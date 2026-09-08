#include "cuda.h"
#include "forward.h"
#include "kernels.h"
#include "model.h"
#include "quant.h"
#include "strategy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The score kernel stages a position tile in shared; the tile is sized by
   context, and this is where the budget stops. */
#define HYBRID_CTX_MAX 8192

/* The gemm assigns eight rows (one warp each) per block, all from one head,
   so a per-head matrix stack must tile into whole blocks. */
#define GEMM_HEAD_ALIGN 8

typedef struct {
    unsigned long long ptr;
    int type; /* CUDA_W_* */
} DeviceTensor;

typedef struct {
    DeviceTensor attn_q;
    DeviceTensor attn_k;    /* GQA */
    DeviceTensor attn_v;    /* GQA */
    DeviceTensor kv_a;      /* MLA: compressed keys+values */
    DeviceTensor k_b;       /* MLA */
    DeviceTensor v_b;       /* MLA */
    DeviceTensor attn_output;
    unsigned long long q_norm;  /* GQA: per-layer, f32 */
    unsigned long long kv_norm; /* f32: MLA kv_a_norm, GQA attn_k_norm */
    int kv_merged;              /* GQA: attn_k/attn_v share one allocation */
} HybridLayer;

typedef struct {
    CudaDevice *dev;
    HybridLayer *layers;
    CudaGeometry geometry;
    unsigned long long geom_on_device;
    unsigned long long normed, query, kv_projected, query_latent, attn_latent,
        attn_out, projected, scores, cache, cos_sin;
    /* The rope table is per-forward, not per-layer: it rides up once and
       the other 47 layers of the same chunk skip their copy. */
    int cos_sin_position, cos_sin_tokens;
} Hybrid;

/* -------------------------------------------------------------- */
/* Weight upload. Q4_K rows whose length is whole 256-blocks stream
   straight into vram and dequantize inside the gemm; every other encoding
   is dequantized here once, to f16 -- half the f32 bytes, which the gemm
   re-reads on every token of decode, and against the encoding's own
   quantization error the half-ulp of f16 is noise. */

static int tensor_rows(const GgufTensor *t) {
    unsigned long long stacked = t->dims[2] ? t->dims[2] : 1;
    return (int)(t->dims[1] * stacked);
}

static size_t f16_upload_bytes(const GgufTensor *t) {
    return (size_t)tensor_rows(t) * (size_t)t->dims[0] * 2;
}

static int upload_tensor(CudaDevice *dev, const GgufTensor *t, DeviceTensor *out,
                         char *err, size_t errsz) {
    int n_in = (int)t->dims[0];
    int usable = t->type == GGML_TYPE_Q4_K && n_in % QK_K == 0;

    size_t bytes = usable ? (size_t)t->n_bytes : f16_upload_bytes(t);
    unsigned long long ptr = cuda_alloc(dev, bytes, err, errsz);
    if (!ptr) return 0;

    if (usable) {
        if (!cuda_copy_to(dev, ptr, t->data, bytes)) {
            snprintf(err, errsz, "%s", cuda_fault(dev));
            return 0;
        }
        out->type = CUDA_W_Q4K;
        out->ptr = ptr;
        return 1;
    }

    /* Dequantized rows keep the stack layout the gemm walks: matrix, row,
    element. The source stride is the quantized row's, not the f16
    destination's. */
    uint16_t *rows = malloc(bytes);
    if (!rows) {
        snprintf(err, errsz, "out of memory dequantizing %s", t->name);
        return 0;
    }
    size_t src_stride = row_bytes(t->type, n_in);
    const unsigned char *src = t->data;
    int stacked = (int)(t->dims[2] ? t->dims[2] : 1);
    float *values = malloc((size_t)n_in * sizeof *values);
    if (!values) {
        snprintf(err, errsz, "out of memory dequantizing %s", t->name);
        free(rows);
        return 0;
    }
    for (int m = 0; m < stacked; m++)
        for (int r = 0; r < (int)t->dims[1]; r++) {
            dequant_row(src, t->type, n_in, values);
            uint16_t *dst = rows + ((size_t)m * t->dims[1] + r) * n_in;
            for (int c = 0; c < n_in; c++) dst[c] = fp32_to_fp16(values[c]);
            src += src_stride;
        }
    free(values);
    int ok = cuda_copy_to(dev, ptr, rows, bytes);
    free(rows);
    if (!ok) {
        snprintf(err, errsz, "%s", cuda_fault(dev));
        return 0;
    }
    out->type = CUDA_W_F16;
    out->ptr = ptr;
    return 1;
}

/* Every weight byte the strategy uploads, for the vram budget check.
   Q4_K tensors stream at quant size; everything else grows to f16 on the
   way up. */
static size_t attention_upload_bytes(const Model *model) {
    size_t total = 0;
    for (int i = 0; i < model->n_layer; i++) {
        const Layer *layer = &model->layers[i];
        const GgufTensor *tensors[6];
        int n = 0;
        tensors[n++] = layer->attn_q;
        tensors[n++] = layer->attn_output;
        if (model->attention == ATTN_MLA) {
            tensors[n++] = layer->kv_a_mqa;
            tensors[n++] = layer->k_b;
            tensors[n++] = layer->v_b;
        } else {
            tensors[n++] = layer->attn_k;
            tensors[n++] = layer->attn_v;
            tensors[n++] = layer->attn_q_norm;
        }
        for (int k = 0; k < n; k++) {
            const GgufTensor *t = tensors[k];
            int usable = t->type == GGML_TYPE_Q4_K &&
                         (int)t->dims[0] % QK_K == 0;
            total += usable ? (size_t)t->n_bytes : f16_upload_bytes(t);
        }
    }
    return total;
}

/* -------------------------------------------------------------- */
/* The attention hook. Same contract as the cpu's: normed in, projected
   out, kv appended to the cache at (layer, position).

   One blocking call per layer: the copy-from at the end drains the kernel
   queue, lands the output the ffn needs, and surfaces any deferred fault.
   The uploads at the top never wait on a kernel -- the previous layer's
   copy-from left the queue empty -- and everything between them is async
   launches. The query and kv prepares run on the gpu (cuda_q_prep /
   cuda_kv_prep), so no intermediate activation crosses the bus and the
   cache slot is written in vram directly. */
void gpu_attention(Runtime *runtime, const Layer *layer,
                          int layer_index, int position, int n_tokens) {
    (void)layer; /* the layer's weights are already on the device */
    const Model *model = runtime->model;
    Hybrid *hy = runtime->device;
    HybridLayer *hl = &hy->layers[layer_index];
    CudaDevice *dev = hy->dev;
    const CudaGeometry *g = &hy->geometry;
    int mla = model->attention == ATTN_MLA;
    int n_embd = model->n_embd;
    int head_dim = model->head_dim_k;
    int q_stride = model->n_head * head_dim;

    cuda_copy_to(dev, hy->normed, runtime->normed,
                 (size_t)n_tokens * n_embd * 4);
    if (position != hy->cos_sin_position || n_tokens != hy->cos_sin_tokens) {
        cuda_copy_to(dev, hy->cos_sin, runtime->cos_sin,
                     (size_t)n_tokens * g->rope_dim * 4);
        hy->cos_sin_position = position;
        hy->cos_sin_tokens = n_tokens;
    }

    cuda_gemm(dev, hl->attn_q.ptr, hy->normed, hy->query, n_embd,
              model->n_head * head_dim, 1, n_embd, 0, q_stride, n_tokens,
              hl->attn_q.type);
    cuda_q_prep(dev, hy->query, hy->cos_sin, hl->q_norm, hy->geom_on_device,
               n_tokens);

    if (mla) {
        cuda_gemm(dev, hl->kv_a.ptr, hy->normed, hy->kv_projected, n_embd,
                  g->cache_width, 1, n_embd, 0, g->cache_width, n_tokens,
                  hl->kv_a.type);
    } else if (hl->kv_merged) {
        cuda_gemm(dev, hl->attn_k.ptr, hy->normed, hy->kv_projected, n_embd,
                  g->cache_width, 1, n_embd, 0, g->cache_width, n_tokens,
                  hl->attn_k.type);
    } else {
        int kv_width = model->n_head_kv * model->head_dim_k;
        cuda_gemm(dev, hl->attn_k.ptr, hy->normed, hy->kv_projected, n_embd,
                  kv_width, 1, n_embd, 0, g->cache_width, n_tokens,
                  hl->attn_k.type);
        cuda_gemm(dev, hl->attn_v.ptr, hy->normed,
                  hy->kv_projected + (size_t)kv_width * 4, n_embd,
                  model->n_head_kv * model->head_dim_v, 1, n_embd, 0,
                  g->cache_width, n_tokens, hl->attn_v.type);
    }

    cuda_kv_prep(dev, hy->kv_projected, hy->cos_sin, hl->kv_norm,
                 hy->cache + (size_t)layer_index * g->n_ctx * g->slot_bytes +
                     (size_t)position * g->slot_bytes,
                 hy->geom_on_device, n_tokens);

    if (mla) {
        cuda_gemm(dev, hl->k_b.ptr, hy->query, hy->query_latent,
                  model->qk_nope_dim, model->kv_lora_rank, model->n_head,
                  q_stride, head_dim, g->latent_stride, n_tokens,
                  hl->k_b.type);
    }
    cuda_attention(dev, hy->cache + (size_t)layer_index * g->n_ctx *
                                        g->slot_bytes,
                   hy->query, mla ? hy->query_latent : hy->query, hy->scores,
                   mla ? hy->attn_latent : hy->attn_out, position, n_tokens, g,
                   hy->geom_on_device);
    if (mla) {
        cuda_gemm(dev, hl->v_b.ptr, hy->attn_latent, hy->attn_out,
                  g->rank, model->head_dim_v, model->n_head, g->latent_stride,
                  g->rank, model->n_head * model->head_dim_v, n_tokens,
                  hl->v_b.type);
    }

    cuda_gemm(dev, hl->attn_output.ptr, hy->attn_out, hy->projected,
              model->n_head * model->head_dim_v, n_embd, 1,
              model->n_head * model->head_dim_v, 0, n_embd, n_tokens,
              hl->attn_output.type);

    cuda_copy_from(dev, runtime->projected, hy->projected,
                   (size_t)n_tokens * n_embd * 4);
    if (cuda_fault(dev)) {
        fprintf(stderr, "gpu fault, cannot continue: %s\n", cuda_fault(dev));
        exit(1);
    }
}

static const float *hybrid_forward(Runtime *runtime, const int *tokens,
                                   int position, int n_tokens) {
    return forward_with(runtime, tokens, position, n_tokens, gpu_attention);
}

/* -------------------------------------------------------------- */
/* Start and stop. */

/* Frees whatever was allocated so far -- partial states included, which is
   what a failed start leaves behind. */
static void hybrid_teardown(Runtime *runtime) {
    Hybrid *hy = runtime->device;
    if (!hy) {
        runtime_stop(runtime);
        return;
    }
    CudaDevice *dev = hy->dev;
    if (dev) {
        cuda_host_unpin(dev, runtime->normed);
        cuda_host_unpin(dev, runtime->projected);
        cuda_host_unpin(dev, runtime->cos_sin);
        for (int i = 0; i < runtime->model->n_layer; i++) {
            HybridLayer *hl = &hy->layers ? &hy->layers[i] : NULL;
            if (!hl) break;
            cuda_free(dev, hl->attn_q.ptr);
            cuda_free(dev, hl->attn_output.ptr);
            cuda_free(dev, hl->kv_a.ptr);
            /* Merged, attn_v points inside attn_k's allocation. */
            cuda_free(dev, hl->attn_k.ptr);
            if (!hl->kv_merged) cuda_free(dev, hl->attn_v.ptr);
            cuda_free(dev, hl->k_b.ptr);
            cuda_free(dev, hl->v_b.ptr);
            cuda_free(dev, hl->q_norm);
            cuda_free(dev, hl->kv_norm);
        }
        cuda_free(dev, hy->normed);
        cuda_free(dev, hy->query);
        cuda_free(dev, hy->kv_projected);
        cuda_free(dev, hy->query_latent);
        cuda_free(dev, hy->attn_latent);
        cuda_free(dev, hy->attn_out);
        cuda_free(dev, hy->projected);
        cuda_free(dev, hy->scores);
        cuda_free(dev, hy->cache);
        cuda_free(dev, hy->cos_sin);
        cuda_free(dev, hy->geom_on_device);
        cuda_stop(dev);
    }
    free(hy->layers);
    free(hy);
    runtime->device = NULL;
    runtime_stop(runtime);
}

static void hybrid_stop(Runtime *runtime) { hybrid_teardown(runtime); }

static int check_geometry(const Model *model, int n_ctx, char *err,
                          size_t errsz) {
    if (n_ctx > HYBRID_CTX_MAX) {
        snprintf(err, errsz,
                 "HYBRID supports contexts up to %d; this run asks for %d",
                 HYBRID_CTX_MAX, n_ctx);
        return 0;
    }
    int head_dim = model->head_dim_k;
    if (head_dim % 4 || model->head_dim_v % 4 || model->qk_rope_dim % 4) {
        snprintf(err, errsz,
                 "HYBRID needs head dimensions divisible by 4; this model "
                 "has %d/%d/%d",
                 head_dim, model->head_dim_v, model->qk_rope_dim);
        return 0;
    }
    if (model->attention == ATTN_MLA &&
        (model->kv_lora_rank % 64 || model->kv_lora_rank > 512)) {
        snprintf(err, errsz,
                 "HYBRID supports kv_lora_rank up to 512 and divisible by "
                 "64; this model has %d",
                 model->kv_lora_rank);
        return 0;
    }
    if (model->attention == ATTN_GQA &&
        (model->head_dim_v % 64 || model->n_head > 32)) {
        snprintf(err, errsz,
                 "HYBRID supports GQA with value heads of 64 and at most 32 "
                 "query heads; this model has %d/%d",
                 model->head_dim_v, model->n_head);
        return 0;
    }
    if (model->attention == ATTN_GQA && model->n_head % model->n_head_kv) {
        snprintf(err, errsz,
                 "GQA with %d query heads over %d kv heads does not divide "
                 "evenly",
                 model->n_head, model->n_head_kv);
        return 0;
    }
    /* The gpu kernels declare their shared memory statically, so the widest
       shape this model can ask for has to fit what they declared. */
    if (model->n_head > 32) {
        snprintf(err, errsz,
                 "HYBRID supports at most 32 query heads; this model has %d",
                 model->n_head);
        return 0;
    }
    if (model->attention == ATTN_GQA &&
        (model->n_head_kv > 8 ||
         model->n_head_kv * model->head_dim_k > 1024)) {
        snprintf(err, errsz,
                 "HYBRID supports GQA with at most 8 kv heads of 1024 key "
                 "elements; this model has %d x %d",
                 model->n_head_kv, model->head_dim_k);
        return 0;
    }
    if (model->n_head * model->head_dim_v > 6144) {
        snprintf(err, errsz,
                 "HYBRID supports attention output rows up to 6144; this "
                 "model has %d",
                 model->n_head * model->head_dim_v);
        return 0;
    }
    return 1;
}

/* A small f32 vector -- a norm weight -- up as its own allocation. */
static unsigned long long upload_f32(CudaDevice *dev, const float *values,
                                      int n, char *err, size_t errsz) {
    unsigned long long ptr = cuda_alloc(dev, (size_t)n * 4, err, errsz);
    if (!ptr) return 0;
    if (!cuda_copy_to(dev, ptr, values, (size_t)n * 4)) {
        snprintf(err, errsz, "%s", cuda_fault(dev));
        cuda_free(dev, ptr);
        return 0;
    }
    return ptr;
}

static int q4k_streamable(const GgufTensor *t) {
    return t->type == GGML_TYPE_Q4_K && (int)t->dims[0] % QK_K == 0;
}

/* GQA's key and value projections share one gemm when both stream as
   Q4_K: stacked in one allocation, they are one matrix of cache_width
   rows, which is exactly the kv projection's layout. */
static int upload_gqa_kv(CudaDevice *dev, const Layer *layer, HybridLayer *hl,
                         char *err, size_t errsz) {
    const GgufTensor *k = layer->attn_k;
    const GgufTensor *v = layer->attn_v;
    hl->kv_merged = 0;
    if (q4k_streamable(k) && q4k_streamable(v)) {
        size_t k_bytes = k->n_bytes;
        unsigned long long ptr = cuda_alloc(dev, k_bytes + v->n_bytes, err,
                                            errsz);
        if (!ptr) return 0;
        if (!cuda_copy_to(dev, ptr, k->data, k_bytes) ||
            !cuda_copy_to(dev, ptr + k_bytes, v->data, v->n_bytes)) {
            snprintf(err, errsz, "%s", cuda_fault(dev));
            cuda_free(dev, ptr);
            return 0;
        }
        hl->attn_k = (DeviceTensor){ptr, CUDA_W_Q4K};
        hl->attn_v = (DeviceTensor){ptr + k_bytes, CUDA_W_Q4K};
        hl->kv_merged = 1;
        return 1;
    }
    return upload_tensor(dev, k, &hl->attn_k, err, errsz) &&
           upload_tensor(dev, v, &hl->attn_v, err, errsz);
}

static int upload_layer(CudaDevice *dev, const Model *model, int index,
                        HybridLayer *hl, char *err, size_t errsz) {
    const Layer *layer = &model->layers[index];
    if (!upload_tensor(dev, layer->attn_q, &hl->attn_q, err, errsz))
        return 0;
    if (!upload_tensor(dev, layer->attn_output, &hl->attn_output, err, errsz))
        return 0;
    if (model->attention == ATTN_MLA) {
        if (!upload_tensor(dev, layer->kv_a_mqa, &hl->kv_a, err, errsz))
            return 0;
        if (!upload_tensor(dev, layer->k_b, &hl->k_b, err, errsz)) return 0;
        if (!upload_tensor(dev, layer->v_b, &hl->v_b, err, errsz)) return 0;
        /* The per-head stacks must tile into whole gemm blocks, or a block
           would straddle two heads and stage the wrong activations. */
        if (layer->k_b->dims[1] % GEMM_HEAD_ALIGN ||
            layer->v_b->dims[1] % GEMM_HEAD_ALIGN) {
            snprintf(err, errsz,
                     "attn_k_b/attn_v_b rows per head must be divisible by "
                     "%d for the gpu gemm",
                     GEMM_HEAD_ALIGN);
            return 0;
        }
        hl->kv_norm = upload_f32(dev, layer->kv_a_norm->data,
                                 model->kv_lora_rank, err, errsz);
        return hl->kv_norm != 0;
    }
    if (!upload_gqa_kv(dev, layer, hl, err, errsz)) return 0;
    hl->q_norm = upload_f32(dev, layer->attn_q_norm->data, model->head_dim_k,
                            err, errsz);
    if (!hl->q_norm) return 0;
    hl->kv_norm = upload_f32(dev, layer->attn_k_norm->data, model->head_dim_k,
                             err, errsz);
    return hl->kv_norm != 0;
}

static Runtime *hybrid_start(const Model *model, const Plan *plan, int n_ctx,
                             int n_threads, char *err, size_t errsz) {
    (void)plan;
    Runtime *runtime = runtime_start(model, n_ctx, n_threads, err, errsz);
    if (!runtime) return NULL;

    if (!check_geometry(model, runtime->n_ctx, err, errsz)) {
        runtime_stop(runtime);
        return NULL;
    }

    Hybrid *hy = calloc(1, sizeof *hy);
    if (!hy) {
        snprintf(err, errsz, "out of memory");
        runtime_stop(runtime);
        return NULL;
    }

    if (!cuda_start(&hy->dev, err, errsz)) {
        free(hy);
        runtime_stop(runtime);
        return NULL;
    }
    runtime->device = hy;
    CudaDevice *dev = hy->dev;

    const Model *m = model;
    CudaGeometry *g = &hy->geometry;
    g->kind = m->attention;
    g->n_head = m->n_head;
    g->n_head_kv = m->attention == ATTN_GQA ? m->n_head_kv : 0;
    g->head_dim_k = m->head_dim_k;
    g->rank = m->attention == ATTN_MLA ? m->kv_lora_rank : 0;
    g->rope_dim = m->qk_rope_dim;
    g->n_segments = runtime->n_segments;
    g->cache_width = runtime->cache_width;
    g->slot_bytes = runtime->cache_width + runtime->n_segments * 4;
    g->q_stride = m->n_head * m->head_dim_k;
    g->latent_stride = m->attention == ATTN_MLA ? m->n_head * m->kv_lora_rank
                                                : 0;
    g->out_stride = m->attention == ATTN_MLA
                        ? m->n_head * m->kv_lora_rank
                        : m->n_head * m->head_dim_v;
    g->fold_width = m->attention == ATTN_MLA ? m->kv_lora_rank
                                             : m->head_dim_v;
    g->kq_scale = m->kq_scale;
    g->n_ctx = runtime->n_ctx;
    g->rms_eps = m->rms_eps;

    int max_tokens = runtime->max_tokens;
    size_t activations = (size_t)max_tokens *
                         ((size_t)m->n_embd * 2 + g->q_stride + g->cache_width +
                          g->latent_stride * 2 + g->out_stride + g->rope_dim) *
                         4;
    size_t weights = attention_upload_bytes(model);
    size_t kv_bytes = (size_t)m->n_layer * g->n_ctx * g->slot_bytes;
    size_t scores_bytes =
        (size_t)m->n_head * max_tokens * g->n_ctx * 4 + sizeof *g;
    size_t needed = weights + kv_bytes + activations + scores_bytes;

    unsigned long long free_bytes = cuda_vram_free(dev);
    if (free_bytes < needed + (64ull << 20)) {
        snprintf(err, errsz,
                 "HYBRID needs %.2f GB of vram, %.2f GB free (serving "
                 "fewer models or planning cpu-stream frees the rest)",
                 needed / 1e9, free_bytes / 1e9);
        hybrid_teardown(runtime);
        return NULL;
    }

    hy->layers = calloc((size_t)m->n_layer, sizeof *hy->layers);
    if (!hy->layers) {
        snprintf(err, errsz, "out of memory");
        hybrid_teardown(runtime);
        return NULL;
    }
    hy->cos_sin_position = -1;

    int ok = 1;
    for (int i = 0; i < m->n_layer && ok; i++)
        ok = upload_layer(dev, m, i, &hy->layers[i], err, errsz);
    if (!ok) {
        hybrid_teardown(runtime);
        return NULL;
    }

    ok &= (hy->normed = cuda_alloc(dev, (size_t)max_tokens * m->n_embd * 4,
                                   err, errsz)) != 0;
    ok &= (hy->query = cuda_alloc(dev, (size_t)max_tokens * g->q_stride * 4,
                                  err, errsz)) != 0;
    ok &= (hy->kv_projected =
               cuda_alloc(dev, (size_t)max_tokens * g->cache_width * 4,
                          err, errsz)) != 0;
    ok &= (hy->attn_out = cuda_alloc(dev, (size_t)max_tokens * g->out_stride * 4,
                                     err, errsz)) != 0;
    ok &= (hy->projected = cuda_alloc(dev, (size_t)max_tokens * m->n_embd * 4,
                                      err, errsz)) != 0;
    ok &= (hy->scores = cuda_alloc(
               dev, (size_t)m->n_head * max_tokens * g->n_ctx * 4, err,
               errsz)) != 0;
ok &= (hy->cache = cuda_alloc(dev, (size_t)m->n_layer * g->n_ctx *
                                              g->slot_bytes,
                                   err, errsz)) != 0;
    ok &= (hy->cos_sin = cuda_alloc(dev, (size_t)max_tokens * g->rope_dim * 4,
                                    err, errsz)) != 0;
    ok &= (hy->geom_on_device = cuda_alloc(dev, sizeof *g, err, errsz)) != 0;
    if (m->attention == ATTN_MLA) {
        ok &= (hy->query_latent =
                   cuda_alloc(dev, (size_t)max_tokens * g->latent_stride * 4,
                              err, errsz)) != 0;
        ok &= (hy->attn_latent =
                   cuda_alloc(dev, (size_t)max_tokens * g->latent_stride * 4,
                              err, errsz)) != 0;
    }
    if (ok) ok = cuda_copy_to(dev, hy->geom_on_device, g, sizeof *g);
    if (!ok) {
        if (!err[0]) snprintf(err, errsz, "%s", cuda_fault(dev));
        hybrid_teardown(runtime);
        return NULL;
    }

    /* The per-layer transfers are small and frequent; a pin that silently
       fell back to pageable staging costs ~1 ms a copy on this driver,
       which is a whole decode rate -- so it is required, not best effort. */
    if (!cuda_host_pin(dev, runtime->normed,
                       (size_t)max_tokens * m->n_embd * 4) ||
        !cuda_host_pin(dev, runtime->projected,
                       (size_t)max_tokens * m->n_embd * 4) ||
        !cuda_host_pin(dev, runtime->cos_sin,
                       (size_t)max_tokens * g->rope_dim * 4)) {
        snprintf(err, errsz,
                 "could not pin the per-layer transfer buffers (another "
                 "process may hold the gpu, or the driver refuses to pin "
                 "this much host memory)");
        hybrid_teardown(runtime);
        return NULL;
    }
    return runtime;
}

const Strategy hybrid = {
    "HYBRID",
    hybrid_start,
    hybrid_forward,
    hybrid_stop,
};