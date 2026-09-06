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
    DeviceTensor kv_a;      /* MLA: compressed keys+values */
    DeviceTensor attn_k;    /* GQA */
    DeviceTensor attn_v;    /* GQA */
    DeviceTensor k_b;       /* MLA */
    DeviceTensor v_b;       /* MLA */
    DeviceTensor attn_output;
    unsigned long long q_norm; /* GQA: per-layer, f32 */
} HybridLayer;

typedef struct {
    CudaDevice *dev;
    HybridLayer *layers;
    CudaGeometry geometry;
    unsigned long long geom_on_device;
    unsigned long long normed, query, kv_projected, query_latent, attn_latent,
        attn_out, projected, scores, cache;
    uint8_t *slot_stage; /* packed slot+scales, staged for upload */
} Hybrid;

/* -------------------------------------------------------------- */
/* Weight upload. Q4_K rows whose length is whole 256-blocks stream
   straight into vram and dequantize inside the gemm; every other encoding
   is dequantized here once, because the gpu would pay the unpack on every
   token for a type the fused kernel does not know. */

static int tensor_rows(const GgufTensor *t) {
    unsigned long long stacked = t->dims[2] ? t->dims[2] : 1;
    return (int)(t->dims[1] * stacked);
}

static size_t f32_upload_bytes(const GgufTensor *t) {
    return (size_t)tensor_rows(t) * (size_t)t->dims[0] * 4;
}

static int upload_tensor(CudaDevice *dev, const GgufTensor *t, DeviceTensor *out,
                         char *err, size_t errsz) {
    int n_in = (int)t->dims[0];
    int usable = t->type == GGML_TYPE_Q4_K && n_in % QK_K == 0;

    size_t bytes = usable ? (size_t)t->n_bytes : f32_upload_bytes(t);
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
       element. The source stride is the quantized row's, not the f32
       destination's. */
    float *rows = malloc(bytes);
    if (!rows) {
        snprintf(err, errsz, "out of memory dequantizing %s", t->name);
        return 0;
    }
    size_t src_stride = row_bytes(t->type, n_in);
    const unsigned char *src = t->data;
    int stacked = (int)(t->dims[2] ? t->dims[2] : 1);
    for (int m = 0; m < stacked; m++)
        for (int r = 0; r < (int)t->dims[1]; r++) {
            dequant_row(src, t->type, n_in,
                        rows + ((size_t)m * t->dims[1] + r) * n_in);
            src += src_stride;
        }
    int ok = cuda_copy_to(dev, ptr, rows, bytes);
    free(rows);
    if (!ok) {
        snprintf(err, errsz, "%s", cuda_fault(dev));
        return 0;
    }
    out->type = CUDA_W_F32;
    out->ptr = ptr;
    return 1;
}

/* Every weight byte the strategy uploads, for the vram budget check.
   Q4_K tensors stream at quant size; everything else grows to f32 on the
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
            total += usable ? (size_t)t->n_bytes : f32_upload_bytes(t);
        }
    }
    return total;
}

/* -------------------------------------------------------------- */
/* The attention hook. Same contract as the cpu's: normed in, projected
   out, kv appended to the cache at (layer, position). */

static void prepare_query(Runtime *runtime, const Layer *layer, int n_tokens) {
    const Model *model = runtime->model;
    int head_dim = model->head_dim_k;
    int q_stride = model->n_head * head_dim;
    int rope = model->qk_rope_dim;

    for (int t = 0; t < n_tokens; t++) {
        const float *cos_sin = runtime->cos_sin + (size_t)t * rope;
        for (int head = 0; head < model->n_head; head++) {
            float *q = runtime->query + (size_t)t * q_stride +
                       (size_t)head * head_dim;
            if (model->attention == ATTN_GQA)
                rmsnorm(q, q, layer->attn_q_norm->data, head_dim,
                        model->rms_eps);
            if (model->attention == ATTN_GQA)
                rope_apply_neox(q, cos_sin, rope);
            else
                rope_apply(q + (head_dim - rope), cos_sin, rope);
        }
    }
}

static void prepare_kv(Runtime *runtime, const Layer *layer, int n_tokens,
                       uint8_t *slot_stage) {
    const Model *model = runtime->model;
    const CudaGeometry *g = &((Hybrid *)runtime->device)->geometry;
    int width = g->cache_width;
    int slot_bytes = g->slot_bytes;

    for (int t = 0; t < n_tokens; t++) {
        float *projected =
            runtime->kv_projected + (size_t)t * g->cache_width;
        uint8_t *slot = slot_stage + (size_t)t * slot_bytes;
        float *scales = (float *)(slot + width);
        const float *cos_sin = runtime->cos_sin + (size_t)t * g->rope_dim;

        if (model->attention == ATTN_MLA) {
            float normed[512];
            rmsnorm(normed, projected, layer->kv_a_norm->data, g->rank,
                    model->rms_eps);
            rope_apply(projected + g->rank, cos_sin, g->rope_dim);
            scales[0] = quantize_cache(slot, normed, g->rank);
            scales[1] =
                quantize_cache(slot + g->rank, projected + g->rank,
                               g->rope_dim);
        } else {
            int head_dim = model->head_dim_k;
            int kv_width = model->n_head_kv * head_dim;
            for (int kv = 0; kv < model->n_head_kv; kv++) {
                float *k = projected + (size_t)kv * head_dim;
                rmsnorm(k, k, layer->attn_k_norm->data, head_dim,
                        model->rms_eps);
                rope_apply_neox(k, cos_sin, model->qk_rope_dim);
                scales[kv] = quantize_cache(slot + (size_t)kv * head_dim, k,
                                            head_dim);
                scales[model->n_head_kv + kv] =
                    quantize_cache(slot + kv_width + (size_t)kv *
                                                        model->head_dim_v,
                                   projected + kv_width +
                                       (size_t)kv * model->head_dim_v,
                                   model->head_dim_v);
            }
        }
    }
}
void gpu_attention(Runtime *runtime, const Layer *layer,
                           int layer_index, int position, int n_tokens) {
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

    /* The query needs its norms and rotation before scoring, and the cpu
       kernels already do both exactly -- so the query round-trips once. */
    cuda_gemm(dev, hl->attn_q.ptr, hy->normed, hy->query, n_embd,
              model->n_head * head_dim, 1, n_embd, 0, q_stride, n_tokens,
              hl->attn_q.type);
    if (!cuda_copy_from(dev, runtime->query, hy->query,
                        (size_t)n_tokens * q_stride * 4))
        return;
    prepare_query(runtime, layer, n_tokens);
    cuda_copy_to(dev, hy->query, runtime->query,
                  (size_t)n_tokens * q_stride * 4);

    if (mla) {
        cuda_gemm(dev, hl->kv_a.ptr, hy->normed, hy->kv_projected, n_embd,
                  g->cache_width, 1, n_embd, 0, g->cache_width, n_tokens,
                  hl->kv_a.type);
    } else {
        int kv_width = model->n_head_kv * model->head_dim_k;
        cuda_gemm(dev, hl->attn_k.ptr, hy->normed, hy->kv_projected, n_embd,
                  kv_width, 1, n_embd, 0, g->cache_width, n_tokens,
                  hl->attn_k.type);
        cuda_gemm(dev, hl->attn_v.ptr, hy->normed,
                  hy->kv_projected + (size_t)kv_width * 4, n_embd,
                  model->n_head_kv * model->head_dim_v, 1, n_embd, 0,
                  g->cache_width, n_tokens,
                  hl->attn_v.type);
    }

    if (!cuda_copy_from(dev, runtime->kv_projected, hy->kv_projected,
                        (size_t)n_tokens * g->cache_width * 4))
        return;
    prepare_kv(runtime, layer, n_tokens, hy->slot_stage);
    cuda_copy_to(dev,
                  hy->cache + (size_t)layer_index * g->n_ctx * g->slot_bytes +
                      (size_t)position * g->slot_bytes,
                  hy->slot_stage, (size_t)n_tokens * g->slot_bytes);

    if (mla) {
        cuda_gemm(dev, hl->k_b.ptr, hy->query, hy->query_latent,
                  model->qk_nope_dim, model->kv_lora_rank, model->n_head,
                  q_stride, head_dim, g->latent_stride, n_tokens,
                  hl->k_b.type);
        cuda_attention(dev, hy->cache + (size_t)layer_index * g->n_ctx *
                                            g->slot_bytes,
                       hy->query, hy->query_latent, hy->scores,
                       hy->attn_latent, position, n_tokens, g,
                       hy->geom_on_device);
        cuda_gemm(dev, hl->v_b.ptr, hy->attn_latent, hy->attn_out,
                  g->rank, model->head_dim_v, model->n_head, g->latent_stride,
                  g->rank, model->n_head * model->head_dim_v, n_tokens,
                  hl->v_b.type);
    } else {
        cuda_attention(dev, hy->cache + (size_t)layer_index * g->n_ctx *
                                            g->slot_bytes,
                       hy->query, hy->query, hy->scores, hy->attn_out,
                       position, n_tokens, g, hy->geom_on_device);
    }

    cuda_gemm(dev, hl->attn_output.ptr, hy->attn_out, hy->projected,
              model->n_head * model->head_dim_v, n_embd, 1,
              model->n_head * model->head_dim_v, 0, n_embd, n_tokens,
              hl->attn_output.type);
    cuda_copy_from(dev, runtime->projected, hy->projected,
                   (size_t)n_tokens * n_embd * 4);
}

static const float *hybrid_forward(Runtime *runtime, const int *tokens,
                                   int position, int n_tokens) {
    Hybrid *hy = runtime->device;
    /* cuda_start ran every kernel shape against checked results, so a fault
       from here on means the driver is failing; continuing would emit
       garbage as if it were logits. */
    if (cuda_fault(hy->dev)) {
        fprintf(stderr, "gpu fault, cannot continue: %s\n",
                cuda_fault(hy->dev));
        exit(1);
    }
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
        cuda_host_unpin(dev, runtime->query);
        cuda_host_unpin(dev, runtime->kv_projected);
        cuda_host_unpin(dev, runtime->projected);
        cuda_host_unpin(dev, hy->slot_stage);
        for (int i = 0; i < runtime->model->n_layer; i++) {
            HybridLayer *hl = &hy->layers ? &hy->layers[i] : NULL;
            if (!hl) break;
            cuda_free(dev, hl->attn_q.ptr);
            cuda_free(dev, hl->attn_output.ptr);
            cuda_free(dev, hl->kv_a.ptr);
            cuda_free(dev, hl->attn_k.ptr);
            cuda_free(dev, hl->attn_v.ptr);
            cuda_free(dev, hl->k_b.ptr);
            cuda_free(dev, hl->v_b.ptr);
            cuda_free(dev, hl->q_norm);
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
        cuda_free(dev, hy->geom_on_device);
        cuda_stop(dev);
    }
    free(hy->layers);
    free(hy->slot_stage);
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
    } else {
        if (!upload_tensor(dev, layer->attn_k, &hl->attn_k, err, errsz))
            return 0;
        if (!upload_tensor(dev, layer->attn_v, &hl->attn_v, err, errsz))
            return 0;
        hl->q_norm = cuda_alloc(dev, (size_t)model->head_dim_k * 4, err,
                                errsz);
        if (!hl->q_norm) return 0;
        if (!cuda_copy_to(dev, hl->q_norm, layer->attn_q_norm->data,
                          (size_t)model->head_dim_k * 4)) {
            snprintf(err, errsz, "%s", cuda_fault(dev));
            return 0;
        }
    }
    return 1;
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
    hy->slot_stage = malloc((size_t)max_tokens * g->slot_bytes);
    if (!hy->layers || !hy->slot_stage) {
        snprintf(err, errsz, "out of memory");
        hybrid_teardown(runtime);
        return NULL;
    }

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

    /* The per-layer transfers are small and frequent; pinning the buffers
       they touch keeps each copy off the driver's pageable staging path. */
    cuda_host_pin(dev, runtime->normed, (size_t)max_tokens * m->n_embd * 4);
    cuda_host_pin(dev, runtime->query, (size_t)max_tokens * g->q_stride * 4);
    cuda_host_pin(dev, runtime->kv_projected,
                  (size_t)max_tokens * g->cache_width * 4);
    cuda_host_pin(dev, runtime->projected, (size_t)max_tokens * m->n_embd * 4);
    cuda_host_pin(dev, hy->slot_stage, (size_t)max_tokens * g->slot_bytes);
    return runtime;
}

const Strategy hybrid = {
    "HYBRID",
    hybrid_start,
    hybrid_forward,
    hybrid_stop,
};