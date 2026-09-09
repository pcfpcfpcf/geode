#include "selftest.h"

#include "cuda.h"
#include "forward.h"
#include "kernels.h"
#include "model.h"
#include "quant.h"
#include "strategy.h"
#include "tokenizer.h"

extern const Strategy hybrid;
void gpu_attention(Runtime *runtime, const Layer *layer,
                  int layer_index, int position, int n_tokens);

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PROMPT_TOKENS_MAX 2048

/* gemv_row dequantizes exactly what the reference does in a different order,
   so only float reassociation separates them. */
#define GEMV_TOLERANCE 1e-3f

/* Rounding an activation to int8 costs a fraction of a percent per dot; well
   past this the quantization is not the explanation and the kernel is wrong. */
#define ACTIVATION_TOLERANCE 0.02f

/* Enough to eyeball against another implementation without burying the checks
   underneath it. */
#define DEQUANT_PREVIEW 8

static int report(const char *name, int ok) {
    printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int check_value(const char *what, float got, float want,
                       float tolerance) {
    float off = fabsf(got - want);
    if (off > tolerance) {
        printf("    %s MISMATCH: got %.6f want %.6f (tolerance %.6f)\n", what,
               got, want, tolerance);
        return 0;
    }
    printf("    %s ok: %.6f (%.3f%% off reference)\n", what, got,
           100.0f * off / (1.0f + fabsf(want)));
    return 1;
}

/* Checks both matrix kernels for one tensor against a naive dequant-then-dot
   over the same row. */
static int check_tensor(const GgufFile *g, const char *name) {
    const GgufTensor *t = gguf_find(g, name);
    if (!t) {
        printf("  %s: NOT FOUND\n", name);
        return 0;
    }
    int n = (int)t->dims[0];
    float *deq = malloc((size_t)n * sizeof *deq);
    float *x = malloc((size_t)n * sizeof *x);
    void *scratch = malloc(activation_bytes(n, 1));
    if (!deq || !x || !scratch) {
        printf("  %s: out of memory for %d elements\n", name, n);
        free(scratch);
        free(x);
        free(deq);
        return 0;
    }

    dequant_row(t->data, t->type, n, deq);
    printf("  %-28s %-5s n=%d\n", name, quant_types[t->type].name, n);
    printf("    dequant[0..%d]:", DEQUANT_PREVIEW - 1);
    for (int i = 0; i < DEQUANT_PREVIEW && i < n; i++) printf(" %.6f", deq[i]);
    printf("\n");

    for (int i = 0; i < n; i++) x[i] = (float)(i % 7) - 3.0f;
    float want = 0;
    for (int i = 0; i < n; i++) want += deq[i] * x[i];

    int ok = check_value("gemv", gemv_row(t->data, t->type, n, x), want,
                         GEMV_TOLERANCE * (1.0f + fabsf(want)));

    /* matmul is the path decode and prefill actually take, and it quantizes
       activations to int8 first, so it is held to int8 tolerance rather than
       gemv's. */
    ActivationBatch activation;
    float quantized = 0;
    activation_set(&activation, scratch, x, (size_t)n, n, 1);
    matmul(&quantized, 1, t->data, t->type, n, 0, 1, &activation);
    ok &= check_value("matvec", quantized, want,
                      ACTIVATION_TOLERANCE * (1.0f + fabsf(want)));

    free(scratch);
    free(x);
    free(deq);
    return ok;
}

int selftest_kernels(const GgufFile *g) {
    int ok = 1;
    ok &= check_tensor(g, "blk.0.attn_q.weight");    /* Q4_K */
    ok &= check_tensor(g, "blk.0.attn_norm.weight"); /* F32 */
    /* Q6_K lives in the dense down-projection for MLA and in the value
       projection for GQA. */
    const GgufTensor *q6 = gguf_find(g, "blk.0.ffn_down.weight");
    if (!q6) q6 = gguf_find(g, "blk.0.attn_v.weight");
    if (q6) ok &= check_tensor(g, q6->name);
    else ok = 0;
    /* Q5_0 only appears in MLA's k_b; GQA models have none. */
    if (gguf_find(g, "blk.0.attn_k_b.weight"))
        ok &= check_tensor(g, "blk.0.attn_k_b.weight");
    return report("kernels", ok);
}

int selftest_tokenizer(const GgufFile *g) {
    static const char *prompts[] = {
        "Hello, world!",
        "Привет, как дела?",
        "The quick brown fox jumps over the lazy dog.",
        "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    };
    char err[256];
    Tokenizer t;
    if (!tokenizer_init(&t, g, err, sizeof err)) {
        printf("tokenizer: %s\n", err);
        return 1;
    }
    printf("tokenizer: %d tokens, bos=%d eos=%d add_bos=%d\n", t.n_tokens,
           t.bos_id, t.eos_id, t.add_bos);

    int ok = 1;
    for (unsigned i = 0; i < sizeof prompts / sizeof *prompts; i++) {
        int ids[512];
        int n = tokenizer_encode(&t, prompts[i], ids, 512);
        char back[1024];
        tokenizer_decode(&t, ids, n, back, sizeof back);

        printf("  \"%s\" -> %d tokens: [", prompts[i], n);
        for (int j = 0; j < n && j < 16; j++) printf("%s%d", j ? " " : "", ids[j]);
        if (n > 16) printf(" ...");
        printf("]\n  decode: \"%s\"\n", back);
        if (strcmp(prompts[i], back) != 0) {
            printf("  ROUND-TRIP MISMATCH\n");
            ok = 0;
        }
    }
    tokenizer_free(&t);
    return report("tokenizer", ok);
}

static int argmax(const float *values, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (values[i] > values[best]) best = i;
    return best;
}

/* A fresh runtime per width, because the kv cache each one leaves behind is
   half of what the next would read. */
static int prefill_logits(const Model *model, const int *ids, int n_prompt,
                          int chunk, float *out, char *err, size_t errsz) {
    Runtime *runtime = runtime_start(model, n_prompt, 0, err, errsz);
    if (!runtime) return 0;

    const float *logits = NULL;
    for (int i = 0; i < n_prompt; i += chunk) {
        int n = n_prompt - i;
        if (n > chunk) n = chunk;
        logits = forward(runtime, ids + i, i, n);
    }
    memcpy(out, logits, (size_t)model->n_vocab * sizeof *out);
    runtime_stop(runtime);
    return 1;
}

/* Long enough to span several chunks and to put tokens either side of a chunk
   boundary in each other's attention. */
#define PREFILL_TEST_PROMPT                                                    \
    "The stored-program computer keeps instructions and data in one memory, "  \
    "which is why a program can be written by another program. Compilers, "    \
    "linkers and operating systems all follow from that single decision, and " \
    "so does most of what makes a machine general rather than special."

/* A chunk changes which weights are read together, not what is computed, so
   prefilling in chunks has to land where feeding the same tokens one at a time
   lands. It currently lands there exactly, but the check is a tolerance rather
   than an equality: every token's experts are summed in the same order at any
   width, and nothing promises a future kernel will keep it that way. */
#define PREFILL_TOLERANCE 0.01f

int selftest_prefill(const GgufFile *g) {
    char err[256];
    Model model;
    if (!model_load(&model, g, err, sizeof err)) {
        printf("prefill: %s\n", err);
        return 1;
    }
    Tokenizer t;
    if (!tokenizer_init(&t, g, err, sizeof err)) {
        printf("prefill: %s\n", err);
        model_free(&model);
        return 1;
    }

    int ids[PROMPT_TOKENS_MAX];
    int n_prompt = tokenizer_encode_prompt(&t, PREFILL_TEST_PROMPT, ids,
                                           PROMPT_TOKENS_MAX);

    float *single = malloc((size_t)model.n_vocab * sizeof *single);
    float *chunked = malloc((size_t)model.n_vocab * sizeof *chunked);
    int ok = single && chunked &&
             prefill_logits(&model, ids, n_prompt, 1, single, err, sizeof err) &&
             prefill_logits(&model, ids, n_prompt, PREFILL_CHUNK, chunked, err,
                            sizeof err);

    if (!ok) {
        printf("prefill: %s\n", single && chunked ? err : "out of memory");
    } else {
        float worst = 0, largest = 0;
        for (int i = 0; i < model.n_vocab; i++) {
            float diff = fabsf(single[i] - chunked[i]);
            if (diff > worst) worst = diff;
            if (fabsf(single[i]) > largest) largest = fabsf(single[i]);
        }
        float tolerance = PREFILL_TOLERANCE * (1.0f + largest);
        int token = argmax(chunked, model.n_vocab);
        int wanted = argmax(single, model.n_vocab);

        printf("prefill: %d tokens, chunk 1 vs %d\n", n_prompt, PREFILL_CHUNK);
        printf("  logits differ by at most %.6f of %.3f (tolerance %.6f)\n",
               worst, largest, tolerance);
        printf("  argmax %d vs %d\n", wanted, token);
        if (worst > tolerance || token != wanted) ok = 0;
    }

    free(single);
    free(chunked);
    tokenizer_free(&t);
    model_free(&model);
    return report("prefill", ok);
}

/* The gpu attention recomputes what the cpu's does: same cache (the kv prep
   is literally the same code), same weights, different arithmetic. The cpu
   quantizes its queries to int8 before scoring; the gpu dots them in f32
   against the same cache, so the gap is the cpu's own quantization error
   compounding through every layer: measured ~5% of logit scale at 26-48
   layers. A broken kernel lands far outside that; the argmax has to survive
   regardless. */
#define HYBRID_TOLERANCE 0.06f

/* Records the first layer's attention input and output so the gpu path can
   be checked against a scalar reference of the same step. */
static float *trace_normed;
static float *trace_projected;
static int trace_saved;

static void tracing_attention_cpu(Runtime *runtime, const Layer *layer,
                                  int layer_index, int position,
                                  int n_tokens) {
    forward_attention_cpu(runtime, layer, layer_index, position, n_tokens);
    if (!trace_saved) {
        int n_embd = runtime->model->n_embd;
        memcpy(trace_normed, runtime->normed, (size_t)n_tokens * n_embd * 4);
        memcpy(trace_projected, runtime->projected,
               (size_t)n_tokens * n_embd * 4);
        trace_saved = 1;
    }
}

static void tracing_attention(Runtime *runtime, const Layer *layer,
                              int layer_index, int position, int n_tokens) {
    gpu_attention(runtime, layer, layer_index, position, n_tokens);
    if (!trace_saved) {
        int n_embd = runtime->model->n_embd;
        memcpy(trace_normed, runtime->normed, (size_t)n_tokens * n_embd * 4);
        memcpy(trace_projected, runtime->projected,
               (size_t)n_tokens * n_embd * 4);
        trace_saved = 1;
    }
}

int selftest_hybrid(const GgufFile *g) {
    char err[256];
    Model model;
    if (!model_load(&model, g, err, sizeof err)) {
        printf("hybrid: %s\n", err);
        return 1;
    }
    Tokenizer t;
    if (!tokenizer_init(&t, g, err, sizeof err)) {
        printf("hybrid: %s\n", err);
        model_free(&model);
        return 1;
    }

    int ids[PROMPT_TOKENS_MAX];
    int n_prompt = tokenizer_encode_prompt(&t, PREFILL_TEST_PROMPT, ids,
                                           PROMPT_TOKENS_MAX);

    float *cpu = malloc((size_t)model.n_vocab * sizeof *cpu);
    float *gpu = malloc((size_t)model.n_vocab * sizeof *gpu);
    if (!cpu || !gpu) {
        printf("hybrid: out of memory\n");
        free(cpu);
        free(gpu);
        tokenizer_free(&t);
        model_free(&model);
        return 1;
    }

    if (!prefill_logits(&model, ids, n_prompt, PREFILL_CHUNK, cpu, err,
                        sizeof err)) {
        printf("hybrid: %s\n", err);
        free(cpu);
        free(gpu);
        tokenizer_free(&t);
        model_free(&model);
        return 1;
    }

    const Strategy *strategy = &hybrid;

    Plan plan;
    plan_default(&plan);
    Runtime *runtime = strategy->start(&model, &plan, n_prompt, 0, err,
                                       sizeof err);
    int ok = runtime != NULL;
    if (!ok) {
        printf("hybrid: %s\n", err);
    } else {
        /* One full chunk through layer 0 on both tiers: the gpu attention
           against the cpu's, same inputs -- a per-layer verdict instead of a
           26-layer one. */
        {
            int n = n_prompt < PREFILL_CHUNK ? n_prompt : PREFILL_CHUNK;
            size_t bytes = (size_t)n * model.n_embd * 4;
            float *cpu_in = malloc(bytes), *cpu_out = malloc(bytes);
            float *gpu_in = malloc(bytes), *gpu_out = malloc(bytes);
            if (cpu_in && cpu_out && gpu_in && gpu_out) {
                Runtime *cpu_runtime =
                    runtime_start(&model, n_prompt, 0, err, sizeof err);
                if (cpu_runtime) {
                    trace_normed = cpu_in;
                    trace_projected = cpu_out;
                    trace_saved = 0;
                    forward_with(cpu_runtime, ids, 0, n,
                                  tracing_attention_cpu, NULL);
                    runtime_stop(cpu_runtime);
                    if (trace_saved) {
                        trace_normed = gpu_in;
                        trace_projected = gpu_out;
                        trace_saved = 0;
                        forward_with(runtime, ids, 0, n, tracing_attention,
                                     NULL);
                        float worst_in = 0, worst_out = 0;
                        int worst_i = 0, off = 0;
                        for (size_t i = 0; i < (size_t)n * model.n_embd;
                             i++) {
                            float d = fabsf(gpu_in[i] - cpu_in[i]);
                            if (d > worst_in) worst_in = d;
                            d = fabsf(gpu_out[i] - cpu_out[i]);
                            if (d > worst_out) {
                                worst_out = d;
                                worst_i = (int)i;
                            }
                            if (d > 1e-3f) off++;
                        }
                        printf("  layer-0 over %d tokens: input worst %.6f, "
                               "output %d off, worst %.6f at %d\n",
                               n, worst_in, off, worst_out, worst_i);
                    }
                }
            }
            free(cpu_in); free(cpu_out); free(gpu_in); free(gpu_out);
        }

        const float *logits = NULL;
        for (int i = 0; i < n_prompt; i += PREFILL_CHUNK) {
            int n = n_prompt - i;
            if (n > PREFILL_CHUNK) n = PREFILL_CHUNK;
            logits = strategy->forward(runtime, ids + i, i, n);
        }
        memcpy(gpu, logits, (size_t)model.n_vocab * sizeof *gpu);
        strategy->stop(runtime);

        float worst = 0, largest = 0;
        for (int i = 0; i < model.n_vocab; i++) {
            float diff = fabsf(cpu[i] - gpu[i]);
            if (diff > worst) worst = diff;
            if (fabsf(cpu[i]) > largest) largest = fabsf(cpu[i]);
        }
        float tolerance = HYBRID_TOLERANCE * (1.0f + largest);
        int token = argmax(gpu, model.n_vocab);
        int wanted = argmax(cpu, model.n_vocab);

        printf("hybrid: %d tokens, gpu vs cpu over %d layers\n", n_prompt,
               model.n_layer);
        printf("  logits differ by at most %.6f of %.3f (tolerance %.6f)\n",
               worst, largest, tolerance);
        printf("  argmax %d vs %d\n", wanted, token);
        if (worst > tolerance || token != wanted) ok = 0;
    }

    free(cpu);
    free(gpu);
    tokenizer_free(&t);
    model_free(&model);
    return report("hybrid", ok);
}

/* The gemm shapes the hybrid executor puts on the gpu -- attention per
   layer plus the base feed-forward -- timed at decode's batch and the
   prefill widths, against the probe's ceilings. The probe measures the
   hardware; this measures the kernels the executor actually launches, so
   the gap per shape is visible instead of folded into one end-to-end
   number. The encoding per shape is the uploader's choice: Q4_K streams
   at quant size, everything else lands as f16. */
#define BENCH_SHAPES 16

typedef struct {
    char name[40];
    int n_in, head_out, n_head, type; /* CUDA_W_* */
} BenchShape;

static void bench_add(BenchShape *shapes, int *n_shapes, const char *what,
                      const GgufTensor *tensor, int n_in, int head_out,
                      int n_head) {
    int type = tensor->type == GGML_TYPE_Q4_K && n_in % QK_K == 0
                   ? CUDA_W_Q4K
                   : CUDA_W_F16;
    for (int i = 0; i < *n_shapes; i++)
        if (shapes[i].n_in == n_in && shapes[i].head_out == head_out &&
            shapes[i].n_head == n_head && shapes[i].type == type)
            return;
    BenchShape *s = &shapes[(*n_shapes)++];
    snprintf(s->name, sizeof s->name, "%-12s %-4s %5d->%-5dx%d", what,
             type == CUDA_W_Q4K ? "q4k" : "f16", n_in, head_out, n_head);
    s->n_in = n_in;
    s->head_out = head_out;
    s->n_head = n_head;
    s->type = type;
}

int selftest_bench(const GgufFile *g) {
    char err[256];
    Model model;
    if (!model_load(&model, g, err, sizeof err)) {
        printf("gemm: %s\n", err);
        return 1;
    }
    CudaDevice *dev;
    if (!cuda_start(&dev, err, sizeof err)) {
        printf("gemm: %s\n", err);
        model_free(&model);
        return 1;
    }

    BenchShape shapes[BENCH_SHAPES];
    int n_shapes = 0;
    const Model *m = &model;
    for (int i = 0; i < m->n_layer; i++) {
        const Layer *layer = &m->layers[i];
        if (m->attention == ATTN_MLA) {
            bench_add(shapes, &n_shapes, "kv_a", layer->kv_a_mqa, m->n_embd,
                      m->kv_lora_rank + m->qk_rope_dim, 1);
            bench_add(shapes, &n_shapes, "k_b", layer->k_b, m->qk_nope_dim,
                      m->kv_lora_rank, m->n_head);
            bench_add(shapes, &n_shapes, "v_b", layer->v_b, m->kv_lora_rank,
                      m->head_dim_v, m->n_head);
        } else {
            bench_add(shapes, &n_shapes, "attn_k", layer->attn_k, m->n_embd,
                      m->n_head_kv * m->head_dim_k, 1);
            bench_add(shapes, &n_shapes, "attn_v", layer->attn_v, m->n_embd,
                      m->n_head_kv * m->head_dim_v, 1);
        }
        bench_add(shapes, &n_shapes, "attn_q", layer->attn_q, m->n_embd,
                  m->n_head * m->head_dim_k, 1);
        bench_add(shapes, &n_shapes, "attn_output", layer->attn_output,
                  m->n_head * m->head_dim_v, m->n_embd, 1);
        const FeedForward *base = model_base_ffn(m, layer);
        if (base) {
            int width = (int)base->gate->dims[1];
            bench_add(shapes, &n_shapes, "ffn_gate", base->gate, m->n_embd,
                      width, 1);
            bench_add(shapes, &n_shapes, "ffn_up", base->up, m->n_embd, width,
                      1);
            bench_add(shapes, &n_shapes, "ffn_down", base->down, width,
                      m->n_embd, 1);
        }
    }

    int batches[] = {1, PREFILL_CHUNK / 4, PREFILL_CHUNK / 2, PREFILL_CHUNK};
    for (unsigned b = 0; b < sizeof batches / sizeof *batches; b++) {
        int n_tokens = batches[b];
        printf("\nbatch %d:\n%-30s %9s %9s %9s\n", n_tokens, "shape",
               "ms/call", "GFLOPS", "GB/s");
        for (int s = 0; s < n_shapes; s++) {
            const BenchShape *sh = &shapes[s];
            int rows = sh->head_out * sh->n_head;
            size_t row_bytes = sh->type == CUDA_W_Q4K
                                   ? (size_t)sh->n_in / 256 * 144
                                   : (size_t)sh->n_in * 2;
            size_t weight_bytes = (size_t)rows * row_bytes;
            size_t x_bytes = (size_t)n_tokens * sh->n_in * 4;
            size_t out_bytes = (size_t)n_tokens * rows * 4;
            unsigned long long dw =
                cuda_alloc(dev, weight_bytes, err, sizeof err);
            unsigned long long dx =
                cuda_alloc(dev, x_bytes, err, sizeof err);
            unsigned long long dout =
                cuda_alloc(dev, out_bytes, err, sizeof err);
            if (!dw || !dx || !dout) {
                printf("gemm: %s\n", "gpu out of memory for the bench");
                cuda_stop(dev);
                model_free(&model);
                return 1;
            }
            /* Timing only: zero q4_k blocks dequant to zero, f16 rows hold
               1.0 -- the instruction count is value-independent. */
            void *fill = calloc(1, weight_bytes);
            if (sh->type == CUDA_W_F16)
                for (size_t i = 0; i < weight_bytes / 2; i++)
                    ((uint16_t *)fill)[i] = fp32_to_fp16(1.0f);
            cuda_copy_to(dev, dw, fill, weight_bytes);
            free(fill);
            float *x = malloc(x_bytes);
            for (size_t i = 0; i < x_bytes / 4; i++) x[i] = 0.01f;
            cuda_copy_to(dev, dx, x, x_bytes);
            free(x);

            int reps = n_tokens == 1 ? 500 : 200;
            for (int i = 0; i < 20; i++)
                cuda_gemm(dev, dw, dx, dout, sh->n_in, sh->head_out,
                          sh->n_head, sh->n_in, sh->n_head > 1 ? sh->n_in : 0,
                          rows, n_tokens, sh->type);
            cuda_sync(dev);
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            double t0 = ts.tv_sec + ts.tv_nsec / 1e9;
            for (int i = 0; i < reps; i++)
                cuda_gemm(dev, dw, dx, dout, sh->n_in, sh->head_out,
                          sh->n_head, sh->n_in, sh->n_head > 1 ? sh->n_in : 0,
                          rows, n_tokens, sh->type);
            cuda_sync(dev);
            clock_gettime(CLOCK_MONOTONIC, &ts);
            double ms = (ts.tv_sec + ts.tv_nsec / 1e9 - t0) * 1e3 / reps;
            printf("%-30s %9.3f %9.1f %9.1f\n", sh->name, ms,
                   2.0 * rows * sh->n_in * n_tokens / ms / 1e6,
                   weight_bytes / ms / 1e6);
            cuda_free(dev, dw);
            cuda_free(dev, dx);
            cuda_free(dev, dout);
        }
    }

    cuda_stop(dev);
    model_free(&model);
    return 0;
}
