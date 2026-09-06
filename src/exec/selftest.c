#include "selftest.h"

#include "forward.h"
#include "kernels.h"
#include "model.h"
#include "quant.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
