#include "forward.h"
#include "gguf.h"
#include "kernels.h"
#include "model.h"
#include "modules.h"
#include "quant.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void cfg(const GgufFile *g, const char *arch, const char *key) {
    char full[160];
    unsigned long long u;
    double f;
    snprintf(full, sizeof full, "%s.%s", arch, key);
    if (gguf_meta_u64(g, full, &u))
        printf("  %-42s %llu\n", key, u);
    else if (gguf_meta_f64(g, full, &f))
        printf("  %-42s %g\n", key, f);
}

static const char *config_keys[] = {
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.kv_lora_rank",
    "attention.key_length_mla",
    "attention.value_length_mla",
    "attention.layer_norm_rms_epsilon",
    "rope.dimension_count",
    "rope.freq_base",
    "rope.scaling.type",
    "rope.scaling.factor",
    "rope.scaling.original_context_length",
    "rope.scaling.yarn_beta_fast",
    "rope.scaling.yarn_beta_slow",
    "rope.scaling.yarn_log_multiplier",
    "expert_count",
    "expert_used_count",
    "expert_shared_count",
    "expert_feed_forward_length",
    "expert_weights_scale",
    "expert_weights_norm",
    "expert_gating_func",
    "leading_dense_block_count",
    "vocab_size",
};

static void dump_layer_tensors(const GgufFile *g, int layer) {
    char prefix[16];
    snprintf(prefix, sizeof prefix, "blk.%d.", layer);
    size_t plen = strlen(prefix);
    for (unsigned long long i = 0; i < g->n_tensors; i++) {
        const GgufTensor *t = &g->tensors[i];
        if (strncmp(t->name, prefix, plen) != 0) continue;
        printf("  %-34s %-5s [", t->name + plen, quant_types[t->type].name);
        for (int d = 0; d < t->n_dims; d++)
            printf("%s%llu", d ? ", " : "", t->dims[d]);
        printf("]  %.2f MB\n", t->n_bytes / 1e6);
    }
}

/* Rounding an activation to int8 costs a fraction of a percent per dot; well
   past this the quantization is not the explanation and the kernel is wrong. */
#define ACTIVATION_TOLERANCE 0.02f

/* Dequant the first 256 elements of a tensor, print them, and check gemv_row
   against a naive dequant-then-dot reference. */
static int check_tensor(const GgufFile *g, const char *name) {
    const GgufTensor *t = gguf_find(g, name);
    if (!t) {
        printf("  %s: NOT FOUND\n", name);
        return 0;
    }
    int n = (int)t->dims[0];
    float *deq = malloc((size_t)n * sizeof(float));
    float *x = malloc((size_t)n * sizeof(float));
    dequant_row(t->data, t->type, n, deq);

    printf("  %-28s %-5s n=%d\n", name, quant_types[t->type].name, n);
    printf("    dequant[0..255]:");
    for (int i = 0; i < 256 && i < n; i++) printf(" %.6f", deq[i]);
    printf("\n");

    int ok = 1;
    for (int i = 0; i < n; i++) x[i] = (float)(i % 7) - 3.0f;
    float want = 0;
    for (int i = 0; i < n; i++) want += deq[i] * x[i];

    float got = gemv_row(t->data, t->type, n, x);
    if (fabsf(got - want) > 1e-3f * (1.0f + fabsf(want))) {
        printf("    gemv MISMATCH: got %.6f want %.6f\n", got, want);
        ok = 0;
    } else {
        printf("    gemv ok: %.6f\n", got);
    }

    /* matvec is the path decode actually takes, and it quantizes activations
       to int8 first, so it is held to int8 tolerance rather than gemv's. */
    void *scratch = malloc(activation_bytes(n) + 1);
    Activation activation;
    float quantized = 0;
    activation_set(&activation, scratch, x, n);
    matvec(&quantized, t->data, t->type, n, 0, 1, &activation);
    float tolerance = ACTIVATION_TOLERANCE * (1.0f + fabsf(want));
    if (fabsf(quantized - want) > tolerance) {
        printf("    matvec MISMATCH: got %.6f want %.6f (tolerance %.6f)\n",
               quantized, want, tolerance);
        ok = 0;
    } else {
        printf("    matvec ok: %.6f (%.3f%% off reference)\n", quantized,
               100.0f * fabsf(quantized - want) / (1.0f + fabsf(want)));
    }
    free(scratch);
    free(x);
    free(deq);
    return ok;
}

static int kernels_test(const GgufFile *g) {
    int ok = 1;
    ok &= check_tensor(g, "blk.0.attn_q.weight");      /* Q4_K */
    ok &= check_tensor(g, "blk.0.ffn_down.weight");    /* Q6_K */
    ok &= check_tensor(g, "blk.0.attn_k_b.weight");    /* Q5_0 */
    ok &= check_tensor(g, "blk.0.attn_norm.weight");    /* F32 */
    printf(ok ? "kernels: PASS\n" : "kernels: FAIL\n");
    return ok ? 0 : 1;
}

static int tokenize_test(const GgufFile *g) {
    char err[256];
    Tokenizer t;
    if (!tokenizer_init(&t, g, err, sizeof err)) {
        printf("tokenizer: %s\n", err);
        return 1;
    }
    printf("tokenizer: %d tokens, bos=%d eos=%d add_bos=%d\n", t.n_tokens,
           t.bos_id, t.eos_id, t.add_bos);

    const char *prompts[] = {
        "Hello, world!",
        "Привет, как дела?",
        "The quick brown fox jumps over the lazy dog.",
        "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    };
    int ok = 1;
    for (unsigned i = 0; i < sizeof prompts / sizeof prompts[0]; i++) {
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
    printf(ok ? "tokenizer: PASS\n" : "tokenizer: FAIL\n");
    return ok ? 0 : 1;
}

#define PROMPT_TOKENS_MAX 2048
#define DEFAULT_PREDICT_TOKENS 32

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int argmax(const float *values, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (values[i] > values[best]) best = i;
    return best;
}

static int run_prompt(const GgufFile *g, const char *prompt, int n_predict,
                      int n_threads) {
    char err[256];
    Model model;
    if (!model_load(&model, g, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    Tokenizer tokenizer;
    if (!tokenizer_init(&tokenizer, g, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        model_free(&model);
        return 1;
    }

    int prompt_ids[PROMPT_TOKENS_MAX];
    int n_prompt = 0;
    if (tokenizer.add_bos) prompt_ids[n_prompt++] = tokenizer.bos_id;
    int n_text = tokenizer_encode(&tokenizer, prompt, prompt_ids + n_prompt,
                                  PROMPT_TOKENS_MAX - n_prompt);
    if (n_text > 0) n_prompt += n_text;
    if (n_prompt < 1) {
        fprintf(stderr, "prompt encoded to %d tokens; expected at least 1\n",
                n_prompt);
        tokenizer_free(&tokenizer);
        model_free(&model);
        return 1;
    }

    Runtime *runtime = runtime_start(&model, n_prompt + n_predict, n_threads,
                                     err, sizeof err);
    if (!runtime) {
        fprintf(stderr, "%s\n", err);
        tokenizer_free(&tokenizer);
        model_free(&model);
        return 1;
    }

    double started = now_seconds();
    const float *logits = NULL;
    for (int i = 0; i < n_prompt; i++)
        logits = forward(runtime, prompt_ids[i], i);
    double prefilled = now_seconds();

    printf("%s", prompt);
    fflush(stdout);

    int generated = 0;
    for (int i = 0; i < n_predict; i++) {
        int token = argmax(logits, model.n_vocab);
        if (token == tokenizer.eos_id) break;
        char text[512];
        tokenizer_decode(&tokenizer, &token, 1, text, sizeof text);
        printf("%s", text);
        fflush(stdout);
        generated++;
        logits = forward(runtime, token, n_prompt + i);
    }
    double finished = now_seconds();

    double prefill_seconds = prefilled - started;
    double decode_seconds = finished - prefilled;
    printf("\n\nprefill: %d tokens in %.2fs (%.1f tok/s)\n", n_prompt,
           prefill_seconds, n_prompt / prefill_seconds);
    if (generated > 0)
        printf("decode:  %d tokens in %.2fs (%.2f tok/s)\n", generated,
               decode_seconds, generated / decode_seconds);

    runtime_stop(runtime);
    tokenizer_free(&tokenizer);
    model_free(&model);
    return 0;
}

int exec_main(int argc, char **argv) {
    int generating = strcmp(argv[0], "exec-run") == 0;
    if (argc < 2 || (generating ? argc > 5 : argc != 2)) {
        fprintf(stderr, "usage: %s MODEL.gguf%s\n", argv[0],
                generating ? " [PROMPT] [N_PREDICT] [THREADS]" : "");
        return 2;
    }
    char err[256];
    GgufFile g;
    if (!gguf_open(&g, argv[1], err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    if (generating) {
        const char *prompt = argc > 2 ? argv[2] : "The capital of France is";
        int n_predict = argc > 3 ? atoi(argv[3]) : DEFAULT_PREDICT_TOKENS;
        if (n_predict < 1) {
            fprintf(stderr, "N_PREDICT is %d; expected at least 1\n",
                    n_predict);
            gguf_close(&g);
            return 2;
        }
        int rc = run_prompt(&g, prompt, n_predict,
                            argc > 4 ? atoi(argv[4]) : 0);
        gguf_close(&g);
        return rc;
    }
    if (strcmp(argv[0], "exec-kernels") == 0) {
        int rc = kernels_test(&g);
        gguf_close(&g);
        return rc;
    }
    if (strcmp(argv[0], "exec-tokenize") == 0) {
        int rc = tokenize_test(&g);
        gguf_close(&g);
        return rc;
    }

    char arch[128] = "";
    char name[256] = "";
    gguf_meta_str(&g, "general.architecture", arch, sizeof arch);
    gguf_meta_str(&g, "general.name", name, sizeof name);
    printf("model: %s (%s), %llu tensors, file %.2f GB\n\n", name, arch,
           g.n_tensors, g.map_size / 1e9);

    printf("config:\n");
    for (unsigned i = 0; i < sizeof config_keys / sizeof config_keys[0]; i++)
        cfg(&g, arch, config_keys[i]);

    unsigned long long hist[64] = {0};
    unsigned long long bytes = 0;
    for (unsigned long long i = 0; i < g.n_tensors; i++) {
        hist[g.tensors[i].type]++;
        bytes += g.tensors[i].n_bytes;
    }
    printf("\nquant histogram:");
    for (int t = 0; t < n_quant_types; t++)
        if (hist[t]) printf("  %s x%llu", quant_types[t].name, hist[t]);
    printf("\ntensor bytes: %.2f GB (file %.2f GB)\n", bytes / 1e9,
           g.map_size / 1e9);

    printf("\nblk.0 (dense) tensors:\n");
    dump_layer_tensors(&g, 0);
    printf("\nblk.1 (moe) tensors:\n");
    dump_layer_tensors(&g, 1);

    gguf_close(&g);
    return 0;
}
