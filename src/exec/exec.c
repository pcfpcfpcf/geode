#include "gguf.h"
#include "kernels.h"
#include "modules.h"
#include "quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    float got = gemv_row(t->data, t->type, n, x);
    float want = 0;
    for (int i = 0; i < n; i++) want += deq[i] * x[i];
    if (fabsf(got - want) > 1e-3f * (1.0f + fabsf(want))) {
        printf("    gemv MISMATCH: got %.6f want %.6f\n", got, want);
        ok = 0;
    } else {
        printf("    gemv ok: %.6f\n", got);
    }
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

int exec_main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    char err[256];
    GgufFile g;
    if (!gguf_open(&g, argv[1], err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    if (strcmp(argv[0], "exec-kernels") == 0) {
        int rc = kernels_test(&g);
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
