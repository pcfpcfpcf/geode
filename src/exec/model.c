#include "model.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Fixed inside ggml's rotation, independent of the model's own multiplier. */
#define YARN_ROTATION_LOG_MULTIPLIER 0.1

/* Every lookup below records the first thing it could not satisfy and returns
   a harmless value, so model_load reports one error instead of unwinding at
   twenty call sites. */
typedef struct {
    const GgufFile *gguf;
    char arch[64];
    char problem[224];
} Loader;

static void fail(Loader *loader, const char *fmt, ...) {
    if (loader->problem[0]) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(loader->problem, sizeof loader->problem, fmt, ap);
    va_end(ap);
}

static int meta_int(Loader *loader, const char *key) {
    char full[160];
    unsigned long long value;
    snprintf(full, sizeof full, "%s.%s", loader->arch, key);
    if (gguf_meta_u64(loader->gguf, full, &value)) return (int)value;
    fail(loader, "missing metadata key %s", full);
    return 0;
}

static int meta_int_or(Loader *loader, const char *key, int fallback) {
    char full[160];
    unsigned long long value;
    snprintf(full, sizeof full, "%s.%s", loader->arch, key);
    return gguf_meta_u64(loader->gguf, full, &value) ? (int)value : fallback;
}

static double meta_float_or(Loader *loader, const char *key, double fallback) {
    char full[160];
    double value;
    snprintf(full, sizeof full, "%s.%s", loader->arch, key);
    return gguf_meta_f64(loader->gguf, full, &value) ? value : fallback;
}

static const GgufTensor *vector(Loader *loader, unsigned long long n,
                                const char *fmt, ...) {
    char name[GGUF_NAME_MAX];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);

    const GgufTensor *tensor = gguf_find(loader->gguf, name);
    if (!tensor) {
        fail(loader, "missing tensor %s", name);
        return NULL;
    }
    if (tensor->dims[0] != n)
        fail(loader, "%s: expected %llu elements, got %llu", name, n,
             tensor->dims[0]);
    return tensor;
}

/* Like vector, but a tensor some architectures simply do not have -- a router
   bias, say -- is not an error. */
static const GgufTensor *optional_vector(Loader *loader, unsigned long long n,
                                         const char *fmt, ...) {
    char name[GGUF_NAME_MAX];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);

    const GgufTensor *tensor = gguf_find(loader->gguf, name);
    if (!tensor) return NULL;
    if (tensor->dims[0] != n)
        fail(loader, "%s: expected %llu elements, got %llu", name, n,
             tensor->dims[0]);
    return tensor;
}

/* GGUF stores dims[0] as the contracted (input) dimension, dims[1] as the
   output dimension and dims[2] as a stack of independent matrices. */
static const GgufTensor *matrix(Loader *loader, unsigned long long n_in,
                                unsigned long long n_out,
                                unsigned long long n_matrices,
                                const char *fmt, ...) {
    char name[GGUF_NAME_MAX];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);

    const GgufTensor *tensor = gguf_find(loader->gguf, name);
    if (!tensor) {
        fail(loader, "missing tensor %s", name);
        return NULL;
    }
    unsigned long long stacked = tensor->dims[2] ? tensor->dims[2] : 1;
    if (tensor->dims[0] != n_in || tensor->dims[1] != n_out ||
        stacked != n_matrices)
        fail(loader, "%s: expected [%llu, %llu, %llu], got [%llu, %llu, %llu]",
             name, n_in, n_out, n_matrices, tensor->dims[0], tensor->dims[1],
             stacked);
    return tensor;
}

static void load_hparams(Loader *loader, Model *model) {
    model->n_layer = meta_int(loader, "block_count");
    model->n_embd = meta_int(loader, "embedding_length");
    model->n_head = meta_int(loader, "attention.head_count");
    model->n_ff = meta_int(loader, "feed_forward_length");
    model->n_dense_layer = meta_int_or(loader, "leading_dense_block_count", 0);

    GgufArray tokens;
    if (gguf_meta_arr(loader->gguf, "tokenizer.ggml.tokens", &tokens))
        model->n_vocab = (int)tokens.remaining;
    else
        model->n_vocab = meta_int(loader, "vocab_size");

    if (model->attention == ATTN_MLA) {
        model->kv_lora_rank = meta_int(loader, "attention.kv_lora_rank");
        model->head_dim_k = meta_int_or(loader, "attention.key_length_mla",
                                        meta_int(loader, "attention.key_length"));
        model->head_dim_v = meta_int_or(loader, "attention.value_length_mla",
                                        meta_int(loader, "attention.value_length"));
        model->qk_rope_dim = meta_int(loader, "rope.dimension_count");
        model->qk_nope_dim = model->head_dim_k - model->qk_rope_dim;
    } else {
        model->n_head_kv = meta_int(loader, "attention.head_count_kv");
        model->head_dim_k = meta_int(loader, "attention.key_length");
        model->head_dim_v = meta_int(loader, "attention.value_length");
        /* Qwen3 rotates the whole head; llama.cpp defaults the same way when
           the metadata omits rope.dimension_count. */
        model->qk_rope_dim =
            meta_int_or(loader, "rope.dimension_count", model->head_dim_k);
    }

    model->n_expert = meta_int_or(loader, "expert_count", 0);
    model->n_expert_used = meta_int_or(loader, "expert_used_count", 0);
    model->n_expert_shared = meta_int_or(loader, "expert_shared_count", 0);
    model->n_ff_expert = meta_int_or(loader, "expert_feed_forward_length", 0);
    model->gating = meta_int_or(loader, "expert_gating_func",
                                model->attention == ATTN_MLA ? GATING_SIGMOID
                                                             : GATING_SOFTMAX);
    model->expert_weights_norm = meta_int_or(loader, "expert_weights_norm",
                                             model->attention == ATTN_MLA ? 0 : 1);
    model->expert_weights_scale =
        (float)meta_float_or(loader, "expert_weights_scale", 1.0);

    if (model->gating != GATING_SOFTMAX && model->gating != GATING_SIGMOID)
        fail(loader,
             "expert_gating_func %d is not supported; this executor routes "
             "with softmax (%d) or sigmoid (%d)",
             model->gating, GATING_SOFTMAX, GATING_SIGMOID);

    model->rms_eps =
        (float)meta_float_or(loader, "attention.layer_norm_rms_epsilon", 1e-6);
    model->rope_freq_base =
        (float)meta_float_or(loader, "rope.freq_base", 10000.0);
    model->rope_orig_ctx =
        meta_int_or(loader, "rope.scaling.original_context_length", 0);
    model->rope_beta_fast =
        (float)meta_float_or(loader, "rope.scaling.yarn_beta_fast", 32.0);
    model->rope_beta_slow =
        (float)meta_float_or(loader, "rope.scaling.yarn_beta_slow", 1.0);

    double scaling_factor = meta_float_or(loader, "rope.scaling.factor", 1.0);
    model->rope_freq_scale =
        scaling_factor > 0 ? 1.0f / (float)scaling_factor : 1.0f;

    /* YaRN corrects attention magnitude in two places. Inside the rotation the
       correction is a hardcoded 0.1 log term divided straight back out by
       attn_factor, so the rotation nets to 1.0 -- which is why rope_apply()
       leaves magnitudes alone. Only the score scale keeps a correction, and it
       uses the multiplier the model was trained with rather than the hardcoded
       one; the two coincide here, leaving plain 1/sqrt(head_dim_k). */
    double log_multiplier =
        meta_float_or(loader, "rope.scaling.yarn_log_multiplier", 0.0);
    double log_scale = log(1.0 / model->rope_freq_scale);
    double attn_factor = 1.0 / (1.0 + YARN_ROTATION_LOG_MULTIPLIER * log_scale);
    double mscale = attn_factor * (1.0 + log_multiplier * log_scale);
    model->kq_scale = (float)(mscale * mscale / sqrt((double)model->head_dim_k));
}

static void load_layer(Loader *loader, Model *model, Layer *layer, int index) {
    int n_embd = model->n_embd;
    int n_expert = model->n_expert;

    layer->attn_norm = vector(loader, n_embd, "blk.%d.attn_norm.weight", index);
    layer->attn_q = matrix(loader, n_embd, model->n_head * model->head_dim_k, 1,
                           "blk.%d.attn_q.weight", index);
    if (model->attention == ATTN_MLA) {
        layer->kv_a_mqa =
            matrix(loader, n_embd, model->kv_lora_rank + model->qk_rope_dim, 1,
                   "blk.%d.attn_kv_a_mqa.weight", index);
        layer->kv_a_norm = vector(loader, model->kv_lora_rank,
                                  "blk.%d.attn_kv_a_norm.weight", index);
        layer->k_b = matrix(loader, model->qk_nope_dim, model->kv_lora_rank,
                            model->n_head, "blk.%d.attn_k_b.weight", index);
        layer->v_b = matrix(loader, model->kv_lora_rank, model->head_dim_v,
                            model->n_head, "blk.%d.attn_v_b.weight", index);
    } else {
        layer->attn_q_norm = vector(loader, model->head_dim_k,
                                    "blk.%d.attn_q_norm.weight", index);
        layer->attn_k = matrix(loader, n_embd,
                               model->n_head_kv * model->head_dim_k, 1,
                               "blk.%d.attn_k.weight", index);
        layer->attn_k_norm = vector(loader, model->head_dim_k,
                                    "blk.%d.attn_k_norm.weight", index);
        layer->attn_v = matrix(loader, n_embd,
                               model->n_head_kv * model->head_dim_v, 1,
                               "blk.%d.attn_v.weight", index);
    }
    layer->attn_output = matrix(loader, model->n_head * model->head_dim_v,
                                n_embd, 1, "blk.%d.attn_output.weight", index);
    layer->ffn_norm = vector(loader, n_embd, "blk.%d.ffn_norm.weight", index);

    layer->has_experts = index >= model->n_dense_layer;
    if (!layer->has_experts) {
        layer->dense.gate = matrix(loader, n_embd, model->n_ff, 1,
                                   "blk.%d.ffn_gate.weight", index);
        layer->dense.up = matrix(loader, n_embd, model->n_ff, 1,
                                 "blk.%d.ffn_up.weight", index);
        layer->dense.down = matrix(loader, model->n_ff, n_embd, 1,
                                   "blk.%d.ffn_down.weight", index);
        return;
    }

    layer->router = matrix(loader, n_embd, n_expert, 1,
                           "blk.%d.ffn_gate_inp.weight", index);
    layer->router_bias =
        optional_vector(loader, n_expert, "blk.%d.exp_probs_b.bias", index);
    layer->experts.gate = matrix(loader, n_embd, model->n_ff_expert, n_expert,
                                 "blk.%d.ffn_gate_exps.weight", index);
    layer->experts.up = matrix(loader, n_embd, model->n_ff_expert, n_expert,
                               "blk.%d.ffn_up_exps.weight", index);
    layer->experts.down = matrix(loader, model->n_ff_expert, n_embd, n_expert,
                                 "blk.%d.ffn_down_exps.weight", index);
    if (model->n_expert_shared > 0) {
        int n_shared = model->n_ff_expert * model->n_expert_shared;
        layer->shared_expert.gate = matrix(loader, n_embd, n_shared, 1,
                                           "blk.%d.ffn_gate_shexp.weight", index);
        layer->shared_expert.up = matrix(loader, n_embd, n_shared, 1,
                                         "blk.%d.ffn_up_shexp.weight", index);
        layer->shared_expert.down = matrix(loader, n_shared, n_embd, 1,
                                           "blk.%d.ffn_down_shexp.weight", index);
    }
}

/* The layer's deterministic feed-forward: the dense block's own ffn, or the
   MoE layer's shared expert. Every token reads it, so strategies place it on
   their fastest tier; everything stochastic stays behind. */
const FeedForward *model_base_ffn(const Model *model, const Layer *layer) {
    if (!layer->has_experts) return &layer->dense;
    return model->n_expert_shared > 0 ? &layer->shared_expert : NULL;
}

int model_load(Model *model, const GgufFile *gguf, char *err, size_t errsz) {
    memset(model, 0, sizeof *model);

    Loader loader;
    memset(&loader, 0, sizeof loader);
    loader.gguf = gguf;
    if (!gguf_meta_str(gguf, "general.architecture", loader.arch,
                       sizeof loader.arch)) {
        snprintf(err, errsz, "no general.architecture in model");
        return 0;
    }
    if (strcmp(loader.arch, "deepseek2") == 0)
        model->attention = ATTN_MLA;
    else if (strcmp(loader.arch, "qwen3moe") == 0)
        model->attention = ATTN_GQA;
    else {
        snprintf(err, errsz,
                 "architecture %s is not supported; this executor runs "
                 "deepseek2 (multi-head latent attention) and qwen3moe "
                 "(grouped-query attention)",
                 loader.arch);
        return 0;
    }

    load_hparams(&loader, model);
    if (loader.problem[0]) {
        snprintf(err, errsz, "%s", loader.problem);
        return 0;
    }

    model->token_embd = matrix(&loader, model->n_embd, model->n_vocab, 1,
                               "token_embd.weight");
    model->output_norm = vector(&loader, model->n_embd, "output_norm.weight");
    model->output =
        matrix(&loader, model->n_embd, model->n_vocab, 1, "output.weight");

    model->layers = calloc((size_t)model->n_layer, sizeof *model->layers);
    if (!model->layers) {
        snprintf(err, errsz, "out of memory for %d layers", model->n_layer);
        return 0;
    }
    for (int index = 0; index < model->n_layer; index++)
        load_layer(&loader, model, &model->layers[index], index);

    if (loader.problem[0]) {
        snprintf(err, errsz, "%s", loader.problem);
        model_free(model);
        return 0;
    }
    return 1;
}

void model_free(Model *model) {
    free(model->layers);
    model->layers = NULL;
}
