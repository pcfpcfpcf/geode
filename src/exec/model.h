#ifndef GEODE_MODEL_H
#define GEODE_MODEL_H

#include "gguf.h"

/* ggml's expert gating enum. */
#define GATING_SOFTMAX 0
#define GATING_SIGMOID 2

/* How a layer attends. MLA compresses keys and values into one latent shared
   by every head; GQA projects separate per-head keys and values. */
typedef enum {
    ATTN_MLA,
    ATTN_GQA,
} AttentionKind;

/* A SwiGLU feed-forward: down(silu(gate(x)) * up(x)). Dense blocks, the shared
   expert and the routed expert stack are all this shape; the expert stack
   carries one matrix per expert in its last dimension. */
typedef struct {
    const GgufTensor *gate;
    const GgufTensor *up;
    const GgufTensor *down;
} FeedForward;

typedef struct {
    const GgufTensor *attn_norm;
    const GgufTensor *attn_q;
    const GgufTensor *attn_q_norm;
    const GgufTensor *attn_k;
    const GgufTensor *attn_k_norm;
    const GgufTensor *attn_v;
    const GgufTensor *kv_a_mqa;
    const GgufTensor *kv_a_norm;
    const GgufTensor *k_b;
    const GgufTensor *v_b;
    const GgufTensor *attn_output;
    const GgufTensor *ffn_norm;

    const GgufTensor *router;
    const GgufTensor *router_bias;
    FeedForward dense;
    FeedForward experts;
    FeedForward shared_expert;

    int has_experts;
} Layer;

typedef struct {
    int n_layer;
    int n_embd;
    int n_head;
    int n_head_kv;
    int n_ff;
    int n_ff_expert;
    int n_vocab;
    int n_dense_layer;

    AttentionKind attention;

    /* Multi-head latent attention. Keys and values are projected down to one
       kv_lora_rank latent shared by every head, plus one rotary key also
       shared by every head -- so a cache slot is kv_lora_rank + qk_rope_dim
       wide, not n_head * head_dim_k. Each head's key splits into qk_nope_dim
       rotary-free dims (reconstructed from the latent through k_b) and
       qk_rope_dim rotary dims. GQA leaves kv_lora_rank and qk_nope_dim zero. */
    int kv_lora_rank;
    int qk_rope_dim;
    int qk_nope_dim;
    int head_dim_k;
    int head_dim_v;

    int n_expert;
    int n_expert_used;
    int n_expert_shared;
    int gating;
    int expert_weights_norm;
    float expert_weights_scale;

    float rms_eps;
    float rope_freq_base;
    float rope_freq_scale;
    int rope_orig_ctx;
    float rope_beta_fast;
    float rope_beta_slow;
    float kq_scale;

    const GgufTensor *token_embd;
    const GgufTensor *output_norm;
    const GgufTensor *output;
    Layer *layers;
} Model;

int model_load(Model *model, const GgufFile *gguf, char *err, size_t errsz);
void model_free(Model *model);

#endif
