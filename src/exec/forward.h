#ifndef GEODE_FORWARD_H
#define GEODE_FORWARD_H

#include "kernels.h"
#include "model.h"
#include "parallel.h"

typedef struct Runtime Runtime;

/* The tier a strategy places attention on. The contract is the CPU one's:
   reads `normed` (and `normed_batch`'s token count is n_tokens) and leaves
   the result in `projected`, for every token of the chunk. */
typedef void (*AttentionFn)(Runtime *runtime, const Layer *layer,
                            int layer_index, int position, int n_tokens);

/* The tier a strategy places the layer's deterministic feed-forward on --
   the dense block's ffn or the MoE layer's shared expert. Same contract:
   reads `normed`, and leaves the layer's whole feed-forward result in
   `projected` for every token, routed experts included. */
typedef void (*FfnFn)(Runtime *runtime, const Layer *layer, int layer_index,
                      int n_tokens);

/* Whatever a strategy attached to this runtime beyond the cpu state -- the
   hybrid's device handle. Opaque: forward knows nothing about the layer
   above it. */

/* Regrouping the chunk by expert is what makes a routed layer affordable: four
   tokens rarely read four *different* experts, so one sweep of the stack per
   chunk beats one per token. `token_begin` indexes both the concatenated token
   list and the gate/up/activated rows. */
typedef struct {
    const FeedForward *ffn;
    int matrix_index;
    int token_begin;
    int n_tokens;
    long long work_begin;
    size_t scratch_begin;
} Branch;

struct Runtime {
    const Model *model;
    void *device;
    ThreadPool *pool;
    RopeConfig rope;
    int n_ctx;
    int cache_width;
    int n_segments;
    int query_quant_width;
    int query_segments;
    int max_tokens;

    uint8_t *cache;
    float *cache_scale;
    float *cache_block;
    float *cos_sin;
    int8_t *query_quants;
    QuantizedQuery *query_scale;

    float *residual;
    float *normed;
    float *query;
    float *query_latent;
    float *kv_projected;
    float *kv_normed;
    float *scores;
    float *attn_latent;
    float *attn_out;
    float *projected;
    float *gate;
    float *up;
    float *activated;
    float *router_probs;
    float *branch_out;
    float *logits;

    Branch *branches;
    int *branch_tokens;
    float *branch_weights;
    int *chosen;
    float *chosen_weight;

    /* `head_scratch` is per worker rather than per chunk: the per-head matrices
       run inside a parallel region rather than across one. */
    ActivationBatch normed_batch;
    ActivationBatch *branch_activations;
    ActivationBatch heads_batch;
    unsigned char *normed_scratch;
    unsigned char *activated_scratch;
    unsigned char *heads_scratch;
    unsigned char *head_scratch;
    size_t head_scratch_bytes;
};

/* One matrix of a stacked tensor: gate/up/down per expert, k_b/v_b per head.
   Matrices are dims[1] rows of dims[0] elements, stacked along dims[2]. */
const void *matrix_at(const GgufTensor *tensor, int index);

Runtime *runtime_start(const Model *model, int n_ctx, int n_threads, char *err,
                       size_t errsz);
void runtime_stop(Runtime *runtime);

/* The widest chunk `forward` accepts. Prefill reads every weight once per
   chunk instead of once per token, so a prompt fed one token at a time costs
   what decoding it would; feeding it in chunks is what makes prefill cheaper
   than decode rather than the same price.

   Wider is not uniformly better, and the curve here is shallow -- 16, 32, 64
   and 100 measured 34, 36, 38 and 36 tok/s on the target box. Two effects
   cross. Attention and the shared expert amortize across the whole chunk, and
   a chunk this wide gives each routed expert about a tile of tokens rather
   than the one or two that fall back to a vector at a time. Against that, 64
   tokens already route to nearly every expert in the stack, so past here an
   extra token brings in as much expert weight as it saves, and the activations
   still have to stay in cache for the row-held-across-vectors loop to pay. */
#define PREFILL_CHUNK 64

/* Runs `n_tokens` consecutive tokens starting at `position` through every
   layer, appending to the kv cache as it goes. Returns logits over the
   vocabulary for the *last* token of the chunk, valid until the next call --
   the earlier ones are never sampled, and the output matrix is the widest in
   the model. Positions must be fed in order from 0: the cache holds no gaps. */
const float *forward(Runtime *runtime, const int *tokens, int position,
                     int n_tokens);

/* The cpu's own attention, for tests that compare a strategy's tier split
   against the reference one layer at a time. */
void forward_attention_cpu(Runtime *runtime, const Layer *layer,
                           int layer_index, int position, int n_tokens);

/* The cpu's own feed-forward: the shared expert and the dense block's ffn
   alongside the routed experts. `skip_base` leaves the deterministic half
   out, for the strategy that placed it on another tier. */
void forward_feed_forward_cpu(Runtime *runtime, const Layer *layer,
                              int n_tokens, int skip_base);

/* The same pass with a tier swapped per hook: a strategy splits the layer
   loop here and nowhere else -- everything around the hooks (embedding,
   rope tables, norms, the output projection) is tier-agnostic. Either hook
   may be NULL for the cpu's own. */
const float *forward_with(Runtime *runtime, const int *tokens, int position,
                          int n_tokens, AttentionFn attention, FfnFn ffn);

#endif
