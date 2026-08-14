#include "forward.h"

#include "kernels.h"
#include "parallel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* One feed-forward whose output is summed into the layer's result: a dense
   block is a single branch of weight 1, an expert block is the routed experts
   plus the shared one. They all read the same input and none reads another's
   output, so every branch's gate and up rows are one parallel region and every
   branch's down rows are a second -- three regions per expert block instead of
   sixteen. `offset` is where the branch's rows start in the concatenated
   gate/up/activated buffers. */
typedef struct {
    const FeedForward *ffn;
    int matrix_index;
    int offset;
    float weight;
} Branch;

struct Runtime {
    const Model *model;
    ThreadPool *pool;
    RopeConfig rope;
    int n_ctx;
    int cache_width;

    uint16_t *cache;

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

    /* Quantization slots, indexed by worker inside a parallel region and by
       branch between the two feed-forward regions -- never both at once, so
       there are as many as the larger of the two needs. Sequential callers use
       slot 0 and are done with it before the region they feed returns. */
    unsigned char *activation_scratch;
    size_t activation_slot_bytes;
    Activation *activations;
};

static void *activation_slot(Runtime *runtime, int slot) {
    return runtime->activation_scratch +
           (size_t)slot * runtime->activation_slot_bytes;
}

static const void *matrix_at(const GgufTensor *tensor, int index) {
    size_t matrix_bytes = (size_t)tensor->dims[1] *
                          row_bytes(tensor->type, (int)tensor->dims[0]);
    return (const unsigned char *)tensor->data + (size_t)index * matrix_bytes;
}

static uint16_t *cache_slot(Runtime *runtime, int layer, int position) {
    return runtime->cache +
           ((size_t)layer * runtime->n_ctx + position) * runtime->cache_width;
}

typedef struct {
    float *out;
    const void *rows;
    unsigned type;
    int n_in;
    int n_out;
    const Activation *x;
} MatvecJob;

static void matvec_worker(void *state, int worker, int n_workers) {
    const MatvecJob *job = state;
    int begin = (int)((long long)job->n_out * worker / n_workers);
    int end = (int)((long long)job->n_out * (worker + 1) / n_workers);
    matvec(job->out, job->rows, job->type, job->n_in, begin, end, job->x);
}

static void run_matvec(Runtime *runtime, float *out, const GgufTensor *tensor,
                       int matrix_index, const float *x) {
    Activation activation;
    activation_set(&activation, activation_slot(runtime, 0), x,
                   (int)tensor->dims[0]);
    MatvecJob job = {out,
                     matrix_at(tensor, matrix_index),
                     tensor->type,
                     (int)tensor->dims[0],
                     (int)tensor->dims[1],
                     &activation};
    pool_run(runtime->pool, matvec_worker, &job);
}

typedef struct {
    Runtime *runtime;
    const Layer *layer;
    int layer_index;
    int position;
    int n_cached;
} AttentionJob;

/* One worker owns whole heads, so scoring, softmax, the weighted sum over the
   cache and the value projection all run without a barrier between them. */
static void attention_worker(void *state, int worker, int n_workers) {
    const AttentionJob *job = state;
    Runtime *runtime = job->runtime;
    const Model *model = runtime->model;
    int rank = model->kv_lora_rank;

    int head_begin = model->n_head * worker / n_workers;
    int head_end = model->n_head * (worker + 1) / n_workers;

    for (int head = head_begin; head < head_end; head++) {
        float *query = runtime->query + (size_t)head * model->head_dim_k;
        float *query_rope = query + model->qk_nope_dim;
        float *query_latent = runtime->query_latent + (size_t)head * rank;
        float *scores = runtime->scores + (size_t)head * runtime->n_ctx;
        float *attn_latent = runtime->attn_latent + (size_t)head * rank;

        Activation activation;
        rope_apply(query_rope, &runtime->rope, job->position);
        activation_set(&activation, activation_slot(runtime, worker), query,
                       model->qk_nope_dim);
        matvec(query_latent, matrix_at(job->layer->k_b, head),
               job->layer->k_b->type, model->qk_nope_dim, 0, rank, &activation);

        for (int p = 0; p < job->n_cached; p++) {
            const uint16_t *slot = cache_slot(runtime, job->layer_index, p);
            float score = dot_fp16(slot, query_latent, rank) +
                          dot_fp16(slot + rank, query_rope, model->qk_rope_dim);
            scores[p] = score * model->kq_scale;
        }
        softmax(scores, job->n_cached);

        memset(attn_latent, 0, (size_t)rank * sizeof *attn_latent);
        for (int p = 0; p < job->n_cached; p++)
            accumulate_fp16(attn_latent,
                            cache_slot(runtime, job->layer_index, p), scores[p],
                            rank);

        activation_set(&activation, activation_slot(runtime, worker),
                       attn_latent, rank);
        matvec(runtime->attn_out + (size_t)head * model->head_dim_v,
               matrix_at(job->layer->v_b, head), job->layer->v_b->type, rank, 0,
               model->head_dim_v, &activation);
    }
}

static void attention(Runtime *runtime, const Layer *layer, int layer_index,
                      int position) {
    const Model *model = runtime->model;
    int rank = model->kv_lora_rank;

    run_matvec(runtime, runtime->query, layer->attn_q, 0, runtime->normed);
    run_matvec(runtime, runtime->kv_projected, layer->kv_a_mqa, 0,
               runtime->normed);

    rmsnorm(runtime->kv_normed, runtime->kv_projected,
            layer->kv_a_norm->data, rank, model->rms_eps);
    rope_apply(runtime->kv_projected + rank, &runtime->rope, position);

    uint16_t *slot = cache_slot(runtime, layer_index, position);
    for (int i = 0; i < rank; i++) slot[i] = fp32_to_fp16(runtime->kv_normed[i]);
    for (int i = 0; i < model->qk_rope_dim; i++)
        slot[rank + i] = fp32_to_fp16(runtime->kv_projected[rank + i]);

    AttentionJob job = {runtime, layer, layer_index, position, position + 1};
    pool_run(runtime->pool, attention_worker, &job);

    run_matvec(runtime, runtime->projected, layer->attn_output, 0,
               runtime->attn_out);
}

static int branch_width(const Branch *branch) {
    return (int)branch->ffn->gate->dims[1];
}

/* Lays the branches end to end and returns how many rows they occupy in
   total, which is the row space the expand region divides between workers. */
static int branch_layout(Branch *branches, int n_branches) {
    int offset = 0;
    for (int b = 0; b < n_branches; b++) {
        branches[b].offset = offset;
        offset += branch_width(&branches[b]);
    }
    return offset;
}

typedef struct {
    Runtime *runtime;
    int n_branches;
    int total_ff;
    const Activation *x;
} ExpandJob;

/* A worker takes a slice of the concatenated row space and runs whichever
   branches it lands in, so the last branch is never left to one thread while
   the rest wait at a barrier. Gate and up share the slice, which lets the
   swiglu happen here rather than in a pass of its own. */
static void expand_worker(void *state, int worker, int n_workers) {
    const ExpandJob *job = state;
    Runtime *runtime = job->runtime;
    int n_embd = runtime->model->n_embd;
    int begin = (int)((long long)job->total_ff * worker / n_workers);
    int end = (int)((long long)job->total_ff * (worker + 1) / n_workers);

    for (int b = 0; b < job->n_branches; b++) {
        const Branch *branch = &runtime->branches[b];
        int lo = begin - branch->offset;
        int hi = end - branch->offset;
        if (lo < 0) lo = 0;
        if (hi > branch_width(branch)) hi = branch_width(branch);
        if (lo >= hi) continue;

        float *gate = runtime->gate + branch->offset;
        float *up = runtime->up + branch->offset;
        matvec(gate, matrix_at(branch->ffn->gate, branch->matrix_index),
               branch->ffn->gate->type, n_embd, lo, hi, job->x);
        matvec(up, matrix_at(branch->ffn->up, branch->matrix_index),
               branch->ffn->up->type, n_embd, lo, hi, job->x);
        swiglu(runtime->activated + branch->offset + lo, gate + lo, up + lo,
               hi - lo);
    }
}

typedef struct {
    Runtime *runtime;
    int n_branches;
} ContractJob;

/* Every branch projects back onto the same n_embd output, so here a worker
   owns output rows instead: it walks all the branches and sums them into the
   rows it owns, and the mixture needs no reduction afterwards. */
static void contract_worker(void *state, int worker, int n_workers) {
    const ContractJob *job = state;
    Runtime *runtime = job->runtime;
    int n_embd = runtime->model->n_embd;
    int begin = (int)((long long)n_embd * worker / n_workers);
    int end = (int)((long long)n_embd * (worker + 1) / n_workers);

    float *out = runtime->projected;
    memset(out + begin, 0, (size_t)(end - begin) * sizeof *out);
    for (int b = 0; b < job->n_branches; b++) {
        const Branch *branch = &runtime->branches[b];
        matvec(runtime->branch_out,
               matrix_at(branch->ffn->down, branch->matrix_index),
               branch->ffn->down->type, branch_width(branch), begin, end,
               &runtime->activations[b]);
        add_scaled(out + begin, runtime->branch_out + begin, branch->weight,
                   end - begin);
    }
}

static void run_branches(Runtime *runtime, int n_branches) {
    int total_ff = branch_layout(runtime->branches, n_branches);

    Activation input;
    activation_set(&input, activation_slot(runtime, 0), runtime->normed,
                   runtime->model->n_embd);
    ExpandJob expand = {runtime, n_branches, total_ff, &input};
    pool_run(runtime->pool, expand_worker, &expand);

    for (int b = 0; b < n_branches; b++)
        activation_set(&runtime->activations[b], activation_slot(runtime, b),
                       runtime->activated + runtime->branches[b].offset,
                       branch_width(&runtime->branches[b]));

    ContractJob contract = {runtime, n_branches};
    pool_run(runtime->pool, contract_worker, &contract);
}

/* Experts are ranked by probability plus a learned bias, but weighted by the
   probability alone -- the bias steers load balancing, not the mixture. */
static int select_experts(Runtime *runtime, const Layer *layer) {
    const Model *model = runtime->model;
    const float *bias = layer->router_bias->data;
    float *probs = runtime->router_probs;
    Branch *branches = runtime->branches;

    for (int e = 0; e < model->n_expert; e++)
        probs[e] = 1.0f / (1.0f + expf(-probs[e]));

    for (int slot = 0; slot < model->n_expert_used; slot++) {
        int best = -1;
        float best_score = 0;
        for (int e = 0; e < model->n_expert; e++) {
            int taken = 0;
            for (int s = 0; s < slot; s++)
                if (branches[s].matrix_index == e) taken = 1;
            if (taken) continue;
            float score = probs[e] + bias[e];
            if (best < 0 || score > best_score) {
                best = e;
                best_score = score;
            }
        }
        branches[slot].ffn = &layer->experts;
        branches[slot].matrix_index = best;
        branches[slot].weight = probs[best];
    }

    float scale = model->expert_weights_scale;
    if (model->expert_weights_norm) {
        float sum = 0;
        for (int slot = 0; slot < model->n_expert_used; slot++)
            sum += branches[slot].weight;
        if (sum > 0) scale /= sum;
    }
    for (int slot = 0; slot < model->n_expert_used; slot++)
        branches[slot].weight *= scale;

    branches[model->n_expert_used].ffn = &layer->shared_expert;
    branches[model->n_expert_used].matrix_index = 0;
    branches[model->n_expert_used].weight = 1.0f;
    return model->n_expert_used + 1;
}

static void feed_forward(Runtime *runtime, const Layer *layer) {
    if (!layer->has_experts) {
        runtime->branches[0].ffn = &layer->dense;
        runtime->branches[0].matrix_index = 0;
        runtime->branches[0].weight = 1.0f;
        run_branches(runtime, 1);
        return;
    }
    run_matvec(runtime, runtime->router_probs, layer->router, 0,
               runtime->normed);
    run_branches(runtime, select_experts(runtime, layer));
}

const float *forward(Runtime *runtime, int token, int position) {
    const Model *model = runtime->model;
    int n_embd = model->n_embd;

    const unsigned char *embedding =
        (const unsigned char *)model->token_embd->data +
        (size_t)token * row_bytes(model->token_embd->type, n_embd);
    dequant_row(embedding, model->token_embd->type, n_embd, runtime->residual);

    for (int index = 0; index < model->n_layer; index++) {
        const Layer *layer = &model->layers[index];

        rmsnorm(runtime->normed, runtime->residual, layer->attn_norm->data,
                n_embd, model->rms_eps);
        attention(runtime, layer, index, position);
        add_scaled(runtime->residual, runtime->projected, 1.0f, n_embd);

        rmsnorm(runtime->normed, runtime->residual, layer->ffn_norm->data,
                n_embd, model->rms_eps);
        feed_forward(runtime, layer);
        add_scaled(runtime->residual, runtime->projected, 1.0f, n_embd);
    }

    rmsnorm(runtime->normed, runtime->residual, model->output_norm->data,
            n_embd, model->rms_eps);
    run_matvec(runtime, runtime->logits, model->output, 0, runtime->normed);
    return runtime->logits;
}

static float *alloc_floats(size_t n, int *ok) {
    float *buffer = calloc(n, sizeof *buffer);
    if (!buffer) *ok = 0;
    return buffer;
}

Runtime *runtime_start(const Model *model, int n_ctx, int n_threads, char *err,
                       size_t errsz) {
    Runtime *runtime = calloc(1, sizeof *runtime);
    if (!runtime) {
        snprintf(err, errsz, "out of memory");
        return NULL;
    }
    runtime->model = model;
    runtime->n_ctx = n_ctx;
    runtime->cache_width = model->kv_lora_rank + model->qk_rope_dim;
    rope_init(&runtime->rope, model->rope_freq_base, model->rope_freq_scale,
              model->qk_rope_dim, model->rope_orig_ctx, model->rope_beta_fast,
              model->rope_beta_slow);

    if (n_threads < 1) n_threads = pool_default_workers();
    runtime->pool = pool_start(n_threads);

    /* The gate/up/activated buffers hold every branch of a block at once, so
       they are sized for whichever block concatenates to the most rows. */
    int n_branches = model->n_expert_used + model->n_expert_shared;
    if (n_branches < 1) n_branches = 1;
    int inner = n_branches * model->n_ff_expert;
    if (model->n_ff > inner) inner = model->n_ff;

    if (!runtime->pool) {
        snprintf(err, errsz, "could not start %d worker threads", n_threads);
        runtime_stop(runtime);
        return NULL;
    }

    int ok = 1;
    int widest = inner > model->n_head * model->head_dim_v
                     ? inner
                     : model->n_head * model->head_dim_v;
    int n_slots = pool_workers(runtime->pool) > n_branches
                      ? pool_workers(runtime->pool)
                      : n_branches;
    runtime->activation_slot_bytes = activation_bytes(widest);
    runtime->activation_scratch =
        calloc((size_t)n_slots, runtime->activation_slot_bytes);
    runtime->activations = calloc((size_t)n_branches,
                                  sizeof *runtime->activations);
    runtime->branches = calloc((size_t)n_branches, sizeof *runtime->branches);
    if (!runtime->activation_scratch || !runtime->activations ||
        !runtime->branches)
        ok = 0;

    size_t cache_elements = (size_t)model->n_layer * n_ctx *
                            runtime->cache_width;
    runtime->cache = calloc(cache_elements, sizeof *runtime->cache);
    if (!runtime->cache) ok = 0;

    runtime->residual = alloc_floats((size_t)model->n_embd, &ok);
    runtime->normed = alloc_floats((size_t)model->n_embd, &ok);
    runtime->query =
        alloc_floats((size_t)model->n_head * model->head_dim_k, &ok);
    runtime->query_latent =
        alloc_floats((size_t)model->n_head * model->kv_lora_rank, &ok);
    runtime->kv_projected = alloc_floats((size_t)runtime->cache_width, &ok);
    runtime->kv_normed = alloc_floats((size_t)model->kv_lora_rank, &ok);
    runtime->scores = alloc_floats((size_t)model->n_head * n_ctx, &ok);
    runtime->attn_latent =
        alloc_floats((size_t)model->n_head * model->kv_lora_rank, &ok);
    runtime->attn_out =
        alloc_floats((size_t)model->n_head * model->head_dim_v, &ok);
    runtime->projected = alloc_floats((size_t)model->n_embd, &ok);
    runtime->gate = alloc_floats((size_t)inner, &ok);
    runtime->up = alloc_floats((size_t)inner, &ok);
    runtime->activated = alloc_floats((size_t)inner, &ok);
    runtime->branch_out = alloc_floats((size_t)model->n_embd, &ok);
    runtime->logits = alloc_floats((size_t)model->n_vocab, &ok);
    if (model->n_expert > 0)
        runtime->router_probs = alloc_floats((size_t)model->n_expert, &ok);

    if (!ok) {
        snprintf(err, errsz,
                 "out of memory for %d-token context (kv cache needs %.2f GB)",
                 n_ctx, cache_elements * sizeof *runtime->cache / 1e9);
        runtime_stop(runtime);
        return NULL;
    }
    return runtime;
}

void runtime_stop(Runtime *runtime) {
    if (!runtime) return;
    pool_stop(runtime->pool);
    free(runtime->cache);
    free(runtime->residual);
    free(runtime->normed);
    free(runtime->query);
    free(runtime->query_latent);
    free(runtime->kv_projected);
    free(runtime->kv_normed);
    free(runtime->scores);
    free(runtime->attn_latent);
    free(runtime->attn_out);
    free(runtime->projected);
    free(runtime->gate);
    free(runtime->up);
    free(runtime->activated);
    free(runtime->router_probs);
    free(runtime->branch_out);
    free(runtime->logits);
    free(runtime->branches);
    free(runtime->activations);
    free(runtime->activation_scratch);
    free(runtime);
}
