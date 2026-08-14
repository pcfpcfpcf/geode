#include "forward.h"

#include "kernels.h"
#include "parallel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    int index;
    float weight;
} ExpertChoice;

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
    float *expert_out;
    float *logits;
    ExpertChoice *chosen;
};

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
    const float *x;
} MatvecJob;

static void matvec_worker(void *state, int worker, int n_workers) {
    const MatvecJob *job = state;
    int begin = (int)((long long)job->n_out * worker / n_workers);
    int end = (int)((long long)job->n_out * (worker + 1) / n_workers);
    matvec(job->out, job->rows, job->type, job->n_in, begin, end, job->x);
}

static void run_matvec(Runtime *runtime, float *out, const GgufTensor *tensor,
                       int matrix_index, const float *x) {
    MatvecJob job = {out,
                     matrix_at(tensor, matrix_index),
                     tensor->type,
                     (int)tensor->dims[0],
                     (int)tensor->dims[1],
                     x};
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

        rope_apply(query_rope, &runtime->rope, job->position);
        matvec(query_latent, matrix_at(job->layer->k_b, head),
               job->layer->k_b->type, model->qk_nope_dim, 0, rank, query);

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

        matvec(runtime->attn_out + (size_t)head * model->head_dim_v,
               matrix_at(job->layer->v_b, head), job->layer->v_b->type, rank, 0,
               model->head_dim_v, attn_latent);
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

static void feed_forward(Runtime *runtime, const FeedForward *ffn,
                         int matrix_index, const float *x, float *out) {
    run_matvec(runtime, runtime->gate, ffn->gate, matrix_index, x);
    run_matvec(runtime, runtime->up, ffn->up, matrix_index, x);
    swiglu(runtime->activated, runtime->gate, runtime->up,
           (int)ffn->gate->dims[1]);
    run_matvec(runtime, out, ffn->down, matrix_index, runtime->activated);
}

/* Experts are ranked by probability plus a learned bias, but weighted by the
   probability alone -- the bias steers load balancing, not the mixture. */
static void select_experts(Runtime *runtime, const float *bias) {
    const Model *model = runtime->model;
    float *probs = runtime->router_probs;

    for (int e = 0; e < model->n_expert; e++)
        probs[e] = 1.0f / (1.0f + expf(-probs[e]));

    for (int slot = 0; slot < model->n_expert_used; slot++) {
        int best = -1;
        float best_score = 0;
        for (int e = 0; e < model->n_expert; e++) {
            int taken = 0;
            for (int s = 0; s < slot; s++)
                if (runtime->chosen[s].index == e) taken = 1;
            if (taken) continue;
            float score = probs[e] + bias[e];
            if (best < 0 || score > best_score) {
                best = e;
                best_score = score;
            }
        }
        runtime->chosen[slot].index = best;
        runtime->chosen[slot].weight = probs[best];
    }

    float scale = model->expert_weights_scale;
    if (model->expert_weights_norm) {
        float sum = 0;
        for (int slot = 0; slot < model->n_expert_used; slot++)
            sum += runtime->chosen[slot].weight;
        if (sum > 0) scale /= sum;
    }
    for (int slot = 0; slot < model->n_expert_used; slot++)
        runtime->chosen[slot].weight *= scale;
}

static void mixture_of_experts(Runtime *runtime, const Layer *layer) {
    const Model *model = runtime->model;

    run_matvec(runtime, runtime->router_probs, layer->router, 0,
               runtime->normed);
    select_experts(runtime, layer->router_bias->data);

    memset(runtime->projected, 0,
           (size_t)model->n_embd * sizeof *runtime->projected);
    for (int slot = 0; slot < model->n_expert_used; slot++) {
        feed_forward(runtime, &layer->experts, runtime->chosen[slot].index,
                     runtime->normed, runtime->expert_out);
        add_scaled(runtime->projected, runtime->expert_out,
                   runtime->chosen[slot].weight, model->n_embd);
    }
    feed_forward(runtime, &layer->shared_expert, 0, runtime->normed,
                 runtime->expert_out);
    add_scaled(runtime->projected, runtime->expert_out, 1.0f, model->n_embd);
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
        if (layer->has_experts)
            mixture_of_experts(runtime, layer);
        else
            feed_forward(runtime, &layer->dense, 0, runtime->normed,
                         runtime->projected);
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

    if (n_threads < 1) n_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    runtime->pool = pool_start(n_threads);

    int inner = model->n_ff > model->n_ff_expert ? model->n_ff
                                                 : model->n_ff_expert;
    int shared = model->n_ff_expert * model->n_expert_shared;
    if (shared > inner) inner = shared;

    int ok = runtime->pool != NULL;
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
    runtime->expert_out = alloc_floats((size_t)model->n_embd, &ok);
    runtime->logits = alloc_floats((size_t)model->n_vocab, &ok);
    if (model->n_expert > 0) {
        runtime->router_probs = alloc_floats((size_t)model->n_expert, &ok);
        runtime->chosen = calloc((size_t)model->n_expert_used,
                                 sizeof *runtime->chosen);
        if (!runtime->chosen) ok = 0;
    }

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
    free(runtime->expert_out);
    free(runtime->logits);
    free(runtime->chosen);
    free(runtime);
}
