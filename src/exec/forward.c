#include "forward.h"

#include "kernels.h"
#include "parallel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const void *matrix_at(const GgufTensor *tensor, int index) {
    size_t matrix_bytes = (size_t)tensor->dims[1] *
                          row_bytes(tensor->type, (int)tensor->dims[0]);
    return (const unsigned char *)tensor->data + (size_t)index * matrix_bytes;
}

/* A slot's latent and its rotary key are quantized apart: they are unrelated
   projections, and one scale over both would cost whichever came out smaller
   most of its eight bits. Their queries split the same way. */
enum { SLOT_LATENT, SLOT_ROTARY, SLOT_SEGMENTS };

static uint8_t *cache_slot(Runtime *runtime, int layer, int position) {
    return runtime->cache +
           ((size_t)layer * runtime->n_ctx + position) * runtime->cache_width;
}

static float *slot_scales(Runtime *runtime, int layer, int position) {
    return runtime->cache_scale +
           ((size_t)layer * runtime->n_ctx + position) * runtime->n_segments;
}

static int8_t *query_quants(Runtime *runtime, int token, int head) {
    return runtime->query_quants +
           ((size_t)token * runtime->model->n_head + head) *
               runtime->query_quant_width;
}

static QuantizedQuery *query_scales(Runtime *runtime, int token, int head) {
    return runtime->query_scale +
           ((size_t)token * runtime->model->n_head + head) *
               runtime->query_segments;
}

/* An integer dot arrives with the bias the cache's unsigned quants put into it
   and with neither operand's scale applied. */
static float scaled_dot(int32_t dot, const QuantizedQuery *query, float scale) {
    return ((float)dot - query->bias) * query->scale * scale;
}

static int cache_rows(int n_cached) {
    return (n_cached + CACHE_ROWS - 1) / CACHE_ROWS * CACHE_ROWS;
}

static float *score_row(Runtime *runtime, int head, int token, int n_tokens) {
    return runtime->scores +
           ((size_t)head * n_tokens + token) * runtime->n_ctx;
}

typedef struct {
    float *out;
    size_t out_stride;
    const void *rows;
    unsigned type;
    int n_in;
    int n_out;
    const ActivationBatch *x;
} MatmulJob;

static void matmul_worker(void *state, int worker, int n_workers) {
    const MatmulJob *job = state;
    int begin = (int)((long long)job->n_out * worker / n_workers);
    int end = (int)((long long)job->n_out * (worker + 1) / n_workers);
    matmul(job->out, job->out_stride, job->rows, job->type, job->n_in, begin,
           end, job->x);
}

/* Input arrives quantized: the matrices reading the same chunk would each
   otherwise pay for it. */
static void run_matmul(Runtime *runtime, float *out, size_t out_stride,
                       const GgufTensor *tensor, int matrix_index,
                       const ActivationBatch *x) {
    MatmulJob job = {out,
                     out_stride,
                     matrix_at(tensor, matrix_index),
                     tensor->type,
                     (int)tensor->dims[0],
                     (int)tensor->dims[1],
                     x};
    pool_run(runtime->pool, matmul_worker, &job);
}

static void normalize(Runtime *runtime, const GgufTensor *weight,
                      int n_tokens) {
    const Model *model = runtime->model;
    for (int t = 0; t < n_tokens; t++)
        rmsnorm(runtime->normed + (size_t)t * model->n_embd,
                runtime->residual + (size_t)t * model->n_embd, weight->data,
                model->n_embd, model->rms_eps);
    activation_set(&runtime->normed_batch, runtime->normed_scratch,
                   runtime->normed, (size_t)model->n_embd, model->n_embd,
                   n_tokens);
}

typedef struct {
    Runtime *runtime;
    const Layer *layer;
    int layer_index;
    int position;
    int n_tokens;
    int n_cached;
} AttentionJob;

/* Cache passes outer, heads inner: the fold's expansion to floats would
   otherwise be multiplied by the head count. Splitting them by data instead --
   scoring by position, folding by latent column -- costs 14% of decode at 900
   and at 2300 positions, so the read stays repeated per worker. Softmax zeroes
   the rows past a token's position, which is what lets a block run whole. */
static void attention_mla_worker(void *state, int worker, int n_workers) {
    const AttentionJob *job = state;
    Runtime *runtime = job->runtime;
    const Model *model = runtime->model;
    int rank = model->kv_lora_rank;
    int n_tokens = job->n_tokens;
    int query_width = model->n_head * model->head_dim_k;
    int latent_width = model->n_head * rank;
    int n_rows = cache_rows(job->n_cached);
    int width = runtime->cache_width;
    float *block = runtime->cache_block + (size_t)worker * CACHE_ROWS * width;
    void *scratch =
        runtime->head_scratch + (size_t)worker * runtime->head_scratch_bytes;

    int head_begin = model->n_head * worker / n_workers;
    int head_end = model->n_head * (worker + 1) / n_workers;

    for (int head = head_begin; head < head_end; head++) {
        float *query = runtime->query + (size_t)head * model->head_dim_k;
        for (int t = 0; t < n_tokens; t++)
            rope_apply(query + (size_t)t * query_width + model->qk_nope_dim,
                       runtime->cos_sin + (size_t)t * model->qk_rope_dim,
                       model->qk_rope_dim);

        ActivationBatch nope;
        activation_set(&nope, scratch, query, (size_t)query_width,
                       model->qk_nope_dim, n_tokens);
        matmul(runtime->query_latent + (size_t)head * rank,
               (size_t)latent_width, matrix_at(job->layer->k_b, head),
               job->layer->k_b->type, model->qk_nope_dim, 0, rank, &nope);

        /* Once per head, then dotted against every cached position: what the
           rounding costs is paid here and what it saves scales with context. */
        for (int t = 0; t < n_tokens; t++) {
            int8_t *quants = query_quants(runtime, t, head);
            QuantizedQuery *quantized = query_scales(runtime, t, head);
            quantize_query(&quantized[SLOT_LATENT], quants,
                           runtime->query_latent + (size_t)t * latent_width +
                               (size_t)head * rank,
                           rank);
            quantize_query(&quantized[SLOT_ROTARY], quants + rank,
                           query + (size_t)t * query_width + model->qk_nope_dim,
                           model->qk_rope_dim);
        }
    }

    for (int p = 0; p < n_rows; p += CACHE_ROWS) {
        const uint8_t *slot = cache_slot(runtime, job->layer_index, p);
        const float *scales = slot_scales(runtime, job->layer_index, p);
        for (int head = head_begin; head < head_end; head++)
            for (int t = 0; t < n_tokens; t++) {
                const int8_t *quants = query_quants(runtime, t, head);
                const QuantizedQuery *query = query_scales(runtime, t, head);
                int32_t latent[CACHE_ROWS], rotary[CACHE_ROWS];
                dot_int8_rows(latent, slot, width, quants, rank);
                dot_int8_rows(rotary, slot + rank, width, quants + rank,
                              model->qk_rope_dim);
                float *row = score_row(runtime, head, t, n_tokens) + p;
                for (int r = 0; r < CACHE_ROWS; r++) {
                    const float *scale =
                        scales + (size_t)r * runtime->n_segments;
                    row[r] = (scaled_dot(latent[r], &query[SLOT_LATENT],
                                         scale[SLOT_LATENT]) +
                              scaled_dot(rotary[r], &query[SLOT_ROTARY],
                                         scale[SLOT_ROTARY])) *
                             model->kq_scale;
                }
            }
    }

    for (int head = head_begin; head < head_end; head++)
        for (int t = 0; t < n_tokens; t++) {
            float *row = score_row(runtime, head, t, n_tokens);
            int attended = job->position + t + 1;
            softmax(row, attended);
            memset(row + attended, 0,
                   (size_t)(n_rows - attended) * sizeof *row);
            memset(runtime->attn_latent + (size_t)t * latent_width +
                       (size_t)head * rank,
                   0, (size_t)rank * sizeof *runtime->attn_latent);
        }

    for (int p = 0; p < n_rows; p += CACHE_ROWS) {
        for (int r = 0; r < CACHE_ROWS; r++)
            expand_int8(
                block + (size_t)r * width,
                cache_slot(runtime, job->layer_index, p + r),
                slot_scales(runtime, job->layer_index, p + r)[SLOT_LATENT],
                rank);
        for (int head = head_begin; head < head_end; head++)
            for (int t = 0; t < n_tokens; t++)
                add_scaled_rows(runtime->attn_latent +
                                    (size_t)t * latent_width +
                                    (size_t)head * rank,
                                block, width,
                                score_row(runtime, head, t, n_tokens) + p,
                                rank);
    }

    for (int head = head_begin; head < head_end; head++) {
        ActivationBatch latent;
        activation_set(&latent, scratch,
                       runtime->attn_latent + (size_t)head * rank,
                       (size_t)latent_width, rank, n_tokens);
        matmul(runtime->attn_out + (size_t)head * model->head_dim_v,
               (size_t)model->n_head * model->head_dim_v,
               matrix_at(job->layer->v_b, head), job->layer->v_b->type, rank, 0,
               model->head_dim_v, &latent);
    }
}

static void attention_mla(Runtime *runtime, const Layer *layer,
                          int layer_index, int position, int n_tokens) {
    const Model *model = runtime->model;
    int rank = model->kv_lora_rank;

    run_matmul(runtime, runtime->kv_projected, (size_t)runtime->cache_width,
               layer->kv_a_mqa, 0, &runtime->normed_batch);

    for (int t = 0; t < n_tokens; t++) {
        float *projected = runtime->kv_projected + (size_t)t * runtime->cache_width;
        float *normed = runtime->kv_normed + (size_t)t * rank;
        rmsnorm(normed, projected, layer->kv_a_norm->data, rank,
                model->rms_eps);
        rope_apply(projected + rank,
                   runtime->cos_sin + (size_t)t * model->qk_rope_dim,
                   model->qk_rope_dim);

        uint8_t *slot = cache_slot(runtime, layer_index, position + t);
        float *scale = slot_scales(runtime, layer_index, position + t);
        scale[SLOT_LATENT] = quantize_cache(slot, normed, rank);
        scale[SLOT_ROTARY] = quantize_cache(slot + rank, projected + rank,
                                            model->qk_rope_dim);
    }

    AttentionJob job = {runtime,  layer,    layer_index,
                        position, n_tokens, position + n_tokens};
    pool_run(runtime->pool, attention_mla_worker, &job);
}

/* GQA scores each query head against its own key head and folds the same
   head's values, so the cache holds every key and value in full rather than a
   shared latent. The slot is n_head_kv keys then n_head_kv values, each
   quantized with its own scale. */
static void attention_gqa_worker(void *state, int worker, int n_workers) {
    const AttentionJob *job = state;
    Runtime *runtime = job->runtime;
    const Model *model = runtime->model;
    int head_dim_k = model->head_dim_k;
    int head_dim_v = model->head_dim_v;
    int n_head_kv = model->n_head_kv;
    int n_rot = model->qk_rope_dim;
    int n_tokens = job->n_tokens;
    int query_width = model->n_head * head_dim_k;
    int out_width = model->n_head * head_dim_v;
    int kv_width = n_head_kv * head_dim_k;
    int n_rows = cache_rows(job->n_cached);
    int width = runtime->cache_width;
    float *block = runtime->cache_block + (size_t)worker * CACHE_ROWS * width;

    int head_begin = model->n_head * worker / n_workers;
    int head_end = model->n_head * (worker + 1) / n_workers;

    for (int head = head_begin; head < head_end; head++) {
        float *query = runtime->query + (size_t)head * head_dim_k;
        for (int t = 0; t < n_tokens; t++)
            rope_apply_neox(query + (size_t)t * query_width,
                            runtime->cos_sin + (size_t)t * n_rot, n_rot);
        for (int t = 0; t < n_tokens; t++) {
            int8_t *quants = query_quants(runtime, t, head);
            QuantizedQuery *quantized = query_scales(runtime, t, head);
            quantize_query(quantized, quants, query + (size_t)t * query_width,
                           head_dim_k);
        }
    }

    for (int p = 0; p < n_rows; p += CACHE_ROWS) {
        const uint8_t *slot = cache_slot(runtime, job->layer_index, p);
        const float *scales = slot_scales(runtime, job->layer_index, p);
        for (int head = head_begin; head < head_end; head++) {
            int kv = head * n_head_kv / model->n_head;
            const uint8_t *k = slot + (size_t)kv * head_dim_k;
            for (int t = 0; t < n_tokens; t++) {
                const int8_t *quants = query_quants(runtime, t, head);
                const QuantizedQuery *query = query_scales(runtime, t, head);
                int32_t dots[CACHE_ROWS];
                dot_int8_rows(dots, k, width, quants, head_dim_k);
                float *row = score_row(runtime, head, t, n_tokens) + p;
                for (int r = 0; r < CACHE_ROWS; r++)
                    row[r] = scaled_dot(dots[r], query,
                                        scales[(size_t)r * runtime->n_segments + kv]) *
                             model->kq_scale;
            }
        }
    }

    for (int head = head_begin; head < head_end; head++)
        for (int t = 0; t < n_tokens; t++) {
            float *row = score_row(runtime, head, t, n_tokens);
            int attended = job->position + t + 1;
            softmax(row, attended);
            memset(row + attended, 0,
                   (size_t)(n_rows - attended) * sizeof *row);
            memset(runtime->attn_out + (size_t)t * out_width +
                       (size_t)head * head_dim_v,
                   0, (size_t)head_dim_v * sizeof *runtime->attn_out);
        }

    int kv_begin = head_begin * n_head_kv / model->n_head;
    int kv_end = (head_end - 1) * n_head_kv / model->n_head + 1;
    for (int kv = kv_begin; kv < kv_end; kv++) {
        int head_lo = kv * model->n_head / n_head_kv;
        int head_hi = (kv + 1) * model->n_head / n_head_kv;
        if (head_lo < head_begin) head_lo = head_begin;
        if (head_hi > head_end) head_hi = head_end;
        for (int p = 0; p < n_rows; p += CACHE_ROWS) {
            for (int r = 0; r < CACHE_ROWS; r++)
                expand_int8(block + (size_t)r * width,
                            cache_slot(runtime, job->layer_index, p + r) +
                                kv_width + (size_t)kv * head_dim_v,
                            slot_scales(runtime, job->layer_index, p + r)
                                [n_head_kv + kv],
                            head_dim_v);
            for (int head = head_lo; head < head_hi; head++)
                for (int t = 0; t < n_tokens; t++)
                    add_scaled_rows(runtime->attn_out + (size_t)t * out_width +
                                        (size_t)head * head_dim_v,
                                    block, width,
                                    score_row(runtime, head, t, n_tokens) + p,
                                    head_dim_v);
        }
    }
}

static void attention_gqa(Runtime *runtime, const Layer *layer, int layer_index,
                          int position, int n_tokens) {
    const Model *model = runtime->model;
    int head_dim_k = model->head_dim_k;
    int head_dim_v = model->head_dim_v;
    int n_head_kv = model->n_head_kv;
    int kv_width = n_head_kv * head_dim_k;

    for (int t = 0; t < n_tokens; t++)
        for (int head = 0; head < model->n_head; head++)
            rmsnorm(runtime->query + (size_t)t * model->n_head * head_dim_k +
                        (size_t)head * head_dim_k,
                    runtime->query + (size_t)t * model->n_head * head_dim_k +
                        (size_t)head * head_dim_k,
                    layer->attn_q_norm->data, head_dim_k, model->rms_eps);

    run_matmul(runtime, runtime->kv_projected, (size_t)runtime->cache_width,
               layer->attn_k, 0, &runtime->normed_batch);
    run_matmul(runtime, runtime->kv_projected + kv_width,
               (size_t)runtime->cache_width, layer->attn_v, 0,
               &runtime->normed_batch);

    for (int t = 0; t < n_tokens; t++) {
        float *k = runtime->kv_projected + (size_t)t * runtime->cache_width;
        for (int kv = 0; kv < n_head_kv; kv++) {
            float *head_k = k + (size_t)kv * head_dim_k;
            rmsnorm(head_k, head_k, layer->attn_k_norm->data, head_dim_k,
                    model->rms_eps);
            rope_apply_neox(head_k,
                            runtime->cos_sin + (size_t)t * model->qk_rope_dim,
                            model->qk_rope_dim);
        }
        uint8_t *slot = cache_slot(runtime, layer_index, position + t);
        float *scale = slot_scales(runtime, layer_index, position + t);
        for (int kv = 0; kv < n_head_kv; kv++) {
            scale[kv] = quantize_cache(slot + (size_t)kv * head_dim_k,
                                       k + (size_t)kv * head_dim_k, head_dim_k);
            scale[n_head_kv + kv] =
                quantize_cache(slot + kv_width + (size_t)kv * head_dim_v,
                               k + kv_width + (size_t)kv * head_dim_v,
                               head_dim_v);
        }
    }

    AttentionJob job = {runtime,  layer,    layer_index,
                        position, n_tokens, position + n_tokens};
    pool_run(runtime->pool, attention_gqa_worker, &job);
}

/* The cpu tier of the attention split, exported for tests that need to run
   one layer of it against a strategy's replacement. */
void forward_attention_cpu(Runtime *runtime, const Layer *layer,
                           int layer_index, int position, int n_tokens) {
    const Model *model = runtime->model;
    run_matmul(runtime, runtime->query,
               (size_t)model->n_head * model->head_dim_k, layer->attn_q, 0,
               &runtime->normed_batch);
    if (model->attention == ATTN_MLA)
        attention_mla(runtime, layer, layer_index, position, n_tokens);
    else
        attention_gqa(runtime, layer, layer_index, position, n_tokens);
    activation_set(&runtime->heads_batch, runtime->heads_scratch,
                   runtime->attn_out,
                   (size_t)model->n_head * model->head_dim_v,
                   model->n_head * model->head_dim_v, n_tokens);
    run_matmul(runtime, runtime->projected, (size_t)model->n_embd,
               layer->attn_output, 0, &runtime->heads_batch);
}

static int branch_width(const Branch *branch) {
    return (int)branch->ffn->gate->dims[1];
}

/* Work is rows times tokens, not rows: the shared expert takes every token while
   a routed one often takes one, so splitting rows evenly would overload whoever
   landed on the shared branch. The product is also the branch's float count. */
static long long branch_layout(Branch *branches, int n_branches) {
    long long work = 0;
    size_t scratch = 0;
    for (int b = 0; b < n_branches; b++) {
        int width = branch_width(&branches[b]);
        branches[b].work_begin = work;
        branches[b].scratch_begin = scratch;
        work += (long long)width * branches[b].n_tokens;
        scratch += activation_bytes(width, branches[b].n_tokens);
    }
    return work;
}

/* The input batch covers the chunk in order, which is what makes a token list an
   index into it. */
static void branch_input(ActivationBatch *batch, const ActivationBatch *input,
                         const Runtime *runtime, const Branch *branch) {
    *batch = *input;
    batch->tokens = runtime->branch_tokens + branch->token_begin;
    batch->n_x = branch->n_tokens;
}

typedef struct {
    Runtime *runtime;
    int n_branches;
    long long total_work;
    const ActivationBatch *x;
} ExpandJob;

/* Slicing the concatenated work space rather than the branches keeps the last
   branch off one thread while the rest wait. */
static void expand_worker(void *state, int worker, int n_workers) {
    const ExpandJob *job = state;
    Runtime *runtime = job->runtime;
    int n_embd = runtime->model->n_embd;
    long long begin = job->total_work * worker / n_workers;
    long long end = job->total_work * (worker + 1) / n_workers;

    for (int b = 0; b < job->n_branches; b++) {
        const Branch *branch = &runtime->branches[b];
        int width = branch_width(branch);
        long long lo = (begin - branch->work_begin) / branch->n_tokens;
        long long hi = (end - branch->work_begin) / branch->n_tokens;
        if (lo < 0) lo = 0;
        if (hi > width) hi = width;
        if (lo >= hi) continue;

        ActivationBatch x;
        branch_input(&x, job->x, runtime, branch);

        float *gate = runtime->gate + branch->work_begin;
        float *up = runtime->up + branch->work_begin;
        float *activated = runtime->activated + branch->work_begin;
        matmul(gate, (size_t)width, matrix_at(branch->ffn->gate, branch->matrix_index),
               branch->ffn->gate->type, n_embd, (int)lo, (int)hi, &x);
        matmul(up, (size_t)width, matrix_at(branch->ffn->up, branch->matrix_index),
               branch->ffn->up->type, n_embd, (int)lo, (int)hi, &x);
        for (int t = 0; t < branch->n_tokens; t++)
            swiglu(activated + (size_t)t * width + lo,
                   gate + (size_t)t * width + lo, up + (size_t)t * width + lo,
                   (int)(hi - lo));
    }
}

typedef struct {
    Runtime *runtime;
    int n_branches;
    int n_tokens;
} ContractJob;

/* Workers own disjoint output columns, so two branches sharing a token never
   collide and the mixture needs no reduction. */
static void contract_worker(void *state, int worker, int n_workers) {
    const ContractJob *job = state;
    Runtime *runtime = job->runtime;
    int n_embd = runtime->model->n_embd;
    int begin = (int)((long long)n_embd * worker / n_workers);
    int end = (int)((long long)n_embd * (worker + 1) / n_workers);
    float *branch_out =
        runtime->branch_out + (size_t)worker * runtime->max_tokens * n_embd;

    for (int t = 0; t < job->n_tokens; t++)
        memset(runtime->projected + (size_t)t * n_embd + begin, 0,
               (size_t)(end - begin) * sizeof *runtime->projected);

    for (int b = 0; b < job->n_branches; b++) {
        const Branch *branch = &runtime->branches[b];
        matmul(branch_out, (size_t)n_embd,
               matrix_at(branch->ffn->down, branch->matrix_index),
               branch->ffn->down->type, branch_width(branch), begin, end,
               &runtime->branch_activations[b]);

        for (int t = 0; t < branch->n_tokens; t++) {
            int token = runtime->branch_tokens[branch->token_begin + t];
            add_scaled(runtime->projected + (size_t)token * n_embd + begin,
                       branch_out + (size_t)t * n_embd + begin,
                       runtime->branch_weights[branch->token_begin + t],
                       end - begin);
        }
    }
}

static void run_branches(Runtime *runtime, int n_branches, int n_tokens,
                         const ActivationBatch *input) {
    ExpandJob expand = {runtime, n_branches,
                        branch_layout(runtime->branches, n_branches), input};
    pool_run(runtime->pool, expand_worker, &expand);

    for (int b = 0; b < n_branches; b++) {
        const Branch *branch = &runtime->branches[b];
        int width = branch_width(branch);
        activation_set(&runtime->branch_activations[b],
                       runtime->activated_scratch + branch->scratch_begin,
                       runtime->activated + branch->work_begin, (size_t)width,
                       width, branch->n_tokens);
    }

    ContractJob contract = {runtime, n_branches, n_tokens};
    pool_run(runtime->pool, contract_worker, &contract);
}

/* Experts are ranked by probability plus a learned bias, but weighted by the
   probability alone -- the bias steers load balancing, not the mixture. */
static void rank_experts(Runtime *runtime, const Layer *layer, int token) {
    const Model *model = runtime->model;
    const float *bias = layer->router_bias ? layer->router_bias->data : NULL;
    int used = model->n_expert_used;
    float *probs = runtime->router_probs + (size_t)token * model->n_expert;
    int *chosen = runtime->chosen + (size_t)token * used;
    float *weights = runtime->chosen_weight + (size_t)token * used;

    if (model->gating == GATING_SOFTMAX) {
        float max = probs[0];
        for (int e = 1; e < model->n_expert; e++)
            if (probs[e] > max) max = probs[e];
        float sum = 0;
        for (int e = 0; e < model->n_expert; e++) {
            probs[e] = expf(probs[e] - max);
            sum += probs[e];
        }
        for (int e = 0; e < model->n_expert; e++) probs[e] /= sum;
    } else {
        for (int e = 0; e < model->n_expert; e++)
            probs[e] = 1.0f / (1.0f + expf(-probs[e]));
    }

    for (int slot = 0; slot < used; slot++) {
        int best = -1;
        float best_score = 0;
        for (int e = 0; e < model->n_expert; e++) {
            int taken = 0;
            for (int s = 0; s < slot; s++)
                if (chosen[s] == e) taken = 1;
            if (taken) continue;
            float score = probs[e] + (bias ? bias[e] : 0.0f);
            if (best < 0 || score > best_score) {
                best = e;
                best_score = score;
            }
        }
        chosen[slot] = best;
        weights[slot] = probs[best];
    }

    float scale = model->expert_weights_scale;
    if (model->expert_weights_norm) {
        float sum = 0;
        for (int slot = 0; slot < used; slot++) sum += weights[slot];
        if (sum > 0) scale /= sum;
    }
    for (int slot = 0; slot < used; slot++) weights[slot] *= scale;
}

static void add_branch_token(Runtime *runtime, int flat, int token,
                             float weight) {
    runtime->branch_tokens[flat] = token;
    runtime->branch_weights[flat] = weight;
}

static int select_branches(Runtime *runtime, const Layer *layer, int n_tokens) {
    const Model *model = runtime->model;
    int used = model->n_expert_used;
    int n_branches = 0;
    int flat = 0;

    for (int t = 0; t < n_tokens; t++) rank_experts(runtime, layer, t);

    for (int e = 0; e < model->n_expert; e++) {
        int begin = flat;
        for (int t = 0; t < n_tokens; t++)
            for (int slot = 0; slot < used; slot++)
                if (runtime->chosen[(size_t)t * used + slot] == e)
                    add_branch_token(
                        runtime, flat++, t,
                        runtime->chosen_weight[(size_t)t * used + slot]);
        if (flat == begin) continue;

        Branch *branch = &runtime->branches[n_branches++];
        branch->ffn = &layer->experts;
        branch->matrix_index = e;
        branch->token_begin = begin;
        branch->n_tokens = flat - begin;
    }

    if (model->n_expert_shared > 0) {
        Branch *shared = &runtime->branches[n_branches++];
        shared->ffn = &layer->shared_expert;
        shared->matrix_index = 0;
        shared->token_begin = flat;
        shared->n_tokens = n_tokens;
        for (int t = 0; t < n_tokens; t++)
            add_branch_token(runtime, flat++, t, 1.0f);
    }
    return n_branches;
}

static void feed_forward(Runtime *runtime, const Layer *layer, int n_tokens) {
    const Model *model = runtime->model;

    if (!layer->has_experts) {
        Branch *branch = &runtime->branches[0];
        branch->ffn = &layer->dense;
        branch->matrix_index = 0;
        branch->token_begin = 0;
        branch->n_tokens = n_tokens;
        for (int t = 0; t < n_tokens; t++)
            add_branch_token(runtime, t, t, 1.0f);
        run_branches(runtime, 1, n_tokens, &runtime->normed_batch);
        return;
    }
    run_matmul(runtime, runtime->router_probs, (size_t)model->n_expert,
               layer->router, 0, &runtime->normed_batch);
    run_branches(runtime, select_branches(runtime, layer, n_tokens), n_tokens,
                 &runtime->normed_batch);
}
const float *forward_with(Runtime *runtime, const int *tokens, int position,
                          int n_tokens, AttentionFn attention) {
    const Model *model = runtime->model;
    int n_embd = model->n_embd;

    for (int t = 0; t < n_tokens; t++) {
        const unsigned char *embedding =
            (const unsigned char *)model->token_embd->data +
            (size_t)tokens[t] * row_bytes(model->token_embd->type, n_embd);
        dequant_row(embedding, model->token_embd->type, n_embd,
                    runtime->residual + (size_t)t * n_embd);
        rope_position(runtime->cos_sin + (size_t)t * model->qk_rope_dim,
                      &runtime->rope, position + t);
    }

    for (int index = 0; index < model->n_layer; index++) {
        const Layer *layer = &model->layers[index];

        normalize(runtime, layer->attn_norm, n_tokens);
        attention(runtime, layer, index, position, n_tokens);
        for (int t = 0; t < n_tokens; t++)
            add_scaled(runtime->residual + (size_t)t * n_embd,
                       runtime->projected + (size_t)t * n_embd, 1.0f, n_embd);

        normalize(runtime, layer->ffn_norm, n_tokens);
        feed_forward(runtime, layer, n_tokens);
        for (int t = 0; t < n_tokens; t++)
            add_scaled(runtime->residual + (size_t)t * n_embd,
                       runtime->projected + (size_t)t * n_embd, 1.0f, n_embd);
    }

    /* Only the last token is sampled, and the output matrix is the widest in
       the model -- projecting the whole chunk would cost more than the layers
       did. */
    rmsnorm(runtime->normed,
            runtime->residual + (size_t)(n_tokens - 1) * n_embd,
            model->output_norm->data, n_embd, model->rms_eps);
    activation_set(&runtime->normed_batch, runtime->normed_scratch,
                   runtime->normed, (size_t)n_embd, n_embd, 1);
    run_matmul(runtime, runtime->logits, (size_t)model->n_vocab, model->output,
               0, &runtime->normed_batch);
    return runtime->logits;
}

const float *forward(Runtime *runtime, const int *tokens, int position,
                      int n_tokens) {
    return forward_with(runtime, tokens, position, n_tokens,
                      forward_attention_cpu);
}

static float *alloc_floats(size_t n, int *ok) {
    float *buffer = calloc(n, sizeof *buffer);
    if (!buffer) *ok = 0;
    return buffer;
}

static unsigned char *alloc_bytes(size_t n, int *ok) {
    unsigned char *buffer = calloc(n, 1);
    if (!buffer) *ok = 0;
    return buffer;
}

static size_t larger(size_t a, size_t b) { return a > b ? a : b; }

Runtime *runtime_start(const Model *model, int n_ctx, int n_threads, char *err,
                       size_t errsz) {
    Runtime *runtime = calloc(1, sizeof *runtime);
    if (!runtime) {
        snprintf(err, errsz, "out of memory");
        return NULL;
    }
    runtime->model = model;
    /* Every position-indexed buffer holds a whole number of blocks, so a cache
       pass can run the block a chunk ends inside of to its end. */
    runtime->n_ctx = n_ctx = cache_rows(n_ctx);
    if (model->attention == ATTN_MLA) {
        runtime->cache_width = model->kv_lora_rank + model->qk_rope_dim;
        runtime->n_segments = SLOT_SEGMENTS;
        runtime->query_quant_width = runtime->cache_width;
        runtime->query_segments = SLOT_SEGMENTS;
    } else {
        runtime->cache_width =
            model->n_head_kv * (model->head_dim_k + model->head_dim_v);
        runtime->n_segments = 2 * model->n_head_kv;
        runtime->query_quant_width = model->head_dim_k;
        runtime->query_segments = 1;
    }
    runtime->max_tokens = n_ctx < PREFILL_CHUNK ? n_ctx : PREFILL_CHUNK;
    rope_init(&runtime->rope, model->rope_freq_base, model->rope_freq_scale,
              model->qk_rope_dim, model->rope_orig_ctx, model->rope_beta_fast,
              model->rope_beta_slow);

    if (n_threads < 1) n_threads = pool_default_workers();
    runtime->pool = pool_start(n_threads);
    if (!runtime->pool) {
        snprintf(err, errsz, "could not start %d worker threads", n_threads);
        runtime_stop(runtime);
        return NULL;
    }

    int max_tokens = runtime->max_tokens;
    int n_workers = pool_workers(runtime->pool);

    /* A chunk can touch every expert at once, not just one token's picks. */
    int n_branches = model->n_expert + model->n_expert_shared;
    if (n_branches < 1) n_branches = 1;
    int per_token = model->n_expert_used + model->n_expert_shared;
    if (per_token < 1) per_token = 1;
    int flat_tokens = max_tokens * per_token;

    /* Sized for whichever block concatenates to the most rows: a chunk through
       the widest dense block, or through every expert it can route to. */
    int inner = model->n_ff;
    if (per_token * model->n_ff_expert > inner)
        inner = per_token * model->n_ff_expert;
    size_t ff_capacity = (size_t)max_tokens * inner;

    int ok = 1;
    runtime->normed_scratch =
        alloc_bytes(activation_bytes(model->n_embd, max_tokens), &ok);
    /* Every branch quantizes its own rows: one block set per (token, expert)
       however the tokens split, plus one per token for the shared branch. */
    runtime->activated_scratch = alloc_bytes(
        larger(activation_bytes(model->n_ff, max_tokens),
               activation_bytes(model->n_ff_expert,
                                max_tokens * model->n_expert_used) +
                   activation_bytes(model->n_ff_expert *
                                        model->n_expert_shared,
                                    max_tokens)),
        &ok);
    runtime->heads_scratch = alloc_bytes(
        activation_bytes(model->n_head * model->head_dim_v, max_tokens), &ok);
    if (model->attention == ATTN_MLA) {
        runtime->head_scratch_bytes =
            larger(activation_bytes(model->qk_nope_dim, max_tokens),
                   activation_bytes(model->kv_lora_rank, max_tokens));
        runtime->head_scratch =
            alloc_bytes((size_t)n_workers * runtime->head_scratch_bytes, &ok);
    }

    runtime->branches = calloc((size_t)n_branches, sizeof *runtime->branches);
    runtime->branch_activations =
        calloc((size_t)n_branches, sizeof *runtime->branch_activations);
    runtime->branch_tokens =
        calloc((size_t)flat_tokens, sizeof *runtime->branch_tokens);
    runtime->branch_weights =
        calloc((size_t)flat_tokens, sizeof *runtime->branch_weights);
    if (!runtime->branches || !runtime->branch_activations ||
        !runtime->branch_tokens || !runtime->branch_weights)
        ok = 0;
    if (model->n_expert > 0) {
        runtime->chosen = calloc((size_t)max_tokens * model->n_expert_used,
                                 sizeof *runtime->chosen);
        runtime->chosen_weight =
            calloc((size_t)max_tokens * model->n_expert_used,
                   sizeof *runtime->chosen_weight);
        runtime->router_probs =
            alloc_floats((size_t)max_tokens * model->n_expert, &ok);
        if (!runtime->chosen || !runtime->chosen_weight) ok = 0;
    }

    size_t slots = (size_t)model->n_layer * n_ctx;
    size_t cache_bytes = slots * (runtime->cache_width +
                                  runtime->n_segments * sizeof(float));
    runtime->cache = alloc_bytes(slots * runtime->cache_width, &ok);
    runtime->cache_scale = alloc_floats(slots * runtime->n_segments, &ok);

    runtime->query_quants =
        calloc((size_t)max_tokens * model->n_head * runtime->query_quant_width,
               sizeof *runtime->query_quants);
    runtime->query_scale =
        calloc((size_t)max_tokens * model->n_head * runtime->query_segments,
               sizeof *runtime->query_scale);
    if (!runtime->query_quants || !runtime->query_scale) ok = 0;

    runtime->cache_block = alloc_floats(
        (size_t)n_workers * CACHE_ROWS * runtime->cache_width, &ok);
    runtime->cos_sin =
        alloc_floats((size_t)max_tokens * model->qk_rope_dim, &ok);
    runtime->residual = alloc_floats((size_t)max_tokens * model->n_embd, &ok);
    runtime->normed = alloc_floats((size_t)max_tokens * model->n_embd, &ok);
    runtime->query = alloc_floats(
        (size_t)max_tokens * model->n_head * model->head_dim_k, &ok);
    runtime->kv_projected =
        alloc_floats((size_t)max_tokens * runtime->cache_width, &ok);
    runtime->scores =
        alloc_floats((size_t)model->n_head * max_tokens * n_ctx, &ok);
    runtime->attn_out = alloc_floats(
        (size_t)max_tokens * model->n_head * model->head_dim_v, &ok);
    if (model->attention == ATTN_MLA) {
        runtime->query_latent = alloc_floats(
            (size_t)max_tokens * model->n_head * model->kv_lora_rank, &ok);
        runtime->kv_normed =
            alloc_floats((size_t)max_tokens * model->kv_lora_rank, &ok);
        runtime->attn_latent = alloc_floats(
            (size_t)max_tokens * model->n_head * model->kv_lora_rank, &ok);
    }
    runtime->projected = alloc_floats((size_t)max_tokens * model->n_embd, &ok);
    runtime->gate = alloc_floats(ff_capacity, &ok);
    runtime->up = alloc_floats(ff_capacity, &ok);
    runtime->activated = alloc_floats(ff_capacity, &ok);
    runtime->branch_out = alloc_floats(
        (size_t)n_workers * max_tokens * model->n_embd, &ok);
    runtime->logits = alloc_floats((size_t)model->n_vocab, &ok);

    if (!ok) {
        snprintf(err, errsz,
                 "out of memory for %d-token context (kv cache needs %.2f GB)",
                 n_ctx, cache_bytes / 1e9);
        runtime_stop(runtime);
        return NULL;
    }
    return runtime;
}

void runtime_stop(Runtime *runtime) {
    if (!runtime) return;
    pool_stop(runtime->pool);
    free(runtime->cache);
    free(runtime->cache_scale);
    free(runtime->cache_block);
    free(runtime->cos_sin);
    free(runtime->query_quants);
    free(runtime->query_scale);
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
    free(runtime->branch_activations);
    free(runtime->branch_tokens);
    free(runtime->branch_weights);
    free(runtime->chosen);
    free(runtime->chosen_weight);
    free(runtime->normed_scratch);
    free(runtime->activated_scratch);
    free(runtime->heads_scratch);
    free(runtime->head_scratch);
    free(runtime);
}
