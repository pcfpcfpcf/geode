#include "sampler.h"

#include <math.h>

/* Ranking the whole vocabulary would cost more than the forward pass that
   produced it, and past a few dozen tokens the tail carries none of the mass. */
#define TOP_K 64

/* Exactly representable in a float, so the draw is uniform over [0, 1). */
#define RANDOM_BITS 24

typedef struct {
    float logit;
    int token;
} Candidate;

static int collect_top_k(const float *logits, int n_vocab, Candidate *top) {
    int n = 0;
    for (int token = 0; token < n_vocab; token++) {
        if (n == TOP_K && logits[token] <= top[n - 1].logit) continue;
        int i = n < TOP_K ? n++ : n - 1;
        for (; i > 0 && logits[token] > top[i - 1].logit; i--)
            top[i] = top[i - 1];
        top[i].logit = logits[token];
        top[i].token = token;
    }
    return n;
}

static void sort_descending(Candidate *top, int n) {
    for (int i = 1; i < n; i++) {
        Candidate moving = top[i];
        int j = i;
        for (; j > 0 && moving.logit > top[j - 1].logit; j--) top[j] = top[j - 1];
        top[j] = moving;
    }
}

/* A token the model already dislikes has a negative logit, which dividing
   would pull back toward zero. */
static float penalized(const Sampler *sampler, int token, float logit) {
    long long n = sampler->n_seen < SAMPLER_HISTORY ? sampler->n_seen
                                                    : SAMPLER_HISTORY;
    for (long long i = 0; i < n; i++) {
        if (sampler->recent[i] != token) continue;
        return logit > 0 ? logit / sampler->repetition_penalty
                         : logit * sampler->repetition_penalty;
    }
    return logit;
}

static uint64_t next_random(Sampler *sampler) {
    sampler->state ^= sampler->state << 13;
    sampler->state ^= sampler->state >> 7;
    sampler->state ^= sampler->state << 17;
    return sampler->state;
}

static float next_uniform(Sampler *sampler) {
    return (float)(next_random(sampler) >> (64 - RANDOM_BITS)) /
           (float)(1u << RANDOM_BITS);
}

void sampler_init(Sampler *sampler, float temperature, float top_p,
                  float repetition_penalty, uint64_t seed) {
    sampler->temperature = temperature;
    sampler->top_p = top_p;
    sampler->repetition_penalty = repetition_penalty;
    sampler->state = seed ? seed : 1;
    sampler->n_seen = 0;
}

void sampler_note(Sampler *sampler, int token) {
    sampler->recent[sampler->n_seen++ % SAMPLER_HISTORY] = token;
}

int sampler_pick(Sampler *sampler, const float *logits, int n_vocab) {
    Candidate top[TOP_K];
    float weight[TOP_K];
    int n = collect_top_k(logits, n_vocab, top);
    if (n < 1) return 0;

    /* Penalizing the candidates rather than the vocabulary: a token the penalty
       would have lifted into them sits far below a nucleus a handful wide. */
    for (int i = 0; i < n; i++)
        top[i].logit = penalized(sampler, top[i].token, top[i].logit);
    sort_descending(top, n);
    if (sampler->temperature <= 0) return top[0].token;

    float total = 0;
    for (int i = 0; i < n; i++) {
        weight[i] = expf((top[i].logit - top[0].logit) / sampler->temperature);
        total += weight[i];
    }

    /* Never empty -- a peak above top_p on its own would otherwise leave
       nothing to draw from. */
    float kept = 0, wanted = total * sampler->top_p;
    int nucleus = 0;
    while (nucleus < n && kept < wanted) kept += weight[nucleus++];

    /* The last candidate answers whatever rounding leaves undrawn. */
    float draw = next_uniform(sampler) * kept;
    for (int i = 0; i < nucleus - 1; i++) {
        draw -= weight[i];
        if (draw < 0) return top[i].token;
    }
    return top[nucleus - 1].token;
}
