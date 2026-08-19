#ifndef GEODE_SAMPLER_H
#define GEODE_SAMPLER_H

#include <stdint.h>

#define SAMPLER_HISTORY 64

/* Greedy decoding makes a reply a fixed point: once a phrase is its own most
   likely continuation nothing can leave it. Drawing from the head of the
   distribution breaks the token-level cycles and re-enters the phrase-level
   ones -- every token of a repeated phrase is individually likely -- so the
   penalty is not an alternative to the draw but the other half of it.
   Temperature 0 restores greedy. */
typedef struct {
    float temperature;
    float top_p;
    float repetition_penalty;
    uint64_t state;
    int recent[SAMPLER_HISTORY];
    long long n_seen;
} Sampler;

void sampler_init(Sampler *sampler, float temperature, float top_p,
                  float repetition_penalty, uint64_t seed);

/* Every token that enters the context, prompt and reply alike. */
void sampler_note(Sampler *sampler, int token);

int sampler_pick(Sampler *sampler, const float *logits, int n_vocab);

#endif
