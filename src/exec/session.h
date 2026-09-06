#ifndef GEODE_SESSION_H
#define GEODE_SESSION_H

#include "chat.h"
#include "model.h"
#include "plan.h"
#include "sampler.h"
#include "strategy.h"
#include "tokenizer.h"

#define SAMPLER_TEMPERATURE 0.7f
#define SAMPLER_TOP_P 0.9f
#define SAMPLER_REPETITION_PENALTY 1.1f

/* Seeded rather than clocked, so a run still reproduces exactly. */
#define SAMPLER_SEED 0x9E3779B97F4A7C15ull

typedef struct {
    Plan plan;
    const Strategy *strategy;
    Model model;
    Tokenizer tokenizer;
    Chat chat;
    Sampler sampler;
    int has_chat;
} Session;

/* Loads the cached plan for `model`, picks the strategy, and opens the
   model, tokenizer and chat. Prints what it chose. */
int session_open(Session *session, const GgufFile *g, const char *model,
                 char *err, size_t errsz);
void session_close(Session *session);

typedef struct {
    int n_prompt;
    int n_generated;
    double prefill_seconds;
    double decode_seconds;
} StreamStats;

/* One generated token's decoded text. Return nonzero to stop generation. */
typedef int (*TokenSink)(void *ctx, const char *text, int length);

/* Prefills the prompt at `position`, samples until the model emits its end
   token or the budget runs out, and returns the position past the last token
   cached -- where the next prompt has to start, since the cache holds no
   gaps. `generated_ids`, when not NULL, receives the id of every token fed
   back to the model, so a caller keeping its own history can extend it by
   exactly what the cache came to hold. */
int session_stream(Session *session, Runtime *runtime, const int *ids,
                   int n_prompt, int position, int n_predict, TokenSink sink,
                   void *sink_ctx, int *generated_ids, StreamStats *stats);

#endif
