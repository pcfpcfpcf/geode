#include "session.h"
#include "gguf.h"
#include "home.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int session_open(Session *session, const GgufFile *g, const char *model,
                 char *err, size_t errsz) {
    char plan_err[256];
    int planned = plan_load(&session->plan, geode_model_home(model, "plan"),
                            plan_err, sizeof plan_err);

    double predicted[2];
    char note[256];
    session->strategy =
        strategy_choose(&session->plan, predicted, note, sizeof note);
    if (!session->strategy) {
        snprintf(err, errsz, "%s", note);
        return 0;
    }

    if (planned) plan_print(&session->plan);
    else printf("plan:     none (%s)\n", plan_err);
    printf("%s\n", note);
    if (predicted[1] > 0)
        printf("predict:  %.1f-%.1f tok/s decode\n", predicted[0], predicted[1]);

    if (!model_load(&session->model, g, err, errsz)) return 0;
    if (!tokenizer_init(&session->tokenizer, g, err, errsz)) {
        model_free(&session->model);
        return 0;
    }
    session->has_chat = chat_init(&session->chat, &session->tokenizer);
    sampler_init(&session->sampler, SAMPLER_TEMPERATURE, SAMPLER_TOP_P,
                 SAMPLER_REPETITION_PENALTY, SAMPLER_SEED);
    return 1;
}

void session_close(Session *session) {
    tokenizer_free(&session->tokenizer);
    model_free(&session->model);
}

int session_stream(Session *session, Runtime *runtime, const int *ids,
                   int n_prompt, int position, int n_predict, TokenSink sink,
                   void *sink_ctx, int *generated_ids, StreamStats *stats) {
    const Strategy *strategy = session->strategy;
    double started = now_seconds();
    const float *logits = NULL;
    for (int i = 0; i < n_prompt; i++) sampler_note(&session->sampler, ids[i]);
    for (int i = 0; i < n_prompt; i += PREFILL_CHUNK) {
        int n = n_prompt - i;
        if (n > PREFILL_CHUNK) n = PREFILL_CHUNK;
        logits = strategy->forward(runtime, ids + i, position + i, n);
    }
    double prefilled = now_seconds();

    /* A reply closes with the message separator; a role separator in the
       middle of it means the model has started echoing the prompt template,
       and letting it continue only spirals. */
    int reply_end = session->has_chat ? session->chat.end : -1;
    int role_end = session->has_chat ? session->chat.start : -1;
    int generated = 0;
    for (int i = 0; i < n_predict; i++) {
        int token =
            sampler_pick(&session->sampler, logits, session->model.n_vocab);
        if (token == session->tokenizer.eos_id || token == reply_end ||
            token == role_end)
            break;
        sampler_note(&session->sampler, token);
        char text[512];
        tokenizer_decode(&session->tokenizer, &token, 1, text, sizeof text);
        if (sink && sink(sink_ctx, text, (int)strlen(text))) break;
        generated++;
        if (generated_ids) generated_ids[generated - 1] = token;
        logits = strategy->forward(runtime, &token, position + n_prompt + i, 1);
    }
    double finished = now_seconds();

    stats->n_prompt = n_prompt;
    stats->n_generated = generated;
    stats->prefill_seconds = prefilled - started;
    stats->decode_seconds = finished - prefilled;

    return position + n_prompt + generated;
}
