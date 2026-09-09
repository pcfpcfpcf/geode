#include "strategy.h"

#include <stdio.h>
#include <string.h>

extern const Strategy cpu_stream;
extern const Strategy hybrid;

/* Adding an executor is a file and a row here; nothing else in exec learns its
   name. RESIDENT and FLASH-STREAM are candidates the planner scores but this
   build does not implement, so a plan naming one falls through to the fastest
   that is implemented. */
static const Strategy *const strategies[] = {
    &cpu_stream,
    &hybrid,
};

/* What runs when the plan says nothing about any strategy this build has. */
#define DEFAULT_STRATEGY (&cpu_stream)

const Strategy *strategy_find(const char *name) {
    for (unsigned i = 0; i < sizeof strategies / sizeof *strategies; i++)
        if (strcmp(strategies[i]->name, name) == 0) return strategies[i];
    return NULL;
}

void strategy_names(char *out, size_t cap) {
    size_t used = 0;
    for (unsigned i = 0; i < sizeof strategies / sizeof *strategies; i++) {
        used += (size_t)snprintf(out + used, cap - used, "%s%s",
                                 i ? ", " : "", strategies[i]->name);
    }
}

static void set_band(double *predicted, const double *from) {
    predicted[0] = from[0];
    predicted[1] = from[1];
}

/* The fastest built strategy the planner was willing to score. */
static const Strategy *best_scored(const Plan *plan, double *predicted) {
    const Strategy *best = NULL;
    predicted[0] = predicted[1] = 0;

    for (int i = 0; i < plan->n_candidates; i++) {
        const Candidate *candidate = &plan->candidates[i];
        if (!candidate->scorable) continue;
        const Strategy *strategy = strategy_find(candidate->strategy);
        if (!strategy || candidate->decode_tok_s[1] <= predicted[1]) continue;
        best = strategy;
        set_band(predicted, candidate->decode_tok_s);
    }
    return best;
}

/* What a caller named by hand runs, plan or no plan: the planner's objection
   is reported in the note, not obeyed -- the caller asked for it. */
static const Strategy *strategy_overridden(const char *override,
                                           const Plan *plan, double *predicted,
                                           char *note, size_t notesz) {
    const Strategy *wanted = strategy_find(override);
    if (!wanted) {
        char built[128];
        strategy_names(built, sizeof built);
        snprintf(note, notesz, "unknown strategy '%s'; built here: %s",
                 override, built);
        return NULL;
    }

    const Candidate *candidate = plan_candidate(plan, wanted->name);
    if (candidate && candidate->scorable)
        set_band(predicted, candidate->decode_tok_s);
    if (candidate && !candidate->scorable)
        snprintf(note, notesz, "strategy: %s (the planner ruled it out: %s)",
                 wanted->name,
                 candidate->reason[0] ? candidate->reason : "no reason given");
    else
        snprintf(note, notesz, "strategy: %s (chosen over the plan)",
                 wanted->name);
    return wanted;
}

const Strategy *strategy_choose(const Plan *plan, const char *override,
                                double *predicted, char *note, size_t notesz) {
    predicted[0] = predicted[1] = 0;

    if (override && override[0])
        return strategy_overridden(override, plan, predicted, note, notesz);

    const Strategy *planned = strategy_find(plan->strategy);
    if (planned) {
        set_band(predicted, plan->predicted_tok_s);
        snprintf(note, notesz, "strategy: %s", planned->name);
        return planned;
    }

    const Strategy *fallback = best_scored(plan, predicted);
    if (fallback) {
        snprintf(note, notesz, "strategy: %s (plan says %s, not built here)",
                 fallback->name, plan->strategy);
        return fallback;
    }

    /* The planner ruled the default out rather than never considering it, so
       its reason is the answer -- running it anyway would thrash. */
    const Candidate *ruled_out = plan_candidate(plan, DEFAULT_STRATEGY->name);
    if (ruled_out && !ruled_out->scorable) {
        snprintf(note, notesz, "%s is not built here, and %s was ruled out: %s",
                 plan->strategy, DEFAULT_STRATEGY->name,
                 ruled_out->reason[0] ? ruled_out->reason : "no reason given");
        return NULL;
    }

    snprintf(note, notesz, "strategy: %s (no plan scored one that is built)",
             DEFAULT_STRATEGY->name);
    return DEFAULT_STRATEGY;
}
