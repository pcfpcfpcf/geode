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

static const Strategy *strategy_named(const char *name) {
    for (unsigned i = 0; i < sizeof strategies / sizeof *strategies; i++)
        if (strcmp(strategies[i]->name, name) == 0) return strategies[i];
    return NULL;
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
        const Strategy *strategy = strategy_named(candidate->strategy);
        if (!strategy || candidate->decode_tok_s[1] <= predicted[1]) continue;
        best = strategy;
        set_band(predicted, candidate->decode_tok_s);
    }
    return best;
}

const Strategy *strategy_choose(const Plan *plan, double *predicted, char *note,
                                size_t notesz) {
    predicted[0] = predicted[1] = 0;

    const Strategy *planned = strategy_named(plan->strategy);
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
