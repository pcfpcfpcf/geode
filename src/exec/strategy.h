#ifndef GEODE_STRATEGY_H
#define GEODE_STRATEGY_H

#include "forward.h"
#include "model.h"
#include "plan.h"

/* One row per executor in SYSTEM_DESIGN's strategy table. The forward pass
   itself is shared -- a strategy differs by where it puts each component, not
   by the arithmetic -- so `forward` and `stop` are hooks for a tier split that
   has to interleave with the layer loop, not per-strategy transformers. */
typedef struct {
    const char *name;
    Runtime *(*start)(const Model *model, const Plan *plan, int n_ctx,
                      int n_threads, char *err, size_t errsz);
    const float *(*forward)(Runtime *runtime, const int *tokens, int position,
                            int n_tokens);
    void (*stop)(Runtime *runtime);
} Strategy;

/* The strategy to run for a plan, or NULL when the plan rules out every one
   this build has -- the planner refuses CPU-STREAM on a box the model does not
   fit in, and running it anyway would thrash rather than be slow.

   `predicted` takes the decode band the planner promised whichever strategy
   comes back, zeroed when it scored none. `note` takes the line to print: the
   plan is only a request, and which executor actually ran is the first thing a
   reader needs. */
const Strategy *strategy_choose(const Plan *plan, double *predicted, char *note,
                                size_t notesz);

#endif
