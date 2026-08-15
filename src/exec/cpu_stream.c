#include "strategy.h"

/* Every component on DRAM, read straight out of the model's mapping. The
   plan's placement has nothing left to choose once there is one tier, so this
   strategy reads none of it. */
static Runtime *cpu_stream_start(const Model *model, const Plan *plan,
                                 int n_ctx, int n_threads, char *err,
                                 size_t errsz) {
    (void)plan;
    return runtime_start(model, n_ctx, n_threads, err, errsz);
}

const Strategy cpu_stream = {
    "CPU-STREAM",
    cpu_stream_start,
    forward,
    runtime_stop,
};
