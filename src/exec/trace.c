#include "trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NANOSECONDS_PER_MS 1000000.0

/* Names and shape of the table, in the enum's order. `nested` marks a stage
   that is part of the one above it -- indented when printed, and left out of
   the accounted total so the top-level stages still sum to the pass.
   `per_layer` marks one the layer loop runs once per layer, which is the
   only place a per-layer column means anything. */
typedef struct {
    const char *name;
    int nested;
    int per_layer;
} StageInfo;

static const StageInfo stage_info[TRACE_STAGES] = {
    [TRACE_EMBED] = {"embed", 0, 0},
    [TRACE_NORM] = {"norm", 0, 1},
    [TRACE_ATTENTION] = {"attention", 0, 1},
    [TRACE_ATTN_UPLOAD] = {"upload", 1, 1},
    [TRACE_ATTN_ISSUE] = {"issue", 1, 1},
    [TRACE_ATTN_DRAIN] = {"drain", 1, 1},
    [TRACE_RESIDUAL] = {"residual", 0, 1},
    [TRACE_FFN] = {"ffn", 0, 1},
    [TRACE_ROUTER] = {"router", 1, 1},
    [TRACE_EXPERTS] = {"experts", 1, 1},
    [TRACE_FFN_ISSUE] = {"gpu issue", 1, 1},
    [TRACE_FFN_DRAIN] = {"gpu drain", 1, 1},
    [TRACE_LOGITS] = {"logits", 0, 0},
};

static uint64_t now_nanoseconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void trace_open(Trace *trace, int n_layer) {
    const char *wanted = getenv("GEODE_TRACE");
    trace->on = wanted && wanted[0] && wanted[0] != '0';
    trace->n_layer = n_layer;
    trace_reset(trace);
}

void trace_reset(Trace *trace) {
    trace->passes = 0;
    trace->tokens = 0;
    trace->wall = 0;
    for (int stage = 0; stage < TRACE_STAGES; stage++)
        trace->nanoseconds[stage] = 0;
}

uint64_t trace_now(const Trace *trace) {
    return trace->on ? now_nanoseconds() : 0;
}

uint64_t trace_mark(Trace *trace, TraceStage stage, uint64_t since) {
    if (!trace->on) return 0;
    uint64_t now = now_nanoseconds();
    trace->nanoseconds[stage] += now - since;
    return now;
}

void trace_pass(Trace *trace, uint64_t since, int n_tokens) {
    if (!trace->on) return;
    trace->wall += now_nanoseconds() - since;
    trace->passes++;
    trace->tokens += n_tokens;
}

static void print_row(const char *name, int nested, double ms, double share,
                      double per_layer) {
    char label[32];
    snprintf(label, sizeof label, "%s%s", nested ? "  " : "", name);
    printf("  %-16s %8.2f %7.1f", label, ms, share);
    if (per_layer > 0)
        printf(" %9.3f\n", per_layer);
    else
        printf(" %9s\n", "-");
}

void trace_report(const Trace *trace, const char *label) {
    if (!trace->on || trace->passes < 1) return;

    double passes = (double)trace->passes;
    double wall_ms = trace->wall / passes / NANOSECONDS_PER_MS;
    uint64_t accounted = 0;
    for (int stage = 0; stage < TRACE_STAGES; stage++)
        if (!stage_info[stage].nested) accounted += trace->nanoseconds[stage];

    printf("\ntrace (%s, %d layers, %lld passes over %lld tokens):\n", label,
           trace->n_layer, trace->passes, trace->tokens);
    printf("  %-16s %8s %7s %9s\n", "stage", "ms/pass", "%", "ms/layer");
    for (int stage = 0; stage < TRACE_STAGES; stage++) {
        const StageInfo *info = &stage_info[stage];
        double ms = trace->nanoseconds[stage] / passes / NANOSECONDS_PER_MS;
        print_row(info->name, info->nested, ms, 100.0 * ms / wall_ms,
                  info->per_layer ? ms / trace->n_layer : 0.0);
    }
    print_row("accounted", 0, accounted / passes / NANOSECONDS_PER_MS,
              100.0 * accounted / passes / NANOSECONDS_PER_MS / wall_ms, 0.0);
    print_row("wall", 0, wall_ms, 100.0, 0.0);
}
