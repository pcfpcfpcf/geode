#ifndef GEODE_TRACE_H
#define GEODE_TRACE_H

#include <stdint.h>

/* Where a forward pass spends its time, stage by stage, so a placement
   decision rests on a measurement rather than on a bandwidth model. Off
   unless GEODE_TRACE is set in the environment: a mark costs a clock read
   and a decode token takes hundreds of them.

   The stages are in print order. Six of them cover a pass end to end and
   sum to it; the rest break the two hooks down for whichever tier the
   strategy put them on. A nested stage is a share of the one above it, not
   time of its own, and a strategy without a gpu leaves its rows at zero --
   which is itself the reading, for a model whose layers have no base ffn to
   overlap the routed experts with. */
typedef enum {
    TRACE_EMBED,       /* token embedding and the rope table */
    TRACE_NORM,        /* both layer norms, and the output norm */
    TRACE_ATTENTION,   /* the attention hook, whichever tier owns it */
    TRACE_ATTN_UPLOAD, /*   its activations crossing up to the device */
    TRACE_ATTN_ISSUE,  /*   its kernel launches, which do not wait */
    TRACE_ATTN_DRAIN,  /*   the copy down that waits on all of them */
    TRACE_RESIDUAL,    /* both residual adds */
    TRACE_FFN,         /* the feed-forward hook, whichever tier owns it */
    TRACE_ROUTER,      /*   the router matmul and the top-k over it */
    TRACE_EXPERTS,     /*   the routed experts on the cpu */
    TRACE_FFN_ISSUE,   /*   the base ffn's launches, if a tier holds it */
    TRACE_FFN_DRAIN,   /*   the copy down that waits on them */
    TRACE_LOGITS,      /* the output projection */
    TRACE_STAGES
} TraceStage;

typedef struct {
    int on;
    int n_layer;
    long long passes;
    long long tokens;
    uint64_t wall;
    uint64_t nanoseconds[TRACE_STAGES];
} Trace;

/* Reads the environment once, for the runtime a strategy is starting. */
void trace_open(Trace *trace, int n_layer);

/* Drops what has been recorded, keeping the runtime traced. Prefill and
   decode have nothing in common but the code they run, so the caller that
   wants one of them clears the other out. */
void trace_reset(Trace *trace);

/* A timestamp to measure the next stage from, or 0 when tracing is off. */
uint64_t trace_now(const Trace *trace);

/* Charges now - `since` to `stage` and returns now, so a chain of stages
   reads the clock once per boundary rather than twice. */
uint64_t trace_mark(Trace *trace, TraceStage stage, uint64_t since);

/* Closes one pass: `n_tokens` through the layer loop, taking `since` to
   now. The wall it accumulates is what the stages are a share of, and the
   gap between them is the time no stage claimed. */
void trace_pass(Trace *trace, uint64_t since, int n_tokens);

/* The stage table, per pass. Silent when tracing is off or no pass ran. */
void trace_report(const Trace *trace, const char *label);

#endif
