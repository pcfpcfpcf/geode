#ifndef GEODE_ROUTER_DUMP_H
#define GEODE_ROUTER_DUMP_H

/* The router's choices, one line per layer per decoded token, written to the
   path in GEODE_ROUTER_DUMP. Off when the variable is unset. It exists to
   answer one question the caching stages rest on: over a few hundred tokens,
   what share of a layer's expert reads would a resident pool of K experts
   have served -- i.e. has load-balanced routing flattened the skew enough to
   make a pool pointless. `scripts/router_hits.py` turns a dump into that
   curve.

   Like the trace, prefill is left out: `router_dump_arm` is called where
   `trace_reset` is, and rows before that call are dropped, so the file holds
   decode only. */

#include <stdio.h>

typedef struct {
    FILE *file;
    int armed;
    int n_expert_used;
    long long rows;
} RouterDump;

/* Reads GEODE_ROUTER_DUMP once and opens the file, writing a header the
   script reads the model's shape from. */
void router_dump_open(RouterDump *dump, int n_layer, int n_expert,
                      int n_expert_used);

/* Starts recording: rows offered before this call (prefill) are dropped. */
void router_dump_arm(RouterDump *dump);

/* One token's picks for one layer: `chosen` holds `n_expert_used` ids. */
void router_dump_row(RouterDump *dump, int layer, const int *chosen);

void router_dump_close(RouterDump *dump);

#endif
