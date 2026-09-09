#include "router_dump.h"

#include <stdlib.h>

void router_dump_open(RouterDump *dump, int n_layer, int n_expert,
                      int n_expert_used) {
    const char *path = getenv("GEODE_ROUTER_DUMP");
    dump->file = NULL;
    dump->armed = 0;
    dump->n_expert_used = n_expert_used;
    dump->rows = 0;
    if (!path || !path[0]) return;

    dump->file = fopen(path, "w");
    if (!dump->file) return;
    fprintf(dump->file, "# geode router dump  n_layer=%d n_expert=%d n_expert_used=%d\n",
            n_layer, n_expert, n_expert_used);
}

void router_dump_arm(RouterDump *dump) {
    if (dump->file) dump->armed = 1;
}

void router_dump_row(RouterDump *dump, int layer, const int *chosen) {
    if (!dump->file || !dump->armed) return;
    fprintf(dump->file, "%d", layer);
    for (int slot = 0; slot < dump->n_expert_used; slot++)
        fprintf(dump->file, " %d", chosen[slot]);
    fputc('\n', dump->file);
    dump->rows++;
}

void router_dump_close(RouterDump *dump) {
    if (!dump->file) return;
    fclose(dump->file);
    dump->file = NULL;
}
