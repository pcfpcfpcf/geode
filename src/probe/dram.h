#ifndef GEODE_DRAM_H
#define GEODE_DRAM_H

typedef struct {
    int node;
    unsigned long long capacity_bytes;
    double read_bw_bytes_s;
    double copy_bw_bytes_s;
    double load_latency_ns;
} DramNode;

int dram_probe(DramNode **out);
void dram_free(DramNode *nodes);

#endif