#ifndef GEODE_GPU_H
#define GEODE_GPU_H

typedef struct {
    char name[64];
    unsigned long long vram_bytes;
    int cc_major, cc_minor;
    int pcie_gen, pcie_width;
    double hbm_bw_bytes_s;
    double pcie_bw_bytes_s;
    int bw_measured;
    char bw_error[128];
} Gpu;

int gpu_probe(Gpu **out);
void gpu_free(Gpu *gpus);

#endif