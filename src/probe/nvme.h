#ifndef GEODE_NVME_H
#define GEODE_NVME_H

typedef struct {
    char name[32];
    unsigned long long capacity_bytes;
    double read_bw_bytes_s;
    int measured;
} NvmeDrive;

int nvme_probe(NvmeDrive **out);
void nvme_free(NvmeDrive *drives);

#endif