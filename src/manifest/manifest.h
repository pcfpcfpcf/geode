#ifndef GEODE_MANIFEST_H
#define GEODE_MANIFEST_H

#include <stddef.h>

typedef struct {
    char model[256];
    char source[1024];
    char variant[64];
    char architecture[128];
    unsigned long long total_params, active_params, n_layer, total_bytes;
    unsigned long long attention_bytes, base_bytes;
    unsigned long long expert_count, expert_used, expert_bytes, routed_bytes;
    unsigned long long kv_bytes_per_ctx_token;
    int mtp_head;
} Manifest;

int manifest_load(const char *path, Manifest *m, char *err, size_t errsz);

#endif
