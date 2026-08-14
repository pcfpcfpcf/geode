#ifndef GEODE_GGUF_H
#define GEODE_GGUF_H

#include <stddef.h>

#define GGUF_MAX_DIMS 4
#define GGUF_NAME_MAX 128

typedef struct {
    char name[GGUF_NAME_MAX];
    unsigned type;
    int n_dims;
    unsigned long long dims[GGUF_MAX_DIMS];
    const void *data;
    unsigned long long n_bytes;
} GgufTensor;

typedef struct {
    int fd;
    void *map;
    unsigned long long map_size;
    GgufTensor *tensors;
    unsigned long long n_tensors;
    const unsigned char *kv_start;
    const unsigned char *kv_end;
    unsigned long long kv_count;
} GgufFile;

int gguf_open(GgufFile *g, const char *path, char *err, size_t errsz);
void gguf_close(GgufFile *g);
const GgufTensor *gguf_find(const GgufFile *g, const char *name);
int gguf_meta_u64(const GgufFile *g, const char *key, unsigned long long *out);
int gguf_meta_f64(const GgufFile *g, const char *key, double *out);
int gguf_meta_str(const GgufFile *g, const char *key, char *out, size_t outsz);

#endif
