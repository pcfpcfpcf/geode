#ifndef GEODE_GGUF_H
#define GEODE_GGUF_H

#include <stddef.h>

#define GGUF_MAX_DIMS 4
#define GGUF_NAME_MAX 128

typedef enum {
    GGUF_U8, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STR, GGUF_ARR, GGUF_U64, GGUF_I64, GGUF_F64
} GgufType;

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

/* Elements are variable width, so an array is walked rather than indexed --
   indexing a 128k-token vocabulary would cost a scan per token. */
typedef struct {
    GgufType elem_type;
    unsigned long long remaining;
    const unsigned char *next;
    const unsigned char *end;
} GgufArray;

int gguf_open(GgufFile *g, const char *path, char *err, size_t errsz);
void gguf_close(GgufFile *g);
const GgufTensor *gguf_find(const GgufFile *g, const char *name);
int gguf_meta_u64(const GgufFile *g, const char *key,
                  unsigned long long *out);
int gguf_meta_f64(const GgufFile *g, const char *key, double *out);
int gguf_meta_str(const GgufFile *g, const char *key, char *out,
                  size_t outsz);
int gguf_meta_arr(const GgufFile *g, const char *key, GgufArray *out);

/* `str` points into the mapping and is not terminated: it lives as long as the
   GgufFile and no longer. */
int gguf_array_next_str(GgufArray *array, const char **str, size_t *len);

#endif
