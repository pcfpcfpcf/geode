#include "gguf.h"
#include "quant.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* GGUF is little-endian on disk; this reader assumes a little-endian host. */

#define GGUF_MAGIC 0x46554747u /* "GGUF" */
#define DEFAULT_ALIGNMENT 32

enum {
    KV_U8, KV_I8, KV_U16, KV_I16, KV_U32, KV_I32,
    KV_F32, KV_BOOL, KV_STR, KV_ARR, KV_U64, KV_I64, KV_F64
};

typedef struct {
    const unsigned char *p;
    const unsigned char *end;
} Cur;

static int cur_take(Cur *c, unsigned long long n, const unsigned char **out) {
    if ((unsigned long long)(c->end - c->p) < n) return 0;
    *out = c->p;
    c->p += n;
    return 1;
}

static int cur_u32(Cur *c, unsigned *out) {
    const unsigned char *b;
    if (!cur_take(c, 4, &b)) return 0;
    memcpy(out, b, 4);
    return 1;
}

static int cur_u64(Cur *c, unsigned long long *out) {
    const unsigned char *b;
    if (!cur_take(c, 8, &b)) return 0;
    memcpy(out, b, 8);
    return 1;
}

static int cur_str(Cur *c, const char **str, unsigned long long *len) {
    const unsigned char *b;
    if (!cur_u64(c, len)) return 0;
    if (!cur_take(c, *len, &b)) return 0;
    *str = (const char *)b;
    return 1;
}

static int skip_value(Cur *c, unsigned type) {
    const unsigned char *b;
    unsigned long long len, count;
    unsigned elem;
    switch (type) {
    case KV_U8: case KV_I8: case KV_BOOL:
        return cur_take(c, 1, &b);
    case KV_U16: case KV_I16:
        return cur_take(c, 2, &b);
    case KV_U32: case KV_I32: case KV_F32:
        return cur_take(c, 4, &b);
    case KV_U64: case KV_I64: case KV_F64:
        return cur_take(c, 8, &b);
    case KV_STR:
        if (!cur_u64(c, &len)) return 0;
        return cur_take(c, len, &b);
    case KV_ARR:
        if (!cur_u32(c, &elem) || !cur_u64(c, &count)) return 0;
        for (unsigned long long i = 0; i < count; i++)
            if (!skip_value(c, elem)) return 0;
        return 1;
    default:
        return 0;
    }
}

static int find_key(const GgufFile *g, const char *key, unsigned *type,
                    Cur *val) {
    Cur c = {g->kv_start, g->kv_end};
    size_t keylen = strlen(key);
    for (unsigned long long i = 0; i < g->kv_count; i++) {
        const char *k;
        unsigned long long klen;
        unsigned t;
        if (!cur_str(&c, &k, &klen) || !cur_u32(&c, &t)) return 0;
        if (klen == keylen && memcmp(k, key, keylen) == 0) {
            *type = t;
            *val = c;
            return 1;
        }
        if (!skip_value(&c, t)) return 0;
    }
    return 0;
}

int gguf_meta_u64(const GgufFile *g, const char *key,
                  unsigned long long *out) {
    unsigned type;
    Cur c;
    const unsigned char *b;
    if (!find_key(g, key, &type, &c)) return 0;
    switch (type) {
    case KV_U8: case KV_BOOL:
        if (!cur_take(&c, 1, &b)) return 0;
        *out = *b;
        return 1;
    case KV_U16:
        if (!cur_take(&c, 2, &b)) return 0;
        { unsigned short s; memcpy(&s, b, 2); *out = s; }
        return 1;
    case KV_U32: case KV_I32:
        if (!cur_take(&c, 4, &b)) return 0;
        { unsigned w; memcpy(&w, b, 4); *out = w; }
        return 1;
    case KV_U64: case KV_I64:
        return cur_u64(&c, out);
    default:
        return 0;
    }
}

int gguf_meta_f64(const GgufFile *g, const char *key, double *out) {
    unsigned type;
    Cur c;
    const unsigned char *b;
    if (!find_key(g, key, &type, &c)) return 0;
    switch (type) {
    case KV_F32:
        if (!cur_take(&c, 4, &b)) return 0;
        { float f; memcpy(&f, b, 4); *out = f; }
        return 1;
    case KV_F64:
        if (!cur_take(&c, 8, &b)) return 0;
        memcpy(out, b, 8);
        return 1;
    default:
        return 0;
    }
}

int gguf_meta_str(const GgufFile *g, const char *key, char *out,
                  size_t outsz) {
    unsigned type;
    Cur c;
    const char *s;
    unsigned long long len;
    if (!find_key(g, key, &type, &c) || type != KV_STR) return 0;
    if (!cur_str(&c, &s, &len)) return 0;
    if (len >= outsz) len = outsz - 1;
    memcpy(out, s, len);
    out[len] = '\0';
    return 1;
}

const GgufTensor *gguf_find(const GgufFile *g, const char *name) {
    for (unsigned long long i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}

static int read_tensor_infos(GgufFile *g, Cur *c, unsigned long long count,
                             char *err, size_t errsz) {
    g->tensors = malloc(count * sizeof *g->tensors);
    if (!g->tensors) {
        snprintf(err, errsz, "out of memory");
        return 0;
    }
    for (unsigned long long i = 0; i < count; i++) {
        GgufTensor *t = &g->tensors[i];
        const char *name;
        unsigned long long name_len, offset;
        unsigned n_dims, type;
        if (!cur_str(c, &name, &name_len) || !cur_u32(c, &n_dims)) return 0;
        if (name_len >= GGUF_NAME_MAX || n_dims < 1 || n_dims > GGUF_MAX_DIMS) {
            snprintf(err, errsz, "bad tensor info at index %llu", i);
            return 0;
        }
        memcpy(t->name, name, name_len);
        t->name[name_len] = '\0';
        t->n_dims = (int)n_dims;
        for (int d = 0; d < GGUF_MAX_DIMS; d++) t->dims[d] = 0;
        for (unsigned d = 0; d < n_dims; d++)
            if (!cur_u64(c, &t->dims[d])) return 0;
        if (!cur_u32(c, &type) || !cur_u64(c, &offset)) return 0;
        if (type >= (unsigned)n_quant_types || !quant_types[type].name) {
            snprintf(err, errsz, "%.100s: unknown tensor type %u", t->name,
                     type);
            return 0;
        }
        t->type = type;
        QuantType qt = quant_types[type];
        unsigned long long n_elements = 1;
        for (int d = 0; d < GGUF_MAX_DIMS; d++)
            n_elements *= t->dims[d] ? t->dims[d] : 1;
        t->n_bytes = (n_elements / qt.block_size) * qt.type_size;
        t->data = (const unsigned char *)g->map + offset; /* fixed by caller */
    }
    return 1;
}

int gguf_open(GgufFile *g, const char *path, char *err, size_t errsz) {
    memset(g, 0, sizeof *g);
    g->fd = open(path, O_RDONLY);
    if (g->fd < 0) {
        snprintf(err, errsz, "%s: %s", path, strerror(errno));
        return 0;
    }
    struct stat st;
    if (fstat(g->fd, &st) || st.st_size <= 0) {
        snprintf(err, errsz, "%s: cannot stat", path);
        goto fail;
    }
    g->map_size = (unsigned long long)st.st_size;
    g->map = mmap(NULL, g->map_size, PROT_READ, MAP_PRIVATE, g->fd, 0);
    if (g->map == MAP_FAILED) {
        snprintf(err, errsz, "%s: mmap: %s", path, strerror(errno));
        goto fail;
    }

    Cur c = {(const unsigned char *)g->map,
             (const unsigned char *)g->map + g->map_size};
    unsigned magic, version;
    unsigned long long tensor_count;
    if (!cur_u32(&c, &magic) || magic != GGUF_MAGIC) {
        snprintf(err, errsz, "%s: not a GGUF file", path);
        goto fail;
    }
    if (!cur_u32(&c, &version) || version < 2 || version > 3 ||
        !cur_u64(&c, &tensor_count) || !cur_u64(&c, &g->kv_count)) {
        snprintf(err, errsz, "%s: bad GGUF header", path);
        goto fail;
    }

    g->kv_start = c.p;
    for (unsigned long long i = 0; i < g->kv_count; i++) {
        const char *k;
        unsigned long long klen;
        unsigned type;
        if (!cur_str(&c, &k, &klen) || !cur_u32(&c, &type) ||
            !skip_value(&c, type)) {
            snprintf(err, errsz, "%s: metadata parse error", path);
            goto fail;
        }
    }
    g->kv_end = c.p;

    if (!read_tensor_infos(g, &c, tensor_count, err, errsz)) {
        if (!err[0]) snprintf(err, errsz, "%s: tensor info parse error", path);
        goto fail;
    }
    g->n_tensors = tensor_count;

    unsigned long long alignment = DEFAULT_ALIGNMENT;
    gguf_meta_u64(g, "general.alignment", &alignment);
    unsigned long long data_off =
        (unsigned long long)(c.p - (const unsigned char *)g->map);
    data_off = (data_off + alignment - 1) / alignment * alignment;
    const unsigned char *data_start =
        (const unsigned char *)g->map + data_off;
    const unsigned char *map_end =
        (const unsigned char *)g->map + g->map_size;
    for (unsigned long long i = 0; i < g->n_tensors; i++) {
        GgufTensor *t = &g->tensors[i];
        unsigned long long rel =
            (unsigned long long)((const unsigned char *)t->data -
                                 (const unsigned char *)g->map);
        if (data_start + rel + t->n_bytes > map_end) {
            snprintf(err, errsz, "%.100s: tensor data out of bounds",
                     t->name);
            goto fail;
        }
        t->data = data_start + rel;
    }
    return 1;

fail:
    gguf_close(g);
    return 0;
}

void gguf_close(GgufFile *g) {
    if (g->map && g->map != MAP_FAILED) munmap(g->map, g->map_size);
    if (g->fd > 0) close(g->fd);
    free(g->tensors);
    memset(g, 0, sizeof *g);
}
