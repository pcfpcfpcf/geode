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

typedef struct {
    const unsigned char *at;
    const unsigned char *end;
} Reader;

static int read_bytes(Reader *reader, unsigned long long n,
                      const unsigned char **out) {
    if ((unsigned long long)(reader->end - reader->at) < n) return 0;
    *out = reader->at;
    reader->at += n;
    return 1;
}

static int read_u32(Reader *reader, unsigned *out) {
    const unsigned char *bytes;
    if (!read_bytes(reader, 4, &bytes)) return 0;
    memcpy(out, bytes, 4);
    return 1;
}

static int read_u64(Reader *reader, unsigned long long *out) {
    const unsigned char *bytes;
    if (!read_bytes(reader, 8, &bytes)) return 0;
    memcpy(out, bytes, 8);
    return 1;
}

static int read_str(Reader *reader, const char **str, unsigned long long *len) {
    const unsigned char *bytes;
    if (!read_u64(reader, len)) return 0;
    if (!read_bytes(reader, *len, &bytes)) return 0;
    *str = (const char *)bytes;
    return 1;
}

static int skip_value(Reader *reader, unsigned type) {
    const unsigned char *bytes;
    unsigned long long len, count;
    unsigned elem_type;
    switch (type) {
    case GGUF_U8: case GGUF_I8: case GGUF_BOOL:
        return read_bytes(reader, 1, &bytes);
    case GGUF_U16: case GGUF_I16:
        return read_bytes(reader, 2, &bytes);
    case GGUF_U32: case GGUF_I32: case GGUF_F32:
        return read_bytes(reader, 4, &bytes);
    case GGUF_U64: case GGUF_I64: case GGUF_F64:
        return read_bytes(reader, 8, &bytes);
    case GGUF_STR:
        if (!read_u64(reader, &len)) return 0;
        return read_bytes(reader, len, &bytes);
    case GGUF_ARR:
        if (!read_u32(reader, &elem_type) || !read_u64(reader, &count)) return 0;
        for (unsigned long long i = 0; i < count; i++)
            if (!skip_value(reader, elem_type)) return 0;
        return 1;
    default:
        return 0;
    }
}

static int find_key(const GgufFile *g, const char *key, unsigned *type,
                    Reader *value) {
    Reader reader = {g->kv_start, g->kv_end};
    size_t key_len = strlen(key);
    for (unsigned long long i = 0; i < g->kv_count; i++) {
        const char *found;
        unsigned long long found_len;
        unsigned found_type;
        if (!read_str(&reader, &found, &found_len) ||
            !read_u32(&reader, &found_type))
            return 0;
        if (found_len == key_len && memcmp(found, key, key_len) == 0) {
            *type = found_type;
            *value = reader;
            return 1;
        }
        if (!skip_value(&reader, found_type)) return 0;
    }
    return 0;
}

int gguf_meta_u64(const GgufFile *g, const char *key,
                  unsigned long long *out) {
    unsigned type;
    Reader reader;
    const unsigned char *bytes;
    if (!find_key(g, key, &type, &reader)) return 0;
    switch (type) {
    case GGUF_U8: case GGUF_BOOL:
        if (!read_bytes(&reader, 1, &bytes)) return 0;
        *out = *bytes;
        return 1;
    case GGUF_U16:
        if (!read_bytes(&reader, 2, &bytes)) return 0;
        { unsigned short half; memcpy(&half, bytes, 2); *out = half; }
        return 1;
    case GGUF_U32: case GGUF_I32:
        if (!read_bytes(&reader, 4, &bytes)) return 0;
        { unsigned word; memcpy(&word, bytes, 4); *out = word; }
        return 1;
    case GGUF_U64: case GGUF_I64:
        return read_u64(&reader, out);
    default:
        return 0;
    }
}

int gguf_meta_f64(const GgufFile *g, const char *key, double *out) {
    unsigned type;
    Reader reader;
    const unsigned char *bytes;
    if (!find_key(g, key, &type, &reader)) return 0;
    switch (type) {
    case GGUF_F32:
        if (!read_bytes(&reader, 4, &bytes)) return 0;
        { float single; memcpy(&single, bytes, 4); *out = single; }
        return 1;
    case GGUF_F64:
        if (!read_bytes(&reader, 8, &bytes)) return 0;
        memcpy(out, bytes, 8);
        return 1;
    default:
        return 0;
    }
}

int gguf_meta_str(const GgufFile *g, const char *key, char *out,
                  size_t outsz) {
    unsigned type;
    Reader reader;
    const char *str;
    unsigned long long len;
    if (!find_key(g, key, &type, &reader) || type != GGUF_STR) return 0;
    if (!read_str(&reader, &str, &len)) return 0;
    if (len >= outsz) len = outsz - 1;
    memcpy(out, str, len);
    out[len] = '\0';
    return 1;
}

int gguf_meta_arr(const GgufFile *g, const char *key, GgufArray *out) {
    unsigned type, elem_type;
    Reader reader;
    if (!find_key(g, key, &type, &reader) || type != GGUF_ARR) return 0;
    if (!read_u32(&reader, &elem_type) || !read_u64(&reader, &out->remaining))
        return 0;
    out->elem_type = (GgufType)elem_type;
    out->next = reader.at;
    out->end = reader.end;
    return 1;
}

int gguf_array_next_str(GgufArray *array, const char **str, size_t *len) {
    if (array->elem_type != GGUF_STR || array->remaining == 0) return 0;

    Reader reader = {array->next, array->end};
    unsigned long long taken_len;
    if (!read_str(&reader, str, &taken_len)) return 0;
    array->next = reader.at;
    array->remaining--;
    *len = (size_t)taken_len;
    return 1;
}

const GgufTensor *gguf_find(const GgufFile *g, const char *name) {
    for (unsigned long long i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}

static int read_tensor_infos(GgufFile *g, Reader *reader,
                             unsigned long long count, char *err,
                             size_t errsz) {
    g->tensors = malloc(count * sizeof *g->tensors);
    if (!g->tensors) {
        snprintf(err, errsz, "out of memory");
        return 0;
    }
    for (unsigned long long i = 0; i < count; i++) {
        GgufTensor *tensor = &g->tensors[i];
        const char *name;
        unsigned long long name_len, offset;
        unsigned n_dims, type;
        if (!read_str(reader, &name, &name_len) || !read_u32(reader, &n_dims))
            return 0;
        if (name_len >= GGUF_NAME_MAX || n_dims < 1 || n_dims > GGUF_MAX_DIMS) {
            snprintf(err, errsz, "bad tensor info at index %llu", i);
            return 0;
        }
        memcpy(tensor->name, name, name_len);
        tensor->name[name_len] = '\0';
        tensor->n_dims = (int)n_dims;
        for (int d = 0; d < GGUF_MAX_DIMS; d++) tensor->dims[d] = 0;
        for (unsigned d = 0; d < n_dims; d++)
            if (!read_u64(reader, &tensor->dims[d])) return 0;
        if (!read_u32(reader, &type) || !read_u64(reader, &offset)) return 0;
        if (type >= (unsigned)n_quant_types || !quant_types[type].name) {
            snprintf(err, errsz, "%.100s: unknown tensor type %u", tensor->name,
                     type);
            return 0;
        }
        tensor->type = type;
        QuantType quant = quant_types[type];
        unsigned long long n_elements = 1;
        for (int d = 0; d < GGUF_MAX_DIMS; d++)
            n_elements *= tensor->dims[d] ? tensor->dims[d] : 1;
        tensor->n_bytes = (n_elements / quant.block_size) * quant.type_size;
        tensor->data = (const unsigned char *)g->map + offset;
    }
    return 1;
}

/* Tensor offsets are relative to the data section, whose start is only known
   once the infos have been read -- so read_tensor_infos parks each offset
   against the mapping and this moves them all once. */
static int place_tensors(GgufFile *g, const unsigned char *infos_end,
                         char *err, size_t errsz) {
    unsigned long long alignment = DEFAULT_ALIGNMENT;
    gguf_meta_u64(g, "general.alignment", &alignment);

    const unsigned char *base = g->map;
    unsigned long long data_off = (unsigned long long)(infos_end - base);
    data_off = (data_off + alignment - 1) / alignment * alignment;

    for (unsigned long long i = 0; i < g->n_tensors; i++) {
        GgufTensor *tensor = &g->tensors[i];
        unsigned long long offset =
            (unsigned long long)((const unsigned char *)tensor->data - base);
        if (data_off + offset + tensor->n_bytes > g->map_size) {
            snprintf(err, errsz, "%.100s: tensor data out of bounds",
                     tensor->name);
            return 0;
        }
        tensor->data = base + data_off + offset;
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
    struct stat status;
    if (fstat(g->fd, &status) || status.st_size <= 0) {
        snprintf(err, errsz, "%s: cannot stat", path);
        goto fail;
    }
    g->map_size = (unsigned long long)status.st_size;
    g->map = mmap(NULL, g->map_size, PROT_READ, MAP_PRIVATE, g->fd, 0);
    if (g->map == MAP_FAILED) {
        snprintf(err, errsz, "%s: mmap: %s", path, strerror(errno));
        goto fail;
    }

    Reader reader = {g->map, (const unsigned char *)g->map +
                                    g->map_size};
    unsigned magic, version;
    unsigned long long tensor_count;
    if (!read_u32(&reader, &magic) || magic != GGUF_MAGIC) {
        snprintf(err, errsz, "%s: not a GGUF file", path);
        goto fail;
    }
    if (!read_u32(&reader, &version) || version < 2 || version > 3 ||
        !read_u64(&reader, &tensor_count) || !read_u64(&reader, &g->kv_count)) {
        snprintf(err, errsz, "%s: bad GGUF header", path);
        goto fail;
    }

    g->kv_start = reader.at;
    for (unsigned long long i = 0; i < g->kv_count; i++) {
        const char *key;
        unsigned long long key_len;
        unsigned type;
        if (!read_str(&reader, &key, &key_len) || !read_u32(&reader, &type) ||
            !skip_value(&reader, type)) {
            snprintf(err, errsz, "%s: metadata parse error", path);
            goto fail;
        }
    }
    g->kv_end = reader.at;

    if (!read_tensor_infos(g, &reader, tensor_count, err, errsz)) {
        if (!err[0]) snprintf(err, errsz, "%s: tensor info parse error", path);
        goto fail;
    }
    g->n_tensors = tensor_count;
    if (!place_tensors(g, reader.at, err, errsz)) goto fail;
    return 1;

fail:
    gguf_close(g);
    return 0;
}

void gguf_close(GgufFile *g) {
    if (g->map && g->map != MAP_FAILED) munmap(g->map, g->map_size);
    if (g->fd >= 0) close(g->fd);
    free(g->tensors);
    memset(g, 0, sizeof *g);
    g->fd = -1;
}
