#include "json.h"
#include "manifest.h"
#include "modules.h"
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* GGUF is little-endian on disk; this reader assumes a little-endian host. */

#define GGUF_MAGIC 0x46554747u /* "GGUF" */
#define KV_DTYPE_BYTES 2       /* KV cache stored as f16 */

typedef enum {
    T_EMBED,
    T_ATTENTION,
    T_EXPERT,
    T_BASE
} TensorClass;

typedef struct {
    TensorClass cls;
    unsigned long long n_elements;
    unsigned long long n_bytes;
    unsigned long long out_dim; /* dim[1]: rows for embed, kv width for attn_k/v */
} TensorInfo;

typedef struct {
    FILE *f;
    char err[256];
} Reader;

static int read_raw(Reader *r, void *buf, size_t n) {
    if (fread(buf, 1, n, r->f) != n) {
        snprintf(r->err, sizeof r->err, "unexpected end of file");
        return 0;
    }
    return 1;
}

static int read_u32(Reader *r, unsigned *out) {
    return read_raw(r, out, 4);
}

static int read_u64(Reader *r, unsigned long long *out) {
    return read_raw(r, out, 8);
}

static int read_string(Reader *r, char *buf, size_t bufsz) {
    unsigned long long len;
    if (!read_u64(r, &len)) return 0;
    if (len >= bufsz) {
        snprintf(r->err, sizeof r->err, "string too long (%llu bytes)", len);
        return 0;
    }
    if (!read_raw(r, buf, len)) return 0;
    buf[len] = '\0';
    return 1;
}

enum {
    GGUF_U8, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STR, GGUF_ARR, GGUF_U64, GGUF_I64, GGUF_F64
};

static int skip_value(Reader *r, unsigned type);

static int skip_array(Reader *r) {
    unsigned elem_type;
    unsigned long long count;
    if (!read_u32(r, &elem_type) || !read_u64(r, &count)) return 0;
    for (unsigned long long i = 0; i < count; i++)
        if (!skip_value(r, elem_type)) return 0;
    return 1;
}

static int skip_value(Reader *r, unsigned type) {
    char strbuf[4096];
    switch (type) {
    case GGUF_U8: case GGUF_I8: case GGUF_BOOL:
        return read_raw(r, strbuf, 1);
    case GGUF_U16: case GGUF_I16:
        return read_raw(r, strbuf, 2);
    case GGUF_U32: case GGUF_I32: case GGUF_F32:
        return read_raw(r, strbuf, 4);
    case GGUF_U64: case GGUF_I64: case GGUF_F64:
        return read_raw(r, strbuf, 8);
    case GGUF_STR: {
        unsigned long long len;
        if (!read_u64(r, &len)) return 0;
        return fseek(r->f, (long)len, SEEK_CUR) == 0;
    }
    case GGUF_ARR:
        return skip_array(r);
    default:
        snprintf(r->err, sizeof r->err, "unknown metadata type %u", type);
        return 0;
    }
}

static int read_scalar_u64(Reader *r, unsigned type, unsigned long long *out) {
    unsigned char b;
    unsigned short s;
    unsigned w;
    unsigned long long q;
    switch (type) {
    case GGUF_U8: case GGUF_BOOL:
        if (!read_raw(r, &b, 1)) return 0;
        *out = b;
        return 1;
    case GGUF_U16:
        if (!read_raw(r, &s, 2)) return 0;
        *out = s;
        return 1;
    case GGUF_U32:
        if (!read_raw(r, &w, 4)) return 0;
        *out = w;
        return 1;
    case GGUF_U64:
        if (!read_raw(r, &q, 8)) return 0;
        *out = q;
        return 1;
    default:
        return skip_value(r, type);
    }
}

static int ends_with(const char *s, const char *suffix) {
    size_t ns = strlen(s), nx = strlen(suffix);
    return ns >= nx && strcmp(s + ns - nx, suffix) == 0;
}

static TensorClass classify(const char *name) {
    if (strcmp(name, "token_embd.weight") == 0) return T_EMBED;
    if (strstr(name, "_exps.")) return T_EXPERT;
    if (strstr(name, ".attn_")) return T_ATTENTION;
    return T_BASE;
}

static void print_manifest(const Manifest *m) {
    printf("model:   %s (%s), %llu layers, %llu experts (%llu used/token)\n",
           m->model, m->architecture, m->n_layer, m->expert_count,
           m->expert_used);
    printf("params:  %.2fB total, %.2fB active\n", m->total_params / 1e9,
           m->active_params / 1e9);
    printf("bytes/token: attention %.1f MB, base %.1f MB, routed %.1f MB "
           "(%llu experts x %.2f MB)\n",
           m->attention_bytes / 1e6, m->base_bytes / 1e6,
           m->routed_bytes / 1e6, m->expert_used, m->expert_bytes / 1e6);
    printf("weights: %.2f GB on disk, kv cache %.1f KB per context token\n",
           m->total_bytes / 1e9, m->kv_bytes_per_ctx_token / 1e3);
}

static void write_manifest_json(FILE *out, const Manifest *m) {
    Json j;
    json_begin(&j, out);
    json_u64(&j, "schema", 1);
    json_string(&j, "model", m->model);
    json_string(&j, "source", m->source);
    json_string(&j, "variant", m->variant);
    json_string(&j, "architecture", m->architecture);
    json_u64(&j, "total_params", m->total_params);
    json_u64(&j, "active_params", m->active_params);
    json_u64(&j, "n_layer", m->n_layer);
    json_open(&j, "components", 0);
    json_open(&j, "attention", 0);
    json_u64(&j, "bytes_per_token", m->attention_bytes);
    json_string(&j, "pattern", "deterministic");
    json_close(&j);
    json_open(&j, "base", 0);
    json_u64(&j, "bytes_per_token", m->base_bytes);
    json_string(&j, "pattern", "deterministic");
    json_close(&j);
    json_open(&j, "experts", 0);
    json_u64(&j, "count", m->expert_count);
    json_u64(&j, "used_per_token", m->expert_used);
    json_u64(&j, "bytes_each", m->expert_bytes);
    json_u64(&j, "bytes_per_token", m->routed_bytes);
    json_string(&j, "pattern", "stochastic");
    json_close(&j);
    json_close(&j);
    json_open(&j, "kv_cache", 0);
    json_u64(&j, "bytes_per_context_token", m->kv_bytes_per_ctx_token);
    json_close(&j);
    json_open(&j, "hash_routed_layers", 1);
    json_close(&j);
    json_u64(&j, "mtp_head", (unsigned long long)m->mtp_head);
    json_open(&j, "variants", 1);
    json_string(&j, NULL, "full-weight");
    json_close(&j);
    json_end(&j);
}

static double jnum(const JVal *obj, const char *key, int *ok) {
    const JVal *v = json_get(obj, key);
    if (!v || v->kind != JV_NUM) {
        *ok = 0;
        return 0;
    }
    return v->num;
}

static void jstr(const JVal *obj, const char *key, char *out, size_t outsz) {
    const JVal *v = json_get(obj, key);
    out[0] = '\0';
    if (v && v->kind == JV_STR) snprintf(out, outsz, "%s", v->str);
}

int manifest_load(const char *path, Manifest *m, char *err, size_t errsz) {
    char *text = json_read_file(path, err, errsz);
    if (!text) return 0;
    JVal *root = json_parse(text, err, errsz);
    free(text);
    if (!root) return 0;

    memset(m, 0, sizeof *m);
    int ok = 1;
    jstr(root, "model", m->model, sizeof m->model);
    jstr(root, "source", m->source, sizeof m->source);
    jstr(root, "variant", m->variant, sizeof m->variant);
    jstr(root, "architecture", m->architecture, sizeof m->architecture);
    m->total_params = (unsigned long long)jnum(root, "total_params", &ok);
    m->active_params = (unsigned long long)jnum(root, "active_params", &ok);
    m->n_layer = (unsigned long long)jnum(root, "n_layer", &ok);
    m->mtp_head = (int)jnum(root, "mtp_head", &ok);

    const JVal *comp = json_get(root, "components");
    const JVal *att = comp ? json_get(comp, "attention") : NULL;
    const JVal *base = comp ? json_get(comp, "base") : NULL;
    const JVal *exp = comp ? json_get(comp, "experts") : NULL;
    const JVal *kv = json_get(root, "kv_cache");
    if (!att || !base || !exp || !kv) ok = 0;
    if (ok) {
        m->attention_bytes = (unsigned long long)jnum(att, "bytes_per_token", &ok);
        m->base_bytes = (unsigned long long)jnum(base, "bytes_per_token", &ok);
        m->expert_count = (unsigned long long)jnum(exp, "count", &ok);
        m->expert_used = (unsigned long long)jnum(exp, "used_per_token", &ok);
        m->expert_bytes = (unsigned long long)jnum(exp, "bytes_each", &ok);
        m->routed_bytes = (unsigned long long)jnum(exp, "bytes_per_token", &ok);
        m->kv_bytes_per_ctx_token =
            (unsigned long long)jnum(kv, "bytes_per_context_token", &ok);
    }
    json_free(root);
    if (!ok) {
        snprintf(err, errsz, "%.200s: missing or bad manifest fields", path);
        return 0;
    }
    return 1;
}

static const char *default_out_path(void) {
    static char buf[1024];
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(buf, sizeof buf, "%s/.geode/manifest.json", home);
    return buf;
}

int manifest_main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s MODEL.gguf [OUT.json]\n", argv[0]);
        return 2;
    }
    const char *gguf_path = argv[1];
    const char *out_path = argc == 3 ? argv[2] : default_out_path();

    Reader r = {NULL, {0}};
    r.f = fopen(gguf_path, "rb");
    if (!r.f) {
        perror(gguf_path);
        return 1;
    }

    unsigned magic, version;
    unsigned long long tensor_count, kv_count;
    if (!read_u32(&r, &magic) || magic != GGUF_MAGIC) {
        fprintf(stderr, "%s: not a GGUF file\n", gguf_path);
        return 1;
    }
    if (!read_u32(&r, &version) || version < 2 || version > 3 ||
        !read_u64(&r, &tensor_count) || !read_u64(&r, &kv_count))
        goto fail;

    char model_name[256] = "";
    char arch[128] = "";
    unsigned long long n_layer = 0, n_expert = 0, n_expert_used = 0;
    for (unsigned long long i = 0; i < kv_count; i++) {
        char key[256];
        unsigned type;
        if (!read_string(&r, key, sizeof key) || !read_u32(&r, &type))
            goto fail;
        if (strcmp(key, "general.name") == 0) {
            if (!read_string(&r, model_name, sizeof model_name)) goto fail;
        } else if (strcmp(key, "general.architecture") == 0) {
            if (!read_string(&r, arch, sizeof arch)) goto fail;
        } else if (ends_with(key, ".block_count")) {
            if (!read_scalar_u64(&r, type, &n_layer)) goto fail;
        } else if (ends_with(key, ".expert_count")) {
            if (!read_scalar_u64(&r, type, &n_expert)) goto fail;
        } else if (ends_with(key, ".expert_used_count")) {
            if (!read_scalar_u64(&r, type, &n_expert_used)) goto fail;
        } else {
            if (!skip_value(&r, type)) goto fail;
        }
    }
    if (model_name[0] == '\0') {
        const char *base = strrchr(gguf_path, '/');
        snprintf(model_name, sizeof model_name, "%s", base ? base + 1 : gguf_path);
    }
    if (n_layer == 0 || n_expert == 0 || n_expert_used == 0) {
        fprintf(stderr, "%s: missing MoE metadata (layers=%llu experts=%llu "
                "used=%llu); not a routed MoE model?\n",
                gguf_path, n_layer, n_expert, n_expert_used);
        return 1;
    }

    TensorInfo *tensors = calloc(tensor_count, sizeof *tensors);
    unsigned long long embed_row_bytes = 0, embed_elements = 0;
    unsigned long long embed_bytes = 0, kv_dim = 0;
    int have_output_head = 0;
    for (unsigned long long i = 0; i < tensor_count; i++) {
        char name[256];
        unsigned n_dims, type;
        if (!read_string(&r, name, sizeof name) || !read_u32(&r, &n_dims))
            goto fail;
        if (n_dims < 1 || n_dims > 4) {
            snprintf(r.err, sizeof r.err, "%.200s: bad dim count %u", name,
                     n_dims);
            goto fail;
        }
        unsigned long long dims[4] = {0, 0, 0, 0};
        for (unsigned d = 0; d < n_dims; d++)
            if (!read_u64(&r, &dims[d])) goto fail;
        unsigned long long offset;
        if (!read_u32(&r, &type) || !read_u64(&r, &offset)) goto fail;
        if (type >= (unsigned)n_quant_types || !quant_types[type].name) {
            snprintf(r.err, sizeof r.err, "%.200s: unknown tensor type %u",
                     name, type);
            goto fail;
        }
        QuantType qt = quant_types[type];
        unsigned long long n_elements = 1;
        for (int d = 0; d < 4; d++) n_elements *= dims[d] ? dims[d] : 1;
        tensors[i].cls = classify(name);
        tensors[i].n_elements = n_elements;
        tensors[i].n_bytes = (n_elements / qt.block_size) * qt.type_size;
        tensors[i].out_dim = n_dims >= 2 ? dims[1] : 0;
        if (tensors[i].cls == T_EMBED && tensors[i].out_dim) {
            embed_row_bytes = tensors[i].n_bytes / tensors[i].out_dim;
            embed_elements = n_elements;
            embed_bytes = tensors[i].n_bytes;
        }
        if (strcmp(name, "output.weight") == 0) have_output_head = 1;
        /* MHA/GQA: separate K and V projections; MLA (deepseek2): one
           compressed kv+rope projection. */
        if (strcmp(name, "blk.0.attn_k.weight") == 0) kv_dim += dims[1];
        if (strcmp(name, "blk.0.attn_v.weight") == 0) kv_dim += dims[1];
        if (strcmp(name, "blk.0.attn_kv_a_mqa.weight") == 0) kv_dim = dims[1];
    }
    fclose(r.f);

    unsigned long long attention_bytes = 0, base_bytes = 0, expert_bytes = 0;
    unsigned long long total_bytes = 0, total_params = 0, expert_params = 0;
    for (unsigned long long i = 0; i < tensor_count; i++) {
        total_bytes += tensors[i].n_bytes;
        total_params += tensors[i].n_elements;
        switch (tensors[i].cls) {
        case T_ATTENTION: attention_bytes += tensors[i].n_bytes; break;
        case T_EXPERT:
            expert_bytes += tensors[i].n_bytes;
            expert_params += tensors[i].n_elements;
            break;
        case T_BASE: base_bytes += tensors[i].n_bytes; break;
        case T_EMBED: break;
        }
    }
    base_bytes += embed_row_bytes; /* one embedding row lookup per token */
    if (!have_output_head)
        base_bytes += embed_bytes; /* tied LM head reads all rows per token */

    if (kv_dim == 0) {
        fprintf(stderr, "%s: no blk.0.attn_k/v.weight or attn_kv_a_mqa.weight; "
                "unsupported attention layout\n", gguf_path);
        return 1;
    }

    unsigned long long expert_bytes_each = expert_bytes / n_expert;
    unsigned long long routed_per_token = expert_bytes_each * n_expert_used;
    unsigned long long active_params =
        total_params - expert_params * (n_expert - n_expert_used) / n_expert
        - embed_elements;
    unsigned long long kv_bytes_per_ctx_token =
        kv_dim * n_layer * KV_DTYPE_BYTES;

    struct stat st;
    if (stat(gguf_path, &st) == 0 && (unsigned long long)st.st_size < total_bytes) {
        fprintf(stderr, "%s: tensor sizes (%llu) exceed file size (%lld); "
                "parse error\n", gguf_path, total_bytes, (long long)st.st_size);
        return 1;
    }

    Manifest info = {0};
    snprintf(info.model, sizeof info.model, "%s", model_name);
    snprintf(info.source, sizeof info.source, "%s", gguf_path);
    snprintf(info.variant, sizeof info.variant, "full-weight");
    snprintf(info.architecture, sizeof info.architecture, "%s", arch);
    info.total_params = total_params;
    info.active_params = active_params;
    info.n_layer = n_layer;
    info.total_bytes = total_bytes;
    info.attention_bytes = attention_bytes;
    info.base_bytes = base_bytes;
    info.expert_count = n_expert;
    info.expert_used = n_expert_used;
    info.expert_bytes = expert_bytes_each;
    info.routed_bytes = routed_per_token;
    info.kv_bytes_per_ctx_token = kv_bytes_per_ctx_token;

    FILE *out = fopen(out_path, "w");
    if (!out) {
        perror(out_path);
        return 1;
    }
    write_manifest_json(out, &info);
    if (fclose(out)) {
        perror(out_path);
        return 1;
    }

    print_manifest(&info);
    printf("wrote %s\n", out_path);

    free(tensors);
    return 0;

fail:
    fprintf(stderr, "%s: %s\n", gguf_path, r.err[0] ? r.err : "parse error");
    return 1;
}
