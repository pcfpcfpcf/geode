#include "json.h"
#include <stdlib.h>
#include <string.h>

static void json_putsep(Json *j) {
    if (j->has_item[j->depth]) fputc(',', j->f);
    fputc('\n', j->f);
    for (int i = 0; i < j->depth; i++) fputs("  ", j->f);
}

static void json_key(Json *j, const char *key) {
    if (!j->is_array[j->depth] && key) {
        fputc('"', j->f);
        fputs(key, j->f);
        fputs("\": ", j->f);
    }
}

void json_begin(Json *j, FILE *f) {
    j->f = f;
    j->depth = 0;
    j->is_array[0] = 0;
    j->has_item[0] = 0;
    fputc('{', j->f);
}

void json_open(Json *j, const char *key, int as_array) {
    json_putsep(j);
    json_key(j, key);
    fputc(as_array ? '[' : '{', j->f);
    j->depth++;
    j->is_array[j->depth] = as_array;
    j->has_item[j->depth] = 0;
}

void json_close(Json *j) {
    fputc('\n', j->f);
    j->depth--;
    for (int i = 0; i < j->depth; i++) fputs("  ", j->f);
    fputc(j->is_array[j->depth + 1] ? ']' : '}', j->f);
    j->has_item[j->depth] = 1;
}

void json_string(Json *j, const char *key, const char *val) {
    json_putsep(j);
    json_key(j, key);
    fputc('"', j->f);
    for (const char *p = val; *p; p++) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', j->f);
            fputc(*p, j->f);
        } else if ((unsigned char)*p < 0x20) {
            fprintf(j->f, "\\u%04x", (unsigned char)*p);
        } else {
            fputc(*p, j->f);
        }
    }
    fputc('"', j->f);
    j->has_item[j->depth] = 1;
}

void json_u64(Json *j, const char *key, unsigned long long val) {
    json_putsep(j);
    json_key(j, key);
    fprintf(j->f, "%llu", val);
    j->has_item[j->depth] = 1;
}

void json_double(Json *j, const char *key, double val) {
    json_putsep(j);
    json_key(j, key);
    fprintf(j->f, "%.1f", val);
    j->has_item[j->depth] = 1;
}

void json_end(Json *j) {
    fputc('\n', j->f);
    fputc('}', j->f);
    fputc('\n', j->f);
}

typedef struct {
    const char *start;
    const char *p;
    char *err;
    size_t errsz;
} Parser;

static void parse_fail(Parser *ps, const char *msg) {
    if (ps->err[0] == '\0')
        snprintf(ps->err, ps->errsz, "%s near byte %ld", msg,
                 (long)(ps->p - ps->start));
}

static void skip_ws(Parser *ps) {
    while (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')
        ps->p++;
}

static JVal *new_val(JValKind kind) {
    JVal *v = calloc(1, sizeof *v);
    v->kind = kind;
    return v;
}

static JVal *parse_value(Parser *ps);

static JVal *parse_string(Parser *ps) {
    if (*ps->p != '"') {
        parse_fail(ps, "expected string");
        return NULL;
    }
    ps->p++;
    size_t cap = 32, n = 0;
    char *buf = malloc(cap);
    while (*ps->p && *ps->p != '"') {
        char c = *ps->p++;
        if (c == '\\') {
            char esc = *ps->p++;
            switch (esc) {
            case 'n': c = '\n'; break;
            case 't': c = '\t'; break;
            case 'r': c = '\r'; break;
            case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'u': {
                if (strlen(ps->p) < 4) goto bad;
                unsigned cp;
                if (sscanf(ps->p, "%4x", &cp) != 1) goto bad;
                ps->p += 4;
                c = cp < 0x80 ? (char)cp : '?';
                break;
            }
            case '"': case '\\': case '/': c = esc; break;
            default: goto bad;
            }
        }
        if (n + 1 >= cap) {
            cap *= 2;
            buf = realloc(buf, cap);
        }
        buf[n++] = c;
    }
    if (*ps->p != '"') goto bad;
    ps->p++;
    buf[n] = '\0';
    JVal *v = new_val(JV_STR);
    v->str = buf;
    return v;
bad:
    free(buf);
    parse_fail(ps, "bad string");
    return NULL;
}

static int append_item(JVal *container, const char *key, JVal *item) {
    int n = container->n_items;
    JVal **items = realloc(container->items, (n + 1) * sizeof *items);
    if (!items) return 0;
    container->items = items;
    container->items[n] = item;
    if (key) {
        char **keys = realloc(container->keys, (n + 1) * sizeof *keys);
        if (!keys) return 0;
        container->keys = keys;
        container->keys[n] = strdup(key);
    }
    container->n_items = n + 1;
    return 1;
}

static JVal *parse_array(Parser *ps) {
    ps->p++;
    JVal *v = new_val(JV_ARR);
    skip_ws(ps);
    if (*ps->p == ']') {
        ps->p++;
        return v;
    }
    for (;;) {
        JVal *item = parse_value(ps);
        if (!item || !append_item(v, NULL, item)) goto bad;
        skip_ws(ps);
        if (*ps->p == ',') {
            ps->p++;
            skip_ws(ps);
            continue;
        }
        if (*ps->p == ']') {
            ps->p++;
            return v;
        }
        goto bad;
    }
bad:
    parse_fail(ps, "bad array");
    json_free(v);
    return NULL;
}

static JVal *parse_object(Parser *ps) {
    ps->p++;
    JVal *v = new_val(JV_OBJ);
    skip_ws(ps);
    if (*ps->p == '}') {
        ps->p++;
        return v;
    }
    for (;;) {
        skip_ws(ps);
        JVal *key = parse_string(ps);
        if (!key) goto bad;
        skip_ws(ps);
        if (*ps->p != ':') {
            json_free(key);
            goto bad;
        }
        ps->p++;
        skip_ws(ps);
        JVal *item = parse_value(ps);
        if (!item || !append_item(v, key->str, item)) {
            json_free(key);
            goto bad;
        }
        json_free(key);
        skip_ws(ps);
        if (*ps->p == ',') {
            ps->p++;
            continue;
        }
        if (*ps->p == '}') {
            ps->p++;
            return v;
        }
        goto bad;
    }
bad:
    parse_fail(ps, "bad object");
    json_free(v);
    return NULL;
}

static JVal *parse_value(Parser *ps) {
    skip_ws(ps);
    switch (*ps->p) {
    case '{': return parse_object(ps);
    case '[': return parse_array(ps);
    case '"': return parse_string(ps);
    case 't':
        if (strncmp(ps->p, "true", 4) == 0) {
            ps->p += 4;
            JVal *v = new_val(JV_BOOL);
            v->num = 1;
            return v;
        }
        break;
    case 'f':
        if (strncmp(ps->p, "false", 5) == 0) {
            ps->p += 5;
            return new_val(JV_BOOL);
        }
        break;
    case 'n':
        if (strncmp(ps->p, "null", 4) == 0) {
            ps->p += 4;
            return new_val(JV_NULL);
        }
        break;
    default: {
        char *end;
        double d = strtod(ps->p, &end);
        if (end != ps->p) {
            ps->p = end;
            JVal *v = new_val(JV_NUM);
            v->num = d;
            return v;
        }
    }
    }
    parse_fail(ps, "bad value");
    return NULL;
}

JVal *json_parse(const char *text, char *err, size_t errsz) {
    Parser ps = {text, text, err, errsz};
    err[0] = '\0';
    JVal *v = parse_value(&ps);
    if (v) {
        skip_ws(&ps);
        if (*ps.p != '\0') {
            parse_fail(&ps, "trailing data");
            json_free(v);
            v = NULL;
        }
    }
    return v;
}

const JVal *json_get(const JVal *obj, const char *key) {
    if (!obj || obj->kind != JV_OBJ) return NULL;
    for (int i = 0; i < obj->n_items; i++)
        if (strcmp(obj->keys[i], key) == 0) return obj->items[i];
    return NULL;
}

void json_free(JVal *v) {
    if (!v) return;
    free(v->str);
    for (int i = 0; i < v->n_items; i++) json_free(v->items[i]);
    free(v->items);
    for (int i = 0; i < v->n_items; i++)
        if (v->keys) free(v->keys[i]);
    free(v->keys);
    free(v);
}

char *json_read_file(const char *path, char *err, size_t errsz) {
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(err, errsz, "cannot open %s", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(size + 1);
    if (fread(buf, 1, size, f) != (size_t)size) {
        snprintf(err, errsz, "cannot read %s", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[size] = '\0';
    fclose(f);
    return buf;
}