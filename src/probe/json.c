#include "json.h"
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