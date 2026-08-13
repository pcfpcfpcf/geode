#ifndef GEODE_JSON_H
#define GEODE_JSON_H

#include <stdio.h>

typedef struct {
    FILE *f;
    int is_array[16];
    int has_item[16];
    int depth;
} Json;

void json_begin(Json *j, FILE *f);
void json_open(Json *j, const char *key, int as_array);
void json_close(Json *j);
void json_string(Json *j, const char *key, const char *val);
void json_u64(Json *j, const char *key, unsigned long long val);
void json_double(Json *j, const char *key, double val);
void json_end(Json *j);

typedef enum { JV_NULL, JV_BOOL, JV_NUM, JV_STR, JV_ARR, JV_OBJ } JValKind;

typedef struct JVal JVal;
struct JVal {
    JValKind kind;
    double num;                 /* JV_NUM; JV_BOOL: 0 or 1 */
    char *str;                  /* JV_STR */
    JVal **items;               /* JV_ARR */
    int n_items;
    char **keys;                /* JV_OBJ: keys[i] -> items[i] */
};

/* Returns NULL and fills err on failure. Caller frees with json_free. */
JVal *json_parse(const char *text, char *err, size_t errsz);
const JVal *json_get(const JVal *obj, const char *key);
void json_free(JVal *v);

/* Read a whole file; returns NULL on failure. */
char *json_read_file(const char *path, char *err, size_t errsz);

#endif