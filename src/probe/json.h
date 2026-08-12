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

#endif