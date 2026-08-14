#ifndef GEODE_TOKENIZER_H
#define GEODE_TOKENIZER_H

#include "gguf.h"

typedef struct {
    int n_tokens;
    const char **tokens;   /* byte-encoded token strings, owned */
    int *token_types;
    int bos_id, eos_id;
    int add_bos;
    /* hash maps */
    struct StrMap *vocab;   /* token string -> id */
    struct StrMap *merges;  /* "left right" -> rank */
} Tokenizer;

int tokenizer_init(Tokenizer *t, const GgufFile *g, char *err, size_t errsz);
void tokenizer_free(Tokenizer *t);
int tokenizer_encode(const Tokenizer *t, const char *text, int *out,
                     int max_out);
int tokenizer_decode(const Tokenizer *t, const int *ids, int n_ids,
                     char *out, int max_out);

#endif
