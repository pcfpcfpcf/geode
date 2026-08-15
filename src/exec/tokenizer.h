#ifndef GEODE_TOKENIZER_H
#define GEODE_TOKENIZER_H

#include "gguf.h"

typedef struct StringMap StringMap;

/* Token text points into the model's mapping rather than being copied, so a
   Tokenizer must not outlive the GgufFile it was built from. */
typedef struct {
    const char *text;
    int length;
} Token;

typedef struct {
    Token *tokens;
    int n_tokens;
    int bos_id;
    int eos_id;
    int add_bos;
    StringMap *vocab;
    StringMap *merges;
} Tokenizer;

int tokenizer_init(Tokenizer *t, const GgufFile *g, char *err, size_t errsz);
void tokenizer_free(Tokenizer *t);

/* Text needing more than max_ids is truncated, so a caller that must not lose
   any compares the result against max_ids. */
int tokenizer_encode(const Tokenizer *t, const char *text, int *ids,
                     int max_ids);

/* The same, behind the leading token the model asks for. */
int tokenizer_encode_prompt(const Tokenizer *t, const char *text, int *ids,
                            int max_ids);
int tokenizer_decode(const Tokenizer *t, const int *ids, int n_ids, char *out,
                     int max_bytes);

#endif
