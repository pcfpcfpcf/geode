#include "tokenizer.h"
#include "unicode_ranges.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GPT-2 byte-level BPE, ported from llama.cpp (llm_tokenizer_bpe). */

#define UC_UNDEFINED  0x0001
#define UC_NUMBER     0x0002
#define UC_LETTER     0x0004
#define UC_WHITESPACE 0x0100

#define REPLACEMENT_CPT 0xFFFD

/* Only reached by a model that names neither; llama.cpp assumes the same two
   in the same place. */
#define DEFAULT_BOS_ID 1
#define DEFAULT_EOS_ID 2

static uint16_t cpt_flags(uint32_t cpt) {
    if (cpt >= 0x110000) return UC_UNDEFINED;
    int lo = 0, hi = n_uc_ranges - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (uc_ranges[mid].start <= cpt) lo = mid;
        else hi = mid - 1;
    }
    uint16_t flags = uc_ranges[lo].start <= cpt ? uc_ranges[lo].flags : 0;
    for (int i = 0; i < n_uc_whitespace; i++)
        if (uc_whitespace[i] == cpt) return flags | UC_WHITESPACE;
    return flags;
}

static int utf8_len(uint8_t lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1;
}

static uint32_t utf8_decode(const char *s, int *len) {
    uint8_t b = (uint8_t)s[0];
    *len = utf8_len(b);
    switch (*len) {
    case 2:
        return ((uint32_t)(b & 0x1F) << 6) | ((uint8_t)s[1] & 0x3F);
    case 3:
        return ((uint32_t)(b & 0x0F) << 12) | (((uint8_t)s[1] & 0x3F) << 6) |
               ((uint8_t)s[2] & 0x3F);
    case 4:
        return ((uint32_t)(b & 0x07) << 18) | (((uint8_t)s[1] & 0x3F) << 12) |
               (((uint8_t)s[2] & 0x3F) << 6) | ((uint8_t)s[3] & 0x3F);
    default:
        return b;
    }
}

static int utf8_encode(uint32_t cpt, char *out) {
    if (cpt < 0x80) {
        out[0] = (char)cpt;
        return 1;
    }
    if (cpt < 0x800) {
        out[0] = (char)(0xC0 | (cpt >> 6));
        out[1] = (char)(0x80 | (cpt & 0x3F));
        return 2;
    }
    if (cpt < 0x10000) {
        out[0] = (char)(0xE0 | (cpt >> 12));
        out[1] = (char)(0x80 | ((cpt >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cpt & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cpt >> 18));
    out[1] = (char)(0x80 | ((cpt >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cpt >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cpt & 0x3F));
    return 4;
}

/* GPT-2 gives every byte a printable codepoint so a token is always valid
   text: 188 bytes stand for themselves and the other 68 are lifted clear of
   them, which keeps every mapped codepoint under CPT_LIMIT. */
#define CPT_LIMIT 0x200
#define NO_BYTE   0x100

static uint32_t cpt_of_byte[256];
static uint16_t byte_of_cpt[CPT_LIMIT];

/* A sentinel rather than a zeroed table, because byte 0 is itself a value the
   inverse has to be able to return. */
static void init_byte_encoding(void) {
    for (int cpt = 0; cpt < CPT_LIMIT; cpt++) byte_of_cpt[cpt] = NO_BYTE;
    int lifted = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
                        b >= 0xAE;
        uint32_t cpt = printable ? (uint32_t)b : (uint32_t)(256 + lifted++);
        cpt_of_byte[b] = cpt;
        byte_of_cpt[cpt] = (uint16_t)b;
    }
}

typedef struct {
    const char *key;
    int key_len;
    int val;
} MapEntry;

/* Sized for its whole contents at init and never grown, which is what lets an
   entry borrow its key from the mapping instead of copying it. */
struct StringMap {
    MapEntry *entries;
    int cap;
};

static unsigned long str_hash(const char *s, int len) {
    unsigned long h = 1469598103934665603UL;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211UL;
    }
    return h;
}

static StringMap *map_create(int n_entries) {
    StringMap *m = calloc(1, sizeof *m);
    if (!m) return NULL;
    m->cap = 16;
    while (m->cap < 2 * n_entries) m->cap *= 2;
    m->entries = calloc((size_t)m->cap, sizeof *m->entries);
    if (!m->entries) {
        free(m);
        return NULL;
    }
    return m;
}

static void map_free(StringMap *m) {
    if (!m) return;
    free(m->entries);
    free(m);
}

/* The matching entry, or the empty one it would go in. Terminates because the
   table is never more than half full. */
static MapEntry *map_slot(const StringMap *m, const char *key, int key_len) {
    unsigned long h = str_hash(key, key_len) & (unsigned long)(m->cap - 1);
    while (m->entries[h].key &&
           (m->entries[h].key_len != key_len ||
            memcmp(m->entries[h].key, key, (size_t)key_len) != 0))
        h = (h + 1) & (unsigned long)(m->cap - 1);
    return &m->entries[h];
}

static void map_put(StringMap *m, const char *key, int key_len, int val) {
    MapEntry *entry = map_slot(m, key, key_len);
    entry->key = key;
    entry->key_len = key_len;
    entry->val = val;
}

static int map_get(const StringMap *m, const char *key, int key_len, int *val) {
    const MapEntry *entry = map_slot(m, key, key_len);
    if (!entry->key) return 0;
    *val = entry->val;
    return 1;
}

typedef struct {
    int *lens;
    int count;
    int cap;
    int start;
} WordList;

static void close_word(WordList *words, int end) {
    if (end > words->start && words->count < words->cap)
        words->lens[words->count++] = end - words->start;
    words->start = end;
}

/* 's|'t|'re|'ve|'m|'ll|'d */
static int contraction_len(const uint32_t *cpts, int n, int pos) {
    if (cpts[pos] != '\'' || pos + 1 >= n) return 0;
    uint32_t a = cpts[pos + 1];
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (pos + 2 >= n) return 0;
    uint32_t b = cpts[pos + 2];
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
        (a == 'l' && b == 'l'))
        return 3;
    return 0;
}

/* `kind` 0 is the run of everything that is neither space, letter nor
   number -- the regex's [^\s\p{L}\p{N}] class. */
static int in_run(uint16_t flags, uint16_t kind) {
    if (kind) return (flags & kind) != 0;
    return flags && !(flags & (UC_WHITESPACE | UC_LETTER | UC_NUMBER));
}

static int whitespace_len(const uint32_t *cpts, int n, int pos) {
    int len = 0;
    while (pos + len < n && (cpt_flags(cpts[pos + len]) & UC_WHITESPACE)) len++;
    return len;
}

/* The GPT-2 pre-tokenizer regex, matched by hand:
     's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
   A single leading space joins the run after it, so " the" is one word. */
static int split_words(const uint32_t *cpts, int n, int *lens, int max_words) {
    static const uint16_t run_kinds[] = {UC_LETTER, UC_NUMBER, 0};
    WordList words = {lens, 0, max_words, 0};

    for (int pos = 0; pos < n;) {
        int contraction = contraction_len(cpts, n, pos);
        if (contraction) {
            pos += contraction;
            close_word(&words, pos);
            continue;
        }

        int lead = cpts[pos] == ' ' && pos + 1 < n;
        uint16_t flags = cpt_flags(cpts[pos + lead]);
        int matched = 0;
        for (unsigned k = 0; k < sizeof run_kinds / sizeof *run_kinds; k++) {
            if (!in_run(flags, run_kinds[k])) continue;
            pos += lead;
            while (pos < n && in_run(cpt_flags(cpts[pos]), run_kinds[k])) pos++;
            close_word(&words, pos);
            matched = 1;
            break;
        }
        if (matched) continue;

        /* \s+(?!\S): a run of spaces gives its last one back to the word that
           follows, unless it ends the text. */
        int spaces = whitespace_len(cpts, n, pos);
        if (spaces > 1 && pos + spaces < n) spaces--;
        pos += spaces > 0 ? spaces : 1;
        close_word(&words, pos);
    }
    return words.count;
}

typedef struct {
    int prev, next;
    int start, len;
} Symbol;

typedef struct {
    int left, right;
    int rank;
    int left_start, left_len;
    int right_start, right_len;
} Bigram;

typedef struct {
    Bigram *entries;
    int count;
} PairHeap;

/* Lowest rank first, leftmost among equals, which is the order the model's
   merge list defines. */
static int outranks(const Bigram *a, const Bigram *b) {
    return a->rank != b->rank ? a->rank < b->rank : a->left < b->left;
}

static void heap_push(PairHeap *heap, Bigram pair) {
    int i = heap->count++;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (!outranks(&pair, &heap->entries[parent])) break;
        heap->entries[i] = heap->entries[parent];
        i = parent;
    }
    heap->entries[i] = pair;
}

static Bigram heap_pop(PairHeap *heap) {
    Bigram top = heap->entries[0];
    Bigram last = heap->entries[--heap->count];
    int i = 0;
    for (;;) {
        int left = 2 * i + 1, right = 2 * i + 2;
        int best = -1;
        if (left < heap->count) best = left;
        if (right < heap->count &&
            (best < 0 || outranks(&heap->entries[right], &heap->entries[left])))
            best = right;
        if (best < 0 || !outranks(&heap->entries[best], &last)) break;
        heap->entries[i] = heap->entries[best];
        i = best;
    }
    heap->entries[i] = last;
    return top;
}

/* A codepoint is at most 4 utf-8 bytes and each byte maps back to a codepoint
   under 0x800, so at most 2 bytes -- byte-encoding octuples a word at worst.
   Every buffer below is sized off that, so nothing here has a fixed cap a long
   word could run past. */
#define MAX_ENCODED_BYTES_PER_CPT 8

/* One symbol per encoded byte at most; the initial pass queues fewer pairs
   than there are symbols and each merge queues two more, so three per symbol
   bounds the heap. */
typedef struct {
    char *bytes;
    Symbol *symbols;
    Bigram *pairs;
    char *key;
} WordScratch;

static void scratch_free(WordScratch *scratch) {
    free(scratch->bytes);
    free(scratch->symbols);
    free(scratch->pairs);
    free(scratch->key);
}

static int scratch_alloc(WordScratch *scratch, int longest_word) {
    if (longest_word < 1) longest_word = 1;
    size_t max_bytes = (size_t)longest_word * MAX_ENCODED_BYTES_PER_CPT;

    memset(scratch, 0, sizeof *scratch);
    scratch->bytes = malloc(max_bytes);
    scratch->symbols = malloc(max_bytes * sizeof *scratch->symbols);
    scratch->pairs = malloc(3 * max_bytes * sizeof *scratch->pairs);
    scratch->key = malloc(2 * max_bytes + 1);
    if (scratch->bytes && scratch->symbols && scratch->pairs && scratch->key)
        return 1;
    scratch_free(scratch);
    return 0;
}

static int byte_encode(const uint32_t *cpts, int start, int len, char *out) {
    int n = 0;
    for (int i = 0; i < len; i++) {
        char utf8[4];
        int utf8_bytes = utf8_encode(cpts[start + i], utf8);
        for (int b = 0; b < utf8_bytes; b++)
            n += utf8_encode(cpt_of_byte[(uint8_t)utf8[b]], out + n);
    }
    return n;
}

typedef struct {
    int *ids;
    int count;
    int cap;
} TokenList;

static void append_token(TokenList *out, int id) {
    if (out->count < out->cap) out->ids[out->count++] = id;
}

static void queue_pair(PairHeap *heap, const Tokenizer *t, const char *word,
                       const Symbol *symbols, int left, int right, char *key) {
    if (left < 0 || right < 0) return;
    const Symbol *a = &symbols[left], *b = &symbols[right];
    if (a->len == 0 || b->len == 0) return;

    memcpy(key, word + a->start, (size_t)a->len);
    key[a->len] = ' ';
    memcpy(key + a->len + 1, word + b->start, (size_t)b->len);

    int rank;
    if (!map_get(t->merges, key, a->len + b->len + 1, &rank)) return;
    Bigram pair = {left, right, rank, a->start, a->len, b->start, b->len};
    heap_push(heap, pair);
}

/* A symbol the vocabulary does not have is emitted as its byte-encoded
   characters, which the vocabulary always has. */
static void emit_symbol(const Tokenizer *t, const char *word,
                        const Symbol *symbol, TokenList *out) {
    int id;
    if (map_get(t->vocab, word + symbol->start, symbol->len, &id)) {
        append_token(out, id);
        return;
    }
    for (int at = 0; at < symbol->len;) {
        int len = utf8_len((uint8_t)word[symbol->start + at]);
        if (map_get(t->vocab, word + symbol->start + at, len, &id))
            append_token(out, id);
        at += len;
    }
}

static void merge_word(const Tokenizer *t, WordScratch *scratch, int n_bytes,
                       TokenList *out) {
    const char *word = scratch->bytes;
    Symbol *symbols = scratch->symbols;
    int n_symbols = 0;

    for (int i = 0; i < n_bytes;) {
        int len = utf8_len((uint8_t)word[i]);
        symbols[n_symbols].prev = n_symbols - 1;
        symbols[n_symbols].next = n_symbols + 1;
        symbols[n_symbols].start = i;
        symbols[n_symbols].len = len;
        i += len;
        n_symbols++;
    }
    if (n_symbols == 0) return;
    symbols[n_symbols - 1].next = -1;

    PairHeap heap = {scratch->pairs, 0};
    for (int i = 1; i < n_symbols; i++)
        queue_pair(&heap, t, word, symbols, i - 1, i, scratch->key);

    while (heap.count > 0) {
        Bigram pair = heap_pop(&heap);
        Symbol *left = &symbols[pair.left], *right = &symbols[pair.right];
        /* Both symbols must still be what they were when the pair was queued;
           an earlier merge may have grown or emptied either. */
        if (left->start != pair.left_start || left->len != pair.left_len ||
            right->start != pair.right_start || right->len != pair.right_len)
            continue;

        left->len += right->len;
        right->len = 0;
        left->next = right->next;
        if (right->next >= 0) symbols[right->next].prev = pair.left;

        queue_pair(&heap, t, word, symbols, left->prev, pair.left,
                   scratch->key);
        queue_pair(&heap, t, word, symbols, pair.left, left->next,
                   scratch->key);
    }

    for (int i = 0; i >= 0; i = symbols[i].next)
        emit_symbol(t, word, &symbols[i], out);
}

int tokenizer_init(Tokenizer *t, const GgufFile *g, char *err, size_t errsz) {
    memset(t, 0, sizeof *t);
    init_byte_encoding();

    GgufArray tokens, merges;
    if (!gguf_meta_arr(g, "tokenizer.ggml.tokens", &tokens) ||
        tokens.elem_type != GGUF_STR) {
        snprintf(err, errsz,
                 "model has no tokenizer.ggml.tokens string array");
        return 0;
    }
    if (!gguf_meta_arr(g, "tokenizer.ggml.merges", &merges) ||
        merges.elem_type != GGUF_STR) {
        snprintf(err, errsz,
                 "model has no tokenizer.ggml.merges string array; this "
                 "tokenizer needs a byte-level BPE merge list");
        return 0;
    }

    t->n_tokens = (int)tokens.remaining;
    t->tokens = calloc((size_t)t->n_tokens, sizeof *t->tokens);
    t->vocab = map_create(t->n_tokens);
    t->merges = map_create((int)merges.remaining);
    if (!t->tokens || !t->vocab || !t->merges) {
        snprintf(err, errsz, "out of memory for a %d-token vocabulary",
                 t->n_tokens);
        tokenizer_free(t);
        return 0;
    }

    for (int id = 0; id < t->n_tokens; id++) {
        const char *text;
        size_t len;
        if (!gguf_array_next_str(&tokens, &text, &len)) {
            snprintf(err, errsz,
                     "tokenizer.ggml.tokens ends after %d of %d tokens", id,
                     t->n_tokens);
            tokenizer_free(t);
            return 0;
        }
        t->tokens[id].text = text;
        t->tokens[id].length = (int)len;
        map_put(t->vocab, text, (int)len, id);
    }

    for (int rank = 0; merges.remaining > 0; rank++) {
        const char *text;
        size_t len;
        if (!gguf_array_next_str(&merges, &text, &len)) {
            snprintf(err, errsz, "tokenizer.ggml.merges ends after %d merges",
                     rank);
            tokenizer_free(t);
            return 0;
        }
        map_put(t->merges, text, (int)len, rank);
    }

    unsigned long long value;
    t->bos_id = gguf_meta_u64(g, "tokenizer.ggml.bos_token_id", &value)
                    ? (int)value
                    : DEFAULT_BOS_ID;
    t->eos_id = gguf_meta_u64(g, "tokenizer.ggml.eos_token_id", &value)
                    ? (int)value
                    : DEFAULT_EOS_ID;
    t->add_bos =
        gguf_meta_u64(g, "tokenizer.ggml.add_bos_token", &value) ? (int)value : 0;
    return 1;
}

void tokenizer_free(Tokenizer *t) {
    free(t->tokens);
    map_free(t->vocab);
    map_free(t->merges);
    memset(t, 0, sizeof *t);
}

int tokenizer_encode(const Tokenizer *t, const char *text, int *ids,
                     int max_ids) {
    TokenList out = {ids, 0, max_ids};
    int n_bytes = (int)strlen(text);
    uint32_t *cpts = malloc(((size_t)n_bytes + 1) * sizeof *cpts);
    int *lens = malloc(((size_t)n_bytes + 1) * sizeof *lens);
    if (!cpts || !lens) {
        free(cpts);
        free(lens);
        return 0;
    }

    int n_cpts = 0;
    for (int at = 0; at < n_bytes;) {
        int len = utf8_len((uint8_t)text[at]);
        if (at + len > n_bytes) {
            cpts[n_cpts++] = REPLACEMENT_CPT;
            at++;
        } else {
            cpts[n_cpts++] = utf8_decode(text + at, &len);
            at += len;
        }
    }

    int n_words = split_words(cpts, n_cpts, lens, n_cpts + 1);
    int longest = 0;
    for (int w = 0; w < n_words; w++)
        if (lens[w] > longest) longest = lens[w];

    WordScratch scratch;
    if (scratch_alloc(&scratch, longest)) {
        int start = 0;
        for (int w = 0; w < n_words; w++) {
            int n_encoded = byte_encode(cpts, start, lens[w], scratch.bytes);
            merge_word(t, &scratch, n_encoded, &out);
            start += lens[w];
        }
        scratch_free(&scratch);
    }

    free(lens);
    free(cpts);
    return out.count;
}

int tokenizer_token_id(const Tokenizer *t, const char *text) {
    int id;
    return map_get(t->vocab, text, (int)strlen(text), &id) ? id : -1;
}

int tokenizer_encode_prompt(const Tokenizer *t, const char *text, int *ids,
                            int max_ids) {
    int n = 0;
    if (t->add_bos && max_ids > 0) ids[n++] = t->bos_id;
    return n + tokenizer_encode(t, text, ids + n, max_ids - n);
}

int tokenizer_decode(const Tokenizer *t, const int *ids, int n_ids, char *out,
                     int max_bytes) {
    int n = 0;
    for (int i = 0; i < n_ids && n < max_bytes - 1; i++) {
        if (ids[i] < 0 || ids[i] >= t->n_tokens) continue;
        const Token *token = &t->tokens[ids[i]];

        for (int at = 0; at < token->length && n < max_bytes - 1;) {
            int len = utf8_len((uint8_t)token->text[at]);
            if (at + len > token->length) break;
            uint32_t cpt = utf8_decode(token->text + at, &len);
            at += len;
            /* A token carrying anything but byte-encoded text -- a special
               token spelled in real unicode -- has no bytes to give back. */
            if (cpt < CPT_LIMIT && byte_of_cpt[cpt] != NO_BYTE)
                out[n++] = (char)byte_of_cpt[cpt];
        }
    }
    out[n] = '\0';
    return n;
}
