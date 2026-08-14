#include "tokenizer.h"
#include "unicode_ranges.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GPT-2 byte-level BPE, ported from llama.cpp (llm_tokenizer_bpe). */

#define UC_UNDEFINED 0x0001
#define UC_NUMBER    0x0002
#define UC_LETTER    0x0004
#define UC_WHITESPACE 0x0100

/* ---- unicode helpers ---- */

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
        if (uc_whitespace[i] == cpt) {
            flags |= UC_WHITESPACE;
            break;
        }
    return flags;
}

static int utf8_len(uint8_t b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

static uint32_t utf8_decode(const char *s, int *len) {
    uint8_t b = (uint8_t)s[0];
    if (b < 0x80) {
        *len = 1;
        return b;
    }
    if ((b & 0xE0) == 0xC0) {
        *len = 2;
        return ((uint32_t)(b & 0x1F) << 6) | ((uint8_t)s[1] & 0x3F);
    }
    if ((b & 0xF0) == 0xE0) {
        *len = 3;
        return ((uint32_t)(b & 0x0F) << 12) |
               (((uint8_t)s[1] & 0x3F) << 6) | ((uint8_t)s[2] & 0x3F);
    }
    *len = 4;
    return ((uint32_t)(b & 0x07) << 18) | (((uint8_t)s[1] & 0x3F) << 12) |
           (((uint8_t)s[2] & 0x3F) << 6) | ((uint8_t)s[3] & 0x3F);
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

/* byte -> unicode codepoint (GPT-2 byte encoding) */
static uint32_t byte_to_cpt[256];
static int byte_to_cpt_ready = 0;

static void init_byte_to_cpt(void) {
    if (byte_to_cpt_ready) return;
    int set[256] = {0};
    for (int ch = 0x21; ch <= 0x7E; ch++) {
        byte_to_cpt[ch] = (uint32_t)ch;
        set[ch] = 1;
    }
    for (int ch = 0xA1; ch <= 0xAC; ch++) {
        byte_to_cpt[ch] = (uint32_t)ch;
        set[ch] = 1;
    }
    for (int ch = 0xAE; ch <= 0xFF; ch++) {
        byte_to_cpt[ch] = (uint32_t)ch;
        set[ch] = 1;
    }
    int n = 0;
    for (int ch = 0; ch < 256; ch++)
        if (!set[ch]) byte_to_cpt[ch] = 256 + n++;
    byte_to_cpt_ready = 1;
}

/* ---- string hash map ---- */

typedef struct {
    char *key;
    int val;
} StrMapEntry;

struct StrMap {
    StrMapEntry *entries;
    int n, cap;
};
typedef struct StrMap StrMap;

static unsigned long str_hash(const char *s, int len) {
    unsigned long h = 1469598103934665603UL;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211UL;
    }
    return h;
}

static void strmap_init(StrMap *m, int cap) {
    m->cap = cap;
    m->n = 0;
    m->entries = calloc(cap, sizeof *m->entries);
}

static void strmap_free(StrMap *m) {
    for (int i = 0; i < m->cap; i++) free(m->entries[i].key);
    free(m->entries);
    m->entries = NULL;
    m->cap = m->n = 0;
}

static void strmap_put(StrMap *m, const char *key, int keylen, int val);

static void strmap_grow(StrMap *m) {
    StrMap old = *m;
    m->cap = old.cap * 2;
    m->n = 0;
    m->entries = calloc(m->cap, sizeof *m->entries);
    for (int i = 0; i < old.cap; i++)
        if (old.entries[i].key) strmap_put(m, old.entries[i].key, -1,
                                           old.entries[i].val);
    strmap_free(&old);
}

static void strmap_put(StrMap *m, const char *key, int keylen, int val) {
    if (m->n * 2 >= m->cap) strmap_grow(m);
    if (keylen < 0) keylen = (int)strlen(key);
    unsigned long h = str_hash(key, keylen) & (m->cap - 1);
    while (m->entries[h].key) {
        if ((int)strlen(m->entries[h].key) == keylen &&
            memcmp(m->entries[h].key, key, keylen) == 0) {
            m->entries[h].val = val;
            return;
        }
        h = (h + 1) & (m->cap - 1);
    }
    m->entries[h].key = malloc(keylen + 1);
    memcpy(m->entries[h].key, key, keylen);
    m->entries[h].key[keylen] = '\0';
    m->entries[h].val = val;
    m->n++;
}

static int strmap_get(const StrMap *m, const char *key, int keylen, int *val) {
    unsigned long h = str_hash(key, keylen) & (m->cap - 1);
    while (m->entries[h].key) {
        if ((int)strlen(m->entries[h].key) == keylen &&
            memcmp(m->entries[h].key, key, keylen) == 0) {
            *val = m->entries[h].val;
            return 1;
        }
        h = (h + 1) & (m->cap - 1);
    }
    return 0;
}

/* ---- GPT-2 regex pre-tokenization ---- */

/* Split text (as codepoints) into words per the GPT-2 regex. Returns the
   number of words; lens[i] = codepoint length of word i. */
static int gpt2_split(const uint32_t *cpts, int n, int *lens, int max_words) {
    int n_words = 0;
    int prev_end = 0;
    int pos = 0;
    while (pos < n) {
        uint32_t cpt = cpts[pos];
        uint16_t flags = cpt_flags(cpt);

        /* 's|'t|'re|'ve|'m|'ll|'d */
        if (cpt == '\'' && pos + 1 < n) {
            uint32_t next = cpts[pos + 1];
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                if (pos + 2 > prev_end && n_words < max_words)
                    lens[n_words++] = pos + 2 - prev_end;
                prev_end = pos + 2;
                pos += 2;
                continue;
            }
            if (pos + 2 < n) {
                uint32_t nn = cpts[pos + 2];
                if ((next == 'r' && nn == 'e') || (next == 'v' && nn == 'e') ||
                    (next == 'l' && nn == 'l')) {
                    if (pos + 3 > prev_end && n_words < max_words)
                        lens[n_words++] = pos + 3 - prev_end;
                    prev_end = pos + 3;
                    pos += 3;
                    continue;
                }
            }
        }

        uint16_t flags2 = (cpt == ' ' && pos + 1 < n) ? cpt_flags(cpts[pos + 1])
                                                       : flags;
        /* ?\p{L}+ */
        if (flags2 & UC_LETTER) {
            pos += (cpt == ' ');
            while (flags2 & UC_LETTER) {
                pos++;
                flags2 = pos < n ? cpt_flags(cpts[pos]) : 0;
            }
            if (pos > prev_end && n_words < max_words)
                lens[n_words++] = pos - prev_end;
            prev_end = pos;
            continue;
        }
        /* ?\p{N}+ */
        if (flags2 & UC_NUMBER) {
            pos += (cpt == ' ');
            while (flags2 & UC_NUMBER) {
                pos++;
                flags2 = pos < n ? cpt_flags(cpts[pos]) : 0;
            }
            if (pos > prev_end && n_words < max_words)
                lens[n_words++] = pos - prev_end;
            prev_end = pos;
            continue;
        }
        /* ?[^\s\p{L}\p{N}]+ */
        if (!(flags2 & (UC_WHITESPACE | UC_LETTER | UC_NUMBER)) && flags2) {
            pos += (cpt == ' ');
            while (!(flags2 & (UC_WHITESPACE | UC_LETTER | UC_NUMBER)) &&
                   flags2) {
                pos++;
                flags2 = pos < n ? cpt_flags(cpts[pos]) : 0;
            }
            if (pos > prev_end && n_words < max_words)
                lens[n_words++] = pos - prev_end;
            prev_end = pos;
            continue;
        }

        /* whitespace runs */
        int nws = 0;
        while (pos + nws < n && (cpt_flags(cpts[pos + nws]) & UC_WHITESPACE))
            nws++;
        /* \s+(?!\S) */
        if (nws > 1 && pos + nws < n) {
            pos += nws - 1;
            if (pos > prev_end && n_words < max_words)
                lens[n_words++] = pos - prev_end;
            prev_end = pos;
            continue;
        }
        /* \s+ */
        if (nws > 0) {
            pos += nws;
            if (pos > prev_end && n_words < max_words)
                lens[n_words++] = pos - prev_end;
            prev_end = pos;
            continue;
        }
        /* no match */
        pos++;
        if (pos > prev_end && n_words < max_words)
            lens[n_words++] = pos - prev_end;
        prev_end = pos;
    }
    return n_words;
}

/* ---- BPE merge ---- */

typedef struct {
    int prev, next;
    int start, len; /* into the byte-encoded word buffer */
} Symbol;

typedef struct {
    int left, right;
    int rank;
    int l_start, l_len, r_start, r_len;
} Bigram;

typedef struct {
    Bigram *d;
    int n, cap;
} Heap;

static void heap_push(Heap *h, Bigram b) {
    if (h->n == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 16;
        h->d = realloc(h->d, h->cap * sizeof *h->d);
    }
    int i = h->n++;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h->d[p].rank < b.rank ||
            (h->d[p].rank == b.rank && h->d[p].left <= b.left))
            break;
        h->d[i] = h->d[p];
        i = p;
    }
    h->d[i] = b;
}

static Bigram heap_pop(Heap *h) {
    Bigram top = h->d[0];
    Bigram last = h->d[--h->n];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = 2 * i + 2;
        int m = -1;
        if (l < h->n) m = l;
        if (r < h->n) {
            if (m < 0 || h->d[r].rank < h->d[l].rank ||
                (h->d[r].rank == h->d[l].rank && h->d[r].left < h->d[l].left))
                m = r;
        }
        if (m < 0) break;
        if (h->d[m].rank < last.rank ||
            (h->d[m].rank == last.rank && h->d[m].left < last.left)) {
            h->d[i] = h->d[m];
            i = m;
        } else {
            break;
        }
    }
    h->d[i] = last;
    return top;
}

/* Byte-encode one word (codepoints [start, start+len)) into buf. Returns
   byte length. */
static int byte_encode_word(const uint32_t *cpts, int start, int len,
                            char *buf) {
    int n = 0;
    for (int i = 0; i < len; i++) {
        char tmp[4];
        int tl = utf8_encode(cpts[start + i], tmp);
        for (int j = 0; j < tl; j++) {
            char enc[4];
            int el = utf8_encode(byte_to_cpt[(uint8_t)tmp[j]], enc);
            memcpy(buf + n, enc, el);
            n += el;
        }
    }
    return n;
}

/* BPE-merge a byte-encoded word; emit token ids into out. */
static int bpe_merge(const Tokenizer *t, const char *word, int wlen, int *out,
                     int max_out) {
    Symbol syms[512];
    int n_sym = 0;
    for (int i = 0; i < wlen;) {
        int l = utf8_len((uint8_t)word[i]);
        syms[n_sym].prev = n_sym - 1;
        syms[n_sym].next = n_sym + 1;
        syms[n_sym].start = i;
        syms[n_sym].len = l;
        i += l;
        n_sym++;
    }
    if (n_sym == 0) return 0;
    syms[0].prev = -1;
    syms[n_sym - 1].next = -1;

    Heap heap = {0};
    /* add initial bigrams */
    for (int i = 1; i < n_sym; i++) {
        Symbol *L = &syms[i - 1], *R = &syms[i];
        char key[1024];
        int kl = L->len + R->len + 1;
        if (kl >= (int)sizeof key) continue;
        memcpy(key, word + L->start, L->len);
        key[L->len] = ' ';
        memcpy(key + L->len + 1, word + R->start, R->len);
        int rank;
        if (strmap_get(t->merges, key, kl, &rank)) {
            Bigram b = {i - 1, i, rank, L->start, L->len, R->start, R->len};
            heap_push(&heap, b);
        }
    }

    while (heap.n > 0) {
        Bigram b = heap_pop(&heap);
        Symbol *L = &syms[b.left], *R = &syms[b.right];
        if (L->len == 0 || R->len == 0) continue;
        if (L->start != b.l_start || L->len != b.l_len ||
            R->start != b.r_start || R->len != b.r_len)
            continue;
        L->len += R->len;
        R->len = 0;
        L->next = R->next;
        if (R->next >= 0) syms[R->next].prev = b.left;
        /* add new bigrams around the merged symbol */
        for (int side = 0; side < 2; side++) {
            int left = side == 0 ? L->prev : b.left;
            int right = side == 0 ? b.left : L->next;
            if (left < 0 || right < 0) continue;
            Symbol *A = &syms[left], *B = &syms[right];
            if (A->len == 0 || B->len == 0) continue;
            char key[1024];
            int kl = A->len + B->len + 1;
            if (kl >= (int)sizeof key) continue;
            memcpy(key, word + A->start, A->len);
            key[A->len] = ' ';
            memcpy(key + A->len + 1, word + B->start, B->len);
            int rank;
            if (strmap_get(t->merges, key, kl, &rank)) {
                Bigram nb = {left, right, rank, A->start, A->len, B->start,
                             B->len};
                heap_push(&heap, nb);
            }
        }
    }
    free(heap.d);

    int n_out = 0;
    for (int i = 0; i != -1; i = syms[i].next) {
        if (syms[i].len == 0) continue;
        int id;
        if (strmap_get(t->vocab, word + syms[i].start, syms[i].len, &id)) {
            if (n_out < max_out) out[n_out++] = id;
} else {
            /* fallback: emit byte tokens for each byte-encoded char */
            for (int j = 0; j < syms[i].len;) {
                int l = utf8_len((uint8_t)word[syms[i].start + j]);
                int bid;
                if (strmap_get(t->vocab, word + syms[i].start + j, l, &bid) &&
                    n_out < max_out)
                    out[n_out++] = bid;
                j += l;
            }
        }
    }
    return n_out;
}

/* ---- public API ---- */

int tokenizer_init(Tokenizer *t, const GgufFile *g, char *err, size_t errsz) {
    memset(t, 0, sizeof *t);
    init_byte_to_cpt();

    GgufArray arr;
    if (!gguf_meta_arr(g, "tokenizer.ggml.tokens", &arr) ||
        arr.elem_type != 8) {
        snprintf(err, errsz, "no tokenizer.ggml.tokens array");
        return 0;
    }
    t->n_tokens = (int)arr.count;
    t->tokens = calloc(t->n_tokens, sizeof *t->tokens);
    t->token_types = calloc(t->n_tokens, sizeof *t->token_types);
    t->vocab = calloc(1, sizeof *t->vocab);
    t->merges = calloc(1, sizeof *t->merges);
    strmap_init(t->vocab, 1 << 18);
    strmap_init(t->merges, 1 << 18);

    /* tokens */
    const unsigned char *p = arr.data;
    const unsigned char *end = (const unsigned char *)g->map + g->map_size;
    for (int i = 0; i < t->n_tokens; i++) {
        unsigned long long len;
        if (p + 8 > end) goto fail;
        memcpy(&len, p, 8);
        p += 8;
        if (p + len > end) goto fail;
        char *s = malloc(len + 1);
        memcpy(s, p, len);
        s[len] = '\0';
        p += len;
        t->tokens[i] = s;
        strmap_put(t->vocab, s, (int)len, i);
    }

    /* token types */
    if (gguf_meta_arr(g, "tokenizer.ggml.token_type", &arr) &&
        arr.elem_type == 5) {
        p = arr.data;
        for (int i = 0; i < t->n_tokens && p + 4 <= end; i++) {
            int v;
            memcpy(&v, p, 4);
            p += 4;
            t->token_types[i] = v;
        }
    }

    /* merges */
    if (gguf_meta_arr(g, "tokenizer.ggml.merges", &arr) &&
        arr.elem_type == 8) {
        p = arr.data;
        for (unsigned long long i = 0; i < arr.count; i++) {
            unsigned long long len;
            if (p + 8 > end) break;
            memcpy(&len, p, 8);
            p += 8;
            if (p + len > end) break;
            strmap_put(t->merges, (const char *)p, (int)len, (int)i);
            p += len;
        }
    }

    unsigned long long u;
    t->bos_id = gguf_meta_u64(g, "tokenizer.ggml.bos_token_id", &u) ? (int)u : 1;
    t->eos_id = gguf_meta_u64(g, "tokenizer.ggml.eos_token_id", &u) ? (int)u : 2;
    t->add_bos = gguf_meta_u64(g, "tokenizer.ggml.add_bos_token", &u) ? (int)u : 0;
    return 1;

fail:
    snprintf(err, errsz, "tokenizer: bad tokens array");
    tokenizer_free(t);
    return 0;
}

void tokenizer_free(Tokenizer *t) {
    for (int i = 0; i < t->n_tokens; i++) free((void *)t->tokens[i]);
    free(t->tokens);
    free(t->token_types);
    if (t->vocab) {
        strmap_free(t->vocab);
        free(t->vocab);
    }
    if (t->merges) {
        strmap_free(t->merges);
        free(t->merges);
    }
    memset(t, 0, sizeof *t);
}

int tokenizer_encode(const Tokenizer *t, const char *text, int *out,
                     int max_out) {
    /* decode to codepoints */
    int n = (int)strlen(text);
    uint32_t *cpts = malloc((n + 1) * sizeof *cpts);
    int nc = 0;
    for (int i = 0; i < n;) {
        int l = utf8_len((uint8_t)text[i]);
        if (i + l > n) {
            cpts[nc++] = 0xFFFD;
            i++;
        } else {
            cpts[nc++] = utf8_decode(text + i, &l);
            i += l;
        }
    }

    int *lens = malloc((nc + 1) * sizeof *lens);
    int n_words = gpt2_split(cpts, nc, lens, nc + 1);

    int n_out = 0;
    int start = 0;
    for (int w = 0; w < n_words; w++) {
        char buf[4096];
        int wlen = byte_encode_word(cpts, start, lens[w], buf);
        n_out += bpe_merge(t, buf, wlen, out + n_out, max_out - n_out);
        start += lens[w];
    }
    free(lens);
    free(cpts);
    return n_out;
}

int tokenizer_decode(const Tokenizer *t, const int *ids, int n_ids, char *out,
                     int max_out) {
    int n = 0;
    for (int i = 0; i < n_ids && n < max_out - 1; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->n_tokens) continue;
        const char *s = t->tokens[id];
        int len = (int)strlen(s);
        /* byte-encoded: decode each unicode char back to a byte */
        for (int j = 0; j < len && n < max_out - 1;) {
            int l = utf8_len((uint8_t)s[j]);
            uint32_t cpt = utf8_decode(s + j, &l);
            /* inverse byte_to_cpt */
            uint8_t b = 0;
            for (int k = 0; k < 256; k++)
                if (byte_to_cpt[k] == cpt) {
                    b = (uint8_t)k;
                    break;
                }
            out[n++] = (char)b;
            j += l;
        }
    }
    out[n] = '\0';
    return n;
}
