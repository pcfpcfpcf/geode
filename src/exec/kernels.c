#include "kernels.h"
#include "quant.h"

#include <math.h>
#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define HAVE_AVX2 1
#include <immintrin.h>
#endif

/* Block layouts and dequant math follow ggml (llama.cpp) exactly. */

typedef struct {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[16];
} block_q5_0;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
} block_q6_K;

/* Activations quantized per 256, mirroring the widest weight block size.
   group_sums holds the sum of each 16 quants: the types that store a constant
   offset per sub-block cancel it against those sums instead of touching the
   quants again. */
typedef struct {
    float scale;
    int8_t qs[QK_K];
    int16_t group_sums[QK_K / 16];
} ActivationBlock;

/* A weight row is quantized in blocks of either 32 (the legacy types) or 256
   (the K-quants), so one activation block spans one K-quant block or eight
   legacy ones. */
#define QUANT_GRANULE 32

/* Fills out[row_begin..row_end) with each weight row dotted against an
   int8-quantized activation. n is elements, not blocks, so each kernel counts
   its own blocks and no caller has to know the type's geometry. The kernel
   owns the row loop because a type whose blocks are small needs several rows
   in flight to fill the fma pipeline and to share the activation loads. */
typedef void (*GemvQ8)(float *out, const void *rows, size_t stride,
                       int row_begin, int row_end, int n,
                       const ActivationBlock *x);

/* One row against one activation vector. The batched path owns its own row and
   vector loops and needs only this much per type. */
typedef float (*RowDotQ8)(const void *row, int n, const ActivationBlock *x);

/* One row against GEMM_TILE activation vectors at once, writing vector t to
   out[t * out_stride]. Unpacking a weight block is about half of what a row
   dot costs, and a tile pays it once for all of its vectors instead of once
   each -- which is the difference between reusing a row from cache and reusing
   the work of decoding it.

   Four, because the tile holds an int32 accumulator and a float accumulator
   per vector alongside the unpacked nibbles and their scales, and eight would
   spill those to the stack and give back more than the sharing wins. */
#define GEMM_TILE 4

typedef void (*RowTileQ8)(float *out, size_t out_stride, const void *row, int n,
                          const ActivationBlock *const *x);

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) {
                man <<= 1;
                exp--;
            }
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d,
                                    uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

void dequant_row(const void *data, unsigned type, int n, float *dst) {
    switch (type) {
    case GGML_TYPE_F32:
        memcpy(dst, data, (size_t)n * 4);
        break;
    case GGML_TYPE_F16: {
        const uint16_t *w = data;
        for (int i = 0; i < n; i++) dst[i] = fp16_to_fp32(w[i]);
        break;
    }
    case GGML_TYPE_Q5_0: {
        const block_q5_0 *b = data;
        int nb = n / 32;
        for (int i = 0; i < nb; i++) {
            float d = fp16_to_fp32(b[i].d);
            uint32_t qh;
            memcpy(&qh, b[i].qh, 4);
            for (int j = 0; j < 16; j++) {
                uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
                int32_t x0 = ((b[i].qs[j] & 0x0F) | xh_0) - 16;
                int32_t x1 = ((b[i].qs[j] >> 4) | xh_1) - 16;
                dst[i * 32 + j] = x0 * d;
                dst[i * 32 + j + 16] = x1 * d;
            }
        }
        break;
    }
    case GGML_TYPE_Q4_K: {
        const block_q4_K *b = data;
        int nb = n / 256;
        for (int i = 0; i < nb; i++) {
            const uint8_t *q = b[i].qs;
            float d = fp16_to_fp32(b[i].d);
            float min = fp16_to_fp32(b[i].dmin);
            int is = 0;
            uint8_t sc, m;
            for (int j = 0; j < 256; j += 64) {
                get_scale_min_k4(is + 0, b[i].scales, &sc, &m);
                float d1 = d * sc;
                float m1 = min * m;
                get_scale_min_k4(is + 1, b[i].scales, &sc, &m);
                float d2 = d * sc;
                float m2 = min * m;
                for (int l = 0; l < 32; l++) {
                    dst[i * 256 + j + l] = d1 * (q[l] & 0xF) - m1;
                    dst[i * 256 + j + 32 + l] = d2 * (q[l] >> 4) - m2;
                }
                q += 32;
                is += 2;
            }
        }
        break;
    }
    case GGML_TYPE_Q6_K: {
        const block_q6_K *b = data;
        int nb = n / 256;
        for (int i = 0; i < nb; i++) {
            float d = fp16_to_fp32(b[i].d);
            const uint8_t *ql = b[i].ql;
            const uint8_t *qh = b[i].qh;
            const int8_t *sc = b[i].scales;
            for (int n2 = 0; n2 < 256; n2 += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int8_t q1 =
                        (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 =
                        (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 =
                        (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 =
                        (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    dst[i * 256 + n2 + l] = d * sc[is + 0] * q1;
                    dst[i * 256 + n2 + l + 32] = d * sc[is + 2] * q2;
                    dst[i * 256 + n2 + l + 64] = d * sc[is + 4] * q3;
                    dst[i * 256 + n2 + l + 96] = d * sc[is + 6] * q4;
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        break;
    }
    default:
        break;
    }
}

/* The scalar paths below are the reference; every vector path must agree with
   them, which is what exec-kernels checks. A scalar dot chains one add per
   element, so it retires at fp-add latency (~1.4 GFLOP/s here) no matter how
   much memory bandwidth is spare -- the vector paths exist to break that
   dependency, not to save instructions. */
#ifdef HAVE_AVX2
static int have_avx2(void) {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("f16c");
}

/* The portable fp16_to_fp32 branches and loops over subnormals. Once per 256
   weights that is noise; once per 32, as the legacy block types need, it costs
   more than the arithmetic it feeds. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline float
half_to_float(uint16_t half) {
    return _cvtsh_ss(half);
}

__attribute__((target("avx2,fma,f16c"), always_inline)) static inline float
hsum256(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

/* Weights stay unsigned nibbles and activations are signed int8, which is
   exactly the operand pairing VPMADDUBSW wants: 32 multiply-accumulates per
   instruction against 8 for float FMA. Q4_K's per-sub-block minimum is a
   constant offset, so it folds into a sum of activations taken from the
   precomputed group sums instead of costing any multiplies. */
/* The eight 6-bit scales and eight 6-bit mins of a Q4_K block, unpacked in one
   go. Decoding them one at a time inside the loop, as the scalar reference
   does, costs more than the arithmetic they feed. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline void
q4k_scales_mins(const uint8_t *packed, __m128i *scales, __m128i *mins) {
    const uint32_t low6 = 0x3f3f3f3fu, low4 = 0x0f0f0f0fu, top2 = 0x03030303u;
    uint32_t u[4];
    memcpy(u, packed, 12);
    u[3] = ((u[2] >> 4) & low4) | (((u[1] >> 6) & top2) << 4);
    uint32_t mins_low = u[1] & low6;
    u[1] = (u[2] & low4) | (((u[0] >> 6) & top2) << 4);
    u[2] = mins_low;
    u[0] &= low6;

    __m128i both = _mm_set_epi32((int)u[3], (int)u[2], (int)u[1], (int)u[0]);
    *scales = _mm_cvtepu8_epi16(both);
    *mins = _mm_cvtepu8_epi16(_mm_srli_si128(both, 8));
}

/* Nibbles and scales are unpacked from the weight block and then folded into
   one activation vector at a time, because a chunk of tokens shares them: the
   unpacking is the half of the work that tiling exists to pay only once.
   `low` holds a group's first 32 weights and `high` its second 32, which is
   the order the activation quants already sit in. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline void
q4k_group(__m256i *sumi, __m256i low, __m256i high, __m256i scale_low,
          __m256i scale_high, const int8_t *qs) {
    *sumi = _mm256_add_epi32(
        *sumi,
        _mm256_madd_epi16(
            _mm256_maddubs_epi16(low, _mm256_loadu_si256((const __m256i *)qs)),
            scale_low));
    *sumi = _mm256_add_epi32(
        *sumi,
        _mm256_madd_epi16(_mm256_maddubs_epi16(
                              high, _mm256_loadu_si256(
                                        (const __m256i *)(qs + 32))),
                          scale_high));
}

/* Folds a finished block into one vector's running totals. Each sub-block's
   minimum is constant across its 32 activations, so all eight collapse into
   one dot product against the group sums rather than an accumulation inside
   the group loop. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline void
q4k_fold(__m256 *acc, __m128 *offset, __m256i sumi, const block_q4_K *b,
         __m128i mins, const ActivationBlock *a) {
    *acc = _mm256_fmadd_ps(_mm256_set1_ps(half_to_float(b->d) * a->scale),
                           _mm256_cvtepi32_ps(sumi), *acc);
    __m256i sums = _mm256_loadu_si256((const __m256i *)a->group_sums);
    __m128i pairs = _mm_hadd_epi16(_mm256_castsi256_si128(sums),
                                   _mm256_extracti128_si256(sums, 1));
    *offset = _mm_fmadd_ps(_mm_set1_ps(half_to_float(b->dmin) * a->scale),
                           _mm_cvtepi32_ps(_mm_madd_epi16(mins, pairs)),
                           *offset);
}

__attribute__((target("avx2,fma,f16c"), always_inline)) static inline float
q4k_total(__m256 acc, __m128 offset) {
    __m128 total = _mm_sub_ps(_mm_add_ps(_mm256_castps256_ps128(acc),
                                         _mm256_extractf128_ps(acc, 1)),
                              offset);
    total = _mm_add_ps(total, _mm_movehl_ps(total, total));
    total = _mm_add_ss(total, _mm_movehdup_ps(total));
    return _mm_cvtss_f32(total);
}

/* Unpacks a block's group `is` into the nibbles and scales its activations
   multiply against. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline void
q4k_unpack(__m256i *low, __m256i *high, __m256i *scale_low,
           __m256i *scale_high, const uint8_t *q, const int16_t *scale_of,
           int is) {
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i packed = _mm256_loadu_si256((const __m256i *)q);
    *low = _mm256_and_si256(packed, nibble_mask);
    *high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask);
    *scale_low = _mm256_set1_epi16(scale_of[is + 0]);
    *scale_high = _mm256_set1_epi16(scale_of[is + 1]);
}

__attribute__((target("avx2,fma,f16c"), always_inline)) static inline float
q4k_row(const void *row, int n, const ActivationBlock *activations) {
    const block_q4_K *blocks = row;
    __m256 acc = _mm256_setzero_ps();
    __m128 offset = _mm_setzero_ps();

    for (int i = 0; i < n / QK_K; i++) {
        const block_q4_K *b = &blocks[i];
        const ActivationBlock *a = &activations[i];
        __m128i scales, mins;
        q4k_scales_mins(b->scales, &scales, &mins);
        int16_t scale_of[8];
        _mm_storeu_si128((__m128i *)scale_of, scales);

        __m256i sumi = _mm256_setzero_si256();
        const uint8_t *q = b->qs;
        for (int j = 0, is = 0; j < QK_K; j += 64, q += 32, is += 2) {
            __m256i low, high, scale_low, scale_high;
            q4k_unpack(&low, &high, &scale_low, &scale_high, q, scale_of, is);
            q4k_group(&sumi, low, high, scale_low, scale_high, a->qs + j);
        }
        q4k_fold(&acc, &offset, sumi, b, mins, a);
    }
    return q4k_total(acc, offset);
}

__attribute__((target("avx2,fma,f16c"))) static void
gemv_q4k_q8_avx2(float *out, const void *rows, size_t stride, int row_begin,
                 int row_end, int n, const ActivationBlock *x) {
    const unsigned char *row =
        (const unsigned char *)rows + (size_t)row_begin * stride;
    for (int r = row_begin; r < row_end; r++, row += stride)
        out[r] = q4k_row(row, n, x);
}

__attribute__((target("avx2,fma,f16c"))) static void
row_tile_q4k(float *out, size_t out_stride, const void *row, int n,
             const ActivationBlock *const *x) {
    const block_q4_K *blocks = row;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    __m128 off0 = _mm_setzero_ps(), off1 = off0, off2 = off0, off3 = off0;

    for (int i = 0; i < n / QK_K; i++) {
        const block_q4_K *b = &blocks[i];
        __m128i scales, mins;
        q4k_scales_mins(b->scales, &scales, &mins);
        int16_t scale_of[8];
        _mm_storeu_si128((__m128i *)scale_of, scales);

        __m256i sumi0 = _mm256_setzero_si256(), sumi1 = sumi0, sumi2 = sumi0,
                sumi3 = sumi0;
        const uint8_t *q = b->qs;
        for (int j = 0, is = 0; j < QK_K; j += 64, q += 32, is += 2) {
            __m256i low, high, scale_low, scale_high;
            q4k_unpack(&low, &high, &scale_low, &scale_high, q, scale_of, is);
            q4k_group(&sumi0, low, high, scale_low, scale_high, x[0][i].qs + j);
            q4k_group(&sumi1, low, high, scale_low, scale_high, x[1][i].qs + j);
            q4k_group(&sumi2, low, high, scale_low, scale_high, x[2][i].qs + j);
            q4k_group(&sumi3, low, high, scale_low, scale_high, x[3][i].qs + j);
        }
        q4k_fold(&acc0, &off0, sumi0, b, mins, &x[0][i]);
        q4k_fold(&acc1, &off1, sumi1, b, mins, &x[1][i]);
        q4k_fold(&acc2, &off2, sumi2, b, mins, &x[2][i]);
        q4k_fold(&acc3, &off3, sumi3, b, mins, &x[3][i]);
    }
    out[0] = q4k_total(acc0, off0);
    out[out_stride] = q4k_total(acc1, off1);
    out[2 * out_stride] = q4k_total(acc2, off2);
    out[3 * out_stride] = q4k_total(acc3, off3);
}

/* Q6_K keeps four low bits and two high bits per weight, in four interleaved
   32-weight parts per 128. Reassembling them costs more than Q4_K's single
   mask, which makes it the type a tile has most to share. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline void
q6k_unpack(__m256i *quants, const uint8_t *ql, const uint8_t *qh) {
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
    const __m256i pair_mask = _mm256_set1_epi8(3);
    __m256i low = _mm256_loadu_si256((const __m256i *)ql);
    __m256i low32 = _mm256_loadu_si256((const __m256i *)(ql + 32));
    __m256i high = _mm256_loadu_si256((const __m256i *)qh);

    quants[0] = _mm256_or_si256(
        _mm256_and_si256(low, nibble_mask),
        _mm256_slli_epi16(_mm256_and_si256(high, pair_mask), 4));
    quants[1] = _mm256_or_si256(
        _mm256_and_si256(low32, nibble_mask),
        _mm256_slli_epi16(
            _mm256_and_si256(_mm256_srli_epi16(high, 2), pair_mask), 4));
    quants[2] = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(low, 4), nibble_mask),
        _mm256_slli_epi16(
            _mm256_and_si256(_mm256_srli_epi16(high, 4), pair_mask), 4));
    quants[3] = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(low32, 4), nibble_mask),
        _mm256_slli_epi16(
            _mm256_and_si256(_mm256_srli_epi16(high, 6), pair_mask), 4));
}

/* Lanes 0-7 cover the first 16 weights of a part and lanes 8-15 the next, and
   those two halves carry different scales. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline __m256i
q6k_scales(const int8_t *sc, int part) {
    return _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_set1_epi16(sc[2 * part + 0])),
        _mm_set1_epi16(sc[2 * part + 1]), 1);
}

/* Q6_K is symmetric around 32 with no stored minimum, so the -32 is recovered
   by multiplying the activations against a constant 32 and subtracting. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline void
q6k_part(__m256i *sumi, __m256i quants, __m256i scales, const int8_t *qs) {
    __m256i x = _mm256_loadu_si256((const __m256i *)qs);
    __m256i product =
        _mm256_sub_epi16(_mm256_maddubs_epi16(quants, x),
                         _mm256_maddubs_epi16(_mm256_set1_epi8(32), x));
    *sumi = _mm256_add_epi32(*sumi, _mm256_madd_epi16(scales, product));
}

__attribute__((target("avx2,fma,f16c"), always_inline)) static inline void
q6k_fold(__m256 *acc, __m256i sumi, const block_q6_K *b,
         const ActivationBlock *a) {
    *acc = _mm256_fmadd_ps(_mm256_set1_ps(half_to_float(b->d) * a->scale),
                           _mm256_cvtepi32_ps(sumi), *acc);
}

__attribute__((target("avx2,fma,f16c"), always_inline)) static inline float
q6k_row(const void *row, int n, const ActivationBlock *activations) {
    const block_q6_K *blocks = row;
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < n / QK_K; i++) {
        const block_q6_K *b = &blocks[i];
        const ActivationBlock *a = &activations[i];
        const uint8_t *ql = b->ql;
        const uint8_t *qh = b->qh;
        const int8_t *sc = b->scales;
        __m256i sumi = _mm256_setzero_si256();

        for (int n2 = 0; n2 < QK_K; n2 += 128, ql += 64, qh += 32, sc += 8) {
            __m256i quants[4];
            q6k_unpack(quants, ql, qh);
            for (int part = 0; part < 4; part++)
                q6k_part(&sumi, quants[part], q6k_scales(sc, part),
                         a->qs + n2 + 32 * part);
        }
        q6k_fold(&acc, sumi, b, a);
    }
    return hsum256(acc);
}

__attribute__((target("avx2,fma,f16c"))) static void
gemv_q6k_q8_avx2(float *out, const void *rows, size_t stride, int row_begin,
                 int row_end, int n, const ActivationBlock *x) {
    const unsigned char *row =
        (const unsigned char *)rows + (size_t)row_begin * stride;
    for (int r = row_begin; r < row_end; r++, row += stride)
        out[r] = q6k_row(row, n, x);
}

__attribute__((target("avx2,fma,f16c"))) static void
row_tile_q6k(float *out, size_t out_stride, const void *row, int n,
             const ActivationBlock *const *x) {
    const block_q6_K *blocks = row;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;

    for (int i = 0; i < n / QK_K; i++) {
        const block_q6_K *b = &blocks[i];
        const uint8_t *ql = b->ql;
        const uint8_t *qh = b->qh;
        const int8_t *sc = b->scales;
        __m256i sumi0 = _mm256_setzero_si256(), sumi1 = sumi0, sumi2 = sumi0,
                sumi3 = sumi0;

        for (int n2 = 0; n2 < QK_K; n2 += 128, ql += 64, qh += 32, sc += 8) {
            __m256i quants[4];
            q6k_unpack(quants, ql, qh);
            for (int part = 0; part < 4; part++) {
                __m256i scales = q6k_scales(sc, part);
                int at = n2 + 32 * part;
                q6k_part(&sumi0, quants[part], scales, x[0][i].qs + at);
                q6k_part(&sumi1, quants[part], scales, x[1][i].qs + at);
                q6k_part(&sumi2, quants[part], scales, x[2][i].qs + at);
                q6k_part(&sumi3, quants[part], scales, x[3][i].qs + at);
            }
        }
        q6k_fold(&acc0, sumi0, b, &x[0][i]);
        q6k_fold(&acc1, sumi1, b, &x[1][i]);
        q6k_fold(&acc2, sumi2, b, &x[2][i]);
        q6k_fold(&acc3, sumi3, b, &x[3][i]);
    }
    out[0] = hsum256(acc0);
    out[out_stride] = hsum256(acc1);
    out[2 * out_stride] = hsum256(acc2);
    out[3 * out_stride] = hsum256(acc3);
}

/* Expands the low 32 bits at `bits` to one byte per bit, 0xFF where set. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline __m256i
bytes_from_bits_32(const uint8_t *bits) {
    uint32_t packed;
    memcpy(&packed, bits, 4);
    const __m256i byte_of_bit =
        _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202,
                          0x0101010101010101, 0x0000000000000000);
    const __m256i keep_one_bit = _mm256_set1_epi64x((int64_t)0x7fbfdfeff7fbfdfeULL);
    __m256i spread =
        _mm256_shuffle_epi8(_mm256_set1_epi32((int)packed), byte_of_bit);
    return _mm256_cmpeq_epi8(_mm256_or_si256(spread, keep_one_bit),
                             _mm256_set1_epi64x(-1));
}

/* Q5_0 keeps a fifth bit for each quant in a 32-bit mask, and is symmetric
   around 16 with no stored minimum -- so like Q6_K the offset comes off the
   activation sums rather than the quants. Its blocks are 32 wide, so eight of
   them share one activation block and pick up two group sums each. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline __m256i
q50_quants(const block_q5_0 *b) {
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);
    /* Low nibbles are elements 0-15 and high nibbles elements 16-31, which is
       the order the activation quants already sit in. */
    __m128i packed = _mm_loadu_si128((const __m128i *)b->qs);
    __m256i nibbles = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_and_si128(packed, nibble_mask)),
        _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask), 1);
    return _mm256_or_si256(
        nibbles,
        _mm256_and_si256(bytes_from_bits_32(b->qh), _mm256_set1_epi8(16)));
}

/* Rows this type carries are short -- 128 elements is four blocks -- so a
   single row leaves the fma pipeline mostly empty waiting on its own
   accumulator, and pays a horizontal sum for four blocks of work. Four rows at
   a time run four independent chains and read each activation block once
   between them. */
#define Q50_ROWS 4

/* One block of one row, folded into that row's running products and its
   running offset. The accumulators are passed by pointer so that the row loop
   below can hold them in named locals: an array of them is what gcc spills. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline void
q50_block(__m256 *acc, float *offset, const block_q5_0 *b, __m256i xq,
          float activation_scale, float group_sums) {
    __m256i sumi = _mm256_madd_epi16(
        _mm256_maddubs_epi16(q50_quants(b), xq), _mm256_set1_epi16(1));
    float scale = half_to_float(b->d) * activation_scale;
    *acc = _mm256_fmadd_ps(_mm256_set1_ps(scale), _mm256_cvtepi32_ps(sumi),
                           *acc);
    *offset -= scale * 16.0f * group_sums;
}

__attribute__((target("avx2,fma,f16c"), always_inline)) static inline float
q50_row(const void *row, int n, const ActivationBlock *activations) {
    const block_q5_0 *blocks = row;
    __m256 acc = _mm256_setzero_ps();
    float offset = 0;
    for (int i = 0; i < n / QUANT_GRANULE; i++) {
        int sub = i % (QK_K / QUANT_GRANULE);
        const ActivationBlock *a = &activations[i / (QK_K / QUANT_GRANULE)];
        q50_block(&acc, &offset, &blocks[i],
                  _mm256_loadu_si256(
                      (const __m256i *)(a->qs + QUANT_GRANULE * sub)),
                  a->scale,
                  (float)(a->group_sums[2 * sub] + a->group_sums[2 * sub + 1]));
    }
    return hsum256(acc) + offset;
}

__attribute__((target("avx2,fma,f16c"))) static void
gemv_q50_q8_avx2(float *out, const void *rows, size_t stride, int row_begin,
                 int row_end, int n, const ActivationBlock *x) {
    const unsigned char *base =
        (const unsigned char *)rows + (size_t)row_begin * stride;
    int blocks = n / QUANT_GRANULE;
    int r = row_begin;

    for (; r + Q50_ROWS <= row_end; r += Q50_ROWS, base += Q50_ROWS * stride) {
        const block_q5_0 *row0 = (const block_q5_0 *)base;
        const block_q5_0 *row1 = (const block_q5_0 *)(base + stride);
        const block_q5_0 *row2 = (const block_q5_0 *)(base + 2 * stride);
        const block_q5_0 *row3 = (const block_q5_0 *)(base + 3 * stride);
        __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0,
               acc3 = acc0;
        float off0 = 0, off1 = 0, off2 = 0, off3 = 0;

        for (int i = 0; i < blocks; i++) {
            int sub = i % (QK_K / QUANT_GRANULE);
            const ActivationBlock *a = &x[i / (QK_K / QUANT_GRANULE)];
            __m256i xq = _mm256_loadu_si256(
                (const __m256i *)(a->qs + QUANT_GRANULE * sub));
            float sums = (float)(a->group_sums[2 * sub] +
                                 a->group_sums[2 * sub + 1]);

            q50_block(&acc0, &off0, &row0[i], xq, a->scale, sums);
            q50_block(&acc1, &off1, &row1[i], xq, a->scale, sums);
            q50_block(&acc2, &off2, &row2[i], xq, a->scale, sums);
            q50_block(&acc3, &off3, &row3[i], xq, a->scale, sums);
        }
        out[r + 0] = hsum256(acc0) + off0;
        out[r + 1] = hsum256(acc1) + off1;
        out[r + 2] = hsum256(acc2) + off2;
        out[r + 3] = hsum256(acc3) + off3;
    }

    for (; r < row_end; r++, base += stride) out[r] = q50_row(base, n, x);
}

__attribute__((target("avx2,fma,f16c"))) static void
expand_fp16_avx2(float *dst, const uint16_t *values, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(
            dst + i,
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(values + i))));
    for (; i < n; i++) dst[i] = fp16_to_fp32(values[i]);
}

/* Four accumulators rather than one: a single chain retires at fma latency and
   leaves three quarters of the pipeline idle on vectors this short. They are
   named rather than held in an array because gcc declines to unroll the loop
   an array would need and spills all four to the stack, which turns every
   fma into a load-modify-store and costs more than the extra chains win. */
__attribute__((target("avx2,fma,f16c"))) static float
dot_f32_avx2(const float *a, const float *b, int n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 0),
                               _mm256_loadu_ps(b + i + 0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                               _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
                               _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
                               _mm256_loadu_ps(b + i + 24), acc3);
    }
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i),
                               acc0);
    float sum = hsum256(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                      _mm256_add_ps(acc2, acc3)));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* Two accumulators per row rather than four: eight chains already cover the fma
   latency, and the rows themselves supply the independence a single dot has to
   find by splitting its own. */
__attribute__((target("avx2,fma,f16c"))) static void
dot_f32_rows_avx2(float *out, const float *rows, size_t stride, const float *b,
                  int n) {
    const float *r0 = rows, *r1 = rows + stride, *r2 = rows + 2 * stride,
                *r3 = rows + 3 * stride;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    __m256 c0 = a0, c1 = a0, c2 = a0, c3 = a0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 x = _mm256_loadu_ps(b + i);
        __m256 y = _mm256_loadu_ps(b + i + 8);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i), x, a0);
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 8), y, c0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i), x, a1);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i + 8), y, c1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i), x, a2);
        c2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 8), y, c2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i), x, a3);
        c3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i + 8), y, c3);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(b + i);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i), x, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i), x, a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i), x, a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i), x, a3);
    }
    out[0] = hsum256(_mm256_add_ps(a0, c0));
    out[1] = hsum256(_mm256_add_ps(a1, c1));
    out[2] = hsum256(_mm256_add_ps(a2, c2));
    out[3] = hsum256(_mm256_add_ps(a3, c3));
    for (; i < n; i++)
        for (int r = 0; r < CACHE_ROWS; r++)
            out[r] += rows[(size_t)r * stride + i] * b[i];
}

/* The four rows fold into two sums that meet at the store, so no chain is
   longer than two: summing them one after another into dst would serialize all
   four behind each other. */
__attribute__((target("avx2,fma,f16c"))) static void
add_scaled_rows_avx2(float *dst, const float *rows, size_t stride,
                     const float *scales, int n) {
    const float *r0 = rows, *r1 = rows + stride, *r2 = rows + 2 * stride,
                *r3 = rows + 3 * stride;
    __m256 s0 = _mm256_set1_ps(scales[0]), s1 = _mm256_set1_ps(scales[1]),
           s2 = _mm256_set1_ps(scales[2]), s3 = _mm256_set1_ps(scales[3]);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 x = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i), s0,
                                   _mm256_loadu_ps(dst + i));
        __m256 y = _mm256_mul_ps(_mm256_loadu_ps(r1 + i), s1);
        __m256 z = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i + 8), s0,
                                   _mm256_loadu_ps(dst + i + 8));
        __m256 w = _mm256_mul_ps(_mm256_loadu_ps(r1 + i + 8), s1);
        x = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i), s2, x);
        y = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i), s3, y);
        z = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i + 8), s2, z);
        w = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i + 8), s3, w);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(x, y));
        _mm256_storeu_ps(dst + i + 8, _mm256_add_ps(z, w));
    }
    for (; i < n; i++)
        for (int r = 0; r < CACHE_ROWS; r++)
            dst[i] += rows[(size_t)r * stride + i] * scales[r];
}

__attribute__((target("avx2,fma,f16c"))) static void
add_scaled_avx2(float *dst, const float *src, float scale, int n) {
    __m256 w = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i), w,
                                         _mm256_loadu_ps(dst + i)));
    for (; i < n; i++) dst[i] += src[i] * scale;
}

/* Horizontal sum of eight int32 lanes. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline int
hsum256i(__m256i v) {
    __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(v),
                               _mm256_extracti128_si256(v, 1));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
    return _mm_cvtsi128_si32(lo);
}

/* Quantizes one block and returns its scale, or 0 when every value was zero
   and the caller should leave the zeroed block alone. The signed extreme is
   the wider of the block's maximum and its negated minimum, which is the
   largest magnitude with the sign of the end that saturates. */
__attribute__((target("avx2,fma,f16c"))) static float
quantize_block_avx2(ActivationBlock *block, const float *x, int count) {
    __m256 hi = _mm256_loadu_ps(x), lo = hi;
    for (int i = 8; i < count; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        hi = _mm256_max_ps(hi, v);
        lo = _mm256_min_ps(lo, v);
    }
    __m128 h = _mm_max_ps(_mm256_castps256_ps128(hi),
                          _mm256_extractf128_ps(hi, 1));
    h = _mm_max_ps(h, _mm_movehl_ps(h, h));
    h = _mm_max_ss(h, _mm_movehdup_ps(h));
    __m128 l = _mm_min_ps(_mm256_castps256_ps128(lo),
                          _mm256_extractf128_ps(lo, 1));
    l = _mm_min_ps(l, _mm_movehl_ps(l, l));
    l = _mm_min_ss(l, _mm_movehdup_ps(l));

    float highest = _mm_cvtss_f32(h), lowest = _mm_cvtss_f32(l);
    float extreme = highest >= -lowest ? highest : lowest;
    if (extreme == 0) return 0;

    float inverse_scale = -127.0f / extreme;
    __m256 scale = _mm256_set1_ps(inverse_scale);
    const __m256i lane_order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (int j = 0; j < count; j += QUANT_GRANULE) {
        __m256i q[4];
        for (int k = 0; k < 4; k++)
            q[k] = _mm256_cvtps_epi32(
                _mm256_mul_ps(_mm256_loadu_ps(x + j + 8 * k), scale));

        /* Summed as int32 before the pack, which is where the group sums are
           cheapest to take. */
        block->group_sums[j / 16 + 0] =
            (int16_t)hsum256i(_mm256_add_epi32(q[0], q[1]));
        block->group_sums[j / 16 + 1] =
            (int16_t)hsum256i(_mm256_add_epi32(q[2], q[3]));

        /* Both packs interleave the two 128-bit lanes; one permute puts the
           32 bytes back in element order. */
        __m256i packed = _mm256_packs_epi16(_mm256_packs_epi32(q[0], q[1]),
                                            _mm256_packs_epi32(q[2], q[3]));
        _mm256_storeu_si256((__m256i *)(block->qs + j),
                            _mm256_permutevar8x32_epi32(packed, lane_order));
    }
    return 1.0f / inverse_scale;
}

/* Out-of-line so that the batched path can reach a type's row dot through a
   table. One indirect call buys a whole row of arithmetic, which is why the
   batched loop can afford what the single-vector kernels above inline. */
__attribute__((target("avx2,fma,f16c"))) static float
row_dot_q50(const void *row, int n, const ActivationBlock *x) {
    return q50_row(row, n, x);
}

__attribute__((target("avx2,fma,f16c"))) static float
row_dot_q4k(const void *row, int n, const ActivationBlock *x) {
    return q4k_row(row, n, x);
}

__attribute__((target("avx2,fma,f16c"))) static float
row_dot_q6k(const void *row, int n, const ActivationBlock *x) {
    return q6k_row(row, n, x);
}

/* One row per weight type that has an integer kernel. Anything missing here
   falls back to the scalar reference, so adding a type is a kernel and a
   row -- nothing else in this file learns its name.

   The two tables are the same arithmetic under different loop orders: a lone
   vector has nothing to amortize and wants several rows in flight, while a
   chunk wants one row held in cache across all of its vectors. Both reach the
   same per-block code, so neither can drift from the other. */
static const GemvQ8 gemv_q8_kernels[] = {
    [GGML_TYPE_Q5_0] = gemv_q50_q8_avx2,
    [GGML_TYPE_Q4_K] = gemv_q4k_q8_avx2,
    [GGML_TYPE_Q6_K] = gemv_q6k_q8_avx2,
};

static const RowDotQ8 row_dot_q8_kernels[] = {
    [GGML_TYPE_Q5_0] = row_dot_q50,
    [GGML_TYPE_Q4_K] = row_dot_q4k,
    [GGML_TYPE_Q6_K] = row_dot_q6k,
};

/* A type with no tile kernel is not a missing feature -- the row dot above
   covers it, one vector at a time. Q5_0 carries only the per-head key
   matrices, whose rows are four blocks long and already fit in cache, so
   sharing their unpacking would buy nothing. */
static const RowTileQ8 row_tile_q8_kernels[] = {
    [GGML_TYPE_Q4_K] = row_tile_q4k,
    [GGML_TYPE_Q6_K] = row_tile_q6k,
};
#endif

static GemvQ8 gemv_q8_kernel(unsigned type) {
#ifdef HAVE_AVX2
    if (have_avx2() && type < sizeof gemv_q8_kernels / sizeof *gemv_q8_kernels)
        return gemv_q8_kernels[type];
#else
    (void)type;
#endif
    return NULL;
}

static RowDotQ8 row_dot_q8_kernel(unsigned type) {
#ifdef HAVE_AVX2
    if (have_avx2() &&
        type < sizeof row_dot_q8_kernels / sizeof *row_dot_q8_kernels)
        return row_dot_q8_kernels[type];
#else
    (void)type;
#endif
    return NULL;
}

static RowTileQ8 row_tile_q8_kernel(unsigned type) {
#ifdef HAVE_AVX2
    if (have_avx2() &&
        type < sizeof row_tile_q8_kernels / sizeof *row_tile_q8_kernels)
        return row_tile_q8_kernels[type];
#else
    (void)type;
#endif
    return NULL;
}

uint16_t fp32_to_fp16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t man = bits & 0x7fffffu;

    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t sub = man >> shift;
        if ((man >> (shift - 1)) & 1) sub++;
        return (uint16_t)(sign | sub);
    }
    uint16_t half = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    if (man & 0x1000u) half++;
    return half;
}

static int activation_blocks(int n) { return (n + QK_K - 1) / QK_K; }

size_t activation_bytes(int n, int n_x) {
    return (size_t)n_x * activation_blocks(n) * sizeof(ActivationBlock);
}

static int batch_index(const ActivationBatch *x, int t) {
    return x->tokens ? x->tokens[t] : t;
}

static const ActivationBlock *batch_blocks(const ActivationBatch *x, int t) {
    return (const ActivationBlock *)x->blocks +
           (size_t)batch_index(x, t) * activation_blocks(x->n);
}

static const float *batch_values(const ActivationBatch *x, int t) {
    return x->values + (size_t)batch_index(x, t) * x->value_stride;
}

/* Scale so the largest-magnitude value lands on -127: taking the signed
   extreme rather than the absolute one keeps the quantization symmetric
   around whichever end actually saturates.

   A vector shorter than QK_K still quantizes -- the last block is zero-filled
   past the end, and zero quants and zero group sums contribute nothing to any
   kernel. Only whole 32-element groups are accepted, which is the smallest
   block any weight type uses and so the smallest span a weight scale can
   cover. */
static void quantize_vector(ActivationBlock *blocks, const float *values,
                            int n) {
#ifdef HAVE_AVX2
    int vector = have_avx2();
#endif
    for (int i = 0; i < activation_blocks(n); i++) {
        ActivationBlock *block = &blocks[i];
        const float *x = values + i * QK_K;
        int count = n - i * QK_K < QK_K ? n - i * QK_K : QK_K;

        memset(block, 0, sizeof *block);
#ifdef HAVE_AVX2
        if (vector) {
            block->scale = quantize_block_avx2(block, x, count);
            continue;
        }
#endif
        float extreme = 0, largest = 0;
        for (int j = 0; j < count; j++) {
            float magnitude = fabsf(x[j]);
            if (magnitude > largest) {
                largest = magnitude;
                extreme = x[j];
            }
        }
        if (largest == 0) continue;

        float inverse_scale = -127.0f / extreme;
        block->scale = 1.0f / inverse_scale;
        for (int j = 0; j < count; j++) {
            int q = (int)lrintf(inverse_scale * x[j]);
            block->qs[j] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
        }
        for (int g = 0; g < count / 16; g++) {
            int sum = 0;
            for (int k = 0; k < 16; k++) sum += block->qs[g * 16 + k];
            block->group_sums[g] = (int16_t)sum;
        }
    }
}

void activation_set(ActivationBatch *batch, void *scratch, const float *values,
                    size_t value_stride, int n, int n_x) {
    batch->values = values;
    batch->value_stride = value_stride;
    batch->tokens = NULL;
    batch->n = n;
    batch->n_x = n_x;
    batch->blocks = NULL;
    if (n % QUANT_GRANULE != 0 || !scratch) return;

    ActivationBlock *blocks = scratch;
    for (int t = 0; t < n_x; t++)
        quantize_vector(blocks + (size_t)t * activation_blocks(n),
                        values + (size_t)t * value_stride, n);
    batch->blocks = blocks;
}

size_t row_bytes(unsigned type, int n) {
    QuantType quant = quant_types[type];
    return (size_t)(n / quant.block_size) * (size_t)quant.type_size;
}

/* The types with no integer kernel, and the lengths that are not a whole
   number of blocks, dot the unquantized vector against each row. */
static void matmul_reference(float *out, size_t out_stride, const void *rows,
                             size_t stride, unsigned type, int n_in,
                             int row_begin, int row_end,
                             const ActivationBatch *x) {
    for (int t = 0; t < x->n_x; t++) {
        const float *values = batch_values(x, t);
        const unsigned char *row =
            (const unsigned char *)rows + (size_t)row_begin * stride;
        float *dst = out + (size_t)t * out_stride;
        for (int r = row_begin; r < row_end; r++, row += stride)
            dst[r] = gemv_row(row, type, n_in, values);
    }
}

/* Rows outer and vectors inner, so a row is read from memory once and stays in
   cache for the whole chunk. Whole tiles also share the cost of unpacking it;
   the remainder takes one vector at a time. A routed expert that only a couple
   of the chunk's tokens chose is all remainder, which is why the chunk wants
   to be wide enough to give each expert a tile's worth. */
static void matmul_chunk(float *out, size_t out_stride, const void *rows,
                         size_t stride, unsigned type, int n_in, int row_begin,
                         int row_end, const ActivationBatch *x,
                         RowDotQ8 row_dot) {
    RowTileQ8 row_tile = x->n_x >= GEMM_TILE ? row_tile_q8_kernel(type) : NULL;
    int tiled = row_tile ? x->n_x - x->n_x % GEMM_TILE : 0;
    const unsigned char *row =
        (const unsigned char *)rows + (size_t)row_begin * stride;

    for (int r = row_begin; r < row_end; r++, row += stride) {
        for (int t = 0; t < tiled; t += GEMM_TILE) {
            const ActivationBlock *lanes[GEMM_TILE];
            for (int lane = 0; lane < GEMM_TILE; lane++)
                lanes[lane] = batch_blocks(x, t + lane);
            row_tile(out + (size_t)t * out_stride + r, out_stride, row, n_in,
                     lanes);
        }
        for (int t = tiled; t < x->n_x; t++)
            out[(size_t)t * out_stride + r] =
                row_dot(row, n_in, batch_blocks(x, t));
    }
}

void matmul(float *out, size_t out_stride, const void *rows, unsigned type,
            int n_in, int row_begin, int row_end, const ActivationBatch *x) {
    size_t stride = row_bytes(type, n_in);
    RowDotQ8 row_dot = x->blocks ? row_dot_q8_kernel(type) : NULL;
    GemvQ8 gemv = row_dot && x->n_x == 1 ? gemv_q8_kernel(type) : NULL;

    if (gemv)
        gemv(out, rows, stride, row_begin, row_end, n_in, batch_blocks(x, 0));
    else if (row_dot)
        matmul_chunk(out, out_stride, rows, stride, type, n_in, row_begin,
                     row_end, x, row_dot);
    else
        matmul_reference(out, out_stride, rows, stride, type, n_in, row_begin,
                         row_end, x);
}

void rmsnorm(float *out, const float *x, const float *weight, int n,
             float eps) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

void softmax(float *values, int n) {
    float max = values[0];
    for (int i = 1; i < n; i++)
        if (values[i] > max) max = values[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        values[i] = expf(values[i] - max);
        sum += values[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) values[i] *= inv;
}

void swiglu(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++)
        out[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

void add_scaled(float *dst, const float *src, float scale, int n) {
#ifdef HAVE_AVX2
    if (have_avx2()) {
        add_scaled_avx2(dst, src, scale, n);
        return;
    }
#endif
    for (int i = 0; i < n; i++) dst[i] += src[i] * scale;
}

float dot_f32(const float *a, const float *b, int n) {
#ifdef HAVE_AVX2
    if (have_avx2()) return dot_f32_avx2(a, b, n);
#endif
    float acc = 0;
    for (int i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}

void dot_f32_rows(float *out, const float *rows, size_t stride, const float *b,
                  int n) {
#ifdef HAVE_AVX2
    if (have_avx2()) {
        dot_f32_rows_avx2(out, rows, stride, b, n);
        return;
    }
#endif
    for (int r = 0; r < CACHE_ROWS; r++)
        out[r] = dot_f32(rows + (size_t)r * stride, b, n);
}

void add_scaled_rows(float *dst, const float *rows, size_t stride,
                     const float *scales, int n) {
#ifdef HAVE_AVX2
    if (have_avx2()) {
        add_scaled_rows_avx2(dst, rows, stride, scales, n);
        return;
    }
#endif
    for (int r = 0; r < CACHE_ROWS; r++)
        add_scaled(dst, rows + (size_t)r * stride, scales[r], n);
}

void expand_fp16(float *dst, const uint16_t *values, int n) {
#ifdef HAVE_AVX2
    if (have_avx2()) {
        expand_fp16_avx2(dst, values, n);
        return;
    }
#endif
    for (int i = 0; i < n; i++) dst[i] = fp16_to_fp32(values[i]);
}

static float corr_dim(int n_dims, int orig_ctx, float n_rot, float base) {
    return (float)n_dims * logf((float)orig_ctx / (n_rot * 2.0f * (float)M_PI)) /
           (2.0f * logf(base));
}

void rope_init(RopeConfig *rope, float freq_base, float freq_scale, int n_dims,
               int orig_ctx, float beta_fast, float beta_slow) {
    rope->freq_scale = freq_scale;
    rope->theta_step = powf(freq_base, -2.0f / (float)n_dims);
    rope->n_dims = n_dims;
    rope->corr_low = floorf(corr_dim(n_dims, orig_ctx, beta_fast, freq_base));
    rope->corr_high = ceilf(corr_dim(n_dims, orig_ctx, beta_slow, freq_base));
}

void rope_position(float *cos_sin, const RopeConfig *rope, int position) {
    int n_pairs = rope->n_dims / 2;
    float theta_extrap = (float)position;
    float span = rope->corr_high - rope->corr_low;
    if (span < 0.001f) span = 0.001f;

    for (int i = 0; i < n_pairs; i++, theta_extrap *= rope->theta_step) {
        float ramp = 1.0f - ((float)i - rope->corr_low) / span;
        if (ramp < 0.0f) ramp = 0.0f;
        if (ramp > 1.0f) ramp = 1.0f;

        float theta_interp = rope->freq_scale * theta_extrap;
        float theta = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
        cos_sin[2 * i] = cosf(theta);
        cos_sin[2 * i + 1] = sinf(theta);
    }
}

/* Interleaved pairing: dimension 2i rotates against 2i+1. DeepSeek emits its
   rotary dimensions in that order, so the split-half pairing that most recent
   architectures use yields text that reads fluently but has lost track of
   position -- it recalls facts and cannot continue "1, 2, 3".
   Magnitude is left alone -- see the mscale note in model.c. */
void rope_apply(float *vec, const float *cos_sin, int n_dims) {
    for (int i = 0; i < n_dims / 2; i++) {
        float cos_theta = cos_sin[2 * i], sin_theta = cos_sin[2 * i + 1];
        float low = vec[2 * i], high = vec[2 * i + 1];
        vec[2 * i] = low * cos_theta - high * sin_theta;
        vec[2 * i + 1] = low * sin_theta + high * cos_theta;
    }
}

float gemv_row(const void *data, unsigned type, int n_in, const float *x) {
    switch (type) {
    case GGML_TYPE_F32:
        return dot_f32(data, x, n_in);
    case GGML_TYPE_F16: {
        const uint16_t *w = data;
        float acc = 0;
        for (int i = 0; i < n_in; i++) acc += fp16_to_fp32(w[i]) * x[i];
        return acc;
    }
    case GGML_TYPE_Q5_0: {
        const block_q5_0 *b = data;
        float acc = 0;
        int nb = n_in / 32;
        for (int i = 0; i < nb; i++) {
            float d = fp16_to_fp32(b[i].d);
            uint32_t qh;
            memcpy(&qh, b[i].qh, 4);
            for (int j = 0; j < 16; j++) {
                uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
                uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
                int32_t x0 = ((b[i].qs[j] & 0x0F) | xh_0) - 16;
                int32_t x1 = ((b[i].qs[j] >> 4) | xh_1) - 16;
                acc += x0 * d * x[i * 32 + j];
                acc += x1 * d * x[i * 32 + j + 16];
            }
        }
        return acc;
    }
    case GGML_TYPE_Q4_K: {
        const block_q4_K *b = data;
        float acc = 0;
        int nb = n_in / 256;
        for (int i = 0; i < nb; i++) {
            const uint8_t *q = b[i].qs;
            float d = fp16_to_fp32(b[i].d);
            float min = fp16_to_fp32(b[i].dmin);
            int is = 0;
            uint8_t sc, m;
            for (int j = 0; j < 256; j += 64) {
                get_scale_min_k4(is + 0, b[i].scales, &sc, &m);
                float d1 = d * sc;
                float m1 = min * m;
                get_scale_min_k4(is + 1, b[i].scales, &sc, &m);
                float d2 = d * sc;
                float m2 = min * m;
                for (int l = 0; l < 32; l++) {
                    acc += (d1 * (q[l] & 0xF) - m1) * x[i * 256 + j + l];
                    acc += (d2 * (q[l] >> 4) - m2) * x[i * 256 + j + 32 + l];
                }
                q += 32;
                is += 2;
            }
        }
        return acc;
    }
    case GGML_TYPE_Q6_K: {
        const block_q6_K *b = data;
        float acc = 0;
        int nb = n_in / 256;
        for (int i = 0; i < nb; i++) {
            float d = fp16_to_fp32(b[i].d);
            const uint8_t *ql = b[i].ql;
            const uint8_t *qh = b[i].qh;
            const int8_t *sc = b[i].scales;
            for (int n2 = 0; n2 < 256; n2 += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int8_t q1 =
                        (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 =
                        (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 =
                        (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 =
                        (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    acc += (d * sc[is + 0] * q1) * x[i * 256 + n2 + l];
                    acc += (d * sc[is + 2] * q2) * x[i * 256 + n2 + l + 32];
                    acc += (d * sc[is + 4] * q3) * x[i * 256 + n2 + l + 64];
                    acc += (d * sc[is + 6] * q4) * x[i * 256 + n2 + l + 96];
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return acc;
    }
    default:
        return 0;
    }
}
