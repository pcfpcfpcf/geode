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

__attribute__((target("avx2,fma,f16c"), always_inline)) static inline float
q4k_row(const void *row, int n, const ActivationBlock *activations) {
    const block_q4_K *blocks = row;
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    __m128 offset_acc = _mm_setzero_ps();

    for (int i = 0; i < n / QK_K; i++) {
        const block_q4_K *b = &blocks[i];
        const ActivationBlock *a = &activations[i];
        const uint8_t *q = b->qs;
        __m256i sumi = _mm256_setzero_si256();
        int is = 0;

        __m128i scales, mins;
        q4k_scales_mins(b->scales, &scales, &mins);
        int16_t scale_of[8];
        _mm_storeu_si128((__m128i *)scale_of, scales);

        /* Each sub-block's minimum is constant across its 32 activations, so
           all eight collapse into one dot product against the group sums
           rather than an accumulation inside the loop below. */
        __m256i sums = _mm256_loadu_si256((const __m256i *)a->group_sums);
        __m128i pairs = _mm_hadd_epi16(_mm256_castsi256_si128(sums),
                                       _mm256_extracti128_si256(sums, 1));
        offset_acc = _mm_fmadd_ps(
            _mm_set1_ps(half_to_float(b->dmin) * a->scale),
            _mm_cvtepi32_ps(_mm_madd_epi16(mins, pairs)), offset_acc);

        for (int j = 0; j < 256; j += 64) {
            __m256i packed = _mm256_loadu_si256((const __m256i *)q);
            __m256i low = _mm256_and_si256(packed, nibble_mask);
            __m256i high =
                _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask);
            __m256i xlow =
                _mm256_loadu_si256((const __m256i *)(a->qs + j));
            __m256i xhigh =
                _mm256_loadu_si256((const __m256i *)(a->qs + j + 32));

            sumi = _mm256_add_epi32(
                sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(low, xlow),
                                        _mm256_set1_epi16(scale_of[is + 0])));
            sumi = _mm256_add_epi32(
                sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(high, xhigh),
                                        _mm256_set1_epi16(scale_of[is + 1])));
            q += 32;
            is += 2;
        }
        acc = _mm256_fmadd_ps(
            _mm256_set1_ps(half_to_float(b->d) * a->scale),
            _mm256_cvtepi32_ps(sumi), acc);
    }
    __m128 total = _mm_sub_ps(_mm_add_ps(_mm256_castps256_ps128(acc),
                                         _mm256_extractf128_ps(acc, 1)),
                              offset_acc);
    total = _mm_add_ps(total, _mm_movehl_ps(total, total));
    total = _mm_add_ss(total, _mm_movehdup_ps(total));
    return _mm_cvtss_f32(total);
}

__attribute__((target("avx2,fma,f16c"))) static void
gemv_q4k_q8_avx2(float *out, const void *rows, size_t stride, int row_begin,
                 int row_end, int n, const ActivationBlock *x) {
    const unsigned char *row =
        (const unsigned char *)rows + (size_t)row_begin * stride;
    for (int r = row_begin; r < row_end; r++, row += stride)
        out[r] = q4k_row(row, n, x);
}

/* Q6_K is symmetric around 32 with no stored minimum, so the -32 is recovered
   by multiplying the activations against a constant 32 and subtracting. */
__attribute__((target("avx2,fma,f16c"), always_inline)) static inline float
q6k_row(const void *row, int n, const ActivationBlock *activations) {
    const block_q6_K *blocks = row;
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
    const __m256i pair_mask = _mm256_set1_epi8(3);
    const __m256i thirty_two = _mm256_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < n / QK_K; i++) {
        const block_q6_K *b = &blocks[i];
        const ActivationBlock *a = &activations[i];
        const uint8_t *ql = b->ql;
        const uint8_t *qh = b->qh;
        const int8_t *sc = b->scales;
        __m256i sumi = _mm256_setzero_si256();

        for (int n2 = 0; n2 < 256; n2 += 128) {
            __m256i low = _mm256_loadu_si256((const __m256i *)ql);
            __m256i low32 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i high = _mm256_loadu_si256((const __m256i *)qh);

            __m256i quants[4];
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

            for (int part = 0; part < 4; part++) {
                __m256i x = _mm256_loadu_si256(
                    (const __m256i *)(a->qs + n2 + 32 * part));
                __m256i product = _mm256_sub_epi16(
                    _mm256_maddubs_epi16(quants[part], x),
                    _mm256_maddubs_epi16(thirty_two, x));
                /* Lanes 0-7 cover the first 16 weights and lanes 8-15 the
                   next, and those two halves carry different scales. */
                __m256i scales = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(_mm_set1_epi16(sc[2 * part + 0])),
                    _mm_set1_epi16(sc[2 * part + 1]), 1);
                sumi = _mm256_add_epi32(sumi,
                                        _mm256_madd_epi16(scales, product));
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
        acc = _mm256_fmadd_ps(
            _mm256_set1_ps(half_to_float(b->d) * a->scale),
            _mm256_cvtepi32_ps(sumi), acc);
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

__attribute__((target("avx2,fma,f16c"))) static void
gemv_q50_q8_avx2(float *out, const void *rows, size_t stride, int row_begin,
                 int row_end, int n, const ActivationBlock *x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const unsigned char *base =
        (const unsigned char *)rows + (size_t)row_begin * stride;
    int blocks = n / QUANT_GRANULE;
    int r = row_begin;

    for (; r + Q50_ROWS <= row_end; r += Q50_ROWS, base += Q50_ROWS * stride) {
        __m256 acc[Q50_ROWS];
        float offset[Q50_ROWS];
        for (int k = 0; k < Q50_ROWS; k++) {
            acc[k] = _mm256_setzero_ps();
            offset[k] = 0;
        }
        for (int i = 0; i < blocks; i++) {
            int sub = i % (QK_K / QUANT_GRANULE);
            const ActivationBlock *a = &x[i / (QK_K / QUANT_GRANULE)];
            __m256i xq = _mm256_loadu_si256(
                (const __m256i *)(a->qs + QUANT_GRANULE * sub));
            float sums = (float)(a->group_sums[2 * sub] +
                                 a->group_sums[2 * sub + 1]);

            for (int k = 0; k < Q50_ROWS; k++) {
                const block_q5_0 *b =
                    &((const block_q5_0 *)(base + (size_t)k * stride))[i];
                __m256i sumi = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(q50_quants(b), xq), ones);
                float scale = half_to_float(b->d) * a->scale;
                acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(scale),
                                         _mm256_cvtepi32_ps(sumi), acc[k]);
                offset[k] -= scale * 16.0f * sums;
            }
        }
        for (int k = 0; k < Q50_ROWS; k++)
            out[r + k] = hsum256(acc[k]) + offset[k];
    }

    for (; r < row_end; r++, base += stride) {
        __m256 acc = _mm256_setzero_ps();
        float offset = 0;
        for (int i = 0; i < blocks; i++) {
            int sub = i % (QK_K / QUANT_GRANULE);
            const ActivationBlock *a = &x[i / (QK_K / QUANT_GRANULE)];
            const block_q5_0 *b = &((const block_q5_0 *)base)[i];
            __m256i sumi = _mm256_madd_epi16(
                _mm256_maddubs_epi16(
                    q50_quants(b),
                    _mm256_loadu_si256(
                        (const __m256i *)(a->qs + QUANT_GRANULE * sub))),
                ones);
            float scale = half_to_float(b->d) * a->scale;
            acc = _mm256_fmadd_ps(_mm256_set1_ps(scale),
                                  _mm256_cvtepi32_ps(sumi), acc);
            offset -= scale * 16.0f * (float)(a->group_sums[2 * sub] +
                                              a->group_sums[2 * sub + 1]);
        }
        out[r] = hsum256(acc) + offset;
    }
}

__attribute__((target("avx2,fma,f16c"))) static float
dot_fp16_avx2(const uint16_t *values, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(values + i))),
            _mm256_loadu_ps(x + i), acc);
    float sum = hsum256(acc);
    for (; i < n; i++) sum += fp16_to_fp32(values[i]) * x[i];
    return sum;
}

__attribute__((target("avx2,fma,f16c"))) static void
accumulate_fp16_avx2(float *dst, const uint16_t *values, float weight, int n) {
    __m256 w = _mm256_set1_ps(weight);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(
            dst + i,
            _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(values + i))),
                w, _mm256_loadu_ps(dst + i)));
    for (; i < n; i++) dst[i] += fp16_to_fp32(values[i]) * weight;
}

/* One row per weight type that has an integer kernel. Anything missing here
   falls back to the scalar reference, so adding a type is a kernel and a
   row -- nothing else in this file learns its name. */
static const GemvQ8 gemv_q8_kernels[] = {
    [GGML_TYPE_Q5_0] = gemv_q50_q8_avx2,
    [GGML_TYPE_Q4_K] = gemv_q4k_q8_avx2,
    [GGML_TYPE_Q6_K] = gemv_q6k_q8_avx2,
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

size_t activation_bytes(int n) {
    return (size_t)activation_blocks(n) * sizeof(ActivationBlock);
}

/* Scale so the largest-magnitude value lands on -127: taking the signed
   extreme rather than the absolute one keeps the quantization symmetric
   around whichever end actually saturates.

   A vector shorter than QK_K still quantizes -- the last block is zero-filled
   past the end, and zero quants and zero group sums contribute nothing to any
   kernel. Only whole 32-element groups are accepted, which is the smallest
   block any weight type uses and so the smallest span a weight scale can
   cover. */
void activation_set(Activation *activation, void *scratch, const float *values,
                    int n) {
    activation->values = values;
    activation->n = n;
    activation->blocks = NULL;
    if (n % QUANT_GRANULE != 0 || !scratch) return;

    ActivationBlock *blocks = scratch;
    for (int i = 0; i < activation_blocks(n); i++) {
        ActivationBlock *block = &blocks[i];
        const float *x = values + i * QK_K;
        int count = n - i * QK_K < QK_K ? n - i * QK_K : QK_K;

        memset(block, 0, sizeof *block);
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
        for (int j = 0; j < count; j++) {
            int q = (int)lrintf(inverse_scale * x[j]);
            block->qs[j] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
        }
        for (int g = 0; g < count / 16; g++) {
            int sum = 0;
            for (int k = 0; k < 16; k++) sum += block->qs[g * 16 + k];
            block->group_sums[g] = (int16_t)sum;
        }
        block->scale = 1.0f / inverse_scale;
    }
    activation->blocks = blocks;
}

size_t row_bytes(unsigned type, int n) {
    QuantType quant = quant_types[type];
    return (size_t)(n / quant.block_size) * (size_t)quant.type_size;
}

void matvec(float *out, const void *rows, unsigned type, int n_in,
            int row_begin, int row_end, const Activation *x) {
    size_t stride = row_bytes(type, n_in);
    const unsigned char *row =
        (const unsigned char *)rows + (size_t)row_begin * stride;

    GemvQ8 gemv = x->blocks ? gemv_q8_kernel(type) : NULL;
    if (gemv) {
        gemv(out, rows, stride, row_begin, row_end, n_in, x->blocks);
        return;
    }
    for (int r = row_begin; r < row_end; r++, row += stride)
        out[r] = gemv_row(row, type, n_in, x->values);
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
    for (int i = 0; i < n; i++) dst[i] += src[i] * scale;
}

float dot_fp16(const uint16_t *values, const float *x, int n) {
#ifdef HAVE_AVX2
    if (have_avx2()) return dot_fp16_avx2(values, x, n);
#endif
    float acc = 0;
    for (int i = 0; i < n; i++) acc += fp16_to_fp32(values[i]) * x[i];
    return acc;
}

void accumulate_fp16(float *dst, const uint16_t *values, float weight, int n) {
#ifdef HAVE_AVX2
    if (have_avx2()) {
        accumulate_fp16_avx2(dst, values, weight, n);
        return;
    }
#endif
    for (int i = 0; i < n; i++) dst[i] += fp16_to_fp32(values[i]) * weight;
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

/* Interleaved pairing: dimension 2i rotates against 2i+1. DeepSeek emits its
   rotary dimensions in that order, so the split-half pairing that most recent
   architectures use yields text that reads fluently but has lost track of
   position -- it recalls facts and cannot continue "1, 2, 3".
   Magnitude is left alone -- see the mscale note in model.c. */
void rope_apply(float *vec, const RopeConfig *rope, int position) {
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
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        float low = vec[2 * i];
        float high = vec[2 * i + 1];
        vec[2 * i] = low * cos_theta - high * sin_theta;
        vec[2 * i + 1] = low * sin_theta + high * cos_theta;
    }
}

float gemv_row(const void *data, unsigned type, int n_in, const float *x) {
    switch (type) {
    case GGML_TYPE_F32: {
        const float *w = data;
        float acc = 0;
        for (int i = 0; i < n_in; i++) acc += w[i] * x[i];
        return acc;
    }
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
