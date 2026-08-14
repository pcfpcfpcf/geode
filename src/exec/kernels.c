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

/* Activations quantized per 256, mirroring the weight block size. group_sums
   holds the sum of each 16 quants: Q4_K needs them in pairs to cancel its
   sub-block minimum without touching the quants again. */
typedef struct {
    float scale;
    int8_t qs[QK_K];
    int16_t group_sums[QK_K / 16];
} ActivationBlock;

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
__attribute__((target("avx2,fma,f16c"))) static float
gemv_q4k_q8_avx2(const block_q4_K *blocks, int nb,
                 const ActivationBlock *activations) {
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    float offset_acc = 0;

    for (int i = 0; i < nb; i++) {
        const block_q4_K *b = &blocks[i];
        const ActivationBlock *a = &activations[i];
        const uint8_t *q = b->qs;
        __m256i sumi = _mm256_setzero_si256();
        int offsets = 0;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, b->scales, &sc1, &m1);
            get_scale_min_k4(is + 1, b->scales, &sc2, &m2);

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
                                        _mm256_set1_epi16(sc1)));
            sumi = _mm256_add_epi32(
                sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(high, xhigh),
                                        _mm256_set1_epi16(sc2)));

            int g = j / 16;
            offsets += m1 * (a->group_sums[g + 0] + a->group_sums[g + 1]);
            offsets += m2 * (a->group_sums[g + 2] + a->group_sums[g + 3]);
            q += 32;
            is += 2;
        }
        acc = _mm256_fmadd_ps(
            _mm256_set1_ps(fp16_to_fp32(b->d) * a->scale),
            _mm256_cvtepi32_ps(sumi), acc);
        offset_acc -= fp16_to_fp32(b->dmin) * a->scale * (float)offsets;
    }
    return hsum256(acc) + offset_acc;
}

/* Q6_K is symmetric around 32 with no stored minimum, so the -32 is recovered
   by multiplying the activations against a constant 32 and subtracting. */
__attribute__((target("avx2,fma,f16c"))) static float
gemv_q6k_q8_avx2(const block_q6_K *blocks, int nb,
                 const ActivationBlock *activations) {
    const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
    const __m256i pair_mask = _mm256_set1_epi8(3);
    const __m256i thirty_two = _mm256_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
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
            _mm256_set1_ps(fp16_to_fp32(b->d) * a->scale),
            _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum256(acc);
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
#endif

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

size_t activation_bytes(int n) {
    return (size_t)(n / QK_K) * sizeof(ActivationBlock);
}

/* Scale so the largest-magnitude value lands on -127: taking the signed
   extreme rather than the absolute one keeps the quantization symmetric
   around whichever end actually saturates. */
void activation_set(Activation *activation, void *scratch, const float *values,
                    int n) {
    activation->values = values;
    activation->n = n;
    activation->blocks = NULL;
    if (n % QK_K != 0 || !scratch) return;

    ActivationBlock *blocks = scratch;
    for (int i = 0; i < n / QK_K; i++) {
        ActivationBlock *block = &blocks[i];
        const float *x = values + i * QK_K;

        float extreme = 0, largest = 0;
        for (int j = 0; j < QK_K; j++) {
            float magnitude = fabsf(x[j]);
            if (magnitude > largest) {
                largest = magnitude;
                extreme = x[j];
            }
        }
        if (largest == 0) {
            memset(block, 0, sizeof *block);
            continue;
        }

        float inverse_scale = -127.0f / extreme;
        for (int j = 0; j < QK_K; j++) {
            int q = (int)lrintf(inverse_scale * x[j]);
            block->qs[j] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
        }
        for (int g = 0; g < QK_K / 16; g++) {
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

#ifdef HAVE_AVX2
    if (x->blocks && have_avx2() &&
        (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K)) {
        int nb = n_in / QK_K;
        for (int r = row_begin; r < row_end; r++, row += stride)
            out[r] = type == GGML_TYPE_Q4_K
                         ? gemv_q4k_q8_avx2((const block_q4_K *)row, nb,
                                            x->blocks)
                         : gemv_q6k_q8_avx2((const block_q6_K *)row, nb,
                                            x->blocks);
        return;
    }
#endif
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
