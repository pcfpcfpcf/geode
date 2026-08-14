#include "kernels.h"
#include "quant.h"

#include <math.h>
#include <string.h>

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

size_t row_bytes(unsigned type, int n) {
    QuantType quant = quant_types[type];
    return (size_t)(n / quant.block_size) * (size_t)quant.type_size;
}

void matvec(float *out, const void *rows, unsigned type, int n_in,
            int row_begin, int row_end, const float *x) {
    const unsigned char *row = (const unsigned char *)rows +
                               (size_t)row_begin * row_bytes(type, n_in);
    size_t stride = row_bytes(type, n_in);
    for (int r = row_begin; r < row_end; r++, row += stride)
        out[r] = gemv_row(row, type, n_in, x);
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
    float acc = 0;
    for (int i = 0; i < n; i++) acc += fp16_to_fp32(values[i]) * x[i];
    return acc;
}

void accumulate_fp16(float *dst, const uint16_t *values, float weight, int n) {
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
