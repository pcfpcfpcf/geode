#ifndef GEODE_KERNELS_H
#define GEODE_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#define QK_K 256

/* corr_low/corr_high are the rotary dimensions between which YaRN crosses
   over from extrapolating the original context to interpolating the scaled
   one; rope_init derives them from beta_fast/beta_slow. */
typedef struct {
    float freq_scale;
    float theta_step;
    int n_dims;
    float corr_low;
    float corr_high;
} RopeConfig;

void rope_init(RopeConfig *rope, float freq_base, float freq_scale, int n_dims,
               int orig_ctx, float beta_fast, float beta_slow);
void rope_apply(float *vec, const RopeConfig *rope, int position);

float fp16_to_fp32(uint16_t half);
uint16_t fp32_to_fp16(float value);

size_t row_bytes(unsigned type, int n);
float gemv_row(const void *data, unsigned type, int n_in, const float *x);
void dequant_row(const void *data, unsigned type, int n, float *dst);
void matvec(float *out, const void *rows, unsigned type, int n_in,
            int row_begin, int row_end, const float *x);

void rmsnorm(float *out, const float *x, const float *weight, int n,
             float eps);
void softmax(float *values, int n);
void swiglu(float *out, const float *gate, const float *up, int n);
void add_scaled(float *dst, const float *src, float scale, int n);

/* The latent kv cache is read as fp16 in both directions: once to score a
   query against every cached position, once to fold those scores back into a
   latent vector. */
float dot_fp16(const uint16_t *values, const float *x, int n);
void accumulate_fp16(float *dst, const uint16_t *values, float weight, int n);

#endif
