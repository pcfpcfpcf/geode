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

/* The rotation for a position is the same for every head and every layer, so
   the transcendentals are paid once per token and rope_apply is left with the
   rotation itself. cos_sin holds a cosine and a sine per rotary pair. */
void rope_position(float *cos_sin, const RopeConfig *rope, int position);
void rope_apply(float *vec, const float *cos_sin, int n_dims);

float fp16_to_fp32(uint16_t half);
uint16_t fp32_to_fp16(float value);

/* An activation vector quantized to int8 once per matvec, so the cost is paid
   once and every row of the weight matrix reuses it. Weight types with no
   integer path, and lengths that are not a whole number of blocks, fall back
   to `values`. */
typedef struct {
    const float *values;
    const void *blocks;
    int n;
} Activation;

size_t activation_bytes(int n);
void activation_set(Activation *activation, void *scratch, const float *values,
                    int n);

size_t row_bytes(unsigned type, int n);
float gemv_row(const void *data, unsigned type, int n_in, const float *x);
void dequant_row(const void *data, unsigned type, int n, float *dst);
void matvec(float *out, const void *rows, unsigned type, int n_in,
            int row_begin, int row_end, const Activation *x);

void rmsnorm(float *out, const float *x, const float *weight, int n,
             float eps);
void softmax(float *values, int n);
void swiglu(float *out, const float *gate, const float *up, int n);
void add_scaled(float *dst, const float *src, float scale, int n);

/* The latent kv cache is stored as fp16 and every attention head reads the
   same rows, so a row is expanded to floats once and then scored and folded
   back in fp32: converting inside each head made the conversion, not the
   arithmetic, scale with head count. */
void expand_fp16(float *dst, const uint16_t *values, int n);
float dot_f32(const float *a, const float *b, int n);

#endif
