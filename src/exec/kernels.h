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
void rope_apply_neox(float *vec, const float *cos_sin, int n_dims);

float fp16_to_fp32(uint16_t half);
uint16_t fp32_to_fp16(float value);

/* A chunk of activation vectors quantized to int8 together, so the cost is
   paid once and every row of the weight matrix reuses all of them. One
   vector's blocks stay contiguous, because a dot walks them while the weight
   row is what has to stay in cache. `values` holds the same vectors
   unquantized, `value_stride` apart, for the weight types with no integer path
   and the lengths that are not a whole number of blocks. `tokens` picks a
   subset -- each routed expert of a chunk sees only the tokens that chose it
   -- and NULL means all of them in order. */
typedef struct {
    const float *values;
    const void *blocks;
    const int *tokens;
    size_t value_stride;
    int n;
    int n_x;
} ActivationBatch;

size_t activation_bytes(int n, int n_x);
void activation_set(ActivationBatch *batch, void *scratch, const float *values,
                    size_t value_stride, int n, int n_x);

size_t row_bytes(unsigned type, int n);
float gemv_row(const void *data, unsigned type, int n_in, const float *x);
void dequant_row(const void *data, unsigned type, int n, float *dst);

/* Fills out[t * out_stride + r] for every vector t of x and every row r in
   [row_begin, row_end). A chunk dots each row against all of its vectors while
   the row is still in cache, so the row is read from memory once per chunk
   rather than once per token -- which is the whole difference between prefill
   and running decode repeatedly. */
void matmul(float *out, size_t out_stride, const void *rows, unsigned type,
            int n_in, int row_begin, int row_end, const ActivationBatch *x);

void rmsnorm(float *out, const float *x, const float *weight, int n,
             float eps);
void softmax(float *values, int n);
void swiglu(float *out, const float *gate, const float *up, int n);
void add_scaled(float *dst, const float *src, float scale, int n);

float dot_f32(const float *a, const float *b, int n);

/* The latent kv cache is stored quantized: eight bits and a scale per segment
   of a slot. Its quants are biased to unsigned because that is the operand
   pairing VPMADDUBSW wants -- 32 multiply-accumulates an instruction against 8
   for float fma -- and scoring, whose other operand is a query it can quantize
   too, reads them straight and pays no expansion at all.

   The fold has no such path: it weights rows by the softmax, not by a
   quantized vector, and the latent it walks is contiguous in rank rather than
   in position. So it expands a row to floats once per block and reuses it
   across heads, the way scoring used to. */
#define CACHE_QUANT_BIAS 128

/* Returns the scale, or 0 when every value was zero. */
float quantize_cache(uint8_t *quants, const float *values, int n);
void expand_int8(float *dst, const uint8_t *quants, float scale, int n);

/* A query quantized to dot against those slots. The query is what gives up a
   bit rather than the cache: VPMADDUBSW sums two products into an int16, and
   2 * 255 * 63 fits where 2 * 255 * 127 saturates. */
#define QUERY_QUANT_MAX 63

typedef struct {
    float scale;
    /* What the cache's own bias contributes to every dot against this query --
       CACHE_QUANT_BIAS times the sum of the quants -- taken back off each
       result before it is scaled. */
    float bias;
} QuantizedQuery;

void quantize_query(QuantizedQuery *query, int8_t *quants, const float *values,
                    int n);

/* CACHE_ROWS rows of a matrix at once, `stride` apart. One row at a time loads
   the vector -- or reloads and rewrites `dst` -- once per multiply, which at
   these widths costs more than the multiply; a block of rows spends those
   loads once and reuses them, and needs no extra memory to do it. */
#define CACHE_ROWS 4
void dot_int8_rows(int32_t *out, const uint8_t *rows, size_t stride,
                   const int8_t *query, int n);
void add_scaled_rows(float *dst, const float *rows, size_t stride,
                     const float *scales, int n);

#endif
