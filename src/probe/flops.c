#include "flops.h"

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define HAVE_AVX2_BENCH 1
#include <immintrin.h>
#endif

#define WORKERS_MAX 64
#define ACCUMULATORS 12 /* covers FMA latency x throughput on 2 FMA ports */
#define BENCH_SECONDS 0.5
#define DEQUANT_ROWS 8 /* activation rows per weight pass (GEMM tiling) */
#define DEQUANT_BUFFER_BYTES (512u << 10) /* cache-resident per worker */
#define Q4K_BLOCK_ELEMS 256
#define Q4K_BLOCK_BYTES 144

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static volatile float sink_f;
static volatile double sink_d;

/* Operands chosen so accumulators grow to +inf: inf operands run at full
   speed and the loop never hits the denormal slow path. */
#ifdef HAVE_AVX2_BENCH
__attribute__((target("avx2,fma"))) static double bench_avx2(double seconds) {
    const __m256 a = _mm256_set1_ps(1.0001f);
    const __m256 b = _mm256_set1_ps(1.0001f);
    __m256 acc[ACCUMULATORS];
    for (int i = 0; i < ACCUMULATORS; i++) acc[i] = _mm256_set1_ps(1.0f + i);
    double best = 0;
    double t_end = now_s() + seconds;
    while (now_s() < t_end) {
        double t0 = now_s(), t1 = t0;
        unsigned long long iters = 0;
        do {
            for (int k = 0; k < 100; k++)
                for (int i = 0; i < ACCUMULATORS; i++)
                    acc[i] = _mm256_fmadd_ps(acc[i], a, b);
            iters += 100;
            t1 = now_s();
        } while (t1 - t0 < 0.15 && t1 < t_end);
        double rate = (double)iters * ACCUMULATORS * 16 / (t1 - t0);
        if (rate > best) best = rate;
    }
    float lanes[8], sum = 0;
    for (int i = 0; i < ACCUMULATORS; i++) {
        _mm256_storeu_ps(lanes, acc[i]);
        sum += lanes[0];
    }
    sink_f = sum;
    return best;
}
#endif

#ifdef HAVE_AVX2_BENCH
/* Real ggml Q4_K block: 256 quants in 144 bytes, 8 sub-blocks of 32 with
   6-bit scales/mins packed into 12 bytes. */
typedef struct {
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

static int q4k_scale(const block_q4_K *b, int j) {
    if (j < 4) return b->scales[j] & 63;
    return (b->scales[j + 4] & 0x0F) | ((b->scales[j - 4] >> 6) << 4);
}

static int q4k_min(const block_q4_K *b, int j) {
    if (j < 4) return b->scales[j + 4] & 63;
    return (b->scales[j + 4] >> 4) | ((b->scales[j] >> 6) << 4);
}

__attribute__((target("avx2,fma,f16c"), always_inline)) static inline float
hsum256(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

/* Same structure as a real dequant GEMM: unpack nibbles once per sub-block,
   fold the scale in, then FMA against every activation row. Min correction
   uses precomputed row sums, outside the inner loop. */
__attribute__((target("avx2,fma,f16c"))) static double
bench_q4k_dequant(const uint8_t *buf, size_t nblocks, const float *x,
                  const float *rowsum, double seconds) {
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);
    float acc[DEQUANT_ROWS] = {0};
    double best = 0;
    double t_end = now_s() + seconds;
    while (now_s() < t_end) {
        double t0 = now_s(), t1 = t0;
        unsigned long long passes = 0;
        do {
            for (size_t nb = 0; nb < nblocks; nb++) {
                const block_q4_K *b =
                    (const block_q4_K *)(buf + nb * Q4K_BLOCK_BYTES);
                float d = _cvtsh_ss(b->d), dmin = _cvtsh_ss(b->dmin);
                float fs[8], fm_sum = 0;
                for (int j = 0; j < 8; j++) {
                    fs[j] = d * q4k_scale(b, j);
                    fm_sum += dmin * q4k_min(b, j);
                }
                __m256 accv[DEQUANT_ROWS];
                for (int r = 0; r < DEQUANT_ROWS; r++) accv[r] = _mm256_setzero_ps();
                for (int j = 0; j < 8; j++) {
                    __m128i bytes =
                        _mm_loadu_si128((const __m128i *)(b->qs + 16 * j));
                    __m128i lo8 = _mm_and_si128(bytes, nibble_mask);
                    __m128i hi8 =
                        _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
                    __m256 vfs = _mm256_set1_ps(fs[j]);
                    __m256 q0 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo8)), vfs);
                    __m256 q1 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(
                            _mm256_cvtepi8_epi32(_mm_srli_si128(lo8, 8))),
                        vfs);
                    __m256 q2 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi8)), vfs);
                    __m256 q3 = _mm256_mul_ps(
                        _mm256_cvtepi32_ps(
                            _mm256_cvtepi8_epi32(_mm_srli_si128(hi8, 8))),
                        vfs);
                    for (int r = 0; r < DEQUANT_ROWS; r++) {
                        const float *xr = x + r * Q4K_BLOCK_ELEMS + 32 * j;
                        accv[r] = _mm256_fmadd_ps(q0, _mm256_loadu_ps(xr), accv[r]);
                        accv[r] = _mm256_fmadd_ps(q1, _mm256_loadu_ps(xr + 8), accv[r]);
                        accv[r] = _mm256_fmadd_ps(q2, _mm256_loadu_ps(xr + 16), accv[r]);
                        accv[r] = _mm256_fmadd_ps(q3, _mm256_loadu_ps(xr + 24), accv[r]);
                    }
                }
                for (int r = 0; r < DEQUANT_ROWS; r++)
                    acc[r] += hsum256(accv[r]) - fm_sum * rowsum[r];
            }
            passes++;
            t1 = now_s();
        } while (t1 - t0 < 0.15 && t1 < t_end);
        double rate = (double)passes * nblocks * Q4K_BLOCK_ELEMS *
                      DEQUANT_ROWS * 2 / (t1 - t0);
        if (rate > best) best = rate;
    }
    float sum = 0;
    for (int r = 0; r < DEQUANT_ROWS; r++) sum += acc[r];
    sink_f = sum;
    return best;
}
#endif

static double bench_scalar(double seconds) {
    const double a = 1.0001, b = 1.0001;
    double acc[ACCUMULATORS];
    for (int i = 0; i < ACCUMULATORS; i++) acc[i] = 1.0 + i;
    double best = 0;
    double t_end = now_s() + seconds;
    while (now_s() < t_end) {
        double t0 = now_s(), t1 = t0;
        unsigned long long iters = 0;
        do {
            for (int k = 0; k < 100; k++)
                for (int i = 0; i < ACCUMULATORS; i++)
                    acc[i] = acc[i] * a + b;
            iters += 100;
            t1 = now_s();
        } while (t1 - t0 < 0.15 && t1 < t_end);
        double rate = (double)iters * ACCUMULATORS * 2 / (t1 - t0);
        if (rate > best) best = rate;
    }
    double sum = 0;
    for (int i = 0; i < ACCUMULATORS; i++) sum += acc[i];
    sink_d = sum;
    return best;
}

typedef struct {
    int cpu;
    double flops;
    double dequant_flops;
} Worker;

static uint64_t prng_state = 0x9E3779B97F4A7C15ull;

static uint64_t prng_next(void) {
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 7;
    prng_state ^= prng_state << 17;
    return prng_state;
}

static void *flops_worker(void *arg) {
    Worker *w = arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(w->cpu, &set);
    sched_setaffinity(0, sizeof set, &set); /* unpinned is fine on failure */
#ifdef HAVE_AVX2_BENCH
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        __builtin_cpu_supports("f16c")) {
        w->flops = bench_avx2(BENCH_SECONDS);

        uint8_t *buf = malloc(DEQUANT_BUFFER_BYTES);
        float *x = malloc(DEQUANT_ROWS * Q4K_BLOCK_ELEMS * sizeof *x);
        float rowsum[DEQUANT_ROWS];
        if (buf && x) {
            for (size_t i = 0; i < DEQUANT_BUFFER_BYTES; i += 8)
                *(uint64_t *)(buf + i) = prng_next();
            for (int r = 0; r < DEQUANT_ROWS; r++) {
                rowsum[r] = 0;
                for (int i = 0; i < Q4K_BLOCK_ELEMS; i++) {
                    x[r * Q4K_BLOCK_ELEMS + i] =
                        (float)(prng_next() & 0xFF) / 128.0f - 1.0f;
                    rowsum[r] += x[r * Q4K_BLOCK_ELEMS + i];
                }
            }
            w->dequant_flops =
                bench_q4k_dequant(buf, DEQUANT_BUFFER_BYTES / Q4K_BLOCK_BYTES,
                                  x, rowsum, BENCH_SECONDS);
        }
        free(buf);
        free(x);
        return NULL;
    }
#endif
    w->flops = bench_scalar(BENCH_SECONDS);
    return NULL;
}

double flops_probe(int *threads, double *q4k_dequant_flops) {
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online < 1) return 0;
    int n = online > WORKERS_MAX ? WORKERS_MAX : (int)online;

    Worker *workers = calloc(n, sizeof *workers);
    pthread_t *tids = malloc(n * sizeof *tids);
    for (int i = 0; i < n; i++) {
        workers[i].cpu = i;
        if (pthread_create(&tids[i], NULL, flops_worker, &workers[i])) {
            n = i;
            break;
        }
    }
    double total = 0, total_dequant = 0;
    for (int i = 0; i < n; i++) {
        pthread_join(tids[i], NULL);
        total += workers[i].flops;
        total_dequant += workers[i].dequant_flops;
    }
    free(workers);
    free(tids);
    *threads = n;
    *q4k_dequant_flops = total_dequant;
    return n > 0 ? total : 0;
}
