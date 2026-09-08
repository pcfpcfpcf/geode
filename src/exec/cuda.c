#include "cuda.h"

#include "kernels.h"
#include "model.h"
#include "quant.h"

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The kernels live below as PTX the driver JITs, so no toolkit is needed at
   build time and any compute capability runs. .target sm_50 is the oldest
   this code cares about; the driver JITs it forward. */

#define GEMM_BLOCK 256          /* 8 warps, one output row each */
#define GEMM_ROWS_PER_BLOCK 8
#define GEMM_TILE_MAX 8
#define GEMM_TILE_COUNT 4       /* 1, 2, 4, 8 */
/* The q4_k scale staging: 16 floats (8 scales, 8 negated mins) per warp. */
#define GEMM_SCALE_SHARED 512

#define SCORE_BLOCK 128
#define SCORE_TILE 8            /* positions staged per block */
#define SOFTMAX_BLOCK 128
#define FOLD_BLOCK 128
#define FOLD_R_TILE 64

#define LOG2_E 1.44269504088896f

struct CudaDevice;

typedef int (*cu_init_t)(unsigned);
typedef int (*cu_dev_get_t)(int *, int);
typedef int (*cu_ctx_create_t)(void **, unsigned, int);
typedef int (*cu_ctx_destroy_t)(void *);
typedef int (*cu_mem_alloc_t)(unsigned long long *, size_t);
typedef int (*cu_mem_free_t)(unsigned long long);
typedef int (*cu_mem_info_t)(unsigned long long *, unsigned long long *);
typedef int (*cu_memcpy_htd_t)(unsigned long long, const void *, size_t);
typedef int (*cu_memcpy_dth_t)(void *, unsigned long long, size_t);
typedef int (*cu_mem_host_register_t)(void *, size_t, unsigned);
typedef int (*cu_mem_host_unregister_t)(void *);
typedef int (*cu_module_load_t)(void **, const void *);
typedef int (*cu_func_get_t)(void **, void *, const char *);
typedef int (*cu_dev_attr_t)(int *, int, int);
typedef int (*cu_occupancy_t)(int *, void *, int, size_t);
typedef int (*cu_launch_t)(void *, unsigned, unsigned, unsigned, unsigned,
                           unsigned, unsigned, unsigned, void *, void **,
                           void **);
typedef int (*cu_ctx_sync_t)(void);
typedef int (*cu_err_str_t)(int, const char **);

/* The calls on the per-token path are resolved once here; a dlsym per copy
   or launch is small but pointless, and caching keeps the hot wrappers to
   one pointer dereference. */
struct CudaDevice {
    void *cuda;
    void *ctx;
    void *module;
    void *gemm_fn[GEMM_TILE_COUNT][2];
    void *score_fn;
    void *softmax_fn;
    void *fold_fn;
    void *q_prep_fn;
    void *kv_prep_fn;
    cu_mem_alloc_t mem_alloc;
    cu_mem_free_t mem_free;
    cu_mem_info_t mem_info;
    cu_memcpy_htd_t memcpy_htd;
    cu_memcpy_dth_t memcpy_dth;
    cu_mem_host_register_t host_register;
    cu_mem_host_unregister_t host_unregister;
    cu_launch_t launch_kernel;
    cu_ctx_sync_t ctx_sync;
    cu_err_str_t err_str;
    cu_dev_attr_t dev_attr;
    cu_occupancy_t occupancy;
    int device;
    int n_sm;
    /* How many blocks of each gemm tile variant the device keeps resident;
       the launcher never launches more, so the kernel's row-tile loop takes
       the rest. */
    int gemm_max_blocks[GEMM_TILE_COUNT][2];
    char fault[192];
};

const char *cuda_fault(const CudaDevice *dev) { return dev->fault[0] ? dev->fault : NULL; }

#define FAULT(dev, ...)                                                       \
    do {                                                                       \
        if (!(dev)->fault[0])                                                  \
            snprintf((dev)->fault, sizeof (dev)->fault, __VA_ARGS__);          \
    } while (0)

static void *sym(void *lib, const char *name) {
    return lib ? dlsym(lib, name) : NULL;
}

/* =============================================================== */
/* geode_gemm_t{1,2,4,8}: out[token][row] = W_row . x[token].

   One entry per token-tile width, generated, so the token loop is unrolled
   into named accumulators -- a dynamic-width loop would spill them to local
   memory and put a load in front of every accumulate.

   One warp computes one output row. The q4_k dot is shaped around the
   format's own geometry so the issue stream, not the encoding, sets the
   pace on Maxwell, where there is no int8 SIMD and a nibble-per-element
   kernel measured instruction-bound at ~17 instructions per element:
   each lane owns the eight nibbles of four consecutive block bytes,
   which are the same four positions of two adjacent 32-wide sub-blocks,
   so the values arrive in one 32-bit load and the per-element scale
   multiply folds into a per-sub-block epilogue --
   out += s*sum(nib*x) - m*sum(x), the x sums kept in registers from
   the same vector loads the dot consumes. x streams from global
   through L2 (a tile is at most 64 KB, resident) rather than being
   re-staged to shared by every block; shared holds only the warp's 16
   scale floats, unpacked once per 256-block. */

/* Per token `t`: eight f32 weights times eight x values read straight
   from global; the token's running x pointer steps one 256-element row
   chunk. */
static void gemm_f32_token(FILE *out, int t) {
    fprintf(out,
            "        ld.global.v4.f32 {%%f8,%%f9,%%f10,%%f11}, [%%rd%d];\n"
            "        ld.global.v4.f32 {%%f12,%%f13,%%f14,%%f15}, [%%rd%d+16];\n"
            "        fma.rn.f32 %%f%d, %%f0, %%f8, %%f%d;\n"
            "        fma.rn.f32 %%f%d, %%f1, %%f9, %%f%d;\n"
            "        fma.rn.f32 %%f%d, %%f2, %%f10, %%f%d;\n"
            "        fma.rn.f32 %%f%d, %%f3, %%f11, %%f%d;\n"
            "        fma.rn.f32 %%f%d, %%f4, %%f12, %%f%d;\n"
            "        fma.rn.f32 %%f%d, %%f5, %%f13, %%f%d;\n"
            "        fma.rn.f32 %%f%d, %%f6, %%f14, %%f%d;\n"
            "        fma.rn.f32 %%f%d, %%f7, %%f15, %%f%d;\n"
            "        add.s64 %%rd%d, %%rd%d, 1024;\n",
            16 + t, 16 + t,
            40 + t, 40 + t, 40 + t, 40 + t, 40 + t, 40 + t, 40 + t, 40 + t,
            40 + t, 40 + t, 40 + t, 40 + t, 40 + t, 40 + t, 40 + t, 40 + t,
            16 + t, 16 + t);
}

/* Eight fmas: weights w0..w0+7 against the x values f8..f15. */
static void gemm_fma8(FILE *out, int acc, int w0) {
    for (int i = 0; i < 8; i++)
        fprintf(out,
                "        fma.rn.f32 %%f%d, %%f%d, %%f%d, %%f%d;\n",
                acc, w0 + i, 8 + i, acc);
}

/* Per token `t` for the two-row kernels: one set of x loads feeds both
   sub-rows' accumulators -- the pair shares a head, so the x segment is
   identical. Sub-row 0's weights sit in f0-f7 (f32) or were converted
   there (f16); sub-row 1's in w0 passed by the caller. */
static void gemm_token2(FILE *out, int t, int w0) {
    fprintf(out,
            "        ld.global.v4.f32 {%%f8,%%f9,%%f10,%%f11}, [%%rd%d];\n"
            "        ld.global.v4.f32 {%%f12,%%f13,%%f14,%%f15}, [%%rd%d+16];\n",
            16 + t, 16 + t);
    gemm_fma8(out, 40 + 2 * t, 0);
    gemm_fma8(out, 41 + 2 * t, w0);
    fprintf(out,
            "        add.s64 %%rd%d, %%rd%d, 1024;\n",
            16 + t, 16 + t);
}

/* Per token `t`: eight nibble weights against the lane's eight x
   values, then the two sub-block folds. f8/f9 hold the raw nibble
   dots; f19/f20 the scales, f21/f22 the negated mins, f23/f24 the x
   sums the folds close over. The block's contribution lands in a
   temp first so the accumulator's serial chain is one add per
   block, not a four-deep fma stack. */
static void gemm_q4_token(FILE *out, int t, int tile) {
    (void)tile;
    fprintf(out,
            "        ld.global.v4.f32 {%%f10,%%f11,%%f12,%%f13}, [%%rd%d];\n"
            "        ld.global.v4.f32 {%%f14,%%f15,%%f16,%%f17}, [%%rd%d+128];\n"
            "        prmt.b32 %%r65, %%r46, %%r63, 0x6540;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        mul.rn.f32 %%f8, %%f18, %%f10;\n"
            "        prmt.b32 %%r65, %%r46, %%r63, 0x6541;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f8, %%f18, %%f11, %%f8;\n"
            "        prmt.b32 %%r65, %%r46, %%r63, 0x6542;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f8, %%f18, %%f12, %%f8;\n"
            "        prmt.b32 %%r65, %%r46, %%r63, 0x6543;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f8, %%f18, %%f13, %%f8;\n"
            "        prmt.b32 %%r65, %%r47, %%r63, 0x6540;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        mul.rn.f32 %%f9, %%f18, %%f14;\n"
            "        prmt.b32 %%r65, %%r47, %%r63, 0x6541;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f9, %%f18, %%f15, %%f9;\n"
            "        prmt.b32 %%r65, %%r47, %%r63, 0x6542;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f9, %%f18, %%f16, %%f9;\n"
            "        prmt.b32 %%r65, %%r47, %%r63, 0x6543;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f9, %%f18, %%f17, %%f9;\n"
            "        add.f32 %%f23, %%f10, %%f11;\n"
            "        add.f32 %%f24, %%f12, %%f13;\n"
            "        add.f32 %%f23, %%f23, %%f24;\n"
            "        add.f32 %%f24, %%f14, %%f15;\n"
            "        add.f32 %%f25, %%f16, %%f17;\n"
            "        add.f32 %%f24, %%f24, %%f25;\n"
            "        mul.f32 %%f26, %%f8, %%f19;\n"
            "        fma.rn.f32 %%f26, %%f23, %%f21, %%f26;\n"
            "        fma.rn.f32 %%f26, %%f9, %%f20, %%f26;\n"
            "        fma.rn.f32 %%f26, %%f24, %%f22, %%f26;\n"
            "        add.f32 %%f%d, %%f%d, %%f26;\n"
            "        add.s64 %%rd%d, %%rd%d, 1024;\n",
            16 + t, 16 + t,
            40 + t, 40 + t, 16 + t, 16 + t);
}

/* Two-row q4 body: sub-row 0 works from r46/r47 and f19-f22 as above,
   then sub-row 1 repeats the dot against r5/r6 (its unpacked nibbles)
   and f27-f30 (its scales) -- the x loads and x sums are shared. */
static void gemm_q4_fold(FILE *out, int acc, int nlo, int nhi, int sc0,
                         int sc1, int mn0, int mn1) {
    fprintf(out,
            "        prmt.b32 %%r65, %%r%d, %%r63, 0x6540;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        mul.rn.f32 %%f8, %%f18, %%f10;\n"
            "        prmt.b32 %%r65, %%r%d, %%r63, 0x6541;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f8, %%f18, %%f11, %%f8;\n"
            "        prmt.b32 %%r65, %%r%d, %%r63, 0x6542;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f8, %%f18, %%f12, %%f8;\n"
            "        prmt.b32 %%r65, %%r%d, %%r63, 0x6543;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f8, %%f18, %%f13, %%f8;\n"
            "        prmt.b32 %%r65, %%r%d, %%r63, 0x6540;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        mul.rn.f32 %%f9, %%f18, %%f14;\n"
            "        prmt.b32 %%r65, %%r%d, %%r63, 0x6541;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f9, %%f18, %%f15, %%f9;\n"
            "        prmt.b32 %%r65, %%r%d, %%r63, 0x6542;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f9, %%f18, %%f16, %%f9;\n"
            "        prmt.b32 %%r65, %%r%d, %%r63, 0x6543;\n"
            "        cvt.rn.f32.u32 %%f18, %%r65;\n"
            "        fma.rn.f32 %%f9, %%f18, %%f17, %%f9;\n"
            "        mul.f32 %%f26, %%f8, %%f%d;\n"
            "        fma.rn.f32 %%f26, %%f23, %%f%d, %%f26;\n"
            "        fma.rn.f32 %%f26, %%f9, %%f%d, %%f26;\n"
            "        fma.rn.f32 %%f26, %%f24, %%f%d, %%f26;\n"
            "        add.f32 %%f%d, %%f%d, %%f26;\n",
            nlo, nlo, nlo, nlo, nhi, nhi, nhi, nhi,
            sc0, mn0, sc1, mn1, acc, acc);
}

static void gemm_q4_token2(FILE *out, int t) {
    fprintf(out,
            "        ld.global.v4.f32 {%%f10,%%f11,%%f12,%%f13}, [%%rd%d];\n"
            "        ld.global.v4.f32 {%%f14,%%f15,%%f16,%%f17}, [%%rd%d+128];\n"
            "        add.f32 %%f23, %%f10, %%f11;\n"
            "        add.f32 %%f24, %%f12, %%f13;\n"
            "        add.f32 %%f23, %%f23, %%f24;\n"
            "        add.f32 %%f24, %%f14, %%f15;\n"
            "        add.f32 %%f25, %%f16, %%f17;\n"
            "        add.f32 %%f24, %%f24, %%f25;\n",
            16 + t, 16 + t);
    gemm_q4_fold(out, 40 + 2 * t, 46, 47, 19, 20, 21, 22);
    gemm_q4_fold(out, 41 + 2 * t, 5, 6, 27, 28, 29, 30);
    fprintf(out,
            "        add.s64 %%rd%d, %%rd%d, 1024;\n",
            16 + t, 16 + t);
}

/* Per token `t`: the running x pointer, seeded from the token's
   segment (row stride, head stride) plus the lane's byte offset held
   in rd3. Tokens past the tile's valid range clamp to token 0: the
   loads must stay in bounds even though the stores are guarded. */
static void gemm_token_ptr(FILE *out, int t) {
    fprintf(out,
            "        add.u32 %%r24, %%r19, %d;\n"
            "        setp.lt.s32 %%p2, %%r24, %%r54;\n"
            "        selp.b32 %%r24, %%r24, 0, %%p2;\n"
            "        mul.wide.u32 %%rd1, %%r24, %%r51;\n"
            "        mul.wide.u32 %%rd2, %%r17, %%r52;\n"
            "        add.s64 %%rd1, %%rd1, %%rd2;\n"
            "        shl.b64 %%rd1, %%rd1, 2;\n"
            "        add.s64 %%rd1, %%rd1, %%rd3;\n"
            "        add.s64 %%rd%d, %%rd11, %%rd1;\n",
            t, 16 + t);
}

/* geode_gemm_t{tile}_r{rows}: out[token][row] = W_row . x[token].

   One entry per token-tile width (unrolled accumulators) and per
   row-pairing width r. r=2 doubles the bytes in flight per warp and
   halves the per-row latency chain -- the kernel would otherwise pay
   a full DRAM round trip per 256-column chunk of every row, and the
   memory system sits idle while it waits -- and it shares the x
   loads between the two rows, which always belong to the same head
   (16-row groups need head_out % 16 == 0 when n_head > 1; the
   launcher falls back to the r=1 kernels otherwise). A row-pair's
   second row can be past the matrix end on a partial tile; p0 marks
   that case and gates the second row's loads and stores. */
static void gemm_entry(FILE *out, int tile, int r) {
    fprintf(out,
            ".visible .entry geode_gemm_t%d_r%d(\n"
            "    .param .u64 p_w, .param .u64 p_x, .param .u64 p_out,\n"
            "    .param .u32 p_n_in, .param .u32 p_head_out,\n"
            "    .param .u32 p_n_head, .param .u32 p_x_stride,\n"
            "    .param .u32 p_x_head_stride, .param .u32 p_out_stride,\n"
            "    .param .u32 p_n_tokens, .param .u32 p_type)\n"
            "{\n"
            "    .reg .pred %%p<8>;\n"
            "    .reg .f32 %%f<56>;\n"
            "    .reg .b16 %%rs<17>;\n"
            "    .reg .u32 %%r<68>;\n"
            "    .reg .u64 %%rd<24>;\n"
            "    .shared .align 16 .b8 geode_shm[%d];\n"
            "    ld.param.u64 %%rd10, [p_w];\n"
            "    ld.param.u64 %%rd11, [p_x];\n"
            "    ld.param.u64 %%rd12, [p_out];\n"
            "    ld.param.u32 %%r48, [p_n_in];\n"
            "    ld.param.u32 %%r49, [p_head_out];\n"
            "    ld.param.u32 %%r50, [p_n_head];\n"
            "    ld.param.u32 %%r51, [p_x_stride];\n"
            "    ld.param.u32 %%r52, [p_x_head_stride];\n"
            "    ld.param.u32 %%r53, [p_out_stride];\n"
            "    ld.param.u32 %%r54, [p_n_tokens];\n"
            "    ld.param.u32 %%r55, [p_type];\n"
            "    mov.u32 %%r10, %%tid.x;\n"
            "    shr.u32 %%r11, %%r10, 5;\n"
            "    and.b32 %%r12, %%r10, 31;\n"
            "    mov.u32 %%r13, geode_shm;\n"
            "    mul.lo.u32 %%r16, %%r49, %%r50;\n"
            "    mov.u32 %%r19, %%ctaid.y;\n"
            "    mul.lo.u32 %%r19, %%r19, %d;\n"
            "    mov.u32 %%r63, 0;\n"
            "    shl.b32 %%r22, %%r11, %d;\n"
            "    add.u32 %%r32, %%r13, %%r22;\n"
            "    mov.u32 %%r64, %%ctaid.x;\n",
            tile, r, r == 2 ? 1024 : 512, tile, r == 2 ? 7 : 6);
    fprintf(out,
            "    $L_row:\n"
            "    shl.b32 %%r14, %%r64, %d;\n",
            3 + (r == 2));
    if (r == 2)
        fprintf(out,
                "    shl.b32 %%r25, %%r11, 1;\n"
                "    add.u32 %%r15, %%r14, %%r25;\n"
                "    add.u32 %%r24, %%r15, 1;\n"
                "    setp.lt.s32 %%p0, %%r24, %%r16;\n");
    else
        fprintf(out, "    add.u32 %%r15, %%r14, %%r11;\n");
    fprintf(out,
            "    setp.ge.s32 %%p5, %%r15, %%r16;\n"
            "    @%%p5 bra $L_row_next;\n"
            "    div.u32 %%r17, %%r14, %%r49;\n"
            "    setp.eq.s32 %%p6, %%r55, 1;\n");
    for (int t = 0; t < tile; t++)
        for (int k = 0; k < r; k++)
            fprintf(out, "    mov.f32 %%f%d, 0f00000000;\n", 40 + t * r + k);
    fprintf(out,
            "    @%%p6 bra $L_q4;\n"
            "    setp.eq.s32 %%p7, %%r55, 2;\n"
            "    @%%p7 bra $L_f16;\n"
            "$L_f32:\n"
            "    shl.b32 %%r26, %%r48, 2;\n"
            "    shl.b32 %%r27, %%r12, 3;\n"
            "    mul.wide.u32 %%rd13, %%r15, %%r26;\n"
            "    add.s64 %%rd13, %%rd13, %%rd10;\n"
            "    cvt.u64.u32 %%rd3, %%r27;\n"
            "    shl.b64 %%rd3, %%rd3, 2;\n"
            "    add.s64 %%rd13, %%rd13, %%rd3;\n");
    if (r == 2)
        fprintf(out,
                "    cvt.u64.u32 %%rd1, %%r26;\n"
                "    add.s64 %%rd0, %%rd13, %%rd1;\n");
    for (int t = 0; t < tile; t++) gemm_token_ptr(out, t);
    fprintf(out,
            "$L_f32_loop:\n"
            "    setp.ge.s32 %%p1, %%r27, %%r48;\n"
            "    @%%p1 bra $L_f32_done;\n"
            "    ld.global.v4.f32 {%%f0,%%f1,%%f2,%%f3}, [%%rd13];\n"
            "    ld.global.v4.f32 {%%f4,%%f5,%%f6,%%f7}, [%%rd13+16];\n");
    if (r == 2)
        fprintf(out,
                "    @%%p0 ld.global.v4.f32 {%%f24,%%f25,%%f26,%%f27}, [%%rd0];\n"
                "    @%%p0 ld.global.v4.f32 {%%f28,%%f29,%%f30,%%f31}, [%%rd0+16];\n");
    for (int t = 0; t < tile; t++)
        if (r == 2) gemm_token2(out, t, 24);
        else gemm_f32_token(out, t);
    fprintf(out,
            "    add.s64 %%rd13, %%rd13, 1024;\n");
    if (r == 2) fprintf(out, "    add.s64 %%rd0, %%rd0, 1024;\n");
    fprintf(out,
            "    add.u32 %%r27, %%r27, 256;\n"
            "    bra $L_f32_loop;\n"
            "$L_f32_done:\n"
            "    bra $L_reduce;\n"
            "$L_f16:\n"
            "    shl.b32 %%r26, %%r48, 1;\n"
            "    shl.b32 %%r27, %%r12, 3;\n"
            "    mul.wide.u32 %%rd13, %%r15, %%r26;\n"
            "    add.s64 %%rd13, %%rd13, %%rd10;\n"
            "    cvt.u64.u32 %%rd2, %%r27;\n"
            "    shl.b64 %%rd2, %%rd2, 1;\n"
            "    add.s64 %%rd13, %%rd13, %%rd2;\n"
            "    cvt.u64.u32 %%rd3, %%r27;\n"
            "    shl.b64 %%rd3, %%rd3, 2;\n");
    if (r == 2)
        fprintf(out,
                "    cvt.u64.u32 %%rd1, %%r26;\n"
                "    add.s64 %%rd0, %%rd13, %%rd1;\n");
    for (int t = 0; t < tile; t++) gemm_token_ptr(out, t);
    fprintf(out,
            "$L_f16_loop:\n"
            "    setp.ge.s32 %%p1, %%r27, %%r48;\n"
            "    @%%p1 bra $L_f16_done;\n"
            "    ld.global.v4.b16 {%%rs1,%%rs2,%%rs3,%%rs4}, [%%rd13];\n"
            "    ld.global.v4.b16 {%%rs5,%%rs6,%%rs7,%%rs8}, [%%rd13+8];\n");
    if (r == 2)
        fprintf(out,
                "    @%%p0 ld.global.v4.b16 {%%rs9,%%rs10,%%rs11,%%rs12}, [%%rd0];\n"
                "    @%%p0 ld.global.v4.b16 {%%rs13,%%rs14,%%rs15,%%rs16}, [%%rd0+8];\n"
                "    cvt.f32.f16 %%f16, %%rs9;\n"
                "    cvt.f32.f16 %%f17, %%rs10;\n"
                "    cvt.f32.f16 %%f18, %%rs11;\n"
                "    cvt.f32.f16 %%f19, %%rs12;\n"
                "    cvt.f32.f16 %%f20, %%rs13;\n"
                "    cvt.f32.f16 %%f21, %%rs14;\n"
                "    cvt.f32.f16 %%f22, %%rs15;\n"
                "    cvt.f32.f16 %%f23, %%rs16;\n");
    fprintf(out,
            "    cvt.f32.f16 %%f0, %%rs1;\n"
            "    cvt.f32.f16 %%f1, %%rs2;\n"
            "    cvt.f32.f16 %%f2, %%rs3;\n"
            "    cvt.f32.f16 %%f3, %%rs4;\n"
            "    cvt.f32.f16 %%f4, %%rs5;\n"
            "    cvt.f32.f16 %%f5, %%rs6;\n"
            "    cvt.f32.f16 %%f6, %%rs7;\n"
            "    cvt.f32.f16 %%f7, %%rs8;\n");
    for (int t = 0; t < tile; t++)
        if (r == 2) gemm_token2(out, t, 16);
        else gemm_f32_token(out, t);
    fprintf(out,
            "    add.s64 %%rd13, %%rd13, 512;\n");
    if (r == 2) fprintf(out, "    add.s64 %%rd0, %%rd0, 512;\n");
    fprintf(out,
            "    add.u32 %%r27, %%r27, 256;\n"
            "    bra $L_f16_loop;\n"
            "$L_f16_done:\n"
            "    bra $L_reduce;\n"
            "$L_q4:\n"
            "    shr.u32 %%r30, %%r48, 8;\n"
            "    mul.lo.u32 %%r25, %%r30, 144;\n"
            "    mul.wide.u32 %%rd13, %%r15, %%r25;\n"
            "    add.s64 %%rd13, %%rd13, %%rd10;\n"
            "    shr.u32 %%r31, %%r12, 3;\n"
            "    and.b32 %%r33, %%r12, 7;\n"
            "    shl.b32 %%r34, %%r33, 2;\n"
            "    shl.b32 %%r40, %%r31, 6;\n"
            "    add.u32 %%r40, %%r40, %%r34;\n"
            "    shl.b32 %%r41, %%r40, 2;\n"
            "    shl.b32 %%r35, %%r31, 5;\n"
            "    add.u32 %%r35, %%r35, %%r34;\n"
            "    add.u32 %%r35, %%r35, 16;\n"
            "    cvt.u64.u32 %%rd3, %%r35;\n"
            "    add.s64 %%rd15, %%rd13, %%rd3;\n"
            "    shl.b32 %%r36, %%r31, 1;\n"
            "    shl.b32 %%r37, %%r36, 2;\n"
            "    add.u32 %%r38, %%r32, %%r37;\n"
            "    mov.u32 %%r39, 0;\n"
            "    cvt.u64.u32 %%rd3, %%r41;\n");
    if (r == 2)
        fprintf(out,
                "    cvt.u64.u32 %%rd1, %%r25;\n"
                "    add.s64 %%rd0, %%rd13, %%rd1;\n"
                "    cvt.u64.u32 %%rd7, %%r35;\n"
                "    add.s64 %%rd7, %%rd0, %%rd7;\n");
    for (int t = 0; t < tile; t++) gemm_token_ptr(out, t);
    fprintf(out,
            "$L_qb:\n"
            "    setp.ge.s32 %%p1, %%r39, %%r30;\n"
            "    @%%p1 bra $L_q_done;\n"
            "    ld.global.u32 %%r45, [%%rd15];\n"
            "    ld.global.b16 %%rs1, [%%rd13];\n"
            "    ld.global.b16 %%rs2, [%%rd13+2];\n"
            "    cvt.f32.f16 %%f34, %%rs1;\n"
            "    cvt.f32.f16 %%f35, %%rs2;\n");
    if (r == 2)
        fprintf(out,
                "    @%%p0 ld.global.u32 %%r66, [%%rd7];\n"
                "    @%%p0 ld.global.b16 %%rs3, [%%rd0];\n"
                "    @%%p0 ld.global.b16 %%rs4, [%%rd0+2];\n"
                "    cvt.f32.f16 %%f36, %%rs3;\n"
                "    cvt.f32.f16 %%f37, %%rs4;\n");
    fprintf(out,
            "    add.s64 %%rd14, %%rd13, 4;\n"
            "    setp.lt.s32 %%p3, %%r12, 8;\n"
            "    @!%%p3 bra $L_noscale;\n"
            "    cvt.u64.u32 %%rd1, %%r12;\n"
            "    add.s64 %%rd1, %%rd14, %%rd1;\n"
            "    ld.global.u8 %%r1, [%%rd1];\n"
            "    ld.global.u8 %%r2, [%%rd1+4];\n"
            "    add.s64 %%rd2, %%rd1, -4;\n"
            "    ld.global.u8 %%r3, [%%rd2];\n"
            "    setp.lt.s32 %%p4, %%r12, 4;\n"
            "    and.b32 %%r4, %%r1, 63;\n"
            "    and.b32 %%r5, %%r2, 63;\n"
            "    and.b32 %%r6, %%r2, 15;\n"
            "    shr.u32 %%r7, %%r3, 6;\n"
            "    shl.b32 %%r7, %%r7, 4;\n"
            "    or.b32 %%r6, %%r6, %%r7;\n"
            "    shr.u32 %%r7, %%r2, 4;\n"
            "    shr.u32 %%r8, %%r1, 6;\n"
            "    shl.b32 %%r8, %%r8, 4;\n"
            "    or.b32 %%r8, %%r7, %%r8;\n"
            "    selp.b32 %%r4, %%r4, %%r6, %%p4;\n"
            "    selp.b32 %%r5, %%r5, %%r8, %%p4;\n"
            "    cvt.rn.f32.u32 %%f18, %%r4;\n"
            "    cvt.rn.f32.u32 %%f19, %%r5;\n"
            "    mul.f32 %%f18, %%f18, %%f34;\n"
            "    mul.f32 %%f19, %%f19, %%f35;\n"
            "    sub.f32 %%f19, 0f00000000, %%f19;\n"
            "    shl.b32 %%r7, %%r12, 2;\n"
            "    add.u32 %%r8, %%r32, %%r7;\n"
            "    add.u32 %%r9, %%r8, 32;\n"
            "    st.shared.f32 [%%r8], %%f18;\n"
            "    st.shared.f32 [%%r9], %%f19;\n");
    if (r == 2)
        fprintf(out,
                "    setp.lt.and.s32 %%p2, %%r12, 8, %%p0;\n"
                "    add.s64 %%rd14, %%rd0, 4;\n"
                "    cvt.u64.u32 %%rd1, %%r12;\n"
                "    add.s64 %%rd1, %%rd14, %%rd1;\n"
                "    @%%p2 ld.global.u8 %%r1, [%%rd1];\n"
                "    @%%p2 ld.global.u8 %%r2, [%%rd1+4];\n"
                "    add.s64 %%rd2, %%rd1, -4;\n"
                "    @%%p2 ld.global.u8 %%r3, [%%rd2];\n"
                "    setp.lt.s32 %%p4, %%r12, 4;\n"
                "    and.b32 %%r4, %%r1, 63;\n"
                "    and.b32 %%r5, %%r2, 63;\n"
                "    and.b32 %%r6, %%r2, 15;\n"
                "    shr.u32 %%r7, %%r3, 6;\n"
                "    shl.b32 %%r7, %%r7, 4;\n"
                "    or.b32 %%r6, %%r6, %%r7;\n"
                "    shr.u32 %%r7, %%r2, 4;\n"
                "    shr.u32 %%r8, %%r1, 6;\n"
                "    shl.b32 %%r8, %%r8, 4;\n"
                "    or.b32 %%r8, %%r7, %%r8;\n"
                "    selp.b32 %%r4, %%r4, %%r6, %%p4;\n"
                "    selp.b32 %%r5, %%r5, %%r8, %%p4;\n"
                "    cvt.rn.f32.u32 %%f18, %%r4;\n"
                "    cvt.rn.f32.u32 %%f19, %%r5;\n"
                "    mul.f32 %%f18, %%f18, %%f36;\n"
                "    mul.f32 %%f19, %%f19, %%f37;\n"
                "    sub.f32 %%f19, 0f00000000, %%f19;\n"
                "    shl.b32 %%r7, %%r12, 2;\n"
                "    add.u32 %%r8, %%r32, %%r7;\n"
                "    add.u32 %%r8, %%r8, 64;\n"
                "    add.u32 %%r9, %%r8, 32;\n"
                "    @%%p2 st.shared.f32 [%%r8], %%f18;\n"
                "    @%%p2 st.shared.f32 [%%r9], %%f19;\n");
    fprintf(out,
            "$L_noscale:\n"
            "    bar.warp.sync 0xffffffff;\n"
            "    and.b32 %%r46, %%r45, 0x0F0F0F0F;\n"
            "    shr.u32 %%r47, %%r45, 4;\n"
            "    and.b32 %%r47, %%r47, 0x0F0F0F0F;\n");
    if (r == 2)
        fprintf(out,
                "    and.b32 %%r5, %%r66, 0x0F0F0F0F;\n"
                "    shr.u32 %%r6, %%r66, 4;\n"
                "    and.b32 %%r6, %%r6, 0x0F0F0F0F;\n");
    fprintf(out,
            "    ld.shared.f32 %%f19, [%%r38];\n"
            "    ld.shared.f32 %%f20, [%%r38+4];\n"
            "    ld.shared.f32 %%f21, [%%r38+32];\n"
            "    ld.shared.f32 %%f22, [%%r38+36];\n");
    if (r == 2)
        fprintf(out,
                "    ld.shared.f32 %%f27, [%%r38+64];\n"
                "    ld.shared.f32 %%f28, [%%r38+68];\n"
                "    ld.shared.f32 %%f29, [%%r38+96];\n"
                "    ld.shared.f32 %%f30, [%%r38+100];\n");
    for (int t = 0; t < tile; t++)
        if (r == 2) gemm_q4_token2(out, t);
        else gemm_q4_token(out, t, tile);
    fprintf(out,
            "    add.s64 %%rd13, %%rd13, 144;\n"
            "    add.s64 %%rd15, %%rd15, 144;\n");
    if (r == 2)
        fprintf(out,
                "    add.s64 %%rd0, %%rd0, 144;\n"
                "    add.s64 %%rd7, %%rd7, 144;\n");
    fprintf(out,
            "    add.u32 %%r39, %%r39, 1;\n"
            "    bra $L_qb;\n"
            "$L_q_done:\n"
            "$L_reduce:\n");
    fprintf(out,
            "    mov.u32 %%r56, 1;\n"
            "    mov.u32 %%r57, 2;\n"
            "    mov.u32 %%r58, 4;\n"
            "    mov.u32 %%r59, 8;\n"
            "    mov.u32 %%r60, 16;\n");
    for (int i = 0; i < tile * r; i++) {
        fprintf(out, "    mov.b32 %%r61, %%f%d;\n", 40 + i);
        for (int round = 0; round < 5; round++)
            fprintf(out,
                    "    shfl.bfly.b32 %%r62, %%r61, %%r%d, 31;\n"
                    "    mov.b32 %%f39, %%r62;\n"
                    "    add.f32 %%f%d, %%f%d, %%f39;\n"
                    "    mov.b32 %%r61, %%f%d;\n",
                    56 + round, 40 + i, 40 + i, 40 + i);
    }
    fprintf(out,
            "    setp.ne.s32 %%p7, %%r12, 0;\n"
            "    @%%p7 bra $L_row_next;\n");
    for (int t = 0; t < tile; t++)
        for (int k = 0; k < r; k++) {
            fprintf(out,
                    "    add.u32 %%r24, %%r19, %d;\n"
                    "    setp.ge.s32 %%p1, %%r24, %%r54;\n"
                    "    @%%p1 bra $L_s%d_%d;\n",
                    t, t, k);
            if (k == 1)
                fprintf(out, "    @!%%p0 bra $L_s%d_%d;\n", t, k);
            if (k == 0)
                fprintf(out, "    cvt.u64.u32 %%rd2, %%r15;\n");
            else
                fprintf(out,
                        "    add.u32 %%r25, %%r15, 1;\n"
                        "    cvt.u64.u32 %%rd2, %%r25;\n");
            fprintf(out,
                    "    mul.wide.u32 %%rd1, %%r24, %%r53;\n"
                    "    add.s64 %%rd1, %%rd1, %%rd2;\n"
                    "    shl.b64 %%rd1, %%rd1, 2;\n"
                    "    add.s64 %%rd1, %%rd1, %%rd12;\n"
                    "    st.global.f32 [%%rd1], %%f%d;\n"
                    "$L_s%d_%d:\n",
                    40 + t * r + k, t, k);
        }
    fprintf(out,
            "$L_row_next:\n"
            "    mov.u32 %%r65, %%nctaid.x;\n"
            "    add.u32 %%r64, %%r64, %%r65;\n"
            "    shl.b32 %%r65, %%r64, %d;\n"
            "    setp.ge.s32 %%p5, %%r65, %%r16;\n"
            "    @%%p5 bra $L_ret;\n"
            "    bra $L_row;\n"
            "$L_ret:\n"
            "    ret;\n"
            "}\n",
            3 + (r == 2));
}

/* =============================================================== */
/* geode_attn_score: every (head, position) dot against the cache, one
   position-tile block at a time. The tile's slots are dequantized to f32
   in shared ONCE and shared by all heads -- a block per (head, position)
   would re-read the whole cache once per head, which is the 32x read
   amplification the cpu path avoids by keeping heads inner. */

static const char attn_score_ptx[] =
    ".visible .entry geode_attn_score(\n"
    "    .param .u64 p_cache, .param .u64 p_query, .param .u64 p_qlat,\n"
    "    .param .u64 p_scores, .param .u32 p_position, .param .u32 p_ntok,\n"
    "    .param .u64 p_geom)\n"
    "{\n"
    "    .reg .pred %p<6>;\n"
    "    .reg .f32 %f<24>;\n"
    "    .reg .u32 %r<44>;\n"
    "    .reg .u64 %rd<20>;\n"
    "    .shared .align 16 .b8 geode_shm[32768];\n"
    "    ld.param.u64 %rd10, [p_cache];\n"
    "    ld.param.u64 %rd11, [p_query];\n"
    "    ld.param.u64 %rd12, [p_qlat];\n"
    "    ld.param.u64 %rd13, [p_scores];\n"
    "    ld.param.u32 %r21, [p_position];\n"
    "    ld.param.u32 %r22, [p_ntok];\n"
    "    ld.param.u64 %rd14, [p_geom];\n"
    "    ld.global.u32 %r10, [%rd14+0];\n"
    "    ld.global.u32 %r11, [%rd14+4];\n"
    "    ld.global.u32 %r12, [%rd14+8];\n"
    "    ld.global.u32 %r13, [%rd14+12];\n"
    "    ld.global.u32 %r14, [%rd14+16];\n"
    "    ld.global.u32 %r15, [%rd14+20];\n"
    "    ld.global.u32 %r16, [%rd14+28];\n"
    "    ld.global.u32 %r17, [%rd14+32];\n"
    "    ld.global.u32 %r18, [%rd14+36];\n"
    "    ld.global.u32 %r19, [%rd14+40];\n"
    "    ld.global.f32 %f10, [%rd14+52];\n"
    "    ld.global.u32 %r20, [%rd14+56];\n"
    "    mov.u32 %r23, %tid.x;\n"
    "    mov.u32 %r24, geode_shm;\n"
    "    mov.u32 %r25, %ctaid.y;\n"
    "    add.u32 %r26, %r21, %r25;\n"
    "    add.u32 %r26, %r26, 1;\n"
    "    mov.u32 %r27, %ctaid.x;\n"
    "    shl.b32 %r27, %r27, 3;\n"
    "    setp.eq.s32 %p5, %r10, 0;\n"
    "    @%p5 bra $L_mla;\n"
    /* GQA: stage the tile's keys, one scale per kv head. */
    "    mul.lo.u32 %r29, %r12, %r13;\n"
    "    mov.u32 %r28, %r23;\n"
    "    mul.lo.u32 %r30, %r29, 8;\n"
    "$L_gq_stage:\n"
    "    setp.ge.s32 %p1, %r28, %r30;\n"
    "    @%p1 bra $L_gq_stage_done;\n"
    "    div.u32 %r31, %r28, %r29;\n"
    "    mul.lo.u32 %r32, %r31, %r29;\n"
    "    sub.u32 %r32, %r28, %r32;\n"
    "    add.u32 %r33, %r27, %r31;\n"
    "    setp.lt.s32 %p2, %r33, %r26;\n"
    "    mul.wide.u32 %rd1, %r33, %r17;\n"
    "    cvt.u64.u32 %rd2, %r32;\n"
    "    add.s64 %rd1, %rd1, %rd2;\n"
    "    add.s64 %rd1, %rd1, %rd10;\n"
    "    mov.u32 %r35, 128;\n"
    "    @%p2 ld.global.u8 %r35, [%rd1];\n"
    "    div.u32 %r34, %r32, %r13;\n"
    "    @!%p2 bra $L_gq_zero;\n"
    "    mul.wide.u32 %rd1, %r33, %r17;\n"
    "    mul.wide.u32 %rd2, %r34, 4;\n"
    "    add.s64 %rd1, %rd1, %rd2;\n"
    "    cvt.u64.u32 %rd3, %r16;\n"
    "    add.s64 %rd1, %rd1, %rd3;\n"
    "    add.s64 %rd1, %rd1, %rd10;\n"
    "    ld.global.f32 %f11, [%rd1];\n"
    "    bra $L_gq_scale;\n"
    "$L_gq_zero:\n"
    "    mov.f32 %f11, 0f00000000;\n"
    "$L_gq_scale:\n"
    "    cvt.rn.f32.u32 %f1, %r35;\n"
    "    sub.f32 %f1, %f1, 0f43000000;\n"
    "    mul.f32 %f1, %f1, %f11;\n"
    "    shl.b32 %r36, %r28, 2;\n"
    "    add.u32 %r36, %r24, %r36;\n"
    "    st.shared.f32 [%r36], %f1;\n"
    "    add.u32 %r28, %r28, 128;\n"
    "    bra $L_gq_stage;\n"
    "$L_gq_stage_done:\n"
    "    bar.sync 0;\n"
    "    mul.lo.u32 %r37, %r11, 8;\n"
    "    mov.u32 %r38, %r23;\n"
    "$L_gq_task:\n"
    "    setp.ge.s32 %p1, %r38, %r37;\n"
    "    @%p1 bra $L_ret;\n"
    "    div.u32 %r31, %r38, %r11;\n"
    "    rem.u32 %r39, %r38, %r11;\n"
    "    add.u32 %r33, %r27, %r31;\n"
    "    setp.ge.s32 %p2, %r33, %r26;\n"
    "    @%p2 bra $L_gq_next;\n"
    "    mul.lo.u32 %r40, %r39, %r12;\n"
    "    div.u32 %r40, %r40, %r11;\n"
    "    mul.wide.u32 %rd1, %r25, %r18;\n"
    "    mul.wide.u32 %rd2, %r39, %r13;\n"
    "    add.s64 %rd1, %rd1, %rd2;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.s64 %rd15, %rd11, %rd1;\n"
    "    mul.lo.u32 %r41, %r40, %r13;\n"
    "    mul.lo.u32 %r42, %r31, %r29;\n"
    "    add.u32 %r42, %r42, %r41;\n"
    "    mov.f32 %f0, 0f00000000;\n"
    "    mov.u32 %r28, 0;\n"
    "$L_gq_dot:\n"
    "    setp.ge.s32 %p3, %r28, %r13;\n"
    "    @%p3 bra $L_gq_dot_done;\n"
    "    ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd15];\n"
    "    add.u32 %r36, %r42, %r28;\n"
    "    shl.b32 %r36, %r36, 2;\n"
    "    add.u32 %r36, %r24, %r36;\n"
    "    ld.shared.v4.f32 {%f5,%f6,%f7,%f8}, [%r36];\n"
    "    fma.rn.f32 %f0, %f1, %f5, %f0;\n"
    "    fma.rn.f32 %f0, %f2, %f6, %f0;\n"
    "    fma.rn.f32 %f0, %f3, %f7, %f0;\n"
    "    fma.rn.f32 %f0, %f4, %f8, %f0;\n"
    "    add.s64 %rd15, %rd15, 16;\n"
    "    add.u32 %r28, %r28, 4;\n"
    "    bra $L_gq_dot;\n"
    "$L_gq_dot_done:\n"
    "    mul.f32 %f0, %f0, %f10;\n"
    "    mul.lo.u32 %r36, %r39, %r22;\n"
    "    add.u32 %r36, %r36, %r25;\n"
    "    mul.lo.u32 %r36, %r36, %r20;\n"
    "    add.u32 %r36, %r36, %r33;\n"
    "    cvt.u64.u32 %rd1, %r36;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.s64 %rd1, %rd1, %rd13;\n"
    "    st.global.f32 [%rd1], %f0;\n"
    "$L_gq_next:\n"
    "    add.u32 %r38, %r38, 128;\n"
    "    bra $L_gq_task;\n"
    /* MLA: stage latent + rotary, two scales. */
    "$L_mla:\n"
    "    mov.u32 %r28, %r23;\n"
    "    mul.lo.u32 %r30, %r16, 8;\n"
    "$L_mla_stage:\n"
    "    setp.ge.s32 %p1, %r28, %r30;\n"
    "    @%p1 bra $L_mla_stage_done;\n"
    "    div.u32 %r31, %r28, %r16;\n"
    "    mul.lo.u32 %r32, %r31, %r16;\n"
    "    sub.u32 %r32, %r28, %r32;\n"
    "    add.u32 %r33, %r27, %r31;\n"
    "    setp.lt.s32 %p2, %r33, %r26;\n"
    "    mov.u32 %r35, 128;\n"
    "    mov.f32 %f11, 0f00000000;\n"
    "    @!%p2 bra $L_mla_store;\n"
    "    mul.wide.u32 %rd1, %r33, %r17;\n"
    "    cvt.u64.u32 %rd2, %r32;\n"
    "    add.s64 %rd1, %rd1, %rd2;\n"
    "    add.s64 %rd1, %rd1, %rd10;\n"
    "    ld.global.u8 %r35, [%rd1];\n"
    "    setp.lt.s32 %p3, %r32, %r14;\n"
    "    mov.u32 %r34, 0;\n"
    "    @!%p3 mov.u32 %r34, 1;\n"
    "    mul.wide.u32 %rd1, %r33, %r17;\n"
    "    mul.wide.u32 %rd2, %r34, 4;\n"
    "    add.s64 %rd1, %rd1, %rd2;\n"
    "    cvt.u64.u32 %rd3, %r16;\n"
    "    add.s64 %rd1, %rd1, %rd3;\n"
    "    add.s64 %rd1, %rd1, %rd10;\n"
    "    ld.global.f32 %f11, [%rd1];\n"
    "$L_mla_store:\n"
    "    cvt.rn.f32.u32 %f1, %r35;\n"
    "    sub.f32 %f1, %f1, 0f43000000;\n"
    "    mul.f32 %f1, %f1, %f11;\n"
    "    shl.b32 %r36, %r28, 2;\n"
    "    add.u32 %r36, %r24, %r36;\n"
    "    st.shared.f32 [%r36], %f1;\n"
    "    add.u32 %r28, %r28, 128;\n"
    "    bra $L_mla_stage;\n"
    "$L_mla_stage_done:\n"
    "    bar.sync 0;\n"
    "    sub.u32 %r43, %r13, %r15;\n"
    "    mul.lo.u32 %r37, %r11, 8;\n"
    "    mov.u32 %r38, %r23;\n"
    "$L_mla_task:\n"
    "    setp.ge.s32 %p1, %r38, %r37;\n"
    "    @%p1 bra $L_ret;\n"
    "    div.u32 %r31, %r38, %r11;\n"
    "    rem.u32 %r39, %r38, %r11;\n"
    "    add.u32 %r33, %r27, %r31;\n"
    "    setp.ge.s32 %p2, %r33, %r26;\n"
    "    @%p2 bra $L_mla_next;\n"
    "    mul.wide.u32 %rd1, %r25, %r19;\n"
    "    mul.wide.u32 %rd2, %r39, %r14;\n"
    "    add.s64 %rd1, %rd1, %rd2;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.s64 %rd16, %rd12, %rd1;\n"
    "    mul.lo.u32 %r42, %r31, %r16;\n"
    "    mov.f32 %f0, 0f00000000;\n"
    "    mov.u32 %r28, 0;\n"
    "$L_mla_lat:\n"
    "    setp.ge.s32 %p3, %r28, %r14;\n"
    "    @%p3 bra $L_mla_lat_done;\n"
    "    ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd16];\n"
    "    add.u32 %r36, %r42, %r28;\n"
    "    shl.b32 %r36, %r36, 2;\n"
    "    add.u32 %r36, %r24, %r36;\n"
    "    ld.shared.v4.f32 {%f5,%f6,%f7,%f8}, [%r36];\n"
    "    fma.rn.f32 %f0, %f1, %f5, %f0;\n"
    "    fma.rn.f32 %f0, %f2, %f6, %f0;\n"
    "    fma.rn.f32 %f0, %f3, %f7, %f0;\n"
    "    fma.rn.f32 %f0, %f4, %f8, %f0;\n"
    "    add.s64 %rd16, %rd16, 16;\n"
    "    add.u32 %r28, %r28, 4;\n"
    "    bra $L_mla_lat;\n"
    "$L_mla_lat_done:\n"
    "    mul.wide.u32 %rd1, %r25, %r18;\n"
    "    mul.wide.u32 %rd2, %r39, %r13;\n"
    "    add.s64 %rd1, %rd1, %rd2;\n"
    "    cvt.u64.u32 %rd3, %r43;\n"
    "    add.s64 %rd1, %rd1, %rd3;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.s64 %rd15, %rd11, %rd1;\n"
    "    add.u32 %r42, %r42, %r14;\n"
    "    mov.f32 %f9, 0f00000000;\n"
    "    mov.u32 %r28, 0;\n"
    "$L_mla_rot:\n"
    "    setp.ge.s32 %p3, %r28, %r15;\n"
    "    @%p3 bra $L_mla_rot_done;\n"
    "    ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd15];\n"
    "    add.u32 %r36, %r42, %r28;\n"
    "    shl.b32 %r36, %r36, 2;\n"
    "    add.u32 %r36, %r24, %r36;\n"
    "    ld.shared.v4.f32 {%f5,%f6,%f7,%f8}, [%r36];\n"
    "    fma.rn.f32 %f9, %f1, %f5, %f9;\n"
    "    fma.rn.f32 %f9, %f2, %f6, %f9;\n"
    "    fma.rn.f32 %f9, %f3, %f7, %f9;\n"
    "    fma.rn.f32 %f9, %f4, %f8, %f9;\n"
    "    add.s64 %rd15, %rd15, 16;\n"
    "    add.u32 %r28, %r28, 4;\n"
    "    bra $L_mla_rot;\n"
    "$L_mla_rot_done:\n"
    "    add.f32 %f0, %f0, %f9;\n"
    "    mul.f32 %f0, %f0, %f10;\n"
    "    mul.lo.u32 %r36, %r39, %r22;\n"
    "    add.u32 %r36, %r36, %r25;\n"
    "    mul.lo.u32 %r36, %r36, %r20;\n"
    "    add.u32 %r36, %r36, %r33;\n"
    "    cvt.u64.u32 %rd1, %r36;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.s64 %rd1, %rd1, %rd13;\n"
    "    st.global.f32 [%rd1], %f0;\n"
    "$L_mla_next:\n"
    "    add.u32 %r38, %r38, 128;\n"
    "    bra $L_mla_task;\n"
    "$L_ret:\n"
    "    ret;\n"
    "}\n";

/* geode_attn_softmax: one (head, token) score row, normalized in place.
   Rows only run to the token's attended length; the rest of the buffer is
   never read. */
static const char attn_softmax_ptx[] =
    ".visible .entry geode_attn_softmax(\n"
    "    .param .u64 p_scores, .param .u32 p_position, .param .u32 p_ntok,\n"
    "    .param .u64 p_geom)\n"
    "{\n"
    "    .reg .pred %p<4>;\n"
    "    .reg .f32 %f<16>;\n"
    "    .reg .u32 %r<32>;\n"
    "    .reg .u64 %rd<16>;\n"
    "    .shared .align 16 .b8 geode_red[128];\n"
    "    ld.param.u64 %rd10, [p_scores];\n"
    "    ld.param.u32 %r10, [p_position];\n"
    "    ld.param.u32 %r11, [p_ntok];\n"
    "    ld.param.u64 %rd11, [p_geom];\n"
    "    ld.global.u32 %r13, [%rd11+56];\n"
    "    mov.u32 %r12, %tid.x;\n"
    "    mov.u32 %r19, %ctaid.y;\n"
    "    add.u32 %r14, %r10, %r19;\n"
    "    add.u32 %r14, %r14, 1;\n"
    "    mov.u32 %r19, %ctaid.x;\n"
    "    mul.lo.u32 %r15, %r19, %r11;\n"
    "    mov.u32 %r19, %ctaid.y;\n"
    "    add.u32 %r15, %r15, %r19;\n"
    "    mul.lo.u32 %r15, %r15, %r13;\n"
    "    cvt.u64.u32 %rd1, %r15;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.s64 %rd12, %rd10, %rd1;\n"
    "    mov.f32 %f10, 0fff7fffff;\n"
    "    mov.u32 %r16, %r12;\n"
    "$L_max:\n"
    "    setp.ge.s32 %p1, %r16, %r14;\n"
    "    @%p1 bra $L_max_done;\n"
    "    mul.wide.u32 %rd1, %r16, 4;\n"
    "    add.s64 %rd1, %rd12, %rd1;\n"
    "    ld.global.f32 %f1, [%rd1];\n"
    "    max.f32 %f10, %f10, %f1;\n"
    "    add.u32 %r16, %r16, 128;\n"
    "    bra $L_max;\n"
    "$L_max_done:\n"
    "    mov.u32 %r20, 1;\n"
    "    mov.u32 %r21, 2;\n"
    "    mov.u32 %r22, 4;\n"
    "    mov.u32 %r23, 8;\n"
    "    mov.u32 %r24, 16;\n"
    "    mov.b32 %r25, %f10;\n"
    "    shfl.bfly.b32 %r26, %r25, %r20, 31;\n"
    "    mov.b32 %f14, %r26;\n"
    "    max.f32 %f10, %f10, %f14;\n"
    "    mov.b32 %r25, %f10;\n"
    "    shfl.bfly.b32 %r26, %r25, %r21, 31;\n"
    "    mov.b32 %f14, %r26;\n"
    "    max.f32 %f10, %f10, %f14;\n"
    "    mov.b32 %r25, %f10;\n"
    "    shfl.bfly.b32 %r26, %r25, %r22, 31;\n"
    "    mov.b32 %f14, %r26;\n"
    "    max.f32 %f10, %f10, %f14;\n"
    "    mov.b32 %r25, %f10;\n"
    "    shfl.bfly.b32 %r26, %r25, %r23, 31;\n"
    "    mov.b32 %f14, %r26;\n"
    "    max.f32 %f10, %f10, %f14;\n"
    "    mov.b32 %r25, %f10;\n"
    "    shfl.bfly.b32 %r26, %r25, %r24, 31;\n"
    "    mov.b32 %f14, %r26;\n"
    "    max.f32 %f10, %f10, %f14;\n"
    "    and.b32 %r19, %r12, 31;\n"
    "    shr.u32 %r18, %r12, 5;\n"
    "    setp.ne.s32 %p1, %r19, 0;\n"
    "    @%p1 bra $L_max_share;\n"
    "    mov.u32 %r17, geode_red;\n"
    "    shl.b32 %r18, %r18, 2;\n"
    "    add.u32 %r17, %r17, %r18;\n"
    "    st.shared.f32 [%r17], %f10;\n"
    "$L_max_share:\n"
    "    bar.sync 0;\n"
    "    mov.u32 %r17, geode_red;\n"
    "    ld.shared.v4.f32 {%f1,%f2,%f3,%f4}, [%r17];\n"
    "    max.f32 %f10, %f1, %f2;\n"
    "    max.f32 %f10, %f10, %f3;\n"
    "    max.f32 %f10, %f10, %f4;\n"
    "    mov.f32 %f11, 0f00000000;\n"
    "    mov.u32 %r16, %r12;\n"
    "$L_exp:\n"
    "    setp.ge.s32 %p1, %r16, %r14;\n"
    "    @%p1 bra $L_exp_done;\n"
    "    mul.wide.u32 %rd1, %r16, 4;\n"
    "    add.s64 %rd1, %rd12, %rd1;\n"
    "    ld.global.f32 %f1, [%rd1];\n"
    "    sub.f32 %f1, %f1, %f10;\n"
    "    mul.f32 %f1, %f1, 0f3FB8AA3B;\n"
    "    ex2.approx.f32 %f1, %f1;\n"
    "    st.global.f32 [%rd1], %f1;\n"
    "    add.f32 %f11, %f11, %f1;\n"
    "    add.u32 %r16, %r16, 128;\n"
    "    bra $L_exp;\n"
    "$L_exp_done:\n"
    "    mov.b32 %r27, %f11;\n"
    "    shfl.bfly.b32 %r28, %r27, %r20, 31;\n"
    "    mov.b32 %f15, %r28;\n"
    "    add.f32 %f11, %f11, %f15;\n"
    "    mov.b32 %r27, %f11;\n"
    "    shfl.bfly.b32 %r28, %r27, %r21, 31;\n"
    "    mov.b32 %f15, %r28;\n"
    "    add.f32 %f11, %f11, %f15;\n"
    "    mov.b32 %r27, %f11;\n"
    "    shfl.bfly.b32 %r28, %r27, %r22, 31;\n"
    "    mov.b32 %f15, %r28;\n"
    "    add.f32 %f11, %f11, %f15;\n"
    "    mov.b32 %r27, %f11;\n"
    "    shfl.bfly.b32 %r28, %r27, %r23, 31;\n"
    "    mov.b32 %f15, %r28;\n"
    "    add.f32 %f11, %f11, %f15;\n"
    "    mov.b32 %r27, %f11;\n"
    "    shfl.bfly.b32 %r28, %r27, %r24, 31;\n"
    "    mov.b32 %f15, %r28;\n"
    "    add.f32 %f11, %f11, %f15;\n"
    "    and.b32 %r19, %r12, 31;\n"
    "    shr.u32 %r18, %r12, 5;\n"
    "    setp.ne.s32 %p1, %r19, 0;\n"
    "    @%p1 bra $L_sum_share;\n"
    "    mov.u32 %r17, geode_red;\n"
    "    shl.b32 %r18, %r18, 2;\n"
    "    add.u32 %r17, %r17, %r18;\n"
    "    st.shared.f32 [%r17+16], %f11;\n"
    "$L_sum_share:\n"
    "    bar.sync 0;\n"
    "    mov.u32 %r17, geode_red;\n"
    "    ld.shared.v4.f32 {%f1,%f2,%f3,%f4}, [%r17+16];\n"
    "    add.f32 %f11, %f1, %f2;\n"
    "    add.f32 %f11, %f11, %f3;\n"
    "    add.f32 %f11, %f11, %f4;\n"
    "    mov.f32 %f13, 0f3F800000;\n"
    "    div.rn.f32 %f12, %f13, %f11;\n"
    "    mov.u32 %r16, %r12;\n"
    "$L_scale:\n"
    "    setp.ge.s32 %p1, %r16, %r14;\n"
    "    @%p1 bra $L_ret;\n"
    "    mul.wide.u32 %rd1, %r16, 4;\n"
    "    add.s64 %rd1, %rd12, %rd1;\n"
    "    ld.global.f32 %f1, [%rd1];\n"
    "    mul.f32 %f1, %f1, %f12;\n"
    "    st.global.f32 [%rd1], %f1;\n"
    "    add.u32 %r16, %r16, 128;\n"
    "    bra $L_scale;\n"
    "$L_ret:\n"
    "    ret;\n"
    "}\n";

/* geode_attn_fold: out[t][head][fold_width] = sum_p scores[h][t][p] * value(p).
   A block owns one 64-wide slice of the fold width, for every head at once:
   each thread accumulates one head's 16 elements across all positions, with
   the tile's dequantized values staged in shared and each score broadcast to
   the 64 elements it weights. Position tiles are the loop; accumulators live
   across them in shared. */
static const char attn_fold_ptx[] =
    ".visible .entry geode_attn_fold(\n"
    "    .param .u64 p_cache, .param .u64 p_scores, .param .u64 p_out,\n"
    "    .param .u32 p_position, .param .u32 p_ntok, .param .u64 p_geom)\n"
    "{\n"
    "    .reg .pred %p<10>;\n"
    "    .reg .f32 %f<16>;\n"
    "    .reg .u32 %r<52>;\n"
    "    .reg .u64 %rd<20>;\n"
    "    .shared .align 16 .b8 geode_shm[32768];\n"
    "    ld.param.u64 %rd10, [p_cache];\n"
    "    ld.param.u64 %rd11, [p_scores];\n"
    "    ld.param.u64 %rd12, [p_out];\n"
    "    ld.param.u32 %r19, [p_position];\n"
    "    ld.param.u32 %r20, [p_ntok];\n"
    "    ld.param.u64 %rd13, [p_geom];\n"
    "    ld.global.u32 %r10, [%rd13+0];\n"
    "    ld.global.u32 %r11, [%rd13+4];\n"
    "    ld.global.u32 %r12, [%rd13+8];\n"
    "    ld.global.u32 %r13, [%rd13+12];\n"
    "    ld.global.u32 %r14, [%rd13+28];\n"
    "    ld.global.u32 %r15, [%rd13+32];\n"
    "    ld.global.u32 %r16, [%rd13+44];\n"
    "    ld.global.u32 %r17, [%rd13+48];\n"
    "    ld.global.u32 %r18, [%rd13+56];\n"
    "    mov.u32 %r21, %tid.x;\n"
    "    mov.u32 %r22, geode_shm;\n"
    "    mov.u32 %r23, %ctaid.y;\n"
    "    add.u32 %r24, %r19, %r23;\n"
    "    add.u32 %r24, %r24, 1;\n"
    "    mov.u32 %r25, %ctaid.x;\n"
    "    shl.b32 %r25, %r25, 6;\n"
    "    setp.eq.s32 %p7, %r10, 0;\n"
    /* accumulator area: n_head * 64 floats */
    "    mul.lo.u32 %r26, %r11, 64;\n"
    "    shl.b32 %r26, %r26, 2;\n"
    "    shr.u32 %r27, %r26, 2;\n"
    "    mov.u32 %r28, %r21;\n"
    "$L_zero:\n"
    "    setp.ge.s32 %p1, %r28, %r27;\n"
    "    @%p1 bra $L_zero_done;\n"
    "    shl.b32 %r29, %r28, 2;\n"
    "    add.u32 %r29, %r22, %r29;\n"
    "    mov.f32 %f1, 0f00000000;\n"
    "    st.shared.f32 [%r29], %f1;\n"
    "    add.u32 %r28, %r28, 128;\n"
    "    bra $L_zero;\n"
    "$L_zero_done:\n"
    "    bar.sync 0;\n"
    /* thread's cells: 16 r's of one head */
    "    div.u32 %r29, %r21, 4;\n"
    "    rem.u32 %r30, %r21, 4;\n"
    "    shl.b32 %r30, %r30, 4;\n"
    "    setp.ge.s32 %p8, %r29, %r11;\n"
    "    mul.lo.u32 %r31, %r29, %r12;\n"
    "    div.u32 %r31, %r31, %r11;\n"
    "    @%p7 mov.u32 %r31, 0;\n"
    "    mov.u32 %r32, 1;\n"
    "    @!%p7 mul.lo.u32 %r32, %r12, 1;\n"
    "    mul.lo.u32 %r33, %r29, %r20;\n"
    "    add.u32 %r33, %r33, %r23;\n"
    "    mul.lo.u32 %r33, %r33, %r18;\n"
    "    mul.lo.u32 %r34, %r29, 64;\n"
    "    add.u32 %r34, %r34, %r30;\n"
    "    shl.b32 %r34, %r34, 2;\n"
    "    add.u32 %r34, %r22, %r34;\n"
    "    mov.u32 %r35, 0;\n"
    "$L_pt:\n"
    "    setp.ge.s32 %p1, %r35, %r24;\n"
    "    @%p1 bra $L_write;\n"
    /* stage 8 positions x kvmax slices x 64 values, dequantized */
    "    mul.lo.u32 %r36, %r32, 64;\n"
    "    mul.lo.u32 %r36, %r36, 8;\n"
    "    mov.u32 %r27, %r21;\n"
    "$L_stage:\n"
    "    setp.ge.s32 %p2, %r27, %r36;\n"
    "    @%p2 bra $L_stage_done;\n"
    "    mul.lo.u32 %r37, %r32, 64;\n"
    "    div.u32 %r38, %r27, %r37;\n"
    "    mul.lo.u32 %r39, %r38, %r37;\n"
    "    sub.u32 %r39, %r27, %r39;\n"
    "    div.u32 %r40, %r39, 64;\n"
    "    mul.lo.u32 %r41, %r40, 64;\n"
    "    sub.u32 %r41, %r39, %r41;\n"
    "    add.u32 %r42, %r35, %r38;\n"
    "    add.u32 %r43, %r25, %r41;\n"
    "    setp.ge.s32 %p3, %r42, %r24;\n"
    "    mov.u32 %r44, 128;\n"
    "    mov.f32 %f11, 0f00000000;\n"
    "    setp.lt.s32 %p4, %r43, %r17;\n"
    "    @%p3 bra $L_st_store;\n"
    "    @!%p4 bra $L_st_zero;\n"
    "    mul.wide.u32 %rd1, %r42, %r15;\n"
    "    cvt.u64.u32 %rd2, %r43;\n"
    "    @!%p7 bra $L_st_gqa;\n"
    "    add.s64 %rd1, %rd1, %rd2;\n"
    "    add.s64 %rd1, %rd1, %rd10;\n"
    "    bra $L_st_byte;\n"
    "$L_st_gqa:\n"
    "    mul.lo.u32 %r28, %r12, %r13;\n"
    "    cvt.u64.u32 %rd3, %r28;\n"
    "    add.s64 %rd1, %rd1, %rd3;\n"
    "    mul.lo.u32 %r28, %r40, %r17;\n"
    "    cvt.u64.u32 %rd3, %r28;\n"
    "    add.s64 %rd1, %rd1, %rd3;\n"
    "    add.s64 %rd1, %rd1, %rd2;\n"
    "    add.s64 %rd1, %rd1, %rd10;\n"
    "$L_st_byte:\n"
    "    ld.global.u8 %r44, [%rd1];\n"
    "    mul.wide.u32 %rd1, %r42, %r15;\n"
    "    cvt.u64.u32 %rd2, %r14;\n"
    "    add.s64 %rd1, %rd1, %rd2;\n"
    "    @%p7 bra $L_st_scale;\n"
    "    add.u32 %r49, %r12, %r40;\n"
    "    mul.wide.u32 %rd3, %r49, 4;\n"
    "    add.s64 %rd1, %rd1, %rd3;\n"
    "$L_st_scale:\n"
    "    add.s64 %rd1, %rd1, %rd10;\n"
    "    ld.global.f32 %f11, [%rd1];\n"
    "    bra $L_st_deq;\n"
    "$L_st_zero:\n"
    "    mov.u32 %r44, 128;\n"
    "    mov.f32 %f11, 0f00000000;\n"
    "$L_st_deq:\n"
    "$L_st_store:\n"
    "    cvt.rn.f32.u32 %f1, %r44;\n"
    "    sub.f32 %f1, %f1, 0f43000000;\n"
    "    mul.f32 %f1, %f1, %f11;\n"
    "    shl.b32 %r28, %r27, 2;\n"
    "    add.u32 %r28, %r22, %r28;\n"
    "    add.u32 %r28, %r28, %r26;\n"
    "    st.shared.f32 [%r28], %f1;\n"
    "    add.u32 %r27, %r27, 128;\n"
    "    bra $L_stage;\n"
    "$L_stage_done:\n"
    "    bar.sync 0;\n"
    "    @%p8 bra $L_pt_end;\n"
    /* fold the tile into the accumulators */
    "    mul.lo.u32 %r45, %r31, 64;\n"
    "    mul.lo.u32 %r37, %r32, 64;\n"
    "    mov.u32 %r38, 0;\n"
    "$L_pl:\n"
    "    setp.ge.s32 %p5, %r38, 8;\n"
    "    @%p5 bra $L_pl_done;\n"
    "    add.u32 %r42, %r35, %r38;\n"
    "    setp.ge.s32 %p6, %r42, %r24;\n"
    "    @%p6 bra $L_pl_next;\n"
    "    mul.lo.u32 %r46, %r38, %r37;\n"
    "    add.u32 %r46, %r46, %r45;\n"
    "    add.u32 %r46, %r46, %r30;\n"
    "    add.u32 %r47, %r33, %r42;\n"
    "    cvt.u64.u32 %rd1, %r47;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.s64 %rd1, %rd1, %rd11;\n"
    "    ld.global.f32 %f10, [%rd1];\n"
    "    shl.b32 %r46, %r46, 2;\n"
    "    add.u32 %r46, %r22, %r46;\n"
    "    add.u32 %r46, %r46, %r26;\n"
    "    ld.shared.f32 %f1, [%r46];\n"
    "    ld.shared.f32 %f2, [%r34];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+4];\n"
    "    ld.shared.f32 %f2, [%r34+4];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+4], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+8];\n"
    "    ld.shared.f32 %f2, [%r34+8];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+8], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+12];\n"
    "    ld.shared.f32 %f2, [%r34+12];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+12], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+16];\n"
    "    ld.shared.f32 %f2, [%r34+16];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+16], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+20];\n"
    "    ld.shared.f32 %f2, [%r34+20];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+20], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+24];\n"
    "    ld.shared.f32 %f2, [%r34+24];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+24], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+28];\n"
    "    ld.shared.f32 %f2, [%r34+28];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+28], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+32];\n"
    "    ld.shared.f32 %f2, [%r34+32];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+32], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+36];\n"
    "    ld.shared.f32 %f2, [%r34+36];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+36], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+40];\n"
    "    ld.shared.f32 %f2, [%r34+40];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+40], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+44];\n"
    "    ld.shared.f32 %f2, [%r34+44];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+44], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+48];\n"
    "    ld.shared.f32 %f2, [%r34+48];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+48], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+52];\n"
    "    ld.shared.f32 %f2, [%r34+52];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+52], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+56];\n"
    "    ld.shared.f32 %f2, [%r34+56];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+56], %f2;\n"
    "    ld.shared.f32 %f1, [%r46+60];\n"
    "    ld.shared.f32 %f2, [%r34+60];\n"
    "    fma.rn.f32 %f2, %f10, %f1, %f2;\n"
    "    st.shared.f32 [%r34+60], %f2;\n"
    "$L_pl_next:\n"
    "    add.u32 %r38, %r38, 1;\n"
    "    bra $L_pl;\n"
    "$L_pl_done:\n"
    "$L_pt_end:\n"
    "    bar.sync 0;\n"
    "    add.u32 %r35, %r35, 8;\n"
    "    bra $L_pt;\n"
    /* write out */
    "$L_write:\n"
    "    bar.sync 0;\n"
    "    shr.u32 %r51, %r26, 2;\n"
    "    mov.u32 %r27, %r21;\n"
    "$L_out:\n"
    "    setp.ge.s32 %p1, %r27, %r51;\n"
    "    @%p1 bra $L_ret;\n"
    "    div.u32 %r48, %r27, 64;\n"
    "    rem.u32 %r41, %r27, 64;\n"
    "    add.u32 %r43, %r25, %r41;\n"
    "    setp.lt.s32 %p2, %r43, %r17;\n"
    "    @!%p2 bra $L_out_next;\n"
    "    mul.lo.u32 %r28, %r48, %r17;\n"
    "    add.u32 %r28, %r28, %r43;\n"
    "    mul.wide.u32 %rd1, %r23, %r16;\n"
    "    cvt.u64.u32 %rd2, %r28;\n"
    "    add.s64 %rd1, %rd1, %rd2;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.s64 %rd1, %rd1, %rd12;\n"
    "    shl.b32 %r28, %r27, 2;\n"
    "    add.u32 %r28, %r22, %r28;\n"
    "    ld.shared.f32 %f1, [%r28];\n"
    "    st.global.f32 [%rd1], %f1;\n"
    "$L_out_next:\n"
    "    add.u32 %r27, %r27, 128;\n"
    "    bra $L_out;\n"
    "$L_ret:\n"
    "    ret;\n"
    "}\n";

/* geode_q_prep: the query half of the old cpu-side prepare, moved on device
   so the query never crosses the bus. One lane per head, one block per
   token: GQA rmsnorms each head against the layer's q_norm then applies
   neox rope; MLA rotates only the tail slice of each head, interleaved
   pairs. Op-for-op the cpu tier's prepare_query, so the self-test holds it
   to near-exact agreement. */
static const char q_prep_ptx[] =
    ".visible .entry geode_q_prep(\n"
    "    .param .u64 p_query, .param .u64 p_cos, .param .u64 p_weight,\n"
    "    .param .u32 p_ntok, .param .u64 p_geom)\n"
    "{\n"
    "    .reg .pred %p<4>;\n"
    "    .reg .f32 %f<16>;\n"
    "    .reg .u32 %r<24>;\n"
    "    .reg .u64 %rd<14>;\n"
    "    ld.param.u64 %rd10, [p_query];\n"
    "    ld.param.u64 %rd11, [p_cos];\n"
    "    ld.param.u64 %rd12, [p_weight];\n"
    "    ld.param.u32 %r20, [p_ntok];\n"
    "    ld.param.u64 %rd13, [p_geom];\n"
    "    ld.global.u32 %r10, [%rd13+0];\n"
    "    ld.global.u32 %r11, [%rd13+4];\n"
    "    ld.global.u32 %r12, [%rd13+12];\n"
    "    ld.global.u32 %r13, [%rd13+20];\n"
    "    ld.global.u32 %r14, [%rd13+36];\n"
    "    ld.global.f32 %f10, [%rd13+60];\n"
    "    mov.u32 %r15, %tid.x;\n"
    "    mov.u32 %r16, %ctaid.x;\n"
    "    setp.ge.s32 %p1, %r15, %r11;\n"
    "    setp.ge.s32 %p2, %r16, %r20;\n"
    "    @%p1 bra $L_ret;\n"
    "    @%p2 bra $L_ret;\n"
    "    mul.lo.u32 %r18, %r16, %r14;\n"
    "    mul.lo.u32 %r19, %r15, %r12;\n"
    "    add.u32 %r18, %r18, %r19;\n"
    "    cvt.u64.u32 %rd1, %r18;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.s64 %rd1, %rd1, %rd10;\n"
    "    mul.lo.u32 %r18, %r16, %r13;\n"
    "    cvt.u64.u32 %rd2, %r18;\n"
    "    shl.b64 %rd2, %rd2, 2;\n"
    "    add.s64 %rd2, %rd2, %rd11;\n"
    "    setp.eq.s32 %p3, %r10, 0;\n"
    "    @%p3 bra $L_mla;\n"
    "    mov.f32 %f0, 0f00000000;\n"
    "    mov.u64 %rd3, %rd1;\n"
    "    mov.u32 %r21, 0;\n"
    "$L_sum:\n"
    "    setp.ge.s32 %p1, %r21, %r12;\n"
    "    @%p1 bra $L_scale;\n"
    "    ld.global.f32 %f1, [%rd3];\n"
    "    fma.rn.f32 %f0, %f1, %f1, %f0;\n"
    "    add.s64 %rd3, %rd3, 4;\n"
    "    add.u32 %r21, %r21, 1;\n"
    "    bra $L_sum;\n"
    "$L_scale:\n"
    "    cvt.rn.f32.u32 %f2, %r12;\n"
    "    div.rn.f32 %f3, %f0, %f2;\n"
    "    add.f32 %f3, %f3, %f10;\n"
    "    sqrt.rn.f32 %f4, %f3;\n"
    "    mov.f32 %f5, 0f3F800000;\n"
    "    div.rn.f32 %f5, %f5, %f4;\n"
    "    mov.u64 %rd3, %rd1;\n"
    "    mov.u32 %r21, 0;\n"
    "$L_norm:\n"
    "    setp.ge.s32 %p1, %r21, %r12;\n"
    "    @%p1 bra $L_rope;\n"
    "    ld.global.f32 %f1, [%rd3];\n"
    "    mul.rn.f32 %f1, %f1, %f5;\n"
    "    cvt.u64.u32 %rd4, %r21;\n"
    "    shl.b64 %rd4, %rd4, 2;\n"
    "    add.s64 %rd4, %rd4, %rd12;\n"
    "    ld.global.f32 %f2, [%rd4];\n"
    "    mul.rn.f32 %f1, %f1, %f2;\n"
    "    st.global.f32 [%rd3], %f1;\n"
    "    add.s64 %rd3, %rd3, 4;\n"
    "    add.u32 %r21, %r21, 1;\n"
    "    bra $L_norm;\n"
    "$L_rope:\n"
    "    shr.u32 %r22, %r13, 1;\n"
    "    mov.u32 %r21, 0;\n"
    "$L_gq_rot:\n"
    "    setp.ge.s32 %p1, %r21, %r22;\n"
    "    @%p1 bra $L_ret;\n"
    "    cvt.u64.u32 %rd4, %r21;\n"
    "    shl.b64 %rd4, %rd4, 3;\n"
    "    add.s64 %rd7, %rd4, %rd2;\n"
    "    ld.global.f32 %f1, [%rd7];\n"
    "    ld.global.f32 %f2, [%rd7+4];\n"
    "    shr.b64 %rd5, %rd4, 1;\n"
    "    add.s64 %rd5, %rd5, %rd1;\n"
    "    add.u32 %r23, %r21, %r22;\n"
    "    cvt.u64.u32 %rd6, %r23;\n"
    "    shl.b64 %rd6, %rd6, 2;\n"
    "    add.s64 %rd6, %rd6, %rd1;\n"
    "    ld.global.f32 %f3, [%rd5];\n"
    "    ld.global.f32 %f4, [%rd6];\n"
    "    mul.rn.f32 %f5, %f3, %f1;\n"
    "    mul.rn.f32 %f6, %f4, %f2;\n"
    "    sub.f32 %f5, %f5, %f6;\n"
    "    st.global.f32 [%rd5], %f5;\n"
    "    mul.rn.f32 %f5, %f3, %f2;\n"
    "    mul.rn.f32 %f6, %f4, %f1;\n"
    "    add.f32 %f5, %f5, %f6;\n"
    "    st.global.f32 [%rd6], %f5;\n"
    "    add.u32 %r21, %r21, 1;\n"
    "    bra $L_gq_rot;\n"
    "$L_mla:\n"
    "    sub.u32 %r23, %r12, %r13;\n"
    "    cvt.u64.u32 %rd5, %r23;\n"
    "    shl.b64 %rd5, %rd5, 2;\n"
    "    add.s64 %rd5, %rd5, %rd1;\n"
    "    shr.u32 %r22, %r13, 1;\n"
    "    mov.u32 %r21, 0;\n"
    "$L_ml_rot:\n"
    "    setp.ge.s32 %p1, %r21, %r22;\n"
    "    @%p1 bra $L_ret;\n"
    "    cvt.u64.u32 %rd4, %r21;\n"
    "    shl.b64 %rd4, %rd4, 3;\n"
    "    add.s64 %rd7, %rd4, %rd2;\n"
    "    ld.global.f32 %f1, [%rd7];\n"
    "    ld.global.f32 %f2, [%rd7+4];\n"
    "    add.s64 %rd6, %rd5, %rd4;\n"
    "    ld.global.f32 %f3, [%rd6];\n"
    "    ld.global.f32 %f4, [%rd6+4];\n"
    "    mul.rn.f32 %f5, %f3, %f1;\n"
    "    mul.rn.f32 %f6, %f4, %f2;\n"
    "    sub.f32 %f5, %f5, %f6;\n"
    "    st.global.f32 [%rd6], %f5;\n"
    "    mul.rn.f32 %f5, %f3, %f2;\n"
    "    mul.rn.f32 %f6, %f4, %f1;\n"
    "    add.f32 %f5, %f5, %f6;\n"
    "    st.global.f32 [%rd6+4], %f5;\n"
    "    add.u32 %r21, %r21, 1;\n"
    "    bra $L_ml_rot;\n"
    "$L_ret:\n"
    "    ret;\n"
    "}\n";

/* geode_kv_prep: the kv half, one lane per cache segment per token. Each
   lane norms its segment in place (the latent against kv_a_norm, GQA keys
   against attn_k_norm), rotates what rotates, and quantizes straight into
   the layer's slot in vram -- bytes and scales land where score and fold
   read them, so the cache never crosses the bus either. Op-for-op the cpu
   tier's prepare_kv, quantize_cache's rounding included. */
static const char kv_prep_ptx[] =
    ".visible .entry geode_kv_prep(\n"
    "    .param .u64 p_kv, .param .u64 p_cos, .param .u64 p_norm,\n"
    "    .param .u64 p_slot, .param .u32 p_ntok, .param .u64 p_geom)\n"
    "{\n"
    "    .reg .pred %p<8>;\n"
    "    .reg .f32 %f<16>;\n"
    "    .reg .b16 %rs<2>;\n"
    "    .reg .u32 %r<32>;\n"
    "    .reg .u64 %rd<16>;\n"
    "    ld.param.u64 %rd10, [p_kv];\n"
    "    ld.param.u64 %rd11, [p_cos];\n"
    "    ld.param.u64 %rd12, [p_norm];\n"
    "    ld.param.u64 %rd13, [p_slot];\n"
    "    ld.param.u32 %r30, [p_ntok];\n"
    "    ld.param.u64 %rd14, [p_geom];\n"
    "    ld.global.u32 %r10, [%rd14+0];\n"
    "    ld.global.u32 %r11, [%rd14+8];\n"
    "    ld.global.u32 %r12, [%rd14+12];\n"
    "    ld.global.u32 %r13, [%rd14+16];\n"
    "    ld.global.u32 %r14, [%rd14+20];\n"
    "    ld.global.u32 %r15, [%rd14+24];\n"
    "    ld.global.u32 %r16, [%rd14+28];\n"
    "    ld.global.u32 %r17, [%rd14+32];\n"
    "    ld.global.u32 %r18, [%rd14+48];\n"
    "    ld.global.f32 %f10, [%rd14+60];\n"
    "    mov.u32 %r19, %tid.x;\n"
    "    mov.u32 %r20, %ctaid.x;\n"
    "    setp.ge.s32 %p1, %r19, %r15;\n"
    "    setp.ge.s32 %p2, %r20, %r30;\n"
    "    mov.u32 %r25, 0;\n"
    "    mov.u32 %r26, 0;\n"
    "    @%p1 bra $L_ret;\n"
    "    @%p2 bra $L_ret;\n"
    "    setp.eq.s32 %p3, %r10, 0;\n"
    "    @%p3 bra $L_mla;\n"
    "    setp.lt.s32 %p4, %r19, %r11;\n"
    "    sub.u32 %r24, %r19, %r11;\n"
    "    selp.b32 %r24, %r19, %r24, %p4;\n"
    "    @%p4 bra $L_key;\n"
    "    mul.lo.u32 %r21, %r11, %r12;\n"
    "    mul.lo.u32 %r27, %r24, %r18;\n"
    "    add.u32 %r21, %r21, %r27;\n"
    "    mov.u32 %r22, %r18;\n"
    "    add.u32 %r23, %r11, %r24;\n"
    "    bra $L_have;\n"
    "$L_key:\n"
    "    mul.lo.u32 %r21, %r24, %r12;\n"
    "    mov.u32 %r22, %r12;\n"
    "    mov.u32 %r23, %r24;\n"
    "    mov.u32 %r26, 1;\n"
    "    mov.u32 %r25, 2;\n"
    "    bra $L_have;\n"
    "$L_mla:\n"
    "    mul.lo.u32 %r21, %r19, %r13;\n"
    "    setp.eq.s32 %p4, %r19, 0;\n"
    "    selp.b32 %r22, %r13, %r14, %p4;\n"
    "    mov.u32 %r23, %r19;\n"
    "    mov.u32 %r27, 1;\n"
    "    selp.b32 %r26, %r27, %r26, %p4;\n"
    "    setp.eq.s32 %p5, %r19, 1;\n"
    "    selp.b32 %r25, %r27, %r25, %p5;\n"
    "$L_have:\n"
    "    mul.lo.u32 %r27, %r20, %r16;\n"
    "    add.u32 %r27, %r27, %r21;\n"
    "    cvt.u64.u32 %rd1, %r27;\n"
    "    shl.b64 %rd1, %rd1, 2;\n"
    "    add.s64 %rd1, %rd1, %rd10;\n"
    "    mul.lo.u32 %r27, %r20, %r17;\n"
    "    add.u32 %r27, %r27, %r21;\n"
    "    cvt.u64.u32 %rd2, %r27;\n"
    "    add.s64 %rd2, %rd2, %rd13;\n"
    "    mul.lo.u32 %r27, %r20, %r17;\n"
    "    add.u32 %r27, %r27, %r16;\n"
    "    mul.lo.u32 %r28, %r23, 4;\n"
    "    add.u32 %r27, %r27, %r28;\n"
    "    cvt.u64.u32 %rd3, %r27;\n"
    "    add.s64 %rd3, %rd3, %rd13;\n"
    "    mul.lo.u32 %r27, %r20, %r14;\n"
    "    cvt.u64.u32 %rd4, %r27;\n"
    "    shl.b64 %rd4, %rd4, 2;\n"
    "    add.s64 %rd4, %rd4, %rd11;\n"
    "    setp.eq.s32 %p4, %r26, 0;\n"
    "    @%p4 bra $L_rope_check;\n"
    "    mov.f32 %f0, 0f00000000;\n"
    "    mov.u64 %rd5, %rd1;\n"
    "    mov.u32 %r28, 0;\n"
    "$L_norm_sum:\n"
    "    setp.ge.s32 %p5, %r28, %r22;\n"
    "    @%p5 bra $L_norm_apply;\n"
    "    ld.global.f32 %f1, [%rd5];\n"
    "    fma.rn.f32 %f0, %f1, %f1, %f0;\n"
    "    add.s64 %rd5, %rd5, 4;\n"
    "    add.u32 %r28, %r28, 1;\n"
    "    bra $L_norm_sum;\n"
    "$L_norm_apply:\n"
    "    cvt.rn.f32.u32 %f2, %r22;\n"
    "    div.rn.f32 %f3, %f0, %f2;\n"
    "    add.f32 %f3, %f3, %f10;\n"
    "    sqrt.rn.f32 %f4, %f3;\n"
    "    mov.f32 %f5, 0f3F800000;\n"
    "    div.rn.f32 %f5, %f5, %f4;\n"
    "    mov.u64 %rd5, %rd1;\n"
    "    mov.u32 %r28, 0;\n"
    "$L_norm_loop:\n"
    "    setp.ge.s32 %p5, %r28, %r22;\n"
    "    @%p5 bra $L_rope_check;\n"
    "    ld.global.f32 %f1, [%rd5];\n"
    "    mul.rn.f32 %f1, %f1, %f5;\n"
    "    cvt.u64.u32 %rd6, %r28;\n"
    "    shl.b64 %rd6, %rd6, 2;\n"
    "    add.s64 %rd6, %rd6, %rd12;\n"
    "    ld.global.f32 %f2, [%rd6];\n"
    "    mul.rn.f32 %f1, %f1, %f2;\n"
    "    st.global.f32 [%rd5], %f1;\n"
    "    add.s64 %rd5, %rd5, 4;\n"
    "    add.u32 %r28, %r28, 1;\n"
    "    bra $L_norm_loop;\n"
    "$L_rope_check:\n"
    "    setp.eq.s32 %p4, %r25, 0;\n"
    "    @%p4 bra $L_quant;\n"
    "    setp.eq.s32 %p5, %r25, 2;\n"
    "    @%p5 bra $L_rope_neox;\n"
    "    shr.u32 %r26, %r22, 1;\n"
    "    mov.u32 %r28, 0;\n"
    "$L_rot_i:\n"
    "    setp.ge.s32 %p5, %r28, %r26;\n"
    "    @%p5 bra $L_quant;\n"
    "    cvt.u64.u32 %rd6, %r28;\n"
    "    shl.b64 %rd6, %rd6, 3;\n"
    "    add.s64 %rd7, %rd4, %rd6;\n"
    "    ld.global.f32 %f1, [%rd7];\n"
    "    ld.global.f32 %f2, [%rd7+4];\n"
    "    add.s64 %rd8, %rd1, %rd6;\n"
    "    ld.global.f32 %f3, [%rd8];\n"
    "    ld.global.f32 %f4, [%rd8+4];\n"
    "    mul.rn.f32 %f5, %f3, %f1;\n"
    "    mul.rn.f32 %f6, %f4, %f2;\n"
    "    sub.f32 %f5, %f5, %f6;\n"
    "    st.global.f32 [%rd8], %f5;\n"
    "    mul.rn.f32 %f5, %f3, %f2;\n"
    "    mul.rn.f32 %f6, %f4, %f1;\n"
    "    add.f32 %f5, %f5, %f6;\n"
    "    st.global.f32 [%rd8+4], %f5;\n"
    "    add.u32 %r28, %r28, 1;\n"
    "    bra $L_rot_i;\n"
    "$L_rope_neox:\n"
    "    shr.u32 %r26, %r14, 1;\n"
    "    mov.u32 %r28, 0;\n"
    "$L_rot_n:\n"
    "    setp.ge.s32 %p5, %r28, %r26;\n"
    "    @%p5 bra $L_quant;\n"
    "    cvt.u64.u32 %rd6, %r28;\n"
    "    shl.b64 %rd6, %rd6, 3;\n"
    "    add.s64 %rd7, %rd4, %rd6;\n"
    "    ld.global.f32 %f1, [%rd7];\n"
    "    ld.global.f32 %f2, [%rd7+4];\n"
    "    cvt.u64.u32 %rd8, %r28;\n"
    "    shl.b64 %rd8, %rd8, 2;\n"
    "    add.s64 %rd8, %rd8, %rd1;\n"
    "    add.u32 %r27, %r28, %r26;\n"
    "    cvt.u64.u32 %rd9, %r27;\n"
    "    shl.b64 %rd9, %rd9, 2;\n"
    "    add.s64 %rd9, %rd9, %rd1;\n"
    "    ld.global.f32 %f3, [%rd8];\n"
    "    ld.global.f32 %f4, [%rd9];\n"
    "    mul.rn.f32 %f5, %f3, %f1;\n"
    "    mul.rn.f32 %f6, %f4, %f2;\n"
    "    sub.f32 %f5, %f5, %f6;\n"
    "    st.global.f32 [%rd8], %f5;\n"
    "    mul.rn.f32 %f5, %f3, %f2;\n"
    "    mul.rn.f32 %f6, %f4, %f1;\n"
    "    add.f32 %f5, %f5, %f6;\n"
    "    st.global.f32 [%rd9], %f5;\n"
    "    add.u32 %r28, %r28, 1;\n"
    "    bra $L_rot_n;\n"
    "$L_quant:\n"
    "    mov.f32 %f0, 0f00000000;\n"
    "    mov.u64 %rd5, %rd1;\n"
    "    mov.u32 %r28, 0;\n"
    "$L_amax:\n"
    "    setp.ge.s32 %p5, %r28, %r22;\n"
    "    @%p5 bra $L_zero_check;\n"
    "    ld.global.f32 %f1, [%rd5];\n"
    "    abs.f32 %f1, %f1;\n"
    "    max.f32 %f0, %f0, %f1;\n"
    "    add.s64 %rd5, %rd5, 4;\n"
    "    add.u32 %r28, %r28, 1;\n"
    "    bra $L_amax;\n"
    "$L_zero_check:\n"
    "    mov.f32 %f2, 0f00000000;\n"
    "    setp.eq.f32 %p5, %f0, %f2;\n"
    "    @%p5 bra $L_all_zero;\n"
    "    mov.f32 %f2, 0f42FE0000;\n"
    "    div.rn.f32 %f2, %f2, %f0;\n"
    "    mov.f32 %f3, 0f3F800000;\n"
    "    div.rn.f32 %f3, %f3, %f2;\n"
    "    mov.u64 %rd5, %rd1;\n"
    "    mov.u64 %rd6, %rd2;\n"
    "    mov.u32 %r28, 0;\n"
    "$L_quant_loop:\n"
    "    setp.ge.s32 %p5, %r28, %r22;\n"
    "    @%p5 bra $L_scale_store;\n"
    "    ld.global.f32 %f1, [%rd5];\n"
    "    mul.rn.f32 %f1, %f1, %f2;\n"
    "    cvt.rni.s32.f32 %r27, %f1;\n"
    "    add.u32 %r27, %r27, 128;\n"
    "    cvt.u16.u32 %rs1, %r27;\n"
    "    st.global.u8 [%rd6], %rs1;\n"
    "    add.s64 %rd5, %rd5, 4;\n"
    "    add.s64 %rd6, %rd6, 1;\n"
    "    add.u32 %r28, %r28, 1;\n"
    "    bra $L_quant_loop;\n"
    "$L_scale_store:\n"
    "    st.global.f32 [%rd3], %f3;\n"
    "    bra $L_ret;\n"
    "$L_all_zero:\n"
    "    mov.u64 %rd6, %rd2;\n"
    "    mov.u32 %r28, 0;\n"
    "$L_zero_loop:\n"
    "    setp.ge.s32 %p5, %r28, %r22;\n"
    "    @%p5 bra $L_zero_scale;\n"
    "    mov.u16 %rs1, 128;\n"
    "    st.global.u8 [%rd6], %rs1;\n"
    "    add.s64 %rd6, %rd6, 1;\n"
    "    add.u32 %r28, %r28, 1;\n"
    "    bra $L_zero_loop;\n"
    "$L_zero_scale:\n"
    "    mov.f32 %f1, 0f00000000;\n"
    "    st.global.f32 [%rd3], %f1;\n"
    "$L_ret:\n"
    "    ret;\n"
    "}\n";

static char *build_module(void) {
    char *text = NULL;
    size_t size = 0;
    FILE *out = open_memstream(&text, &size);
    if (!out) return NULL;
    fputs(".version 6.0\n.target sm_50\n.address_size 64\n", out);
    for (int tile = 1; tile <= GEMM_TILE_MAX; tile *= 2) {
        gemm_entry(out, tile, 1);
        gemm_entry(out, tile, 2);
    }
    fputs(attn_score_ptx, out);
    fputs(attn_softmax_ptx, out);
    fputs(attn_fold_ptx, out);
    fputs(q_prep_ptx, out);
    fputs(kv_prep_ptx, out);
    fclose(out);
    return text;
}

/* =============================================================== */
/* Host plumbing. */

unsigned long long cuda_vram_free(CudaDevice *dev) {
    unsigned long long free_bytes = 0, total = 0;
    if (!dev->mem_info || dev->mem_info(&free_bytes, &total)) return 0;
    return free_bytes;
}

unsigned long long cuda_alloc(CudaDevice *dev, size_t bytes, char *err,
                              size_t errsz) {
    unsigned long long ptr = 0;
    if (!dev->mem_alloc || dev->mem_alloc(&ptr, bytes)) {
        snprintf(err, errsz, "gpu out of memory for %zu bytes", bytes);
        return 0;
    }
    return ptr;
}

void cuda_free(CudaDevice *dev, unsigned long long ptr) {
    if (dev->mem_free) dev->mem_free(ptr);
}

int cuda_copy_to(CudaDevice *dev, unsigned long long dst, const void *src,
                 size_t bytes) {
    if (!dev->memcpy_htd || dev->memcpy_htd(dst, src, bytes)) {
        FAULT(dev, "copy to gpu failed");
        return 0;
    }
    return 1;
}

/* The default-stream copy is synchronous: it drains the kernel queue, so a
   deferred execution fault lands here. */
int cuda_copy_from(CudaDevice *dev, void *dst, unsigned long long src,
                   size_t bytes) {
    if (!dev->memcpy_dth || dev->memcpy_dth(dst, src, bytes)) {
        FAULT(dev, "copy from gpu failed");
        return 0;
    }
    return 1;
}

int cuda_host_pin(CudaDevice *dev, void *ptr, size_t bytes) {
    return dev->host_register && !dev->host_register(ptr, bytes, 0);
}

void cuda_host_unpin(CudaDevice *dev, void *ptr) {
    if (dev->host_unregister) dev->host_unregister(ptr);
}

int cuda_sync(CudaDevice *dev) {
    if (!dev->ctx_sync || dev->ctx_sync()) {
        FAULT(dev, "gpu execution faulted");
        return 0;
    }
    return 1;
}

/* A launch that reports failure is fatal: cuda_start ran each kernel shape
   with checked results, so an in-flight failure means the driver is gone.
   Launches are asynchronous; the queue is drained by the copy that ends each
   attention call, which is where a deferred fault surfaces. */
static void launch(CudaDevice *dev, void *fn, unsigned grid_x, unsigned grid_y,
                   unsigned block, unsigned shared, void **params) {
    if (!dev->launch_kernel) {
        FAULT(dev, "CUDA driver API incomplete");
        return;
    }
    int rc = dev->launch_kernel(fn, grid_x, grid_y, 1, block, 1, 1, shared,
                                NULL, params, NULL);
    if (rc) {
        const char *why = NULL;
        if (dev->err_str && !dev->err_str(rc, &why) && why)
            FAULT(dev, "kernel launch failed: %s", why);
        else
            FAULT(dev, "kernel launch failed: %d", rc);
    }
}

void cuda_gemm(CudaDevice *dev, unsigned long long w, unsigned long long x,
               unsigned long long out, int n_in, int head_out, int n_head,
               int x_row_stride, int x_head_stride, int out_row_stride,
               int n_tokens, int type) {
    if (dev->fault[0]) return;

    int tile = 1;
    for (int t = 2; t <= GEMM_TILE_MAX; t *= 2)
        if (t <= n_tokens) tile = t;
    /* A row pair always lives inside one head when head_out is a
       multiple of 16 (or there is no head structure at all); anything
       else takes the single-row kernels. */
    int r = n_head == 1 || head_out % 16 == 0 ? 2 : 1;

    unsigned rows = (unsigned)head_out * (unsigned)n_head;
    unsigned rpb = (unsigned)(GEMM_ROWS_PER_BLOCK * r);
    unsigned tiles = (rows + rpb - 1) / rpb;
    unsigned resident =
        (unsigned)dev->gemm_max_blocks[__builtin_ctz(tile)][r - 1];
    unsigned grid_x = tiles < resident ? tiles : resident;
    unsigned grid_y = (unsigned)((n_tokens + tile - 1) / tile);

    unsigned long long w64 = w, x64 = x, out64 = out;
    unsigned n_in32 = n_in, head_out32 = head_out, n_head32 = n_head;
    unsigned x_str = x_row_stride, x_hstr = x_head_stride;
    unsigned out_str = out_row_stride;
    unsigned ntok = n_tokens, type32 = type;
    void *params[] = {&w64,      &x64,     &out64,    &n_in32,
                      &head_out32, &n_head32, &x_str,   &x_hstr,
                      &out_str,  &ntok,    &type32};
    launch(dev, dev->gemm_fn[__builtin_ctz(tile)][r - 1], grid_x, grid_y,
           GEMM_BLOCK, 0, params);
}

void cuda_attention(CudaDevice *dev, unsigned long long cache,
                    unsigned long long query, unsigned long long query_latent,
                    unsigned long long scores, unsigned long long out,
                    int position, int n_tokens, const CudaGeometry *geom,
                    unsigned long long geometry_on_device) {
    if (dev->fault[0]) return;

    unsigned attended = position + n_tokens;
    unsigned long long cache64 = cache, query64 = query, qlat64 = query_latent;
    unsigned long long scores64 = scores, geom64 = geometry_on_device;
    unsigned pos32 = position, ntok32 = n_tokens;
    void *score_params[] = {&cache64, &query64, &qlat64,  &scores64,
                            &pos32,   &ntok32,  &geom64};
    launch(dev, dev->score_fn, (attended + SCORE_TILE - 1) / SCORE_TILE,
           n_tokens, SCORE_BLOCK, 0, score_params);

    unsigned long long scores_in = scores;
    void *softmax_params[] = {&scores_in, &pos32, &ntok32, &geom64};
    launch(dev, dev->softmax_fn, geom->n_head, n_tokens, SOFTMAX_BLOCK, 0,
           softmax_params);

    unsigned long long out64 = out, cache_in = cache, scores_f = scores;
    void *fold_params[] = {&cache_in, &scores_f, &out64, &pos32, &ntok32,
                           &geom64};
    launch(dev, dev->fold_fn,
           (geom->fold_width + FOLD_R_TILE - 1) / FOLD_R_TILE, n_tokens,
           FOLD_BLOCK, 0, fold_params);

}

void cuda_q_prep(CudaDevice *dev, unsigned long long query,
                 unsigned long long cos_sin, unsigned long long weight,
                 unsigned long long geometry_on_device, int n_tokens) {
    if (dev->fault[0]) return;
    unsigned long long q64 = query, cos64 = cos_sin, w64 = weight;
    unsigned long long g64 = geometry_on_device;
    unsigned ntok = n_tokens;
    void *params[] = {&q64, &cos64, &w64, &ntok, &g64};
    launch(dev, dev->q_prep_fn, n_tokens, 1, 32, 0, params);
}

void cuda_kv_prep(CudaDevice *dev, unsigned long long kv_projected,
                  unsigned long long cos_sin, unsigned long long norm,
                  unsigned long long slot_base,
                  unsigned long long geometry_on_device, int n_tokens) {
    if (dev->fault[0]) return;
    unsigned long long kv64 = kv_projected, cos64 = cos_sin, n64 = norm;
    unsigned long long slot64 = slot_base, g64 = geometry_on_device;
    unsigned ntok = n_tokens;
    void *params[] = {&kv64, &cos64, &n64, &slot64, &ntok, &g64};
    launch(dev, dev->kv_prep_fn, n_tokens, 1, 32, 0, params);
}

/* =============================================================== */
/* Startup: context, module load, and a self-test that runs the gemm in
   both weight encodings against the cpu reference, so a kernel bug or a
   broken driver JIT fails here with a name, not as silent garbage later. */

#define SELF_TEST_N_IN 512
/* More row tiles than the device keeps resident, so the gemm's row-tile
   loop -- several tiles per block -- is exercised, not just its first
   iteration. */
#define SELF_TEST_ROWS 1023
#define SELF_TEST_TOKENS 11
#define SELF_TEST_TOLERANCE 1e-3

static int check_gemm(CudaDevice *dev, int type, char *err, size_t errsz) {
    int n = SELF_TEST_N_IN;
    int rows = SELF_TEST_ROWS;
    int n_tokens = SELF_TEST_TOKENS;
    /* The Q4_K and F16 encodings each carry their own row size. */
    size_t dev_row_bytes = type == CUDA_W_Q4K ? (size_t)n / 256 * 144
                           : type == CUDA_W_F16 ? (size_t)n * 2
                                                : (size_t)n * 4;
    float *x = malloc((size_t)n_tokens * n * sizeof *x);
    float *w = malloc((size_t)rows * n * sizeof *w);
    float *got = malloc((size_t)n_tokens * rows * sizeof *got);
    float *want = malloc((size_t)n_tokens * rows * sizeof *want);
    if (!x || !w || !got || !want) {
        snprintf(err, errsz, "out of memory for gpu self-test");
        free(x); free(w); free(got); free(want);
        return 0;
    }

    for (int i = 0; i < n_tokens * n; i++)
        x[i] = (float)(i % 13) - 6.0f;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < n; c++)
            w[(size_t)r * n + c] = (float)(((r * 7 + c * 3) % 21) - 10) / 8.0f;

    if (type == CUDA_W_Q4K) {
        /* Requantize the f32 rows as Q4_K blocks with d=1.5, dmin=0.25 (both
           exact in fp16), scale i+1 and min i, laid out the way
           get_scale_min_k4 in kernels.c unpacks them: bytes 0-3 carry scales
           0-3 and scales 4-7's top bits, bytes 4-7 the mins the same way,
           bytes 8-11 the low nibbles of scales 4-7 and mins 4-7. */
        uint8_t *packed = calloc((size_t)rows, (size_t)n / 256 * 144);
        if (!packed) {
            snprintf(err, errsz, "out of memory for gpu self-test");
            free(x); free(w); free(got); free(want);
            return 0;
        }
        for (int r = 0; r < rows; r++) {
            uint8_t *row = packed + (size_t)r * (n / 256) * 144;
            for (int b = 0; b < n / 256; b++) {
                uint8_t *block = row + b * 144;
                uint16_t d16 = 0x3E00, dmin16 = 0x3400;
                memcpy(block, &d16, 2);
                memcpy(block + 2, &dmin16, 2);
                for (int j = 0; j < 4; j++) {
                    int sc_low = (j + 1) * 7, mn_low = j * 7;
                    int sc_high = (j + 5) * 7, mn_high = (j + 4) * 7;
                    block[4 + j] =
                        (uint8_t)((sc_low & 63) | ((sc_high & 0x30) << 2));
                    block[8 + j] =
                        (uint8_t)((mn_low & 63) | ((mn_high & 0x30) << 2));
                    block[12 + j] =
                        (uint8_t)((sc_high & 0xF) | ((mn_high & 0xF) << 4));
                }
                for (int j = 0; j < 128; j++)
                    block[16 + j] =
                        (uint8_t)((j & 15) | (((j * 5 + b) % 16) << 4));
            }
        }
        free(w);
        w = (float *)packed;
    } else if (type == CUDA_W_F16) {
        uint16_t *packed = malloc((size_t)rows * n * 2);
        if (!packed) {
            snprintf(err, errsz, "out of memory for gpu self-test");
            free(x); free(w); free(got); free(want);
            return 0;
        }
        for (int i = 0; i < rows * n; i++) {
            packed[i] = fp32_to_fp16(w[i]);
            float back = fp16_to_fp32(packed[i]);
            float tolerance = 1e-3f * (1.0f + fabsf(w[i]));
            if (fabsf(back - w[i]) > tolerance) {
                snprintf(err, errsz,
                         "f32->f16 conversion drifted: %.7f became %.7f",
                         w[i], back);
                free(packed);
                free(x); free(w); free(got); free(want);
                return 0;
            }
        }
        free(w);
        w = (float *)packed;
        dev_row_bytes = (size_t)n * 2;
    }

    for (int r = 0; r < rows; r++) {
        float reference[SELF_TEST_N_IN];
        if (type == CUDA_W_Q4K)
            dequant_row((const uint8_t *)w + (size_t)r * dev_row_bytes,
                        GGML_TYPE_Q4_K, n, reference);
        else if (type == CUDA_W_F16) {
            const uint16_t *row =
                (const uint16_t *)w + (size_t)r * n;
            for (int c = 0; c < n; c++) reference[c] = fp16_to_fp32(row[c]);
        } else
            dequant_row((const uint8_t *)w + (size_t)r * dev_row_bytes,
                        GGML_TYPE_F32, n, reference);
        for (int t = 0; t < n_tokens; t++)
            want[(size_t)t * rows + r] =
                dot_f32(reference, x + (size_t)t * n, n);
    }

    unsigned long long dw = cuda_alloc(dev, (size_t)rows * dev_row_bytes,
                                       err, errsz);
    unsigned long long dx = cuda_alloc(dev, (size_t)n_tokens * n * 4, err, errsz);
    unsigned long long dout = cuda_alloc(dev, (size_t)n_tokens * rows * 4, err, errsz);
    if (!dw || !dx || !dout) {
        free(x); free(w); free(got); free(want);
        return 0;
    }
    if (!cuda_copy_to(dev, dw, w, (size_t)rows * dev_row_bytes) ||
        !cuda_copy_to(dev, dx, x, (size_t)n_tokens * n * 4)) {
        snprintf(err, errsz, "%s", dev->fault);
        free(x); free(w); free(got); free(want);
        return 0;
    }
    cuda_gemm(dev, dw, dx, dout, n, rows, 1, n, n, rows, n_tokens, type);
    if (cuda_fault(dev)) {
        snprintf(err, errsz, "%s", dev->fault);
        free(x); free(w); free(got); free(want);
        return 0;
    }
    int ok = cuda_copy_from(dev, got, dout, (size_t)n_tokens * rows * 4);
    if (ok) {
        for (int t = 0; t < n_tokens && ok; t++)
            for (int r = 0; r < rows; r++) {
                float got_one = got[(size_t)t * rows + r];
                float want_one = want[(size_t)t * rows + r];
                float tolerance =
                    SELF_TEST_TOLERANCE * (1.0f + fabsf(want_one));
                if (fabsf(got_one - want_one) > tolerance) {
                    snprintf(err, errsz,
                             "gpu gemm self-test mismatch (%s encoding): "
                             "token %d row %d got %.6f want %.6f",
                             type == CUDA_W_Q4K ? "q4_k"
                             : type == CUDA_W_F16 ? "f16" : "f32",
                             t, r, got_one, want_one);
                    ok = 0;
                    break;
                }
            }
    } else {
        snprintf(err, errsz, "%s", dev->fault);
    }

    cu_mem_free_t mem_free = sym(dev->cuda, "cuMemFree");
    if (mem_free) {
        mem_free(dw);
        mem_free(dx);
        mem_free(dout);
    }
    free(x);
    free(w);
    free(got);
    free(want);
    return ok;
}

/* The attention trio against a scalar reference over a toy geometry:
   two heads, a short context, and a chunk that starts mid-cache so the
   position masking is exercised on both ends. */
static int check_attention(CudaDevice *dev, int kind, char *err,
                           size_t errsz) {
    CudaGeometry g;
    memset(&g, 0, sizeof g);
    g.kind = kind;
    g.n_head = 32;
    g.n_head_kv = kind == ATTN_GQA ? 4 : 0;
    g.head_dim_k = 16;
    g.rank = kind == ATTN_MLA ? 64 : 0;
    g.rope_dim = kind == ATTN_MLA ? 8 : 0;
    g.n_segments = kind == ATTN_MLA ? 2 : 2 * g.n_head_kv;
    g.cache_width =
        kind == ATTN_MLA ? g.rank + g.rope_dim
                         : g.n_head_kv * (g.head_dim_k + 8);
    g.slot_bytes = g.cache_width + g.n_segments * 4;
    g.q_stride = g.n_head * g.head_dim_k;
    g.latent_stride = kind == ATTN_MLA ? g.n_head * g.rank : 0;
    g.out_stride = kind == ATTN_MLA ? g.n_head * g.rank
                                    : g.n_head * 8;
    g.fold_width = kind == ATTN_MLA ? g.rank : 8;
    g.kq_scale = 0.25f;
    g.n_ctx = 64;

    int position = 5, n_tokens = 20;
    int width = g.cache_width;

    size_t cache_bytes = (size_t)g.n_ctx * g.slot_bytes;
    size_t query_floats = (size_t)n_tokens * g.q_stride;
    size_t qlat_floats =
        kind == ATTN_MLA ? (size_t)n_tokens * g.latent_stride : query_floats;
    size_t out_floats = (size_t)n_tokens * g.out_stride;
    size_t scores_floats = (size_t)g.n_head * n_tokens * g.n_ctx;

    uint8_t *slot = malloc(cache_bytes);
    float *query = malloc(query_floats * 4);
    float *qlat = malloc(qlat_floats * 4);
    float *gpu_out = malloc(out_floats * 4);
    float *want = calloc(out_floats, 4);
    float *scores = malloc(scores_floats * 4);
    if (!slot || !query || !qlat || !gpu_out || !want || !scores) {
        snprintf(err, errsz, "out of memory for attention self-test");
        free(slot); free(query); free(qlat); free(gpu_out); free(want);
        free(scores);
        return 0;
    }

    uint32_t rng = 0x12345678u;
    for (size_t i = 0; i < cache_bytes; i++) {
        rng = rng * 1664525u + 1013904223u;
        slot[i] = (uint8_t)(rng >> 24);
    }
    for (size_t i = 0; i < cache_bytes / 4; i++)
        ((float *)slot)[i] = ((float)(slot[i * 4]) - 128.0f) / 512.0f;
    for (size_t i = 0; i < query_floats; i++) {
        rng = rng * 1664525u + 1013904223u;
        query[i] = (float)((int)(rng >> 27) - 16) / 8.0f;
    }
    for (size_t i = 0; i < qlat_floats; i++) {
        rng = rng * 1664525u + 1013904223u;
        qlat[i] = (float)((int)(rng >> 27) - 16) / 8.0f;
    }

    /* Reference: dequantize each slot segment with its own scale, score,
       softmax over the attended span only, fold. */
    float *want_probs_all =
        calloc((size_t)g.n_head * n_tokens * g.n_ctx, 4);
    for (int t = 0; t < n_tokens; t++) {
        int attended = position + t + 1;
        float *probs = malloc((size_t)attended * sizeof *probs);
        for (int head = 0; head < g.n_head; head++) {
            for (int p = 0; p < attended; p++) {
                const uint8_t *bytes = slot + (size_t)p * g.slot_bytes;
                const float *scales =
                    (const float *)(bytes + width);
                float dot = 0;
                if (kind == ATTN_MLA) {
                    for (int i = 0; i < g.rank; i++)
                        dot += ((float)bytes[i] - 128.0f) * scales[0] *
                               qlat[(size_t)t * g.latent_stride +
                                    (size_t)head * g.rank + i];
                    const float *rot =
                        query + (size_t)t * g.q_stride +
                        (size_t)head * g.head_dim_k + g.head_dim_k -
                        g.rope_dim;
                    for (int i = 0; i < g.rope_dim; i++)
                        dot += ((float)bytes[g.rank + i] - 128.0f) *
                               scales[1] * rot[i];
                } else {
                    int kv = head * g.n_head_kv / g.n_head;
                    for (int i = 0; i < g.head_dim_k; i++)
                        dot += ((float)bytes[kv * g.head_dim_k + i] -
                                128.0f) *
                               scales[kv] *
                               query[(size_t)t * g.q_stride +
                                     (size_t)head * g.head_dim_k + i];
                }
                probs[p] = dot * g.kq_scale;
            }
            float max = probs[0];
            for (int p = 1; p < attended; p++)
                if (probs[p] > max) max = probs[p];
            float sum = 0;
            for (int p = 0; p < attended; p++) {
                probs[p] = expf(probs[p] - max);
                sum += probs[p];
            }
            for (int p = 0; p < attended; p++) probs[p] /= sum;
            memcpy(want_probs_all + ((size_t)head * n_tokens + t) * g.n_ctx,
                   probs, (size_t)attended * sizeof *probs);
            for (int p = 0; p < attended; p++) {
                const uint8_t *bytes = slot + (size_t)p * g.slot_bytes;
                const float *scales = (const float *)(bytes + width);
                for (int i = 0; i < g.fold_width; i++) {
                    float value;
                    if (kind == ATTN_MLA)
                        value = ((float)bytes[i] - 128.0f) * scales[0];
                    else {
                        int kv = head * g.n_head_kv / g.n_head;
                        int kv_width = g.n_head_kv * g.head_dim_k;
                        value = ((float)bytes[kv_width +
                                              kv * g.fold_width + i] -
                                 128.0f) *
                                scales[g.n_head_kv + kv];
                    }
                    want[(size_t)t * g.out_stride + (size_t)head *
                                                        g.fold_width +
                         i] += probs[p] * value;
                }
            }
        }
        free(probs);
    }

    unsigned long long d_cache = cuda_alloc(dev, cache_bytes, err, errsz);
    unsigned long long d_query = cuda_alloc(dev, query_floats * 4, err, errsz);
    unsigned long long d_qlat = cuda_alloc(dev, qlat_floats * 4, err, errsz);
    unsigned long long d_scores =
        cuda_alloc(dev, scores_floats * 4, err, errsz);
    unsigned long long d_out = cuda_alloc(dev, out_floats * 4, err, errsz);
    unsigned long long d_geom = cuda_alloc(dev, sizeof g, err, errsz);
    if (!d_cache || !d_query || !d_qlat || !d_scores || !d_out || !d_geom) {
        free(slot); free(query); free(qlat); free(gpu_out); free(want);
        free(scores);
        return 0;
    }
    int ok = cuda_copy_to(dev, d_cache, slot, cache_bytes) &&
             cuda_copy_to(dev, d_query, query, query_floats * 4) &&
             cuda_copy_to(dev, d_qlat, qlat, qlat_floats * 4) &&
             cuda_copy_to(dev, d_geom, &g, sizeof g);
    if (ok) {
        cuda_attention(dev, d_cache, d_query, d_qlat, d_scores, d_out,
                       position, n_tokens, &g, d_geom);
        ok = !cuda_fault(dev) &&
             cuda_copy_from(dev, gpu_out, d_out, out_floats * 4);
    }
    if (ok) {
        ok = cuda_copy_from(dev, scores, d_scores, scores_floats * 4);
        if (ok) {
            for (int t = 0; ok && t < n_tokens; t++)
                for (int head = 0; ok && head < g.n_head; head++) {
                    int attended = position + t + 1;
                    const float *row =
                        scores + ((size_t)head * n_tokens + t) * g.n_ctx;
                    float *ref =
                        want_probs_all + (size_t)head * n_tokens *
                                             (size_t)(g.n_ctx) +
                        (size_t)t * g.n_ctx;
                    for (int p = 0; ok && p < attended; p++) {
                        float tolerance =
                            0.02f * (1.0f + fabsf(ref[p]));
                        if (fabsf(row[p] - ref[p]) > tolerance) {
                            snprintf(err, errsz,
                                     "gpu attention scores mismatch (%s): "
                                     "t=%d head=%d p=%d got %.6f want %.6f",
                                     kind == ATTN_MLA ? "mla" : "gqa", t,
                                     head, p, row[p], ref[p]);
                            ok = 0;
                        }
                    }
                }
        }
    }
    if (ok) {
        for (size_t i = 0; i < out_floats; i++) {
            float tolerance = 0.02f * (1.0f + fabsf(want[i]));
            if (fabsf(gpu_out[i] - want[i]) > tolerance) {
                snprintf(err, errsz,
                         "gpu attention self-test mismatch (%s): out[%zu] "
                         "got %.6f want %.6f",
                         kind == ATTN_MLA ? "mla" : "gqa", i, gpu_out[i],
                         want[i]);
                ok = 0;
                break;
            }
        }
    } else if (!err[0]) {
        snprintf(err, errsz, "%s", cuda_fault(dev));
    }

    cuda_free(dev, d_cache);
    cuda_free(dev, d_query);
    cuda_free(dev, d_qlat);
    cuda_free(dev, d_scores);
    cuda_free(dev, d_out);
    cuda_free(dev, d_geom);
    free(slot); free(query); free(qlat); free(gpu_out); free(want);
    free(scores); free(want_probs_all);
    return ok;
}

/* The gemm's stacked-matrix mode, which the plain self-test above does not
   reach: n_head matrices share one weight row array, and each head reads its
   own x segment -- the strides must not be confused with n_in. */
static int check_gemm_heads(CudaDevice *dev, int type, int head_out,
                           char *err, size_t errsz) {
    /* Q4_K rows must be whole 256-wide blocks, so the two encodings get
       different widths; x segments deliberately overlap (x_head_stride <
       n_in) to catch the strides being confused with n_in. head_out 16
       takes the row-paired kernels, 24 the single-row fallback. */
    const int n_head = 4;
    const int n_in = type == CUDA_W_Q4K ? 512 : 48;
    const int x_head_stride = type == CUDA_W_Q4K ? 128 : 24;
    const int n_tokens = 2;
    int rows = n_head * head_out;
    int x_row = (n_head - 1) * x_head_stride + n_in;

    size_t w_bytes = type == CUDA_W_Q4K
                         ? (size_t)rows * (n_in / 256) * 144
                         : (size_t)rows * n_in * 4;
    uint8_t *w = malloc(w_bytes);
    float *x = malloc((size_t)n_tokens * x_row * 4);
    float *got = malloc((size_t)n_tokens * rows * 4);
    float *want = malloc((size_t)n_tokens * rows * 4);
    if (!w || !x || !got || !want) {
        snprintf(err, errsz, "out of memory for gpu self-test");
        free(w); free(x); free(got); free(want);
        return 0;
    }

    for (int i = 0; i < n_tokens * x_row; i++)
        x[i] = (float)((i * 7) % 17 - 8) / 4.0f;
    for (size_t i = 0; i < w_bytes; i++)
        w[i] = (uint8_t)(i * 11 + 3);

    if (type == CUDA_W_Q4K) {
        for (int r = 0; r < rows; r++) {
            uint8_t *block = w + (size_t)r * (n_in / 256) * 144;
            uint16_t d16 = 0x3C00, dmin16 = 0x3800;
            memcpy(block, &d16, 2);
            memcpy(block + 2, &dmin16, 2);
            for (int j = 0; j < 12; j++) block[4 + j] = (uint8_t)(j * 5 + 1);
            for (int j = 0; j < 128; j++) block[16 + j] = (uint8_t)(j * 3);
        }
    }

    for (int t = 0; t < n_tokens; t++)
        for (int r = 0; r < rows; r++) {
            int head = r / head_out;
            float reference[1024];
            dequant_row(w + (size_t)r * (type == CUDA_W_Q4K
                                             ? (size_t)n_in / 256 * 144
                                             : (size_t)n_in * 4),
                        type == CUDA_W_Q4K ? GGML_TYPE_Q4_K : GGML_TYPE_F32,
                        n_in, reference);
            float dot = 0;
            for (int c = 0; c < n_in; c++)
                dot += reference[c] * x[(size_t)t * x_row + head * x_head_stride + c];
            want[(size_t)t * rows + r] = dot;
        }

    unsigned long long dw = cuda_alloc(dev, w_bytes, err, errsz);
    unsigned long long dx = cuda_alloc(dev, (size_t)n_tokens * x_row * 4, err, errsz);
    unsigned long long dout =
        cuda_alloc(dev, (size_t)n_tokens * rows * 4, err, errsz);
    if (!dw || !dx || !dout) {
        free(w); free(x); free(got); free(want);
        return 0;
    }
    int ok = cuda_copy_to(dev, dw, w, w_bytes) &&
             cuda_copy_to(dev, dx, x, (size_t)n_tokens * x_row * 4);
    if (ok) {
        cuda_gemm(dev, dw, dx, dout, n_in, head_out, n_head, x_row,
                  x_head_stride, rows, n_tokens, type);
        ok = !cuda_fault(dev) &&
             cuda_copy_from(dev, got, dout, (size_t)n_tokens * rows * 4);
    }
    if (ok) {
        for (size_t i = 0; i < (size_t)n_tokens * rows; i++) {
            float tolerance = 0.02f * (1.0f + fabsf(want[i]));
            if (fabsf(got[i] - want[i]) > tolerance) {
                snprintf(err, errsz,
                         "gpu stacked gemm mismatch (%s): out[%zu] got "
                         "%.6f want %.6f",
                         type == CUDA_W_Q4K ? "q4_k" : "f32", i, got[i],
                         want[i]);
                ok = 0;
                break;
            }
        }
    } else if (!err[0]) {
        snprintf(err, errsz, "%s", cuda_fault(dev));
    }

    cuda_free(dev, dw);
    cuda_free(dev, dx);
    cuda_free(dev, dout);
    free(w); free(x); free(got); free(want);
    return ok;
}

/* The prep kernels replace code the cpu tier still has, so the reference is
   that code: rmsnorm/rope/quantize_cache called the way the old cpu-side
   prepares called them. The gpu runs first, from the raw inputs; the cpu
   reference then works the same buffers in place, and the two outputs are
   compared. Token 0's projection is all zeros so the scale-0 path is
   exercised too. */
static int check_prep(CudaDevice *dev, int kind, char *err, size_t errsz) {
    CudaGeometry g;
    memset(&g, 0, sizeof g);
    g.kind = kind;
    g.n_head = 32;
    g.n_head_kv = kind == ATTN_GQA ? 4 : 0;
    g.head_dim_k = 16;
    g.rank = kind == ATTN_MLA ? 64 : 0;
    g.rope_dim = 8;
    g.n_segments = kind == ATTN_MLA ? 2 : 2 * g.n_head_kv;
    g.cache_width =
        kind == ATTN_MLA ? g.rank + g.rope_dim
                         : g.n_head_kv * (g.head_dim_k + 8);
    g.slot_bytes = g.cache_width + g.n_segments * 4;
    g.q_stride = g.n_head * g.head_dim_k;
    g.fold_width = kind == ATTN_MLA ? g.rank : 8;
    g.rms_eps = 1e-6f;
    g.n_ctx = 64;

    const int n_tokens = 20;
    int norm_width = kind == ATTN_MLA ? g.rank : g.head_dim_k;
    size_t query_bytes = (size_t)n_tokens * g.q_stride * 4;
    size_t kv_bytes = (size_t)n_tokens * g.cache_width * 4;
    size_t slot_bytes = (size_t)n_tokens * g.slot_bytes;
    float *query = malloc(query_bytes);
    float *query_got = malloc(query_bytes);
    float *kv = malloc(kv_bytes);
    float *qnorm = malloc((size_t)g.head_dim_k * 4);
    float *kvnorm = malloc((size_t)norm_width * 4);
    float *cos_sin = malloc((size_t)n_tokens * g.rope_dim * 4);
    uint8_t *slot_got = malloc(slot_bytes);
    uint8_t *slot_ref = calloc(1, slot_bytes);
    if (!query || !query_got || !kv || !qnorm || !kvnorm || !cos_sin ||
        !slot_got || !slot_ref) {
        snprintf(err, errsz, "out of memory for prep self-test");
        free(query); free(query_got); free(kv); free(qnorm); free(kvnorm);
        free(cos_sin); free(slot_got); free(slot_ref);
        return 0;
    }

    uint32_t rng = 0x9e3779b9u;
    for (size_t i = 0; i < query_bytes / 4; i++) {
        rng = rng * 1664525u + 1013904223u;
        query[i] = (float)((int)(rng >> 27) - 16) / 8.0f;
    }
    for (size_t i = 0; i < kv_bytes / 4; i++) {
        rng = rng * 1664525u + 1013904223u;
        kv[i] = (float)((int)(rng >> 27) - 16) / 8.0f;
    }
    memset(kv, 0, (size_t)g.cache_width * 4);
    for (int i = 0; i < g.head_dim_k; i++) {
        rng = rng * 1664525u + 1013904223u;
        qnorm[i] = 1.0f + (float)(rng >> 28) / 8.0f;
    }
    for (int i = 0; i < norm_width; i++) {
        rng = rng * 1664525u + 1013904223u;
        kvnorm[i] = 1.0f + (float)(rng >> 28) / 8.0f;
    }
    for (size_t i = 0; i < (size_t)n_tokens * g.rope_dim; i++) {
        rng = rng * 1664525u + 1013904223u;
        cos_sin[i] = (float)((int)(rng >> 27) - 16) / 16.0f;
    }

    unsigned long long d_query = cuda_alloc(dev, query_bytes, err, errsz);
    unsigned long long d_kv = cuda_alloc(dev, kv_bytes, err, errsz);
    unsigned long long d_qnorm =
        cuda_alloc(dev, (size_t)g.head_dim_k * 4, err, errsz);
    unsigned long long d_kvnorm =
        cuda_alloc(dev, (size_t)norm_width * 4, err, errsz);
    unsigned long long d_cos =
        cuda_alloc(dev, (size_t)n_tokens * g.rope_dim * 4, err, errsz);
    unsigned long long d_slot = cuda_alloc(dev, slot_bytes, err, errsz);
    unsigned long long d_geom = cuda_alloc(dev, sizeof g, err, errsz);
    int ok = d_query && d_kv && d_qnorm && d_kvnorm && d_cos && d_slot &&
             d_geom;
    if (ok)
        ok = cuda_copy_to(dev, d_query, query, query_bytes) &&
             cuda_copy_to(dev, d_kv, kv, kv_bytes) &&
             cuda_copy_to(dev, d_qnorm, qnorm, (size_t)g.head_dim_k * 4) &&
             cuda_copy_to(dev, d_kvnorm, kvnorm, (size_t)norm_width * 4) &&
             cuda_copy_to(dev, d_cos, cos_sin,
                          (size_t)n_tokens * g.rope_dim * 4) &&
             cuda_copy_to(dev, d_geom, &g, sizeof g);
    if (ok) {
        cuda_q_prep(dev, d_query, d_cos, d_qnorm, d_geom, n_tokens);
        cuda_kv_prep(dev, d_kv, d_cos, d_kvnorm, d_slot, d_geom, n_tokens);
        ok = !cuda_fault(dev) &&
             cuda_copy_from(dev, query_got, d_query, query_bytes) &&
             cuda_copy_from(dev, slot_got, d_slot, slot_bytes);
    }
    cuda_free(dev, d_query);
    cuda_free(dev, d_kv);
    cuda_free(dev, d_qnorm);
    cuda_free(dev, d_kvnorm);
    cuda_free(dev, d_cos);
    cuda_free(dev, d_slot);
    cuda_free(dev, d_geom);
    if (!ok) {
        if (!err[0]) snprintf(err, errsz, "%s", cuda_fault(dev));
        free(query); free(query_got); free(kv); free(qnorm); free(kvnorm);
        free(cos_sin); free(slot_got); free(slot_ref);
        return 0;
    }

    for (int t = 0; t < n_tokens; t++) {
        const float *cos_sin_t = cos_sin + (size_t)t * g.rope_dim;
        for (int head = 0; head < g.n_head; head++) {
            float *q = query + (size_t)t * g.q_stride +
                       (size_t)head * g.head_dim_k;
            if (kind == ATTN_GQA) {
                rmsnorm(q, q, qnorm, g.head_dim_k, g.rms_eps);
                rope_apply_neox(q, cos_sin_t, g.rope_dim);
            } else {
                rope_apply(q + g.head_dim_k - g.rope_dim, cos_sin_t,
                           g.rope_dim);
            }
        }
        float *projected = kv + (size_t)t * g.cache_width;
        uint8_t *slot = slot_ref + (size_t)t * g.slot_bytes;
        float *scales = (float *)(slot + g.cache_width);
        if (kind == ATTN_MLA) {
            float normed[64];
            rmsnorm(normed, projected, kvnorm, g.rank, g.rms_eps);
            rope_apply(projected + g.rank, cos_sin_t, g.rope_dim);
            scales[0] = quantize_cache(slot, normed, g.rank);
            scales[1] =
                quantize_cache(slot + g.rank, projected + g.rank, g.rope_dim);
        } else {
            int kv_width = g.n_head_kv * g.head_dim_k;
            for (int kvh = 0; kvh < g.n_head_kv; kvh++) {
                float *k = projected + (size_t)kvh * g.head_dim_k;
                rmsnorm(k, k, kvnorm, g.head_dim_k, g.rms_eps);
                rope_apply_neox(k, cos_sin_t, g.rope_dim);
                scales[kvh] =
                    quantize_cache(slot + (size_t)kvh * g.head_dim_k, k,
                                   g.head_dim_k);
                scales[g.n_head_kv + kvh] =
                    quantize_cache(slot + kv_width + (size_t)kvh * 8,
                                   projected + kv_width + (size_t)kvh * 8, 8);
            }
        }
    }

    ok = 1;
    for (size_t i = 0; i < query_bytes / 4 && ok; i++) {
        float tolerance = 1e-4f * (1.0f + fabsf(query[i]));
        if (fabsf(query_got[i] - query[i]) > tolerance) {
            snprintf(err, errsz,
                     "gpu q_prep mismatch (%s): query[%zu] got %.6f want %.6f",
                     kind == ATTN_MLA ? "mla" : "gqa", i, query_got[i],
                     query[i]);
            ok = 0;
        }
    }
    /* The scales are floats compared on their own terms below; a last-ulp
       scale difference would otherwise fail the byte rule without meaning
       anything. */
    for (int t = 0; t < n_tokens && ok; t++)
        for (int i = 0; i < g.cache_width && ok; i++) {
            size_t at = (size_t)t * g.slot_bytes + i;
            if ((int)slot_got[at] - (int)slot_ref[at] > 1 ||
                (int)slot_ref[at] - (int)slot_got[at] > 1) {
                snprintf(err, errsz,
                         "gpu kv_prep mismatch (%s): token %d slot byte %d "
                         "got %d want %d",
                         kind == ATTN_MLA ? "mla" : "gqa", t, i, slot_got[at],
                         slot_ref[at]);
                ok = 0;
            }
        }
    for (int t = 0; t < n_tokens && ok; t++) {
        const float *got = (const float *)(slot_got + (size_t)t * g.slot_bytes +
                                          g.cache_width);
        const float *want = (const float *)(slot_ref +
                                            (size_t)t * g.slot_bytes +
                                            g.cache_width);
        for (int s = 0; s < g.n_segments; s++) {
            float tolerance = 1e-5f * (1.0f + fabsf(want[s]));
            if (fabsf(got[s] - want[s]) > tolerance) {
                snprintf(err, errsz,
                         "gpu kv_prep scale mismatch (%s): token %d segment "
                         "%d got %.7f want %.7f",
                         kind == ATTN_MLA ? "mla" : "gqa", t, s, got[s],
                         want[s]);
                ok = 0;
                break;
            }
        }
    }

    free(query); free(query_got); free(kv); free(qnorm); free(kvnorm);
    free(cos_sin); free(slot_got); free(slot_ref);
    return ok;
}

int cuda_start(CudaDevice **out, char *err, size_t errsz) {
    CudaDevice *dev = calloc(1, sizeof *dev);
    if (!dev) {
        snprintf(err, errsz, "out of memory");
        return 0;
    }
    dev->cuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!dev->cuda) {
        snprintf(err, errsz, "no CUDA driver: %s", dlerror());
        free(dev);
        return 0;
    }

    cu_init_t init = sym(dev->cuda, "cuInit");
    cu_dev_get_t device_get = sym(dev->cuda, "cuDeviceGet");
    cu_ctx_create_t ctx_create = sym(dev->cuda, "cuCtxCreate");
    cu_module_load_t module_load = sym(dev->cuda, "cuModuleLoadData");
    cu_func_get_t func_get = sym(dev->cuda, "cuModuleGetFunction");
    dev->mem_alloc = sym(dev->cuda, "cuMemAlloc");
    dev->mem_free = sym(dev->cuda, "cuMemFree");
    dev->mem_info = sym(dev->cuda, "cuMemGetInfo");
    dev->memcpy_htd = sym(dev->cuda, "cuMemcpyHtoD");
    dev->memcpy_dth = sym(dev->cuda, "cuMemcpyDtoH");
    dev->host_register = sym(dev->cuda, "cuMemHostRegister");
    dev->host_unregister = sym(dev->cuda, "cuMemHostUnregister");
    dev->launch_kernel = sym(dev->cuda, "cuLaunchKernel");
    dev->ctx_sync = sym(dev->cuda, "cuCtxSynchronize");
    dev->err_str = sym(dev->cuda, "cuGetErrorString");
    dev->dev_attr = sym(dev->cuda, "cuDeviceGetAttribute");
    dev->occupancy = sym(dev->cuda, "cuOccupancyMaxActiveBlocksPerMultiprocessor");
    /* Until the device says otherwise, every tile launches one block per
       row tile -- the shape that cannot pipeline. */
    for (int i = 0; i < GEMM_TILE_COUNT; i++)
        for (int j = 0; j < 2; j++) dev->gemm_max_blocks[i][j] = -1;
    if (!init || !device_get || !ctx_create || !module_load || !func_get ||
        !dev->mem_alloc || !dev->mem_free || !dev->mem_info ||
        !dev->memcpy_htd || !dev->memcpy_dth || !dev->host_register ||
        !dev->host_unregister || !dev->launch_kernel || !dev->ctx_sync ||
        !dev->err_str) {
        snprintf(err, errsz, "CUDA driver API incomplete");
        cuda_stop(dev);
        return 0;
    }

    /* cuDevicePrimaryCtxRetain leaves allocations failing with "invalid
       device context" on some driver/gpu combos (seen: driver 580, Maxwell);
       cuCtxCreate works everywhere. */
    if (init(0) || device_get(&dev->device, 0) || ctx_create(&dev->ctx, 0, 0)) {
        snprintf(err, errsz, "could not create a CUDA context");
        cuda_stop(dev);
        return 0;
    }

    char *module_text = build_module();
    if (!module_text) {
        snprintf(err, errsz, "out of memory for kernel module");
        cuda_stop(dev);
        return 0;
    }
/* The JIT's own log says which line of PTX it choked on; capture it so
       a broken kernel reports its cause instead of a bare boolean. */
    typedef int (*cu_module_load_ex_t)(void **, const void *, unsigned,
                                       const int *options, void *values[]);
    cu_module_load_ex_t module_load_ex =
        sym(dev->cuda, "cuModuleLoadDataEx");
    enum { CU_JIT_ERROR_LOG_BUFFER = 5,
           CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6 };
char jit_log[2048] = "";
    int jit_options[2] = {CU_JIT_ERROR_LOG_BUFFER,
                          CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void *jit_values[2] = {jit_log, (void *)(sizeof jit_log - 1)};
    int rc = module_load_ex ? module_load_ex(&dev->module, module_text, 2,
                                               jit_options, jit_values)
                             : 1;
    if (rc) {
        FILE *dump = fopen("/tmp/geode-kernels.ptx", "w");
        if (dump) {
            fwrite(module_text, 1, strlen(module_text), dump);
            fclose(dump);
        }
        cu_err_str_t err_str = sym(dev->cuda, "cuGetErrorString");
        const char *why = NULL;
        if (err_str && !err_str(rc, &why) && why)
            snprintf(err, errsz, "kernel JIT failed: %s %s", why, jit_log);
        else
            snprintf(err, errsz, "kernel JIT failed: %d %s", rc, jit_log);
        cuda_stop(dev);
        return 0;
    }
    free(module_text);

    int ok = 1;
    for (int tile = 1, i = 0; tile <= GEMM_TILE_MAX; tile *= 2, i++) {
        for (int j = 0; j < 2; j++) {
            char name[32];
            snprintf(name, sizeof name, "geode_gemm_t%d_r%d", tile, j + 1);
            ok &= func_get(&dev->gemm_fn[i][j], dev->module, name) == 0;
        }
    }
    ok &= func_get(&dev->score_fn, dev->module, "geode_attn_score") == 0;
    ok &= func_get(&dev->softmax_fn, dev->module, "geode_attn_softmax") == 0;
    ok &= func_get(&dev->fold_fn, dev->module, "geode_attn_fold") == 0;
    ok &= func_get(&dev->q_prep_fn, dev->module, "geode_q_prep") == 0;
    ok &= func_get(&dev->kv_prep_fn, dev->module, "geode_kv_prep") == 0;
    if (!ok) {
        snprintf(err, errsz, "kernel module is missing an entry point");
        cuda_stop(dev);
        return 0;
    }

    /* How many gemm blocks stay resident decides the launch width; missing
       calls fall back to one block per row tile, correct but unpipelined. */
    enum { CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16 };
    if (dev->dev_attr && dev->occupancy &&
        !dev->dev_attr(&dev->n_sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                       dev->device) &&
        dev->n_sm > 0)
        for (int i = 0; i < GEMM_TILE_COUNT; i++)
            for (int j = 0; j < 2; j++) {
                int per_sm = 0;
                if (dev->occupancy(&per_sm, dev->gemm_fn[i][j], GEMM_BLOCK,
                                   0) ||
                    per_sm < 1)
                    continue; /* leave the fallback */
                dev->gemm_max_blocks[i][j] = per_sm * dev->n_sm;
            }


    if (!check_gemm(dev, CUDA_W_F32, err, errsz) ||
        !check_gemm(dev, CUDA_W_Q4K, err, errsz) ||
        !check_gemm(dev, CUDA_W_F16, err, errsz) ||
        !check_gemm_heads(dev, CUDA_W_F32, 16, err, errsz) ||
        !check_gemm_heads(dev, CUDA_W_Q4K, 16, err, errsz) ||
        !check_gemm_heads(dev, CUDA_W_F32, 24, err, errsz) ||
        !check_gemm_heads(dev, CUDA_W_Q4K, 24, err, errsz) ||
        !check_attention(dev, ATTN_MLA, err, errsz) ||
        !check_attention(dev, ATTN_GQA, err, errsz) ||
        !check_prep(dev, ATTN_MLA, err, errsz) ||
        !check_prep(dev, ATTN_GQA, err, errsz)) {
        cuda_stop(dev);
        return 0;
    }

    *out = dev;
    return 1;
}

void cuda_stop(CudaDevice *dev) {
    if (!dev) return;
    if (dev->module) {
        typedef int (*cu_module_unload_t)(void *);
        cu_module_unload_t unload = sym(dev->cuda, "cuModuleUnload");
        if (unload) unload(dev->module);
    }
    if (dev->ctx) {
        cu_ctx_destroy_t destroy = sym(dev->cuda, "cuCtxDestroy");
        if (destroy) destroy(dev->ctx);
    }
    if (dev->cuda) dlclose(dev->cuda);
    free(dev);
}