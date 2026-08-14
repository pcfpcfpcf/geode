#include "gpu.h"

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CUDA_GPU_MAX 8
#define COPY_BYTES (128u << 20)

typedef struct {
    char busId[32];
    unsigned domain, bus, device;
    unsigned pciDeviceId, pciSubSystemId;
    unsigned reserved0, reserved1, reserved2;
    unsigned extPciDeviceId, extPciSubSystemId;
    char vendorName[32];
    unsigned reserved3, reserved4, reserved5;
} NvmlPciInfo;

typedef struct {
    unsigned long long total, free, used;
} NvmlMemory;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *sym(void *lib, const char *name) {
    return lib ? dlsym(lib, name) : NULL;
}

static int pcie_link_of(const char *bus_id, int *gen, int *width) {
    char path[256];
    snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/current_link_speed",
             bus_id);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    double gt_s;
    if (fscanf(f, "%lf", &gt_s) != 1) {
        fclose(f);
        return -1;
    }
    fclose(f);
    static const double gens[] = {2.5, 5.0, 8.0, 16.0, 32.0, 64.0};
    *gen = 0;
    for (size_t i = 0; i < sizeof gens / sizeof *gens; i++)
        if (gt_s >= gens[i]) *gen = (int)i + 1;
    *width = 0;
    snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/current_link_width",
             bus_id);
    f = fopen(path, "r");
    if (f) {
        if (fscanf(f, "%d", width) != 1) *width = 0;
        fclose(f);
    }
    return 0;
}

/* cuDevicePrimaryCtxRetain + cuCtxSetCurrent leaves allocations failing
   with "invalid device context" on some driver/gpu combos (seen: driver
   580, Maxwell); cuCtxCreate works everywhere. */
static int cuda_context(void *cuda, int ordinal, void **ctx_out, char *err,
                        size_t errsz) {
    typedef int (*cuinit_t)(unsigned);
    typedef int (*cuget_t)(int *, int);
    typedef int (*cuctx_t)(void **, unsigned, int);
    typedef int (*curc_t)(int, const char **);

    cuinit_t cuInit = sym(cuda, "cuInit");
    cuget_t cuDeviceGet = sym(cuda, "cuDeviceGet");
    cuctx_t cuCtxCreate = sym(cuda, "cuCtxCreate");
    curc_t cuGetError = sym(cuda, "cuGetErrorString");
    if (!cuInit || !cuDeviceGet || !cuCtxCreate) {
        snprintf(err, errsz, "CUDA driver API incomplete");
        return -1;
    }
    int r = cuInit(0);
    if (r) {
        snprintf(err, errsz, "cuInit: %d", r);
        return -1;
    }
    int dev;
    if (cuDeviceGet(&dev, ordinal)) {
        snprintf(err, errsz, "cuDeviceGet failed");
        return -1;
    }
    r = cuCtxCreate(ctx_out, 0, dev);
    if (r || !*ctx_out) {
        const char *es = NULL;
        if (cuGetError && !cuGetError(r, &es) && es)
            snprintf(err, errsz, "cuCtxCreate: %d (%s)", r, es);
        else
            snprintf(err, errsz, "cuCtxCreate: %d", r);
        return -1;
    }
    return 0;
}

static int cuda_bandwidth(void *cuda, double *hbm, double *pcie_bw,
                          char *err, size_t errsz) {
    *hbm = 0.0;
    *pcie_bw = 0.0;
    typedef int (*cumem_t)(unsigned long long *, size_t);
    typedef int (*cumemhost_t)(void **, size_t);
    typedef int (*cucopy_t)(unsigned long long, unsigned long long, size_t);
    typedef int (*cucopyhh_t)(unsigned long long, void *, size_t);
    typedef int (*cucopyh_t)(void *, unsigned long long, size_t);
    typedef int (*curc_t)(int, const char **);

    cumem_t cuMemAlloc = sym(cuda, "cuMemAlloc");
    cumemhost_t cuMemAllocHost = sym(cuda, "cuMemAllocHost");
    cucopy_t cuDtoD = sym(cuda, "cuMemcpyDtoD");
    cucopyhh_t cuHtoD = sym(cuda, "cuMemcpyHtoD");
    cucopyh_t cuDtoH = sym(cuda, "cuMemcpyDtoH");
    typedef int (*cusync_t)(void);
    cusync_t cuCtxSynchronize = sym(cuda, "cuCtxSynchronize");
    curc_t cuGetError = sym(cuda, "cuGetErrorString");

    if (!cuMemAlloc || !cuMemAllocHost || !cuDtoD || !cuHtoD || !cuDtoH ||
        !cuCtxSynchronize) {
        snprintf(err, errsz, "CUDA driver API incomplete");
        return -1;
    }

    unsigned long long d1 = 0, d2 = 0;
    int r = cuMemAlloc(&d1, COPY_BYTES);
    if (r) {
        sprintf(err, "cuMemAlloc: %d", r);
        const char *es = NULL;
        if (cuGetError && !cuGetError(r, &es) && es)
            sprintf(err, "cuMemAlloc: %d (%s)", r, es);
        return -1;
    }
    if (cuMemAlloc(&d2, COPY_BYTES)) {
        sprintf(err, "cuMemAlloc: %d", r);
        return -1;
    }
    void *host = NULL;
    if (cuMemAllocHost(&host, COPY_BYTES)) {
        sprintf(err, "cuMemAllocHost failed");
        return -1;
    }

    cuDtoD(d2, d1, COPY_BYTES);
    cuCtxSynchronize();
    double t0 = now_s();
    for (int i = 0; i < 4; i++) cuDtoD(d2, d1, COPY_BYTES);
    cuCtxSynchronize();
    double dt = now_s() - t0;
    if (dt > 0) *hbm = 4.0 * COPY_BYTES / dt;

    t0 = now_s();
    for (int i = 0; i < 4; i++) {
        cuHtoD(d1, host, COPY_BYTES);
        cuDtoH(host, d2, COPY_BYTES);
    }
    cuCtxSynchronize();
    dt = now_s() - t0;
    if (dt > 0) *pcie_bw = 8.0 * COPY_BYTES / dt;
    return 0;
}

/* FP32 FMA throughput measured by a kernel the driver JITs from embedded
   PTX, so any compute capability works without a build-time toolchain.
   Loop trip count is fixed; on datacenter GPUs the launch may be too short
   to amortize overhead, which only makes the number pessimistic. */
#define FLOPS_KERNEL_ITERS 200000
#define FLOPS_ACCUMULATORS 12
#define FLOPS_THREADS 256
#define FLOPS_BLOCKS_PER_SM 8
#define FLOPS_BENCH_SECONDS 0.5

static const char flops_ptx[] =
    ".version 6.0\n"
    ".target sm_50\n"
    ".address_size 64\n"
    ".visible .entry geode_flops_kernel(.param .u64 geode_out)\n"
    "{\n"
    "    .reg .pred %p<2>;\n"
    "    .reg .f32 %f<16>;\n"
    "    .reg .u32 %r<5>;\n"
    "    .reg .u64 %rd<3>;\n"
    "    ld.param.u64 %rd1, [geode_out];\n"
    "    mov.u32 %r1, %ctaid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mov.u32 %r3, %tid.x;\n"
    "    mad.lo.s32 %r1, %r1, %r2, %r3;\n"
    "    mov.f32 %f1, 0f3F800347;\n"  /* 1.0001f */
    "    mov.f32 %f2, 0f3A83126F;\n"  /* 0.001f */
    "    cvt.rn.f32.u32 %f3, %r1;\n"
    "    mul.f32 %f3, %f3, %f2;\n"
    "    mov.f32 %f4, %f3;\n"
    "    mov.f32 %f5, %f3;\n"
    "    mov.f32 %f6, %f3;\n"
    "    mov.f32 %f7, %f3;\n"
    "    mov.f32 %f8, %f3;\n"
    "    mov.f32 %f9, %f3;\n"
    "    mov.f32 %f10, %f3;\n"
    "    mov.f32 %f11, %f3;\n"
    "    mov.f32 %f12, %f3;\n"
    "    mov.f32 %f13, %f3;\n"
    "    mov.f32 %f14, %f3;\n"
    "    mov.u32 %r4, 200000;\n"
    "$L_geode_loop:\n"
    "    fma.rn.f32 %f3, %f3, %f1, %f2;\n"
    "    fma.rn.f32 %f4, %f4, %f1, %f2;\n"
    "    fma.rn.f32 %f5, %f5, %f1, %f2;\n"
    "    fma.rn.f32 %f6, %f6, %f1, %f2;\n"
    "    fma.rn.f32 %f7, %f7, %f1, %f2;\n"
    "    fma.rn.f32 %f8, %f8, %f1, %f2;\n"
    "    fma.rn.f32 %f9, %f9, %f1, %f2;\n"
    "    fma.rn.f32 %f10, %f10, %f1, %f2;\n"
    "    fma.rn.f32 %f11, %f11, %f1, %f2;\n"
    "    fma.rn.f32 %f12, %f12, %f1, %f2;\n"
    "    fma.rn.f32 %f13, %f13, %f1, %f2;\n"
    "    fma.rn.f32 %f14, %f14, %f1, %f2;\n"
    "    add.s32 %r4, %r4, -1;\n"
    "    setp.ne.s32 %p1, %r4, 0;\n"
    "    @%p1 bra $L_geode_loop;\n"
    "    add.f32 %f3, %f3, %f4;\n"
    "    add.f32 %f5, %f5, %f6;\n"
    "    add.f32 %f7, %f7, %f8;\n"
    "    add.f32 %f9, %f9, %f10;\n"
    "    add.f32 %f11, %f11, %f12;\n"
    "    add.f32 %f13, %f13, %f14;\n"
    "    add.f32 %f3, %f3, %f5;\n"
    "    add.f32 %f7, %f7, %f9;\n"
    "    add.f32 %f11, %f11, %f13;\n"
    "    add.f32 %f3, %f3, %f7;\n"
    "    add.f32 %f3, %f3, %f11;\n"
    "    cvt.u64.u32 %rd2, %r1;\n"
    "    shl.b64 %rd2, %rd2, 2;\n"
    "    add.s64 %rd2, %rd1, %rd2;\n"
    "    st.global.f32 [%rd2], %f3;\n"
    "    ret;\n"
    "}\n";

/* Q4_K dequant+dot shaped kernel, warp-cooperative like a real mmq GEMM:
   the activation tile lives in shared memory with one row per lane (full
   bank utilization), and each warp dequantizes its weight blocks into a
   shared staging area once per sub-block. A naive thread-per-block layout
   measures ~10% of peak: every lane re-reads the same activations from L2
   and the kernel goes load-bound. Min correction is dropped: values are
   garbage either way, only the instruction stream matters. */
#define DQ_THREADS 512          /* 16 warps per block */
#define DQ_PAIRS_PER_WARP 128   /* block-pairs per warp per launch */
#define DQ_BENCH_SECONDS 0.5

static const char dequant_ptx[] =
    ".version 6.0\n"
    ".target sm_50\n"
    ".address_size 64\n"
    /* Row stride is padded to 260 floats: at 256 every row starts in bank
       0 and all 32 lanes conflict 32-way on every float4 load. */
    ".shared .align 16 .b8 geode_x[33280];\n"   /* 32 rows x 260 floats */
    ".shared .align 16 .b8 geode_q[4096];\n"    /* 16 warps x 64 floats */
    ".visible .entry geode_dequant_kernel(\n"
    "    .param .u64 p_weights, .param .u64 p_x, .param .u64 p_out)\n"
    "{\n"
    "    .reg .pred %p<4>;\n"
    "    .reg .f32 %f<24>;\n"
    "    .reg .u32 %r<20>;\n"
    "    .reg .u64 %rd<12>;\n"
    "    ld.param.u64 %rd1, [p_weights];\n"
    "    ld.param.u64 %rd2, [p_x];\n"
    "    ld.param.u64 %rd3, [p_out];\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    and.b32 %r2, %r1, 31;\n"                /* lane */
    "    shr.u32 %r3, %r1, 5;\n"                 /* warp in block */
    "    mov.u32 %r13, geode_x;\n"
    "    mov.u32 %r14, geode_q;\n"
    /* stage the activation tile: 512 threads x 16 floats, packed global
       rows into padded shared rows */
    "    shr.u32 %r15, %r1, 4;\n"                 /* row = tid/16 */
    "    mul.lo.u32 %r15, %r15, 1040;\n"
    "    and.b32 %r16, %r1, 15;\n"
    "    shl.b32 %r16, %r16, 6;\n"                /* col byte = (tid%16)*64 */
    "    add.u32 %r15, %r15, %r16;\n"
    "    add.u32 %r15, %r15, %r13;\n"
    "    mul.wide.u32 %rd7, %r1, 64;\n"
    "    add.s64 %rd7, %rd7, %rd2;\n"
    "    ld.global.v4.f32 {%f3,%f4,%f5,%f6}, [%rd7];\n"
    "    st.shared.v4.f32 [%r15], {%f3,%f4,%f5,%f6};\n"
    "    ld.global.v4.f32 {%f3,%f4,%f5,%f6}, [%rd7+16];\n"
    "    st.shared.v4.f32 [%r15+16], {%f3,%f4,%f5,%f6};\n"
    "    ld.global.v4.f32 {%f3,%f4,%f5,%f6}, [%rd7+32];\n"
    "    st.shared.v4.f32 [%r15+32], {%f3,%f4,%f5,%f6};\n"
    "    ld.global.v4.f32 {%f3,%f4,%f5,%f6}, [%rd7+48];\n"
    "    st.shared.v4.f32 [%r15+48], {%f3,%f4,%f5,%f6};\n"
    "    bar.sync 0;\n"
    /* per-lane x row base, per-warp staging base */
    "    mul.lo.u32 %r11, %r2, 1040;\n"
    "    add.u32 %r11, %r11, %r13;\n"
    "    mul.lo.u32 %r12, %r3, 256;\n"
    "    add.u32 %r12, %r12, %r14;\n"
    /* weights: warp global id x (pairs x 2 blocks x 144B) */
    "    mov.u32 %r4, %ctaid.x;\n"
    "    shl.b32 %r4, %r4, 4;\n"
    "    add.u32 %r4, %r4, %r3;\n"
    "    mul.wide.u32 %rd4, %r4, 36864;\n"       /* wgid x 128 pairs x 288B */
    "    add.s64 %rd4, %rd4, %rd1;\n"
    "    mov.f32 %f12, 0f00000000;\n"
    "    mov.f32 %f13, 0f00000000;\n"
    "    mov.f32 %f14, 0f00000000;\n"
    "    mov.f32 %f15, 0f00000000;\n"
    "    mov.u32 %r7, 128;\n"                    /* pairs remaining */
    "$L_dq_pair:\n"
    /* running addresses, advanced per sub-block: value bytes, scales, x */
    "    shr.u32 %r10, %r2, 1;\n"                /* lane/2: byte in sub-block */
    "    cvt.u64.u32 %rd5, %r10;\n"
    "    add.s64 %rd5, %rd4, %rd5;\n"
    "    add.s64 %rd5, %rd5, 16;\n"              /* qs + lane/2 */
    "    mov.u64 %rd6, %rd4;\n"
    "    add.s64 %rd6, %rd6, 4;\n"               /* scales */
    "    mov.u32 %r17, %r11;\n"                  /* x row base */
    "    mov.u32 %r5, 0;\n"                      /* sub-block j */
    "$L_dq_j:\n"
    /* unpack one value per lane per block, scale folded, into staging */
    "    ld.global.u8 %r8, [%rd5];\n"            /* block 0 value byte */
    "    ld.global.u8 %r9, [%rd5+144];\n"        /* block 1 value byte */
    "    and.b32 %r10, %r2, 1;\n"
    "    setp.eq.s32 %p1, %r10, 0;\n"
    "    and.b32 %r10, %r8, 15;\n"
    "    shr.u32 %r8, %r8, 4;\n"
    "    selp.b32 %r8, %r10, %r8, %p1;\n"        /* even lane: lo, odd: hi */
    "    and.b32 %r10, %r9, 15;\n"
    "    shr.u32 %r9, %r9, 4;\n"
    "    selp.b32 %r9, %r10, %r9, %p1;\n"
    "    ld.global.u8 %r10, [%rd6];\n"           /* block 0 scales[j] */
    "    ld.global.u8 %r16, [%rd6+144];\n"       /* block 1 scales[j] */
    "    and.b32 %r10, %r10, 63;\n"
    "    and.b32 %r16, %r16, 63;\n"
    "    cvt.rn.f32.u32 %f1, %r10;\n"
    "    cvt.rn.f32.u32 %f2, %r16;\n"
    "    cvt.rn.f32.u32 %f11, %r8;\n"
    "    mul.f32 %f11, %f11, %f1;\n"
    "    cvt.rn.f32.u32 %f16, %r9;\n"
    "    mul.f32 %f16, %f16, %f2;\n"
    "    mul.lo.u32 %r15, %r2, 4;\n"
    "    add.u32 %r15, %r15, %r12;\n"
    "    st.shared.f32 [%r15], %f11;\n"
    "    st.shared.f32 [%r15+128], %f16;\n"
    "    bar.warp.sync 0xffffffff;\n"
    /* row dot: lane's row against all 64 staged quants, 8 float4 chunks */
    "    mov.u32 %r15, %r17;\n"
#define DQ_CHUNK(off, acc)                                                \
    "    ld.shared.v4.f32 {%f3,%f4,%f5,%f6}, [%r15+" #off "];\n"          \
    "    ld.shared.v4.f32 {%f7,%f8,%f9,%f10}, [%r12+" #off "];\n"         \
    "    fma.rn.f32 %f" #acc ", %f7, %f3, %f" #acc ";\n"                  \
    "    fma.rn.f32 %f" #acc ", %f8, %f4, %f" #acc ";\n"                  \
    "    fma.rn.f32 %f" #acc ", %f9, %f5, %f" #acc ";\n"                  \
    "    fma.rn.f32 %f" #acc ", %f10, %f6, %f" #acc ";\n"                 \
    "    ld.shared.v4.f32 {%f7,%f8,%f9,%f10}, [%r12+" #off "+128];\n"     \
    "    fma.rn.f32 %f" #acc ", %f7, %f3, %f" #acc ";\n"                  \
    "    fma.rn.f32 %f" #acc ", %f8, %f4, %f" #acc ";\n"                  \
    "    fma.rn.f32 %f" #acc ", %f9, %f5, %f" #acc ";\n"                  \
    "    fma.rn.f32 %f" #acc ", %f10, %f6, %f" #acc ";\n"
    DQ_CHUNK(0, 12) DQ_CHUNK(16, 13) DQ_CHUNK(32, 14) DQ_CHUNK(48, 15)
    DQ_CHUNK(64, 12) DQ_CHUNK(80, 13) DQ_CHUNK(96, 14) DQ_CHUNK(112, 15)
    "    add.s64 %rd5, %rd5, 16;\n"
    "    add.s64 %rd6, %rd6, 1;\n"
    "    add.u32 %r17, %r17, 128;\n"
    "    add.s32 %r5, %r5, 1;\n"
    "    setp.lt.s32 %p1, %r5, 8;\n"
    "    @%p1 bra $L_dq_j;\n"
    "    add.s64 %rd4, %rd4, 288;\n"
    "    add.s32 %r7, %r7, -1;\n"
    "    setp.ne.s32 %p2, %r7, 0;\n"
    "    @%p2 bra $L_dq_pair;\n"
    "    add.f32 %f12, %f12, %f13;\n"
    "    add.f32 %f14, %f14, %f15;\n"
    "    add.f32 %f12, %f12, %f14;\n"
    "    mov.u32 %r4, %ctaid.x;\n"
    "    shl.b32 %r4, %r4, 9;\n"
    "    add.u32 %r4, %r4, %r1;\n"               /* global tid */
    "    cvt.u64.u32 %rd10, %r4;\n"
    "    shl.b64 %rd10, %rd10, 2;\n"
    "    add.s64 %rd10, %rd3, %rd10;\n"
    "    st.global.f32 [%rd10], %f12;\n"
    "    ret;\n"
    "}\n";

static int cuda_dequant(void *cuda, int ordinal, double *flops, char *err,
                        size_t errsz) {
    *flops = 0.0;
    typedef int (*cumem_t)(unsigned long long *, size_t);
    typedef int (*cufree_t)(unsigned long long);
    typedef int (*cumod_t)(void *, const void *);
    typedef int (*cufunc_t)(void *, void *, const char *);
    typedef int (*cuattr_t)(int *, int, int);
    typedef int (*culaunch_t)(void *, unsigned, unsigned, unsigned, unsigned,
                              unsigned, unsigned, unsigned, void *, void **,
                              void **);
    typedef int (*cusync_t)(void);
    typedef int (*cuhtd_t)(unsigned long long, const void *, size_t);
    typedef int (*curc_t)(int, const char **);

    cumem_t cuMemAlloc = sym(cuda, "cuMemAlloc");
    cufree_t cuMemFree = sym(cuda, "cuMemFree");
    cumod_t cuModuleLoadData = sym(cuda, "cuModuleLoadData");
    cufunc_t cuModuleGetFunction = sym(cuda, "cuModuleGetFunction");
    cuattr_t cuDeviceGetAttribute = sym(cuda, "cuDeviceGetAttribute");
    culaunch_t cuLaunchKernel = sym(cuda, "cuLaunchKernel");
    cusync_t cuCtxSynchronize = sym(cuda, "cuCtxSynchronize");
    cuhtd_t cuHtoD = sym(cuda, "cuMemcpyHtoD");
    curc_t cuGetError = sym(cuda, "cuGetErrorString");

    if (!cuMemAlloc || !cuMemFree || !cuModuleLoadData ||
        !cuModuleGetFunction || !cuDeviceGetAttribute || !cuLaunchKernel ||
        !cuCtxSynchronize || !cuHtoD) {
        snprintf(err, errsz, "CUDA driver API incomplete");
        return -1;
    }

    void *module = NULL;
    int r = cuModuleLoadData(&module, dequant_ptx);
    if (r) {
        const char *es = NULL;
        if (cuGetError && !cuGetError(r, &es) && es)
            snprintf(err, errsz, "ptx jit: %d (%s)", r, es);
        else
            snprintf(err, errsz, "ptx jit: %d", r);
        return -1;
    }
    void *kernel = NULL;
    if (cuModuleGetFunction(&kernel, module, "geode_dequant_kernel")) {
        snprintf(err, errsz, "cuModuleGetFunction failed");
        return -1;
    }
    int sm_count = 0;
    if (cuDeviceGetAttribute(&sm_count, 16 /* multiprocessor count */,
                             ordinal) ||
        sm_count <= 0) {
        snprintf(err, errsz, "cuDeviceGetAttribute failed");
        return -1;
    }
    int blocks = sm_count; /* one block per sm: ~36KB shared each */
    long long threads = (long long)blocks * DQ_THREADS;
    long long warps = threads / 32;
    size_t weights_bytes = (size_t)warps * DQ_PAIRS_PER_WARP * 288;
    size_t x_bytes = 32 * 256 * 4;

    unsigned long long d_weights = 0, d_x = 0, d_out = 0;
    if (cuMemAlloc(&d_weights, weights_bytes) ||
        cuMemAlloc(&d_x, x_bytes) ||
        cuMemAlloc(&d_out, threads * 4)) {
        snprintf(err, errsz, "cuMemAlloc failed");
        return -1;
    }
    uint8_t *h_weights = malloc(weights_bytes);
    float *h_x = malloc(x_bytes);
    if (!h_weights || !h_x) {
        snprintf(err, errsz, "out of memory");
        free(h_weights);
        free(h_x);
        return -1;
    }
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < weights_bytes; i += 8) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        *(uint64_t *)(h_weights + i) = rng;
    }
    for (int i = 0; i < 32 * 256; i++) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        h_x[i] = (float)(rng & 0xFF) / 128.0f - 1.0f;
    }
    cuHtoD(d_weights, h_weights, weights_bytes);
    cuHtoD(d_x, h_x, x_bytes);
    free(h_weights);
    free(h_x);

    void *params[] = {&d_weights, &d_x, &d_out};
    cuLaunchKernel(kernel, blocks, 1, 1, DQ_THREADS, 1, 1, 0, NULL,
                   params, NULL);
    cuCtxSynchronize(); /* warmup: absorb jit + first-launch cost */

    double best = 0;
    double t_end = now_s() + DQ_BENCH_SECONDS;
    while (now_s() < t_end) {
        double t0 = now_s();
        r = cuLaunchKernel(kernel, blocks, 1, 1, DQ_THREADS, 1, 1, 0,
                           NULL, params, NULL);
        if (r || cuCtxSynchronize()) break;
        double dt = now_s() - t0;
        double rate = (double)warps * DQ_PAIRS_PER_WARP * 2 * 256.0 * 32 *
                      2 / dt;
        if (rate > best) best = rate;
    }
    cuMemFree(d_weights);
    cuMemFree(d_x);
    cuMemFree(d_out);
    if (best <= 0) {
        snprintf(err, errsz, "kernel launch failed");
        return -1;
    }
    *flops = best;
    return 0;
}

/* Tiled SGEMM: what a real executor's prefill actually runs on a gpu
   without tensor cores (dequant amortizes to noise at prefill context
   lengths). 64x64 block tiles, 4x4 register micro-tile, 8-deep k-tiles,
   A stored row-major in shared (stride 9), B padded to stride 68 for
   16B-aligned float4 access. */
#define SG_DIM 1024
#define SG_KTILES (SG_DIM / 8)
#define SG_BENCH_SECONDS 0.5

#define SG_STEP(a0, a1, a2, a3, bo)                                       \
    "    ld.shared.f32 %f0, [%r16+" #a0 "];\n"                            \
    "    ld.shared.f32 %f1, [%r16+" #a1 "];\n"                            \
    "    ld.shared.f32 %f2, [%r16+" #a2 "];\n"                            \
    "    ld.shared.f32 %f3, [%r16+" #a3 "];\n"                            \
    "    ld.shared.v4.f32 {%f4,%f5,%f6,%f7}, [%r17+" #bo "];\n"           \
    "    fma.rn.f32 %f16, %f0, %f4, %f16;\n"                              \
    "    fma.rn.f32 %f17, %f0, %f5, %f17;\n"                              \
    "    fma.rn.f32 %f18, %f0, %f6, %f18;\n"                              \
    "    fma.rn.f32 %f19, %f0, %f7, %f19;\n"                              \
    "    fma.rn.f32 %f20, %f1, %f4, %f20;\n"                              \
    "    fma.rn.f32 %f21, %f1, %f5, %f21;\n"                              \
    "    fma.rn.f32 %f22, %f1, %f6, %f22;\n"                              \
    "    fma.rn.f32 %f23, %f1, %f7, %f23;\n"                              \
    "    fma.rn.f32 %f24, %f2, %f4, %f24;\n"                              \
    "    fma.rn.f32 %f25, %f2, %f5, %f25;\n"                              \
    "    fma.rn.f32 %f26, %f2, %f6, %f26;\n"                              \
    "    fma.rn.f32 %f27, %f2, %f7, %f27;\n"                              \
    "    fma.rn.f32 %f28, %f3, %f4, %f28;\n"                              \
    "    fma.rn.f32 %f29, %f3, %f5, %f29;\n"                              \
    "    fma.rn.f32 %f30, %f3, %f6, %f30;\n"                              \
    "    fma.rn.f32 %f31, %f3, %f7, %f31;\n"

static const char sgemm_ptx[] =
    ".version 6.0\n"
    ".target sm_50\n"
    ".address_size 64\n"
    ".shared .align 16 .b8 geode_s[4480];\n" /* As 64x9 f32, then Bs 8x68 */
    ".visible .entry geode_sgemm_kernel(\n"
    "    .param .u64 p_a, .param .u64 p_b, .param .u64 p_c,\n"
    "    .param .u32 p_kt)\n"
    "{\n"
    "    .reg .pred %p<4>;\n"
    "    .reg .f32 %f<32>;\n"
    "    .reg .u32 %r<24>;\n"
    "    .reg .u64 %rd<12>;\n"
    "    ld.param.u64 %rd1, [p_a];\n"
    "    ld.param.u64 %rd2, [p_b];\n"
    "    ld.param.u64 %rd3, [p_c];\n"
    "    ld.param.u32 %r20, [p_kt];\n"
    "    mov.u32 %r1, %tid.x;\n"
    "    mov.u32 %r2, %ctaid.x;\n"
    "    mov.u32 %r3, %ctaid.y;\n"
    "    shr.u32 %r4, %r1, 4;\n"              /* ty */
    "    and.b32 %r5, %r1, 15;\n"             /* tx */
    "    mov.u32 %r13, geode_s;\n"
    "    mov.f32 %f16, 0f00000000;\n"
    "    mov.f32 %f17, 0f00000000;\n"
    "    mov.f32 %f18, 0f00000000;\n"
    "    mov.f32 %f19, 0f00000000;\n"
    "    mov.f32 %f20, 0f00000000;\n"
    "    mov.f32 %f21, 0f00000000;\n"
    "    mov.f32 %f22, 0f00000000;\n"
    "    mov.f32 %f23, 0f00000000;\n"
    "    mov.f32 %f24, 0f00000000;\n"
    "    mov.f32 %f25, 0f00000000;\n"
    "    mov.f32 %f26, 0f00000000;\n"
    "    mov.f32 %f27, 0f00000000;\n"
    "    mov.f32 %f28, 0f00000000;\n"
    "    mov.f32 %f29, 0f00000000;\n"
    "    mov.f32 %f30, 0f00000000;\n"
    "    mov.f32 %f31, 0f00000000;\n"
    "    setp.lt.s32 %p1, %r1, 128;\n"
    /* A loader (tid<128): row = tid/2, half = tid%2 */
    "    shr.u32 %r6, %r1, 1;\n"
    "    and.b32 %r7, %r1, 1;\n"
    "    shl.b32 %r8, %r3, 6;\n"
    "    add.u32 %r8, %r8, %r6;\n"
    "    shl.b32 %r8, %r8, 12;\n"             /* (by*64+row) x 4096 */
    "    shl.b32 %r9, %r7, 4;\n"              /* half x 16 */
    "    add.u32 %r8, %r8, %r9;\n"
    "    cvt.u64.u32 %rd4, %r8;\n"
    "    add.s64 %rd4, %rd4, %rd1;\n"
    "    mul.lo.u32 %r14, %r6, 36;\n"         /* shared As: row x 36B */
    "    add.u32 %r14, %r14, %r9;\n"
    "    add.u32 %r14, %r14, %r13;\n"
    /* B loader (tid>=128): t2 = tid-128, row = t2/16, colv = t2%16 */
    "    add.s32 %r10, %r1, -128;\n"
    "    shr.u32 %r11, %r10, 4;\n"
    "    and.b32 %r12, %r10, 15;\n"
    "    shl.b32 %r8, %r2, 8;\n"              /* bx x 256 */
    "    shl.b32 %r9, %r12, 4;\n"             /* colv x 16 */
    "    add.u32 %r8, %r8, %r9;\n"
    "    shl.b32 %r9, %r11, 12;\n"            /* row x 4096 */
    "    add.u32 %r8, %r8, %r9;\n"
    "    cvt.u64.u32 %rd5, %r8;\n"
    "    add.s64 %rd5, %rd5, %rd2;\n"
    "    mul.lo.u32 %r15, %r11, 272;\n"       /* shared Bs: 2304 + row x 272 */
    "    shl.b32 %r18, %r12, 4;\n"
    "    add.u32 %r15, %r15, %r18;\n"
    "    add.u32 %r15, %r15, %r13;\n"
    "    add.u32 %r15, %r15, 2304;\n"
    /* compute bases */
    "    mul.lo.u32 %r16, %r4, 144;\n"        /* As + ty x 144 */
    "    add.u32 %r16, %r16, %r13;\n"
    "    shl.b32 %r17, %r5, 4;\n"             /* Bs + tx x 16 */
    "    add.u32 %r17, %r17, %r13;\n"
    "    add.u32 %r17, %r17, 2304;\n"
    "    mov.u32 %r19, %r20;\n"
    "$L_sg_tile:\n"
    "    @%p1 bra $L_sg_load_a;\n"
    "    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd5];\n"
    "    st.shared.v4.f32 [%r15], {%f0,%f1,%f2,%f3};\n"
    "    add.s64 %rd5, %rd5, 32768;\n"
    "    bra.uni $L_sg_loaded;\n"
    "$L_sg_load_a:\n"
    "    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd4];\n"
    "    st.shared.f32 [%r14], %f0;\n"
    "    st.shared.f32 [%r14+4], %f1;\n"
    "    st.shared.f32 [%r14+8], %f2;\n"
    "    st.shared.f32 [%r14+12], %f3;\n"
    "    add.s64 %rd4, %rd4, 32;\n"
    "$L_sg_loaded:\n"
    "    bar.sync 0;\n"
    SG_STEP(0, 36, 72, 108, 0)
    SG_STEP(4, 40, 76, 112, 272)
    SG_STEP(8, 44, 80, 116, 544)
    SG_STEP(12, 48, 84, 120, 816)
    SG_STEP(16, 52, 88, 124, 1088)
    SG_STEP(20, 56, 92, 128, 1360)
    SG_STEP(24, 60, 96, 132, 1632)
    SG_STEP(28, 64, 100, 136, 1904)
    "    bar.sync 0;\n"
    "    add.s32 %r19, %r19, -1;\n"
    "    setp.ne.s32 %p2, %r19, 0;\n"
    "    @%p2 bra $L_sg_tile;\n"
    /* store C tile: rows by*64+ty*4+i, cols bx*64+tx*4 */
    "    shl.b32 %r6, %r3, 6;\n"
    "    shl.b32 %r7, %r4, 2;\n"
    "    add.u32 %r6, %r6, %r7;\n"
    "    shl.b32 %r6, %r6, 12;\n"
    "    shl.b32 %r7, %r2, 8;\n"
    "    add.u32 %r6, %r6, %r7;\n"
    "    shl.b32 %r7, %r5, 4;\n"
    "    add.u32 %r6, %r6, %r7;\n"
    "    cvt.u64.u32 %rd6, %r6;\n"
    "    add.s64 %rd6, %rd6, %rd3;\n"
    "    st.global.v4.f32 [%rd6], {%f16,%f17,%f18,%f19};\n"
    "    st.global.v4.f32 [%rd6+4096], {%f20,%f21,%f22,%f23};\n"
    "    st.global.v4.f32 [%rd6+8192], {%f24,%f25,%f26,%f27};\n"
    "    st.global.v4.f32 [%rd6+12288], {%f28,%f29,%f30,%f31};\n"
    "    ret;\n"
    "}\n";

static int cuda_sgemm(void *cuda, double *flops, char *err, size_t errsz) {
    *flops = 0.0;
    typedef int (*cumem_t)(unsigned long long *, size_t);
    typedef int (*cufree_t)(unsigned long long);
    typedef int (*cumod_t)(void *, const void *);
    typedef int (*cufunc_t)(void *, void *, const char *);
    typedef int (*culaunch_t)(void *, unsigned, unsigned, unsigned, unsigned,
                              unsigned, unsigned, unsigned, void *, void **,
                              void **);
    typedef int (*cusync_t)(void);
    typedef int (*cuhtd_t)(unsigned long long, const void *, size_t);
    typedef int (*cudth_t)(void *, unsigned long long, size_t);
    typedef int (*curc_t)(int, const char **);

    cumem_t cuMemAlloc = sym(cuda, "cuMemAlloc");
    cufree_t cuMemFree = sym(cuda, "cuMemFree");
    cumod_t cuModuleLoadData = sym(cuda, "cuModuleLoadData");
    cufunc_t cuModuleGetFunction = sym(cuda, "cuModuleGetFunction");
    culaunch_t cuLaunchKernel = sym(cuda, "cuLaunchKernel");
    cusync_t cuCtxSynchronize = sym(cuda, "cuCtxSynchronize");
    cuhtd_t cuHtoD = sym(cuda, "cuMemcpyHtoD");
    cudth_t cuDtoH = sym(cuda, "cuMemcpyDtoH");
    curc_t cuGetError = sym(cuda, "cuGetErrorString");

    if (!cuMemAlloc || !cuMemFree || !cuModuleLoadData ||
        !cuModuleGetFunction || !cuLaunchKernel || !cuCtxSynchronize ||
        !cuHtoD || !cuDtoH) {
        snprintf(err, errsz, "CUDA driver API incomplete");
        return -1;
    }

    void *module = NULL;
    int r = cuModuleLoadData(&module, sgemm_ptx);
    if (r) {
        const char *es = NULL;
        if (cuGetError && !cuGetError(r, &es) && es)
            snprintf(err, errsz, "ptx jit: %d (%s)", r, es);
        else
            snprintf(err, errsz, "ptx jit: %d", r);
        return -1;
    }
    void *kernel = NULL;
    if (cuModuleGetFunction(&kernel, module, "geode_sgemm_kernel")) {
        snprintf(err, errsz, "cuModuleGetFunction failed");
        return -1;
    }

    size_t mat_bytes = (size_t)SG_DIM * SG_DIM * 4;
    unsigned long long d_a = 0, d_b = 0, d_c = 0;
    if (cuMemAlloc(&d_a, mat_bytes) || cuMemAlloc(&d_b, mat_bytes) ||
        cuMemAlloc(&d_c, mat_bytes)) {
        snprintf(err, errsz, "cuMemAlloc failed");
        return -1;
    }
    float *h_a = malloc(mat_bytes);
    float *h_b = malloc(mat_bytes);
    if (!h_a || !h_b) {
        snprintf(err, errsz, "out of memory");
        free(h_a);
        free(h_b);
        return -1;
    }
    uint64_t rng = 0x243F6A8885A308D3ull;
    for (int i = 0; i < SG_DIM * SG_DIM; i++) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        h_a[i] = (float)(rng & 0xFF) / 512.0f;
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        h_b[i] = (float)(rng & 0xFF) / 512.0f;
    }
    cuHtoD(d_a, h_a, mat_bytes);
    cuHtoD(d_b, h_b, mat_bytes);

    int ktiles = SG_KTILES;
    void *params[] = {&d_a, &d_b, &d_c, &ktiles};
    r = cuLaunchKernel(kernel, 16, 16, 1, 256, 1, 1, 0, NULL, params, NULL);
    if (r || cuCtxSynchronize()) {
        snprintf(err, errsz, "kernel launch failed");
        free(h_a);
        free(h_b);
        return -1;
    }

    /* Self-check: C[0][0] must equal dot(A row 0, B col 0). */
    double expect = 0;
    for (int k = 0; k < SG_DIM; k++)
        expect += (double)h_a[k] * h_b[k * SG_DIM];
    float got = 0;
    cuDtoH(&got, d_c, 4);
    double tol = fabs(expect) * 1e-3 + 1e-6;
    free(h_a);
    free(h_b);
    if (fabs(got - expect) > tol) {
        snprintf(err, errsz, "sgemm self-check failed (got %g, want %g)",
                 got, expect);
        return -1;
    }

    double best = 0;
    double t_end = now_s() + SG_BENCH_SECONDS;
    while (now_s() < t_end) {
        double t0 = now_s();
        r = cuLaunchKernel(kernel, 16, 16, 1, 256, 1, 1, 0, NULL, params,
                           NULL);
        if (r || cuCtxSynchronize()) break;
        double dt = now_s() - t0;
        double rate = 2.0 * SG_DIM * SG_DIM * SG_DIM / dt;
        if (rate > best) best = rate;
    }
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);
    if (best <= 0) {
        snprintf(err, errsz, "kernel launch failed");
        return -1;
    }
    *flops = best;
    return 0;
}

static int cuda_flops(void *cuda, int ordinal, double *flops, char *err,
                      size_t errsz) {
    *flops = 0.0;
    typedef int (*cumem_t)(unsigned long long *, size_t);
    typedef int (*cufree_t)(unsigned long long);
    typedef int (*cumod_t)(void *, const void *);
    typedef int (*cufunc_t)(void *, void *, const char *);
    typedef int (*cuattr_t)(int *, int, int);
    typedef int (*culaunch_t)(void *, unsigned, unsigned, unsigned, unsigned,
                              unsigned, unsigned, unsigned, void *, void **,
                              void **);
    typedef int (*cusync_t)(void);
    typedef int (*curc_t)(int, const char **);

    cumem_t cuMemAlloc = sym(cuda, "cuMemAlloc");
    cufree_t cuMemFree = sym(cuda, "cuMemFree");
    cumod_t cuModuleLoadData = sym(cuda, "cuModuleLoadData");
    cufunc_t cuModuleGetFunction = sym(cuda, "cuModuleGetFunction");
    cuattr_t cuDeviceGetAttribute = sym(cuda, "cuDeviceGetAttribute");
    culaunch_t cuLaunchKernel = sym(cuda, "cuLaunchKernel");
    cusync_t cuCtxSynchronize = sym(cuda, "cuCtxSynchronize");
    curc_t cuGetError = sym(cuda, "cuGetErrorString");

    if (!cuMemAlloc || !cuMemFree || !cuModuleLoadData ||
        !cuModuleGetFunction || !cuDeviceGetAttribute || !cuLaunchKernel ||
        !cuCtxSynchronize) {
        snprintf(err, errsz, "CUDA driver API incomplete");
        return -1;
    }

    void *module = NULL;
    int r = cuModuleLoadData(&module, flops_ptx);
    if (r) {
        const char *es = NULL;
        if (cuGetError && !cuGetError(r, &es) && es)
            snprintf(err, errsz, "ptx jit: %d (%s)", r, es);
        else
            snprintf(err, errsz, "ptx jit: %d", r);
        return -1;
    }
    void *kernel = NULL;
    r = cuModuleGetFunction(&kernel, module, "geode_flops_kernel");
    if (r) {
        snprintf(err, errsz, "cuModuleGetFunction: %d", r);
        return -1;
    }
    int sm_count = 0;
    if (cuDeviceGetAttribute(&sm_count, 16 /* multiprocessor count */,
                             ordinal) ||
        sm_count <= 0) {
        snprintf(err, errsz, "cuDeviceGetAttribute failed");
        return -1;
    }
    int blocks = sm_count * FLOPS_BLOCKS_PER_SM;
    long long total_threads = (long long)blocks * FLOPS_THREADS;

    unsigned long long out = 0;
    if (cuMemAlloc(&out, total_threads * 4)) {
        snprintf(err, errsz, "cuMemAlloc failed");
        return -1;
    }
    void *params[] = {&out};

    cuLaunchKernel(kernel, blocks, 1, 1, FLOPS_THREADS, 1, 1, 0, NULL,
                   params, NULL);
    cuCtxSynchronize(); /* warmup: absorb jit + first-launch cost */

    double best = 0;
    double t_end = now_s() + FLOPS_BENCH_SECONDS;
    while (now_s() < t_end) {
        double t0 = now_s();
        r = cuLaunchKernel(kernel, blocks, 1, 1, FLOPS_THREADS, 1, 1, 0,
                           NULL, params, NULL);
        if (r || cuCtxSynchronize()) break;
        double dt = now_s() - t0;
        double rate = (double)total_threads * FLOPS_KERNEL_ITERS *
                      FLOPS_ACCUMULATORS * 2 / dt;
        if (rate > best) best = rate;
    }
    cuMemFree(out);
    if (best <= 0) {
        snprintf(err, errsz, "kernel launch failed");
        return -1;
    }
    *flops = best;
    return 0;
}

int gpu_probe(Gpu **out) {
    void *cuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    void *nvml = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!cuda && !nvml) {
        *out = NULL;
        return 0;
    }

    typedef int (*ccnt_t)(int *);
    typedef int (*ncnt_t)(unsigned *);
    typedef int (*cuinit_t)(unsigned);
    int count = 0;
    cuinit_t cuInit = sym(cuda, "cuInit");
    if (cuInit) cuInit(0);
    ccnt_t cuDevCount = sym(cuda, "cuDeviceGetCount");
    if (cuDevCount) {
        cuDevCount(&count);
    } else {
        ncnt_t nvmlCount = sym(nvml, "nvmlDeviceGetCount_v2");
        if (nvmlCount) {
            unsigned c = 0;
            if (nvmlCount(&c) == 0) count = (int)c;
        }
    }
    if (count <= 0 || count > CUDA_GPU_MAX) {
        *out = NULL;
        return 0;
    }

    Gpu *gpus = calloc((size_t)count, sizeof *gpus);

    typedef int (*cuget_t)(int *, int);
    typedef int (*cuname_t)(char *, int, int);
    typedef int (*cucc_t)(int *, int *, int);
    typedef int (*cutot_t)(unsigned long long *, int);
    typedef int (*nhnd_t)(unsigned, void *);
    typedef int (*nname_t)(void *, char *, unsigned);
    typedef int (*nmem_t)(void *, NvmlMemory *);
    typedef int (*npci_t)(void *, NvmlPciInfo *);
    typedef int (*ninit_t)();
    typedef int (*nshut_t)();

    cuget_t cuDevGet = sym(cuda, "cuDeviceGet");
    cuname_t cuDevName = sym(cuda, "cuDeviceGetName");
    cucc_t cuDevCC = sym(cuda, "cuDeviceComputeCapability");
    cutot_t cuDevTotalMem = sym(cuda, "cuDeviceTotalMem");

    ninit_t nvmlInit = sym(nvml, "nvmlInit_v2");
    nshut_t nvmlShutdown = sym(nvml, "nvmlShutdown");
    nhnd_t nvmlHandle = sym(nvml, "nvmlDeviceGetHandleByIndex_v2");
    nname_t nvmlName = sym(nvml, "nvmlDeviceGetName");
    nmem_t nvmlMem = sym(nvml, "nvmlDeviceGetMemoryInfo");
    npci_t nvmlPci = sym(nvml, "nvmlDeviceGetPciInfo_v3");

    void *nvml_handle = NULL;
    int nvml_ready = 0;
    if (nvml && nvmlInit && nvmlInit() == 0 && nvmlHandle) nvml_ready = 1;

    for (int i = 0; i < count; i++) {
        Gpu *g = &gpus[i];
        if (nvml_ready) {
            void *h = NULL;
            if (nvmlHandle((unsigned)i, &h) == 0) {
                nvml_handle = h;
                char name[64] = "";
                if (nvmlName && nvmlName(h, name, sizeof name) == 0) {
                    NvmlMemory mem;
                    NvmlPciInfo pci;
                    if (nvmlMem && nvmlMem(h, &mem) == 0)
                        g->vram_bytes = mem.total;
                    if (nvmlPci && nvmlPci(h, &pci) == 0)
                        pcie_link_of(pci.busId, &g->pcie_gen, &g->pcie_width);
                    snprintf(g->name, sizeof g->name, "%s", name);
                }
            }
        }
        if (!g->name[0] && cuda && cuDevGet && cuDevName) {
            int dev;
            if (cuDevGet(&dev, i) == 0)
                cuDevName(g->name, sizeof g->name, dev);
        }
        if (cuda && cuDevGet && cuDevTotalMem && g->vram_bytes == 0) {
            int dev;
            unsigned long long total = 0;
            if (cuDevGet(&dev, i) == 0 && cuDevTotalMem(&total, dev) == 0)
                g->vram_bytes = total;
        }
        if (cuda && cuDevGet && cuDevCC) {
            int dev, maj = 0, min = 0;
            if (cuDevGet(&dev, i) == 0 && cuDevCC(&maj, &min, dev) == 0) {
                g->cc_major = maj;
                g->cc_minor = min;
            }
        }

        char err[128] = "";
        void *ctx = NULL;
        if (cuda_context(cuda, i, &ctx, err, sizeof err) == 0) {
            g->bw_measured =
                cuda_bandwidth(cuda, &g->hbm_bw_bytes_s,
                               &g->pcie_bw_bytes_s, err, sizeof err) == 0;
            if (!g->bw_measured)
                snprintf(g->bw_error, sizeof g->bw_error, "%s", err);
            g->flops_measured =
                cuda_flops(cuda, i, &g->fp32_flops, err, sizeof err) == 0;
            if (!g->flops_measured)
                snprintf(g->flops_error, sizeof g->flops_error, "%s", err);
            g->dequant_measured =
                cuda_dequant(cuda, i, &g->q4k_dequant_flops, err,
                             sizeof err) == 0;
            if (!g->dequant_measured)
                snprintf(g->dequant_error, sizeof g->dequant_error, "%s",
                         err);
            g->sgemm_measured =
                cuda_sgemm(cuda, &g->sgemm_flops, err, sizeof err) == 0;
            if (!g->sgemm_measured)
                snprintf(g->sgemm_error, sizeof g->sgemm_error, "%s", err);
            typedef int (*cud_t)(void *);
            cud_t cuCtxDestroy = sym(cuda, "cuCtxDestroy");
            if (cuCtxDestroy) cuCtxDestroy(ctx);
        } else {
            snprintf(g->bw_error, sizeof g->bw_error, "%s", err);
            snprintf(g->flops_error, sizeof g->flops_error, "%s", err);
        }
    }
    if (nvml_handle && nvmlShutdown) nvmlShutdown();
    *out = gpus;
    return count;
}

void gpu_free(Gpu *gpus) { free(gpus); }