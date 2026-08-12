#include "gpu.h"

#include <dlfcn.h>
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

static int cuda_bandwidth(void *cuda, int ordinal, double *hbm,
                          double *pcie_bw, char *err, size_t errsz) {
    *hbm = 0.0;
    *pcie_bw = 0.0;
    typedef int (*cuinit_t)(unsigned);
    typedef int (*cuget_t)(int *, int);
    typedef int (*cuprim_t)(void **, int);
    typedef int (*cucur_t)(void *);
    typedef int (*cumem_t)(unsigned long long *, size_t);
    typedef int (*cumemhost_t)(void **, size_t);
    typedef int (*cucopy_t)(unsigned long long, unsigned long long, size_t);
    typedef int (*cucopyhh_t)(unsigned long long, void *, size_t);
    typedef int (*cucopyh_t)(void *, unsigned long long, size_t);
    typedef int (*curc_t)(int, const char **);

    cuinit_t cuInit = sym(cuda, "cuInit");
    cuget_t cuDeviceGet = sym(cuda, "cuDeviceGet");
    cuprim_t cuPrimRet = sym(cuda, "cuDevicePrimaryCtxRetain");
    cucur_t cuSetCur = sym(cuda, "cuCtxSetCurrent");
    cumem_t cuMemAlloc = sym(cuda, "cuMemAlloc");
    cumemhost_t cuMemAllocHost = sym(cuda, "cuMemAllocHost");
    cucopy_t cuDtoD = sym(cuda, "cuMemcpyDtoD");
    cucopyhh_t cuHtoD = sym(cuda, "cuMemcpyHtoD");
    cucopyh_t cuDtoH = sym(cuda, "cuMemcpyDtoH");
    curc_t cuGetError = sym(cuda, "cuGetErrorString");

    if (!cuInit || !cuDeviceGet || !cuPrimRet || !cuSetCur || !cuMemAlloc ||
        !cuMemAllocHost || !cuDtoD || !cuHtoD || !cuDtoH) {
        snprintf(err, errsz, "CUDA driver API incomplete");
        return -1;
    }
    if (cuInit(0)) {
        snprintf(err, errsz, "cuInit failed");
        return -1;
    }
    int dev;
    if (cuDeviceGet(&dev, ordinal)) {
        snprintf(err, errsz, "cuDeviceGet failed");
        return -1;
    }
    void *ctx = NULL;
    int r = cuPrimRet(&ctx, dev);
    if (r || !ctx) {
        snprintf(err, errsz, "context: %d", r);
        return -1;
    }
    if ((r = cuSetCur(ctx))) {
        snprintf(err, errsz, "cuCtxSetCurrent: %d", r);
        return -1;
    }

    unsigned long long d1 = 0, d2 = 0;
    r = cuMemAlloc(&d1, COPY_BYTES);
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
    double t0 = now_s();
    for (int i = 0; i < 4; i++) cuDtoD(d2, d1, COPY_BYTES);
    double dt = now_s() - t0;
    if (dt > 0) *hbm = 4.0 * COPY_BYTES / dt;

    t0 = now_s();
    for (int i = 0; i < 4; i++) {
        cuHtoD(d1, host, COPY_BYTES);
        cuDtoH(host, d2, COPY_BYTES);
    }
    dt = now_s() - t0;
    if (dt > 0) *pcie_bw = 8.0 * COPY_BYTES / dt;
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
        g->bw_measured =
            cuda_bandwidth(cuda, i, &g->hbm_bw_bytes_s,
                           &g->pcie_bw_bytes_s, err, sizeof err) == 0;
        if (!g->bw_measured) snprintf(g->bw_error, sizeof g->bw_error, "%s",
                                      err);
    }
    if (nvml_handle && nvmlShutdown) nvmlShutdown();
    *out = gpus;
    return count;
}

void gpu_free(Gpu *gpus) { free(gpus); }