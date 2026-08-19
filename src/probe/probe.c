#include "dram.h"
#include "flops.h"
#include "gpu.h"
#include "json.h"
#include "modules.h"
#include "nvme.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void fmt_size(double bytes, char *out, size_t outsz) {
    static const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0;
    double v = bytes;
    while (v >= 1e3 && i < 4) {
        v /= 1e3;
        i++;
    }
    snprintf(out, outsz, "%.1f %s", v, units[i]);
}

typedef struct {
    DramNode *dram;
    int ndram;
    NvmeDrive *nvme;
    int nnvme;
    Gpu *gpu;
    int ngpu;
    char cpu_model[256];
    double flops, dequant_flops;
    int flops_threads;
} ProbeInfo;

static void cpu_model(char *out, size_t outsz) {
    out[0] = '\0';
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *p = strchr(line, ':');
            if (p) {
                p++;
                while (*p == ' ') p++;
                size_t n = strcspn(p, "\n");
                if (n >= outsz) n = outsz - 1;
                memcpy(out, p, n);
                out[n] = '\0';
            }
            break;
        }
    }
    fclose(f);
}

static void print_summary(const ProbeInfo *p) {
    printf("probed hardware:\n");
    if (p->flops > 0 && p->dequant_flops > 0)
        printf("  cpu   %-40s  %6.1f GFLOPS fp32, %6.1f q4k dequant (%.0f%%, "
               "%d threads)\n",
               p->cpu_model, p->flops / 1e9, p->dequant_flops / 1e9,
               100.0 * p->dequant_flops / p->flops, p->flops_threads);
    else if (p->flops > 0)
        printf("  cpu   %-40s  %6.1f GFLOPS fp32 (%d threads)\n",
               p->cpu_model, p->flops / 1e9, p->flops_threads);
    else
        printf("  cpu   %-40s  flops unmeasured\n", p->cpu_model);
    for (int i = 0; i < p->ndram; i++) {
        const DramNode *d = &p->dram[i];
        char size[32];
        fmt_size((double)d->capacity_bytes, size, sizeof size);
        printf("  dram  node%d  %12s  %8.1f GB/s read  %7.1f GB/s copy  %4.0f ns\n",
               d->node, size, d->read_bw_bytes_s / 1e9,
               d->copy_bw_bytes_s / 1e9, d->load_latency_ns);
    }
    for (int i = 0; i < p->nnvme; i++) {
        const NvmeDrive *n = &p->nvme[i];
        char size[32];
        fmt_size((double)n->capacity_bytes, size, sizeof size);
        if (n->measured)
            printf("  nvme  %-8s  %12s  %8.1f GB/s seq read\n", n->name,
                   size, n->read_bw_bytes_s / 1e9);
        else
            printf("  nvme  %-8s  %12s  unmeasured (raw read denied; no "
                   "writable fs path)\n",
                   n->name, size);
    }
    for (int i = 0; i < p->ngpu; i++) {
        const Gpu *g = &p->gpu[i];
        char size[32];
        fmt_size((double)g->vram_bytes, size, sizeof size);
        if (g->bw_measured)
            printf("  gpu   %-24s  %12s  %6.1f GB/s hbm  pcie-gen%d x%d  %5.1f "
                   "GB/s\n",
                   g->name, size, g->hbm_bw_bytes_s / 1e9, g->pcie_gen,
                   g->pcie_width, g->pcie_bw_bytes_s / 1e9);
        else
            printf("  gpu   %-24s  %12s  bandwidth unmeasured (%s)\n",
                   g->name, size, g->bw_error[0] ? g->bw_error : "no CUDA driver");
        if (g->flops_measured && g->dequant_measured && g->sgemm_measured)
            printf("  gpu   %-24s  %12s  %6.1f GFLOPS fp32, %6.1f q4k "
                   "dequant, %6.1f sgemm (%.0f%%)\n",
                   "", size, g->fp32_flops / 1e9, g->q4k_dequant_flops / 1e9,
                   g->sgemm_flops / 1e9,
                   100.0 * g->sgemm_flops / g->fp32_flops);
        else if (g->flops_measured)
            printf("  gpu   %-24s  %12s  %6.1f GFLOPS fp32\n", "", size,
                   g->fp32_flops / 1e9);
        else if (g->bw_measured)
            printf("  gpu   %-24s  %12s  flops unmeasured (%s)\n", "", size,
                   g->flops_error[0] ? g->flops_error : "?");
    }
}

static void write_json(FILE *f, const ProbeInfo *p) {
    Json j;
    json_begin(&j, f);
    json_u64(&j, "schema", 1);
    json_open(&j, "cpu", 0);
    json_string(&j, "model", p->cpu_model);
    json_double(&j, "fp32_flops", p->flops);
    json_double(&j, "q4k_dequant_flops", p->dequant_flops);
    json_u64(&j, "flops_threads", (unsigned long long)p->flops_threads);
    json_close(&j);

    json_open(&j, "dram", 1);
    for (int i = 0; i < p->ndram; i++) {
        const DramNode *d = &p->dram[i];
        json_open(&j, NULL, 0);
        json_u64(&j, "node", d->node);
        json_u64(&j, "capacity_bytes", d->capacity_bytes);
        json_double(&j, "read_bw_bytes_s", d->read_bw_bytes_s);
        json_double(&j, "copy_bw_bytes_s", d->copy_bw_bytes_s);
        json_double(&j, "load_latency_ns", d->load_latency_ns);
        json_close(&j);
    }
    json_close(&j);

    json_open(&j, "nvme", 1);
    for (int i = 0; i < p->nnvme; i++) {
        const NvmeDrive *n = &p->nvme[i];
        json_open(&j, NULL, 0);
        json_string(&j, "dev", n->name);
        json_u64(&j, "capacity_bytes", n->capacity_bytes);
        json_double(&j, "read_bw_bytes_s", n->read_bw_bytes_s);
        json_close(&j);
    }
    json_close(&j);

    json_open(&j, "gpu", 1);
    for (int i = 0; i < p->ngpu; i++) {
        const Gpu *g = &p->gpu[i];
        char cap[64];
        snprintf(cap, sizeof cap, "%d.%d", g->cc_major, g->cc_minor);
        json_open(&j, NULL, 0);
        json_string(&j, "name", g->name);
        json_u64(&j, "vram_bytes", g->vram_bytes);
        json_string(&j, "compute_capability", cap);
        json_open(&j, "pcie_link", 0);
        json_u64(&j, "gen", g->pcie_gen);
        json_u64(&j, "width", g->pcie_width);
        json_double(&j, "bw_bytes_s", g->pcie_bw_bytes_s);
        json_close(&j);
        json_double(&j, "hbm_bw_bytes_s", g->hbm_bw_bytes_s);
        json_u64(&j, "bw_measured", g->bw_measured);
        json_double(&j, "fp32_flops", g->fp32_flops);
        json_u64(&j, "flops_measured", g->flops_measured);
        json_double(&j, "q4k_dequant_flops", g->q4k_dequant_flops);
        json_u64(&j, "dequant_measured", g->dequant_measured);
        json_double(&j, "sgemm_flops", g->sgemm_flops);
        json_u64(&j, "sgemm_measured", g->sgemm_measured);
        json_close(&j);
    }
    json_close(&j);
    json_end(&j);
}

static void probe_free(ProbeInfo *p) {
    dram_free(p->dram);
    nvme_free(p->nvme);
    gpu_free(p->gpu);
}

static const char *default_path(void) {
    static char buf[1024];
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(buf, sizeof buf, "%s/.geode/probe.json", home);
    return buf;
}

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s [OUT.json]\n", argv0);
}

int probe_main(int argc, char **argv) {
    const char *out_path = default_path();
    if (argc > 2) {
        usage(argv[0]);
        return 2;
    }
    if (argc == 2) out_path = argv[1];

    ProbeInfo p = {0};
    p.ndram = dram_probe(&p.dram);
    p.nnvme = nvme_probe(&p.nvme);
    p.ngpu = gpu_probe(&p.gpu);
    p.flops = flops_probe(&p.flops_threads, &p.dequant_flops);
    cpu_model(p.cpu_model, sizeof p.cpu_model);

    print_summary(&p);

    char dir[1024];
    snprintf(dir, sizeof dir, "%s", out_path);
    char *slash = strrchr(dir, '/');
    if (slash && slash != dir) {
        *slash = '\0';
        mkdir(dir, 0700);
    }

    FILE *f = fopen(out_path, "w");
    if (!f) {
        perror(out_path);
        return 1;
    }
    write_json(f, &p);
    if (fclose(f)) {
        perror(out_path);
        return 1;
    }
    printf("wrote %s\n", out_path);

    probe_free(&p);
    return 0;
}