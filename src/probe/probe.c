#include "dram.h"
#include "flops.h"
#include "gpu.h"
#include "json.h"
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

static void print_summary(const DramNode *dram, int ndram,
                          const NvmeDrive *nvme, int nnvme, const Gpu *gpu,
                          int ngpu, double flops, double dequant_flops,
                          int flops_threads) {
    printf("probed hardware:\n");
    char model[256];
    cpu_model(model, sizeof model);
    if (flops > 0 && dequant_flops > 0)
        printf("  cpu   %-40s  %6.1f GFLOPS fp32, %6.1f q4k dequant (%.0f%%, "
               "%d threads)\n",
               model, flops / 1e9, dequant_flops / 1e9,
               100.0 * dequant_flops / flops, flops_threads);
    else if (flops > 0)
        printf("  cpu   %-40s  %6.1f GFLOPS fp32 (%d threads)\n", model,
               flops / 1e9, flops_threads);
    else
        printf("  cpu   %-40s  flops unmeasured\n", model);
    for (int i = 0; i < ndram; i++) {
        char size[32];
        fmt_size((double)dram[i].capacity_bytes, size, sizeof size);
        printf("  dram  node%d  %12s  %8.1f GB/s read  %7.1f GB/s copy  %4.0f ns\n",
               dram[i].node, size, dram[i].read_bw_bytes_s / 1e9,
               dram[i].copy_bw_bytes_s / 1e9, dram[i].load_latency_ns);
    }
    for (int i = 0; i < nnvme; i++) {
        char size[32];
        fmt_size((double)nvme[i].capacity_bytes, size, sizeof size);
        if (nvme[i].measured)
            printf("  nvme  %-8s  %12s  %8.1f GB/s seq read\n", nvme[i].name,
                   size, nvme[i].read_bw_bytes_s / 1e9);
        else
            printf("  nvme  %-8s  %12s  unmeasured (raw read denied; no "
                   "writable fs path)\n",
                   nvme[i].name, size);
    }
    for (int i = 0; i < ngpu; i++) {
        char size[32];
        fmt_size((double)gpu[i].vram_bytes, size, sizeof size);
        if (gpu[i].bw_measured)
            printf("  gpu   %-24s  %12s  %6.1f GB/s hbm  pcie-gen%d x%d  %5.1f "
                   "GB/s\n",
                   gpu[i].name, size, gpu[i].hbm_bw_bytes_s / 1e9,
                   gpu[i].pcie_gen, gpu[i].pcie_width,
                   gpu[i].pcie_bw_bytes_s / 1e9);
        else
            printf("  gpu   %-24s  %12s  bandwidth unmeasured (%s)\n",
                   gpu[i].name, size,
                   gpu[i].bw_error[0] ? gpu[i].bw_error : "no CUDA driver");
        if (gpu[i].flops_measured)
            printf("  gpu   %-24s  %12s  %6.1f GFLOPS fp32\n", "", size,
                   gpu[i].fp32_flops / 1e9);
        else if (gpu[i].bw_measured)
            printf("  gpu   %-24s  %12s  flops unmeasured (%s)\n", "", size,
                   gpu[i].flops_error[0] ? gpu[i].flops_error : "?");
    }
}

static void write_json(FILE *f, const DramNode *dram, int ndram,
                       const NvmeDrive *nvme, int nnvme, const Gpu *gpu,
                       int ngpu, double flops, double dequant_flops,
                       int flops_threads) {
    char model[256] = "";
    cpu_model(model, sizeof model);

    Json j;
    json_begin(&j, f);
    json_u64(&j, "schema", 1);
    json_open(&j, "cpu", 0);
    json_string(&j, "model", model);
    json_double(&j, "fp32_flops", flops);
    json_double(&j, "q4k_dequant_flops", dequant_flops);
    json_u64(&j, "flops_threads", (unsigned long long)flops_threads);
    json_close(&j);

    json_open(&j, "dram", 1);
    for (int i = 0; i < ndram; i++) {
        json_open(&j, NULL, 0);
        json_u64(&j, "node", dram[i].node);
        json_u64(&j, "capacity_bytes", dram[i].capacity_bytes);
        json_double(&j, "read_bw_bytes_s", dram[i].read_bw_bytes_s);
        json_double(&j, "copy_bw_bytes_s", dram[i].copy_bw_bytes_s);
        json_double(&j, "load_latency_ns", dram[i].load_latency_ns);
        json_close(&j);
    }
    json_close(&j);

    json_open(&j, "nvme", 1);
    for (int i = 0; i < nnvme; i++) {
        json_open(&j, NULL, 0);
        json_string(&j, "dev", nvme[i].name);
        json_u64(&j, "capacity_bytes", nvme[i].capacity_bytes);
        json_double(&j, "read_bw_bytes_s", nvme[i].read_bw_bytes_s);
        json_close(&j);
    }
    json_close(&j);

    json_open(&j, "gpu", 1);
    for (int i = 0; i < ngpu; i++) {
        char cap[64];
        snprintf(cap, sizeof cap, "%d.%d", gpu[i].cc_major, gpu[i].cc_minor);
        json_open(&j, NULL, 0);
        json_string(&j, "name", gpu[i].name);
        json_u64(&j, "vram_bytes", gpu[i].vram_bytes);
        json_string(&j, "compute_capability", cap);
        json_open(&j, "pcie_link", 0);
        json_u64(&j, "gen", gpu[i].pcie_gen);
        json_u64(&j, "width", gpu[i].pcie_width);
        json_double(&j, "bw_bytes_s", gpu[i].pcie_bw_bytes_s);
        json_close(&j);
        json_double(&j, "hbm_bw_bytes_s", gpu[i].hbm_bw_bytes_s);
        json_u64(&j, "bw_measured", gpu[i].bw_measured);
        json_double(&j, "fp32_flops", gpu[i].fp32_flops);
        json_u64(&j, "flops_measured", gpu[i].flops_measured);
        json_close(&j);
    }
    json_close(&j);
    json_end(&j);
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

int main(int argc, char **argv) {
    const char *out_path = default_path();
    if (argc > 2) {
        usage(argv[0]);
        return 2;
    }
    if (argc == 2) out_path = argv[1];

    DramNode *dram = NULL;
    NvmeDrive *nvme = NULL;
    Gpu *gpu = NULL;
    int flops_threads = 0;
    int ndram = dram_probe(&dram);
    int nnvme = nvme_probe(&nvme);
    int ngpu = gpu_probe(&gpu);
    double dequant_flops = 0;
    double flops = flops_probe(&flops_threads, &dequant_flops);

    print_summary(dram, ndram, nvme, nnvme, gpu, ngpu, flops, dequant_flops,
                  flops_threads);

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
    write_json(f, dram, ndram, nvme, nnvme, gpu, ngpu, flops, dequant_flops,
               flops_threads);
    if (fclose(f)) {
        perror(out_path);
        return 1;
    }
    printf("wrote %s\n", out_path);

    dram_free(dram);
    nvme_free(nvme);
    gpu_free(gpu);
    return 0;
}