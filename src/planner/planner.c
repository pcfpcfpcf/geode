#include "home.h"
#include "json.h"
#include "manifest.h"
#include "modules.h"
#include "plan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tunable surface of the planner. */
#define DRAM_USABLE_FRACTION 0.80  /* headroom for OS + runtime */
#define VRAM_USABLE_FRACTION 0.90
#define INTERLEAVE_BW_PENALTY 0.50 /* interleaved NUMA sends ~half of reads
                                      across the socket link */
#define NVME_MISS_LATENCY_S 100e-6
#define PREFETCH_HIDDEN_FRACTION 0.90 /* fraction of miss latency the
                                         one-layer-ahead prefetch hides */
#define H_ASSUMED 0.50           /* conservative pool hit-rate assumption */
#define PESSIMISM_LO 0.70        /* predictions are pessimistic by policy:
                                    band low end = 0.70 x point estimate */
#define HYBRID_SYNC_FACTOR 0.90  /* PCIe transfers + pipeline bubbles */
#define HYBRID_GEMV_EFFICIENCY 0.50 /* attention decode reads are
                                       latency-chained gemv rows plus fixed
                                       per-layer kernel overhead, not a
                                       streaming sweep: measured 34.8 GB/s
                                       probe bandwidth delivers ~12-18 GB/s
                                       effective on an M1200, and the cpu
                                       side's byte model carries its own
                                       slack, so this single factor absorbs
                                       the end-to-end gap (measured: 13.5
                                       tok/s, model now lands ~15) */
#define MTP_ACCEPTANCE 2.0       /* only used when manifest has a draft head */
#define GPU_DEQUANT_EFFICIENCY 0.50 /* assumed until a gpu dequant-shaped
                                       bench exists; dequant GEMM never hits
                                       the fp32 fma peak */

static const double h_sweep[] = {0.0, 0.50, 0.85};
static const int n_h_sweep = sizeof h_sweep / sizeof h_sweep[0];

typedef struct {
    int n_nodes;
    double eff_bw;              /* bytes/s after NUMA policy */
    unsigned long long usable;  /* bytes the plan may occupy */
    const char *policy;
    int replicas;
} DramPlan;

typedef struct {
    double read_bw;
    unsigned long long capacity;
} NvmeTier;

typedef struct {
    int present;
    int bw_measured;
    int flops_measured;
    int dequant_measured;
    int sgemm_measured;
    char name[128];
    unsigned long long vram;
    double hbm_bw;
    double fp32_flops;
    double dequant_flops;
    double sgemm_flops;
} GpuTier;

typedef struct {
    DramPlan dram;
    NvmeTier nvme;
    GpuTier gpu;
    double cpu_flops;
    const char *cpu_flops_kind;
} HardwareView;

typedef struct {
    long context, batch;
    unsigned long long kv_total, weights_total, weights_kv;
    unsigned long long kv_per_token, bytes_per_step;
    unsigned long long hybrid_dram_step, hybrid_gpu_step;
    double flops_bound, gpu_flops_bound;
    double cpu_prefill, gpu_prefill;
    int gpu_estimated;
} Workload;

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

static double jnum(const JVal *obj, const char *key, int *ok) {
    const JVal *v = json_get(obj, key);
    if (!v || v->kind != JV_NUM) {
        *ok = 0;
        return 0;
    }
    return v->num;
}

static void jstr(const JVal *obj, const char *key, char *out, size_t outsz) {
    const JVal *v = json_get(obj, key);
    out[0] = '\0';
    if (v && v->kind == JV_STR)
        snprintf(out, outsz, "%s", v->str);
}

static int load_probe(const char *path, HardwareView *hw,
                      unsigned long long weights_kv_bytes,
                      char *err, size_t errsz) {
    char *text = json_read_file(path, err, errsz);
    if (!text) return 0;
    JVal *root = json_parse(text, err, errsz);
    free(text);
    if (!root) return 0;

    hw->cpu_flops = 0;
    hw->cpu_flops_kind = "unmeasured";
    const JVal *jcpu = json_get(root, "cpu");
    if (jcpu) {
        int ok = 1;
        /* dequant-shaped measurement is the honest bound for quantized
           inference; plain fp32 fma is the fallback */
        hw->cpu_flops = jnum(jcpu, "q4k_dequant_flops", &ok);
        if (ok && hw->cpu_flops > 0) {
            hw->cpu_flops_kind = "q4k dequant";
        } else {
            hw->cpu_flops = jnum(jcpu, "fp32_flops", &ok);
            if (ok && hw->cpu_flops > 0) hw->cpu_flops_kind = "fp32 fma";
            else hw->cpu_flops = 0; /* old probe: fields absent */
        }
    }

    DramPlan *dram = &hw->dram;
    const JVal *jdram = json_get(root, "dram");
    if (!jdram || jdram->kind != JV_ARR || jdram->n_items == 0) {
        snprintf(err, errsz, "%.200s: no dram nodes", path);
        json_free(root);
        return 0;
    }
    dram->n_nodes = jdram->n_items;
    double sum_bw = 0;
    unsigned long long sum_cap = 0, min_cap = ~0ULL;
    for (int i = 0; i < jdram->n_items; i++) {
        int ok = 1;
        sum_bw += jnum(jdram->items[i], "read_bw_bytes_s", &ok);
        unsigned long long cap =
            (unsigned long long)jnum(jdram->items[i], "capacity_bytes", &ok);
        if (!ok) {
            snprintf(err, errsz, "%.200s: bad dram entry", path);
            json_free(root);
            return 0;
        }
        sum_cap += cap;
        if (cap < min_cap) min_cap = cap;
    }
    /* Replicate when every node can hold a full copy; otherwise interleave
       and pay the socket-link penalty on ~half of weight reads. */
    if (dram->n_nodes >= 2 &&
        weights_kv_bytes <= (unsigned long long)(min_cap * DRAM_USABLE_FRACTION)) {
        dram->policy = "replicate";
        dram->replicas = dram->n_nodes;
        dram->eff_bw = sum_bw;
        dram->usable = (unsigned long long)(min_cap * DRAM_USABLE_FRACTION);
    } else if (dram->n_nodes >= 2) {
        dram->policy = "interleave";
        dram->replicas = 1;
        dram->eff_bw = sum_bw * INTERLEAVE_BW_PENALTY;
        dram->usable = (unsigned long long)(sum_cap * DRAM_USABLE_FRACTION);
    } else {
        dram->policy = "single";
        dram->replicas = 1;
        dram->eff_bw = sum_bw;
        dram->usable = (unsigned long long)(sum_cap * DRAM_USABLE_FRACTION);
    }

    NvmeTier *nvme = &hw->nvme;
    nvme->read_bw = 0;
    nvme->capacity = 0;
    const JVal *jnvme = json_get(root, "nvme");
    if (jnvme && jnvme->kind == JV_ARR)
        for (int i = 0; i < jnvme->n_items; i++) {
            int ok = 1;
            nvme->read_bw += jnum(jnvme->items[i], "read_bw_bytes_s", &ok);
            nvme->capacity +=
                (unsigned long long)jnum(jnvme->items[i], "capacity_bytes", &ok);
        }

    GpuTier *gpu = &hw->gpu;
    gpu->present = 0;
    const JVal *jgpu = json_get(root, "gpu");
    if (jgpu && jgpu->kind == JV_ARR && jgpu->n_items > 0) {
        int ok = 1;
        const JVal *g = jgpu->items[0];
        gpu->present = 1;
        gpu->vram = (unsigned long long)jnum(g, "vram_bytes", &ok);
        gpu->hbm_bw = jnum(g, "hbm_bw_bytes_s", &ok);
        gpu->bw_measured = (int)jnum(g, "bw_measured", &ok);
        gpu->fp32_flops = jnum(g, "fp32_flops", &ok);
        gpu->flops_measured = (int)jnum(g, "flops_measured", &ok);
        gpu->dequant_flops = jnum(g, "q4k_dequant_flops", &ok);
        gpu->dequant_measured = (int)jnum(g, "dequant_measured", &ok);
        gpu->sgemm_flops = jnum(g, "sgemm_flops", &ok);
        gpu->sgemm_measured = (int)jnum(g, "sgemm_measured", &ok);
        jstr(g, "name", gpu->name, sizeof gpu->name);
    }

    json_free(root);
    return 1;
}

static double mtp_multiplier(const Manifest *m) {
    return m->mtp_head ? MTP_ACCEPTANCE : 1.0;
}

static void score_resident(Candidate *c, const Manifest *m,
                           unsigned long long total_bytes,
                           unsigned long long bytes_per_token,
                           const GpuTier *gpu, double gpu_flops_bound) {
    snprintf(c->strategy, sizeof c->strategy, "RESIDENT");
    c->scorable = 0;
    if (!gpu->present) {
        snprintf(c->reason, sizeof c->reason, "no gpu");
        return;
    }
    if (!gpu->bw_measured) {
        snprintf(c->reason, sizeof c->reason, "gpu bandwidth unmeasured");
        return;
    }
    char need[32], have[32];
    fmt_size((double)total_bytes, need, sizeof need);
    fmt_size(gpu->vram * VRAM_USABLE_FRACTION, have, sizeof have);
    if (total_bytes > gpu->vram * VRAM_USABLE_FRACTION) {
        snprintf(c->reason, sizeof c->reason, "weights+kv %s > vram %s", need,
                 have);
        return;
    }
    c->scorable = 1;
    c->decode_tok_s[1] = gpu->hbm_bw / bytes_per_token;
    if (gpu_flops_bound < c->decode_tok_s[1]) c->decode_tok_s[1] = gpu_flops_bound;
    c->decode_tok_s[1] *= mtp_multiplier(m);
}

static void score_cpu_stream(Candidate *c, const Manifest *m,
                             unsigned long long total_bytes,
                             unsigned long long bytes_per_token,
                             const DramPlan *dram, double flops_bound) {
    snprintf(c->strategy, sizeof c->strategy, "CPU-STREAM");
    c->scorable = 0;
    if (total_bytes > dram->usable) {
        char need[32], have[32];
        fmt_size((double)total_bytes, need, sizeof need);
        fmt_size((double)dram->usable, have, sizeof have);
        snprintf(c->reason, sizeof c->reason, "weights+kv %s > dram %s", need,
                 have);
        return;
    }
    c->scorable = 1;
    c->decode_tok_s[1] = dram->eff_bw / bytes_per_token;
    if (flops_bound < c->decode_tok_s[1]) c->decode_tok_s[1] = flops_bound;
    c->decode_tok_s[1] *= mtp_multiplier(m);
}

static void score_hybrid(Candidate *c, const Manifest *m,
                         unsigned long long kv_total,
                         unsigned long long gpu_fixed_step,
                         unsigned long long dram_step,
                         const DramPlan *dram, const GpuTier *gpu,
                         double flops_bound) {
    snprintf(c->strategy, sizeof c->strategy, "HYBRID");
    c->scorable = 0;
    if (!gpu->present) {
        snprintf(c->reason, sizeof c->reason, "no gpu");
        return;
    }
    if (!gpu->bw_measured) {
        snprintf(c->reason, sizeof c->reason, "gpu bandwidth unmeasured");
        return;
    }
    unsigned long long on_gpu =
        m->attention_bytes + m->base_bytes + kv_total;
    if (on_gpu > gpu->vram * VRAM_USABLE_FRACTION) {
        snprintf(c->reason, sizeof c->reason,
                 "attention+base+kv do not fit in vram");
        return;
    }
    /* Decode's gpu side holds the deterministic reads: attention+kv every
    layer, serially, plus the base ffn (dense block's or shared expert) --
    whose gemms are launched before the cpu's routed matmuls run, so the
    two tiers read in parallel and the step pays whichever of the two
    finishes last. Routed experts stream from DRAM exactly as in
    CPU-STREAM. */
    c->scorable = 1;
    double eff_bw = gpu->hbm_bw * HYBRID_GEMV_EFFICIENCY;
    double step = (double)gpu_fixed_step / eff_bw;
    double routed_time = (double)dram_step / dram->eff_bw;
    double shared_time = (double)m->base_bytes / eff_bw;
    step += routed_time > shared_time ? routed_time : shared_time;
    c->decode_tok_s[1] = 1.0 / step;
    if (flops_bound < c->decode_tok_s[1]) c->decode_tok_s[1] = flops_bound;
    c->decode_tok_s[1] *= HYBRID_SYNC_FACTOR * mtp_multiplier(m);
}

/* The design-doc formula with h folded into bytes: equivalent to
   min(bw, flops, latency) x 1/(1-h) at the disk bound, but stays safe when
   another tier binds. */
static double flash_rate_at_h(const Manifest *m, const DramPlan *dram,
                              const NvmeTier *nvme,
                              unsigned long long att_base,
                              unsigned long long kv_per_token, long batch,
                              double h, double flops_bound) {
    double dram_bytes = att_base + batch * (kv_per_token + m->routed_bytes * h);
    double dram_rate = dram->eff_bw / dram_bytes;
    double disk_rate =
        h >= 1.0 ? INFINITY
                 : nvme->read_bw / (batch * m->routed_bytes * (1.0 - h));
    double misses = batch * m->n_layer * m->expert_used * (1.0 - h);
    double serialized = misses * (1.0 - PREFETCH_HIDDEN_FRACTION);
    double latency_rate =
        serialized <= 0.0 ? INFINITY : 1.0 / (NVME_MISS_LATENCY_S * serialized);
    double rate = dram_rate;
    if (disk_rate < rate) rate = disk_rate;
    if (latency_rate < rate) rate = latency_rate;
    if (flops_bound < rate) rate = flops_bound;
    return rate * mtp_multiplier(m);
}

static void score_flash_stream(Candidate *c, const Manifest *m,
                               unsigned long long kv_total,
                               unsigned long long weights_kv,
                               unsigned long long kv_per_token, long batch,
                               const DramPlan *dram, const NvmeTier *nvme,
                               double flops_bound) {
    snprintf(c->strategy, sizeof c->strategy, "FLASH-STREAM");
    c->scorable = 0;
    /* Design doc: FLASH-STREAM is the "else" branch — streaming from disk
       what could be held in dram is never better. */
    if (weights_kv <= dram->usable) {
        snprintf(c->reason, sizeof c->reason, "model fits in dram");
        return;
    }
    unsigned long long att_base = m->attention_bytes + m->base_bytes;
    if (att_base + kv_total > dram->usable) {
        snprintf(c->reason, sizeof c->reason,
                 "attention+base+kv do not fit in dram");
        return;
    }
    if (nvme->read_bw <= 0) {
        snprintf(c->reason, sizeof c->reason, "nvme bandwidth unmeasured");
        return;
    }
    c->scorable = 1;
    c->decode_tok_s[1] = flash_rate_at_h(m, dram, nvme, att_base, kv_per_token,
                                         batch, H_ASSUMED, flops_bound);
}

static void print_report(const Manifest *m, const HardwareView *hw,
                         const Workload *wl, const Plan *plan) {
    char size[32];
    if (hw->cpu_flops > 0)
        printf("probed:  cpu %.0f GFLOPS %s, ", hw->cpu_flops / 1e9,
               hw->cpu_flops_kind);
    else
        printf("probed:  cpu flops unmeasured, ");
    printf("dram %.0f GB/s eff (%s), usable ", hw->dram.eff_bw / 1e9,
           hw->dram.policy);
    fmt_size((double)hw->dram.usable, size, sizeof size);
    printf("%s, nvme %.1f GB/s", size, hw->nvme.read_bw / 1e9);
    if (hw->gpu.present)
        printf(", gpu %s %.1f GB%s", hw->gpu.name, hw->gpu.vram / 1e9,
               hw->gpu.bw_measured ? "" : " (unmeasured)");
    printf("\n");
    printf("model:   %s (%s), %.2fB total / %.2fB active, %llu layers, "
           "%llu experts (%llu/token)\n",
           m->model, m->variant, m->total_params / 1e9, m->active_params / 1e9,
           m->n_layer, m->expert_count, m->expert_used);
    printf("workload: context %ld, batch %ld; kv cache ", wl->context,
           wl->batch);
    fmt_size((double)wl->kv_total, size, sizeof size);
    printf("%s\n\n", size);

    printf("%-13s %-9s %-14s %s\n", "strategy", "scorable", "decode tok/s",
           "why");
    const Candidate *best =
        plan->strategy[0] ? plan_candidate(plan, plan->strategy) : NULL;
    for (int i = 0; i < plan->n_candidates; i++) {
        const Candidate *c = &plan->candidates[i];
        if (c->scorable)
            printf("%-13s %-9s %.0f-%-9.0f %s\n", c->strategy, "yes",
                   c->decode_tok_s[0], c->decode_tok_s[1],
                   c == best ? "<- best" : "");
        else
            printf("%-13s %-9s %-14s %s\n", c->strategy, "no", "-", c->reason);
    }
    printf("\n");

    if (!best) {
        printf("no scorable strategy on this hardware\n");
        return;
    }

    if (strcmp(best->strategy, "FLASH-STREAM") == 0) {
        unsigned long long att_base = m->attention_bytes + m->base_bytes;
        printf("hit-rate dial (routed %.2f GB/token, nvme %.1f GB/s):\n",
               m->routed_bytes / 1e9, hw->nvme.read_bw / 1e9);
        for (int i = 0; i < n_h_sweep; i++)
            printf("  h=%-4.0f%% -> %.1f tok/s\n", h_sweep[i] * 100,
                   flash_rate_at_h(m, &hw->dram, &hw->nvme, att_base,
                                   wl->kv_per_token, wl->batch, h_sweep[i],
                                   wl->flops_bound));
        if (strstr(m->variant, "decomposed") == NULL)
            printf("note: streaming full experts; decomposition cuts routed "
                   "bytes ~7x\n");
        printf("\n");
    }

    printf("plan:    %s, numa %s", plan->strategy, plan->numa_policy);
    if (plan->numa_replicas > 1) printf(" (%d replicas)", plan->numa_replicas);
    printf("%s\n", m->mtp_head ? ", mtp on" : "");
    printf("predict: %.0f-%.0f tok/s decode (calibrating)",
           plan->predicted_tok_s[0], plan->predicted_tok_s[1]);
    if (best->prefill_tok_s > 0) {
        const char *device = "cpu";
        if (strcmp(best->strategy, "RESIDENT") == 0) device = hw->gpu.name;
        else if (strcmp(best->strategy, "HYBRID") == 0) device = "gpu+cpu";
        int estimated = wl->gpu_estimated &&
                       strcmp(best->strategy, "CPU-STREAM") != 0 &&
                       strcmp(best->strategy, "FLASH-STREAM") != 0;
        printf(", prefill ~%.0f tok/s on %s%s, TTFT ~%.0fs @ %ld ctx",
               best->prefill_tok_s, device,
               estimated ? " (estimated from fp32 peak)" : "",
               wl->context / best->prefill_tok_s, wl->context);
    } else {
        printf(", prefill unmeasured (probe has no flops measurement)");
    }
    printf("\n");
}

static const char *default_probe_path(void) {
    static char buf[1024];
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(buf, sizeof buf, "%s/.geode/probe.json", home);
    return buf;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s MODEL.gguf [--context N] [--batch N] [--out plan.json]\n",
            argv0);
}

static void add_placement(Plan *plan, const char *component, const char *tier,
                          const char *policy) {
    Placement *p = &plan->placements[plan->n_placements++];
    snprintf(p->component, sizeof p->component, "%s", component);
    snprintf(p->tier, sizeof p->tier, "%s", tier);
    snprintf(p->policy, sizeof p->policy, "%s", policy);
}

int planner_main(int argc, char **argv) {
    const char *probe_path = default_probe_path();
    const char *manifest_path = NULL;
    const char *out_path = NULL;
    long context = 4096, batch = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            context = strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch = strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (argv[i][0] != '-') {
            if (manifest_path) {
                usage(argv[0]);
                return 2;
            }
            /* geode_model_home shares one buffer; copy before the next call
               overwrites it. */
            static char manifest_buf[1024], out_buf[1024];
            snprintf(manifest_buf, sizeof manifest_buf, "%s",
                     geode_model_home(argv[i], "manifest"));
            snprintf(out_buf, sizeof out_buf, "%s",
                     geode_model_home(argv[i], "plan"));
            manifest_path = manifest_buf;
            if (!out_path) out_path = out_buf;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!manifest_path) {
        usage(argv[0]);
        return 2;
    }
    if (context <= 0 || batch <= 0) {
        fprintf(stderr, "context and batch must be positive\n");
        return 2;
    }

    char err[256];
    Manifest m;
    if (!manifest_load(manifest_path, &m, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    Workload wl = {0};
    wl.context = context;
    wl.batch = batch;
    wl.kv_total = m.kv_bytes_per_ctx_token * (unsigned long long)context *
                 (unsigned long long)batch;
    wl.weights_total =
        m.attention_bytes + m.base_bytes + m.expert_bytes * m.expert_count;
    wl.weights_kv = wl.weights_total + wl.kv_total;

    HardwareView hw;
    if (!load_probe(probe_path, &hw, wl.weights_kv, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    /* Experts are dequantized and multiplied on the cpu in every strategy
       except RESIDENT, so the cpu flops bound applies to all of them. */
    wl.flops_bound = hw.cpu_flops > 0
                         ? hw.cpu_flops / (2.0 * m.active_params * batch)
                         : INFINITY;
    /* Decode-side kernels are fused dequant GEMV/GEMM; prefill on a gpu
       without tensor cores runs dequant + SGEMM, with dequant amortized to
       noise at prefill context lengths. Each phase gets the measurement
       with the matching shape; fp32 peak scaled by an assumed efficiency
       is the fallback for old probes. */
    double gpu_decode_flops = 0;
    double gpu_prefill_flops = 0;
    if (hw.gpu.dequant_measured && hw.gpu.dequant_flops > 0)
        gpu_decode_flops = hw.gpu.dequant_flops;
    if (hw.gpu.sgemm_measured && hw.gpu.sgemm_flops > 0)
        gpu_prefill_flops = hw.gpu.sgemm_flops;
    if (hw.gpu.flops_measured && (!gpu_decode_flops || !gpu_prefill_flops)) {
        double fallback = hw.gpu.fp32_flops * GPU_DEQUANT_EFFICIENCY;
        if (!gpu_decode_flops) gpu_decode_flops = fallback;
        if (!gpu_prefill_flops) gpu_prefill_flops = fallback;
        wl.gpu_estimated = 1;
    }
    wl.gpu_flops_bound = gpu_decode_flops > 0
                             ? gpu_decode_flops / (2.0 * m.active_params * batch)
                             : INFINITY;

    /* Prefill is GEMM-shaped, computed wherever each component lives. */
    wl.cpu_prefill = hw.cpu_flops > 0 ? hw.cpu_flops / (2.0 * m.active_params) : 0;
    wl.gpu_prefill =
        gpu_prefill_flops > 0 ? gpu_prefill_flops / (2.0 * m.active_params) : 0;

    /* Per decode step: attention+base are read once and amortize over the
       batch, but each token routes independently, so expert reads grow with
       the expected union of hit experts; kv reads scale with batch. Rates
       below are per sequence (latency-oriented). */
    double experts_hit =
        m.expert_count *
        (1.0 - pow(1.0 - (double)m.expert_used / m.expert_count, batch));
    wl.kv_per_token = m.kv_bytes_per_ctx_token * (unsigned long long)context;
    unsigned long long routed_per_step =
        (unsigned long long)(experts_hit * m.expert_bytes);
    wl.bytes_per_step = m.attention_bytes + m.base_bytes + routed_per_step +
                        wl.kv_per_token * (unsigned long long)batch;
    wl.hybrid_dram_step = routed_per_step;
    wl.hybrid_gpu_step = m.attention_bytes + wl.kv_per_token * (unsigned long long)batch;

    Plan plan = {0};
    plan.n_ctx = (int)context;
    plan.batch = (int)batch;
    plan.n_candidates = 4;
    Candidate *cands = plan.candidates;
    score_resident(&cands[0], &m, wl.weights_kv, wl.bytes_per_step, &hw.gpu,
                   wl.gpu_flops_bound);
    score_cpu_stream(&cands[1], &m, wl.weights_kv, wl.bytes_per_step, &hw.dram,
                     wl.flops_bound);
score_hybrid(&cands[2], &m, wl.kv_total, wl.hybrid_gpu_step,
                 wl.hybrid_dram_step, &hw.dram, &hw.gpu, wl.flops_bound);
    score_flash_stream(&cands[3], &m, wl.kv_total, wl.weights_kv,
                       wl.kv_per_token, batch, &hw.dram, &hw.nvme,
                       wl.flops_bound);

    /* Params split approximated by byte split; the Q4_K/Q6_K mix varies
       little across components. HYBRID prefill serializes: attention+base
       on the gpu, routed experts on the cpu. */
    double active_bytes = m.attention_bytes + m.base_bytes + m.routed_bytes;
    double gpu_frac = (m.attention_bytes + m.base_bytes) / active_bytes;
    cands[0].prefill_tok_s = wl.gpu_prefill;
    cands[1].prefill_tok_s = wl.cpu_prefill;
    cands[2].prefill_tok_s =
        (wl.gpu_prefill > 0 && wl.cpu_prefill > 0)
            ? 1.0 / (gpu_frac / wl.gpu_prefill + (1.0 - gpu_frac) / wl.cpu_prefill)
            : 0;
    cands[3].prefill_tok_s = wl.cpu_prefill;

    int best_idx = -1;
    for (int i = 0; i < plan.n_candidates; i++)
        if (cands[i].scorable &&
            (best_idx < 0 || cands[i].decode_tok_s[1] > cands[best_idx].decode_tok_s[1]))
            best_idx = i;
    for (int i = 0; i < plan.n_candidates; i++)
        if (cands[i].scorable)
            cands[i].decode_tok_s[0] = cands[i].decode_tok_s[1] * PESSIMISM_LO;

    if (best_idx >= 0) {
        Candidate *best = &cands[best_idx];
        snprintf(plan.strategy, sizeof plan.strategy, "%s", best->strategy);
        plan.predicted_tok_s[0] = best->decode_tok_s[0];
        plan.predicted_tok_s[1] = best->decode_tok_s[1];
        plan.predicted_prefill_tok_s = best->prefill_tok_s;
        snprintf(plan.numa_policy, sizeof plan.numa_policy, "%s", hw.dram.policy);
        plan.numa_replicas = hw.dram.replicas;

        if (best_idx == 0) {
            add_placement(&plan, "all", "vram", "resident");
        } else if (best_idx == 3) {
            add_placement(&plan, "attention", "dram", "resident");
            add_placement(&plan, "base", "dram", "resident");
            add_placement(&plan, "experts", "nvme", "pool+prefetch");
        } else {
            int hybrid = best_idx == 2;
            add_placement(&plan, "attention", hybrid ? "vram" : "dram", "resident");
            add_placement(&plan, "base", hybrid ? "vram" : "dram", "resident");
            add_placement(&plan, "experts", "dram", "resident");
            add_placement(&plan, "kv", hybrid ? "vram" : "dram", "resident");
        }
    }

    print_report(&m, &hw, &wl, &plan);

    if (best_idx < 0) return 1;

    FILE *out = fopen(out_path, "w");
    if (!out) {
        perror(out_path);
        return 1;
    }
    plan_write(out, &plan);
    if (fclose(out)) {
        perror(out_path);
        return 1;
    }
    printf("wrote %s\n", out_path);
    return 0;
}
