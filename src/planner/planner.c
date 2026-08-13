#include "json.h"

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
#define POOL_MAX_BYTES (12ULL << 30)
#define H_ASSUMED 0.50           /* conservative pool hit-rate assumption */
#define PESSIMISM_LO 0.70        /* predictions are pessimistic by policy:
                                    band low end = 0.70 x point estimate */
#define HYBRID_SYNC_FACTOR 0.90  /* PCIe transfers + pipeline bubbles */
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
    char name[128];
    unsigned long long vram;
    double hbm_bw;
    double fp32_flops;
} GpuTier;

typedef struct {
    char model[256];
    char variant[64];
    unsigned long long total_params, active_params, n_layer;
    unsigned long long attention_bytes, base_bytes;
    unsigned long long expert_count, expert_used, expert_bytes, routed_bytes;
    unsigned long long kv_bytes_per_ctx_token;
    int mtp_head;
} Manifest;

typedef struct {
    const char *name;
    int scorable;
    char reason[160];
    double point;               /* decode tok/s point estimate */
} Candidate;

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

static int load_probe(const char *path, DramPlan *dram, NvmeTier *nvme,
                      GpuTier *gpu, double *cpu_flops,
                      const char **cpu_flops_kind,
                      unsigned long long weights_kv_bytes,
                      char *err, size_t errsz) {
    char *text = json_read_file(path, err, errsz);
    if (!text) return 0;
    JVal *root = json_parse(text, err, errsz);
    free(text);
    if (!root) return 0;

    *cpu_flops = 0;
    *cpu_flops_kind = "unmeasured";
    const JVal *jcpu = json_get(root, "cpu");
    if (jcpu) {
        int ok = 1;
        /* dequant-shaped measurement is the honest bound for quantized
           inference; plain fp32 fma is the fallback */
        *cpu_flops = jnum(jcpu, "q4k_dequant_flops", &ok);
        if (ok && *cpu_flops > 0) {
            *cpu_flops_kind = "q4k dequant";
        } else {
            *cpu_flops = jnum(jcpu, "fp32_flops", &ok);
            if (ok && *cpu_flops > 0) *cpu_flops_kind = "fp32 fma";
            else *cpu_flops = 0; /* old probe: fields absent */
        }
    }

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
        jstr(g, "name", gpu->name, sizeof gpu->name);
    }

    json_free(root);
    return 1;
}

static int load_manifest(const char *path, Manifest *m, char *err,
                         size_t errsz) {
    char *text = json_read_file(path, err, errsz);
    if (!text) return 0;
    JVal *root = json_parse(text, err, errsz);
    free(text);
    if (!root) return 0;

    int ok = 1;
    jstr(root, "model", m->model, sizeof m->model);
    jstr(root, "variant", m->variant, sizeof m->variant);
    m->total_params = (unsigned long long)jnum(root, "total_params", &ok);
    m->active_params = (unsigned long long)jnum(root, "active_params", &ok);
    m->n_layer = (unsigned long long)jnum(root, "n_layer", &ok);
    m->mtp_head = (int)jnum(root, "mtp_head", &ok);

    const JVal *comp = json_get(root, "components");
    const JVal *att = comp ? json_get(comp, "attention") : NULL;
    const JVal *base = comp ? json_get(comp, "base") : NULL;
    const JVal *exp = comp ? json_get(comp, "experts") : NULL;
    const JVal *kv = json_get(root, "kv_cache");
    if (!att || !base || !exp || !kv) ok = 0;
    if (ok) {
        m->attention_bytes =
            (unsigned long long)jnum(att, "bytes_per_token", &ok);
        m->base_bytes = (unsigned long long)jnum(base, "bytes_per_token", &ok);
        m->expert_count = (unsigned long long)jnum(exp, "count", &ok);
        m->expert_used = (unsigned long long)jnum(exp, "used_per_token", &ok);
        m->expert_bytes = (unsigned long long)jnum(exp, "bytes_each", &ok);
        m->routed_bytes =
            (unsigned long long)jnum(exp, "bytes_per_token", &ok);
        m->kv_bytes_per_ctx_token =
            (unsigned long long)jnum(kv, "bytes_per_context_token", &ok);
    }
    json_free(root);
    if (!ok) {
        snprintf(err, errsz, "%.200s: missing or bad manifest fields", path);
        return 0;
    }
    return 1;
}

static double mtp_multiplier(const Manifest *m) {
    return m->mtp_head ? MTP_ACCEPTANCE : 1.0;
}

static void score_resident(Candidate *c, const Manifest *m,
                           unsigned long long total_bytes,
                           unsigned long long bytes_per_token,
                           const GpuTier *gpu, double gpu_flops_bound) {
    c->name = "RESIDENT";
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
    c->point = gpu->hbm_bw / bytes_per_token;
    if (gpu_flops_bound < c->point) c->point = gpu_flops_bound;
    c->point *= mtp_multiplier(m);
}

static void score_cpu_stream(Candidate *c, const Manifest *m,
                             unsigned long long total_bytes,
                             unsigned long long bytes_per_token,
                             const DramPlan *dram, double flops_bound) {
    c->name = "CPU-STREAM";
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
    c->point = dram->eff_bw / bytes_per_token;
    if (flops_bound < c->point) c->point = flops_bound;
    c->point *= mtp_multiplier(m);
}

static void score_hybrid(Candidate *c, const Manifest *m,
                         unsigned long long kv_total,
                         unsigned long long bytes_per_step,
                         unsigned long long gpu_bytes_per_step,
                         const DramPlan *dram, const GpuTier *gpu,
                         double flops_bound) {
    c->name = "HYBRID";
    c->scorable = 0;
    if (!gpu->present) {
        snprintf(c->reason, sizeof c->reason, "no gpu");
        return;
    }
    if (!gpu->bw_measured) {
        snprintf(c->reason, sizeof c->reason, "gpu bandwidth unmeasured");
        return;
    }
    unsigned long long on_gpu = m->attention_bytes + kv_total;
    if (on_gpu > gpu->vram * VRAM_USABLE_FRACTION) {
        snprintf(c->reason, sizeof c->reason,
                 "attention+kv do not fit in vram");
        return;
    }
    /* Decode is expert-byte-bound: experts stream from DRAM exactly as in
       CPU-STREAM; the gpu only takes attention+kv reads off DRAM. */
    c->scorable = 1;
    c->point = dram->eff_bw / bytes_per_step;
    double gpu_rate = gpu->hbm_bw / gpu_bytes_per_step;
    if (gpu_rate < c->point) c->point = gpu_rate;
    if (flops_bound < c->point) c->point = flops_bound;
    c->point *= HYBRID_SYNC_FACTOR * mtp_multiplier(m);
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
    c->name = "FLASH-STREAM";
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
    c->point = flash_rate_at_h(m, dram, nvme, att_base, kv_per_token, batch,
                               H_ASSUMED, flops_bound);
}

static const char *default_probe_path(void) {
    static char buf[1024];
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(buf, sizeof buf, "%s/.geode/probe.json", home);
    return buf;
}

static const char *default_manifest_path(void) {
    static char buf[1024];
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(buf, sizeof buf, "%s/.geode/manifest.json", home);
    return buf;
}

static const char *default_out_path(void) {
    static char buf[1024];
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(buf, sizeof buf, "%s/.geode/plan.json", home);
    return buf;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [probe.json] [manifest.json] [--context N] [--batch N] "
            "[--out plan.json]\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *probe_path = default_probe_path();
    const char *manifest_path = default_manifest_path();
    const char *out_path = default_out_path();
    long context = 4096, batch = 1;
    int positional = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            context = strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch = strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (argv[i][0] != '-') {
            if (positional == 0) probe_path = argv[i];
            else if (positional == 1) manifest_path = argv[i];
            else {
                usage(argv[0]);
                return 2;
            }
            positional++;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (context <= 0 || batch <= 0) {
        fprintf(stderr, "context and batch must be positive\n");
        return 2;
    }

    char err[256];
    Manifest m;
    if (!load_manifest(manifest_path, &m, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    unsigned long long kv_total = m.kv_bytes_per_ctx_token *
                                  (unsigned long long)context *
                                  (unsigned long long)batch;
    unsigned long long weights_total =
        m.attention_bytes + m.base_bytes + m.expert_bytes * m.expert_count;
    unsigned long long weights_kv = weights_total + kv_total;

    DramPlan dram;
    NvmeTier nvme;
    GpuTier gpu;
    double cpu_flops;
    const char *cpu_flops_kind;
    if (!load_probe(probe_path, &dram, &nvme, &gpu, &cpu_flops,
                    &cpu_flops_kind, weights_kv, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    /* Experts are dequantized and multiplied on the cpu in every strategy
       except RESIDENT, so the cpu flops bound applies to all of them. */
    double flops_bound = cpu_flops > 0
                             ? cpu_flops / (2.0 * m.active_params * batch)
                             : INFINITY;
    double gpu_flops_bound =
        gpu.flops_measured ? gpu.fp32_flops * GPU_DEQUANT_EFFICIENCY /
                                 (2.0 * m.active_params * batch)
                           : INFINITY;

    /* Prefill is GEMM-shaped: pick the device with the higher calibrated
       rate. The gpu number rests on an assumed dequant efficiency. */
    double cpu_prefill = cpu_flops > 0 ? cpu_flops / (2.0 * m.active_params) : 0;
    double gpu_prefill =
        gpu.flops_measured ? gpu.fp32_flops * GPU_DEQUANT_EFFICIENCY /
                                 (2.0 * m.active_params)
                           : 0;
    double prefill_rate = cpu_prefill;
    const char *prefill_device = "cpu";
    int prefill_estimated = 0;
    if (gpu_prefill > prefill_rate) {
        prefill_rate = gpu_prefill;
        prefill_device = gpu.name;
        prefill_estimated = 1;
    }

    /* Per decode step: attention+base are read once and amortize over the
       batch, but each token routes independently, so expert reads grow with
       the expected union of hit experts; kv reads scale with batch. Rates
       below are per sequence (latency-oriented). */
    double experts_hit =
        m.expert_count *
        (1.0 - pow(1.0 - (double)m.expert_used / m.expert_count, batch));
    unsigned long long kv_per_token =
        m.kv_bytes_per_ctx_token * (unsigned long long)context;
    unsigned long long routed_per_step =
        (unsigned long long)(experts_hit * m.expert_bytes);
    unsigned long long bytes_per_step = m.attention_bytes + m.base_bytes +
                                        routed_per_step +
                                        kv_per_token * (unsigned long long)batch;
    unsigned long long hybrid_dram_step =
        m.base_bytes + routed_per_step;
    unsigned long long hybrid_gpu_step =
        m.attention_bytes + kv_per_token * (unsigned long long)batch;

    Candidate cands[4];
    score_resident(&cands[0], &m, weights_kv, bytes_per_step, &gpu,
                   gpu_flops_bound);
    score_cpu_stream(&cands[1], &m, weights_kv, bytes_per_step, &dram,
                     flops_bound);
    score_hybrid(&cands[2], &m, kv_total, hybrid_dram_step, hybrid_gpu_step,
                 &dram, &gpu, flops_bound);
    score_flash_stream(&cands[3], &m, kv_total, weights_kv, kv_per_token,
                       batch, &dram, &nvme, flops_bound);

    Candidate *best = NULL;
    for (int i = 0; i < 4; i++)
        if (cands[i].scorable && (!best || cands[i].point > best->point))
            best = &cands[i];

    char size[32];
    if (cpu_flops > 0)
        printf("probed:  cpu %.0f GFLOPS %s, ", cpu_flops / 1e9,
               cpu_flops_kind);
    else
        printf("probed:  cpu flops unmeasured, ");
    printf("dram %.0f GB/s eff (%s), usable ", dram.eff_bw / 1e9,
           dram.policy);
    fmt_size((double)dram.usable, size, sizeof size);
    printf("%s, nvme %.1f GB/s", size, nvme.read_bw / 1e9);
    if (gpu.present)
        printf(", gpu %s %.1f GB%s", gpu.name, gpu.vram / 1e9,
               gpu.bw_measured ? "" : " (unmeasured)");
    printf("\n");
    printf("model:   %s (%s), %.2fB total / %.2fB active, %llu layers, "
           "%llu experts (%llu/token)\n",
           m.model, m.variant, m.total_params / 1e9, m.active_params / 1e9,
           m.n_layer, m.expert_count, m.expert_used);
    printf("workload: context %ld, batch %ld; kv cache ", context, batch);
    fmt_size((double)kv_total, size, sizeof size);
    printf("%s\n\n", size);

    printf("%-13s %-9s %-14s %s\n", "strategy", "scorable", "decode tok/s",
           "why");
    for (int i = 0; i < 4; i++) {
        Candidate *c = &cands[i];
        if (c->scorable)
            printf("%-13s %-9s %.0f-%-9.0f %s\n", c->name, "yes",
                   c->point * PESSIMISM_LO, c->point,
                   c == best ? "<- best" : "");
        else
            printf("%-13s %-9s %-14s %s\n", c->name, "no", "-", c->reason);
    }
    printf("\n");

    if (!best) {
        printf("no scorable strategy on this hardware\n");
        return 1;
    }

    if (best == &cands[3]) {
        unsigned long long att_base = m.attention_bytes + m.base_bytes;
        printf("hit-rate dial (routed %.2f GB/token, nvme %.1f GB/s):\n",
               m.routed_bytes / 1e9, nvme.read_bw / 1e9);
        for (int i = 0; i < n_h_sweep; i++)
            printf("  h=%-4.0f%% -> %.1f tok/s\n", h_sweep[i] * 100,
                   flash_rate_at_h(&m, &dram, &nvme, att_base, kv_per_token,
                                   batch, h_sweep[i], flops_bound));
        if (strstr(m.variant, "decomposed") == NULL)
            printf("note: streaming full experts; decomposition cuts routed "
                   "bytes ~7x\n");
        printf("\n");
    }

    printf("plan:    %s, numa %s", best->name, dram.policy);
    if (dram.replicas > 1) printf(" (%d replicas)", dram.replicas);
    printf("%s\n", m.mtp_head ? ", mtp on" : "");
    printf("predict: %.0f-%.0f tok/s decode (calibrating)",
           best->point * PESSIMISM_LO, best->point);
    if (prefill_rate > 0)
        printf(", prefill ~%.0f tok/s on %s%s, TTFT ~%.0fs @ %ld ctx",
               prefill_rate, prefill_device,
               prefill_estimated ? " (estimated from fp32 peak)" : "",
               context / prefill_rate, context);
    else
        printf(", prefill unmeasured (probe has no flops measurement)");
    printf("\n");

    FILE *out = fopen(out_path, "w");
    if (!out) {
        perror(out_path);
        return 1;
    }
    Json j;
    json_begin(&j, out);
    json_u64(&j, "schema", 1);
    json_string(&j, "strategy", best->name);
    json_open(&j, "workload", 0);
    json_u64(&j, "context", (unsigned long long)context);
    json_u64(&j, "batch", (unsigned long long)batch);
    json_close(&j);
    json_open(&j, "placement", 1);
    if (best == &cands[0]) {
        json_open(&j, NULL, 0);
        json_string(&j, "component", "all");
        json_string(&j, "tier", "vram");
        json_string(&j, "policy", "resident");
        json_close(&j);
    } else if (best == &cands[3]) {
        unsigned long long resident =
            m.attention_bytes + m.base_bytes + kv_total;
        unsigned long long pool = dram.usable - resident;
        if (pool > POOL_MAX_BYTES) pool = POOL_MAX_BYTES;
        json_open(&j, NULL, 0);
        json_string(&j, "component", "attention");
        json_string(&j, "tier", "dram");
        json_string(&j, "policy", "resident");
        json_close(&j);
        json_open(&j, NULL, 0);
        json_string(&j, "component", "base");
        json_string(&j, "tier", "dram");
        json_string(&j, "policy", "resident");
        json_close(&j);
        json_open(&j, NULL, 0);
        json_string(&j, "component", "experts");
        json_string(&j, "tier", "nvme");
        json_string(&j, "policy", "pool+prefetch");
        json_u64(&j, "pool_bytes", pool);
        json_string(&j, "spec_width", "confidence-gated");
        json_string(&j, "pool_partitioning", "per-sequence");
        json_close(&j);
    } else {
        int hybrid = best == &cands[2];
        json_open(&j, NULL, 0);
        json_string(&j, "component", "attention");
        json_string(&j, "tier", hybrid ? "vram" : "dram");
        json_string(&j, "policy", "resident");
        json_close(&j);
        json_open(&j, NULL, 0);
        json_string(&j, "component", "base");
        json_string(&j, "tier", "dram");
        json_string(&j, "policy", "resident");
        json_close(&j);
        json_open(&j, NULL, 0);
        json_string(&j, "component", "experts");
        json_string(&j, "tier", "dram");
        json_string(&j, "policy", "resident");
        json_close(&j);
        json_open(&j, NULL, 0);
        json_string(&j, "component", "kv");
        json_string(&j, "tier", hybrid ? "vram" : "dram");
        json_string(&j, "policy", "resident");
        json_close(&j);
    }
    json_close(&j);
    json_open(&j, "numa", 0);
    json_string(&j, "policy", dram.policy);
    json_u64(&j, "replicas", (unsigned long long)dram.replicas);
    json_close(&j);
    json_open(&j, "predicted_tok_s", 1);
    json_double(&j, NULL, best->point * PESSIMISM_LO);
    json_double(&j, NULL, best->point);
    json_close(&j);
    if (prefill_rate > 0) {
        json_double(&j, "predicted_prefill_tok_s", prefill_rate);
        json_string(&j, "prefill_device", prefill_device);
    }
    json_open(&j, "candidates", 1);
    for (int i = 0; i < 4; i++) {
        json_open(&j, NULL, 0);
        json_string(&j, "strategy", cands[i].name);
        json_u64(&j, "scorable", cands[i].scorable);
        if (cands[i].scorable) {
            json_open(&j, "decode_tok_s", 1);
            json_double(&j, NULL, cands[i].point * PESSIMISM_LO);
            json_double(&j, NULL, cands[i].point);
            json_close(&j);
        } else {
            json_string(&j, "reason", cands[i].reason);
        }
        json_close(&j);
    }
    json_close(&j);
    json_end(&j);
    if (fclose(out)) {
        perror(out_path);
        return 1;
    }
    printf("wrote %s\n", out_path);
    return 0;
}
