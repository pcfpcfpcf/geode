# GEODE — System Design

Hardware-adaptive inference for frontier MoE models. One binary, any box:
probe → plan → execute → serve. Target: batch 1–8, latency-oriented.

## 1. Problem

MoE decode is bandwidth-bound. Bytes/token read from the bottleneck tier sets
the tok/s ceiling. Stock runtimes (ollama/llama.cpp) leave 3–7× on the table:
no NUMA awareness, no MoE placement, no speculation, no measurement.

## 2. Core model

**Hardware = tiers.** Each tier: `{capacity, bandwidth, latency}` — measured, not detected.

| Tier | BW | Capacity | Latency |
|---|---|---|---|
| HBM (GPU) | ~1 TB/s | GBs | ~µs |
| DRAM (CPU) | 0.2–0.9 TB/s | 100s GB | ~100 ns |
| NVMe | ~25 GB/s aggregate | TBs | ~100 µs |

**Model = components.** After decomposition, each has `{bytes/token, access pattern}`:

| Component | Pattern | Placement rule |
|---|---|---|
| attention + router | deterministic, hot | fastest tier, resident |
| W_base + shared expert | deterministic, hot | fastest tier, resident |
| routed deltas (A_iB_i) | stochastic, cold | pool + prefetch from next tier down |
| KV cache | capacity-driven | largest fast tier that fits context |

**Decomposition:** `W_i = W_base + A_iB_i` (rank ~192, INT4).
- In RAM: bandwidth optimization (~7× fewer routed bytes/token).
- On disk: capacity enabler. Full experts ≈146GB → deltas ≈19–25GB.
  Streaming full experts from flash ≈1 tok/s; streaming deltas is viable.

**Placement = optimization:** assign components to tiers to minimize
bytes/token on the bottleneck tier, subject to capacity.

**NUMA rule — replicate, don't interleave.** If a fast tier has ≥2 NUMA nodes
and capacity allows, hold a full replica per node and pin threads socket-local.
Interleaving sends ~half of weight reads across UPI/xGMI (10–20× narrower than
local DRAM). Substrate contract applied intra-box: ship KB activations across
sockets, never MB weights.
- Single stream: alternate layers between sockets → per-socket weight traffic
  halves.
- Multi-tenant: one replica per socket = isolated tenants at full local BW.
- Trade: replica RAM vs. bigger delta pool → planner solves numerically, not
  by rule.

## 3. Components

```
┌─ OFFLINE ─────────────────────────────────────────────┐
│ ④ MODEL PIPELINE                                       │
│   HF weights → decompose → requantize → model package  │
│   package = weights + manifest.json                    │
└──────────────────────┬─────────────────────────────────┘
                       │ pull once
┌─ RUNTIME (customer box) ──────────────────────────────┐
│ ① PROBE + PLANNER     tiers × manifest → plan.json     │
│ ② EXECUTORS           one strategy, picked by planner  │
│ ③ TOOL UI             CLI + OpenAI API + stats         │
└────────────────────────────────────────────────────────┘
```

### ① Probe + Planner

Probe (first run, ~30s, cached `~/.geode/probe.json`):
- per-GPU achieved HBM bandwidth
- per-NUMA-node achieved DRAM bandwidth
- GPU interconnect (PCIe/NVLink) topology + BW
- per-drive NVMe sequential read

Planner, per phase (decode and prefill scored separately):
```
bw_bound    = min_over_tiers( tier_bw / tier_bytes_per_token )
flops_bound = peak_flops / (2 × active_params × batch)
latency_bound = 1 / (miss_latency × serialized_misses)   # flash-stream, low h

predicted_tok/s = min(bw_bound, flops_bound, latency_bound)
                  × hit_rate_factor(h)        # flash-stream only
                  × mtp_multiplier            # if draft head exists
```
- flops term is required: prefill is GEMM-shaped even at batch 1; overlapping
  requests push decode toward GEMM.
- latency term is required: at low hit rates FLASH-STREAM binds on miss
  latency (~100µs), not bandwidth. Omitting it over-promises exactly where
  the system is weakest.
- **Predictions are pessimistic by policy.** Print a calibrated band, never a
  number the box can't beat. Probe reports achieved (not theoretical) BW;
  expect 65–80% of peak. A tool that promises 85–110 and delivers 70 is
  worse than one that promises nothing.

Enumerate candidate strategies → score → emit `plan.json` + print prediction.
Re-plan if achieved diverges >30% from predicted.

### ② Executors

| Strategy | Weights in | Condition | Backend |
|---|---|---|---|
| RESIDENT | VRAM | fits in VRAM | llama.cpp / vLLM |
| CPU-STREAM | DRAM | fits in DRAM | **built in-house** |
| HYBRID | CPU-STREAM + attention/KV in VRAM | GPU present, prefill-heavy workload | **built in-house** |
| FLASH-STREAM | DRAM pool + NVMe | else | **built in-house** |

**HYBRID accelerates both prefill and decode.** Prefill: attention/KV GEMM on
the GPU. Decode gains twice, and the two gains are independent. First, moving
attention+KV to VRAM takes their bytes off the DRAM path — 495 of 1139 MB per
token on the target box at 4k ctx — which stands on its own with no expert
cache at all. Second, the parallel-tier expert cache (see below) splits routed
expert reads across HBM and DRAM, so the two bandwidth sources stack.
Attention placement is a *variable* the planner scores, not a strategy
constant.

Same box ± GPU:

| | Decode | Cold prefill | KV capacity | Sync cost |
|---|---|---|---|---|
| CPU-STREAM | 1.0× (baseline) | 1.0× | all of DRAM | — |
| HYBRID | 1.8× without expert cache, 2.4× with (see below); **grows with ctx** | 3–6× | capped by VRAM; offload back to RAM negates | per-layer PCIe transfers + pipeline bubbles |

Decode multipliers are the bandwidth model of the table below. The planner
predicts less — 1.5× point estimate on the target box — because it also charges
the cpu dequant-flops bound and a PCIe sync factor. Stage 1 measures which one
the box obeys.

Planner rule: add GPU to the plan iff `prefill_savings > pcie_overhead` at the
workload's context length and prefill:decode ratio. Warm-cache agentic loops
(prefill amortized) → GPU earns little. Cold-heavy workloads (fresh 32k docs)
→ GPU pays for itself.

**Expert cache policy — balance, don't maximize.** The pool concept
generalizes beyond FLASH-STREAM: cache hot experts at *every* tier boundary,
not just DRAM above NVMe. For HYBRID this means a VRAM expert cache above
DRAM — attention+KV resident in VRAM as before, plus as many hot experts as
the balance point calls for. Both tiers read in parallel; the bottleneck is
`max(gpu_time, dram_time)`, not the sum.

The optimal admission is the hit rate that load-balances the two tiers —
not the highest hit rate the cache can hold:
```
h_balanced = (bw_fast × (fixed_slow + movable) − bw_slow × fixed_fast)
             / (movable × (bw_fast + bw_slow))
```
where `fixed_fast`/`fixed_slow` is the non-expert work already bound to each
tier and `movable` is the routed expert bytes/token. Past this point, every
expert promoted to the fast tier makes it *slower* — you're moving work from
an underutilized tier to an overutilized one.

On the target box (HBM 34.8 GB/s, DRAM 24.3 GB/s; fixed GPU 495 MB =
attention+KV@4k, fixed DRAM 287 MB = base, movable 356 MB = routed).
CPU-STREAM reads all 1139 MB from DRAM — 46.8 ms, 21.4 tok/s — and is the
baseline both right-hand columns divide by:

| h | GPU ms | DRAM ms | bottleneck | tok/s | vs CPU-STREAM 21.4 | vs ollama 8.75 |
|---|---|---|---|---|---|---|
| 0 (no cache) | 14.2 | 26.5 | DRAM | 37.8 | 1.8× | 4.3× |
| **0.49** | **19.3** | **19.3** | **either** | **51.9** | **2.4×** | **5.9×** |
| 0.85 | 22.9 | 14.0 | GPU | 43.6 | 2.0× | 5.0× |
| 1.0 | 24.4 | 11.8 | GPU | 40.9 | 1.9× | 4.7× |

h=0.85 is worse than h=0.49. "Cache as much as fits" over-caches into
GPU-bound territory and throws away the parallel-bandwidth advantage.

The h=0 row is also why the Stage-1 gate is winnable before this cache
exists: attention+KV in VRAM alone clears CPU-STREAM by 1.8×.

**Bandwidth ratio sets the policy.** The same formula covers FLASH-STREAM
(DRAM above NVMe), but a 12× bandwidth ratio puts `h_balanced` near 1 and
makes the curve sharply asymmetric: under-caching is catastrophic (NVMe time
explodes), over-caching is nearly free (DRAM sits idle). There, "maximize h"
is the right one-sided approximation. For HYBRID (HBM above DRAM, 1.4× apart)
the curve is symmetric and you must hit the balance point — over-caching
hurts as much as under-caching. Same mechanism, different operating point,
determined entirely by the bandwidth ratio between adjacent tiers.

**Predictor target shifts accordingly.** The predictor does not aim for the
highest possible h — it aims for `h_balanced` on this hardware. For
FLASH-STREAM (far tiers) this collapses to "maximize h" and the existing
Stage-3 kill criterion (`hit rate <70% with trace pinning`) still holds. For
HYBRID (close tiers) over-prediction is a performance bug, not just wasted
cache, and the kill criterion is "achieved h within tolerance of
`h_balanced`" — missing high hurts as much as missing low.

FLASH-STREAM internals:
- **Pool:** 6–12GB DRAM of hot/predicted deltas. **Design constraint: pool is
  per-sequence-partitioned from day one**, even single-tenant. Interleaved
  tenants shuffle routing streams → predictor degrades, hit rate falls to the
  static-pinning floor. Partitioned API costs nothing now; retrofitting is a
  rewrite. Scope line: N ≤ num_sockets tenants run isolated (NUMA replicas);
  interleaved-on-shared-pool is the degraded regime — say so in the pitch.
- **Prefetch:** one full layer ahead. Routing for layer L known only at layer L
  (~0.2ms budget vs ~0.1ms NVMe latency) → prediction required.
- **Predictor (3 layers of defense):**
  1. hash-routed layers → deterministic, free perfect prefetch
  2. trace-pinned pool → static prediction, paid in capacity
  3. TAGE/Markov dynamic predictor → volatile middle
- Predictor outputs: what to prefetch, what stays resident, speculation width.
- **Width gating:** speculation width spends disk BW (80× scarcer than HBM).
  High confidence → exact top-k; low confidence → widen, budgeted.
- **Layout compiler:** trace → disk placement. Co-firing experts adjacent;
  hot experts duplicated along stream. Capacity is cheap; burn it for
  sequentiality.

Hit-rate dial (0.47GB delta bytes/token, 25 GB/s):
```
h=0%   → 19ms/token → latency-bound, 1–5 tok/s real
h=50%  → 9.4ms      → ~50–80 tok/s
h=85%  → 2.8ms      → disk ceiling > compute ceiling → compute-bound
```
Knee at h≈80–85%: prediction moves the bottleneck from disk to compute.

### ③ Tool UI

- Verbs: `run`, `serve`, `pull`, `ps` (ollama-compatible mental model).
- OpenAI-compatible API on :11434.
- Prints plan + prediction before serving:
```
probed:  2× Xeon 16ch DDR5 (460 GB/s achieved of 614 peak), 700GB RAM,
         4× NVMe (23 GB/s)
plan:    CPU-STREAM, Q4, MTP on, per-NUMA replica + socket-local pinning
predict: 60–90 tok/s decode (calibrating), TTFT ~4s @ 4k
```
- Live stats: tok/s, pool hit rate, draft acceptance. Publish alongside
  every benchmark number.
- Graceful degradation: if best plan is slow, say why and what hardware
  change lifts it.

### ④ Model pipeline (offline, the IP)

**CI-shaped, not artisanal.** New MoE drops → pipeline runs unattended:
1. **Day one:** convert to `full-q4` (hours). Planner already handles full
   weights → "yes" immediately on any new model.
2. **Later:** decompose routed experts (Fisher-weighted base + SVD deltas,
   sweep r ∈ {128, 192, 256}) → requantize INT4. Risk: stacked quant on
   natively-FP4 weights — kill criterion, gate on task evals, not perplexity.
3. Ship `decomposed` variant only when the eval gate passes. Decomposition is
   an upgrade, not a prerequisite — **except on the disk tier, where it is
   mandatory** (no day-one story sub-RAM; that tier is ours alone anyway).
4. Emit package: weights (both variants) + `manifest.json` + **signed eval
   report** (`full-q4` vs `decomposed` on a fixed public suite + customer task
   set). The eval report is the compliance answer for gov-adjacent buyers and
   automates the Stage-2 kill criterion: it becomes a diff against a signed
   baseline, not a judgment call.

## 4. Contracts (write these first)

**manifest.json** — offline ↔ runtime. Components carry `bytes_per_token`
(decode reads every layer once per token). Full-weight packages report
`experts`; decomposed packages report `deltas` instead:
```json
{
  "model": "deepseek-v4-flash",
  "components": {
    "attention": {"bytes_per_token": "...", "pattern": "deterministic"},
    "base":      {"bytes_per_token": "...", "pattern": "deterministic"},
    "experts":   {"count": 256, "used_per_token": 8,
                  "bytes_each": "...", "bytes_per_token": "...",
                  "pattern": "stochastic"},
    "deltas":    {"count": 256, "bytes_each": "...", "rank": 192}
  },
  "kv_cache":  {"bytes_per_context_token": "..."},
  "hash_routed_layers": [0, 1, 2],
  "mtp_head": true,
  "variants": ["full-q4", "decomposed-r192-q4"],
  "eval_report": {
    "suite": "geode-eval-v1 + customer-set",
    "full-q4": {"score": 0.0},
    "decomposed-r192-q4": {"score": 0.0, "delta_vs_full": 0.0},
    "signature": "..."
  }
}
```

**plan.json** — planner ↔ executor:
```json
{
  "strategy": "FLASH-STREAM",
  "placement": [
    {"component": "attention", "tier": "dram", "policy": "resident"},
    {"component": "base",      "tier": "dram", "policy": "resident"},
    {"component": "deltas",    "tier": "nvme", "policy": "pool+prefetch",
     "pool_bytes": "8G", "spec_width": "confidence-gated",
     "pool_partitioning": "per-sequence"}
  ],
  "numa": {"policy": "replicate", "replicas": 2},
  "predicted_tok_s": [60, 90]
}
```

## 5. Speculation (decode multiplier)

- MTP/draft head where present (V4 has one): draft base-only (r=0, no delta
  reads), verify batches one delta read over k tokens.
- Effective tok/s ×= acceptance (target ≥2.0 tokens/verify; kill: <2.0 →
  ship without).
- Prefill: expert-major reorder (sort tokens by expert) → one sequential
  delta sweep per layer per chunk. Deterministic, no prediction needed.
- Prefix cache: persist KV across agentic re-turns (~10GB @ 1M ctx).

## 6. Build order (each stage gates the next)

| Stage | Deliverable | Kill criterion |
|---|---|---|
| 0 | probe + bench harness; baseline on target box | box BW too low → re-scope promise |
| 1 | **in-house executor**: CPU-STREAM first (all on DRAM), then attention+KV on GPU → HYBRID. Minimal scope — one arch, Q4_K_M, batch 1–8; port ggml kernels as reference | HYBRID achieved < CPU-STREAM achieved |
| 2 | decomposition pipeline | quality loss on task evals → full-weight fallback |
| 3 | expert cache: pool, predictor, balance admission, parallel-tier read. First application: HYBRID (VRAM above DRAM) | achieved h outside tolerance of `h_balanced` |
| 4 | FLASH-STREAM executor (NVMe prefetch, layout, latency term) — expert cache applied to the disk tier | hit rate <70% with trace pinning |
| 5 | MTP speculation | acceptance <2.0 |
| 6 | RESIDENT parity, polish, registry | — |

**Stage 1 is in-house by necessity, not ambition.** llama.cpp offloads whole
layers — component-level placement is impossible inside its scheduler — and
kt-kernel won't run on Maxwell. Placement is the product; it can't be delegated
to a backend. Scope stays minimal: one architecture, one quant, batch 1–8,
kernels ported from ggml as reference. The moat still starts at Stage 2
(decomposition) and 3 (expert cache) — the executor is the vehicle, not the IP.

## 7. Competition / gap

| Tool | UX | MoE hetero | Planner | Measures HW | Disk tier |
|---|---|---|---|---|---|
| Ollama | ✅ | ❌ | ❌ | ❌ | ❌ |
| KTransformers | ❌ (YAML) | ✅ | manual | ❌ | ❌ (KV only) |
| FlexGen | ❌ | ❌ | ✅ (throughput) | ❌ | ✅ (archived 2024) |
| **GEODE** | ✅ | integrated | ✅ auto, latency | ✅ | ✅ weights |

Gap = planner-as-product + measured hardware + printed prediction +
disk-tier weights via decomposition.
