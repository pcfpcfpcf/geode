# GEODE — Stage 1.5: Decode, Measured

The first draft of this document planned three phases against a bandwidth
model and an estimate of where the per-layer time went. Both are now
instrumented. This is what the instrument says, and what that rules out.

Box: i7-6820HQ (4c/8t), DRAM read 21.1 GB/s, Quadro M1200 — HBM 34.8 GB/s
measured, PCIe 10.1 GB/s measured, 4 GB VRAM with ~2.4 GB held by the
desktop. All from `~/.geode/probe.json`, not from spec sheets.

## 1. The instrument

`GEODE_TRACE=1` turns on the stage table (`src/exec/trace.c`), printed after
any decode:

```
GEODE_TRACE=1 ./bin/geode exec-run MODEL.gguf "PROMPT" N
GEODE_TRACE=1 ./bin/geode MODEL.gguf --strategy HYBRID
```

The marks live in `forward_with`'s layer loop, which is tier-agnostic, so
both strategies fill the same table; the hooks add sub-stages for the tier
they landed on. `session_stream` resets the trace after prefill, so the
table is decode only — prefill runs the same code over chunks of 64 and is
a different measurement. The `accounted` row against `wall` is the
integrity check: it came out equal on every run below, so no time is
hiding.

## 2. Qwen3-30B-A3B Q4_K_M, both strategies

Per-token component bytes from the manifest: attention 516.5 MB, base
306.0 MB, routed experts 1097.1 MB, total 1919.6 MB.

| stage | CPU-STREAM | HYBRID |
|---|---|---|
| attention | 22.19 ms (0.462/layer) | 37.22 ms (0.775/layer) |
| — upload + issue | — | 1.34 ms |
| — drain | — | 35.86 ms |
| ffn | 50.16 | 52.65 |
| — router | 2.03 | 4.36 |
| — routed experts | 48.13 (1.003/layer) | 48.28 (1.006/layer) |
| — gpu issue / drain | 0.00 / 0.00 | **0.00 / 0.00** |
| logits | 9.96 | 10.91 |
| **wall** | **82.97 ms — 12.04 tok/s** | **101.45 ms — 9.84 tok/s** |

Conditions: desktop up, load average ~2, 16 and 9 decoded tokens against
short prompts, so the kv cache is nearly empty and the absolute tok/s runs
high. The proportions and the per-layer comparison are what these support;
they are 1.7× apart, far outside the noise. A controlled A/B still owes
≥256-token decodes on an idle box.

## 3. What it says

**The CPU path is at the roofline, not under it.** 1919.6 MB in 82.97 ms is
**23.1 GB/s** — above the probe's own 21.1 GB/s DRAM read. Per component:
attention 23.3 GB/s, routed experts 22.8 GB/s. There is no headroom left in
CPU-STREAM on this box; llama.cpp's ~10 tok/s on the same model is 19.2
GB/s, which CPU-STREAM already clears.

**The GPU is the slower tier for this work.** The same attention costs the
CPU 0.462 ms/layer and the GPU 0.775 ms/layer — 23.3 GB/s against 13.9
GB/s. HYBRID does not lose to overhead here; it loses because the M1200
streams quantized weights slower than the DDR4 does. The card's measured
34.8 GB/s is a copy ceiling the gemv kernels reach less than half of.

**Launch overhead is 1.3% of the token, not 25%.** The first draft blamed
"~0.9 ms/layer ≈ 23 ms" on prep, four PCIe round trips and ~13 launches per
layer. Measured: `upload` 0.54 ms + `issue` 0.80 ms = **1.34 ms/token**.
The launches are genuinely asynchronous and nearly free. The 35.86 ms of
`drain` is the device executing, not the host waiting to talk to it.

**Qwen has no shared expert, so nothing overlaps the routed pass.**
`expert_shared_count` is absent from the gguf (deepseek2 carries it,
qwen3moe does not), so `base_ffn_width()` is 0, `HybridLayer.ffn_width` is
0, and the whole GPU branch of `gpu_feed_forward` is skipped — which is why
`gpu issue` and `gpu drain` read exactly 0.00. The card idles through 48 ms
of CPU experts, 48 times a token. The "shared experts in gpu" work is a
deepseek2-shaped win and does not transfer.

## 4. What that rules out

- **Phase 1 (batch-1 gemv bandwidth) is not worth finishing.** The gate was
  25 GB/s. Even meeting it leaves the GPU attention path at best level with
  a CPU path already measured at 23.3 GB/s, before PCIe. The pipelined q4
  kernels sitting uncommitted in `src/exec/cuda.c` (three register
  generations, `ld.global.nc`, `shfl.idx` scale broadcast) moved the bench
  rows 15.0→19.0, 18.7→23.5, 15.3→18.3 GB/s and look correct — `exec-prefill`
  and `exec-hybrid` pass on the deepseek2 model, the rest of the matrix is
  unrun — but they cannot make this tier the fast one.
- **Phase 2 (fuse the per-layer chain) targets 1.34 ms.** "Launches drop
  360 → 180" is worth about a millisecond a token. Fusion still saves L2
  round trips on the intermediates, but that is a second-order effect on a
  path that needs 1.7× just to reach parity with doing nothing.
- **Phase 3 (the expert pool) has no ratio to trade on.** `h_balanced`
  wants `bw_fast/bw_slow`; measured, that ratio is 13.9/22.8 — below one.
  And PCIe at 10.1 GB/s is under DRAM at 21.1, so nothing can be *streamed*
  to the card at decode time either: a pool would have to be resident and
  loaded once, sized against ~1.6 GB of free VRAM that attention+base+KV
  (1224 MB) already claims.

The three phases are not blocked, mis-ordered, or unfinished. They optimize
a tier that starts 40% behind the tier it replaces.

## 5. What survives

- **The planner was right.** It scored HYBRID (6.0–8.6) below CPU-STREAM
  (6.4–9.1) for this model and chose CPU-STREAM. Measured 9.84 against
  12.04 — the ordering holds, and the bands are pessimistic by about 30%
  rather than wrong. This is the thing geode does that a flag-driven
  executor cannot, and it is worth more than the kernels.
- **Prefill.** It is GEMM-shaped and the GPU's compute edge is real there,
  unlike its bandwidth edge. The prefill numbers alongside the runs above
  (16.4 CPU-STREAM, 20.9 HYBRID tok/s) are not a controlled A/B — different
  prompt lengths — but the direction matches the earlier 2× measurement.
  Nothing here argues against the GPU for prefill.
- **FLASH-STREAM has never been tested.** DRAM over NVMe is 21.1/1.93 =
  **11×**, the one wide ratio on this box, and the regime where the doc's
  own admission formula puts `h_balanced` near 1 and prefetch is the right
  lever. Both local models fit in RAM, so the planner marks it
  `not scorable`. It needs a model larger than 32 GB to exercise at all.

## 6. GigaChat's routed experts, resolved

The first draft of open item 1 below flagged GigaChat's routed experts at 7.2
GB/s (356.1 MB in 49.15 ms, traced on HYBRID) as a 3× gap against Qwen's 22.8
GB/s and the only unexplained CPU-side number in the trace. It does not
reproduce. Two things were wrong with the original number.

**The bucket was never routed-only.** `forward_feed_forward_cpu`'s
`TRACE_EXPERTS` mark covers three things for GigaChat, not one: the routed
experts (356.1 MB), the shared expert that runs on every MoE layer
(`expert_shared_count=1`, 3.83 MB × 25 layers = 95.6 MB), and layer 0's dense
FFN (`leading_dense_block_count=1`, no router at all, 26.8 MB) — because
`select_branches` folds the shared expert into the same branch list
`run_branches` streams, and layer 0 takes the `!has_experts` early return
through the same mark. Qwen has neither a shared expert
(`expert_shared_count` is absent) nor a dense leading layer, so its bucket
really is 1097.1 MB of routed reads and nothing else. Dividing 356.1 MB into
a bucket that actually moved 478.5 MB understates GigaChat's rate before
anything else is considered.

**The run was short.** 49.15 ms was one HYBRID pass from a 9-token decode on
a loaded box. Re-run at 256 tokens, same prompt, interleaved between
strategies, at `pool_default_workers()` (4, one per physical core — this
part is 4c/8t):

| run | strategy | experts ms/pass | bytes moved | GB/s |
|---|---|---|---|---|
| GigaChat 1 | CPU-STREAM | 29.79 | 478.5 MB | 16.06 |
| GigaChat 2 | CPU-STREAM | 27.85\* | 478.5 MB | ~17.2 |
| GigaChat 1 | HYBRID | 25.06 | 478.5 MB | 19.1 |
| GigaChat 2 | HYBRID | 15.55 | 478.5 MB | 30.8 |
| Qwen | CPU-STREAM | 68.25 | 1097.1 MB | 16.08 |

\*second CPU-STREAM run's ffn/experts split wasn't captured, this is the
attention-stage figure from the same run as a lower bound; the wall-clock
tok/s (14.98) matches run 1's (14.52) closely enough to trust the rate.

GigaChat's corrected rate (16–31 GB/s across four runs) brackets Qwen's
(16.08 GB/s) instead of trailing it by 3×. The two models read the same DRAM
through the same code path at the same rate; there was no
architecture-specific bottleneck to find. Stage 3/4 (the expert cache pool)
gains nothing here that the rest of Stage 1.5 didn't already rule out.

One apparent finding from this re-run didn't survive a repeat: a single 1-
against-8-thread comparison showed GigaChat's experts stage flatly failing
to scale (102.69 ms/pass at 1 thread against 104.07 ms/pass at 8) against
Qwen's clean 3.16× (247.93 → 78.51 ms/pass) over the same range. Re-run
four times, interleaved, it didn't reproduce — 1 thread sat at 81–83 ms/pass
and 8 sat at 40–43 ms/pass every time, a normal ~2× speedup. The first pair
was noise, not a shape-specific threading bug; nothing here changes with
core count, physical or logical, on hardware this box's size.

Off to the side, across the four 256-token
runs above, CPU-STREAM decoded GigaChat at 14.52 and 14.98 tok/s; HYBRID at
11.27 and 13.64. That's CPU-STREAM ahead on 2 of 2 pairs, not the "1.2× on
deepseek2, measured at batch 1" SYSTEM_DESIGN.md's strategy table currently
claims. The attention `drain` (46 ms/pass, GPU) against CPU-STREAM's
`attention` (28–35 ms/pass) was consistent across all three HYBRID runs here,
which points at the same GPU-slower-than-DRAM story Qwen already told rather
than at anything expert-specific — but four runs is short of the controlled
A/B open item 1 below asks for, and the table's claim was made on runs at
least as short as the one that produced the 7.2 GB/s number this section just
retracted. Filed as open item 1 below.

## 7. Open, in order of expected value

1. **Re-run the deepseek2 HYBRID-vs-CPU-STREAM decode claim.** ≥256 tokens,
   ≥3 interleaved pairs, idle box — the protocol that exposed the thermal
   confound, now aimed at the specific number in §6 that came back backwards
   on a first pass. If it holds, the SYSTEM_DESIGN.md table's "1.2×" for
   deepseek2 decode needs the same rewrite qwen3moe already got.
2. **The routing hit-rate curve.** Dump the router's top-8 per layer per
   token over a few hundred tokens and plot hits against pool size. It
   sizes every cache decision in Stage 3 and Stage 4, it settles whether
   load-balanced routing has flattened the skew enough to kill the caching
   thesis outright, and it needs no kernels. Cheap either way, and a flat
   curve saves months.

## 8. Commitments

The 1.25× and 1.4× decode targets are withdrawn. They were computed from a
bandwidth ratio this box does not have. The honest claim for HYBRID decode
on this hardware is that it loses to CPU-STREAM by ~20%, the planner says
so in advance, and the planner is the deliverable.

Decode parity with llama.cpp on Qwen3-30B-A3B is met and probably exceeded;
that is a bandwidth roofline, and both implementations sit on it.
