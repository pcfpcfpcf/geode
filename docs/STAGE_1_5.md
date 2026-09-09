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

## 6. Open, in order of expected value

1. **GigaChat's routed experts run at 7.2 GB/s.** Traced on HYBRID: 356.1
   MB of routed expert reads in 49.15 ms, against Qwen's 22.8 GB/s for the
   same kind of work — and GigaChat does *fewer, larger* per-branch reads
   (3.42 MB over 104 branches, against 2.86 MB over 384). Three times off
   roofline on the README's actual target model is the only unexplained
   CPU-side gap in the trace, and the only place a large decode win is
   still plausibly sitting. Next measurement: the same model on CPU-STREAM,
   traced, to see whether the gap is the expert path itself or something
   the hybrid split introduces.
2. **The routing hit-rate curve.** Dump the router's top-8 per layer per
   token over a few hundred tokens and plot hits against pool size. It
   sizes every cache decision in Stage 3 and Stage 4, it settles whether
   load-balanced routing has flattened the skew enough to kill the caching
   thesis outright, and it needs no kernels. Cheap either way, and a flat
   curve saves months.
3. **A controlled decode A/B**, ≥256 tokens, ≥3 interleaved pairs, idle
   box — the protocol that exposed the thermal confound, now that there is
   a stage table to read alongside the totals.

## 7. Commitments

The 1.25× and 1.4× decode targets are withdrawn. They were computed from a
bandwidth ratio this box does not have. The honest claim for HYBRID decode
on this hardware is that it loses to CPU-STREAM by ~20%, the planner says
so in advance, and the planner is the deliverable.

Decode parity with llama.cpp on Qwen3-30B-A3B is met and probably exceeded;
that is a bandwidth roofline, and both implementations sit on it.
