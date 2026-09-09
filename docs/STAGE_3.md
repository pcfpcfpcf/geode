# GEODE — Stage 3: Expert Decomposition

Stages 1 and 2 are done. The in-house executors sit on the DRAM roofline, and
STAGE_1_5.md establishes there is no more decode speed to find on the
executors alone for a model that already fits in RAM. Every remaining win —
and the entire premise of running a model that does *not* fit — is downstream
of one thing that is not built: the offline pipeline that decomposes routed
experts into a shared base plus low-rank deltas.

This doc is the plan for that pipeline, and first for the experiment that
decides whether it is worth building.

## 1. What decomposition buys

An MoE layer's experts are not independent — trained from related
initializations, reading the same residual stream, they share a large common
component. So each expert's weight splits:

```
W_e = W_base + ΔW_e ,   ΔW_e ≈ A_e B_e    (A_e: d_out×r, B_e: r×d_in)
```

`W_base` is one matrix per layer, read once per token — it joins attention and
the shared expert in the deterministic, resident set. Only the small deltas
are read stochastically. For GLM/DeepSeek expert shapes (wide `d_model`,
narrow `d_ff`) at r≈128 this is a 7–9× cut in routed bytes/token and a
comparable cut in total expert storage.

That cut changes what "fits" means at every tier boundary. DeepSeek-V4-Flash
(284B total, 13B active) as the worked example:

| hardware | full weight | decomposed r≈128 |
|---|---|---|
| XPS 15, 32 GB RAM | 180 GB on NVMe → ~0.3 tok/s | ~21–30 GB in RAM → ~6 tok/s |
| 4×RTX 3090, 96 GB VRAM | ~63 GB offloaded over PCIe → ~15–25 tok/s | 0 offloaded, all VRAM → ~80–150 tok/s, more with MTP |
| 1×3090 + 128 GB RAM | ~10% of experts in VRAM | ~100% in VRAM |

Same lever, 6–20× either way. It is the one optimization in the design that is
not hardware-conditional: HYBRID, the pool, and MTP each pay only in some
bandwidth regime (STAGE_1_5.md §4); decomposition pays in all of them.

## 2. The load-bearing unknown

Low-rank truncation plus INT4 requantization is lossy. The whole product
rests on it being lossy in a way that does *not* cost the model's coding and
agentic ability — the capability these targets are chosen for.

- **Gate on task evals, not perplexity.** Perplexity moves too little to
  catch a multi-point drop on SWE-bench-style work. The gate is a coding
  suite run on `full-q4` against the decomposed variant.
- **Set the tolerance before the run.** Decide the acceptable aggregate delta
  and the per-category regression limit up front. A tolerance chosen after
  seeing the numbers is not a gate.
- **Kill criterion:** decomposed variant outside tolerance at every tried
  rank → ship `full-weight` only for that model. The full-weight path already
  works; decomposition is an upgrade, not a dependency — except on the NVMe
  tier, where there is no full-weight story and the model does not run without
  it.

This section is why the sequence in §5 starts on a 30B model, not on the 284B
target.

## 3. The pipeline

Offline, one run per model drop, unattended.

**a. Ingest.** Parse the GGUF, classify every tensor — routed expert / shared
expert / attention / dense-leading / embed — the split
`src/manifest/manifest.c` already computes for the full-weight path.
Dequantize experts to a bf16/fp32 working copy, streamed layer by layer (one
layer's experts are a few GB; the whole model never resides).

**b. Calibrate.** Run a few hundred domain-relevant sequences (code-heavy for
these targets), ~512 tokens each, capturing per-expert input activation
second-moment matrices `C_e = E[x xᵀ]` and, optionally, diagonal Fisher
`E[(∂L/∂w)²]`. These make the decomposition data-aware — it minimizes error
on the activations the model sees, not on the raw weights.

**c. Choose `W_base` per layer.**
- No shared expert (qwen3moe): `W_base` is the importance-weighted mean of the
  layer's experts — `argmin Σ_e ‖(W_e − W_base) C_e^½‖²_F`.
- Shared expert present (deepseek2): the shared expert *is* the anchor.
  Decompose `W_e = W_shared + ΔW_e`, or fit a small residual base on top of
  `W_shared`. Test both.

**d. Low-rank delta per expert.** `R_e = W_e − W_base`; activation-aware
truncated SVD: `R_e C_e^½ = U S Vᵀ`, keep top-r, `A_e = U[:,:r]√S`,
`B_e = √S Vᵀ[:r] C_e^{-½}`. The `C_e` sandwich is what makes low rank survive
— plain SVD of `R_e` minimizes weight error and falls apart below r≈256.
Sweep r ∈ {128, 192, 256}, and lower where experts are fine-grained.

**e. Requantize.** `W_base` → group-wise INT4 (Q4_K shape); afford Q5/Q6 here
if the gate is tight, there is only one per layer. `A_e`, `B_e` → INT4,
watching the singular-value dynamic range — may need per-column scales or `S`
held separately.

**f. Eval gate.** §2. `full-q4` vs `decomposed-rN-q4` on the fixed coding
suite. Emit the signed report — it is both the ship decision and the
compliance artifact.

**g. Package.** Manifest with a `deltas` component (`count`, `bytes_each`,
`rank`) replacing `experts`; both variant streams; the signed report. Disk
layout for the NVMe tier (co-firing experts adjacent, hot ones duplicated
along the stream) is a Stage 4 concern.

## 4. The runtime path

`W_base[layer]` loads resident, alongside attention and the shared expert. At
decode the router picks its experts; each branch computes
`y = W_base·x + A_e·(B_e·x)` — a change to the per-branch matmul in
`run_branches`, not a new code path. The low-rank multiply adds a little FLOP,
but the base matmul is shared across the layer's active experts, so decomposed
active FLOP/token comes out *below* full-weight.

Validate decomposed logits against the Python reference to a fixed tolerance,
the way `exec-prefill` validates chunked prefill against single-token.

## 5. Validation sequence

Three models, smallest first, each answering a question the next one assumes.

**1. Qwen3-30B-A3B — method and eval harness.**
- Already runs in geode; already instrumented (router dump, trace).
- The pessimistic architecture: qwen3moe has no shared expert and no dense
  leading layer, so `W_base` is learned purely from expert averaging with
  nothing to anchor it. If decomposition holds coding here, the shared-expert
  models are easier.
- Build here: the Python pipeline (steps b–e), the coding eval harness
  (HumanEval + MBPP + an aider-style task set), the C runtime path with its
  reference check.
- Output: coding delta vs `full-q4` at r ∈ {128, 192, 256}, plus a look at
  smaller r since Qwen's experts are not fine-grained.

**2. GigaChat-10B (`model.gguf`) — the target architecture, small.**
- deepseek2: MLA attention, `expert_shared_count=1`,
  `leading_dense_block_count=1`, fine-grained experts. DeepSeek-V4-Flash's
  architecture at 1/28th the size, already on disk.
- Tests the `W_e = W_shared + ΔW_e` variant — the anchored decomposition the
  real target will use — and confirms the runtime handles shared-expert +
  dense-leading + MLA in the decomposed path.

**3. DeepSeek-V4-Flash-0731 — the target.**
- Only after 1 and 2 clear their gates.
- Decompose from the **Q8 GGUF** (~180 GB, near-lossless). Q8 → SVD → INT4
  stacks mildly; close enough to a clean signal, and it saves the ~570 GB
  bf16 pull.
- Calibration forward passes on 284B are ~10–20 h on the XPS — run this one on
  a rented GPU box. Method development stays local, on models 1 and 2.

**Why not straight to DeepSeek.** The 180 GB pull buys a slow-iterating
experiment on the model where a bug in the activation-aware SVD, or a rank
that is too aggressive, costs the most to discover. Models 1 and 2 cost
nothing — they are in the repo directory — and model 1 is the harder case for
the method, so a pass there is real evidence.

## 6. Kill criteria

| step | proceed if | else |
|---|---|---|
| Qwen r-sweep | coding delta within tolerance at some r ≤ 256 | decomposition is dead on this class of model — stop, reconsider the thesis |
| GigaChat | shared-expert-anchored decomposition at least matches Qwen's result | debug the deepseek2 path before scaling |
| DeepSeek-V4-Flash | signed eval report within tolerance | ship `full-weight` only; NVMe tier unavailable for this model |
| runtime path | decomposed logits within tolerance of the reference | fix before any perf claim |

## 7. What exists, what's missing

**Exists:** tensor classification and bytes/token accounting (`manifest.c`);
planner hooks for `"decomposed"` variants and the FLASH-STREAM hit-rate math;
per-branch expert streaming in `run_branches`; the router dump and trace from
Stage 1.5.

**Missing:** the entire offline pipeline (no decompose command, no calibration
harness, no SVD, no requantizer); the base+delta runtime matmul; the coding
eval harness; the manifest `deltas` schema wiring.
