# GAP-ANALYSIS.md — our Swift port vs the Python-MLX oracle

**Produced 2026-08-04** by a full file:line audit of `dgrauet/ltx-2-mlx` v0.14.12 against
`ltx-2-mlx-swift` v0.2.0. Scale marker: oracle 25,716 LoC Python; Swift 7,703 LoC (shipping cores
~4,900 after the gate CLI + bench harness). The port was always scoped as *"a Swift-integration
job, not a model port"* — this document is the honest inventory of what that scoping left out, and
of the places the port is **more serial than the thing it mirrored**.

Companion docs: `TILING-PLAN.md` (modality tiling, the DiT attention wall) · `SPEED-PLAN.md`
(S2/S3 already carry two of these) · `../CLAUDE.md` (methodology, the mirror-the-loop-structure rule).

---

## 1. The capability inventory, in one table

| Area | Oracle | Swift | Note |
|---|---|---|---|
| Distilled pipeline (1-stage + 2-stage) | ✅ | ✅ | **this is what we ship** |
| One-stage / two-stage **dev + CFG** | ✅ | ❌ | structure exists, guidance does not |
| **res_2s** second-order sampler (HQ) | ✅ | ❌ | `DenoiseLoop.swift:9` defers it explicitly |
| a2v (audio→video) | ✅ 🟡beta | ❌ | oracle admits sync varies with prompt |
| Keyframe interpolation | ✅ | ❌ | needs the attention-mask system |
| Retake / extend | ✅ 🟡beta | ❌ | DiT *primitive* (per-token σ) landed; pipeline didn't |
| IC-LoRA | ✅ | ✅ reduced | no sub-1.0 attention-strength |
| HDR IC-LoRA / LogC3 | ✅ | ❌ | needs CFG two-stage first |
| LipDub | ✅ 🔴exp | ⚠️ ingest primitives only | both independently concluded: mux the real dub |
| Training (`ltx-trainer`, 4,238 LoC) | ✅ | ❌ | out of scope for an inference port |
| Upsamplers | **3** (x2, x1.5 rational, temporal x2) | **1** (x2, hardcoded filename) | oracle *can load* all three; we cannot |
| Guidance (CFG/STG/modality/rescale) | ✅ whole subsystem | ❌ **none** | see §2 |
| Attention masks (`mask_utils`) | ✅ | ❌ `mask: .none` hardcoded | blocks 4 features |
| Step count / sigma schedule | configurable + `ltx2_schedule` token-adaptive | **2 hardcoded tables** | 8 steps / 3 steps |
| Decode tiling | budget-driven auto-engage | fixed per-tier constant | **our tiling math is better**, our *policy* is absent |
| Encoder tiling | ✅ `tiled_encode` | ❌ | |
| Streaming decode → muxer | ✅ generator, never materializes | ❌ concatenates everything | |

---

## 2. 🚨 The largest single gap: there is no guidance subsystem at all

`LTX2Pipeline.swift:200 encodePrompt` encodes **one** prompt and returns one `(video, audio)`
embed pair — there is no negative encode anywhere. No CFG, no STG, no modality guidance, no
rescale. The oracle ships `MultiModalGuider` + a 4-kind `PerturbationType` (skip a2v-cross,
v2a-cross, video-self, audio-self attention) with LTX-2.3 defaults `cfg_scale=3.0`,
`stg_scale=1.0`, `stg_blocks=[28]`, `rescale=0.7`, `modality_scale=3.0`, audio `cfg=7.0`.

**Why it dominates the ranking: it is the shared prerequisite for six oracle pipelines** — dev
one-stage, two-stage, HQ, a2v, keyframe, retake are all "distilled **+ CFG**", and we shipped only
the distilled half. It is also the single biggest quality lever we are missing.

🔑 **Design note for when it lands:** the oracle runs its 2–4 guidance passes as *separate,
sequential, batch-1 full 48-block forwards* (`samplers.py:775`, `:798`; res_2s repeats the set
twice per outer step). **Batch them on the batch axis instead** (B=2 for CFG). Besides halving
latency, under block streaming it **halves granule I/O** — the streamer reads each block's weights
once for B=2 rather than twice. Do not inherit the oracle's serial pass structure.

---

## 3. Serial constructs — where we mirrored, and where we are WORSE

The port's doctrine (*"mirror the oracle's LOOP structure, not just its math"*) means most serial
structure is deliberate and correct. The audit separates the three cases.

### 3a. Correctly mirrored — leave alone
DiT 48 blocks, connector 8 blocks, Gemma 48 layers, VAE ladders, vocoder upsample stages, denoise
steps — all genuine residual/sequential chains in both.

⚠️ **Correction to an earlier read of mine.** I flagged the audio-VAE mel `for b in range(B): for
c in range(C)` loop as a discarded parallelization win. **On cost, that was wrong: B·C = 2**, so the
loop is two iterations and the performance stake is nil. The bit-exactness rationale
(`AudioVAE.swift:175-178`: batching changes the rfft kernel split → 7e-7 log-mel drift → 0.19
latent maxAbs through the 10-conv stack) stands, and the mirror should **stay**. The interesting
question it raises — *whether the oracle's per-channel loop or our batched version was closer to
PyTorch* — survives as a correctness curiosity for the PyTorch-oracle work, not as a perf item.

### 3b. Parallelizable, mirrored, worth revisiting later
- **Vocoder MRF resblocks** (`Vocoder.swift:89`): 3 independent dilation chains read the same `x`
  and are summed — a pure fan-out, batchable. Small share of wall-clock; low priority.
- **BWE per-channel resample** (`Vocoder.swift:69`): C=2, trivially batchable as `(B·C, T)` — the
  file already does exactly that reshape 30 lines earlier.
- **RoPE per-axis loop** (`RoPE.swift:35`): 3 axes; one broadcast would replace it.
- **Connector `.item()` inside a batch loop** (`Connector.swift:250-251`): a device→host sync at
  the top of the connector. B=1 in production, so one flush, not N.

### 3c. 🚨 Swift is MORE serial than the oracle — pure regressions, no bit-exactness defence

| Swift | What | Oracle equivalent |
|---|---|---|
| **`FrameCodec.swift:40-41`** | **per-pixel scalar CPU RGB→BGRA repack, per frame**, after a per-frame `asArray` device→host copy (`:29`) | one bulk `memoryview` memcpy (`video_vae.py:617`) |
| `VideoInput.swift:56` | per-pixel RGBA8→float on every ingested reference frame | vectorized `np.asarray`/PIL |
| `ImageInput.swift:50,87` | per-pixel i2v init frame | same |
| `AudioVAE.swift:191` | nested host loop building an `[Int32]` frame-index array, scales with audio length × 1024 | `mx.arange` broadcast (`processor.py:147`) |
| `AudioVAE.swift:159-162` | 64×513 = 32,832 scalar iterations for the mel filterbank | numpy-vectorized over `fft_freqs` |
| `DenoiseLoop.swift:94,143` | **blocking `eval`** every denoise step | **`mx.async_eval`** (`samplers.py:169,563,572,699,879`) |

**`FrameCodec` is the worst by a wide margin**: ~43.6 M scalar iterations plus 121 GPU→CPU round
trips at 704×512×121f, for work the oracle does as a single bulk copy. It is already scoped
internally as `SPEED-PLAN.md` **S2 step 2 (zero-copy handoff)** and still open. The `asyncEval`
swap is two lines and restores host/GPU overlap on *every step of every run*.

---

## 4. `eval` barriers — faithful, but with two deltas

The port reproduces the oracle's Metal-watchdog discipline at every site (Gemma per-layer,
connector per-block, DiT every 8, VAE stage/tile boundaries) and **adds one the oracle lacks**
(chunked wide text projection, `Connector.swift:139-147`, from the I5 cold-run trip). Two deltas:

1. **Every cadence is hardcoded in Swift.** The oracle exposes `LTX2_GEMMA_EVAL_EVERY` /
   `LTX2_DIT_EVAL_EVERY` (and documents `=0` so Ultra owners can recover lazy-graph throughput).
   `Gemma3+AllHiddenStates.swift:58` *names* the env var in a comment but never reads it.
2. **Blocking vs async** — §3c above.

---

## 5. Inherited assumptions — the highest-risk list

Ranked by what could silently break us. Full list in the audit; the load-bearing ones:

1. 🚨 **We hardcode values the oracle learned the hard way to read from the checkpoint.** The
   oracle was bitten **twice** by exactly this: `av_ca_timestep_scale_multiplier` (every LTX-2.3
   checkpoint ships `1000.0`; upstream's *dataclass default* is `1.0` — copying the default was a
   real numerical bug, fixed by `from_checkpoint_config`) and `spatial_padding_mode` (trained with
   `zeros`, hardcoded `reflect`, caused *"cumulative temporal divergence… the keyframe
   hold-cut-decay regression"*). **We inherited the corrected VALUES but not the MECHANISM** —
   `DiT.swift:154` and `VideoVAE.swift:239` are literals. A checkpoint revision breaks us silently
   and would not break the oracle.
2. **`cross_attention_adaln` is `false` on disk and overridden to `true` in code** — a silent
   override of shipped metadata, inherited.
3. **Point-coord positions `(B,T,axes)` vs upstream's intervals `(B,axes,T,2)`** — the oracle's
   most-propagated divergence, and we took it wholesale (`ReferenceConditioning.swift:31`).
   "Mathematically equivalent for non-degenerate intervals" — unverified by us.
4. **Keyframe positions deliberately omit an upstream causal fix** (tracking Lightricks PR #192).
   Unexposed today; a trap waiting when keyframe is ported.
5. ⚠️ **Our connector norm violates our own rule.** `Connector.swift:270-274` uses a composed
   mean/rsqrt `rms0` where the oracle uses `mx.fast.rms_norm(weight=None)` — precisely the pattern
   `../CLAUDE.md` warns costs ~1e-5/layer, drifting 16 blocks deep. It runs fp32 and the gate
   passes, so severity is low, but it is a self-inflicted deviation.

✅ **One place we are better than the oracle:** its tiled decode is *not* bit-equivalent to untiled
(causal `Conv3dBlock` replicates each tile's first frame; the trapezoidal blend *"smooths the seam
visually but does not reconstruct the exact signal"*). Our halo+crop is gate-measured **bit-exact
at halo 16**, ~74 dB at halo 5. Our LoRA design (low-rank add at `dense()`, survives quant,
hot-swappable) is likewise arguably better than the oracle's fuse-into-weights.

---

## 6. Ranked work order

| # | Item | Why | Effort |
|---|---|---|---|
| 1 | ✅ **DONE 2026-08-04 — `FrameCodec` on-device repack.** 121f @704×512 encode **3.5 s → 0.2 s (~15×)**, byte-identical (`--frame-codec-gate`, 1,441,792 bytes). See below. | biggest pure loss in the port, zero correctness risk, was scoped as S2 step 2 | ~2 h (est. 1–2 d) |
| 2 | ⛔ **`asyncEval` in the denoise loop — ASSESSED AND DROPPED 2026-08-04.** The win is bounded by host graph-build time, which is milliseconds against a **~6.1 s** denoise step (SPEED-PLAN S1 at compact24) — call it ~0.1%, unmeasurable against this box's noise floor. It would also **break the profiler's per-step timing** (the span would close before the GPU finishes), and the oracle's `mx.async_eval` is there for *memory* ("force computation"), which our blocking `eval` already provides. Not a win; not doing it. | — |
| 3 | **CFG + negative prompt** | unlocks six pipelines + the biggest quality lever; batch the passes (§2) | 1–2 w |
| 4 | **Configurable steps + `ltx2_schedule`** | prerequisite for #3 (CFG needs 15–30 steps, not 8) | 2–3 d |
| 5 | **Read hyperparams from checkpoint config** | closes the silent-breakage class that bit the oracle twice; also the cheap route to the other two upsamplers | 2–4 d |
| 6 | **Attention-mask system** | blocking prerequisite for keyframe, retake, IC strength, and STG | 1 w |
| 7 | **Modality tiling** (`TILING-PLAN.md`) | the DiT attention wall; unlocks 1080p 8s+ which we cannot do at all | 1–2 w |
| 8 | **Streaming decode → muxer** + budget-driven auto-tiling | stops materializing the whole pixel volume; pairs with #1 | 3–5 d |

---

## 7. ✅ Closed: #1 `FrameCodec` on-device repack (2026-08-04)

**The change.** RGB→BGR channel reverse + opaque alpha now run **on-device** (`MLX.take` +
`concatenated`), so the host receives bytes already in the pixel buffer's layout and does one
`memcpy` per row — or one for the whole plane, which is what actually fires, since every LTX width
is a multiple of 32 px ⇒ `width*4` is always 64-byte aligned ⇒ `stride == rowBytes`. The row-loop
branch is kept as a correct fallback but is dead in production.

**Measured** (`--encode-stress 121`, synthetic frames at 704×512, encode timed in isolation,
3 runs each — CPU-bound work, so the spread is tight, unlike this box's GPU thermal band):

| | wall | per frame |
|---|---|---|
| before (host per-pixel loop) | 3.5 / 3.5 / 3.4 s | ~29 ms |
| **after (on-device)** | **0.3 / 0.2 / 0.2 s** | **~1.7 ms** |

**≈15× on the encode stage.** Both encoders re-validated with the two-track audio path that caused
the historic interleave deadlock: hardware+audio 0.3 s, software+audio (113f) 0.7 s.

**Correctness: byte-identity, not eyeball.** `--frame-codec-gate` recomputes the old host-side
per-pixel repack for the same input and asserts the on-device path is byte-identical —
**1,441,792 bytes, zero mismatches**. That is the correct gate for this class of change: the
optimization must be a *pure refactor*, and byte-identity proves it, where an H.264 round-trip
would be lossy and prove less. Random input is deliberate — values far outside [-1,1] exercise the
clip path at both ends.

🔑 **Why this existed at all, and the lesson.** This loop was **not inherited** — the oracle does a
single bulk `memoryview` copy (`video_vae.py:617`). It was a CPU-shaped solution written on a
platform with unified memory and an idle GPU: the natural way to write pixel repacking in C-family
code, which happens to be the wrong way here. **Mirroring the oracle's structure is a correctness
discipline, not a performance one — and where we *diverged* from the oracle without a parity
reason is exactly where the platform-mismatched code accumulated.** §3c lists the remaining
instances (`VideoInput`, `ImageInput`, the audio frame-index builder, blocking `eval`).

⚠️ **Diminishing returns from here.** The remaining ~1.7 ms/frame is 121 device→host copies
(~1.4 MB each) plus the H.264 encode itself. Batching all frames into one GPU op + one copy would
save some of the 121 round trips but holds T·H·W·4 bytes live (174 MB at 704×512×121; **3.8 GB at
4K×113**), which fights the streaming-decode work in #8. Not worth it standalone — revisit only as
part of #8's decode→muxer handoff, where the frames are already being produced incrementally.

---

⚠️ **Sequencing note:** #7 (modality tiling) was my initial recommendation before this audit. It is
still the only path to 1080p 8s+, but #1/#2 are far cheaper wins and #3 unlocks strictly more, so
tiling drops to mid-list. #7 also has an unresolved prerequisite — whether a per-tile attention
mask keeps `MLXFast.scaledDotProductAttention` on the fused kernel or drops it to the materialized
`[B,H,Tq,Tk]` path, which would recreate the very tensor tiling exists to avoid (`TILING-PLAN.md`
open questions). That probe should run before any tiling code is written; note it also overlaps
#6, since the mask system is what would introduce masks in the first place.
