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
| Upsamplers | **3** (x2, x1.5 rational, temporal x2) | ✅ **3** (2026-08-05: sidecar-config-driven `Upsampler.load`, weight-key fallback; `--upsampler-variants-gate` cosine 1.000000 both) | row CLOSED — was a download + a config read, exactly as the correction predicted; **two-stage wiring landed 2026-08-05** (`--two-stage-variants-gate` 11/11: variant-aware stage-1 geometry, pre-flight rejects, x2 bit-identical to pre-wiring — x1.5/temporal two-stage is OUR composition, no oracle/upstream consumer) |
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

### 🚨 …but the premise check kills the 1–2 week estimate: **CFG needs a checkpoint we do not have**

Verified 2026-08-04, before writing any code:

- **CFG is a *dev*-model feature.** The oracle's `--distilled` mode *"skips CFG entirely"*, and
  `ti2vid_two_stages.py:92-99` is explicit: *"Stage 1: Dev model + CFG guidance… Requires
  `dev_transformer` **and** `distilled_lora`."* The distilled path exists precisely because the
  CFG behaviour is already baked into the weights — running CFG on top of it would double-apply
  guidance.
- **We hold the distilled variant only *locally*.** `/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx{,-q8,-q4}`
  each contain exactly one transformer: `transformer-distilled.safetensors`. The shipped
  `config.json` declares `"variants": {"distilled": …, "dev": …}`.
- 🚨 **CORRECTION (same session, before acting on it): the dev weights DO exist, and they are a
  DOWNLOAD, not a conversion project.** I first concluded "no MLX conversion of dev exists" — that
  was **wrong**. `dgrauet/ltx-2.3-mlx-q8` contains **`transformer-dev.safetensors`**, plus
  `ltx-2.3-22b-distilled-lora-384.safetensors` (+ a `-1.1`), `spatial_upscaler_x1_5_v1_0.safetensors`
  and `temporal_upscaler_x2_v1_0.safetensors`. The repo is **~93 GB; we pulled ~47 GB of it.**
  🔑 **The error: I listed the local store, searched HF for a repo whose *name* contained "dev",
  found none, and concluded the artifact didn't exist — but dev is a FILE inside the repos we
  already use, not a separate repo. A partial download is indistinguishable from an unavailable
  artifact when you only look at the local filesystem.** This is PORT-QUEUE's availability lesson
  inverted: that one says a ✅ licence check is not a downloadability check; this one says a local
  `ls` is not an availability check. **Probe the remote file list, not the local directory.**
- **So the real work order is much cheaper than stated above:** `hf` download the dev transformer
  (+~20 GB at q8) → port guidance. No conversion, no re-hosting job. ⚠️ The weight-durability rule
  still applies at ship time — a shipped `WeightSource` must point at a namespace we control, so
  mirroring would be part of *productising* it, not of experimenting with it.
- ✅ **Same correction unblocks the upsamplers cheaply:** the x1.5 spatial and x2 temporal
  upsamplers the audit lists as "oracle can load 3, we have 1" are **also just files we never
  downloaded**, not a porting gap in the weights.
- 🚨 **And the runtime cost probably disqualifies it as a product tier anyway.** Dev + CFG is
  **30 steps × 2 forwards = 60** against distilled's **8** — roughly **7.5× the compute**. Against
  our own measured 121f @704×512 distilled median of **173 s** (`probes/bench_e2e_ladder-pruna-121f_…`),
  that extrapolates to **~20+ minutes per clip**. For a macOS product that ships the fast path
  first, that is not a v1 feature.

🟡 **Verdict after the correction: CFG is DEFERRED on COST, not on availability.** The weights are
one download away. What still argues against it, and is unaffected by the correction:

- **~7.5× the compute** (60 forwards vs 8) ⇒ ~20+ min/clip extrapolated from our measured 173 s at
  121f/704×512. For a product shipping the fast path first, that is not a v1 tier.
- **+~20 GB resident per quant** for a second transformer, against tiers already tight on memory.
- The distilled path we ship is *the* product path; CFG buys a quality tier nobody has asked for
  yet.

**Revisit when** a quality tier is explicitly wanted and ~20 min/clip is acceptable. ⚠️ **Do not
"try CFG on the distilled checkpoint" as a shortcut** — the distilled weights already have the
guidance behaviour baked in, so applying CFG on top double-applies it; if that experiment is ever
run it needs a perceptual gate, not a cosine.

✅ **Cheap, unblocked follow-on regardless:** download the x1.5 spatial + x2 temporal upsamplers and
wire the config-driven variant selection (#5) — that is the audit's "oracle can load 3, we have 1"
gap closed for the price of a download plus a config read.

✅ **Consequence: modality tiling (#7) becomes the top capability item**, because it unlocks
1080p 8s+ **on the distilled checkpoint we already hold** — no new weights, no new storage, and its
memory premise is now measured-and-confirmed (`TILING-PLAN.md`, `--sdpa-mask-probe`).

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
| 3 | ⛔ **CFG + negative prompt — DEFERRED 2026-08-04 on a premise check.** Needs the **dev** checkpoint, which we do not have and which exists in no MLX conversion; acquiring it is a Tier-3 convert + re-host + ~38 GB/quant, and dev+CFG runs ~7.5× our distilled compute (~20+ min/clip). See §2. | — |
| 4 | **Configurable steps + `ltx2_schedule`** | ⬇️ demoted with #3 (its main driver was CFG's 15–30 steps). Still worth having on its own — the distilled path has **two hardcoded sigma tables** and no step knob at all, so there is no quality/speed dial anywhere in the port | 2–3 d |
| 5 | **Read hyperparams from checkpoint config** | closes the silent-breakage class that bit the oracle twice; also the cheap route to the other two upsamplers | 2–4 d |
| 6 | **Attention-mask system** | blocking prerequisite for keyframe, retake, IC strength, and STG | 1 w |
| 7 | **Modality tiling** (`TILING-PLAN.md`) | the DiT attention wall; unlocks 1080p 8s+ which we cannot do at all | 1–2 w |
| 8 | 🟡 **Streaming decode → muxer — BUILT 2026-08-05, default OFF (opt-in `LTX_STREAM_MUX=1`), premise measured and found phase-bound.** Full seam shipped: `decodeChunked(sink:)` (one code path with the array form) → `LTX2Pipeline.StreamingSinks` (audio decoded FIRST so the writer's track closes before frame 1) → `MP4StreamWriter` (init/attachAudio/appendSync/finish; `encodeMP4` is now a composition over it) → wrapper streamed lane + `--t2v-spot` receipt gate. **Measured at 704×512×121f: run peak 54.82 (streamed) vs 54.88/55.93 GB (materialized) — NO run-peak win, because denoise sets the peak and the ~1 GB of pixel volume + transpose twin this removes sits in a phase below it** (the modality-tiling lesson, third instance). Wall-clock: NOISE by the sign-flip rule (whichever lane runs second is slower — session ramp). All gates green through the refactor (chunk 60.4 dB, e2e, encode-stress both encoders, tile gates). **Where it should win — future receipt needed before default-on: geometries where ASSEMBLY is the peak phase** (4K-class decode flows: 3.8 GB volume + twin; low-tier evicted-DiT where the floor is ~0.5 GB). Budget-driven auto-tiling remains open. Receipts `probes/20260805_streammux_{ab,ba}.out`. | done (opt-in) |

---

## 8. 🔍 The run peak at shipping geometries is GEMMA's encode, not the DiT (2026-08-04)

Found while measuring modality tiling. Profiler, untiled 704×512×9f, **worst phys by phase**:

| phase | worst phys |
|---|---|
| **encode (Gemma + connector)** | **52.5 GB** ← the run peak |
| denoise | 41.8 GB |
| vae-decode | 42.0 GB |

Only past ~65f does denoise take the peak. **Every DiT-side memory lever we have — block streaming,
modality tiling — is therefore incapable of moving the run peak at the geometries we actually ship.**

### Where the 52.5 GB goes

```
DiT.load done (lazy)      phys= 0.10 GB     ← lazy safetensors, nothing faulted
DiT kernel warmup         phys=40.58 GB     ← warmup faults in all 38 GB of bf16 weights
encode/gemma    act=45.6  phys=48.3 GB      ← DiT (38) + Gemma-3-12B-4bit (7.6) CO-RESIDENT
encode/connector act=42.0 cache=10.1 phys=52.5 GB   ← the peak: 42 active + 10.1 GB of MLX CACHE
```

Two independent terms, and neither is the DiT's own working set:

**(a) The DiT is resident throughout text encode — ~38 GB of it — for no reason other than load
order.** `load()` warms the DiT (faulting every page), then `t2v()` encodes the prompt. The oracle
hit exactly this and fixed it by moving the text-encoder lifecycle out of `load()` entirely
(its `1a30f74`; documented in its CLAUDE.md as *the* fix that unblocked production quality).
⚠️ Our situation differs in one way that matters: our DiT is the deliberate **persistent resident
floor** across requests, so "encode before load" only helps the first request; helping later ones
means evict → encode → reload, paying the warmup each time. That is the existing
`dropDiTIfSequential()` pattern applied to a second seam, and it is a **tier decision, not a free
win** — worth it where 10 GB decides admission, not otherwise.

**(b) 10.1 GB of the peak is reclaimable MLX buffer cache**, not live data — and the existing
`LTX_CACHE_LIMIT_GB` lever already reaches it.

### Cache-cap sweep — memory measured, timing REFUSED

704×512×9f, run peak by cap (each an independent process):

| cap | peak phys |
|---|---|
| uncapped | 52.49 / 52.55 GB |
| 8 GB | 52.55 GB (no effect) |
| 4 GB | 49.04 GB |
| **2 GB** | **48.47 / 48.02 GB** |
| 1 GB | 48.14 GB |

**Capping at ≤2 GB reliably removes ~4.0–4.5 GB from the run peak.** Reproducible (2 GB measured
twice, 0.45 GB apart); the knee is between 4 and 8 GB.

🚨 **The wall-clock effect is UNMEASURED and this data cannot answer it.** The identical 2 GB config
came out **17.0 s** in one sweep and **30.8 s** in the next — an **81% swing** on the same binary and
config, after hours of continuous heavy jobs. That is thermal drift, the exact failure the
noise-floor receipt documents (`BENCH.md`: this box swings ±23% at 704×512×9f, worse when saturated).
**Do not read a speed verdict out of the tables above.** Owed: a `--bench-e2e` run (ABBA, cooldowns,
cooled box) with `env.LTX_CACHE_LIMIT_GB` as the arm.

### Owed / next

1. ✅ **RUN 2026-08-05 on the rested box (`probes/bench_e2e_cachecap_20260805-133154`) — the cap is
   FREE at this geometry.** blocks=4/runs=1 ABBA, 60 s cooldowns, every run `nominal→nominal`:
   base median **15.4 s** (15.2–15.5) / **52.85 GB** vs cap2 median **15.5 s** (15.4–15.5) /
   **48.80 GB** → **Δmed +0.0 s ≤ 0.3 s spread = NOISE; −4.05 GB peak; outputs bit-identical**
   (cos 1.000000 / maxAbs 0 — the cap is output-invisible, receipted). Note the protocol datum:
   a rested box + one-run-per-block measured a **0.3 s (2%) spread** where the saturated box
   swung 81% — the noise floor is a property of protocol × thermal state, not of the machine.
   ⚠️ Scope: 9f one-stage bf16 on a 128 GB host. **A default change hinges on 121f** (the pool
   serves per-step denoise transients; a 2 GB cap could thrash where 8 steps × nv=5632 reuse
   buffers) — that arm is next, then the QuantFootprint re-measure per note 3.

   ✅ **121f arm RUN same morning (`probes/bench_e2e_cachecap121_20260805-134617`) — the cap is at
   worst free and plausibly ~5% FASTER, and it saves 10.88 GB:** base median **157.1 s / 66.75 GB**
   vs cap2 **149.4 s / 55.87 GB** → Δmed **−7.7 s > 7.0 s spread = MEASURABLE** (same sign in both
   blocks; treat the speed claim as directional — the margin is one spread-width), outputs again
   **bit-identical**. The cap's benefit *grows* with geometry (−4.05 GB @9f → −10.88 GB @121f)
   because the uncapped pool accumulates per-step transients across the denoise loop.
   ⚠️ Ratchet-detector observation: block floors climbed 2.12 → 3.70 → 5.29 GB across the 121f
   session (flat 0.68–0.99 at 9f) — a mild per-process residency ratchet at big geometry that
   `clearCache` doesn't fully return; resets on process exit; worth a look if CLI sessions ever
   chain many big generations.

   🔑 **RESOLUTION — no default change is needed, because the ENGINE already ships this exact cap.**
   `MLXServeEngine`'s `GPUCacheConfiguration.automatic` resolves to **`min(2 GB, 5% of budget)`**
   (`mlx-engine-swift/Sources/MLXServeCore/GPUCachePolicy.swift:78`) and is applied at engine init
   (ENGINE-NEEDS N5, engine ≥0.21.0). The shipping app path has had the 2 GB cap all along; these
   receipts **validate the engine's default with model-specific evidence** (free at 9f, ≥free at
   121f, output-invisible, −4 to −11 GB).
   🚨 **The real consequence is a measurement-basis correction: the bare CLI lane (RunLTX2
   speed-bench/mem-bench/bench-e2e) is the ONLY uncapped path**, so CLI-measured peaks at big
   geometry OVERSTATE the shipping path by up to ~11 GB — e.g. the ladder's bf16@121f "67.76 GB"
   is a CLI-uncapped figure; the app-path equivalent is ~55.9 GB. **From now on: bench arms that
   claim shipping-path relevance should set `env.LTX_CACHE_LIMIT_GB=2` (and receipts print arm env,
   so the basis is visible); uncapped arms remain valid for comparing against historical
   receipts.** QuantFootprints were measured through the engine path and are unaffected.
2. ✅ **MEASURED 2026-08-05, and CLOSED — the encode stack without the DiT is 13.3 GB, and the
   architecture already does the right thing on both tiers.** `--text-encode-gate` (production-shaped
   sequential Gemma→connector, DiT never loaded) peaks at **13,312 MB phys** (external `footprint`
   sampler, gate PASS at production cosines). So the 48.8 GB (capped) encode-phase peak decomposes
   as ~13.3 GB text stack + ~35 GB resident DiT. **But nothing needs building:**
   - **Sequential/low tiers already encode without the DiT.** `dropDiTIfSequential`'s own docstring:
     it fires *"before the connector loads (encode is the T3b-measured peak: connector int8 quantize
     scratch + a warmed-resident DiT ≈ 26 GB co-resident) AND after the last denoise step"* — T3b
     found and solved this exact co-residency where memory is scarce. Every entry point encodes
     before `ensureDiT()`, so a dropped DiT stays dropped through encode and reloads in seconds.
   - **Resident tiers keep the DiT through encode BY DESIGN** — it is the persistent floor across
     requests on 64–128 GB hosts. The desktop's 48.8 GB is the cost of that choice, not a bug.
   **Gemma-quant note (asked 2026-08-05): quantization is not a lever here and is already pulled.**
   Gemma is `mlx-community/gemma-3-12b-it-4bit` (7.5 GB on disk) in both the wrapper default and the
   CLI; the connector int8-quantizes at load (T3b). The whole text stack is 13.3 GB against an
   encode peak set by the 38 GB DiT — even deleting Gemma entirely would move it by at most a third
   of the co-residency term. Going below 4-bit (~1.5–2 GB at best) would put quantization noise
   directly into the **49-hidden-state conditioning stream** the DiT was trained against — quality
   risk for near-zero structural gain. And a smaller Gemma is off the table: the connector is
   trained against Gemma-3-**12B**'s 3840-dim states specifically.
3. ⚠️ **Any change here invalidates the declared `QuantFootprint`** — those are *measured* values
   (`MLXLTX2Package`), so re-measure before touching the manifest.

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
