# LTX-2.3 — Speed Plan (Release 2)

**Status: OPENED EARLY (operator decision 2026-07-04)** — the cheap, gate-safe subset (S2 step 1,
S3 ceiling probe, S6 harness) runs in parallel with MVP M1; the expensive items (S2 step 2, S3
build, S4) still wait for S1's target-hardware verdict. Memory was Release 1's optimization theme
and is closed; speed is Release 2's. The S1 baseline rides the M1 session:
**[LTX_TESTING/M1-TARGET-HARDWARE-PLAN.md](../LTX_TESTING/M1-TARGET-HARDWARE-PLAN.md)** is the
runnable plan (one 24 GB session closes M1 + S1 together).

**Opening trigger:** MVP-READINESS **M1** produces the first honest wall-clock numbers on target
hardware (24/32 GB laptops). S1 formalizes that baseline; S2–S5 are ranked by expected
gain-per-risk and only worth their cost if S1 says the UX needs them. Current desktop reference
(M5 Max, 128 GB): compact24 121f ≈ **64 s** · balanced32 161f ≈ 115 s · standard64 161f two-stage
≈ 188 s · max128 481f i2v ≈ 960 s.

**Standing method:** profile first (`MLX_PROFILE=1`, the shared `MLXProfiling` instrument — spans
already cover every stage and denoise step), change one lever, re-gate parity (`--dit-full-gate`,
`--e2e-gate`, `--vae-chunk-gate`), re-measure the tier table. No lever ships without a quality
gate. Known split today: denoise dominates; decode is chunk-bound; encode is seconds.

---

## S1 — Real-device wall-clock baseline  ✅ CLOSED (2026-07-05, target hardware measured)

**Verdict: 16.1 s per output-second at compact24 ⇒ ≤ 20 ⇒ SHIP AS-IS; remaining speed work is
at-leisure Release 2, not MVP-blocking.** Measured on the actual target (MBP **M5 Pro** 24 GB,
macOS 27.0, weights on TB5 external — session bundle
`/Volumes/Satechi/Testing/ltx-portable/results-MacBook-Pro-20260705-133428/`):

| stage (compact24 clamp 512×288×121f int4, warm) | seconds | share |
|---|---|---|
| text-encode (Gemma 2.4 + connector 1.9) | 4.3 | 5% |
| denoise (8 steps, ~5.9–6.3 s/step, flat) | **49.0** | **60%** |
| vae-decode (4 chunks, 4.9–7.3 s each) | **25.7** | **32%** |
| encode-mp4 (hardware, S2 default in production) | 0.9 | 1% |
| **total run (warm)** | **81.1** | |

(Beware the profiler *group* summary for decode: 45 nested regions double-count parent+children —
read the `vae-decode/video` parent span, 25.7 s, not the 74.6 s group sum.)

Baseline facts that re-rank the lanes below:
- **First run 110.2 s vs warm 85→81 s** (R2–R4): shape-specialization delta ≈ 25 s lands in the
  first generation; repeats got FASTER — **no sustained-load throttling at the compact24 envelope
  on the M5 Pro** (the desktop's ~2× degradation was at the 4× bigger 704×512 shape). Batch UX at
  compact24 is fine as-is.
- **Denoise is still the wall (60%) but decode is now a first-class target (32%)** — a decode
  pass (chunk sizing, decoder kernels, `MLX_PROFILE_DEEP=vae`) is worth ~⅓ of the remaining
  budget and was not in the original S2–S6 ranking. Logged as the first Release-2-at-leisure
  candidate alongside S4's elementwise chains.
- Target-hardware `[STEP-DELTA]` confirms the S3 kill on-device (x0-cos 0.61–0.85 for steps 1–5).

## S2 — Hardware H.264 + zero-copy frame handoff  (effort S–M · expected: encode ~2–4× + power)

Software x264-style encode WAS the default (historical artifact of the stall misdiagnosis; the
audio-first fix made hardware safe again — both validated). Two steps:
1. ✅ **DONE (2026-07-04): default flipped to hardware** (`encodeMP4(software: false)`;
   `LTX_ENCODE=software` is the opt-out). Re-gated: `--encode-stress 113 --audio` hardware
   **PASS 0.4 s** / software **PASS 0.7 s**. Remaining nicety: an SSIMULACRA2 A/B vs software on
   one real clip — ride it along the next real generation (bitstreams differ in rate control;
   hardware leg came out smaller on the synthetic clip, which is expected).
2. **Zero-copy handoff:** replace per-frame GPU→CPU `asArray` + Swift pixel copy with
   `VTCompressionSession` fed by IOSurface-backed pixel buffers (the `h00mankind/MetalVideoEngine`
   pattern, already noted in the skill). Removes the last CPU copy of every frame.
Quality gate: bitstream sanity + SSIMULACRA2 vs software encode on one clip.

## S3 — Step-output caching in the distilled denoise  ❌ KILLED (2026-07-04, measured ceiling ≈ 0)

TeaCache-style residual caching: skip/reuse DiT block outputs when consecutive-step inputs are
close. The plan's kill rule was "model the ceiling before building; step-to-step cosine < ~0.95 ⇒
stop." **The probe (`LTX_STEP_DELTA=1`, DenoiseLoop.swift — ships in both denoise loops, works in
any run) says stop, decisively.** Measured int4 704×512×121f t2v seed 42, deterministic across two
runs:

| step | σ | video in-cos (DiT input) | video x0-cos (prediction) | audio x0-cos |
|---|---|---|---|---|
| 1 | 0.994 | **1.0000** | **0.3453** | 0.9134 |
| 2 | 0.988 | 1.0000 | 0.3773 | 0.9793 |
| 3 | 0.981 | 1.0000 | 0.7492 | 0.9681 |
| 4 | 0.975 | 1.0000 | 0.8619 | 0.9751 |
| 5 | 0.909 | 0.9978 | 0.9024 | 0.9287 |
| 6 | 0.725 | 0.9701 | 0.9716 | 0.9781 |
| 7 | 0.422 | 0.8603 | **0.9908** | 0.9921 |

Two kills in one table: (a) the x0 predictions — the thing a cache would reuse — are far below
0.95 for 5 of 7 transitions; each distilled step does real, distinct work. (b) Worse, the DiT
*inputs* are near-identical early (in-cos 1.0000) exactly where the outputs differ most — an
input-distance trigger (the TeaCache mechanism) would fire precisely where reuse is most wrong.
Only steps 6→7 approach reusability, worth < 1.1× at best. Do not build. The lever stays useful
as a probe if a future non-distilled/CFG path (more steps, redundant tails) ever lands. The M1
session still pastes its `[STEP-DELTA]` lines (free confirmation the property holds on target
hardware — expect identical numbers; it's seed-deterministic model behavior, not hardware).

## S4 — Denoise-loop kernel work  (effort M–L · expected: 10–25% · profile-driven)

The per-step floor. Candidates, strictly in profile order:
- ✅ **SDPA path CHECKED (2026-07-04): fused at every shape, NOT the wall at ≤161f.**
  `RunLTX2 --sdpa-probe` (fused entry vs manual compose, B=1 H=32 D=128 bf16, M5 Max):
  N=2112 → 2.5 ms vs 5.6 ms (2.2×) · N=5632 → 13.1 ms vs 38.5 ms (3.0×) · N=21120 → 197 ms vs
  2276 ms (11.6×). No silent fallback anywhere. Sizing: at 704×512×121f (N=5632), 48 layers
  × 13.1 ms ≈ 0.6 s of a ~10 s int8 step — attention is <10% of the step; the wall is the
  dense/quantized matmuls + elementwise chains. At 481f (N=21120) SDPA grows to ≈ 9.4 s/step-class
  — it matters for max128 long clips, but the fused path is already the fast one.
- **Remaining unfused elementwise chains** in the hand-rolled blocks (the fused-norms sweep took
  the big ones; a fresh `MLX_PROFILE_DEEP=dit` pass will show stragglers — AdaLN modulate,
  gate-mul-add chains are `compile`/fusion candidates).
- **Per-token timestep path (i2v)**: measured heavier than t2v (+2.3 GB act; also slower) —
  check for a scalar fast-path when all generated tokens share sigma.
Every change re-gates `--dit-full-gate` (cosine ≥ 0.9999) before e2e.

## S5 — Load-time & perceived latency  (effort S · expected: UX, not throughput)

- Cold load on internal SSD is already prewarm-protected; measure it on target hardware (M4
  records time-to-first-video). If the DiT mmap re-fault after low-tier evict shows up in traces,
  consider keeping the DiT resident on 32 GB when the request stream is interactive.
- **Progress fidelity:** per-step denoise progress already exists via profiler spans — surface
  step *k/8* in the GUI (perceived speed is half of consumer speed).
- OPTIONAL preview: decode a low-res/first-chunk preview at ~step 6 for early feedback. Real
  cost (an extra partial decode) — only if user testing asks for it.

## S6 — Quant ladder for speed (NOT memory)  (effort S to evaluate · expected: unknown, measure)

int4 exists for memory; on M-series, int8/int4 `quantizedMatmul` can also be *faster* than bf16
at these matrix shapes (bandwidth-bound). One afternoon: time 8 steps at fixed shape across
bf16/int8/int4 on target hardware. If int8 is both smaller AND faster at equal-visual quality
(it is quality-transparent per the tier work, unlike int4's documented sample divergence),
consider making int8 the standard64 default *and* the balanced32 recommendation.
**Harness SHIPPED (2026-07-04): `RunLTX2 --speed-bench [bf16|int8|int4] [w] [h] [frames]`**
(default 704×512×121, one-stage; run 1 compiles kernels, run 2 is the measured leg; add
`MLX_PROFILE=1` for the per-step split). NB the 24 GB machine only fits int4, so the cross-quant
comparison lives on the desktop; what transfers to target hardware is the ratio, not the absolute
seconds.

**MEASURED (2026-07-04, M5 Max/128 GB, two rounds: bf16→int8→int4 back-to-back, then
int4→int8→bf16 with 5-min cooldowns). Verdict: int8 is ~1.3× faster than int4 per step at
704×512×121f — and the dominant speed phenomenon is sustained-load degradation, not quant.**

| leg (cooled run1, warm shader cache) | full generate | steps (8) | peak |
|---|---|---|---|
| int8 | **95.3 s / 98.6 s** (two rounds — reproducible) | 8.1 → 10.9 s | 36.8 GB |
| int4 | **140.4 s** | 11.3 → 17.8 s | 27.5 GB |
| bf16 | ~~_no datum_ — bare-gate watchdog, twice_~~ → **168.5 s** (2026-08-03, weights on PCI-E SSD; **uncooled**, 3rd bf16 run back-to-back — treat as an upper bound vs the cooled rows) | — | 66.2 GB |

- **int8 > int4 for speed, consistently** (per-step median ≈ 9.9 s vs ≈ 12.9 s, both rounds
  agree). Combined with int8 being quality-transparent (vs int4's documented sample divergence):
  **int8 confirmed as the standard64 default recommendation, and worth preferring anywhere the
  envelope fits it.** (24 GB compact24 stays int4 — int8 doesn't fit that budget.)
- **Sustained-load degradation dominates everything else:** in EVERY leg, the immediate second
  run degraded ~1.7–2.4× (int8 run2 226–233 s, steps ballooning to 25–44 s; int4 run2 188–245 s)
  with phys flat (no paging); a 5-min cooldown fully restored run1 speeds. Consequences:
  (a) single-clip interactive UX ≈ the run1 numbers (int8 ≈ 19 s per output-second at this
  BIG shape; the compact24 envelope is ~2.7× smaller); (b) batch/repeated generation is a
  DIFFERENT, ~2× worse regime; (c) every wall-clock claim in this doc must state thermal
  condition; (d) the M1 plan's 3×-drift leg is the highest-information speed measurement on the
  laptop — the desktop already degrades this much on a Studio-class chassis.
- **~~Open bug (harness, reproducible 2×)~~ → CLOSED 2026-08-03 as ISSUES.md I9.** The bare
  `--speed-bench bf16` leg died at its first nv=5632 forward with the Metal watchdog
  (`kIOGPUCommandBufferCallbackErrorTimeout`) despite file prewarm + nv=1 DiT warmup, while the
  app/wrapper path ran bf16 at this shape fine. **The diagnosis recorded here — "shape-
  specialization compile + heaviest compute in one go" — was wrong: it is I/O, not compute.**
  MLX safetensors arrays are lazy, so weight bytes are pulled *inside* the first generation's
  command buffers; the weight tree sits on the **USB** `DEV_ARCHIVE` volume (~250–475 MB/s), and
  bf16 is the only quant whose working set (66.2 GB at this shape) stops the OS from serving those
  reads out of page cache. Restaging the tree on a PCI-E SSD makes the identical run pass 3/3 —
  that is where the bf16 row above comes from. **The prewarm + nv=1 warmup were never the gap**,
  and a target-shape warmup would not have fixed it. Full A/B and mechanism: ISSUES.md I9.
  ⚠️ Corollary for every number in this section: **the int8/int4 rows above were also USB-bound**,
  so their run-1 legs include weight page-in. State the storage volume on any re-measurement.

**⟲ RE-MEASURED 2026-08-03 (late) — first protocol-grade ladder, from the Satechi store, via
`--bench-e2e` (receipt `probes/bench_e2e_ladder-pruna-121f_20260804-012008.{md,json}`).** Four arms
(bf16 · bf16+pruna · q8 · q4) × 2 ABBA blocks × (1 excluded warmup + 2 measured), 60 s cooldowns,
704×512×121f one-stage, 32 generations, zero aborts. ⚠️ **This is the SUSTAINED regime** (3 gens
per visit, 60 s cooldowns — between July's 5-min-cooled run-1 legs and its back-to-back run-2s),
which the numbers reflect; the within-block heat ramp is visible in every arm (run 2 ≈ +10–15%
over run 1; warmups — the coolest slot — often ran FASTER than the measured runs that followed).

| arm | median (min–max) | peak phys | vs July's cooled leg |
|---|---|---|---|
| bf16 | **173.3 s** (148.1–185.4) | 67.76 GB | first protocol datum (chip's 168.5 s single-shot agrees) |
| bf16+pruna | 160.3 s (148.7–168.5) | **60.63 GB** | — (see S8 answer below) |
| q8 | 160.2 s (147.4–168.2) | 50.93 GB | July cooled 95.3/98.6 s · July sustained 226–233 s |
| q4 | 149.5 s (139.6–158.3) | 41.88 GB | July cooled 140.4 s · July sustained 188–245 s |

- 🔑 **The July "int8 is 1.3× faster than int4" ordering INVERTS under sustained load** — q4's
  median beat q8's by 10.7 s here (⚠️ NOISE by the session's own spread rule, so direction only —
  but July's own run-2 data agrees: int8 226–233 vs int4 188–245). Reading: int8's bandwidth
  advantage is a **cooled-regime** property; under thermal cap the smaller checkpoint at least
  ties. **The standard64 int8 recommendation stands for single-clip interactive UX and should NOT
  be extrapolated to batch/queued generation without a cooled-regime A/B.**
- 🚨 **Doctrine refinement — "q8 reproduces bf16" does NOT hold e2e at this geometry.** Same seed,
  same binding, both arms bit-deterministic (see below), yet q8-vs-bf16 final pixels land at
  **cos 0.878** (q4: 0.832; pruna-on-same-trajectory: 0.9989 — the sanity control). Per-forward
  q8 ≈ bf16 (~0.9999) remains true; over an 8-step × 121f trajectory the divergence compounds into
  a **different (equally valid) sample**. q8 = *closest* tier, not *reproducing* tier, at long
  geometries. (The 9f behavior is unmeasured — geometry may matter.)
- ✅ **I8 narrowed:** q8 AND q4 were bit-identical across all 4 measured runs *including a fresh
  pipeline load in block 1* (`cos(first)=1.000000` throughout) — so the documented q8
  nondeterminism is **cross-process only**, not cross-load-within-process, at e2e granularity.
- ⚠️ **Open question — quant-arm peaks are +14 GB vs July** (q8 50.9 vs 36.8; q4 41.9 vs 27.5)
  while bf16 matches its own recent single-shot (+1.6). Leading hypothesis: July's peaks were
  cooled single-generation samples, vs this session's peak-across-3-generations with no
  `clearCache` between measured runs (pool growth); testable with `--runs 1 --blocks 4`. Not
  explained, not yet investigated — do not quote either peak set as "the" footprint without regime.

---

## S7 — Alternative community checkpoints (Sulphur-class)  (effort S to evaluate · expected: unknown, measure · **DEFERRED until MVP is finalized**)

**INVESTIGATIVE — opened 2026-07-05, deliberately parked as post-MVP follow-up (operator decision).**

Background: a previous review adapted **Sulphur 2** (SulphurAI's fine-tune of LTX 2.3) and found
it essentially swappable with the official weights — same architecture, same config, only the
weights differ, so the existing MLX conversion path applied unchanged. Sulphur is reportedly
faster at generation; we held off, but it establishes the class: architecture-identical
full-checkpoint fine-tunes/merges of LTX 2.3 that drop into our pipeline. Known candidates as of
2026-07 (best ongoing index: [awesome-ltx2](https://github.com/wildminder/awesome-ltx2)):

| candidate | what it is | speed relevance | caveats |
|---|---|---|---|
| [Sulphur 2](https://civitai.com/models/2601098/sulphur-2-base) (SulphurAI) | LTX 2.3 fine-tune; dev + distill LoRA/fused | the speed story is `sulphur_distil` on the dev base, NOT the dev weights alone — if the earlier review only ran dev, the distill combo is the un-run benchmark | uncensored-community provenance; content-posture diligence |
| [DaSiWa-LTX2.3](https://huggingface.co/darksidewalker/DaSiWa-LTX2.3) (darksidewalker) | LoRA-fused checkpoint family, **4-step distilled** + 20–30-step standard variants | most directly speed-oriented (official distill is 8-step) | custom license terms beyond base LTX — attribution, branding, and "Generation-as-a-Service" restrictions; READ before production |
| [LTX2.3-10Eros](https://huggingface.co/TenStrip/LTX2.3-10Eros) (TenStrip) | I2V-optimized layer-scaled merge (Sulphur as merge base) | quality/adherence, not speed | same NSFW provenance as Sulphur |
| [Singularity OmniCine](https://civitai.com/models/2610733/singularity-ltx-23omnicinev1) | ~100k-step I2V / FLF / ref-to-video optimization | quality, not throughput | evaluate only if I2V quality becomes the lever |
| linoyts fused-union-control | distilled + Canny/Depth/Pose fused | n/a | only if control signals reach the roadmap |

Evaluation notes when this opens:
- "Faster" in this class means **fewer steps via distillation**, not faster per-step math — the
  honest matrix is official-distilled (8-step) vs Sulphur-distill vs DaSiWa 4-step at matched
  visual quality on target hardware, using the S6 `--speed-bench` harness + existing parity gates.
- GGUF/NVFP4 packagings floating around Civitai are CUDA-ecosystem artifacts — irrelevant here;
  we convert from the safetensors checkpoints exactly as we did for Sulphur.
- Fine-tunes shift the output distribution even on benign prompts — a content-posture pass is an
  evaluation item in its own right for the Sulphur-lineage models (ties into the M5 license work).
- **Possible product follow-on:** if keeping MLX conversions compliant stays as simple as the
  Sulphur experience suggests (same architecture ⇒ same conversion recipe ⇒ same QuantFootprint
  process), a **weight-selection** capability (user picks official vs alternative checkpoint per
  the existing tier/quant picker pattern) is on the table — scope only after the evaluation says
  a candidate is worth carrying.

## S8 — VAE decode pass  (effort M · expected: up to ~⅓ of remaining budget · promoted from S1's finding)

S1 re-ranked the lanes: at the compact24 envelope on target hardware, **decode is 32% of the run**
(25.7 s of 81.1 — read the `vae-decode/video` parent span, not the double-counting group sum),
second only to denoise. Not in the original S2–S6 ranking; formalized here. Candidates, profile
order (`MLX_PROFILE_DEEP=vae`):
- **PrunaVAED lean decoder — PORTED + GATED 2026-08-03; the largest lever in this lane.** A
  pruned, distillation-finetuned drop-in for the video decoder (encoder + latent format unchanged;
  latent stats verified bit-identical to stock). Measured on THIS desktop (M5 Max 128 GB, full
  unchunked decode, fp32, median of 3):

  | tier | stock | pruna | speedup | activation Δ (peak−floor) |
  |---|---|---|---|---|
  | 512×288×121f | 3.468 s | 1.753 s | **1.98×** | 4.79 → 1.92 GB |
  | 704×512×121f | 9.548 s | 4.565 s | **2.09×** | 19.47 → 12.64 GB |

  Beats the upstream H100 claim (1.68–1.7×), and holds on the **chunked** production path:
  704×512×113f chunk 5 halo 4 goes **17.1 s → 7.9 s (2.16×)**, peak Δ 30.65 → 22.37 GB, seam gate
  green (60.8 dB vs stock 60.4).

  ⚠️ **The e2e win is NOT established — do not quote one.** A desktop `--speed-bench` A/B at
  512×288×121f could not resolve it: run-to-run totals varied ±15 s (stock 45.2 / 39.7 s, pruna
  79.1 / 55.4 s), far larger than the ~1.7 s decode saving, so the totals measured machine noise.
  **Decode's share is the whole story and it is tier-dependent:** unchunked on this 128 GB desktop
  decode is only ~8–9% of a ~40 s run (≈4% e2e for a 2× decoder), whereas S1 measured **32% on the
  M5 Pro compact24 target** precisely because it is chunked there and pays halo recompute. So the
  payoff should concentrate on the low, memory-constrained tiers — that is the hypothesis, and
  BRIDGE-LTX-014 is the live measurement that settles it.

  Two measurement traps, both hit here: prewarm the pruna file (unwarmed it cold-faults mid-decode,
  7092 → 4536 ms), and **never compare decode timings across variants under `MLX_PROFILE=1`** —
  profiling forces a per-up-block `eval` that breaks fusion and inflates decode (5530 ms profiled
  vs 3468 ms unprofiled, same stock decode), compressing the gap to ~1.2×. Profile for the split,
  measure the ratio unprofiled. Select via `LTX_VAE_DECODER=pruna` (no app change) or
  `LTX2Configuration.vaeDecoderPath`; weights need `scripts/convert_pruna_vae_decoder.py` first
  (diffusers → ltx-core dialect) and are not auto-materializable yet. Quality is
  near-identical-not-bit-exact by construction (upstream: PSNR 39.23 dB / SSIM 0.981 / LPIPS
  0.0087 at 720p), so acceptance is perceptual — **the perceptual A/B on a real clip is still
  owed**; only numerical parity vs the reference implementation is done so far.
  Gate: `--vae-decode-pruna-gate`. Bench: `--vae-decode-bench`.
- **Chunk sizing on target:** compact24 uses chunk 4 (a MEMORY choice); with 15 GB peak vs 16.8
  budget there is headroom to try chunk 6/8 — fewer halo recomputes = fewer decoded-twice frames.
  Free experiment via `LTX_VAE_CHUNK`; keep the seam gate green (`--vae-chunk-gate`).
- **Decode-scoped cacheLimit interplay:** the decode runs under `cacheLimit=0` (T3b lever) which
  forces buffer reuse at some allocation cost — measure a small nonzero cap on target now that the
  envelope has headroom.
- **Decoder kernel stragglers:** hand-rolled conv/norm chains in the up-blocks (the fused-norms
  sweep covered pixelNorm; a deep profile will show what else repeats 45×).
Quality gates: `--vae-decode-gate` + `--vae-chunk-gate` byte-level seam checks.

**⟲ THE DESKTOP e2e QUESTION ANSWERED 2026-08-03 (the "±15 s swamps it" gap this section named —
receipt `probes/bench_e2e_ladder-pruna-121f_20260804-012008`, protocol details under S6's
re-measurement).** Stock vs Pruna decoder, same bf16 DiT, 704×512×121f one-stage unchunked,
2 ABBA blocks × 2 measured runs:

- **Time: Δmedian −13.0 s (173.3 → 160.3 s) — NOISE** against the session's 37.3 s spread. The
  direction is right and the magnitude is plausibly real (the component A/B predicts ~5 s of it),
  but under a sustained-load session it is **not separable from thermal drift** — which is the
  receipt-grade version of what this section suspected: **on desktop, the e2e time win does not
  clear the noise floor.** The time payoff remains a LOW-tier/chunked hypothesis (decode = 32% of
  compact24), and its validation belongs to the target-hardware app campaign, not this box.
- **Memory: peak phys 67.76 → 60.63 GB = −7.1 GB e2e, consistent in both blocks** (61.2/61.3 vs
  68.6/68.7). Memory is far less drift-sensitive than time, so **this is the solid desktop
  deliverable of the lever** — it matches the decode-activation drop measured at the component
  level and is the number the footprint re-declaration item above should start from.
- Sanity control: pruna-vs-stock final pixels at **cos 0.9989** (same DiT trajectory, decode-only
  delta) against q8's 0.878 / q4's 0.832 sample divergence — the decoder swap changes rendering,
  not the sample, exactly as designed. ⟲ **The perceptual A/B ran the same night and PASSED**
  (Xcode agent, bridge mailbox 2026-08-03): bf16-DiT-isolated decoder A/B at compact24-chunked
  512×288×121f, pruna engagement MD5-confirmed, frames perceptually identical, 8×-amplified diff
  near-nil, **SSIM 0.976 / PSNR 34.6 dB avg** (conservative — over H.264 frames), audio track
  bit-identical. Two perceptual caveats stay open before promotion: **playback-motion eyeball**
  (shimmer isn't visible in stills) and a **face-containing prompt**. Clips:
  `/tmp/ltx-ab/{stock,pruna}.mp4`.

**Pruna lever status after tonight, in one place:** ported + numerically gated ✅ · perceptual
A/B ✅ (2 caveats above) · desktop e2e memory **−7.1 GB** receipted ✅ · desktop e2e time = NOISE
(as this section predicted) · **remaining before any default change: the 24 GB target's wall-clock
+ peak (bridge asks #2/#3, queued for the next M5 Pro session), the two perceptual caveats, and
the footprint re-declaration.** Stock stays the default meanwhile.

## S9 — i2v wall-clock on target  (effort S to re-measure, then rides S4 · **data gap**)

The M1 session's i2v+adapter run (R5) recorded **≈525 s true** for a ~5 s clip at the compact24
clamp — ~6× the t2v path and far over the 20 s/os line — but the figure is CONTAMINATED (a 52-min
silent adapter download overlapped; "true" is an estimate) and entangled with the LTX-012 memory
exception. First step is a clean re-measure (adapter pre-cached, profiled) to split
per-token-timestep cost from LoRA/apply/download noise. If the per-token path is confirmed as the
multiplier, S4's "scalar fast-path when all generated tokens share sigma" is the fix and doubles
as the LTX-012 activation lever. Until measured, treat "i2v is ~6× slower" as UNVERIFIED.

## Out of scope (Release 3+ / product lane)

- On-device LoRA training, tiering UX beyond the picker, product shell — the Phosphene-blueprint
  lane, not a speed item.
- A smaller/distilled-further checkpoint (would re-open the 16 GB tier) — upstream-dependent;
  watch Lightricks releases (community-checkpoint candidates are tracked separately in S7).
- Multi-clip batching/queueing — product feature with speed *implications*, scope when asked for.

## Exit

Release 2 closes when: ~~S1 baseline recorded~~ ✅ (16.1 s/os, 2026-07-05) → the chosen subset of
the open lanes (S2 step 2, S4 elementwise/i2v, S5, S8, S9; S3 killed, S6 answered, S7 parked)
lands with parity gates green → the MVP-READINESS M1 table is re-measured on the same hardware
showing the improvement, and the README tier table gains a wall-clock column (**owed now that S1
data exists** — an at-leisure doc task, don't wait for the speed work).
