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
| bf16 | _no datum_ — bare-gate watchdog, twice (see below) | — | — |

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
- **Open bug (harness, reproducible 2×):** the bare `--speed-bench bf16` leg dies at its first
  nv=5632 forward with the Metal watchdog (`kIOGPUCommandBufferCallbackErrorTimeout`) despite
  file prewarm + nv=1 DiT warmup; the app/wrapper path runs bf16 at this shape fine. Gate gap
  (first big-shape bf16 buffer = shape-specialization compile + heaviest compute in one go), not
  a pipeline regression. Fix when a bf16 datum is actually needed: target-shape 1-step warmup
  inside the gate; bf16 is the Unconstrained path, so it gates nothing tier-shaped today.

---

## Out of scope (Release 3+ / product lane)

- On-device LoRA training, tiering UX beyond the picker, product shell — the Phosphene-blueprint
  lane, not a speed item.
- A smaller/distilled-further checkpoint (would re-open the 16 GB tier) — upstream-dependent;
  watch Lightricks releases.
- Multi-clip batching/queueing — product feature with speed *implications*, scope when asked for.

## Exit

Release 2 closes when: S1 baseline recorded → the chosen subset of S2–S6 lands with parity gates
green → the MVP-READINESS M1 table is re-measured on the same hardware showing the improvement,
and the README tier table gains a wall-clock column.
