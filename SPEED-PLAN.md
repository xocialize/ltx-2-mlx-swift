# LTX-2.3 — Speed Plan (Release 2, deferred)

**Status: DEFERRED — do not open before MVP** ([MVP-READINESS.md](MVP-READINESS.md)). Memory was
Release 1's optimization theme and is closed; speed is Release 2's. This doc exists so the lane is
scoped *now*, while the profiling context is fresh, and executed *later* against real-device data.

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

## S1 — Real-device wall-clock baseline  (gate for everything below)

Feeds directly from MVP M1's table. Deliverable: per-tier `seconds-per-output-second` on target
hardware + the stage split (encode-prompt / denoise / decode / mp4) from one profiled run per
tier. Decision rule of thumb: ≤ 20 s per output-second at compact24 = ship as-is, revisit speed
in Release 2 at leisure; ≥ ~35 s = pull S2+S3 into the MVP window.

## S2 — Hardware H.264 + zero-copy frame handoff  (effort S–M · expected: encode ~2–4× + power)

Software x264-style encode is the DEFAULT today (historical artifact of the stall
misdiagnosis; the audio-first fix made hardware safe again — both validated). Two steps:
1. **Flip the default to hardware** (`LTX_ENCODE=hardware` exists; one-line default change +
   re-run `--encode-stress N --audio` both ways). Near-free win: encode time and, more
   importantly for laptops, CPU power draw during the encode tail.
2. **Zero-copy handoff:** replace per-frame GPU→CPU `asArray` + Swift pixel copy with
   `VTCompressionSession` fed by IOSurface-backed pixel buffers (the `h00mankind/MetalVideoEngine`
   pattern, already noted in the skill). Removes the last CPU copy of every frame.
Quality gate: bitstream sanity + SSIMULACRA2 vs software encode on one clip.

## S3 — Step-output caching in the distilled denoise  (effort M · expected: 1.2–1.4× · RISK: quality)

TeaCache-style residual caching: skip/reuse DiT block outputs when consecutive-step inputs are
close. Honest expectations: the classic 1.5–2× wins come from 30–50-step CFG pipelines; LTX-2.3
distilled runs **8 steps, no CFG**, so redundancy between steps is lower — model the ceiling
before building (dump per-step latent deltas from one profiled run; if step-to-step cosine is
already < ~0.95, stop here). If pursued: cache the joint AV blocks, threshold on timestep-embed
distance, and gate on SSIMULACRA2 + the e2e cosine vs uncached (same seed).

## S4 — Denoise-loop kernel work  (effort M–L · expected: 10–25% · profile-driven)

The per-step floor. Candidates, strictly in profile order:
- **SDPA path at long seqLen** (704×512×481f ⇒ ~21k video tokens): confirm the fused
  `MLXFast.scaledDotProductAttention` path is hit at every shape (no silent fallback), and
  whether fp32 softmax accumulation is being paid where bf16 would gate-pass (the Wan profiling
  program found fp32-SDPA to be THE wall — same check here).
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
