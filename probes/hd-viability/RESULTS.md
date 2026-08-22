# HD viability — 5s@720p watermark (2026-08-22, AB-R-0117)

Cheap first data point before committing a slot to the long HD runs (2.3 spent 962 s on 10s@720p and
**51.9 min** on 10s@1080p).

🚨 **NOT an acceptance number.** Run via `LTX_ENVELOPE_OVERRIDE=1`; **no 2.5 profile permits
1280×704** — every tier caps at 704×512. Acceptance means "≤ budget AT A GEOMETRY THE PROFILE
ALLOWS". This exists only to decide whether the cap should MOVE, the path `maxFrames` took to 481.

## Measured — 1280×704×121f (5.04 s), max128, STREAMED

| | |
|---|---|
| **peak** | **43.70 GB** (resident 0.79 + activation 42.91) |
| gate | **STREAM** — N=14206 vs N_min 3035 (4.7×) |
| wall | 187.1 s — indicative only, single arm, unknown thermal state |

## 🔑 The encoder floor is overtaken — peak is RESOLUTION-driven, not frame-driven

AB-R-0116 measured 704×512 streamed **flat at 24.81 GB from 161f to 481f** (encoder-bound). Here, at
2.5× the pixels but **a quarter of the frames**, the peak nearly doubles.

**Fewer total tokens, higher peak.** The peak is set by the per-step working set, which scales with
SPATIAL area — VAE decode is temporally chunked (`vaeChunkFrames` 8), so frames amortise and
resolution does not.

⟲ "Flat across frames" is retired as a general property. It holds *while the encoder floor
dominates*; 720p is past that crossover.

## 🔑 The 2.3 cross-check validates the model to 1.1 GB

| | activation |
|---|---|
| LTX-2.3, 1280×704×**241f** (AB-R-0024) | **41.79 GB** |
| LTX-2.5 streamed, 1280×704×**121f** (this run) | **42.91 GB** |

Two ports, two frame counts, same resolution — **within 1.1 GB**. Confirms activation is
resolution-driven (2× the frames moved ~1 GB) and that 2.3/2.5 share an activation profile, as
AB-R-0002's byte-identical VAE/latent-space finding predicts.

🔑 **So 10s@720p should also cost ~43.7 GB.** Cheap to confirm — and that run is really a
FALSIFICATION TEST of the frame-amortisation model the 1080p projection rests on.

🔑 **Streaming is worth ~37 GB at 720p**: 2.3 spent **80.36 GB** total (act 41.79 + ~38.6 resident
DiT) where 2.5 streamed spends **43.70** for the same picture.

## 🚨 1080p lands on an UNRESOLVED RULE, not on the hardware

| route | 1080p activation |
|---|---|
| linear-in-pixels from this run (×2.32) | **99.5 GB** |
| LTX-2.3 *measured* at 1920×1088×241f | **93.96 GB** |

So 2.5 streamed ≈ **95–100 GB**. Against max128:

| rate | budget | verdict |
|---|---|---|
| 0.7× | 89.6 | ❌ OVER by 5–11 GB |
| **0.85×** | **108.8** | ✅ WITHIN by 9–14 GB |

⚠️ **The sources disagree and the harness says so** (`T2VSpot25.swift:240`) — it reports 89.6 and
refuses to bake either in, since that "would settle a live question by side effect." **That question
is now load-bearing.** → **AB-T-0076 blocks AB-T-0077.** Do not spend a 52-minute slot producing a
number nobody can interpret.

## Context for the stretch goal

2.3 ran 10s@1080p at **132.71 GB — 4.3 GB of headroom on 137 GB**, in 51.9 min. 2.5 streamed should
bring the same clip to **~95–100 GB**: a boundary row becoming a comfortable one *on the machine*,
with the budget rate deciding whether the governor agrees.

⚠️ Wall time at HD under streaming is **unmeasured**. Never quote the memory win without it.


---

# 10s MEASURED — 720p and 1080p (2026-08-22, AB-R-0118)

Fresh boot, serial, interleaved. `LTX_ENVELOPE_OVERRIDE=1` throughout — **no acceptance numbers**.
Every streamed arm verified STREAM from its last gate line; every `request` line verified at the
geometry asked for.

| arm | lane | peak | wall |
|---|---|---|---|
| 720p 241f streamed ×2 | STREAM | **44.94 / 44.86** | 926 / 977 s |
| 720p 241f **resident** | resident | **50.79** | 875 s |
| 1080p 241f streamed ×2 | STREAM | **97.54 / 96.96** | 3212 / 3353 s |

## ✅ Falsification test PASSED

Predicted ~43.7 GB for 10s@720p from frame-amortisation; **measured 44.94** — 121f → 241f costs
**+1.24 GB for 2× frames**. Model holds. The 1080p projection (95–100) was likewise accurate at
**97.54**.

## 🚨 1080p landed in the MIDDLE band — AB-T-0076 provably gates the verdict

| rate | budget | 97.54 |
|---|---|---|
| 0.7× | 89.6 | ❌ OVER by 7.9 |
| 0.85× | 108.8 | ✅ WITHIN by 11.3 |

Measuring did **not** dissolve the dependency. ⚠️ Which makes AB-T-0076's "do not pick the number
that makes 1080p pass" warning more important, not less.

## ⟲ RETRACTED — "streaming is worth ~37 GB at 720p"

That compared 2.5-streamed against **LTX-2.3's** 80.36 GB receipt, which **predates
`evictDiTBeforeDecode` on every tier** (2026-08-13) — two levers, not one.

**Controlled, same binary, one variable: 50.79 resident vs 44.94 streamed = 5.85 GB.** The other
~31 GB was DiT eviction, which both arms already get.

🔑 **The benefit shrinks as resolution rises** — 20.5 GB at 704×512 (45.29→24.81), 5.85 GB at 720p.
Streaming removes WEIGHTS, and as resolution grows the peak migrates from denoise (DiT resident) to
decode (DiT already evicted), where there is no weight left to remove.

⚠️ **What streaming buys at 1080p is UNMEASURED** — the resident 1080p arm was skipped deliberately
(~134 GB on a 137 GB box, unattended). Do not claim a streaming win at 1080p.

## Wall time — indicative only

1080p **~54 min** vs 2.3's 51.9 min. Fresh boot is the worst case for timing and n=1 per arm; the
720p reps (926/977 s) straddle the resident arm (875 s) closely enough that **no timing claim
survives**. Same order as 2.3, never a delta.

## Claim status

- ✅ **10s@720p ≈ 45 GB — comfortable on max128 under either rate.** Quotable.
- ⏸️ **10s@1080p = 97.54 GB — runs, and fits the MACHINE with ~39 GB spare** (2.3 had 4.3). Whether it
  fits the TIER is AB-T-0076's call.
- ❌ The ~37 GB streaming claim is retracted.
