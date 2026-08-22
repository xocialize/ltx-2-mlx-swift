# Phase 2d — max128 STREAMED (2026-08-21 fresh boot, AB-R-0116)

The last unmeasured tier+mode. Fresh boot (2 min uptime, 95% free, 0 swap), serial, no
`MLX_PROFILE`, no burn-in — the correct posture for a MEMORY measurement.

**All 6 streamed arms genuinely STREAMED**, confirmed from the LAST `[BlockStreamer] gate:` line in
every log — never from the `lane=` label, which reports the CONFIG and was caught printing
`lane=STREAMED` on a fallback during the 9f smoke test.

| arm | reps | peak (worst) | vs 89.60 GB budget |
|---|---|---|---|
| resident 161f | 3 | **45.29** | 51% |
| resident 481f (control, same boot) | 1 | **48.59** | 54% |
| **streamed 161f** | 3 | **24.81** | 28% |
| **streamed 481f — the frame cap** | 3 | **24.81** | 28% |

## 🔑 Streamed peak is IDENTICAL at 161f and 481f — the ENCODER-FLOOR signature

24.80 / 24.81 across a **3× frame increase**: the same number to 0.01 GB. AB-R-0090 identified a
geometry-independent floor at **24.79–24.82 GB** as the bf16 12B Gemma-4 encoder, on exactly this
evidence (256×256×9 and 512×288×121 agreeing to 0.01 GB). max128 runs the bf16 encoder by profile
advice, so **once the DiT is streamed away the encoder is the floor and frame count stops mattering.**

⚠️ Resident DOES scale — 45.29 → 48.59 (+3.30 GB) over the same 3×. The invariance belongs to the
streamed lane, not to the geometry.

## The declared charge is conservative by 51 GB

`MLXLTX25Package` charges max128 **76 GB** (bf16 40 resident + 36 activation) against a measured
worst case of **48.59 resident / 24.81 streamed**.

🔑 **On max128 the enhancer can be CO-RESIDENT**: LTX streamed 24.81 + enhancer 7.19 = **32.0 GB** of
89.60. Even resident + enhancer = 55.8. This is the tier where AB-R-0115's evict-then-`clearCache`
discipline is *not* forced — unlike standard64, where 42.27 + 7.19 = 49.46 busts a 44.8 budget.

## 🚨 TIMING FROM THIS RUN IS VOID — my harness design, not the box

| 161f resident | 93.2 · 131.5 · **164.2** s |
|---|---|
| 161f streamed | 155.0 · 157.3 · 162.3 s |

The resident arm spans **93 → 164 s on identical work**, monotonically — the documented fresh-boot
thermal ramp. ⚠️ **And the arms ran all-resident-then-all-streamed**, handing the cold/fast part of
the session to one and the warm/slow part to the other. The doctrine is explicit that **ABBA cancels
symmetric drift, not a one-way ramp** — and this was not even ABBA. Any apparent streaming penalty
here is confounded with warm-up and is **not quotable**. The 481f arms (598/580/567 vs a 554 resident
control) only hint the penalty shrinks at large N; n=1 control, same ramp. **Claim unmade.**

🔑 The trade was still correct — burn-in would have corrupted the memory numbers, which were the
point. A timing claim needs its own run: burned in, ABBA-interleaved.

**What IS supported** (measured, not timed): at 481f the gate reads **N=21973, S=5.14 GiB/s vs
C(N)=0.33 GiB/s → N_min≈1404** — ~**16× headroom**. Streaming at the frame cap is the regime
streaming exists for, not a marginal call.

## ⚠️ To reconcile — not a contradiction

AB-R-0073 records 481f **@96** = 66.01 GB native; this reads **48.59** resident at 481f. Different
regimes: ours is plain t2v at the harness default **fps 24**; AB-R-0073's was the native arm of a
**DFR r=2** comparison at 96 fps playback, so audio token counts and pipeline differ. Neither refutes
the other. Re-measure in this regime before quoting 66.01 for max128 sizing.

## ⚠️ A measurement is not a decision

max128's profile advises RESIDENT and this does not change that — streaming here was an explicit
override, the AB-D-0035 escape hatch working as designed. Whether max128 should stream by DEFAULT
(halves peak, unlocks co-residency, timing cost unmeasured) is an operator call that needs a proper
timed run first.
