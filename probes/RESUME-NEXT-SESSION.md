# RESUME — next session starts here

> ✅ **2026-08-21 — LTX-2.5 is PUBLISHED, gate-green, and shipping on all four tiers by decision
> (AB-D-0035).** Everything committed and pushed.
> **Next slot = Phase 1 of the consumer-app plan: the AUDIO VERDICT.** One script, ~1 hour,
> needs a GPU slot **and your ears**: `probes/audio25/run-audio-verdict.sh`

## Where things stand

**Published and reachability-gated** (`--weight-sources-reach-gate` PASS, 9/9 reachable AND
complete across 18 configuration arms):

| | |
|---|---|
| `mlx-community/ltx-2.5-mlx` · `-ditq8` · `-q8` | 2.5 weights |
| `xocialize/ltx-2.5-granules` (58.6 GB, stamped + verified) | streaming |
| `xocialize/ltx-2.3-mlx{,-q8,-q4}` · `ltx-2.3-granules` | 2.3, now incl. all three upscaler variants |

Streaming is the default on advised tiers, auto-followed from the profile with an explicit override
(`textEncoderQuant`, `forceStreamGate`, `streamedBlocks` — all tri-state, `nil` = follow).
Board: package **38/38**, reach **PASS**, stream-tiny, pipeline-25, 2.3 q4 parity.

## 🚨 The one thing that blocks a consumer app — and it is NOT ours to fix

**AB-T-0069 / AB-A-0012 (→ MLXEngine).** `QuantFootprint` is per-QUANT; the measured footprint now
varies by TIER because low tiers stream. The governor therefore charges 28.0 GB for a config that
measures 15.5, and **REFUSES compact24 (16.8) and balanced32 (22.4)** — the two tiers the whole
low-tier effort unlocked.

It fails **CLOSED** (denies a fitting config) rather than open, so nothing breaks — it just denies
the machines the work was for. ⚠️ **Do NOT "fix" it by declaring the streamed numbers
unconditionally**: that under-declares the resident path (max128 stays resident) and would admit
runs the governor should refuse. Failing open is worse.

**Until it resolves, 2.5 is a `standard64`+`max128` product in practice.** Consider shipping those
two first and adding low tiers when the ask lands.

## The plan (agreed 2026-08-21)

### Phase 1 — AUDIO VERDICT ← **start here**

```bash
cd LTX_DEV/ltx-2-mlx-swift && ./probes/audio25/run-audio-verdict.sh
```

6 clips (3 speech, 3 ambient) + an mlx-whisper WER leg on the speech arms. **You listen; that is the
verdict.**

**Why it is open:** 2.3 distilled audio is prosodic babble, not intelligible speech — a MODEL
property. AB-R-0002 found 2.5's audio STACK byte-identical to 2.3's, which is often misread as
settling it. **It does not: the stack is the decoder; generation comes from the DiT, and 2.5's DiT
is a different model.** Unmeasured, not inherited.

**Decides:** does the app promise *speech*, *ambient/foley only*, or *audio off by default*? All
three ship; only one is honest. ⚠️ The ambient arms are the control — non-speech audio can be
excellent while speech is babble, and judging only speech prompts would condemn the feature on its
hardest case.

### Phase 2 — consumer-path items (~half a day, mostly desk work)

⟲ **Smaller than first scoped.** Downloads are engine-side: `WeightMaterializer` already reports
`fraction`/`totalBytes` with a free-space preflight. Enhancer eviction is an app-side
`evict(package:)` call, not an engine change.

- **2a** Enhancer load→generate→**evict**. Prove the **7.19 GB actually returns** (measure phys
  before/after — do not trust the call). Also settle **AB-A-0009**: is the governed enhancement path
  seed-pinnable? Open with 5 replies, and reproducibility matters for a product.
- **2b** i2v end-to-end through the package. Works in-harness; what is unverified is adapter
  **materialization** — confirm the i2v adapter is not `licenseGated` and downloads on a cold run.
- **2c** compact24 + i2v advisory (92% of budget) — documented in `probes/tier25-matrix/README.md`,
  app-side to surface. i2v-specific; t2v on that tier is 87%.
- **2d** `max128` streamed — the one tier defaulting to resident, never measured streamed. Doubles
  as the on-ramp to Phase 3.

### Phase 3 — 128 GB now, 768 GB speculative (HOLD until M5 Ultra specs are real)

Measured: 481f@96 (20 s, the frame cap) = **66.01 GB peak**, native — so 128 GB already has ~60 GB
spare at the longest clip any profile permits.

🔑 **The wall is time, not memory.** Attention compute is O(N²) while activation is ~O(N), so
lengthening clips hits wall-clock first: 481f took 533 s, and doubling frames is ~4× attention work
(~35 min for 40 s). **Hypothesis to test, not assume:** more RAM buys **co-residency** (LTX +
enhancer + upscalers, no eviction, no streaming) and **batch throughput** far more than single-clip
length. Also note the frame cap is a DECLARED limit, not a measured one.

⚠️ Do not build for a rumoured 768 GB machine before its specs are real.

## Protocol reminders (earned, do not re-learn)

- **Serial only** for anything streaming — S is a shared-resource measurement; overlap voids every
  S/stall number (verdicts survive).
- **No `MLX_PROFILE`** for acceptance numbers — it breaks fusion and inflated a decode span *above*
  the clean whole-run peak.
- **No burn-in for memory**; burn-in is for TIMING claims only. A fresh boot is the worst case for
  timing stability but right for memory safety.
- **Worst-case, never mean** — compact24's `.auto` headroom once ranged 1.33×…0.59×.
