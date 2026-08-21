# RESUME — next session starts here

> ✅ **2026-08-21 — LTX-2.5 is PUBLISHED, gate-green, and shipping on all four tiers by decision
> (AB-D-0035).** Everything committed and pushed.
> ✅ **PHASE 1 IS FULLY CLOSED — AUDIO IS SHIPPABLE, INCLUDING ON THE LOW TIERS.**
> 2.5 generates intelligible, prompt-specified speech (AB-D-0036), and the **int8 encoder is
> CLEARED for speech** by a blind operator listen of the full encoder ladder (AB-D-0037).
> **Next = Phase 2** — desk work and short runs; **no dedicated GPU slot needed.**

## ⏸️ Session paused for an operator reboot (display-resolution issue on wake)

Nothing is running. All three repos clean and pushed. Nothing to restart or re-measure.

**Artifacts on the Desktop, all reboot-safe:**

| path | |
|---|---|
| `~/Desktop/ltx25-audio-verdict/` (12 MB) | Phase 1 clips — 3 speech, 3 ambient, +2 known-text |
| `~/Desktop/ltx25-enc-audio-ab/` (16 MB) | encoder ladder, 12 clips, arm-named + `manifest.tsv` |
| `~/Desktop/ltx25-enc-BLIND/` (16 MB) | the blind set (verdict already given — safe to delete) |
| `~/Desktop/ltx25-enc-BLIND-KEY.tsv` | its key, no longer withheld |

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

### ✅ Phase 1 — AUDIO VERDICT — **CLOSED. It reversed the expectation twice.**

**1. 2.5 generates intelligible, prompt-specified speech (AB-D-0036).** Operator approved every
clip; voices gender/age-appropriate; the workshop foley matched the depicted action (joint AV
coherence, not merely "audio exists"). ⟲ 2.3's babble is a MODEL property and AB-R-0002's
"2.5's audio STACK is byte-identical" never settled 2.5 — the stack is the DECODER; generation is
the DiT's.

**2. The int8 encoder is CLEARED for speech (AB-D-0037).** 4 encoder arms × 3 low-prior sentences;
q8 matched bf16 exactly (WER 0.12/0.00/0.00, the 0.12 being whisper writing *"at 7."*), and a blind
operator listen could not separate ANY arm — **including a poison encoder at 0.829 connector
cosine**. Both instruments agree. Full write-up: `probes/enc25-audio/RESULTS.md`.

🚨 **Two corrections live in AB-R-0110 — read it before quoting the Phase 1 numbers.** speech-05 was
filed first as a "homophone artifact" and then as *corroborated* because the operator heard it
correctly before reading my note. **Both were wrong.** `spine` is not a homophone of `Spain`, the
audio genuinely does say *spine*, and the **stock sentence** primed the human judge — not my note —
so the "independent" check shared the prior it was meant to test. **Count is 2 of 3 exact, 1 with a
real word error.**

🔑 **Mandatory going forward: LOW-PRIOR, homophone-free known-text sentences.** A famous sentence
correlates the errors of the model, the ASR AND the human judge at once, leaving no independent
check anywhere in the loop.

🔑 **WER measures WHICH WORDS — encoder damage does not appear there.** Proven: the metric cannot
separate bf16 from an encoder we rejected. What separates them is nothing the ear could find either.

⚠️ **Scope limit, in the test design not the listening:** the encoder-ladder prompts specified almost
no visual content (*"a person looks directly at the camera and says clearly: …"*), so **visual
adherence under a degraded encoder is UNTESTED**. A degraded encoder's failure mode is *ignoring the
prompt*, which a quality-preference judgement cannot see. AB-R-0104 (q8 only) is still the video-axis
evidence.

⚠️ Still unmeasured on audio: seed-to-seed consistency, multi-speaker/dialogue, lip-sync under
scrutiny, non-English, and arbitrary (non-stock) sentences — `speech-06-arbitrary` is in the script
and is one short run from closing the last of those.

### 📋 NEW — the q4 ENCODER rejection is worth re-examining (opportunity, NOT a reversal)

int4 was rejected on the **connector gate** (0.996728 vs a 0.999 bar) and **never perceptually
tested**. It now has its first perceptual evidence — indistinguishable, blind — on a narrow prompt
class. **Under STREAMING the encoder is the binding memory term** (that is what took compact24 from
24.79 → 14.57 GB), and q4 is **7.6 GB vs q8's 13 GB**, aimed straight at **compact24 i2v's 92%-of-
budget thinness**.

⚠️ Do NOT reverse on this. Needed first: (a) an A/B on **visually specified** prompts judging
**ADHERENCE**, not preference; (b) the real tier measurement — under eviction the encoder sits
largely outside the peak (AB-R-0034: int8 moved the resident-DiT peak 0.18 GB); only the STREAMED
regime makes it binding; (c) the same quant class was rejected in `qwen-image-edit-swift`.

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
