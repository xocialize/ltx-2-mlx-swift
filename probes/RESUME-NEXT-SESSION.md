# RESUME — next session starts here

> ✅ **2026-08-21 — LTX-2.5 is PUBLISHED, gate-green, and shipping on all four tiers by decision
> (AB-D-0035).** Everything committed and pushed.
> ✅ **PHASE 1 CLOSED 2026-08-21 — AUDIO IS SHIPPABLE (AB-D-0036).** LTX-2.5 generates
> intelligible, prompt-specified speech; operator approved all three scripted arms plus the ambient
> foley. **Next slot = Phase 2** (desk work, short runs — no dedicated slot needed).

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

### ✅ Phase 1 — AUDIO VERDICT — **DONE, and it reversed the expectation**

**LTX-2.5 generates intelligible, prompt-specified speech.** Three known-text arms:

| arm | transcript | WER |
|---|---|---|
| speech-01 | *"The quick brown fox jumps over the lazy dog."* | **0.00** |
| speech-04 | *"Pack my box with five dozen liquor jugs."* | **0.00** |
| speech-05 | *"the rain in Spain stays mainly on the plain"* → *"…rein and spine…on the plane"* | 0.56 — **artifact** |

🔑 **speech-05 is a TRANSCRIPTION artifact, not a generation failure**, confirmed twice: it is the
famous elocution line, and **the operator heard "Spain" BEFORE reading any note about it** — the
correct hearing preceded the explanation, so it was not suggestion.

⟲ **"Homophone artifact" was too broad a label, corrected.** *rein/rain* and *plane/plain* are true
homophones and cost nothing. **`spine` is NOT a homophone of `Spain`** — /spaɪn/ vs /speɪn/ differ
in the vowel — so that substitution is a real mis-transcription, not a spelling coin-flip. The
operator's ear resolves it and the perceptual verdict is theirs by doctrine, so the shipping
question stands; the record should not claim all three subs were free.

**Operator verdict: all three approved.** Voices gender/age-appropriate to the person on screen, and
the workshop foley was correct AND matched the video — joint AV coherence, not just "audio exists".

⟲ This kills the inherited assumption. 2.3's babble is a MODEL property, and AB-R-0002's
"2.5's audio STACK is byte-identical" was read as settling 2.5 too. It never did — the stack is the
DECODER; generation is the DiT's, and 2.5's DiT differs.

⚠️ **WER is the wrong instrument for homophone-dense lines** — it logged a false negative here.
Read it phonetically, or script lines without homophones.

🚨 **WHAT PHASE 1 DID *NOT* ESTABLISH — all three known-text arms are FAMOUS stock sentences**
(two pangrams and an elocution line). A model can land those partly off its language PRIOR rather
than off the prompt, so they probe **intelligibility** well and **prompt-following** weakly.
Consumers type arbitrary dialogue, which has no such prior. So the claim earned is *"speech is
intelligible and well-voiced"* — real, and the question that was open — **not** *"it says the
specific arbitrary thing you typed"*. `speech-06-arbitrary` is now in the script (homophone-free,
no stock-phrase prior) and is one short run from closing it.

⚠️ **Also unmeasured:** seed-to-seed consistency, multi-speaker/dialogue, lip-sync under scrutiny,
non-English, and — cheapest and most load-bearing — **speech under the int8 ENCODER**. All audio
arms ran the bf16 encoder; the low tiers auto-follow to int8, and AB-R-0104's A/B judged VIDEO only.
If int8 conditioning degrades speech, it degrades it exactly where the app most wants audio.

⚠️ **The probe script did not reproduce its own run** — arms 04/05 were run ad-hoc and never written
back, and the harness logged no prompt, so the receipts could not regenerate the verdict. Both fixed
(prompts recorded, per-arm `expectations.tsv`, arms 04–06 in the file). This repo already earned
"print RESOLVED parameters, not intended ones"; the harness that proved audio works violated it.

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
