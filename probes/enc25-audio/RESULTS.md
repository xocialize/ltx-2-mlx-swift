# int8 encoder × speech — 12 runs, 4 encoder arms × 3 sentences (2026-08-21, AB-R-0111)

Closes the gap Phase 1 left: every audio arm there ran **bf16**, but the low tiers AUTO-FOLLOW to
int8 (AB-D-0035), and AB-R-0104's blind A/B judged **video**.

Tier/geometry/DiT-quant fixed (standard64, 704×512×121, int8 DiT), seed pinned 42, interleaved by
sentence. Sentences are low-prior and homophone-free per AB-R-0110. **The encoder is the only
variable.**

| arm | tree | connector cosine | WER ×3 |
|---|---|---|---|
| bf16 | `ltx-2.5-mlx` 22 GB | 0.999879 (floor) | 0.12 / 0.00 / 0.00 |
| **q8 — under test** | `-q8` 13 GB | 0.999820 | **0.12 / 0.00 / 0.00** |
| q4 — control, REJECTED | `-q4` 7.6 GB | 0.996728 | 0.12 / 0.00 / 0.00 |
| **poison — control, BROKEN** | `-poison` 5.9 GB | **0.829329** | **0.12 / 0.00 / 0.00** |

Every 0.12 is one artifact: whisper writes *"at 7."* for *"at seven"*, identically in all four arms.

## 🚨 The controls did not move — and the pre-registered rule said that voids the result

Registered in the script: *"q4/poison ≈ bf16 → the instrument is BLIND; the q8 result means
nothing."* So the first duty was to disbelieve it. Three checks:

1. **Plumbing.** Each arm loaded its intended tree; the `LTX_ENC_TREE` hatch announced itself on the
   control arms only. Not a silent fallback to bf16.
2. **Did the swap change anything?** Frame-40 hashes differ across all four arms, and the audio
   waveforms are entirely different renderings — correlation vs bf16: q8 **0.116**, q4 **0.062**,
   poison **0.004**.
3. **Negative control — the decisive one.** Is whisper hallucinating the target from a weak signal?
   Scored s1's sentence against clips that never contained it: ambient-rain → *"Thank you."*,
   ambient-workshop → *"Okay."* (whisper's classic low-signal outputs), WER 1.00; `s2-bf16`
   transcribed its OWN line and scored 1.12 against the wrong expectation. **Whisper reports what is
   there; it does not invent the target.**

⟲ **The rule was too crude.** "Controls didn't move" has two readings — *the instrument is broken*
or *this is not the axis the damage appears on* — and I had collapsed them into the first. The
negative control separates them. The chain is **not** blind: **WER measures WHICH WORDS, and encoder
damage does not appear there.**

## Established

✅ **int8 does not change which words are spoken** — 3/3 identical to bf16. Speech CONTENT is safe on
the tiers that auto-follow to int8.

🔑 **Word content survives GROSS encoder corruption.** An encoder at 0.829 connector cosine — one
this project REJECTED, whose video is a visibly different generation and whose audio correlates
0.004 with bf16 — still says the correct line. The encoder governs **how** a line is delivered far
more than **what** is said.

⚠️ **The 0.116 → 0.062 → 0.004 ordering is NOT a quality metric.** It measures divergence from bf16,
and two different-but-equally-good renderings also correlate ~0. Suggestive of dose-response; one
sentence, three points. Not banked.

## NOT established — this is what decides the tier

**Voice quality, naturalness and speaker plausibility.** WER structurally cannot see them, and this
run proves it: the metric cannot separate bf16 from an encoder we rejected.

Blind set: `~/Desktop/ltx25-enc-BLIND/` — 12 clips, `setN-A..D`, arm order shuffled per set, key
withheld until after the verdict.

🔑 **The poison arm is now a CALIBRATION arm, worth more than its original role.** Cannot tell it
from bf16 by ear → encoder precision does not matter for audio, q8 trivially safe. Can tell it →
the ear discriminates, and q8's position on that scale is the answer. Either way it is decisive,
which is exactly what the WER leg failed to be.

## Reproducing

```bash
./probes/enc25-audio/run-enc-audio-ab.sh
```
⚠️ `LTX_ENC_TREE` is a CONTROL-ARM hatch — it bypasses the config's own encoder resolution and
must never produce an acceptance number (package-gate case 18 exists to stop this harness
re-deriving what it should be reading).
