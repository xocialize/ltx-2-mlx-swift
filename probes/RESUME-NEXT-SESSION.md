# RESUME — next session starts here

> ✅ **LTX-2.5 is PUBLISHED, gate-green, ADMISSIBLE ON ALL FOUR TIERS, and Phase 1 + most of
> Phase 2 are closed.** Everything committed and pushed; three repos clean.
>
> ▶️ **NEXT = PHASE 2d, `probes/max128-streamed/run-max128-streamed.sh`** — the last unmeasured
> tier+mode. It is a memory ACCEPTANCE measurement, so it **needs the dedicated slot**: serial only,
> no `MLX_PROFILE`, no burn-in, worst-case never mean. Harness is staged, parses, smoke-tested,
> and aborts if another `RunLTX2` is live.

## ⏸️ Paused for an operator reboot

Nothing running. `mlxengine-video-ltx`, `ltx-2-mlx-swift`, `ltx-2-mlx` all clean and pushed.
Granules present: **35 G bf16 · 19 G q8** at `/Volumes/Satechi/Models/ltx-granules-25/`.

🚨 **Two traps the 9f smoke test found — read before quoting ANY number from that run:**
1. **The SPLIT line's `lane=` reports the CONFIG, not the gate outcome.** At 9f the gate fell back
   resident on both passes and still printed `lane=STREAMED, peak 44.10 GB` — a RESIDENT number
   under a STREAMED label. **Confirm the LAST `[BlockStreamer] gate:` line says STREAM first.**
2. **Read the bind line for sweep sizes, never scale them.** bf16's sweep is **34.56 GiB**
   (measured); scaling the on-disk DiT ratio gave 36.38e9, ~2% out, because int8 quantizes only the
   block Linears while the sweep also carries tensors that never quantize.

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

## ✅ THE CONSUMER-APP BLOCKER IS GONE — AB-T-0069 / AB-A-0012 / AB-T-0070 all DONE

⟲ **This section said "the one thing that blocks a consumer app" for most of 2026-08-21. STALE.**
All three landed the same day.

**AB-T-0069** (ltx-2-mlx-swift v0.9.0 `347d4b5`, receipt AB-R-0107) — the streamed low tiers are
**admissible through the governor**:

| tier | charge now | corridor [worst measured, budget] | before |
|---|---|---|---|
| compact24 | **16.20** | [15.49, 16.8] ✅ | 28.0 → REFUSED |
| balanced32 | **18.00** | [17.14, 22.4] ✅ | 28.0 → REFUSED |
| standard64 | 28 | covers streamed 19.43 AND `.auto` fallback 27.31 | 39 (stale 2.3 hint) |
| max128 | 76 | unchanged | 76 |

🔑 **No engine change and no PackageID split were needed** — `FootprintConfigured` existed since
contract 1.13/1.14 and this config already conformed. The bug was a hard-`nil` `residentBytesHint`
sitting under a comment asserting the exact assumption streaming broke. Three fail-closed guards:
streamed numbers only where streaming is GUARANTEED (pinned, never `.auto`), family-keyed, and a
`load()` guard that throws when a pinned tier has no resolvable granule tree. Gate **43/43**.

**AB-T-0070** (mlx-engine-swift **v0.46.0 / contract 1.34.0**) — per-volume disk characterization at
prepare + `QuantFootprint.minSustainedReadBytesPerSecond` + below-floor refusal **before any command
buffer**. **I9 is now refusable instead of a mid-generation abort.** Our pin (`from: "0.32.0"`)
already resolves it.

### 📋 It left US an action item — now filed as **AB-T-0075**

LTX must DECLARE its floors; the engine only enforces them. ⚠️ And the follow-up's wording
(*"bf16/resident lanes"*) **predates AB-T-0069**: a resident lane reads its checkpoint once, but a
**PINNED streamed tier reads granules throughout generation and cannot fall back** — `.forceStream`
is exactly what removed the coin flip. compact24's 16.20 GB admission *depends* on streaming; its
`.auto` fallback peak is 31.92 against a 16.8 budget. **A disk too slow to stream does not slow that
tier down, it invalidates the footprint it was admitted on.** ⚠️ The floor VALUE is unmeasured
between ~475 MB/s (USB, bf16 0/7 crash) and PCIe — and it refuses hardware, so it is an operator
call. See AB-T-0075.

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

### 📋 QUEUED — **AB-T-0074**: re-examine the q4 ENCODER rejection (operator-authorized)

int4 was rejected on the **connector gate** (0.996728 vs a 0.999 bar) and **never perceptually
tested**. It now has its first perceptual evidence — indistinguishable, blind — on a narrow prompt
class. **Under STREAMING the encoder is the binding memory term** (that is what took compact24 from
24.79 → 14.57 GB), and q4 is **7.6 GB vs q8's 13 GB**, aimed straight at **compact24 i2v's 92%-of-
budget thinness**.

**Operator authorized the re-exam** on the strength of the blind ladder judging BOTH axes
("great both visually and audio… couldn't pick favorites"), with q4 and poison both in the set.

⚠️ **Re-examination, NOT a reversal.** Four preconditions, full detail in AB-T-0074:
1. **ADHERENCE A/B on visually specified prompts** — judge *"contains what was asked for"*, not
   *"looks better"*. Keep **poison** in as the calibration arm; if poison passes adherence too, the
   test is the problem. ⚠️ The ladder's prompts had almost no visual content, which is exactly why
   the operator's favourable video read does not settle this.
2. **Measure the tier saving** — under eviction the encoder is largely outside the peak (AB-R-0034:
   int8 moved the resident-DiT peak 0.18 GB); only the STREAMED regime makes it binding. ≥3 reps,
   worst-case, `.forceStream`, t2v AND i2v.
3. **Reconcile** the same quant class being rejected in `qwen-image-edit-swift`.
4. **Re-run the deciding gate number** — connector output at VALID positions, not the diluted
   whole-array figure (AB-L-0017).

### Phase 2 — ✅ 2a / 2b / 2c DONE · ▶️ 2d is the only item left

⟲ **Most of Phase 2 needed no GPU — it was already answered in committed receipts nobody re-read.**

- **2a ✅ enhancer evict (AB-R-0115).** phys **0.24 → 7.43 → 0.42 GB** after release + `clearCache`.
  The 7.19 GB returns; residue 0.18 GB.
  🚨 **But the SEAM arm, which releases WITHOUT `clearCache`, leaves 4.13 GB charged** — the MLX
  buffer-pool ratchet. **"Evict the enhancer" is an INCOMPLETE instruction**: pair it with a cache
  clear or ~3 of the 7.19 GB never comes back. Load-bearing because LTX peak 42.27 + enhancer 7.19
  = **49.46 GB against standard64's 44.8** — the two cannot be co-resident, so the tier depends on
  the enhancer fully leaving.
  ✅ Determinism settled too: **AB-A-0009 answered in the POSITIVE** — `LLMParameters.seed` (engine
  v0.45.0 / contract 1.33.0), gated live on `GemmaLLMPackage` itself. ⚠️ **Available ≠ used: the
  CALLER must set it.** Reproducibility is possible, not automatic.
- **2b ✅ i2v materialization (AB-R-0114).** NOT `licenseGated`, auto-materializes via
  `LoRACache.ensure()`, and resolves **unauthenticated at 200** with a byte-identical size — the
  check AB-T-0067 exists to enforce.
  ⚠️ The fetch is inside `run()`, so a cold install **stalls the first i2v generation for 4.93 GB**
  and the engine's free-space preflight knows nothing about it. 🚨 Do NOT add it to `weightSources`
  — every t2v-only install would download it. Request-scoped ≠ config-scoped. App-side remedy
  documented in `probes/tier25-matrix/README.md`.
- **2c ✅** compact24 + i2v advisory — documented; app-side to surface.
- **2d ▶️ max128 STREAMED — THE DEDICATED-SLOT ITEM.** The one tier whose profile advises RESIDENT,
  so it has never been measured streamed. Two things ride on it: its declared **76 GB** charge may
  be enormously conservative (more co-residency), and **481f@96 is the longest clip any profile
  permits** (66.01 GB native, AB-R-0073) — the Phase 3 on-ramp. ⚠️ 481f is **3× the largest frame
  count ever measured on 2.5 streamed**; flat 9→161f is strong evidence but does not extend
  automatically. ⚠️ Streaming max128 is an **explicit override** (its profile advises resident) —
  that is the AB-D-0035 escape hatch working, not a profile change. **A measurement is not a
  decision.**

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
