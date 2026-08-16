# Swift claims bench matrix (session task #7) — arm status and results

The six-arm matrix is specified in `LTX_TESTING/LTX25-PORT-PLAN.md` §V ("Bench matrix"), tracked by
**AB-T-0005**. This file records what the **Swift** side can run and what it found. Protocol for
every arm: matched output spec, bf16, stock decoder, cache 2 GB, ABBA blocks ≥2, ≥2 runs/arm/block,
30 s cooldowns, unprofiled totals, thermal strata logged.

## Arm reachability (scoped 2026-08-14)

| arm | claim | needs | Swift status |
|---|---|---|---|
| **A** perf parity 2.3 vs 2.5 | §V item 1 | model axis on `--bench-e2e` | ✅ **RUN — PARITY** |
| **B** DFR efficiency | C2 | `dfr=` axis + `DFRPipeline` | ✅ **RUN — DFR DOMINATED** |
| **C** decode triangle | C3 | DiffVAE decoder | ✅ ported + gated 2026-08-14 — runnable |
| **D** duration honesty | C6 | duration head | ✅ ported + gated 2026-08-14 — runnable |
| **E** enhancer overhead | C5 | `GemmaTextGenerator` | ✅ exists — runnable |
| **F** multishot cost | C1 | keyframe slots | ✅ ported + gated — runnable |

⚠️ **Arms C and D were BLOCKED when this matrix was first scoped** — the DiffVAE decoder and the
duration head shipped in the 2.5 weight tree but had no Swift port at all, so task #7 was not
executable as written. Both were ported and gated on 2026-08-14 (`--diffvae-gate`,
`--duration-gate`), which unblocked them. Recorded because "blocked on a port" and "blocked on a
measurement" are different states and the task description implied the latter.

---

## Arm A — 2.3-distilled vs 2.5-distilled, perf parity gate ✅ PARITY

`RunLTX2 --bench-e2e --arm ltx23:model=ltx23,quant=bf16,cache=2 --arm ltx25:model=ltx25,quant=bf16,cache=2
--size 704x512 --frames 121 --fps 24 --two-stage --blocks 2 --runs 2 --cooldown 30`

Receipts: `probes/bench_e2e_armA-perf-parity_20260814-161527.{md,json}`, log
`probes/armA-perf-parity.out`. Host Mac17,7 / 128 GB, git `ef5b901`.

| arm | median | range | peak (⚠️ UNEVICTED) |
|---|---|---|---|
| ltx23 (2.3 distilled) | **99.0 s** | 67.7–107.5 | 56.58 GB |
| ltx25 (2.5 distilled) | **98.5 s** | 95.1–103.2 | 64.56 GB |

**Δmed −0.4 s — NOISE (sign flips between block sets).**

🔑 **The pre-registered expectation is met.** §V item 1: *"Vanilla distilled t2v path: expect ≈
PARITY, not a speedup … Any measured regression on this path is a PORT bug, not the model."* There
is no regression — the 2.5 Swift port costs nothing measurable e2e against 2.3 at matched spec.
This is a NULL result being confirmed, which is only meaningful because the expectation was
registered before the measurement.

### Reading the numbers honestly

⚠️ **The intra-arm spread is far worse than the documented ±15 s noise floor, and it is
SYSTEMATIC, not random.** Two effects, both visible:

1. **Cold start.** ltx23's first measured run was **67.7 s at `nominal→fair`** — every other run in
   the session was `fair→fair` and landed in 95–107.5 s. That single run is 30 s faster than its own
   arm's next run. It is retained (the median is robust to it) but must never be quoted alone.
2. **Intra-block drift the thermal LABEL cannot see.** Within *every* block, run 2 was 6–8 s slower
   than run 1 despite both being labelled `fair→fair`: ltx25 95.3→101.8 and 95.1→103.2; ltx23
   100.5→107.5. The `fair` stratum spans a real performance range, so matching on the thermal label
   is NOT sufficient — run position within a block is a confound in its own right.

**ABBA is what makes the comparison survive both.** ltx23 ran first in block 0, ltx25 first in
block 1, so each arm took the cold-start advantage exactly once and the effects cancel. The
harness's own verdict ("sign flips between block sets") is the direct evidence that they did.
**Any future single-block or single-run arm in this matrix is uninterpretable** — the effects above
are larger than most deltas worth measuring.

### Two things that must NOT be quoted from this receipt

⚠️ **The peaks are UNEVICTED and are not comparable to the footprint receipts.** `--bench-e2e` does
not set `LTX_EVICT_DIT`, and `LTX2Pipeline.evictDiTBeforeDecode` defaults to `false`, so 2.5 reads
64.56 GB here against the **43.11 GB** measured for the same geometry under the evicted+capped
shipping regime (AB-R-0039/0041). Different regimes. The ~8 GB arm-to-arm gap (56.6 vs 64.6) is the
expected bf16 12B Gemma-4 encoder vs 2.3's 4-bit Gemma-3, consistent with AB-R-0034's unevicted 2×2.

⚠️ **`output ltx25 vs ltx23: cos=0.528606` is meaningless and is labelled `divergent-by-design`.**
These are different models. The label exists because `expectsIdenticalOutput` now requires matching
`model`; before that change the harness compared two bf16/stock arms, expected ≈1.0, and would have
reported a 0.53 cosine as catastrophic failure. Arm A compares **wall-clock only**.

**Provenance note:** the receipt records git as `ef5b901+dirty`. The dirt is the harness's own
output files being written during the run (`probes/armA-perf-parity.out` under `tee`), not source —
source was committed clean at `ef5b901` immediately before launch.

---

## Arm B — C2 efficiency: 2.5-DFR-1-round vs 2.5-native ❌ DFR DOMINATED

`RunLTX2 --bench-e2e --arm t25-native:model=ltx25,quant=bf16,cache=2
--arm t25-dfr1:model=ltx25,quant=bf16,cache=2,dfr=1 --size 704x512 --frames 241 --fps 48
--two-stage --blocks 2 --runs 2 --cooldown 30`

**Matched output spec 241f@48**, reached two ways: native generates it directly; DFR generates
121f@24 and densifies one temporal round. The request geometry is DERIVED from the single target
(`BenchArm.request`) and the delivered frame count is verified per run, so matched output is a
property of the harness rather than of the operator's arithmetic.

Receipts: `probes/bench_e2e_armB-c2-dfr-vs-native_20260814-170202.{md,json}`, log
`probes/armB-c2-dfr-vs-native.out`.

| arm | median | range | peak (⚠️ UNEVICTED) |
|---|---|---|---|
| t25-native | **239.7 s** | 225.7–257.2 | 65.53 GB |
| t25-dfr1 | **402.1 s** | 385.4–403.0 | 66.12 GB |

**Δmed +162.3 s > spread 31.5 s — MEASURABLE. DFR at one round costs ×1.68 native.**

### The claim does not survive, and it misses its own pre-registered bar

§V item 2 pre-registered: *"FFN-side token-steps ≈ wash vs native at one round; the structural win
is attention locality + no stage-2 refine on densified frames + it's trained-for … DFR beating
352.7 s at matched output spec is the claim's honest test."*

- Expected ≈ **wash** (+0%) at one round. Measured **+68%**.
- The stated honest test was beating **352.7 s**. DFR runs **402.1 s** — and 2.5-native reaches the
  same deliverable in **239.7 s**.

🔑 **This CONFIRMS the oracle rather than discovering anything.** `ltx-2-mlx/docs/claims/C2-dfr.md`
already concludes *"Temporal rounds DOMINATED — do not use"*, with native 2.8× cheaper for the
identical deliverable AND the operator judging native better on quality — dominated on both axes.
The Swift side now reproduces the cost half independently, at a different geometry
(704×512×241f@48 vs the oracle's 512×768×97) and a different implementation. Swift measures **×1.68**
where Python measured **×2.767**, so our DFR is *relatively* cheaper than the oracle's — but the
direction and the verdict are identical. The port CLAUDE.md's standing note ("rounds are a
QUALITY/parity feature, NOT a speed lever") is now Swift-measured, not inherited.

### Confidence — why this one is not thermal

Arm A established that this harness's noise is systematic (cold start, intra-block drift). None of
it can account for this:

- The delta is **5× the combined spread** (162.3 s vs 31.5 s).
- **It survived order reversal.** DFR ran second in block 0 and FIRST in block 1 — its early-slot
  time (401.6 s) is within ~1 s of its late-slot times (403.0 / 402.5). Slot position moves DFR by
  ~1 s; the arm gap is 162 s.
- Both arms self-reproduce at `cos(first)=1.000000` across blocks.
- The one cold-start artifact is visible and excluded: native's block-0 warmup ran 149.8 s at
  `nominal→fair` vs 197.0 s for its block-1 warmup at `fair→fair` — a 47 s within-arm swing that
  never touches the statistics because warmups are excluded.

### ✅ The memory half of the claim is clean

**DFR costs +0.59 GB peak (66.12 vs 65.53).** The historical "+31 GB for DFR" figure was entirely
the DiT staying resident under the decoder, not the feature — `dropDiTIfSequential` fixed it and
this run confirms the fix at a 241f geometry. Consistent with the oracle's +0.90 GB at rounds=0.

### Scope — what this does NOT settle

⚠️ **One round only, one clip length.** The oracle names the untested regime explicitly: *"The one
regime that could still favour them is longer clips, where native high-fps generation runs out of
sequence length."* At 241f@48 native has no such trouble, so this measurement cannot speak to it.
A round-2 arm (`dfr=2`, delivering 481f@96 from 121f@24) is the natural next probe and the harness
now supports it directly.
⚠️ **Cost only, not quality.** §V also asks for DFR-vs-native at matched WALL-CLOCK budget,
operator-judged. That is a blinded A/B, not a bench arm. The oracle's operator already judged
native better; the Swift side has not re-run that.
⚠️ Peaks are **UNEVICTED** — not comparable to AB-R-0039/0041's shipping-regime footprints.
⚠️ `output t25-dfr1 vs t25-native: cos=0.769262` is `divergent-by-design` and meaningless as a
quality signal: different computation paths. `expectsIdenticalOutput` requires matching `dfrRounds`
precisely so this is never read as a regression.

---

## Arm B (completion) — the 2.3 baselines, finally under ABBA ✅ CLAIM'S OWN BAR WAS CORRUPTED

Run on a **fresh boot, otherwise-idle box** (2026-08-15), same 241f@48 matched output spec, 3 arms
× 2 ABBA blocks × (1 warmup + 2 measured). Receipts:
`probes/bench_e2e_armB-23baselines-abba_20260815-054515.{md,json}`, log
`probes/armB-23baselines-abba.out`.

| arm | median | range | spread | peak (⚠️ UNEVICTED) |
|---|---|---|---|---|
| t23-native | **224.5 s** | 213.4–235.1 | 21.7 | 58.57 GB |
| t25-native | **230.4 s** | 228.6–234.0 | **5.4** | 67.14 GB |
| t23-temporal-x2 | **357.4 s** | 345.5–368.2 | 22.7 | 59.15 GB |

- `t25-native` vs `t23-native`: **Δmed +2.7 s ≤ spread 22.5 s — NOISE** (drift-adjusted; the v2
  detector flagged an −8.6 s session drift rivalling Δ and decided on the adjusted figure).
- `t23-temporal` vs `t23-native`: **Δmed +133.0 s > spread 22.7 s — MEASURABLE, ×1.59.**

### 🚨 The 2.3 baselines the C2 claim was stated against were BOTH wrong

|  | old single (blocks=1/runs=1) | ABBA median | error |
|---|---|---|---|
| 2.3-native 241f | 352.7 s | **224.5 s** | **+57% inflated** |
| 2.3-temporal-x2 | 399.0 s | **357.4 s** | +12% inflated |
| implied ratio | ×1.131 | **×1.592** | **understated the cost of densification by 1.41×** |

🔑 **They were wrong in a DIRECTIONAL way, not merely noisy.** Because native was inflated far more
than temporal, the old pair made densification look nearly free (×1.13) when it actually costs
×1.59. That is very likely why C2 looked plausible enough to be worth testing at all. §V named
*"DFR beating 352.7 s at matched output spec"* as the claim's honest test — **that bar was itself an
artifact**; the real bar is ~225 s, and DFR's 402.1 s misses it by ×1.8 rather than by 50 s.

### 🔑 Being TRAINED-FOR bought nothing on cost

| generation | native | densified | ratio |
|---|---|---|---|
| 2.3 — hand-wired temporal-x2 | 224.5 s | 357.4 s | **×1.59** |
| 2.5 — trained-for DFR (AB-R-0044) | 239.7 s | 402.1 s | **×1.68** |

Densification costs essentially the same relative price on both generations. §V item 2's structural
argument for C2 rested substantially on 2.5's rounds being *trained-for* (keyframe-slot SFT +
distilled LoRA) where 2.3's was our hand wiring — **that distinction does not show up in cost.**

### Parity extends from 121f to 241f

Arm A found 2.3 ≈ 2.5 at 121f (Δ −0.4 s). This finds the same at 241f (Δ +2.7 s, ×1.027) — so 2.5
does **not** scale better with frame count, and the apparent 352.7-vs-239.7 advantage was entirely
the corrupted baseline. One consistent story across a 2× frame increase.

### Method notes

✅ **Cross-session reproducibility, measured.** `t25-native` was re-run here as an anchor: 230.4 s
median vs **239.7 s** last session (Δ 3.9%), with peaks matching to the decimal (66.62 GB warmup
both sessions). Arm B's +162.3 s DFR result was internal to one session; this shows the anchor does
not drift between them.
✅ **A quiet box tightens the spread 4×** — `t25-native` spread 5.4 s here vs 31.5 s last session,
same arm, same geometry. Worth scheduling long timing arms onto a fresh boot.
⚠️ **HARNESS DEFECT FOUND AND FIXED — third instance of the same class.** The receipt line
`output t23-temporal vs t23-native: cos=0.502494 (same-weights)` asserts these arms should be
identical. They differ only by `env.LTX_UPSAMPLER`, which selects a different upsampler checkpoint
AND a different stage-1 geometry. `expectsIdenticalOutput` did not consider `env` — after `model`
(arm A) and `dfrRounds` (arm B), this is the third axis found missing from that predicate. `env` is
an open-ended escape hatch and cannot be allowlisted safely, so **any env difference now forfeits
the identity expectation.** The timing result is unaffected; the quality label was wrong.

---

## Arm D — duration honesty (C6) ✅ SWIFT REPRODUCES THE ORACLE'S SHAPE

`RunLTX2 --duration-battery` — 10 prompts encoded FRESH through the real chain (Gemma-4 → connector
→ duration head). Distinct from `--duration-gate`, which is a parity check on banked encodings: a
parity gate would pass just as green on a head that returned a constant. Log:
`probes/armD-duration-battery.out`.

| axis | Swift (arm D) | oracle (AB-R-0015) | agrees? |
|---|---|---|---|
| responds to pacing | **YES** — long/short **×1.270** | YES, ~+37% | ✅ |
| explicit stated duration | **IGNORED** — "3 second" 4.08 s vs "15 second" **3.88 s** (×0.952) | IGNORED — 4.56 vs 4.28 s | ✅ |
| range | 3.88–5.65 s | 3.4–6.4 s | ✅ |
| short vs medium | 11.3% apart | TIED | ⚠️ marginal |

🔑 **The explicit-duration inversion reproduces.** Asking for *15 seconds* yields a SHORTER
prediction than asking for *3 seconds* — on both stacks. A head that merely ignored the number would
give ~equal values; getting the same inversion is strong evidence the Swift port is faithful rather
than coincidentally near.

⚠️ **C6's verdict is the ORACLE's (PARTIAL) and is not re-litigated here.** This battery asks only
whether the Swift port reproduces that shape. A Swift result that *disagreed* — e.g. suddenly
responding to explicit durations — would indicate a PORT bug, not a better model.
⚠️ The short-vs-medium row is the one non-match: 11.3% apart where the oracle found a tie. Different
prompt sets, so not a contradiction — recorded rather than smoothed over.
⚠️ 0 of 10 cases hit the [1,20]s clamp. The oracle exercised it via a `video_only` case predicting
24.8 s; every prompt here is both-modality, so the clamp is untested by this battery.

---

## Arm F — multishot cost (C1) ✅ FREE, and it doubles as the 121f NULL FLOOR

`--arm-prompt <arm>=<text>` (new): two arms identical in every axis except prompt text.
Receipts: `probes/bench_e2e_armF-c1-multishot-cost_20260815-141705.{md,json}`.

| arm | median | range | spread |
|---|---|---|---|
| f-single | 83.7 s | 66.7–100.4 | 33.7 |
| f-multishot | 92.9 s | 80.1–102.9 | 22.8 |

**Δmed +9.3 s — NOISE (sign flips between block sets).**

🔑 **C1 answered: multishot prompting costs NOTHING at inference.** This was mechanically expected —
prompts pad to 1024 tokens, so both arms do identical work — which is exactly why the arm is
valuable: a MEASURABLE result would have indicated a harness artifact, not a model finding. Combined
with the oracle's C1 (slots are the real multishot surface, +10.7% for 3 slots), the picture is:
**shot-holding is learned behaviour, not an inference cost.**

🔑 **It is also a genuine NULL PAIR at 121f** — identical compute, so any delta is pure noise. That
retires an inference: the "noise scales with geometry" section of `BENCH.md` had extrapolated the
121f floor from cold-start artifacts. Measured: **Δmed ≤ ~10 s is NOISE at 121f, with within-arm
spreads to ~34 s.** See `BENCH.md` for the fresh-boot ramp finding this produced.

⚠️ `output f-multishot vs f-single: cos=0.877864 (divergent-by-design)` — correct, and only because
the identity predicate now accounts for prompt. Different prompts = different conditioning = different
video. This is the FOURTH axis added to that predicate (`model`, `dfrRounds`, `env`, now prompt).

---

## Arm B (dfr=2) — C2's LAST OPEN REGIME, CLOSED ❌ no crossover exists in the envelope

Fresh boot + **3 burn-in generations** before measuring (see `BENCH.md`), 704×512×**481f@96**
matched output, 2 ABBA blocks × (1 warmup + 2 measured). Receipts:
`probes/bench_e2e_armB-dfr2-longclip_20260816-043520.{md,json}`, log `probes/armB-dfr2-longclip.out`.

Native generates 481f@96 directly (**21,472 video tokens**); DFR-2 generates the same 121f@24 base
as r=1 and densifies **twice**.

| arm | median | range | spread | peak (⚠️ UNEVICTED) |
|---|---|---|---|---|
| t25-native | **533.1 s** | 521.8–537.0 | 15.1 (2.8%) | 66.01 GB |
| t25-dfr2 | **936.2 s** | 933.6–940.8 | **7.2 (0.8%)** | 67.15 GB |

**Δmed +403.0 s > spread 15.1 s — MEASURABLE. ×1.756 native.**

### 🔑 The regime is closed, not merely unfavourable

`C2-dfr.md` left exactly one escape hatch: *"the one regime that could still favour them is longer
clips, where native high-fps generation runs out of sequence length."* **481f@96 IS
`max128.maxFrames` — the longest clip any shipping tier supports** — and native handles it
comfortably (66.01 GB peak, 533 s, spread 2.8%). It does not run out of anything.

**There is no supported geometry where temporal rounds win on time.** The hypothetical crossover
would need a clip longer than the product can request.

### Pre-registration scored honestly

Predicted before the run: *"DFR-2 ≈ 1000 s vs native ≈ 500–600 s → ~×1.9, losing by MORE than r=1,
because rounds compound superlinearly while native scales ~linearly in tokens."*

| | predicted | measured | verdict |
|---|---|---|---|
| direction (worse than r=1) | worse | ×1.756 vs ×1.678 | ✅ correct |
| DFR-2 wall-clock | ~1000 s | 936.2 s | ⚠️ over by 7% |
| ratio | ~×1.9 | ×1.756 | ⚠️ over |
| mechanism | rounds compound, native ~linear | native ×2.224 (241f→481f), DFR ×2.328 (r1→r2) | ❌ **mostly wrong** |

🔑 **The mechanism claim does not survive.** Native did NOT scale linearly — it more than doubled
(×2.224) for 2× the tokens, essentially matching DFR's ×2.328. DFR grows *slightly* faster, which is
why the ratio crept from 1.678 → 1.756, but that is a **+4.6% drift per round**, not the compounding
I described. Recorded because the directional call being right does not make the reasoning right,
and the reasoning is what would have been reused.

### 🔑 The real finding: a stable densification tax

| configuration | native | densified | ratio |
|---|---|---|---|
| 2.3 hand-wired temporal-x2 @241f | 224.5 s | 357.4 s | **×1.592** |
| 2.5 trained-for DFR r=1 @241f | 239.7 s | 402.1 s | **×1.678** |
| 2.5 trained-for DFR r=2 @481f | 533.1 s | 936.2 s | **×1.756** |

**Densification costs ~1.6–1.8× native across two generations, two mechanisms (hand-wired vs
trained-for) and two round counts.** That stability is a stronger result than any single ratio: the
penalty is a property of densification itself, not of a particular implementation or clip length.

### Method notes

✅ **Burn-in works, and the receipt shows it.** Both warmups ran `fair→fair` (not `nominal→fair`),
and spreads were **0.8–2.8%** against the 50% within-arm swing a cold-start session produced at 121f
(AB-R-0067). Three throwaway generations before measuring is cheap and decisive.
✅ **Ordering excluded again.** DFR-2 ran second in block 0 and FIRST in block 1; its early-slot time
(933.6 s) sits within ~7 s of its late-slot times (940.8 / 934.1), against a 403 s arm gap.
✅ **Memory half stays clean:** DFR-2 costs **+1.14 GB** peak for TWO rounds (67.15 vs 66.01),
consistent with r=1's +0.59 GB. The historical "+31 GB for DFR" remains fully explained as the DiT
sitting resident under the decoder.
🔑 **Peak is FLAT to 481f** — native reads 66.01 GB at 481f@96 vs ~64–65 GB at 241f@48 for 2× the
tokens. This is also the **first 2.5 measurement at `max128.maxFrames`**, previously flagged as
extrapolated (AB-R-0041). ⚠️ UNEVICTED, so it is not the shipping-regime figure — but it bounds it.
⚠️ `cos=0.213226 (divergent-by-design)` — two rounds of densification diverge the sample far more
than one (r=1 read 0.769). Correct and expected; not a quality signal.

---

## Arm C — decode triangle (C3) ❌ DiffVAE costs ×10–12 time and +23.9 GB for LOWER fidelity

`RunLTX2 --decode-triangle` (combined: time + fidelity) and `--decode-triangle-arm conv|diffvae`
(one decoder per process: memory). Real clip `hard-cut-14s.mp4` at 704×512×25f, read **in place off
Satechi** (AB-L-0041). Logs: `probes/armC-decode-triangle{,-reversed,-perarm}.out`.

**Decoder-isolated by construction:** the clip is VAE-encoded ONCE and both decoders receive
byte-identical latents, then each is scored against the **real source** — not against each other,
since two decoders can disagree while one is simply right.

### Cost (per-process; reproduced twice per arm)

| decoder | decode | attributable memory |
|---|---|---|
| conv | **1.76 / 2.03 s** | **2.34 GB** |
| DiffVAE | **20.46 / 21.80 s** | **26.26 GB** |

**×10–12 time, +23.9 GB.**

### Fidelity vs the REAL source (identical under both decoder orderings)

| decoder | PSNR | SSIM |
|---|---|---|
| conv | **46.57 dB** | **0.9994** |
| DiffVAE | 35.91 dB | 0.9923 |

**conv is 10.66 dB more faithful.** Reproduces the oracle's direction (AB-R-0013: 42.6 vs 36.8 dB)
on a different clip and implementation; absolute values differ because the corpus differs.

🔑 **The stochasticity objection is measured, not waved away.** DiffVAE is 1-step-x0 from a noise
canvas, so "you just drew a bad canvas" is the obvious rebuttal to any metric. A second draw with a
different key moves **PSNR +0.02 dB and SSIM −0.0000** — the conv-vs-DiffVAE gap is **~500× the
draw-to-draw spread**. The verdict is not a draw artifact.

### 🚨 A retracted number, and the control that caught it

The first run reported *"DiffVAE costs +26.43 GB peak"* from a COMBINED run. **That was invalid.**
`PhysSampler` reads PROCESS phys, so whichever decoder runs second carries the first's resident
weights and the accumulated MLX pool:

| order | diffvae peak | conv peak | delta it "showed" |
|---|---|---|---|
| conv first | 41.03 GB | 14.61 GB | **+26.43 GB** |
| diffvae first | 38.62 GB | **41.18 GB** | **−2.56 GB** |

The second arm read ~41 GB either way. Fixed with **one decoder per process**
(`--decode-triangle-arm`), which reproduces exactly (2.34 / 26.26 GB attributable, twice each).

⚠️ **The retracted number was numerically CLOSE to the truth** (+26.43 vs the real +23.9) — and that
is the trap. It was close by accident of ordering, and the identical method yielded −2.56 GB when
reversed. **A roughly-right number from an invalid method is still a wrong finding**, and only the
reversal control distinguished them. Same family as AB-R-0041 (post-run "resident floor" measuring
mmap retention) and AB-L-0041 (an A/B where argument order was the entire effect) — three instances
of *a process-level metric attributed to a per-component cause*.

### Verdict

**On cost and fidelity, DiffVAE loses decisively.** §V's framing — *"DiffVAE decode is HEAVIER by
design … conv stays the efficiency default; DiffVAE is a quality tier"* — is supported, and the
price of that tier is now quantified: **~11× decode time and ~24 GB**. Since decode is 8–32% of a
run, defaulting to DiffVAE would roughly double a run at best.

C3's claim ("sharper faces, textures, on-screen text") therefore rests **entirely** on a perceptual
win that no measurement here or in the oracle supports — and one specific sub-claim, on-screen text,
was already refuted: **identical under Vision OCR** (AB-R-0014). ⚠️ PSNR ≠ perceptual remains true;
a generative decoder may trade fidelity for apparent sharpness. That case would need an operator
A/B, and it would have to be worth ×11 time and 24 GB.

🔑 **DiffVAE is ported and gated but deliberately NOT wired into `LTX2Pipeline`. This measurement
says that is the right call** — wiring it should wait on a perceptual case, not precede one.
