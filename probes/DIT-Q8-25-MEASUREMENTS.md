# LTX-2.5 q8 DiT — the two measurements AB-R-0038 owed

Both were listed as **owed before any lower tier is declared**. Raw logs are the `.out` files
beside this one; the harness is `probes/dit25-quant-probe.sh` + `RunLTX2 --dit25-probe{,-compare}`.
The probe outputs themselves (`probes/dit25probe*/*.safetensors`) are forward-outputs of the LTX-2
weights — weight-derivatives, gitignored like `parity/goldens/`.

⚠️ **`dit25-quant-probe.out` was produced by the FIRST version of the driver, which had two
defects** — kept as the raw record, and both are fixed in the committed harness. `seq $((i+1))
$REPS` is wrong on macOS: **BSD `seq` counts DOWN when start > end**, so the last iteration of each
block emitted a nonexistent index (one crashed invocation, visible in the log) and a **self-pair**,
and a file compared with itself reports `bitExact=yes` unconditionally — a verdict that cannot
fail, which is this project's oldest trap wearing a new hat. The driver now enumerates pairs
explicitly and `--dit25-probe-compare` refuses a self-comparison outright. **None of the numbers
below come from those two lines**; the three real pairs per block ran correctly, and the
later 2.3 and small-N runs used the fixed harness throughout.

---

## 1. Footprint AT 121f — the 25.03 GB number was 9-FRAME

`LTX_CACHE_LIMIT_GB=2 LTX_EVICT_DIT=1 RunLTX2 --mem-bench25 <tree> 704 512 121`
(unprofiled / evicted / capped — the regime the engine actually runs).

| 704×512 PEAK | 9f | 121f | **161f** | 9f → 161f |
|---|---|---|---|---|
| bf16 DiT | 42.54 (AB-R-0036) | 43.11 | **43.26** | +0.72 GB (+1.7%) |
| **q8 DiT** | 25.03 (AB-R-0037) | 25.60 | **25.55** | **+0.52 GB (+2.1%)** |

**q8 is flat across frame count in exactly the way bf16 is.** The inference AB-R-0038 rested on now
has a measurement under it — and 161f is `standard64.maxFrames`, so the declaration covers that
profile's **full frame range** rather than a point inside it. ⚠️ `max128.maxFrames` is **481** and
remains UNMEASURED on 2.5; flatness over 9→161f is strong evidence but 481f is a 3× extension, and
this document exists because that class of extrapolation was worth replacing with a measurement.

✅ **CONTROL — the bf16 arm was re-run at 121f on the SAME BINARY**, in the same session, before
the binary was touched again. It read **43.11 GB against AB-R-0036's 43.10 GB**: the instrument
reproduces a cross-session figure to 0.01 GB, and it moves **−17.51 GB (−40.6%)** between arms at
this identical geometry. So the +0.57 GB frame-count flatness is a real flat, not a metric that
cannot discriminate — this project's recurring trap, and the reason AB-R-0036 demanded a
must-fail control for its own flat curve.

| same binary | PEAK 121f | PEAK 161f | phys-after-load 121f | 161f | "resident floor" 121f | 161f |
|---|---|---|---|---|---|---|
| bf16 | 43.11 | 43.26 | 39.92 | 39.89 | 11.38 | **2.34** |
| q8 | 25.60 | 25.55 | 22.09 | 22.09 | 11.39 | **2.43** |

🚨 **The "resident floor" line is not a resident floor — DO NOT DECLARE FROM IT.** It swings
**~9 GB between two adjacent geometries, on both arms at once**, while `phys-after-load` moves
0.00–0.03 GB over the same change. It is a single sample taken post-warmup-run and
post-`clearCache`, so under eviction it measures whatever mmap pages the OS happened to retain.
Two independent tells: at 121f it reads the *same* ~11.4 GB for a 38 GB and a 20 GB checkpoint, and
at 161f it drops to ~2.4 GB for both.

⚠️ **The harness actively invites this mistake.** Its `DECLARE → residentBytes ≈ …` line is computed
from that floor, so at 161f it recommends **2.34 GB resident for the 40 GB bf16 DiT** — an ~17×
under-declaration the memory governor would act on. `MLXLTX25Package` declares from
**`phys-after-load`** (the quantity that is stable across geometry) and **PEAK**; the derived
`activation = peak − floor` column is unusable at both geometries.

---

## 2. Per-forward cosine, repeated — the SPREAD is ZERO

`RunLTX2 --dit25-probe <arm> <out>` — **one arm per process, 3 processes per arm.** That is the
design, not an accident of scripting: 2.3's `--dit-q8-gate` spread appeared ACROSS process
invocations, while in-process the int8 path *usually* self-repeats bit-exactly. A gate loading both
arms in one process would have reported a spread of exactly zero and proved nothing.

⟲ **CORRECTED 2026-08-16 (AB-T-0042).** This paragraph previously said the int8 path "self-repeats
bit-exactly (the HV2 streaming acceptance memcmp depends on that)" — stated unconditionally, and
false. AB-R-0052 measured the 2.3 q8 **resident** reference failing to self-repeat **in 3 of 7
in-process runs** (cos 0.9971–1.0000). The one-arm-per-process design above is unaffected and still
right — a same-process comparison genuinely cannot see cross-process spread — but the reason given
for it was wrong, and the HV2 acceptance does **not** depend on in-process bit-exactness: the
streaming gate measures the resident self-repeat each run and picks its bar from that
(`StreamGates.swift:321`), rather than assuming it. See `../CLAUDE.md`'s streaming row.

Inputs are in-distribution and **saved into every output file**, so "both arms saw the same input"
is a *checked* control (`inputs bit-identical ✅`), not an assumption about RNG: real Gemma-4
49-state golden → real 2.5 connector for text; `Positions.video/audio` at the stage-2 grid;
`firstLatentFrameKeyframesMask` exactly as `t2vTwoStage` builds it; unit-variance latent from a
fixed key at the stage-2 entry sigma. N = 5632 video / 126 audio tokens, σ = 0.909375.

### Run-to-run (3 processes per arm, all pairs)

| arm | video maxAbs | verdict |
|---|---|---|
| 2.5 bf16 vs itself | 0.000000 | **bit-exact** |
| 2.5 q8 vs itself | 0.000000 | **bit-exact** |
| 2.3 bf16 vs itself | 0.000000 | **bit-exact** |
| 2.3 q8 vs itself | 0.000000 | **bit-exact** |

**Spread = 0 on every arm.** 2.3's documented int8 nondeterminism does not reproduce here, on
either checkpoint. The answer to "report the spread, not one number" is that at this geometry the
number IS a point.

⚠️ **Read `maxAbs`, not the cosine, for identity.** Bit-identical arrays print cosine `1.00000012`
and `0.99999994` — fp rounding in the reduction. A cosine-only report would have shown a spread
that does not exist.

### The quant delta, and a same-metric comparison point

| q8 vs bf16 (all 9 cross-pairs identical to 8 dp) | video cosine | angular error 1−cos | video maxAbs | audio cosine |
|---|---|---|---|---|
| **LTX-2.3** | 0.99237692 | 7.62e-3 | 3.342 | 0.99992329 |
| **LTX-2.5** | **0.99824488** | **1.76e-3** | 0.966 | 0.99981642 |

🔑 **2.5's q8 DiT sits 4.3× closer to its own bf16 than 2.3's q8 DiT does to its own** — same code,
same N, same σ, same quant recipe (transformer_blocks-only int8 g64; the 2.5 recipe was read off
the 2.3 checkpoint precisely so this comparison would be legitimate).

The 2.3 arm exists because **0.998245 has no meaning on its own.** Read against the quant-ladder
doctrine's "q8 ≈ bf16, single-forward cosine ~0.9999" it looks like a miss; but that figure was not
measured this way, and comparing to it would be comparing instruments. Measured the same way, the
precedent everyone was worried about is the *worse* of the two.

⚠️ **This does NOT promote q8-2.5 to a faithful tier.** AB-R-0038's classification stands on the
blinded A/B: a per-forward delta of 1.76e-3 still compounds over a distilled trajectory into a
different sample, and the operator judged those samples equally good. **Lean tier. Anything needing
bit-reproducibility against bf16 stays bf16.**

---

## 3. 🚨 Side finding — int8 nondeterminism is a SMALL-N effect, and it INVERTS the documented hypothesis

The parent CLAUDE.md's leading hypothesis for the flaky `--dit-q8-gate` is *"nondeterminism in the
int8 path that only manifests inside the full 48-block graph, not on a single op."* **The "full
graph" half of that is backwards.** Measured directly — same harness, same comparison structure,
only the geometry changed (`probes/dit-quant-probe-smalln.out`, 448×320×9 → 280 video / 9 audio
tokens, vs 5632 / 126 above):

| arm | self-repeat @ **N=280** | self-repeat @ **N=5632** |
|---|---|---|
| 2.3 bf16 | bit-exact (maxAbs 0.000000) | bit-exact |
| 2.5 bf16 | bit-exact (maxAbs 0.000000) | bit-exact |
| **2.3 q8** | **DIFFERS** — maxAbs 0.047 / 0.836 / 0.863 | **bit-exact** |
| **2.5 q8** | **DIFFERS** — maxAbs 0.597 / 1.189 / 1.267 | **bit-exact** |

Both geometries are the full 48-block int8 graph. The small one varies and the large one does not,
on **both** checkpoints. bf16 is bit-exact at both sizes, so this is the int8 path specifically —
consistent with the already-ruled-out finding that a single `quantizedMatmul` is bit-identical over
50 repeats (the effect is in the graph, just not the way "full graph" implied).

Corroborating, five consecutive `--dit-q8-gate` runs on this binary (N=192,
`probes/dit-q8-gate-repeat.out`):

    0.999904 / 0.999915 / 0.999895 / 0.999907 / 0.999869    → spread 4.6e-5, 5/5 PASS

Still nondeterministic, but ~76× tighter than the documented 0.996388–0.999915 (3.5e-3).

🔑 **Operational consequence: the flakiness afflicts the GATES, not the shipping path.** Gate
fixtures are tiny (N=192); shipping geometries are large (N=5632 at `standard64`'s envelope, where
both q8 checkpoints are bit-exact run to run). The parent CLAUDE.md's warning that *"q8 output may
not be run-to-run reproducible"* was extrapolated from a small-N gate and does not hold at the
sizes the engine actually runs.

🔑 **And it explains why a single small-N q8 cosine is worthless.** At N=280 the q8-vs-bf16 delta
is *swamped* by q8's own run-to-run noise — 2.5 q8-vs-bf16 reads 0.99519 / 0.99867 / 0.99713
depending purely on which q8 process you compare against, while q8-vs-q8 reads 0.99275–0.99649.
**The self-noise is larger than the quant signal.** A gate that reports one number there is
reporting a draw, which is exactly what "0.9999 PASS" was in June.

⚠️ Not claimed: the mechanism (kernel/reduction selection at small N is the obvious suspect, and is
unproven), nor that the historical 3.5e-3 spread was wrong — the old condition was not re-run. What
is claimed is what was measured on this binary today.

---

## 4. Tier scope — AB-D-0014's two-tier decision STANDS; AB-R-0038's consequence was wrong

AB-R-0038 concluded *"`balanced32` is now in range on the measured number and `compact24` is ~1 GB
away."* **Both halves are wrong**, and the error is that 25.03 GB was read against raw RAM sizes
rather than against the governor budgets. The acceptance rule is
`LOW-TIER-PLAN.md:148` — **"every declared profile's measured stage-max ≤ 0.7× its tier"**
(`max128` at 0.85×, per `LTX2Configuration.swift:279`).

| tier | budget | q8 DiT weight floor | q8 measured peak | verdict |
|---|---|---|---|---|
| compact24 | **16.8** GB | 22.09 | 25.60 | ❌ out by 8.8 GB — the *weights alone* are 5.3 GB over budget |
| balanced32 | **22.4** GB | 22.09 | 25.60 | ❌ out — the weight floor leaves **0.31 GB** for all activation |
| standard64 | **44.8** GB | 22.09 | 25.60 | ✅ **57% of budget** (bf16 was 43.11 = 96%) |
| max128 | 108.8 GB | 22.09 | 25.60 | ✅ ample |

🔑 **The lower tiers are blocked by a floor that no envelope reduction can move.** `balanced32`
would run a smaller envelope (576×320, one-stage) than the 704×512×121 measured here, so its
activation term is smaller — but peak cannot go below the resident weights, and those are 20.60 GB
on disk / 22.09 GB as process phys-after-load against a 22.4 GB budget. There is no geometry that
fixes that. `compact24` is not "~1 GB away"; it is 8.8 GB away.

✅ **What the q8 DiT actually buys — and it is worth more than the lower tiers were.**
`standard64`'s declared quant has always been `.int8` (`LTX2Configuration.swift:62`), but no
quantized 2.5 DiT existed, so 2.5 has been running bf16 there at **43.11 GB against a 44.8 GB
budget — 96%, i.e. 1.69 GB of headroom at the *tested* envelope**, with the profile's own envelope
allowing 161f (untested, and above the 121f measured here). q8 takes that to **25.60 GB / 57%**.
The honest framing is not "q8 revives low tiers" but **"q8 makes the tier 2.5 already ships on
actually safe"**, and gives `max128` room it did not have.

⚠️ **Still owed before `balanced32`/`compact24` can be closed for good rather than deferred:** a
`--mem-bench25` run at each low tier's OWN envelope and one-stage path. The harness currently
forces `t2vTwoStage`, so it cannot express `oneStage: true` — that is a harness gap, not a result.
The conclusion above does not depend on it (the floor argument is envelope-independent), but the
formal "measured stage-max per profile" acceptance line does.
