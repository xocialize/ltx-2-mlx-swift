# Swift claims bench matrix (session task #7) — arm status and results

The six-arm matrix is specified in `LTX_TESTING/LTX25-PORT-PLAN.md` §V ("Bench matrix"), tracked by
**AB-T-0005**. This file records what the **Swift** side can run and what it found. Protocol for
every arm: matched output spec, bf16, stock decoder, cache 2 GB, ABBA blocks ≥2, ≥2 runs/arm/block,
30 s cooldowns, unprofiled totals, thermal strata logged.

## Arm reachability (scoped 2026-08-14)

| arm | claim | needs | Swift status |
|---|---|---|---|
| **A** perf parity 2.3 vs 2.5 | §V item 1 | model axis on `--bench-e2e` | ✅ **RUN — see below** |
| **B** DFR efficiency | C2 | `DFRPipeline` | ✅ ported + gated — runnable |
| **C** decode triangle | C3 | **DiffVAE decoder** | ❌ **NOT PORTED** |
| **D** duration honesty | C6 | **duration head** | ❌ **NOT PORTED** |
| **E** enhancer overhead | C5 | `GemmaTextGenerator` | ✅ exists — runnable |
| **F** multishot cost | C1 | keyframe slots | ✅ ported + gated — runnable |

⚠️ **Task #7 is not fully executable as written.** Arms C and D need Swift ports that do not exist.
Both weight files ship in the 2.5 tree — `vae_diffusion_decoder.safetensors` and
`duration_head.safetensors` — and neither is referenced anywhere in `Sources/LTX2`. Both components
ARE implemented in the Python-MLX oracle, so this is a port, not a research task. Spun out as its
own work item; do not treat arms C/D as blocked-on-measurement.

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
