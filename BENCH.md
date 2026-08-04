# BENCH.md — the end-to-end metrics harness (`RunLTX2 --bench-e2e`)

**Added 2026-08-03.** The measurement *protocol* as an executable, so it stops being re-learned by
hand. What exists already, and what none of it can do:

| tool | what it is | why it can't answer "is arm B faster than arm A" |
|---|---|---|
| `--speed-bench` / `--mem-bench` | single-arm: warmup + ONE measured run / ONE footprint declaration | no repeats, no second arm |
| `--vae-decode-bench` | stock-vs-pruna decoder A/B, median-of-3 | decoder-only, not end-to-end; no alternation |
| `LTX_TESTING/portable/run-matrix.sh` + `RunReport` + `analyze-results.py` | the app-side campaign driver: fixed legs R1–R4 (back-to-back drift repeats) + S1 (profiled split), JSON reports, ship/pull thresholds | one *configuration* per campaign — legs are repeats of it, not competing arms; verdicts are absolute (peak ≤ 16.8 GB, spos ≤ 20) not comparative; needs the app |

The missing piece is the comparative protocol: desktop run-to-run totals vary **±15 s**
(SPEED-PLAN S8: the Pruna decoder's ~1.7 s e2e saving was unmeasurable from a single pair), and
session drift once manufactured a **+48%** phantom regression (BLOCKSTREAM-EXPANSION-EVAL §1.3 —
in-regime it was ≈+7%). `--bench-e2e` is that piece: multi-arm, alternated, repeated, receipted,
with an explicit noise verdict. It reuses the fleet's primitives rather than re-deriving them —
`PhysSampler` + `prewarmFiles` (main.swift:727/753), the mem-bench floor/peak recipe
(main.swift:798–821), the quant-band comparison doctrine (StreamGates.swift:311–320), and the
`arms.sh` cooldown/bracket pattern (mlxengine-todo/probes/private/v12b3g/arms.sh).

## What one invocation does

```
RunLTX2 --bench-e2e \
  --arm base:quant=bf16 --arm lean:quant=bf16,decoder=pruna \
  --blocks 2 --runs 2 --cooldown 45 --size 704x512 --frames 24 [--two-stage] [--label pruna-ab]
```

Per **block** (arms alternate in ABBA order across blocks): apply the arm's env + cache cap →
**prewarm** exactly the files that arm faults → **load** (timed) → **one excluded warmup
generation** (shape-specific kernel compile lands there) → **`--runs` measured generations**
(same seed) → drop pipeline + `clearCache` → record the **post-drop floor** (ratchet detector) →
**cooldown**. Every run row: wall, sampled `phys_footprint` high-water (25 ms), MLX active/cache,
thermal state before/after, output mean/std, cosine vs the arm's first measured output.

Receipts land in `probes/bench_e2e_<label>_<stamp>.{md,json}` with the env fingerprint
(host, RAM, OS, git SHA±dirty, geometry, seed, prompt, protocol).

## The doctrine it encodes (with receipts)

| Rule | Why (receipt) |
|---|---|
| **MLX_PROFILE force-disabled** | profiler spans force per-span `eval`, breaking fusion — stock decode 3468 → 5530 ms, compressing a real 2× decoder gap to ~1.2× (S8 trap b). Profile for the *split*, bench for the *ratio*. |
| **Prewarm per arm, per block** | unwarmed weights cold-fault off the archive mid-run — the A/B measures paging, not the arm (S8 trap a; `--speed-bench`'s pruna note). |
| **Warmup run excluded** | first forward pays single-threaded Metal kernel compile per shape — 162 s cold vs 1.5 s steady (PROFILING.md §1); and never profile the first turn generally. |
| **ABBA block order + repeats** | "repeat the arms, in both orders — sign-flip = noise" (§1.3's compounding rule). A bracketed baseline cannot rescue an unrepeated treatment. |
| **Cooldown + thermal column** | thermal drift lands inside whichever arm runs hot; recording state makes drift attributable instead of invisible. |
| **`phys_footprint`, not `Memory.peakMemory`** | MLX-peak counts cumulative allocations and under-reads the admission basis ~2.5–2.9× vs the OS figure (mem-bench header; the Wan profiler lesson). Peak is a mid-denoise transient — sampled, not phase-boundary. |
| **Post-drop floor per block** | catches residency ratchets (the repeated-fallback phys ratchet class) — the floor should return to ~framework baseline every block. |
| **Same-seed outputs compared, never gated on cross-quant** | determinism doctrine: only **bf16** same-config arms carry "expect ≈1" (measured exactly deterministic, 5-run spread 0). q8 is **nondeterministic** (int8 graph-order sensitivity, spread ~3.5e-3 — the flaky-gate saga): its cross-arm cosine is only readable against its own intra-arm spread, which the per-run `cos(arm-first)` column provides. q4 and pruna **diverge by design** — recorded, never failed. |

## Invocation discipline

Run the **bare binary with output redirected, never piped** (`arms.sh` doctrine — a pipeline's exit
code is the last stage's, and `libc++abi` aborts lose block-buffered stdout):

```
.build/out/Products/Debug/RunLTX2 --bench-e2e … > probes/bench_<label>.log 2>&1
```

A mid-run death keeps its evidence anyway: every run row is appended to
`probes/bench_e2e_<label>_<stamp>.partial.jsonl` as it completes (deleted on clean exit, superseded
by the final receipts).

> ✅ **First catch (2026-08-03, this harness's own smoke run): ISSUES.md I9 — RESOLVED the same
> evening.** The GPU-watchdog abort was **weight page-in latency from the USB `DEV_ARCHIVE` volume**,
> not the harness, not the int8 connector, and not the OS seed. MLX safetensors arrays are lazy, so
> the weight bytes are pulled **inside the first generation's Metal command buffers**; when the
> working set is large enough that the OS can't serve them from page cache, they fall back to a
> ~250–475 MB/s USB device mid-command-buffer and the watchdog fires. **bf16 is the only quant whose
> working set crosses that line** (52.6 GB @ 9f, 66.2 GB @ 121f) — int8 and int4 pass on the same
> volume. Every row of the original ledger happened to be bf16, which is why it read as universal.
>
> **Run benches off fast storage** and the whole class disappears (bf16 3/3 green from a PCI-E SSD,
> including the 121f shape SPEED-PLAN S6 had recorded as `no datum`).
>
> ✅ **Permanently fixed the same evening: the weight trees were RELOCATED off the USB archive** to
> `/Volumes/Satechi/Models/{dgrauet,ltx-lora-cache,mlx-community}` — SHA-256 verified 83/83 files,
> DEV_ARCHIVE originals then deleted so stale paths fail loudly. All gate/dumper path literals were
> repointed; `LTX_BENCH_BASE`/`LTX_BENCH_GEMMA` remain for pointing a bench at another tree. There
> is no staging step any more — the defaults are already fast. See ISSUES.md I9.
>
> ⚠️ **Lasting caveat: state the storage volume in every receipt.** Every CLI number on this box
> from before 2026-08-03 was USB-bound and carried weight page-in inside its first generation
> (prewarm 62–101 s from USB vs 3.8–9.0 s from the SSD), so those run-1 legs are **not comparable**
> to anything measured now.

## Verdict semantics

For each arm vs the first arm: `Δmedian` of measured walls, against `spread = max(range_A, range_B)`,
plus per-block sign consistency. **NOISE** if the sign flips between block sets or `|Δmed| ≤ spread`;
**MEASURABLE** otherwise. A **null run** (`--arm A:quant=bf16 --arm B:quant=bf16`, the default when
no arms are given) measures the protocol's own noise floor — run it once per machine/geometry before
trusting any verdict at that geometry, and re-run it when the toolchain or OS changes.

## The machine's noise floor (taken 2026-08-03, post-I9, weights on the Satechi store)

`probes/bench_e2e_nullfloor-ssd_20260804-010416.{md,json}` — identical bf16 arms, default protocol
(blocks=2 ABBA, runs=2, cooldown 45 s), 704×512×9f, everything read from the PCI-E store:

- **A median 20.1 s (16.6–22.8) · B median 28.6 s (26.6–30.8) → the null pair read
  "MEASURABLE +8.5 s" — a FALSE POSITIVE, and the receipt shows the mechanism:** the box stepped
  `nominal → fair` inside block 0 (A's first two runs were the session's only nominal runs at
  16.6–17.8 s; everything after sat 22–31 s, recovering monotonically). ABBA cancels *symmetric*
  drift; a **one-way thermal step early in the session aliases into the arm delta**, and the
  sign-flip check is blind to it (both blocks land the same sign).
- **Operative rule until the verdict logic grows a drift detector: at this geometry on this box,
  treat any Δmedian ≤ ~8.5 s (and any spread ≤ ~6 s) as NOISE regardless of the printed verdict** —
  that is precisely what the null run is for. Levers claiming less than that need more blocks
  (4+), longer cooldowns, or a thermally-quiet box.
- Everything else validated clean: **12/12 bf16 generations from the SSD store, zero watchdog
  aborts** (the I9 fix end-to-end); intra-arm `cos(first)=1.000000` on all 8 measured runs and
  cross-arm `cos=1.000000 / maxAbs=0` (bf16 bit-determinism reconfirmed through 4 separate
  pipeline loads); block floors 0.68–0.94 GB (no residency ratchet); prewarm from the store
  **10.0 s cold → 2.2–3.7 s warm** (vs 203 s first-touch on the old USB volume); peak
  52.5–52.9 GB, matching the I9 working-set analysis.
- v2 verdict-logic items this run motivates: a monotone-**drift detector** (regress wall on global
  run index; flag when |slope·session| ≳ Δmed), **thermal-state stratification** (compare only
  matching states), and defaulting `--blocks` to 4 when a session starts `nominal`.

**Long-geometry addendum (from the 121f ladder session, `bench_e2e_ladder-pruna-121f_20260804-012008`):**
at sustained-load geometries the dominant confound moves **inside the block** — every arm showed
run 2 ≈ +10–15% over run 1, and warmups (the coolest slot, right after cooldown) often ran *faster*
than the measured runs that followed them. That signature = intra-block heat ramp; it inflates
per-arm spread (bf16: 37.3 s ≈ 21% of median) and pushes real deltas under the NOISE bar.
**For geometries where a generation runs minutes, prefer `--runs 1 --blocks 4` over
`--runs 2 --blocks 2`** — same sample count, but every measured run sits in the same
first-run-after-cooldown thermal slot. Also note which regime a session measured (S6 July: 5-min
cooldowns ≈ *cooled/interactive*; 60 s cooldowns + multi-run blocks ≈ *sustained/batch* — the two
regimes rank quants differently, see SPEED-PLAN S6).

## What this harness is NOT

- **Not a stage profiler.** Totals are end-to-end by design. For the split, run the same arm once
  under `MLX_PROFILE=csv` separately (PROFILING.md) and never compare those timings across variants.
- **Not a quality judge.** Cross-arm cosine is a fingerprint, not an oracle. Perceptual judgment
  of divergent-by-design arms (q4, pruna) stays a human/no-reference-metric job (determinism
  doctrine; the S8 "perceptual A/B on a real clip" item remains open).
- **Not the streaming stall meter.** `--stream-auto-gate`/`--stream-budget-gate` own stall
  receipts; streaming arms here (via `env.LTX_STREAM_GRANULES=…`) get time/memory/output rows only.
  ⚠️ Streaming arms also carry the one-rung-per-process fallback ratchet — prefer one streaming arm
  per invocation until the harness grows process isolation.

## Open items (v2 candidates)

- `--isolate`: re-exec each block as a child process (kills cross-arm state bleed entirely; needed
  before multi-arm streaming benches — the fallback phys ratchet is one-rung-per-process).
- Stage-split capture: parse `MLX_PROFILE_CSV` from a dedicated split run appended to the receipt,
  clearly marked as profiled-and-inflated. The rollup already exists —
  `RunReport.stageSplit(fromCSV:)` in `LTXVideoTesting/RunReport.swift:178` — reuse it.
- Schema convergence: the JSON receipt should adopt/extend the app's `RunReport` schema
  (`RunReport.swift:97`, incl. `MachineFingerprint` with lowPowerMode + external-volume flags)
  rather than keep a parallel shape; `analyze-results.py` could then ingest both.
- MP4 encode arm: `FrameCodec` lives in `MLXLTX2`; the encode stage has its own deadlock history
  (PROFILING.md §2) and belongs in the receipt eventually.
- Low Power Mode preflight (run-matrix.sh checks it; the harness only records thermal state).
