# RESUME — next session starts here

> ✅ **UPDATE 2026-08-16 — the `dfr=2` run below is DONE (AB-R-0073): C2's last open regime is
> CLOSED.** DFR r=2 costs **×1.756** native at 481f@96 (`max128.maxFrames`); no supported geometry
> favours temporal rounds. **Remaining arms: C (decode triangle) and E (enhancer overhead).**
> The burn-in protocol below is VALIDATED — spreads were 0.8–2.8% vs a 50% cold-start swing — so
> keep using it for any long timing run.

Everything is committed, pushed and gate-green. This file exists so the next session does not
rediscover the protocol or re-derive what is already measured.

## The next run: bench matrix `dfr=2` (the last open C2 regime)

The ONLY regime `ltx-2-mlx/docs/claims/C2-dfr.md` leaves open for temporal rounds: *"longer clips,
where native high-fps generation runs out of sequence length."* Arm B tested r=1 at 241f@48, where
native is comfortable, and DFR lost (×1.68). r=2 is where the comparison could invert.

```bash
cd LTX_DEV/ltx-2-mlx-swift
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
xcrun swift build --product RunLTX2

# 1. ALWAYS dry-run first — resolves every arm, loads no weights, touches no GPU
./.build/debug/RunLTX2 --bench-e2e \
  --arm t25-native:model=ltx25,quant=bf16,cache=2 \
  --arm t25-dfr2:model=ltx25,quant=bf16,cache=2,dfr=2 \
  --size 704x512 --frames 481 --fps 96 --two-stage --blocks 2 --runs 2 --dry-run

# expected: t25-native request=481f@96 → delivers 481f@96
#           t25-dfr2   request=121f@24 → delivers 481f@96
```

### 🚨 BURN THE BOX IN FIRST — this is the step that is easy to skip after a reboot

A fresh boot is right for MEMORY safety (this is the largest working set yet, and I9 is a
paging-triggered **abort**, not a slowdown) but it is the **worst case for timing stability**: a
cold box warms monotonically through the session, and **ABBA cancels symmetric drift, not a one-way
ramp** (AB-R-0067; measured at 121f, one arm spanned 66.7→100.4 s on identical work).

**Run 2–3 throwaway generations before the measured arms.** The per-block warmup does NOT cover
this — the ramp spans the session, not the block. Something like:

```bash
for i in 1 2; do ./.build/debug/RunLTX2 --e2e25 704 512 121 >/dev/null 2>&1; done
```

Then launch the real thing (drop `--dry-run`, add `--cooldown 30 --label armB-dfr2 --out probes`).
Budget **2.5–3 h**: ~12 generations, native ~500–600 s and DFR-2 possibly ~1000 s each.

### Pre-registered expectation (written BEFORE the run — do not quietly revise it after)

> **DFR-2 should lose by MORE than DFR-1 did.** Each round roughly doubles the tile count so rounds
> compound superlinearly, while native scales ~linearly in tokens (481f@96 = 21,472 video tokens vs
> 241f@48's 10,912). Rough decomposition from arm B: DFR-1 = 402 s ≈ ~100 s base + ~300 s round;
> round 2 densifies 241→481 at twice the tiles, so ≈ +600 s → **DFR-2 ≈ 1000 s vs native ≈ 500–600 s.**
>
> If that holds, the "long clips could favour rounds" regime requires native to be **infeasible**
> (out of sequence length / memory), not merely slower — and at 481f@96 on 128 GB native is
> comfortable. **That would close C2's last open regime** rather than leave it hanging.

⚠️ A result that *contradicts* this (DFR-2 winning) is the interesting one and must not be explained
away — but check the delivered frame count first (the harness verifies it per run and aborts on a
mismatch, so a passing run genuinely delivered 481f).

## Matrix status — 5 of 6 arms run

| arm | claim | status |
|---|---|---|
| A | §V item 1 | ✅ PARITY (Δmed −0.4 s at 121f; confirmed again at 241f, Δ +2.7 s) |
| B | C2 | ✅ DFR r=1 DOMINATED ×1.68 · 2.3 baselines corrected (AB-R-0066) |
| C | C3 | ⬜ **UNRUN** — decode triangle, conv vs DiffVAE |
| D | C6 | ✅ Swift reproduces the oracle's shape (AB-R-0067) |
| E | C5 | ⬜ UNRUN — enhancer overhead |
| F | C1 | ✅ multishot is FREE (AB-R-0067) |

**Arm C no longer needs clips staged locally** — the "AVFoundation can't read off /Volumes/Satechi"
claim was measured FALSE and withdrawn (AB-L-0041; probe in `probes/avfoundation-volume-probe/`).
Read the corpus in place.

## Protocol reminders that cost real time to learn

- **`--dry-run` before every real invocation.** Verifying a plan should never cost a run; it exists
  because a "just check the spec parses" invocation once started a multi-minute generation on a
  busy machine.
- **Absolute wall-clock is NOT comparable across sessions, only within.** The same 121f work read
  95–103 s contended vs 66.7 s cold. Quote deltas, not absolutes.
- **Noise floors are geometry-specific**: ≤8.5 s at 9f, ≤~10 s at 121f (measured, with within-arm
  spreads to ~34 s). Do not carry the 9f figure to longer geometries. See `BENCH.md`.
- **`--bench-e2e` runs UNEVICTED.** Its peaks are not comparable to the footprint receipts
  (AB-R-0039/0041), which are evicted + cache-capped.
- **Adding an arm axis? Update `expectsIdenticalOutput`.** It has been missing an axis four times
  now (`model`, `dfrRounds`, `env`, prompt). That is this harness's most repeatable defect.
- **Never `pkill -f RunLTX2`** — it matches other sessions' gate runs. Kill by PID.

## Other open threads (not blocking the above)

- **AB-T-0009** — BlockStreamKit drift; partly addressed already (the swift repo moved to a
  versioned URL dep at `7c5a32f`), consumers may still need the sweep.
- **`--mem-bench25`'s `DECLARE →` line** still derives `residentBytes` from the post-run floor,
  which is an artifact (AB-R-0041). It should emit the post-load floor plus a separate `retain=`,
  per `mlx-swift-integration`'s `references/package-efficiency.md`. Ignore that line until then.
- **mlx-swift-lm PR #530** — drop the `gemma4-encoder-spi` fork when it lands.
- **Encoder publish** — operator-cleared, encoder-only, intentionally on hold (AB-D-0013).
