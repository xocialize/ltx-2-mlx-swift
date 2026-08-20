# RESUME — next session starts here

> ✅ **2026-08-20 — the int8 encoder PASSED a BLIND perceptual A/B (AB-R-0104), and both 2.5 wiring
> gaps are CLOSED (AB-R-0105).** Everything is committed, pushed and gate-green at **v0.8.0**.
> **One measurement gap remains** before AB-D-0014's tier table can be rewritten, and it is a
> single unattended script: `probes/tier25-matrix/run-i2v-and-footprint.sh` (~45–60 min, needs a
> dedicated GPU slot).

Closed since the last entry: the perceptual A/B (AB-T-0062 → AB-R-0104), the tier matrix
(AB-R-0090), the `.auto`-gate flip and both wiring gaps (AB-R-0105), and **DFR is formally
DEFERRED** for the base release (AB-D-0034 — dominated ×1.6–1.8, operator prefers native, and it is
not reachable from the package so no tier declaration covers it).

## The next run — one command

```bash
cd LTX_DEV/ltx-2-mlx-swift && ./probes/tier25-matrix/run-i2v-and-footprint.sh
```

18 runs, serial, unattended. It measures **i2v** (never measured under this config; adds the
~4.9 GB i2v-adapter, so it cannot be inferred from the t2v row) and re-declares the **t2v
footprint** with the corrected instrument. The script carries the full protocol in its header.

## Where things stand

| tier | budget | shipping candidate (streamed DiT + int8 encoder) | |
|---|---|---|---|
| compact24 | 16.8 | **14.56 ×3** — requires `.forceStream` | ✅ |
| balanced32 | 22.4 | 15.36 / 15.38 / 15.39 | ✅ |
| standard64 | 44.8 | 19.44 / 19.50 / 19.59 | ✅ |

Also standing on its own: streaming makes **2.5 bf16 standard64-admissible** (45.27 ❌ → 24.81 ✅).

**Both levers are required and neither works alone** — resident is DiT-bound at ~31.9 (so the
encoder swap is invisible there); streaming removes that floor and exposes a **geometry-independent
~24.8 GB encoder floor**; only removing both clears a low tier.

## ⚠️ Three traps this cost us — do not re-learn them

1. **`.auto` flips run-to-run at small N.** compact24 read peak 14.57 / **33.63** / 14.58 across
   three runs because C(N) is noisy there (3.34 / 7.43 / 3.52 GiB/s), moving N_min 1823→4133→1919.
   Not a warm-vs-cold-box effect — reps 2 and 3 ran back-to-back on one boot and disagreed.
   `.forceStream` fixes it at **no measurable time cost** (a spurious fallback is not free either:
   it loads 18.38 GiB). **Read WORST-case headroom, never the mean.**
2. **The footprint instrument was wrong twice.** The post-run sample is unstable (AB-R-0078);
   `phys-after-load` — the fix that receipt prescribes — is ALSO wrong here because it was written
   for a NON-evicting regime, and under eviction it reads a transient the run never returns to
   (22.13 GB against a 14.59 GB peak). Correct is the **low-water during the run**. ⚠️ PEAK was
   never affected, so every acceptance number in the table above stands.
3. **A perceptual A/B on an ENCODER is a preference test, not a similarity test.** Changing the
   encoder changes conditioning → trajectory → output, so the arms are *different, equally valid*
   videos by construction. SSIM/PSNR between them measures nothing. (Contrast the pruna A/B, which
   swapped a DECODER on identical latents — there SSIM 0.9825 honestly meant reconstruction.)

## After the script lands

Take the **worst-case** resident/activation per tier from the `DECLARE →` lines, then edit
`MLXLTX25Package`'s `QuantFootprint` and AB-D-0014's tier table. Nothing else blocks it.

⚠️ The int8 encoder is opt-in by design: `LTX2Configuration.textEncoderQuant` (default `.bf16`),
with `LTX2Profile.recommendedTextEncoderQuant` as **advice only**. The profile advises, the config
decides — same doctrine as `recommendedQuant`. Deciding whether the engine should *follow* that
advice automatically for low tiers is a separate call, and it has not been made.
