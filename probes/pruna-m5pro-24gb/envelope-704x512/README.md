# AB-T-0298 on the M5 Pro 24 GB: compact24's envelope widened to 704×512×121 under the lean decoder

**2026-10-15, MacBook Pro (M5 Pro, 16-core GPU, 24 GB), macOS 27.2, release `RunLTX2` with the profile change in
this commit (compact24 `maxWidth`/`maxHeight` 512×288 → 704×512; `maxFrames` 121 unchanged), GPU otherwise idle.**
The acceptance runs AB-T-0298 asks for: at a geometry the profile now ALLOWS (no `LTX_ENVELOPE_OVERRIDE`; the
harness would print a warning otherwise, and the script flags it), with the lean decoder the tier defaults to
since v0.19.0 (AB-D-0147), both modes.

## Setup

As the t2v and i2v receipts one directory up (`../README.md`, `../i2v/README.md`): the shipping compact24
configuration (streamed q8 DiT from the v2 granule tree with `LTX_NO_CHECKPOINT=1`, int8 Gemma-4, `.forceStream`),
`--t2v-spot25` as the staged instrument, one fresh process per run under `/usr/bin/time -l`, swap and thermal state
around each run, 45 s cooldown, prompt "a cat playing piano", seed 42; i2v adds the synthetic first frame and the
4.9 GB adapter. Budget on this host 19.07 GB; compact24's declared charge 16.20 GB (0.8 + 15.4, AB-R-0107).

- **Measured rows**: the lean decoder (profile default, lever unset), four per mode, after two burn-ins and a
  warm-up per mode.
- **Diagnostic rows** (never acceptance numbers): `LTX_VAE_DECODER=stock` at the same geometry, two per mode —
  the reason the widened envelope is lean-only. The footprint hints fail closed with an explicit stock path on this
  tier (gate case 80) for the same reason.

## Runs — 704×512×121, acceptance-grade (the profile allows it; no override warning on any run)

### t2v

| run | decoder | gen s | wall s | resident GB | activation GB | peak GB | footprint GB | swap Δ GB | vs 19.07 GB |
|---|---|---|---|---|---|---|---|---|---|
| burn-in 1 | lean | 147.4 | 176.3 | 0.94 | 13.58 | 14.52 | 14.60 | +0.34 | within |
| burn-in 2 | lean | 151.0 | 179.1 | 0.90 | 13.59 | 14.49 | 14.51 | 0 | within |
| warm-up | lean | 149.8 | 177.7 | 0.90 | 13.60 | 14.50 | 14.51 | −0.02 | within |
| rep 1 | **lean** | **149.1** | 177.1 | 0.89 | 13.60 | **14.49** | 14.49 | 0 | within |
| diag 1 | stock | 180.1 | 209.0 | 0.87 | 20.75 | **21.62** | 21.64 | +0.63 | OVER |
| rep 2 | **lean** | **148.8** | 176.8 | 0.89 | 13.61 | **14.49** | 14.49 | −0.15 | within |
| rep 3 | **lean** | **150.7** | 181.5 | 0.94 | 13.54 | **14.48** | 14.51 | −0.05 | within |
| diag 2 | stock | 183.8 | 213.0 | 0.86 | 20.66 | **21.53** | 21.62 | +0.08 | OVER |
| rep 4 | **lean** | **151.4** | 179.8 | 0.83 | 13.66 | **14.49** | 14.50 | −0.05 | within |

lean, 4 measured: median **149.9 s** (148.8–151.4, range 2.6), peak max **14.49 GB**, resident max 0.94, activation max
13.66, footprint max 14.51 GB, swap flat. stock, 2 diagnostic: median 181.9 s, peak 21.53–21.62 GB, over this host's
budget by 2.5 GB, 0.1–0.6 GB pushed to swap. Against the tier's declared charge (16.20 GB) the lean peak leaves
**1.71 GB**; against this host's 19.07 GB budget, 4.58 GB.

### i2v

| run | decoder | gen s | wall s | resident GB | activation GB | peak GB | footprint GB | swap Δ GB | vs 19.07 GB |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | lean | 176.4 | 208.4 | 0.68 | 17.20 | **17.88** | 17.88 | −0.03 | within |
| rep 1 | **lean** | **176.7** | 208.5 | 0.58 | 16.58 | **17.16** | 17.16 | −0.01 | within |
| diag 1 | stock | 207.0 | 239.7 | 0.86 | 19.22 | **20.08** | 20.10 | +0.34 | OVER |
| rep 2 | **lean** | **172.8** | 204.5 | 0.50 | 16.63 | **17.13** | 17.13 | −0.03 | within |
| rep 3 | **lean** | **175.8** | 207.1 | 0.76 | 16.09 | **16.85** | 16.85 | −0.07 | within |
| diag 2 | stock | 207.8 | 240.8 | 0.58 | 19.47 | **20.06** | 20.17 | +0.73 | OVER |
| rep 4 | **lean** | **174.1** | 206.1 | 0.49 | 16.37 | **16.85** | 16.85 | −0.18 | within |

lean, 4 measured: median **174.9 s** (172.8–176.7, range 3.9), peak max **17.16 GB** (17.88 on the excluded warm-up),
resident max 0.76, activation max 16.63, swap flat. stock, 2 diagnostic: 207.4 s, peak 20.06–20.08 GB, over this
host's budget, 0.3–0.7 GB to swap. Against the tier's declared charge (16.20 GB) the lean i2v peak is **0.96 GB
over** (1.68 over counting the warm-up); against this host's 19.07 GB budget it is 1.9 GB inside.

## Findings

1. **t2v at 704×512×121 under the lean decoder is acceptance-grade and comfortable**: four runs at 148.8–151.4 s
   with the peak pinned at 14.48–14.49 GB — 1.71 GB inside the tier's 16.20 GB declared charge and 4.58 GB inside
   this host's 19.07 GB budget, swap flat, no override, the profile allowing the geometry. The same runs with the
   stock decoder peak 21.5–21.6 GB, over budget and swapping: the widened envelope exists only because the lean
   decoder is the tier's default (and the footprint hints fail closed with an explicit stock path, gate case 80).
2. **i2v at 704×512×121 fits this machine but not the tier's charge.** Lean i2v peaks 16.85–17.16 GB measured
   (17.88 on the warm-up), within the 19.07 GB budget with no swap — but the image conditioning scales with pixels:
   +2.7 GB over t2v at this envelope against +1.4 GB at 512×288 (AB-R-0562). A wide i2v envelope would need the
   compact24 charge raised from 16.20 to about **18.0 GB** (resident 0.8 + activation 17.2, covering the warm-up),
   which clears a real 24 GB Mac's 19.07 GB budget by 1.1 GB (94 %) but busts the 16.80 GB nominal bar the fleet
   holds the charge to (gate case 39; the engine's no-Metal / synthetic-profile fallback) — a charge decision, not
   a measurement, and the operator's (AB-R-0107 accepted 96 % of budget once; AB-R-0038 called 96 % unsafe).
3. **Shipped shape: a mode-aware envelope.** compact24 is 704×512×121 for plain t2v and stays 512×288×121 for
   image-, reference- and audio-conditioned runs (`LTX2Profile.imageConditionedMaxWidth/Height`; the resolver
   takes `imageConditioned:`), so the declared charge stays 16.20 GB and every acceptance number behind it stands:
   t2v wide 14.49 GB, i2v narrow 15.85 GB (AB-R-0562). Hosts previewing an i2v output size pass the flag.
4. **The lean decoder's time win scales with the geometry**: −32 s per clip at 704×512 in both modes (t2v −17.6 %,
   i2v −15.6 %) against −12 s at 512×288.
5. **The box**: fair thermal state throughout the measured rows, nominal only at the start; swap flat on every lean
   row, 0.1–0.7 GB per stock row; no run threw.

## Verdict

**Widen t2v, hold i2v, charge unchanged.** compact24 ships a 704×512×121 text-to-video envelope under the lean
decoder, acceptance-grade with 1.7 GB inside its declared charge; image-conditioned runs keep the 512×288×121
envelope, where i2v was measured inside the charge (AB-R-0562). Widening i2v too is possible on a real 24 GB Mac
(17.2 GB measured, 19.07 budget, no swap) but requires the operator to raise the tier's charge to ~18.0 GB past the
16.80 GB nominal bar — recorded here with the numbers, left as the operator's call.

## Files

`run.log`, `logs/*.full.log`, `run-envelope.sh`; `../analyze.py` parses this log (lean rows count as the Pruna arm).
