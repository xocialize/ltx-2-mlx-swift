# bench-e2e receipt — armA-perf-parity

- started: 2026-08-14 16:15:27 +0000 (UTC stamp 20260814-161527)
- host: Mac17,7 · 137 GB · Version 27.0 (Build 26A5406e)
- repo: ef5b901+dirty · geometry: 704x512x121f fps=24 two-stage · seed 42
- prompt: "a cat playing piano"
- protocol: blocks=2 (ABBA order) · runs/block=2 (+1 excluded warmup) · cooldown 30s · MLX_PROFILE off
- arms:
  - `ltx23`: quant=bf16 decoder=stock cache=2GB
  - `ltx25`: quant=bf16 decoder=stock cache=2GB

## Per-run

| arm | block | run | wall s | peak phys GB | mlx act/cache GB | thermal | cos(arm-first) |
|---|---|---|---|---|---|---|---|
| ltx23 | 0 | warm | 61.7 | 56.16 | 38.5/0.0 | nominal→nominal | — |
| ltx23 | 0 | 1 | 67.7 | 56.30 | 38.5/0.0 | nominal→fair | — |
| ltx23 | 0 | 2 | 97.5 | 56.86 | 39.0/0.0 | fair→fair | 1.000000 |
| ltx25 | 0 | warm | 94.6 | 65.06 | 39.0/0.0 | fair→fair | — |
| ltx25 | 0 | 1 | 95.3 | 64.04 | 39.0/0.0 | fair→fair | — |
| ltx25 | 0 | 2 | 101.8 | 64.56 | 39.6/0.0 | fair→fair | 1.000000 |
| ltx25 | 1 | warm | 92.5 | 65.62 | 39.6/0.0 | fair→fair | — |
| ltx25 | 1 | 1 | 95.1 | 64.56 | 39.6/0.0 | fair→fair | 1.000000 |
| ltx25 | 1 | 2 | 103.2 | 65.61 | 39.6/0.0 | fair→fair | 1.000000 |
| ltx23 | 1 | warm | 92.4 | 56.34 | 39.6/0.0 | fair→fair | — |
| ltx23 | 1 | 1 | 100.5 | 56.10 | 39.6/0.0 | fair→fair | 1.000000 |
| ltx23 | 1 | 2 | 107.5 | 57.05 | 39.6/0.0 | fair→fair | 1.000000 |

## Blocks (load + ratchet detector)

| arm | block | prewarm s | load s | post-drop floor GB |
|---|---|---|---|---|
| ltx23 | 0 | 2.3 | 1.3 | 4.08 |
| ltx25 | 0 | 11.9 | 1.5 | 5.25 |
| ltx25 | 1 | 6.1 | 1.5 | 5.30 |
| ltx23 | 1 | 9.4 | 1.3 | 3.75 |

## Arm stats (measured runs only)

| arm | median s | min–max s | median peak GB |
|---|---|---|---|
| ltx23 | 99.0 | 67.7–107.5 | 56.58 |
| ltx25 | 98.5 | 95.1–103.2 | 64.56 |

## Session drift & thermal strata (v2)

- pooled drift: +3.696 s/run → fitted span +25.9 s across the session
- `ltx23`: raw med 99.0 s · drift-adjusted med 92.9 s · within-arm block span 21.4 s
- `ltx25`: raw med 98.5 s · drift-adjusted med 99.2 s · within-arm block span 0.6 s
- thermal `fair`: ltx25 Δ-1.9s
- thermal `nominal`: ltx25: single-arm (1/0)

## Verdicts (vs `ltx23`)

- `ltx25`: Δmed -0.4s — NOISE (sign flips between block sets)

## Cross-arm output (first measured run, vs `ltx23`)

| arm | cosine | maxAbs | class |
|---|---|---|---|
| ltx25 | 0.528606 | 1.94179 | not gated (q8 nondeterminism / divergent-by-design) |

> Doctrine: totals here are UNPROFILED. For a stage split, run the same arm once under
> `MLX_PROFILE=csv` separately and never compare its timings across variants (PROFILING.md).
