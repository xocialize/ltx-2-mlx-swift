# bench-e2e receipt — armF-c1-multishot-cost

- started: 2026-08-15 14:17:05 +0000 (UTC stamp 20260815-141705)
- host: Mac17,7 · 137 GB · Version 27.0 (Build 26A5406e)
- repo: 628d92e+dirty · geometry: 704x512x121f fps=24 two-stage · seed 42
- prompt: "a cat playing piano"
- protocol: blocks=2 (ABBA order) · runs/block=2 (+1 excluded warmup) · cooldown 30s · MLX_PROFILE off
- arms:
  - `f-single`: quant=bf16 decoder=stock cache=2GB
  - `f-multishot`: quant=bf16 decoder=stock cache=2GB

## Per-run

| arm | block | run | wall s | peak phys GB | mlx act/cache GB | thermal | cos(arm-first) |
|---|---|---|---|---|---|---|---|
| f-single | 0 | warm | 60.5 | 64.40 | 38.5/0.0 | nominal→nominal | — |
| f-single | 0 | 1 | 68.8 | 63.30 | 38.5/0.0 | nominal→fair | — |
| f-single | 0 | 2 | 66.7 | 63.90 | 39.0/0.0 | fair→fair | 1.000000 |
| f-multishot | 0 | warm | 73.1 | 65.14 | 39.0/0.0 | fair→fair | — |
| f-multishot | 0 | 1 | 80.1 | 64.10 | 39.0/0.0 | fair→fair | — |
| f-multishot | 0 | 2 | 92.9 | 64.58 | 39.6/0.0 | fair→fair | 1.000000 |
| f-multishot | 1 | warm | 89.0 | 65.60 | 39.6/0.0 | fair→fair | — |
| f-multishot | 1 | 1 | 93.0 | 64.59 | 39.6/0.0 | fair→fair | 1.000000 |
| f-multishot | 1 | 2 | 102.9 | 65.61 | 39.6/0.0 | fair→fair | 1.000000 |
| f-single | 1 | warm | 94.6 | 65.72 | 39.6/0.0 | fair→fair | — |
| f-single | 1 | 1 | 98.5 | 64.60 | 39.6/0.0 | fair→fair | 1.000000 |
| f-single | 1 | 2 | 100.4 | 65.62 | 39.6/0.0 | fair→fair | 1.000000 |

## Blocks (load + ratchet detector)

| arm | block | prewarm s | load s | post-drop floor GB |
|---|---|---|---|---|
| f-single | 0 | 8.0 | 1.6 | 3.04 |
| f-multishot | 0 | 6.1 | 1.3 | 5.31 |
| f-multishot | 1 | 6.0 | 1.4 | 3.70 |
| f-single | 1 | 6.1 | 1.4 | 3.75 |

## Arm stats (measured runs only)

| arm | median s | min–max s | median peak GB |
|---|---|---|---|
| f-single | 83.7 | 66.7–100.4 | 64.25 |
| f-multishot | 92.9 | 80.1–102.9 | 64.58 |

## Session drift & thermal strata (v2)

- pooled drift: +5.349 s/run → fitted span +37.4 s across the session
- `f-single`: raw med 83.7 s · drift-adjusted med 83.4 s · within-arm block span 31.7 s
- `f-multishot`: raw med 92.9 s · drift-adjusted med 92.6 s · within-arm block span 11.5 s
- thermal `fair`: f-multishot Δ-5.6s
- thermal `nominal`: f-multishot: single-arm (1/0)

## Verdicts (vs `f-single`)

- `f-multishot`: Δmed +9.3s — NOISE (sign flips between block sets)

## Cross-arm output (first measured run, vs `f-single`)

| arm | cosine | maxAbs | class |
|---|---|---|---|
| f-multishot | 0.877864 | 2.23167 | not gated (q8 nondeterminism / divergent-by-design) |

> Doctrine: totals here are UNPROFILED. For a stage split, run the same arm once under
> `MLX_PROFILE=csv` separately and never compare its timings across variants (PROFILING.md).
