# bench-e2e receipt — armB-dfr2-longclip

- started: 2026-08-16 04:35:20 +0000 (UTC stamp 20260816-043520)
- host: Mac17,7 · 137 GB · Version 27.0 (Build 26A5406e)
- repo: b9eebae+dirty · geometry: 704x512x481f fps=96 two-stage · seed 42
- prompt: "a cat playing piano"
- protocol: blocks=2 (ABBA order) · runs/block=2 (+1 excluded warmup) · cooldown 30s · MLX_PROFILE off
- arms:
  - `t25-native`: quant=bf16 decoder=stock cache=2GB
  - `t25-dfr2`: quant=bf16 decoder=stock cache=2GB

## Per-run

| arm | block | run | wall s | peak phys GB | mlx act/cache GB | thermal | cos(arm-first) |
|---|---|---|---|---|---|---|---|
| t25-native | 0 | warm | 515.3 | 64.35 | 40.1/0.1 | fair→fair | — |
| t25-native | 0 | 1 | 534.3 | 62.74 | 40.1/0.1 | fair→fair | — |
| t25-native | 0 | 2 | 521.8 | 64.89 | 42.1/0.1 | fair→fair | 1.000000 |
| t25-dfr2 | 0 | warm | 894.8 | 66.68 | 42.1/0.0 | fair→fair | — |
| t25-dfr2 | 0 | 1 | 940.8 | 65.07 | 42.1/0.0 | fair→fair | — |
| t25-dfr2 | 0 | 2 | 934.1 | 67.15 | 44.2/0.0 | fair→fair | 1.000000 |
| t25-dfr2 | 1 | warm | 890.1 | 68.76 | 44.2/0.0 | fair→fair | — |
| t25-dfr2 | 1 | 1 | 933.6 | 67.15 | 44.2/0.0 | fair→fair | 1.000000 |
| t25-dfr2 | 1 | 2 | 938.2 | 67.14 | 44.2/0.0 | fair→fair | 1.000000 |
| t25-native | 1 | warm | 502.2 | 68.70 | 44.2/0.1 | fair→fair | — |
| t25-native | 1 | 1 | 532.0 | 67.14 | 44.2/0.1 | fair→fair | 1.000000 |
| t25-native | 1 | 2 | 537.0 | 67.15 | 44.2/0.1 | fair→fair | 1.000000 |

## Blocks (load + ratchet detector)

| arm | block | prewarm s | load s | post-drop floor GB |
|---|---|---|---|---|
| t25-native | 0 | 4.6 | 1.3 | 4.54 |
| t25-dfr2 | 0 | 5.5 | 1.2 | 6.78 |
| t25-dfr2 | 1 | 7.7 | 1.2 | 6.78 |
| t25-native | 1 | 7.7 | 1.3 | 6.78 |

## Arm stats (measured runs only)

| arm | median s | min–max s | median peak GB |
|---|---|---|---|
| t25-native | 533.1 | 521.8–537.0 | 66.01 |
| t25-dfr2 | 936.2 | 933.6–940.8 | 67.15 |

## Session drift & thermal strata (v2)

- pooled drift: +0.730 s/run → fitted span +5.1 s across the session
- `t25-native`: raw med 533.1 s · drift-adjusted med 532.3 s · within-arm block span 6.4 s
- `t25-dfr2`: raw med 936.2 s · drift-adjusted med 935.8 s · within-arm block span 1.5 s
- thermal `fair`: t25-dfr2 Δ+403.0s

## Verdicts (vs `t25-native`)

- `t25-dfr2`: Δmed +403.0s > spread 15.1s — MEASURABLE

## Cross-arm output (first measured run, vs `t25-native`)

| arm | cosine | maxAbs | class |
|---|---|---|---|
| t25-dfr2 | 0.213226 | 2.28981 | not gated (q8 nondeterminism / divergent-by-design) |

> Doctrine: totals here are UNPROFILED. For a stage split, run the same arm once under
> `MLX_PROFILE=csv` separately and never compare its timings across variants (PROFILING.md).
