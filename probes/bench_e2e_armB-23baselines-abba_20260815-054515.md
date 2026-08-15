# bench-e2e receipt — armB-23baselines-abba

- started: 2026-08-15 05:45:15 +0000 (UTC stamp 20260815-054515)
- host: Mac17,7 · 137 GB · Version 27.0 (Build 26A5406e)
- repo: 7c5a32f+dirty · geometry: 704x512x241f fps=48 two-stage · seed 42
- prompt: "a cat playing piano"
- protocol: blocks=2 (ABBA order) · runs/block=2 (+1 excluded warmup) · cooldown 30s · MLX_PROFILE off
- arms:
  - `t23-native`: quant=bf16 decoder=stock cache=2GB
  - `t23-temporal`: quant=bf16 decoder=stock cache=2GB env=LTX_UPSAMPLER=temporal_upscaler_x2_v1_0
  - `t25-native`: quant=bf16 decoder=stock cache=2GB

## Per-run

| arm | block | run | wall s | peak phys GB | mlx act/cache GB | thermal | cos(arm-first) |
|---|---|---|---|---|---|---|---|
| t23-native | 0 | warm | 140.8 | 56.61 | 39.0/0.0 | nominal→fair | — |
| t23-native | 0 | 1 | 213.4 | 56.42 | 39.0/0.0 | fair→fair | — |
| t23-native | 0 | 2 | 235.1 | 57.51 | 40.1/0.0 | fair→fair | 1.000000 |
| t23-temporal | 0 | warm | 341.7 | 57.59 | 40.1/0.0 | fair→fair | — |
| t23-temporal | 0 | 1 | 366.4 | 57.79 | 40.1/0.0 | fair→fair | — |
| t23-temporal | 0 | 2 | 368.2 | 58.85 | 41.1/0.0 | fair→fair | 1.000000 |
| t25-native | 0 | warm | 208.7 | 66.62 | 41.1/0.0 | fair→fair | — |
| t25-native | 0 | 1 | 234.0 | 66.09 | 41.1/0.0 | fair→fair | — |
| t25-native | 0 | 2 | 228.6 | 67.16 | 42.2/0.0 | fair→fair | 1.000000 |
| t25-native | 1 | warm | 202.2 | 67.66 | 42.2/0.0 | fair→fair | — |
| t25-native | 1 | 1 | 229.1 | 67.14 | 42.2/0.0 | fair→fair | 1.000000 |
| t25-native | 1 | 2 | 231.8 | 67.13 | 42.2/0.0 | fair→fair | 1.000000 |
| t23-temporal | 1 | warm | 328.0 | 59.50 | 42.2/0.0 | fair→fair | — |
| t23-temporal | 1 | 1 | 348.5 | 59.45 | 42.2/0.0 | fair→fair | 1.000000 |
| t23-temporal | 1 | 2 | 345.5 | 59.67 | 42.2/0.0 | fair→fair | 1.000000 |
| t23-native | 1 | warm | 196.4 | 59.63 | 42.2/0.0 | fair→fair | — |
| t23-native | 1 | 1 | 223.8 | 59.87 | 42.2/0.0 | fair→fair | 1.000000 |
| t23-native | 1 | 2 | 225.1 | 59.63 | 42.2/0.0 | fair→fair | 1.000000 |

## Blocks (load + ratchet detector)

| arm | block | prewarm s | load s | post-drop floor GB |
|---|---|---|---|---|
| t23-native | 0 | 9.5 | 1.3 | 3.55 |
| t23-temporal | 0 | 3.1 | 1.2 | 7.82 |
| t25-native | 0 | 11.8 | 1.5 | 8.86 |
| t25-native | 1 | 6.1 | 1.4 | 5.74 |
| t23-temporal | 1 | 9.4 | 1.2 | 5.76 |
| t23-native | 1 | 7.5 | 1.2 | 8.91 |

## Arm stats (measured runs only)

| arm | median s | min–max s | median peak GB |
|---|---|---|---|
| t23-native | 224.5 | 213.4–235.1 | 58.57 |
| t23-temporal | 357.4 | 345.5–368.2 | 59.15 |
| t25-native | 230.4 | 228.6–234.0 | 67.14 |

## Session drift & thermal strata (v2)

- pooled drift: -0.782 s/run → fitted span -8.6 s across the session
- `t23-native`: raw med 224.5 s · drift-adjusted med 228.4 s · within-arm block span 0.2 s
- `t23-temporal`: raw med 357.4 s · drift-adjusted med 357.0 s · within-arm block span 20.3 s
- `t25-native`: raw med 230.4 s · drift-adjusted med 231.1 s · within-arm block span 0.8 s
- thermal `fair`: t23-temporal Δ+133.0s · t25-native Δ+6.0s

## Verdicts (vs `t23-native`)

- `t23-temporal`: Δmed +133.0s > spread 22.7s — MEASURABLE
- `t25-native`: Δmed +2.7s ≤ spread 22.5s — NOISE  ⚠ drift span -8.6s rivals Δ — decided on drift-adjusted Δ +2.7s

## Cross-arm output (first measured run, vs `t23-native`)

| arm | cosine | maxAbs | class |
|---|---|---|---|
| t23-temporal | 0.502494 | 1.76364 | same-weights bf16 (expect ≈1) |
| t25-native | 0.642648 | 1.78666 | not gated (q8 nondeterminism / divergent-by-design) |

> Doctrine: totals here are UNPROFILED. For a stage split, run the same arm once under
> `MLX_PROFILE=csv` separately and never compare its timings across variants (PROFILING.md).
