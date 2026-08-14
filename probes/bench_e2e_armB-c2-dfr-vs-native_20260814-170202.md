# bench-e2e receipt — armB-c2-dfr-vs-native

- started: 2026-08-14 17:02:02 +0000 (UTC stamp 20260814-170202)
- host: Mac17,7 · 137 GB · Version 27.0 (Build 26A5406e)
- repo: de83da1+dirty · geometry: 704x512x241f fps=48 two-stage · seed 42
- prompt: "a cat playing piano"
- protocol: blocks=2 (ABBA order) · runs/block=2 (+1 excluded warmup) · cooldown 30s · MLX_PROFILE off
- arms:
  - `t25-native`: quant=bf16 decoder=stock cache=2GB
  - `t25-dfr1`: quant=bf16 decoder=stock cache=2GB

## Per-run

| arm | block | run | wall s | peak phys GB | mlx act/cache GB | thermal | cos(arm-first) |
|---|---|---|---|---|---|---|---|
| t25-native | 0 | warm | 149.8 | 64.29 | 39.0/0.0 | nominal→fair | — |
| t25-native | 0 | 1 | 257.2 | 63.77 | 39.0/0.0 | fair→fair | — |
| t25-native | 0 | 2 | 251.0 | 64.93 | 40.1/0.0 | fair→fair | 1.000000 |
| t25-dfr1 | 0 | warm | 379.9 | 65.66 | 40.1/0.0 | fair→fair | — |
| t25-dfr1 | 0 | 1 | 403.0 | 65.13 | 40.1/0.0 | fair→fair | — |
| t25-dfr1 | 0 | 2 | 402.5 | 66.11 | 41.1/0.0 | fair→fair | 1.000000 |
| t25-dfr1 | 1 | warm | 379.0 | 66.64 | 41.1/0.0 | fair→fair | — |
| t25-dfr1 | 1 | 1 | 401.6 | 66.12 | 41.1/0.0 | fair→fair | 1.000000 |
| t25-dfr1 | 1 | 2 | 385.4 | 66.12 | 41.1/0.0 | fair→fair | 1.000000 |
| t25-native | 1 | warm | 197.0 | 66.62 | 41.1/0.0 | fair→fair | — |
| t25-native | 1 | 1 | 225.7 | 66.12 | 41.1/0.0 | fair→fair | 1.000000 |
| t25-native | 1 | 2 | 228.4 | 66.12 | 41.1/0.0 | fair→fair | 1.000000 |

## Blocks (load + ratchet detector)

| arm | block | prewarm s | load s | post-drop floor GB |
|---|---|---|---|---|
| t25-native | 0 | 5.6 | 1.4 | 4.56 |
| t25-dfr1 | 0 | 6.0 | 1.4 | 4.72 |
| t25-dfr1 | 1 | 6.3 | 1.4 | 4.72 |
| t25-native | 1 | 6.2 | 1.4 | 4.72 |

## Arm stats (measured runs only)

| arm | median s | min–max s | median peak GB |
|---|---|---|---|
| t25-native | 239.7 | 225.7–257.2 | 65.53 |
| t25-dfr1 | 402.1 | 385.4–403.0 | 66.12 |

## Session drift & thermal strata (v2)

- pooled drift: -4.546 s/run → fitted span -31.8 s across the session
- `t25-native`: raw med 239.7 s · drift-adjusted med 240.5 s · within-arm block span 27.1 s
- `t25-dfr1`: raw med 402.1 s · drift-adjusted med 398.2 s · within-arm block span 9.2 s
- thermal `fair`: t25-dfr1 Δ+162.3s

## Verdicts (vs `t25-native`)

- `t25-dfr1`: Δmed +162.3s > spread 31.5s — MEASURABLE

## Cross-arm output (first measured run, vs `t25-native`)

| arm | cosine | maxAbs | class |
|---|---|---|---|
| t25-dfr1 | 0.769262 | 1.98680 | not gated (q8 nondeterminism / divergent-by-design) |

> Doctrine: totals here are UNPROFILED. For a stage split, run the same arm once under
> `MLX_PROFILE=csv` separately and never compare its timings across variants (PROFILING.md).
