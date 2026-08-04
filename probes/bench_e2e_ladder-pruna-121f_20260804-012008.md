# bench-e2e receipt — ladder-pruna-121f

- started: 2026-08-04 01:20:08 +0000 (UTC stamp 20260804-012008)
- host: Mac17,7 · 137 GB · Version 27.0 (Build 26A5388g)
- repo: 7a37941+dirty · geometry: 704x512x121f fps=24 one-stage · seed 42
- prompt: "a cat playing piano"
- protocol: blocks=2 (ABBA order) · runs/block=2 (+1 excluded warmup) · cooldown 60s · MLX_PROFILE off
- arms:
  - `bf16`: quant=bf16 decoder=stock
  - `pruna`: quant=bf16 decoder=pruna
  - `q8`: quant=q8 decoder=stock
  - `q4`: quant=q4 decoder=stock

## Per-run

| arm | block | run | wall s | peak phys GB | mlx act/cache GB | thermal | cos(arm-first) |
|---|---|---|---|---|---|---|---|
| bf16 | 0 | warm | 89.1 | 66.24 | 38.5/0.0 | nominal→nominal | — |
| bf16 | 0 | 1 | 148.1 | 66.14 | 38.5/0.0 | nominal→fair | — |
| bf16 | 0 | 2 | 182.2 | 66.94 | 39.0/0.0 | fair→fair | 1.000000 |
| pruna | 0 | warm | 130.9 | 59.44 | 39.0/0.0 | fair→fair | — |
| pruna | 0 | 1 | 153.7 | 59.62 | 39.0/0.0 | fair→fair | — |
| pruna | 0 | 2 | 167.0 | 60.10 | 39.6/0.0 | fair→fair | 1.000000 |
| q8 | 0 | warm | 128.2 | 50.15 | 22.2/0.0 | fair→fair | — |
| q8 | 0 | 1 | 157.3 | 50.00 | 22.2/0.0 | fair→fair | — |
| q8 | 0 | 2 | 168.2 | 50.68 | 22.7/0.0 | fair→fair | 1.000000 |
| q4 | 0 | warm | 105.7 | 41.35 | 13.4/0.0 | fair→fair | — |
| q4 | 0 | 1 | 142.4 | 41.46 | 13.4/0.0 | fair→fair | — |
| q4 | 0 | 2 | 158.3 | 41.86 | 13.9/0.0 | fair→fair | 1.000000 |
| q4 | 1 | warm | 105.0 | 42.00 | 13.9/0.0 | fair→fair | — |
| q4 | 1 | 1 | 139.6 | 41.90 | 13.9/0.0 | fair→fair | 1.000000 |
| q4 | 1 | 2 | 156.5 | 42.00 | 13.9/0.0 | fair→fair | 1.000000 |
| q8 | 1 | warm | 109.5 | 51.16 | 23.2/0.0 | fair→fair | — |
| q8 | 1 | 1 | 147.4 | 51.30 | 23.2/0.0 | fair→fair | 1.000000 |
| q8 | 1 | 2 | 163.0 | 51.17 | 23.2/0.0 | fair→fair | 1.000000 |
| pruna | 1 | warm | 105.2 | 61.27 | 40.6/0.0 | fair→fair | — |
| pruna | 1 | 1 | 148.7 | 61.16 | 40.6/0.0 | fair→fair | 1.000000 |
| pruna | 1 | 2 | 168.5 | 61.22 | 40.6/0.0 | fair→fair | 1.000000 |
| bf16 | 1 | warm | 124.2 | 68.59 | 40.6/0.0 | fair→fair | — |
| bf16 | 1 | 1 | 164.5 | 68.74 | 40.6/0.0 | fair→fair | 1.000000 |
| bf16 | 1 | 2 | 185.4 | 68.58 | 40.6/0.0 | fair→fair | 1.000000 |

## Blocks (load + ratchet detector)

| arm | block | prewarm s | load s | post-drop floor GB |
|---|---|---|---|---|
| bf16 | 0 | 2.2 | 1.4 | 4.65 |
| pruna | 0 | 3.3 | 1.3 | 5.28 |
| q8 | 0 | 4.2 | 0.7 | 5.84 |
| q4 | 0 | 2.5 | 0.5 | 6.39 |
| q4 | 1 | 1.1 | 0.5 | 6.40 |
| q8 | 1 | 1.4 | 0.6 | 6.43 |
| pruna | 1 | 5.8 | 1.2 | 6.45 |
| bf16 | 1 | 2.7 | 1.2 | 6.46 |

## Arm stats (measured runs only)

| arm | median s | min–max s | median peak GB |
|---|---|---|---|
| bf16 | 173.3 | 148.1–185.4 | 67.76 |
| pruna | 160.3 | 148.7–168.5 | 60.63 |
| q8 | 160.2 | 147.4–168.2 | 50.93 |
| q4 | 149.5 | 139.6–158.3 | 41.88 |

## Verdicts (vs `bf16`)

- `pruna`: Δmed -13.0s ≤ spread 37.3s — NOISE
- `q8`: Δmed -13.2s ≤ spread 37.3s — NOISE
- `q4`: Δmed -23.9s ≤ spread 37.3s — NOISE

## Cross-arm output (first measured run, vs `bf16`)

| arm | cosine | maxAbs | class |
|---|---|---|---|
| pruna | 0.998917 | 1.15211 | not gated (q8 nondeterminism / divergent-by-design) |
| q8 | 0.878022 | 2.04879 | not gated (q8 nondeterminism / divergent-by-design) |
| q4 | 0.832259 | 2.01041 | not gated (q8 nondeterminism / divergent-by-design) |

> Doctrine: totals here are UNPROFILED. For a stage split, run the same arm once under
> `MLX_PROFILE=csv` separately and never compare its timings across variants (PROFILING.md).
