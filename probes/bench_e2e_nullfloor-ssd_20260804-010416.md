# bench-e2e receipt — nullfloor-ssd

- started: 2026-08-04 01:04:16 +0000 (UTC stamp 20260804-010416)
- host: Mac17,7 · 137 GB · Version 27.0 (Build 26A5388g)
- repo: 7a37941+dirty · geometry: 704x512x9f fps=24 one-stage · seed 42
- prompt: "a cat playing piano"
- protocol: blocks=2 (ABBA order) · runs/block=2 (+1 excluded warmup) · cooldown 45s · MLX_PROFILE off
- arms:
  - `A`: quant=bf16 decoder=stock
  - `B`: quant=bf16 decoder=stock

## Per-run

| arm | block | run | wall s | peak phys GB | mlx act/cache GB | thermal | cos(arm-first) |
|---|---|---|---|---|---|---|---|
| A | 0 | warm | 15.5 | 52.51 | 38.0/0.0 | nominal→nominal | — |
| A | 0 | 1 | 16.6 | 52.55 | 38.0/0.0 | nominal→nominal | — |
| A | 0 | 2 | 17.8 | 52.70 | 38.1/0.0 | nominal→nominal | 1.000000 |
| B | 0 | warm | 28.7 | 52.78 | 38.1/0.0 | fair→fair | — |
| B | 0 | 1 | 29.2 | 52.79 | 38.1/0.0 | fair→fair | — |
| B | 0 | 2 | 30.8 | 52.83 | 38.1/0.0 | fair→fair | 1.000000 |
| B | 1 | warm | 30.7 | 52.85 | 38.1/0.0 | fair→fair | — |
| B | 1 | 1 | 28.0 | 52.85 | 38.1/0.0 | fair→fair | 1.000000 |
| B | 1 | 2 | 26.6 | 52.85 | 38.1/0.0 | fair→fair | 1.000000 |
| A | 1 | warm | 23.4 | 52.86 | 38.1/0.0 | fair→fair | — |
| A | 1 | 1 | 22.4 | 52.87 | 38.1/0.0 | fair→fair | 1.000000 |
| A | 1 | 2 | 22.8 | 52.87 | 38.1/0.0 | fair→fair | 1.000000 |

## Blocks (load + ratchet detector)

| arm | block | prewarm s | load s | post-drop floor GB |
|---|---|---|---|---|
| A | 0 | 10.0 | 1.8 | 0.78 |
| B | 0 | 2.2 | 1.4 | 0.91 |
| B | 1 | 2.5 | 1.5 | 0.94 |
| A | 1 | 3.7 | 1.3 | 0.68 |

## Arm stats (measured runs only)

| arm | median s | min–max s | median peak GB |
|---|---|---|---|
| A | 20.1 | 16.6–22.8 | 52.78 |
| B | 28.6 | 26.6–30.8 | 52.84 |

## Verdicts (vs `A`)

- `B`: Δmed +8.5s > spread 6.2s — MEASURABLE

## Cross-arm output (first measured run, vs `A`)

| arm | cosine | maxAbs | class |
|---|---|---|---|
| B | 1.000000 | 0.00000 | same-weights bf16 (expect ≈1) |

> Doctrine: totals here are UNPROFILED. For a stage split, run the same arm once under
> `MLX_PROFILE=csv` separately and never compare its timings across variants (PROFILING.md).
