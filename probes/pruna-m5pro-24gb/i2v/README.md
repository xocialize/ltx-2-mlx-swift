# AB-T-0297 step 1 on the M5 Pro 24 GB: compact24's i2v row, stock vs Pruna decoder

**2026-10-15, MacBook Pro (M5 Pro, 16-core GPU, 24 GB), macOS 27.2, release `RunLTX2` at 44d0192 (v0.18.0 plus the
`LTX_MODELS_ROOT` override), GPU otherwise idle.** The row AB-R-0561 did not measure and AB-D-0147 made the
condition for promoting the Pruna decoder: compact24 i2v at its own envelope, stock vs Pruna, AB-R-0561's protocol.

## Setup

Identical to the t2v receipt one directory up (`../README.md`), plus:

- **`LTX_I2V=1`**: `--t2v-spot25`'s i2v path — a synthetic clean first frame (a footprint arm; shapes drive cost, so
  image content is irrelevant, as AB-R-0106 did) and the **~4.9 GB i2v-adapter LoRA** resident, staged from the
  archive's `ltx-lora-cache/i2v-adapter.safetensors` (4,932,895,568 bytes, verified) under the same root override.
  The harness prewarms it with the components (12 files against t2v's 11) and the run reports `mode: i2v`.
- **Decoder selection** is the same mechanism AB-R-0561 verified on a profiled smoke run; every measured run keeps its
  clip (`logs/*.mp4`, kept on the station, not committed) and the arms are read from the timing signature as well.
- **Protocol**: two burn-in generations and one warm-up excluded, then four stock and four Pruna runs in ABBA + BAAB
  order at 512×288×121, 45 s cooldown, one fresh process per run under `/usr/bin/time -l`, swap and the process
  thermal state sampled around each run. Prompt "a cat playing piano", seed 42.
- **Budget** on this host: `recommendedMaxWorkingSetSize` = 19.07 GB (printed with its source); the compact24 tier's
  **declared charge is 16.20 GB** (0.8 + 15.4, AB-R-0107, from AB-R-0106's i2v worst case 0.49 + 15.02 = 15.49 on the
  Max), which is the number AB-T-0297's step-2 condition is measured against.

## Runs — compact24's own envelope, 512×288×121, i2v

| run | decoder | gen s | wall s | resident GB | activation GB | peak GB | footprint GB | swap Δ | vs 19.07 GB |
|---|---|---|---|---|---|---|---|---|---|
| burn-in 1 | stock | 87.1 | 120.3 | 0.77 | 15.31 | 16.08 | 16.08 | 0 | within |
| burn-in 2 | stock | 87.6 | 118.4 | 0.46 | 15.96 | **16.42** | 16.42 | 0 | within |
| warm-up | stock | 88.1 | 117.8 | 0.48 | 15.62 | 16.10 | 16.10 | 0 | within |
| rep 1 (A) | **stock** | **87.7** | 117.5 | 0.37 | 15.71 | 16.08 | 16.08 | 0 | within |
| rep 1 (B) | **pruna** | **76.0** | 105.7 | 0.34 | 15.51 | 15.85 | 15.85 | −0.01 | within |
| rep 2 (B) | **pruna** | **76.1** | 105.6 | 0.81 | 15.03 | 15.84 | 15.84 | 0 | within |
| rep 2 (A) | **stock** | **89.5** | 119.6 | 0.38 | 15.72 | 16.10 | 16.10 | 0 | within |
| rep 3 (B) | **pruna** | **75.9** | 105.5 | 0.65 | 15.19 | 15.83 | 15.83 | 0 | within |
| rep 3 (A) | **stock** | **89.0** | 119.0 | 0.75 | 15.33 | 16.07 | 16.07 | −0.02 | within |
| rep 4 (A) | **stock** | **86.4** | 116.0 | 0.70 | 15.37 | 16.07 | 16.07 | 0 | within |
| rep 4 (B) | **pruna** | **76.2** | 105.3 | 0.46 | 15.37 | 15.83 | 15.83 | −0.01 | within |

| arm | n | median gen | min–max (range) | peak max | resident max | activation max | footprint max |
|---|---|---|---|---|---|---|---|
| stock | 4 | **88.3 s** | 86.4–89.5 (3.1 s) | 16.10 GB | 0.75 GB | 15.72 GB | 16.10 GB |
| pruna | 4 | **76.0 s** | 75.9–76.2 (0.3 s) | 15.85 GB | 0.81 GB | 15.51 GB | 15.85 GB |

**Δmedian pruna − stock = −12.3 s (−13.9 %) against stock's within-arm range of 3.1 s → MEASURABLE.** Thermal state
fair for the measured runs (nominal at the start); swap did not move; every run within this host's 19.07 GB budget.

## Findings

1. **Pruna is the same class of win on i2v as on t2v: −12.3 s of 88.3 s (−13.9 %), MEASURABLE**, every Pruna run
   faster than every stock run by at least 10 s, with a 0.3 s spread on the Pruna arm. The i2v path costs about
   11 s over t2v on both decoders (the adapter and the image conditioning), so the decode saving is the same
   absolute 12 s and a slightly smaller share.
2. **The i2v row is the tight one, and Pruna buys it 0.25–0.35 GB.** i2v peaks 16.07–16.10 GB stock vs 15.83–15.85 GB
   Pruna — about 1.4 GB over the t2v peaks, landing in activation, as AB-R-0106 found on the Max (0.9–1.7 GB,
   activation not residency). One excluded burn-in run with stock reached **16.42 GB, over the tier's declared
   16.20 GB charge**; no Pruna run came within 0.35 GB of it.
3. **AB-T-0297's step-2 condition holds: Pruna's i2v peak, 15.85 GB worst of four, fits the compact24 charge of
   16.20 GB (0.8 + 15.4, AB-R-0107) with 0.35 GB to spare**, so no re-declaration is required. Per component, the
   worst Pruna resident is 0.81 GB and the worst activation 15.51 GB against hints of 0.8 and 15.4: the sum covers
   the measured peak, and each component sits within 0.1 GB of its hint on this machine's split. If the operator
   wants per-component headroom rather than sum-level, 0.9 + 15.6 = 16.5 GB still clears the legacy 16.80 GB and
   this host's 19.07 GB budgets.
4. **Everything else as in the t2v receipt**: streaming from the local q8 granules under `.forceStream`, no fallback,
   load 29 s per fresh process, decoder selection by the same verified mechanism, clips kept per run.

## Verdict

**Promote.** With t2v (AB-R-0561) and now i2v both measurable wins and the i2v peak inside the declared charge, the
condition AB-D-0147 set for making Pruna the default decoder on compact24 and balanced32 for both modes is met.

## Files

`run.log`, `logs/*.full.log` (every run complete), `logs/smoke-i2v-pruna-256x256x9.full.log` (the setup check),
`run-pruna-i2v-ab.sh` (the exact protocol); `../analyze.py` parses this log too.
