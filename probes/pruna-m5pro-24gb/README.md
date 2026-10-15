# AB-A-0005 on the M5 Pro 24 GB: stock vs Pruna decoder at compact24 — wall-clock, peak, and the lean-decoder declaration

**2026-10-15, MacBook Pro (M5 Pro, 16-core GPU, 24 GB), macOS 27.2, release `RunLTX2` at v0.18.0 (2f49d4e) plus
the `LTX_MODELS_ROOT` harness override in this change, GPU otherwise idle.** The two items BRIDGE-LTX-014 left open
and AB-A-0005 carried: (1) the 24 GB-target wall-clock and peak with the Pruna decoder against stock, alternating
repeats, at the compact24 tier's own geometry; (2) the lean-decoder split from the staged instrument, so the
`QuantFootprint` / `peakActivationBytesHint` for a Pruna variant would be measured rather than guessed.

## Setup, and what differs from the Mac Studio

- **Weights**: the archive's own trees, copied from the Satechi volume and verified file-for-file (`ltx-granules-25/q8`,
  `xocialize/ltx-2.5-mlx` components, `xocialize/ltx-2.5-mlx-q8`'s int8 Gemma-4, and `dgrauet/ltx-2.3-mlx/
  vae_decoder_pruna.safetensors` — the video VAE is byte-identical across 2.3 and 2.5, AB-R-0002 — linked into the 2.5
  trees as `vae_decoder_pruna.safetensors`). No transformer checkpoint: the granule tree is a v2 manifest, so the run
  uses `LTX_NO_CHECKPOINT=1`, the hosted deployment shape the harness already supports.
- **Instrument**: `--t2v-spot25` at `LTX_TIER=compact24 LTX_QUANT=int8 LTX_STREAM_25=1 LTX_ENC=q8` — the shipping
  low-tier configuration (streamed q8 DiT, int8 encoder, `.forceStream` from the profile), one fresh process per run
  under `/usr/bin/time -l`, swap and thermal state sampled around each run, 45 s cooldown. The budget is the governor's
  own answer on this host (`recommendedMaxWorkingSetSize` = 19.07 GB), printed with its source (AB-T-0081); the
  legacy 0.7×nominal 16.80 GB is printed alongside for comparison with the Max's receipts.
- **Decoder selection**: `LTX_VAE_DECODER=stock|pruna`. The fallback note only prints under `MLX_PROFILE=1`, which
  breaks fusion, so it was checked once on a profiled smoke run (`logs/smoke-pruna-profiled.full.log`:
  `LTX_VAE_DECODER=pruna → vae_decoder_pruna.safetensors`; the stock run carries no note) and every measured run keeps
  its clip (`logs/*.mp4`) so a silent fallback would show as a stock-identical output.
- **Protocol** (AB-R-0067): three throwaway burn-in generations first (a freshly started box ramps across a session,
  which ABBA cannot cancel), one excluded warm-up per geometry (shape-specific kernel compile), then the measured arms
  in ABBA + BAAB order — four stock and four Pruna runs at the tier's clamped envelope, 512×288×121. Prompt "a cat
  playing piano", seed 42, as the desktop Pruna receipt (`bench_e2e_ladder-pruna-121f_20260804-012008`).
- **Diagnostic, never an acceptance number**: the desktop receipt's 704×512×121 geometry, forced past the profile's
  resolution clamp with `LTX_ENVELOPE_OVERRIDE=1` (the harness announces it), two runs per arm in BAAB order — the
  decode-heavy case, and what a 24 GB box does above its envelope.
- **Toolchain deviation**: this station's active Xcode is the 27A5209h beta (Swift 6.4.0.23), which cannot compile
  swift-collections 1.7.1; the release Xcode 27A266a compiles it but has no Metal toolchain installed here. The build
  therefore pins swift-collections to 1.6.0 locally (every dependent allows ≥ 1.0.0; the pin is not committed) and
  uses the beta toolchain that also built the SenseNova receipt on this station (AB-R-0551). Nothing numeric depends
  on swift-collections.

## Runs — measured, compact24's own envelope (512×288×121, t2v)

Generation time is the harness's `run` (encode → denoise → decode, after load); wall is `time -l`'s real (load
included); resident / activation / peak are the staged instrument's split (`resident(held)` = low-water during the
run, AB-R-0106's corrected instrument); footprint is `time -l`'s peak `phys_footprint`. Burn-in and warm-up rows are
excluded from every statistic.

| run | decoder | gen s | wall s | resident GB | activation GB | peak GB | footprint GB | swap Δ | budget 19.07 GB |
|---|---|---|---|---|---|---|---|---|---|
| burn-in 1 | stock | 78.9 | 107.6 | 0.85 | 13.64 | 14.49 | 14.49 | 0 | within |
| burn-in 2 | stock | 77.9 | 106.7 | 0.90 | 13.68 | 14.58 | 14.58 | +0.05 | within |
| burn-in 3 | stock | 77.2 | 105.5 | 0.99 | 13.67 | 14.66 | 14.66 | 0 | within |
| warm-up | stock | 76.6 | 105.0 | 0.93 | 13.75 | 14.68 | 14.68 | 0 | within |
| rep 1 (A) | **stock** | **76.6** | 104.9 | 0.87 | 13.70 | 14.57 | 14.57 | 0 | within |
| rep 1 (B) | **pruna** | **64.1** | 92.1 | 0.87 | 13.63 | 14.50 | 14.53 | 0 | within |
| rep 2 (B) | **pruna** | **65.3** | 93.2 | 0.88 | 13.60 | 14.49 | 14.49 | 0 | within |
| rep 2 (A) | **stock** | **76.5** | 104.9 | 1.00 | 13.56 | 14.56 | 14.56 | −0.02 | within |
| rep 3 (B) | **pruna** | **64.6** | 92.7 | 1.12 | 13.37 | 14.49 | 14.51 | 0 | within |
| rep 3 (A) | **stock** | **78.2** | 106.8 | 0.89 | 13.69 | 14.58 | 14.58 | 0 | within |
| rep 4 (A) | **stock** | **75.3** | 103.6 | 0.91 | 13.76 | 14.67 | 14.67 | 0 | within |
| rep 4 (B) | **pruna** | **63.4** | 91.3 | 0.99 | 13.51 | 14.50 | 14.51 | 0 | within |

| arm | n | median gen | min–max (range) | peak max | resident max | activation max | footprint max |
|---|---|---|---|---|---|---|---|
| stock | 4 | **76.5 s** | 75.3–78.2 (2.9 s) | 14.67 GB | 1.00 GB | 13.76 GB | 14.67 GB |
| pruna | 4 | **64.3 s** | 63.4–65.3 (1.9 s) | 14.50 GB | 1.12 GB | 13.63 GB | 14.53 GB |

**Δmedian pruna − stock = −12.2 s (−15.9 %) against stock's within-arm range of 2.9 s → MEASURABLE** (the bench-e2e
doctrine: |Δmed| must exceed the baseline arm's spread; the desktop run's −13.0 s was NOISE against a 37.3 s spread).
The thermal state read nominal before the protocol and fair for the measured runs (`logs/thermal-samples.log`, 30 s
samples, 26 nominal / 27 fair over the session); the mirrored block order carries any residual drift, and the two
arms never overlap (every Pruna run is faster than every stock run, by at least 10 s). Swap did not move.

For scale, the Mac Studio's compact24 t2v runs at the same geometry (AB-R-0106 logs) were 65.2–67.3 s with the
stock decoder at a 14.56–14.58 GB peak; this 24 GB M5 Pro runs the same work in 75–78 s stock and 63–65 s with
Pruna.

## Findings

1. **Pruna is a real time win on the 24 GB tier: −12.2 s of 76.5 s (−15.9 %) per clip, MEASURABLE.** This is
   the result the ask predicted and the desktop could not resolve: on the low tier decode is a large enough share of
   a short run, and this box is stable enough (within-arm range 2–3 s against the desktop's 37 s), for a 12 s
   difference to clear the noise bar by 4×. The win concentrates where the ask said it would.
2. **Memory does not move: peak 14.50 GB with Pruna vs 14.67 GB stock (−0.17 GB), both within the 19.07 GB budget
   this machine's governor enforces, and within the legacy 16.80 GB figure.** On the streamed low tier the peak stage
   is the encoder-bound activation (13.4–13.8 GB), not the decode window, so a leaner decoder changes the time of the
   decode phase and barely its memory. The desktop's −7.1 GB peak saving was a resident-DiT, bf16, unevicted
   regime that the shipping low tiers no longer run.
3. **The lean-decoder declaration, measured (item 2 of the ask): worst case over four runs, resident 1.12 GB +
   activation 13.63 GB = 14.75 GB (peak 14.50) for t2v at 512×288×121 with Pruna, against stock's 1.00 + 13.76 =
   14.76 GB.** The current compact24 declaration (16.20 = 0.8 + 15.4, AB-R-0107, from AB-R-0106's i2v worst case)
   covers both with 1.45 GB to spare, so promoting Pruna on this tier needs **no re-declaration**; the split's
   movement here (resident ~0.5 GB higher, activation ~0.5 GB lower than the Max's 0.49 / 14.15) is the two
   instruments' low-water sampling landing on different stages, with the sum within 0.1 GB. i2v with Pruna was not
   measured (the ~4.9 GB i2v-adapter was not staged); its activation is content-sensitive and the i2v row is the one
   AB-R-0106 flagged at 92 %, so a Pruna i2v row is the remaining gap if the promotion is meant to cover i2v.
4. **Decoder selection is verified, not assumed.** The profiled smoke run logs `LTX_VAE_DECODER=pruna →
   vae_decoder_pruna.safetensors` and the stock run no note; every Pruna clip differs from every stock clip (16/16
   pairs), while clips within an arm are not byte-identical either (0/6 pairs in each arm, the port's known run-to-run
   wobble, AB-R-0052), so the per-run evidence of the decoder is the timing signature itself: the two arms are
   disjoint by more than 10 s on every run.
5. **The 24 GB machine runs this tier with headroom and no swap.** Peak 14.5–14.7 GB against 19.07 GB, the process
   footprint equal to the MLX peak, swap flat across all twelve runs, load 28 s per fresh process from the local
   disk, streaming from the q8 granules under `.forceStream` with no fallback.

## Diagnostic — the desktop receipt's geometry, forced past the clamp (704×512×121, `LTX_ENVELOPE_OVERRIDE=1`; never an acceptance number)

| run | decoder | gen s | wall s | resident GB | activation GB | peak GB | footprint GB | swap Δ GB | vs 19.07 GB |
|---|---|---|---|---|---|---|---|---|---|
| warm-up | stock | 183.8 | 213.2 | 0.87 | 19.98 | 20.85 | 21.00 | +2.73 | OVER |
| rep 1 (B) | **pruna** | **150.5** | 179.3 | 0.90 | 13.58 | **14.48** | 14.50 | −0.24 | within |
| rep 1 (A) | **stock** | **180.1** | 209.5 | 1.02 | 20.10 | **21.11** | 21.12 | +1.40 | OVER |
| rep 2 (A) | **stock** | **180.7** | 209.8 | 0.88 | 20.03 | **20.91** | 20.93 | +0.91 | OVER |
| rep 2 (B) | **pruna** | **148.3** | 182.7 | 0.91 | 13.58 | **14.49** | 14.61 | −0.09 | within |

stock: median 180.4 s (range 0.6), peak 20.9–21.1 GB, over budget, 0.9–2.7 GB pushed to swap on every run.
pruna: median 149.4 s (range 2.2), peak 14.5 GB, within budget with 4.6 GB spare, swap flat or shrinking.
**Δmedian −31.0 s (−17.2 %) against a 0.6 s spread → MEASURABLE**, and the peak moves by **−6.6 GB**.

6. **Where decode binds, Pruna moves the peak, not just the clock.** At the tier's own envelope the activation peak is
   the encoder/denoise stage (13.4–13.8 GB) and Pruna changes it by 0.2 GB. At 704×512×121 the decode window is the
   binding stage for the stock decoder (activation 20.0–20.1 GB), and the lean decoder takes the peak from 21.1 to
   14.5 GB — the desktop's −7.1 GB class of saving, now in the regime that matters: a run that is over budget and
   swapping on a 24 GB machine with stock finishes within budget, without swap, and 17 % faster with Pruna. That is
   the strongest argument for Pruna on the 24 GB tier: it is what would let compact24's resolution envelope grow
   from 512×288 toward 704×512 at 121 frames on 24 GB machines. It is a diagnostic — acceptance needs the profile to
   permit the geometry (a profile change) and the repeat protocol at that geometry; AB-D-0014's tier table is the
   operator's.

## Verdict

Both owed items landed on the machine they were waiting for. **(1)** Pruna on the 24 GB tier is a measurable time
win: −12.2 s of 76.5 s (−15.9 %) at compact24's envelope with the peak unchanged, and −17.2 % with a −6.6 GB peak at
the decode-heavy 704×512 geometry. **(2)** The lean-decoder declaration for compact24 t2v, worst case over four
runs, is resident 1.12 GB + activation 13.63 GB (peak 14.50 GB) — inside the tier's current 16.20 GB charge with
1.45 GB to spare, so a t2v promotion needs no re-declaration; i2v with Pruna is the one row not measured here.
Stock stays the default until the operator promotes it, as the ask required; the data says promote on compact24
(and by the same mechanism balanced32) for t2v, measure the Pruna i2v row first if the promotion is to cover i2v,
and treat a wider compact24 resolution envelope under Pruna as the follow-up worth a profile change.

## Files

`run.log` (every run as printed, with swap and thermal state before and after), `logs/*.full.log` (the complete
output per run), `logs/*.mp4` (each run's clip — kept on the M5 Pro station, not committed: ~20 MB), `logs/smoke-*` (the setup checks), `run-pruna-ab.sh` (the exact
protocol).
