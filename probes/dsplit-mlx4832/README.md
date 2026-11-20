# mlx#4832 follow-up: the d-split "wrong attention" is fully masked rows (2026-11-20)

Context: AB-R-1009 reported that with ml-explore/mlx#4903 applied, the head-dim-split NAX SDPA
kernel drops LTX's `--gemma-gate` (Gemma 3 12B 4-bit, 49 states) from mean cosine 0.999866 to
0.924872. bazyli (ml-explore/mlx#4832, issuecomment-7033226457) could not reproduce on an M5 Pro
and asked for the layer-0 tensors plus a prebuilt-metallib run on our M5 Max.

**Machine:** M5 Max 128 GB, macOS 27.3. **Stack:** Python 3.12, pip `mlx==0.32.4`,
`mlx-metal==0.32.4` (prebuilt metallib, no JIT), `mlx-lm==0.32.2`. Gemma weights:
`/Volumes/DEV_VOL1/Models/Models/mlx-community/gemma-3-12b-it-4bit`. Goldens:
`parity/goldens/text_encode/goldens.safetensors` (made on mlx 0.31.1; 21 valid tokens, 1003 left-pad).

## Findings

1. **Not JIT, not mlx-swift.** The prebuilt metallib reproduces it: fused tap mean cosine
   **0.929045** (worst 0.838906 @ layer 33), against 0.924872 for Swift JIT + #4903.
   Chunking queries into 512-row pieces (below the d-split threshold) reproduces the goldens
   **bit-for-bit** (`gemma3_tap.py`, `tap_*.json`).
2. **Only fully masked (left-padded) query rows differ.** Fused tap: valid positions mean
   cosine **0.999998** (worst 0.999984), padded positions **0.916744** (worst 0.790905).
3. **Per layer, both paths on identical inputs** (`per_layer.py`, `per_layer.json`): valid rows
   agree at every layer (cos ≥ 0.99977). Padded rows agree exactly in most layers and diverge in
   a few: layer 4 0.933, layer 28 0.949, layer 2 0.973, layer 37 0.972, layer 17 0.978.
4. **Mechanism.** In a fully masked row every logit is about -1e9 + s (bf16(-1e9) = -998244352).
   - The unfused path computes `q*scale`, the matmul and `+mask` in bf16 (spacing 2^22 near 1e9), so s
     vanishes and the row becomes a uniform average over the keys tied at the row's largest mask
     value. Measured on layer 4: uniform vs chunked output cos **0.999997** (`emulate.py`).
   - The steel/NAX kernels compute `S*scale*log2e + log2e*mask` in fp32
     (`steel_attention_nax.h`, both `attention_nax` and `attention_nax_dsplit`). Near
     log2e·mask = -1.44e9 the fp32 spacing is 128, so scaled scores below 64/log2e ≈ **44.4** round
     away (uniform row) and larger ones survive and take the row.
   - Neither is the exact answer: exact fp64 softmax vs chunked cos 0.565, vs fused 0.607 (layer 4).
   - ⚠️ My fp32 emulation of the NAX kernel matches the real fused output only at cos 0.962, so
     it does not reproduce the exact surviving key sets. The uniform half is exact; the NAX half
     rests on the kernel source plus the synthetic threshold below.
5. **Not specific to d-split** (`synthetic.py`, `synthetic.out`, no weights): with head dim 128 the
   regular fused kernel leaves the uniform answer once scores pass ~44 (score max 23.0 → uniform,
   49.6 → cos 0.174), at qL 1024 and at qL 256. With head dim 256 only qL ≥ 1024 (d-split) does,
   because shorter queries go unfused. Valid rows agree everywhere. Head-256 + qL ≥ 1024 + array
   mask simply went unfused until v0.32.4, so our goldens encode the uniform convention.

## Consequence for LTX

Benign for the pipeline as far as the documented mechanisms go: `GemmaFeaturesExtractorV2` zeroes
padded positions and `_replace_padding_with_registers` overwrites them (pad-token probe receipt:
pad-position hidden states can differ by 1e5 with bit-identical connector output). Valid positions
match at ≥ 0.99998. **Not measured here:** connector output under d-split; the 2.3 connector weights
are not on this station (AB-T-0494). `--gemma-gate` compares all 1024 positions whole-array, so it
reads this difference as a failure. The fleet fork keeping d-split off (AB-D-0264) is now a
gate-convention choice, not a correctness fix.

## `--gemma-gate` now reports the split (2026-11-20)

`RunLTX2 --gemma-gate` prints valid-only and padded-only cosines next to the whole-array one.
PASS/FAIL stays on whole-array so earlier receipts remain comparable. Shipping stack (fork
mlx-swift 0.33.1, d-split off), Release, `gemma-gate-fork0331.out`:

| | mean | worst |
|---|---|---|
| whole-array | 0.999866 | 0.999338 (layer 25), unchanged from the 0.32.3 baseline |
| valid-only, 21 positions | 0.999999 | 0.999992 (layer 48) |
| padded-only, 1003 positions | 0.999850 | 0.999268 (layer 25) |

**Not run:** a Swift build with the d-split route on. The valid-only figure for that arm comes from
the Python prebuilt run above (0.999998). Building it needs an mlx-swift checkout whose vendored mlx
has #4903 without the fleet's disable patch.

## Files

- `gemma3_tap.py <goldens> <gemma_dir> <out> fused|chunked`: 49-state tap, cosine split valid/pad; fused mode saves layer-0 tensors
- `per_layer.py <goldens> <gemma_dir> <out> [save_layer]`: fused vs chunked at every layer on golden-path inputs
- `layer0_check.py`, `emulate.py <layerNN.safetensors>`: path comparisons and fp32/bf16 emulations
- `synthetic.py`: self-contained repro (seeded)
- `DRAFT-reply-4832.md`: the reply as drafted; the operator edited it slightly and posted it as
  ml-explore/mlx#4832 issuecomment-7045337162, which is the version of record
- Tensors (weight-derived activations, gitignored): `parity/goldens/dsplit_mlx4832/layer0_sdpa_inputs.safetensors`,
  `layer04_sdpa_inputs.safetensors` (q, k, v, mask, fused and chunked outputs, attention_mask; scale 0.0625)
