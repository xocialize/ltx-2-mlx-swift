# CLAUDE.md — ltx-2-mlx-swift (the Swift port)

Package-level navigation. **Methodology, quirks, license, and the determinism doctrine are
in the parent [`../CLAUDE.md`](../CLAUDE.md)** (auto-loads) — don't duplicate them here.

## Source map (`Sources/LTX2/` — engine-agnostic functional cores)

| File | What | Gate |
|---|---|---|
| `RoPE.swift` | split-type RoPE (log-spaced freq grid, fractional positions) — shared by connector + DiT | (via others) |
| `GemmaEncoder.swift` | Gemma-3 load (stock mlx-swift-lm) + combined causal+padding mask + tokenize | `--gemma-gate` |
| `Gemma3+AllHiddenStates.swift` | the 49-state encoder tap itself — uniform mask on every layer, per-layer `eval` (watchdog), per-layer `Task.checkCancellation()`. Ours, via `@_spi(GemmaEncoder) import MLXLLM` | `--gemma-gate` |
| `Connector.swift` | `GemmaFeaturesExtractorV2` + `Embeddings1DConnector` (49-layer RMS, dual project, gated attn, GEGLU, registers) — **fp32** | `--connector-gate` |
| `DiT.swift` | joint-AV Diffusion Transformer (48 blocks, AdaLN ×4 kinds, self/text-cross/AV-cross attn) — **bf16**. Quant-aware `dense()` (q8/q4, bits auto). Optional **per-token timesteps** (i2v). | `--dit-tiny`, `--dit-full`, `--dit-q8`, `--dit-q4`, `--dit-pertoken` |
| `DenoiseLoop.swift` | distilled Euler (X0Model + euler_step). `run` = uniform-mask t2v; **`runConditioned`** = i2v (per-token σ + clean-latent re-blend) | `--denoise-gate` (+ i2v via `--dit-pertoken`) |
| `VideoVAE.swift` | 128-ch video VAE decoder + encoder (pixel-shuffle, PixelNorm, causal/non-causal) + `denormalizeLatent`/`normalizeLatent` — **fp32** | `--vae-decode`, `--vae-encode` |
| `AudioVAE.swift` | audio VAE decoder (Conv2d, causal-height, latent→mel) + encoder (waveform→Slaney-mel→latent, LipDub reference audio) — **fp32** | `--audio-vae-decode-gate`, `--audio-vae-encode-gate` |
| `Vocoder.swift` | BigVGAN v2 + Hann-sinc resampler + MelSTFT + BWE (mel→48kHz) — **fp32** | `--vocoder-gate` |
| `Upsampler.swift` | spatial-x2 latent upsampler (Conv3d, GroupNorm, PixelShuffle2D) — **fp32** | `--upsampler`, `--upscale-step` |
| `Positions.swift` | pixel-space video/audio positions + `distilledSigmas`/`stage2Sigmas` + LipDub audio-ref patchify/negative-time positions | (via `--audio-vae-encode-gate`) |
| `LTX2Pipeline.swift` | assembles all of the above → `t2v` (one-stage) / `t2vTwoStage` / **`i2v`** (first-frame conditioning); loads + decodes audio | `--e2e-gate`, `--audio-decode` |

`Sources/MLXLTX2/` — the engine wrapper: `MLXLTX2Package` (ModelPackage, `.textToVideo` incl. i2v
via `initImage`), `LTX2Configuration` (+ `WeightPrewarming` conformance), `FrameCodec` (frames→H.264
+ AAC-muxed MP4), `ImageInput` (decode/preprocess the i2v init frame). `Sources/RunLTX2/` — the
parity-gate CLI. `parity/` — Python golden dumpers + (gitignored) `goldens/`.

## Conventions

- Build: `DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun swift build`.
- Cores are functional (`[String: MLXArray]` + explicit ops keyed by oracle weight-key strings);
  fp32 except the DiT (bf16). New component → port from oracle → `parity/dump_*` golden →
  `RunLTX2 --*-gate` (cosine ≥0.999) → commit.
- `mlx-swift-lm` is a **local path-dep** (`../mlx-swift-lm`) checked out at plain **upstream
  `main`** — no fork, no local patches (upstream #387 / `6608a35` exposes `@_spi(GemmaEncoder)`;
  the tap is ours). **Transitional**: no release tag contains #387 yet, so don't ship off this
  pin; adopting a tag is a one-line `Package.swift` edit. Goldens are gitignored + regenerable —
  `parity/dump_text_encode_goldens.py` writes both the `.npy` set and the packed
  `goldens.safetensors` the Swift gates read. **Page the weights in first**
  (`cat <model>/*.safetensors > /dev/null`) or the oracle dump trips the GPU watchdog.
- **Repo is Apache-2.0 and published** (`xocialize/ltx-2-mlx-swift`) — the port code is our own
  implementation (license stance reversed 2026-06-16, see `../CLAUDE.md` §License). **Never commit**
  the converted weights or `parity/goldens/` — those are LTX-2 weight-derivatives (Community-licensed)
  and stay gitignored.
