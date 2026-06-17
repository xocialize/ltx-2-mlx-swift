# ltx-2-mlx-swift

Swift/MLX port of Lightricks **LTX-2.3** — a DiT-based foundation model that jointly
generates synchronized video **and** audio. Mirrors the Python oracle
[`dgrauet/ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx); integrates into MLXEngine as a
`ModelPackage`.

Unlike the Wan family this is a **standalone substrate** (Gemma-3 text encoder, 128-channel
VAE, joint-AV DiT, BigVGAN + BWE audio) — it does **not** reuse `wan-core`.

## License

- **This port code is licensed [Apache-2.0](LICENSE).** It is our own implementation. Lightricks
  releases their own LTX-2 *inference code* (`ltx-core` / `ltx-pipelines`, per the LTX-Desktop
  `NOTICES.md`) under Apache-2.0 — inference code is not treated as a derivative of the weights.
- **The model weights are NOT included and are NOT Apache-2.0.** LTX-2.3 weights ship under the
  **[LTX-2 Community License](https://huggingface.co/Lightricks/LTX-2.3)** (source-available, with a
  §2 revenue gate and a §A.20 non-compete). Obtain them from Lightricks / `mlx-community` and comply
  with that license — those terms bind any *distribution* of generated output, independent of this code.
- Parity goldens (`parity/goldens/`) are forward-outputs of the weights → weight-derivatives →
  gitignored, never distributed.

## Capabilities

Distilled **text-to-video** (one-stage + two-stage upsampled) and **image-to-video** (first-frame
conditioning via `T2VRequest.initImage`), each across a **bf16 / int8 / int4** quant ladder. Output
is an MP4 with synchronized 48 kHz stereo audio (the joint DiT denoises video + audio together).

Pipeline: `prompt → Gemma-3 (49 hidden states) → connector → joint-AV DiT → distilled Euler
denoise → 128-ch video VAE decode (+ audio VAE → BigVGAN + BWE) → MP4`.

## Parity

Every component is gated numerically against the oracle (per-op cosine ≥ 0.999; most hit ~1.0):
connector, gemma, text-encode, dit-tiny, dit-full, **dit-q8 / dit-q4** (quant), **dit-pertoken**
(i2v per-token timesteps), vae-decode / vae-encode, denoise, e2e, audio-vae-decode, vocoder,
audio-decode, upsampler, upscale-step. (MLX-Swift ↔ MLX-Python is not bit-identical across a
multi-step trajectory — that's expected FP non-determinism; the gate is per-op + perceptual.)

## Layout

- `Sources/LTX2` — engine-agnostic functional cores (RoPE, Gemma, Connector, DiT, DenoiseLoop,
  Video/Audio VAE, Vocoder, Upsampler, Positions, LTX2Pipeline).
- `Sources/MLXLTX2` — the MLXEngine `ModelPackage` wrapper (`MLXLTX2Package`, `LTX2Configuration`,
  `FrameCodec`, `ImageInput`).
- `Sources/RunLTX2` — CLI parity-gate driver.
- `parity/dump_*_goldens.py` — oracle golden dumpers (run in the `ltx-2-mlx` uv env).

## Build / gates

Depends on the [`xocialize/mlx-swift-lm`](https://github.com/xocialize/mlx-swift-lm) fork
(branch `ltx/gemma-all-hidden-states`, the `allHiddenStates` text-encoder seam) as a local
path dep at `../mlx-swift-lm`.

```bash
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
xcrun swift run -c release RunLTX2 --dit-full-gate
```
