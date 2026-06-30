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

## Memory

Per-stage **load → use → evict** (MLXEngine 1.14 efficiency contract): only the DiT backbone stays
resident through the denoise; the Gemma+Connector text encoder evicts before it, and the VAE-decode
stack / two-stage encoder+upsampler load only around their phases. Because weights are lazy-mmap, the
DiT isn't even materialized during text-encode — encoder and DiT are never co-resident, so the peak is
the DiT denoise **alone**. The manifest declares a **split footprint** (`residentBytes` weights +
`peakActivationBytes` transient); the engine reserves one transient across residents (serialized
inference), so LTX co-resides with other models far better than a single-number floor allows.

Measured in-app (M5 Max / 128 GB, two-stage **704×512×9f**, seed 42; peak scales with resolution×frames):

| Quant | Resident (DiT) | Activation | **Peak** | vs. old all-resident |
|---|---|---|---|---|
| bf16 | 38.9 GB | 13.1 GB | **52.0 GB** | 82.8 GB |
| int8 (q8) | 21.5 GB | 12.1 GB | **33.6 GB** | 62.4 GB |
| int4 (q4) | 12.2 GB | 15.1 GB | **27.3 GB** | 53.3 GB |

Activation is ~dtype-independent (same bf16 compute). int4 is artifact-free but **diverges the distilled
sample** vs bf16/q8 (larger per-step quant error). Re-measure (`RunLTX2 --mem-bench`) before claiming a
larger res/frame envelope fits a tier.

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
xcrun swift run -c release RunLTX2 --mem-bench bf16   # split-footprint harness [bf16|int8|int4]
```

> The `--mem-bench` CLI can trip the Metal GPU watchdog on the full two-stage run (beta-OS flakiness);
> the reliable measurement surface is the in-app autorun (`LTX_AUTORUN=1 LTX_QUANT=bf16` in
> LTXVideoTesting), which has the engine's weight prewarm + governor.
