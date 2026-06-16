# ltx-2-mlx-swift

Swift/MLX port of Lightricks **LTX-2.3** — a DiT-based foundation model that jointly
generates synchronized video **and** audio. Mirrors the Python oracle
[`dgrauet/ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx); intended for integration
into MLXEngine.

Unlike the Wan family this is a **standalone substrate** (Gemma-3 text encoder, 128-channel
VAE, joint-AV DiT, BigVGAN + BWE audio) — it does **not** reuse `wan-core`.

> ⚠️ **License posture — eval-only.** LTX-2.3 ships under the **LTX-2 Community License**
> (source-available, non-Apache: §2 revenue gate, §3 derivative-copyleft, §A.20 non-compete).
> This port is treated as an **eval-only / gated specialty** for internal capability
> evaluation — **not shippable** and **not published** (the port code + converted weights are
> "Derivatives" under §3). Kept local by design.

## Status

Text-encode front-end **ported and parity-validated** against the oracle:

| Component | Gate | Result (cosine vs oracle) |
|---|---|---|
| Connector (dual-projection + 8× gated-attn/GEGLU + split-RoPE + registers) | `--connector-gate` | video 0.999988 / audio 0.999668 |
| Gemma 3 49-state extraction (Path-A reuse of `mlx-swift-lm`) | `--gemma-gate` | worst layer 0.999338 |
| Composed token_ids → Gemma → connector | `--text-encode-gate` | video 0.999989 / audio 0.999652 |

Remaining (Path-B cores): joint-AV DiT, 128-ch video VAE (streaming), audio VAE + BigVGAN +
BWE, neural upsampler; then the distilled denoise loop and the MLXEngine `ModelPackage` wrapper.

## Layout

- `Sources/LTX2` — engine-agnostic core (`RoPE`, `Connector`, `GemmaEncoder`).
- `Sources/RunLTX2` — CLI parity-gate driver.
- `parity/dump_text_encode_goldens.py` — generates oracle goldens (run in the `ltx-2-mlx` uv env).

## Build / gates

Depends on the [`xocialize/mlx-swift-lm`](https://github.com/xocialize/mlx-swift-lm) fork
(branch `ltx/gemma-all-hidden-states`, the `allHiddenStates` text-encoder seam) as a local
path dep at `../mlx-swift-lm`.

```bash
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
xcrun swift run -c release RunLTX2 --text-encode-gate
```
