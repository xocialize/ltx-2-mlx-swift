#!/usr/bin/env python
"""Full-scale DiT parity fixture: real distilled weights, production config.

Loads the real LTX-2.3 distilled transformer (bf16) at the checkpoint config,
feeds modest random latents + REAL text embeds (the connector outputs captured by
dump_text_encode_goldens) + pixel-space positions, runs one forward, and dumps
inputs + velocity outputs. The Swift port loads the SAME transformer file and
gates against these. Confirms production dims + real weights + dtype (the regime
the tiny gate can't cover — cf. the connector's 188160-projection bf16 NaN).

Weights are NOT dumped (35GB — Swift loads transformer-distilled.safetensors直接).

Run in the oracle uv env:
    cd ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx && \
        uv run python ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/dump_dit_full_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
from ltx_core_mlx.utils.positions import compute_video_positions, compute_audio_positions
from ltx_core_mlx.utils.weights import load_split_safetensors

MODEL_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
TE = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/text_encode/goldens.safetensors")
OUT = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/dit_full")

# Modest sizes to bound memory: 3 latent frames at 8x8 → Nv=192; Na=16.
F, H, W, Na, FPS = 3, 8, 8, 16, 24.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = LTXModelConfig.from_checkpoint_dir(MODEL_DIR)
    print("config: layers=%d video_dim=%d audio_dim=%d" % (cfg.num_layers, cfg.video_dim, cfg.audio_dim))
    model = LTXModel(cfg)
    weights = load_split_safetensors(MODEL_DIR / "transformer-distilled.safetensors", prefix="transformer.")
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    te = mx.load(str(TE))
    video_text = te["video_embeds"]   # (1, 1024, 4096)
    audio_text = te["audio_embeds"]   # (1, 1024, 2048)

    Nv = F * H * W
    video_positions = compute_video_positions(F, H, W, frame_rate=FPS)  # (1, Nv, 3)
    audio_positions = compute_audio_positions(Na)                        # (1, Na, 1)

    mx.random.seed(7)
    video_latent = mx.random.normal((1, Nv, cfg.video_patch_channels)).astype(mx.float32)
    audio_latent = mx.random.normal((1, Na, cfg.audio_patch_channels)).astype(mx.float32)
    sigma = mx.array([0.7]).astype(mx.float32)

    video_v, audio_v = model(
        video_latent=video_latent, audio_latent=audio_latent, timestep=sigma,
        video_text_embeds=video_text, audio_text_embeds=audio_text,
        video_positions=video_positions, audio_positions=audio_positions,
    )
    mx.eval(video_v, audio_v)

    io = {
        "video_latent": video_latent, "audio_latent": audio_latent, "sigma": sigma,
        "video_text": video_text, "audio_text": audio_text,
        "video_positions": video_positions, "audio_positions": audio_positions,
        "video_v": video_v, "audio_v": audio_v,
    }
    mx.save_safetensors(str(OUT / "io.safetensors"), {k: v.astype(mx.float32) for k, v in io.items()})
    print("wrote", OUT)
    print("Nv=%d Na=%d" % (Nv, Na))
    print("video_v", video_v.shape, "mean=%.5f std=%.5f" % (float(video_v.mean()), float(video_v.std())))
    print("audio_v", audio_v.shape, "mean=%.5f std=%.5f" % (float(audio_v.mean()), float(audio_v.std())))


if __name__ == "__main__":
    main()
