#!/usr/bin/env python
"""Two-stage upscale-step parity: half-res latent → denorm → upsample → renorm (fp32).

Validates the new numerical composition in LTX2Pipeline.t2vTwoStage:
vae_encoder.denormalize_latent → LatentUpsampler(2×) → vae_encoder.normalize_latent.

Run in the oracle uv env:
    cd ~/Development/ltx-2-mlx && \
        uv run python ~/Development/ltx-2-mlx-swift/parity/dump_upscale_step_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.model.video_vae.ops import remap_encoder_weight_keys
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.weights import load_split_safetensors

MODEL_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
OUT = Path("/Users/dustinnielson/Development/ltx-2-mlx-swift/parity/goldens/upscale_step")
F, H, W = 2, 4, 4


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    enc = VideoEncoder()
    ew = remap_encoder_weight_keys(load_split_safetensors(MODEL_DIR / "vae_encoder.safetensors", prefix="vae_encoder."))
    enc.load_weights(list({k: v.astype(mx.float32) for k, v in ew.items()}.items()))
    up = LatentUpsampler(in_channels=128, mid_channels=1024, num_blocks_per_stage=4,
                         spatial_upsample=True, spatial_scale=2.0, rational_resampler=False)
    raw = load_split_safetensors(MODEL_DIR / "spatial_upscaler_x2_v1_1.safetensors")
    stem = "spatial_upscaler_x2_v1_1."
    if raw and all(k.startswith(stem) for k in raw):
        raw = {k[len(stem):]: v for k, v in raw.items()}
    up.load_weights(list({k: v.astype(mx.float32) for k, v in raw.items()}.items()))
    mx.eval(enc.parameters(), up.parameters())

    mx.random.seed(41)
    half = mx.random.normal((1, 128, F, H, W)).astype(mx.float32)  # normalized half-res latent
    # distilled.py upscale step (BCFHW ↔ NHWC transposes around the stat ops)
    denorm = enc.denormalize_latent(half.transpose(0, 2, 3, 4, 1)).transpose(0, 4, 1, 2, 3)
    upscaled = up(denorm)
    renorm = enc.normalize_latent(upscaled.transpose(0, 2, 3, 4, 1)).transpose(0, 4, 1, 2, 3)
    mx.eval(renorm)

    mx.save_safetensors(str(OUT / "io.safetensors"), {"half": half, "renorm": renorm})
    print("wrote", OUT)
    print("half", half.shape, "→ renorm", renorm.shape, "std=%.5f" % float(renorm.std()))


if __name__ == "__main__":
    main()
