#!/usr/bin/env python
"""Spatial-x2 latent upsampler parity fixture: latent → 2×-spatial latent (fp32).

Run in the oracle uv env:
    cd ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx && \
        uv run python ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/dump_upsampler_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.utils.weights import load_split_safetensors

MODEL_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
OUT = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/upsampler")
F, H, W = 2, 4, 4


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # spatial_x2, mid_channels=1024 (matches the v1.1 checkpoint).
    up = LatentUpsampler(in_channels=128, mid_channels=1024, num_blocks_per_stage=4,
                         spatial_upsample=True, temporal_upsample=False,
                         spatial_scale=2.0, rational_resampler=False)
    raw = load_split_safetensors(MODEL_DIR / "spatial_upscaler_x2_v1_1.safetensors")
    stem = "spatial_upscaler_x2_v1_1."
    if raw and all(k.startswith(stem) for k in raw):
        raw = {k[len(stem):]: v for k, v in raw.items()}
    raw = {k: v.astype(mx.float32) for k, v in raw.items()}
    up.load_weights(list(raw.items()))
    mx.eval(up.parameters())

    mx.random.seed(31)
    latent = mx.random.normal((1, 128, F, H, W)).astype(mx.float32)
    out = up(latent)  # (1, 128, F, 2H, 2W)
    mx.eval(out)

    mx.save_safetensors(str(OUT / "io.safetensors"), {"latent": latent, "out": out})
    print("wrote", OUT)
    print("latent", latent.shape, "→ out", out.shape, "std=%.5f" % float(out.std()))


if __name__ == "__main__":
    main()
