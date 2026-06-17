#!/usr/bin/env python
"""Video VAE encode parity fixture: encode small pixels → latent (fp32).

Run in the oracle uv env:
    cd ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx && \
        uv run python ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/dump_vae_encode_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.video_vae.ops import remap_encoder_weight_keys
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.weights import load_split_safetensors

MODEL_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
OUT = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/vae_encode")
F, H, W = 9, 128, 128  # 1+8k frames → latent F=2; 128/32 = 4


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    enc = VideoEncoder()
    w = load_split_safetensors(MODEL_DIR / "vae_encoder.safetensors", prefix="vae_encoder.")
    w = remap_encoder_weight_keys(w)
    w = {k: v.astype(mx.float32) for k, v in w.items()}
    enc.load_weights(list(w.items()))
    mx.eval(enc.parameters())

    mx.random.seed(5)
    pixels = mx.clip(mx.random.normal((1, 3, F, H, W)), -1.0, 1.0).astype(mx.float32)
    latent = enc.encode(pixels)
    mx.eval(latent)

    mx.save_safetensors(str(OUT / "io.safetensors"), {"pixels": pixels, "latent": latent})
    print("wrote", OUT)
    print("pixels", pixels.shape, "→ latent", latent.shape,
          "mean=%.5f std=%.5f" % (float(latent.mean()), float(latent.std())))


if __name__ == "__main__":
    main()
