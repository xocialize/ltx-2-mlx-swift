#!/usr/bin/env python
"""Video VAE decode parity fixture: decode a small latent → pixels (fp32).

Run in the oracle uv env:
    cd ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx && \
        uv run python ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/dump_vae_decode_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
from ltx_core_mlx.utils.weights import load_split_safetensors

MODEL_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
OUT = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/vae_decode")
F, H, W = 2, 4, 4  # tiny latent → output (2*8-7)=9 frames, 128x128


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dec = VideoDecoder()  # causal=False (LTX-2.3 default), zeros spatial pad
    w = load_split_safetensors(MODEL_DIR / "vae_decoder.safetensors", prefix="vae_decoder.")
    w = {k: v.astype(mx.float32) for k, v in w.items()}
    dec.load_weights(list(w.items()))
    mx.eval(dec.parameters())

    mx.random.seed(3)
    latent = mx.random.normal((1, 128, F, H, W)).astype(mx.float32)
    pixels = dec.decode(latent)
    mx.eval(pixels)

    mx.save_safetensors(str(OUT / "io.safetensors"), {"latent": latent, "pixels": pixels})
    print("wrote", OUT)
    print("latent", latent.shape, "→ pixels", pixels.shape,
          "mean=%.5f std=%.5f min=%.3f max=%.3f" % (float(pixels.mean()), float(pixels.std()), float(pixels.min()), float(pixels.max())))


if __name__ == "__main__":
    main()
