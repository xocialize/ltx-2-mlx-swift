#!/usr/bin/env python
"""IC reference-ingest parity fixture (IC-LORA-PLAN P2): looped-still → VAE encode → tokens+positions.

Mirrors the oracle's `iclora_utils.append_ic_lora_reference_video_conditionings` encode glue
(video → encoder.encode → transpose(0,2,3,4,1).reshape(1,-1,128) + compute_video_positions) with
the REAL vae_encoder weights (fp32, the bit-exact-gated component) on a seeded still tiled to
8k+1 frames — gating the GLUE (tiling, ordering, patchify, positions), not the encoder itself.

Run in the oracle uv env:
    cd ~/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx && \
        uv run python ~/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/dump_ic_ingest_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.video_vae.ops import remap_encoder_weight_keys
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.positions import compute_video_positions
from ltx_core_mlx.utils.weights import load_split_safetensors

MODEL_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
OUT = Path("/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/ic_ingest")
F, H, W = 17, 96, 96   # 8k+1 frames (k=2); 96/32 = 3 → latent (3, 3, 3), Nr = 27


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    enc = VideoEncoder()
    w = load_split_safetensors(MODEL_DIR / "vae_encoder.safetensors", prefix="vae_encoder.")
    w = remap_encoder_weight_keys(w)
    w = {k: v.astype(mx.float32) for k, v in w.items()}
    enc.load_weights(list(w.items()))
    mx.eval(enc.parameters())

    mx.random.seed(11)
    still = mx.clip(mx.random.normal((1, 3, 1, H, W)), -1.0, 1.0).astype(mx.float32)
    video = mx.broadcast_to(still, (1, 3, F, H, W))          # looped-still tiling

    latent = enc.encode(video)                                # (1,128,fLat,hLat,wLat) normalized
    ref_tokens = latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)   # iclora_utils line 144
    ref_positions = compute_video_positions(latent.shape[2], latent.shape[3], latent.shape[4])
    mx.eval(ref_tokens, ref_positions)

    mx.save_safetensors(str(OUT / "io.safetensors"), {
        "still": still.astype(mx.float32),
        "ref_tokens": ref_tokens.astype(mx.float32),
        "ref_positions": ref_positions.astype(mx.float32),
        "dims": mx.array([float(F), float(H), float(W)]),
    })
    print("wrote", OUT / "io.safetensors")
    print("tokens", ref_tokens.shape, "mean=%.5f std=%.5f" % (float(ref_tokens.mean()), float(ref_tokens.std())))


if __name__ == "__main__":
    main()
