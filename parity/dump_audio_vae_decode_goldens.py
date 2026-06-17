#!/usr/bin/env python
"""Audio VAE decode parity fixture: audio latent → mel (fp32).

Run in the oracle uv env:
    cd ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx && \
        uv run python ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/dump_audio_vae_decode_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

MODEL_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
OUT = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/audio_vae_decode")
T = 8


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dec = AudioVAEDecoder()
    # Faithful oracle load: decoder convs under "audio_vae.decoder.", per-channel
    # stats under "audio_vae." (outside decoder), then remap _mean_of_means → mean_of_means.
    w = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.decoder.")
    allAudio = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.")
    for k, v in allAudio.items():
        if k.startswith("per_channel_statistics."):
            w[k] = v
    w = remap_audio_vae_keys(w)
    w = {k: v.astype(mx.float32) for k, v in w.items()}
    dec.load_weights(list(w.items()))
    mx.eval(dec.parameters())

    mx.random.seed(9)
    latent = mx.random.normal((1, 8, T, 16)).astype(mx.float32)
    mel = dec.decode(latent)  # (1, 2, T', 64)
    mx.eval(mel)

    mx.save_safetensors(str(OUT / "io.safetensors"), {"latent": latent, "mel": mel})
    print("wrote", OUT)
    print("latent", latent.shape, "→ mel", mel.shape,
          "mean=%.5f std=%.5f" % (float(mel.mean()), float(mel.std())))


if __name__ == "__main__":
    main()
