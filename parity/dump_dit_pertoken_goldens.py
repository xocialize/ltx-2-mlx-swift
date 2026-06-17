#!/usr/bin/env python
"""per-token-timestep DiT parity fixture (the i2v foundation).

Runs the real bf16 distilled transformer with PER-TOKEN timesteps (video_timesteps /
audio_timesteps) instead of a scalar sigma — the path i2v uses to hold conditioned tokens
"clean" (timestep 0) while the rest follow the schedule (timestep sigma). Reuses the dit_full
inputs; builds a mixed mask (first frame's tokens = 0, the rest = sigma) for BOTH modalities to
exercise the full per-token branch (adaln_single + av_ca_video, audio mirror). Dumps the
velocities AND the per-token timestep vectors so the Swift gate feeds identical inputs.

Run in the oracle uv env:
    cd ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx && \
        uv run python ../ltx-2-mlx-swift/parity/dump_dit_pertoken_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig

DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
IO = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/dit_full/io.safetensors")
OUT = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/dit_pertoken")

# dit_full fixture geometry: video Nv=192 (F=3,H=8,W=8 → first frame = 64 tokens), audio Na=16.
VIDEO_FRAME0_TOKENS = 64
AUDIO_FRAME0_TOKENS = 8


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = LTXModelConfig.from_checkpoint_dir(DIR)
    model = LTXModel(cfg)
    import mlx.core as _mx
    w = _mx.load(str(DIR / "transformer-distilled.safetensors"))
    w = {k[len("transformer."):] if k.startswith("transformer.") else k: v for k, v in w.items()}
    model.load_weights(list(w.items()))
    mx.eval(model.parameters())

    io = mx.load(str(IO))
    sigma = io["sigma"]                       # scalar schedule value (shape (1,))
    sig = float(sigma.reshape(-1)[0])
    Nv = io["video_latent"].shape[1]
    Na = io["audio_latent"].shape[1]

    # denoise_mask: 0 = conditioned/clean (timestep 0), 1 = generated (timestep sigma).
    vmask = mx.ones((1, Nv)); vmask[:, :VIDEO_FRAME0_TOKENS] = 0.0
    amask = mx.ones((1, Na)); amask[:, :AUDIO_FRAME0_TOKENS] = 0.0
    video_timesteps = vmask * sig             # (1, Nv)
    audio_timesteps = amask * sig             # (1, Na)

    vv, av = model(
        video_latent=io["video_latent"], audio_latent=io["audio_latent"], timestep=sigma,
        video_text_embeds=io["video_text"], audio_text_embeds=io["audio_text"],
        video_positions=io["video_positions"], audio_positions=io["audio_positions"],
        video_timesteps=video_timesteps, audio_timesteps=audio_timesteps)
    mx.eval(vv, av)

    mx.save_safetensors(str(OUT / "io.safetensors"), {
        "video_timesteps": video_timesteps.astype(mx.float32),
        "audio_timesteps": audio_timesteps.astype(mx.float32),
        "video_v": vv.astype(mx.float32), "audio_v": av.astype(mx.float32)})
    print("wrote", OUT)
    print("video_v", vv.shape, "std=%.5f" % float(vv.std()), " audio_v", av.shape, "std=%.5f" % float(av.std()))


if __name__ == "__main__":
    main()
