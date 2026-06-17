#!/usr/bin/env python
"""Small-scale DiT parity fixture: a tiny seeded LTXModel forward.

Builds a tiny LTXModel (2 layers, small dims), random-initialised with a fixed
seed, runs ONE forward (video+audio latents + sigma + text embeds + positions →
velocities), and dumps the model weights + the inputs + the outputs so the Swift
port can inject identical weights/inputs and compare its velocity output.

Validates the DiT forward MATH without the 35GB transformer download. Full-scale
parity (real weights) follows once the transformer lands.

Run in the oracle uv env:
    cd ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx && \
        uv run python ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/dump_dit_tiny_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig

OUT_DIR = Path(__file__).resolve().parent / "goldens" / "dit_tiny"

# Tiny config — small enough to debug fast, exercises every code path.
CFG = LTXModelConfig(
    num_layers=2,
    video_dim=64, video_num_heads=2, video_head_dim=32,
    audio_dim=32, audio_num_heads=2, audio_head_dim=16,
    av_cross_num_heads=2, av_cross_head_dim=16,
    video_patch_channels=8, audio_patch_channels=8,
    ff_mult=2.0,
    timestep_embedding_dim=32,
    timestep_scale_multiplier=1000.0,
    av_ca_timestep_scale_multiplier=1000.0,  # checkpoint value (issue #37)
    rope_theta=10000.0, rope_type="split",
    positional_embedding_max_pos=(20, 2048, 2048),
    audio_positional_embedding_max_pos=(20,),
    norm_eps=1e-6,
)

B, Nv, Na, Nt = 1, 12, 6, 8


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mx.random.seed(0)

    model = LTXModel(CFG)
    mx.eval(model.parameters())

    # Fixed seeded inputs.
    mx.random.seed(1)
    # fp32 inputs; LTXModel.__call__ casts per LTX2_DIT_FP32 (bf16 by default).
    video_latent = mx.random.normal((B, Nv, CFG.video_patch_channels)).astype(mx.float32)
    audio_latent = mx.random.normal((B, Na, CFG.audio_patch_channels)).astype(mx.float32)
    sigma = mx.array([0.7]).astype(mx.float32)
    video_text = mx.random.normal((B, Nt, CFG.video_dim)).astype(mx.float32)
    audio_text = mx.random.normal((B, Nt, CFG.audio_dim)).astype(mx.float32)
    # Integer positions (the same values are loaded by Swift — any valid grid works).
    vp = mx.arange(Nv).astype(mx.int32)
    video_positions = mx.stack([vp % 4, (vp // 4) % 2, vp % 3], axis=-1)[None]  # (B,Nv,3)
    audio_positions = mx.arange(Na).astype(mx.int32)[None, :, None]              # (B,Na,1)

    video_v, audio_v = model(
        video_latent=video_latent,
        audio_latent=audio_latent,
        timestep=sigma,
        video_text_embeds=video_text,
        audio_text_embeds=audio_text,
        video_positions=video_positions,
        audio_positions=audio_positions,
    )
    mx.eval(video_v, audio_v)

    # Save weights (flattened param tree → keys match the functional Swift port).
    weights = {k: v for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(OUT_DIR / "weights.safetensors"), {k: v.astype(mx.float32) for k, v in weights.items()})

    io = {
        "video_latent": video_latent, "audio_latent": audio_latent, "sigma": sigma,
        "video_text": video_text, "audio_text": audio_text,
        "video_positions": video_positions.astype(mx.int32),
        "audio_positions": audio_positions.astype(mx.int32),
        "video_v": video_v, "audio_v": audio_v,
    }
    mx.save_safetensors(str(OUT_DIR / "io.safetensors"), {k: v.astype(mx.float32) for k, v in io.items()})

    print("wrote", OUT_DIR)
    print("num weight tensors:", len(weights))
    print("video_v", video_v.shape, "mean=%.5f std=%.5f" % (float(video_v.mean()), float(video_v.std())))
    print("audio_v", audio_v.shape, "mean=%.5f std=%.5f" % (float(audio_v.mean()), float(audio_v.std())))
    # Print a few weight keys so the Swift port can match them.
    for k in sorted(weights)[:6]:
        print("  key:", k, list(weights[k].shape))


if __name__ == "__main__":
    main()
