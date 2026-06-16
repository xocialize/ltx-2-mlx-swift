#!/usr/bin/env python
"""Distilled denoise-loop parity fixture (tiny scale, fp32).

Reuses the SAME tiny seeded LTXModel as dump_dit_tiny_goldens (seed 0 → identical
weights, so Swift loads dit_tiny/weights.safetensors), runs the distilled Euler
denoise_loop over a short sigma schedule with a uniform mask (t2v), and dumps the
inputs + the final denoised latents. Validates X0Model + euler_step + sigma
pairing, independent of weights (the DiT is already gated).

Run in the oracle uv env (LTX2_DIT_FP32 forces fp32 to match the fp32 Swift DiT):
    cd ~/Development/ltx-2-mlx && LTX2_DIT_FP32=1 \
        uv run python ~/Development/ltx-2-mlx-swift/parity/dump_denoise_goldens.py
"""

from __future__ import annotations

import os

os.environ.setdefault("LTX2_DIT_FP32", "1")

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from ltx_pipelines_mlx.utils.samplers import denoise_loop

OUT = Path("/Users/dustinnielson/Development/ltx-2-mlx-swift/parity/goldens/dit_denoise")

CFG = LTXModelConfig(
    num_layers=2,
    video_dim=64, video_num_heads=2, video_head_dim=32,
    audio_dim=32, audio_num_heads=2, audio_head_dim=16,
    av_cross_num_heads=2, av_cross_head_dim=16,
    video_patch_channels=8, audio_patch_channels=8,
    ff_mult=2.0, timestep_embedding_dim=32,
    timestep_scale_multiplier=1000.0, av_ca_timestep_scale_multiplier=1000.0,
    rope_theta=10000.0, rope_type="split",
    positional_embedding_max_pos=(20, 2048, 2048),
    audio_positional_embedding_max_pos=(20,), norm_eps=1e-6,
)
B, Nv, Na, Nt = 1, 12, 6, 8
SIGMAS = [1.0, 0.5, 0.0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mx.random.seed(0)
    model = LTXModel(CFG)
    mx.eval(model.parameters())
    x0m = X0Model(model)

    mx.random.seed(2)
    video_latent = mx.random.normal((B, Nv, CFG.video_patch_channels)).astype(mx.float32)
    audio_latent = mx.random.normal((B, Na, CFG.audio_patch_channels)).astype(mx.float32)
    video_text = mx.random.normal((B, Nt, CFG.video_dim)).astype(mx.float32)
    audio_text = mx.random.normal((B, Nt, CFG.audio_dim)).astype(mx.float32)
    vp = mx.arange(Nv).astype(mx.int32)
    video_positions = mx.stack([vp % 4, (vp // 4) % 2, vp % 3], axis=-1)[None]
    audio_positions = mx.arange(Na).astype(mx.int32)[None, :, None]

    video_state = LatentState(
        latent=video_latent, clean_latent=mx.zeros_like(video_latent),
        denoise_mask=mx.ones((B, Nv, 1)), positions=video_positions)
    audio_state = LatentState(
        latent=audio_latent, clean_latent=mx.zeros_like(audio_latent),
        denoise_mask=mx.ones((B, Na, 1)), positions=audio_positions)

    out = denoise_loop(x0m, video_state, audio_state, video_text, audio_text,
                       sigmas=SIGMAS, show_progress=False)
    mx.eval(out.video_latent, out.audio_latent)

    io = {
        "video_latent": video_latent, "audio_latent": audio_latent,
        "video_text": video_text, "audio_text": audio_text,
        "video_positions": video_positions.astype(mx.float32),
        "audio_positions": audio_positions.astype(mx.float32),
        "sigmas": mx.array(SIGMAS).astype(mx.float32),
        "video_final": out.video_latent.astype(mx.float32),
        "audio_final": out.audio_latent.astype(mx.float32),
    }
    mx.save_safetensors(str(OUT / "io.safetensors"), io)
    print("wrote", OUT, "sigmas", SIGMAS)
    print("video_final", out.video_latent.shape, "mean=%.5f std=%.5f" % (float(out.video_latent.mean()), float(out.video_latent.std())))


if __name__ == "__main__":
    main()
