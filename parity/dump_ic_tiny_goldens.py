#!/usr/bin/env python
"""IC-LoRA reference-conditioning parity fixture (tiny scale, fp32) — IC-LORA-PLAN P1.

Reuses the SAME tiny seeded LTXModel as dump_dit_tiny_goldens (seed 0 → Swift loads
dit_tiny/weights.safetensors) and drives the ORACLE's real IC path:
LatentState → VideoConditionByReferenceLatent.apply (append ref tokens: clean latent,
denoise_mask = 1−strength, positions spatial-scaled by downscale_factor) → denoise_loop
(per-token timesteps from the non-uniform mask) → slice off ref tokens.

Two cases:
  A: strength 1.0, downscale 2  (position scaling; attention mask stays None)
  B: strength 0.7, downscale 1  (partial denoise mask → per-token σ at refs)

Run in the oracle uv env:
    cd ~/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx && LTX2_DIT_FP32=1 \
        uv run python ~/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/dump_ic_tiny_goldens.py
"""

from __future__ import annotations

import os

os.environ.setdefault("LTX2_DIT_FP32", "1")

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from ltx_core_mlx.utils.positions import compute_video_positions
from ltx_pipelines_mlx.utils.samplers import denoise_loop

OUT = Path(
    "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/ic_tiny"
)

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
B, Na, Nt = 1, 6, 8
SIGMAS = [1.0, 0.5, 0.0]

# Target latent grid (F, H, W) and per-case reference grids.
TGT = (2, 4, 4)          # Nv = 32
CASES = {
    "a": {"ref": (2, 2, 2), "downscale": 2, "strength": 1.0},   # Nr = 8
    "b": {"ref": (2, 4, 4), "downscale": 1, "strength": 0.7},   # Nr = 32
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mx.random.seed(0)
    model = LTXModel(CFG)
    mx.eval(model.parameters())
    x0m = X0Model(model)

    F, H, W = TGT
    Nv = F * H * W
    io: dict[str, mx.array] = {}

    for name, case in CASES.items():
        rF, rH, rW = case["ref"]
        Nr = rF * rH * rW

        mx.random.seed(3 if name == "a" else 4)
        video_latent = mx.random.normal((B, Nv, CFG.video_patch_channels)).astype(mx.float32)
        audio_latent = mx.random.normal((B, Na, CFG.audio_patch_channels)).astype(mx.float32)
        video_text = mx.random.normal((B, Nt, CFG.video_dim)).astype(mx.float32)
        audio_text = mx.random.normal((B, Nt, CFG.audio_dim)).astype(mx.float32)
        ref_tokens = mx.random.normal((B, Nr, CFG.video_patch_channels)).astype(mx.float32)

        video_positions = compute_video_positions(F, H, W)
        ref_positions = compute_video_positions(rF, rH, rW)

        video_state = LatentState(
            latent=video_latent, clean_latent=mx.zeros_like(video_latent),
            denoise_mask=mx.ones((B, Nv, 1)), positions=video_positions)
        cond = VideoConditionByReferenceLatent(
            reference_latent=ref_tokens, reference_positions=ref_positions,
            downscale_factor=case["downscale"], strength=case["strength"])
        video_state = cond.apply(video_state, spatial_dims=(F, H, W))
        assert video_state.attention_mask is None, "basic IC path must not build an attention mask"

        audio_positions = mx.arange(Na).astype(mx.int32)[None, :, None]
        audio_state = LatentState(
            latent=audio_latent, clean_latent=mx.zeros_like(audio_latent),
            denoise_mask=mx.ones((B, Na, 1)), positions=audio_positions)

        out = denoise_loop(x0m, video_state, audio_state, video_text, audio_text,
                           sigmas=SIGMAS, show_progress=False)
        mx.eval(out.video_latent, out.audio_latent)

        io[f"{name}_video_latent"] = video_latent
        io[f"{name}_audio_latent"] = audio_latent
        io[f"{name}_video_text"] = video_text
        io[f"{name}_audio_text"] = audio_text
        io[f"{name}_ref_tokens"] = ref_tokens
        io[f"{name}_video_positions"] = video_positions.astype(mx.float32)
        io[f"{name}_ref_positions"] = ref_positions.astype(mx.float32)
        io[f"{name}_ext_positions"] = video_state.positions.astype(mx.float32)
        io[f"{name}_audio_positions"] = audio_positions.astype(mx.float32)
        io[f"{name}_params"] = mx.array(
            [float(case["downscale"]), float(case["strength"]), float(Nv), float(Nr)])
        io[f"{name}_video_final_full"] = out.video_latent.astype(mx.float32)   # (B, Nv+Nr, C)
        io[f"{name}_video_final"] = out.video_latent[:, :Nv, :].astype(mx.float32)
        io[f"{name}_audio_final"] = out.audio_latent.astype(mx.float32)
        print(f"case {name}: Nv={Nv} Nr={Nr} downscale={case['downscale']} strength={case['strength']}",
              "video_final mean=%.5f std=%.5f" % (float(io[f'{name}_video_final'].mean()),
                                                  float(io[f'{name}_video_final'].std())))

    io["sigmas"] = mx.array(SIGMAS).astype(mx.float32)
    mx.save_safetensors(str(OUT / "io.safetensors"), io)
    print("wrote", OUT / "io.safetensors")


if __name__ == "__main__":
    main()
