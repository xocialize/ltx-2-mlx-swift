#!/usr/bin/env python
"""Pixel-masked IC conditioning parity fixture (tiny scale, fp32) — the outpaint/inpaint path.

Covers the three pieces `Sources/LTX2/ICAttentionMask.swift` ports from upstream Lightricks/LTX-2
(ltx-core / ltx-pipelines @ 9ec55f9), then drives the ORACLE's real IC denoise with the result:

  1. downsample_mask_video_to_latent  — area (block mean at the integral 32-px ratio every pipeline
     canvas has) + causal temporal (frame 0 alone, then means over t = (F_pix-1)/(F_lat-1)).
  2. build_attention_mask             — target<->target 1, target<->ref = cross (both directions),
     ref<->ref 1.
  3. multiplicative -> additive        — BOTH conventions: ltx-core `log(m)` with m<=0 -> finfo.min,
     and diffusers `(1-m) * -10000`.

⚠️ The oracle is used ONLY as the DiT + denoise loop here. Its own pipeline hands the
MULTIPLICATIVE [0,1] mask straight to `mx.fast.scaled_dot_product_attention` as an ADDITIVE bias
(samplers.py -> model.py -> attention.py), which is a +1 nudge, not a mask — so this dumper converts
upstream's way first and passes the oracle an additive bias, which its attention consumes correctly.

The pixel mask is built to put FRACTIONAL weights on the boundary cells (0.75, 0.125, ...), the
only tokens where the two conventions disagree, and to differ between frame 0 and frames 1..8 so
the causal temporal split is exercised.

Run in the oracle uv env:
    cd /Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx && LTX2_DIT_FP32=1 \
        uv run python ../ltx-2-mlx-swift/parity/dump_ic_mask_goldens.py
"""

from __future__ import annotations

import os

os.environ.setdefault("LTX2_DIT_FP32", "1")

from pathlib import Path

import mlx.core as mx
import numpy as np

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from ltx_core_mlx.utils.positions import compute_video_positions
from ltx_pipelines_mlx.utils.samplers import denoise_loop

HERE = Path(__file__).resolve().parent
OUT = HERE / "goldens" / "ic_mask"
TINY_WEIGHTS = HERE / "goldens" / "dit_tiny" / "weights.safetensors"

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
F_LAT, H_LAT, W_LAT = 2, 4, 4          # target AND reference grid (downscale 1) -> Nv = Nr = 32
F_PIX, H_PIX, W_PIX = 9, 128, 128      # 8k+1 frames, 32 px per latent cell


def pixel_mask() -> np.ndarray:
    m = np.ones((F_PIX, H_PIX, W_PIX), dtype=np.float32)
    m[:, :, :40] = 0.0            # left margin: cell 0 -> 0, cell 1 -> 24/32 = 0.75
    m[1:, :, 100:] = 0.0          # right margin on frames 1..8 only: cell 3 -> 4/32 = 0.125
    m[5:, :16, :] = 0.0           # top band on frames 5..8 only -> temporal mean over 1..8 = 0.75x
    return m


def downsample_to_latent(mask: np.ndarray) -> np.ndarray:
    """upstream iclora_utils.downsample_mask_video_to_latent at an integral ratio (area == block mean)."""
    f, h, w = mask.shape
    bh, bw = h // H_LAT, w // W_LAT
    spatial = mask.reshape(f, H_LAT, bh, W_LAT, bw).mean(axis=(2, 4))
    first = spatial[:1]
    t = (f - 1) // (F_LAT - 1)
    rest = spatial[1:].reshape(F_LAT - 1, t, H_LAT, W_LAT).mean(axis=1)
    return np.concatenate([first, rest], axis=0).reshape(-1)


def build_mult(nv: int, cross: np.ndarray) -> np.ndarray:
    """upstream mask_utils.build_attention_mask, one reference group, no existing mask."""
    nr = cross.shape[0]
    n = nv + nr
    m = np.zeros((n, n), dtype=np.float32)
    m[:nv, :nv] = 1.0
    m[nv:, nv:] = 1.0
    m[:nv, nv:] = cross[None, :]
    m[nv:, :nv] = cross[:, None]
    return m


def additive(mult: np.ndarray, convention: str) -> np.ndarray:
    if convention == "ltxcore":   # transformer_args._prepare_self_attention_mask (fp32 here)
        fin = np.finfo(np.float32)
        bias = np.full_like(mult, fin.min)
        pos = mult > 0
        bias[pos] = np.log(np.maximum(mult[pos], fin.tiny))
        return bias
    if convention == "diffusers":  # transformer_ltx2.py
        return (1.0 - mult) * -10000.0
    raise ValueError(convention)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mx.random.seed(0)
    model = LTXModel(CFG)
    model.load_weights(list(mx.load(str(TINY_WEIGHTS)).items()), strict=True)
    mx.eval(model.parameters())
    x0m = X0Model(model)

    nv = F_LAT * H_LAT * W_LAT
    nr = nv
    mx.random.seed(7)
    video_latent = mx.random.normal((B, nv, CFG.video_patch_channels)).astype(mx.float32)
    audio_latent = mx.random.normal((B, Na, CFG.audio_patch_channels)).astype(mx.float32)
    video_text = mx.random.normal((B, Nt, CFG.video_dim)).astype(mx.float32)
    audio_text = mx.random.normal((B, Nt, CFG.audio_dim)).astype(mx.float32)
    ref_tokens = mx.random.normal((B, nr, CFG.video_patch_channels)).astype(mx.float32)
    video_positions = compute_video_positions(F_LAT, H_LAT, W_LAT)
    audio_positions = mx.arange(Na).astype(mx.int32)[None, :, None]

    pm = pixel_mask()
    cross = downsample_to_latent(pm)
    mult = build_mult(nv, cross)
    print("cross weights (f,h,w):", np.round(cross.reshape(F_LAT, H_LAT, W_LAT), 4).tolist())

    io: dict[str, mx.array] = {
        "pixel_mask": mx.array(pm),
        "cross": mx.array(cross),
        "mult": mx.array(mult),
        "video_latent": video_latent, "audio_latent": audio_latent,
        "video_text": video_text, "audio_text": audio_text,
        "ref_tokens": ref_tokens,
        "video_positions": video_positions.astype(mx.float32),
        "ref_positions": video_positions.astype(mx.float32),
        "audio_positions": audio_positions.astype(mx.float32),
        "sigmas": mx.array(SIGMAS, dtype=mx.float32),
    }

    for conv in ("ltxcore", "diffusers"):
        bias = additive(mult, conv)
        io[f"bias_{conv}"] = mx.array(bias)
        video_state = LatentState(
            latent=video_latent, clean_latent=mx.zeros_like(video_latent),
            denoise_mask=mx.ones((B, nv, 1)), positions=video_positions)
        cond = VideoConditionByReferenceLatent(
            reference_latent=ref_tokens, reference_positions=video_positions,
            downscale_factor=1, strength=1.0)
        video_state = cond.apply(video_state, spatial_dims=(F_LAT, H_LAT, W_LAT))
        audio_state = LatentState(
            latent=audio_latent, clean_latent=mx.zeros_like(audio_latent),
            denoise_mask=mx.ones((B, Na, 1)), positions=audio_positions)
        out = denoise_loop(x0m, video_state, audio_state, video_text, audio_text,
                           sigmas=SIGMAS, video_attention_mask=mx.array(bias)[None, None],
                           show_progress=False)
        mx.eval(out.video_latent, out.audio_latent)
        io[f"{conv}_video_final_full"] = out.video_latent.astype(mx.float32)
        io[f"{conv}_video_final"] = out.video_latent[:, :nv, :].astype(mx.float32)
        io[f"{conv}_audio_final"] = out.audio_latent.astype(mx.float32)
        print(f"{conv}: video |x| mean {float(mx.abs(out.video_latent[:, :nv]).mean()):.5f}")

    d = io["ltxcore_video_final"] - io["diffusers_video_final"]
    print(f"conventions differ on target tokens: maxAbs {float(mx.abs(d).max()):.5f} (must be > 0)")
    mx.save_safetensors(str(OUT / "io.safetensors"), io)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
