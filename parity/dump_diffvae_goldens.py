#!/usr/bin/env python
"""LTX-2.5 DiffVAE parity fixture (arm C / claim C3), dumped from the Python-MLX oracle.

Six stages, in dependency order, so a failure localizes instead of just saying "the decode
is wrong":

  [A] one deterministic NABlock          (det_stages.3.0, dim 512, kernel (3,5,5))
  [B] one LinearPixelShuffleUpsample     (upsamples.3, stride (2,2,2), leading-frame drop)
  [C] one stage-5 DiffusionNABlock       (diff_blocks.0, dim 256, kernel 11^3)
  [D] the full stage-5 stack             (8 blocks + norm_out + conv_out + unpatchify)
  [E] the stage-1-to-4 ladder            (ghost pad -> stages 1-4 -> ghost crop)
  [F] the FULL end-to-end decode         (1-step x0 at t=1.0 from a FIXED noise canvas)

⚠️ THE NOISE IS AN INPUT, NOT A SEED. DiffVAE is a stochastic 1-step-x0 decoder that starts
from pure noise, and MLX-Swift and MLX-Python RNG streams are not bit-identical
(determinism doctrine, ISSUES I1) — so a seeded gate would be measuring the RNG, not the
port. The canvas is dumped into the fixture and injected on the Swift side, exactly as
`--ancestral-step-gate` does for the sampler. A gate that let each side draw its own noise
and compared pixels would be meaningless, and LTX25-PORT-PLAN §V C3 says so explicitly:
never gate this decoder on pixel equality.

The [A]-[D] inputs are reused verbatim from the oracle's torch-reference bank when it is
present, so all three implementations (torch / Python-MLX / Swift) are fed identical
tensors and the torch outputs can ride along for reporting.

    cd .../ltx-2-mlx && uv run --no-sync python .../ltx-2-mlx-swift/parity/dump_diffvae_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "goldens" / "diffvae"
MODEL_DIR = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
REF_BANK = Path("/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx/.work/ltx25-ref/goldens")

# Matches the oracle's own gate fixture: the minimum legal latent (3,7,7) -> 17x224x224
# pixels. Small enough for a CLI gate, large enough that the ghost pad, the ghost crop and
# the stage-5 kernel clamp are all actually exercised.
LATENT_T, LATENT_H, LATENT_W = 3, 7, 7
TILE = (1, 8)          # gather-path query tiling for the component stages
DECODE_TILE = (1, 8)   # ditto for the full decode


def _load(name: str, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Reuse the torch bank's tensor when present, else draw a fresh deterministic one."""
    p = REF_BANK / f"{name}.npy"
    if p.exists():
        a = np.load(p)
        if tuple(a.shape) == shape:
            return a.astype(np.float32)
        print(f"  ! {name} bank shape {a.shape} != {shape}; drawing fresh")
    return rng.standard_normal(shape).astype(np.float32)


def main() -> None:
    from ltx_core_mlx.model.video_vae.diffusion_video_decoder import (
        DiffusionNABlock,
        DiffusionVideoDecoder,
        LinearPixelShuffleUpsample,
        NABlock,
        patchify_hw,
        timestep_embedding,
        unpatchify_hw,
    )
    from ltx_core_mlx.utils.weights import load_split_safetensors

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(21)

    w = {k: v.astype(mx.float32) for k, v in load_split_safetensors(
        MODEL_DIR / "vae_diffusion_decoder.safetensors", prefix="vae_diffusion_decoder.").items()}
    print(f"  loaded {len(w)} weight tensors")

    def sub(prefix: str) -> list:
        return [(k[len(prefix):], v) for k, v in w.items() if k.startswith(prefix)]

    out: dict[str, mx.array] = {}

    # ---- [A] one deterministic NABlock -------------------------------------------------
    xa_np = _load("dv_nablock_in", (1, 4, 8, 8, 512), rng)
    xa = mx.array(xa_np)
    blk = NABlock(512, (3, 5, 5), 64)
    blk.load_weights(sub("det_stages.3.0."), strict=True)
    ya = blk(xa)
    mx.eval(ya)
    out["nablock_in"], out["nablock_out"] = xa, ya

    # ---- [B] one upsample (stride (2,2,2), 512 -> 256, leading-frame drop) --------------
    up = LinearPixelShuffleUpsample(512, (2, 2, 2), 2)
    up.load_weights(sub("upsamples.3."), strict=True)
    yb = up(xa, True)
    mx.eval(yb)
    out["upsample_out"] = yb

    # ---- [C]/[D] stage-5 ---------------------------------------------------------------
    # T,H,W must each be >= the stage-5 kernel (11,11,11): upstream guards on it.
    T5 = H5 = W5 = 12
    ctx = mx.array(_load("dv_diffblock_ctx", (1, T5, H5, W5, 256), rng))
    xt = mx.array(_load("dv_diffblock_xt", (1, 3, T5, H5 * 4, W5 * 4), rng))

    conv_in_x_t = nn.Linear(48, 256)
    conv_in_x_t.load_weights(sub("conv_in_x_t."), strict=True)
    x0 = conv_in_x_t(patchify_hw(xt, 4).transpose(0, 2, 3, 4, 1))

    temb = nn.Sequential(nn.Linear(256, 384), nn.SiLU(), nn.Linear(384, 384))
    temb.load_weights([
        ("layers.0.weight", w["t_embedder.timestep_embedder.linear1.weight"]),
        ("layers.0.bias", w["t_embedder.timestep_embedder.linear1.bias"]),
        ("layers.2.weight", w["t_embedder.timestep_embedder.linear2.weight"]),
        ("layers.2.bias", w["t_embedder.timestep_embedder.linear2.bias"]),
    ], strict=True)
    adaln = nn.Linear(384, 7 * 256)
    adaln.load_weights([("weight", w["shared_adaln.proj.weight"]),
                        ("bias", w["shared_adaln.proj.bias"])], strict=True)
    ch = adaln(nn.silu(temb(timestep_embedding(mx.array([1.0]) * 1000.0))))
    shared = [ch[:, i * 256:(i + 1) * 256].reshape(-1, 1, 1, 1, 256) for i in range(7)]

    one = DiffusionNABlock(256, (11, 11, 11), 256, 64)
    one.load_weights(sub("diff_blocks.0."), strict=True)
    yc = one(x0, ctx, shared, tile=TILE)
    mx.eval(yc)
    out["diffblock_ctx"], out["diffblock_xt"] = ctx, xt
    out["diffblock_out"] = yc

    x = x0
    for i in range(8):
        b = DiffusionNABlock(256, (11, 11, 11), 256, 64)
        b.load_weights(sub(f"diff_blocks.{i}."), strict=True)
        x = b(x, ctx, shared, tile=TILE)
        mx.eval(x)
    norm_out = nn.RMSNorm(256, eps=1e-6)
    norm_out.load_weights(sub("norm_out."), strict=True)
    conv_out = nn.Linear(256, 48)
    conv_out.load_weights(sub("conv_out."), strict=True)
    yd = unpatchify_hw(conv_out(norm_out(x)).transpose(0, 4, 1, 2, 3), 4)
    mx.eval(yd)
    out["stage5_out"] = yd

    # ---- [E]/[F] the whole decoder ------------------------------------------------------
    dec = DiffusionVideoDecoder()
    dec.load_weights([(k, v) for k, v in w.items()], strict=True)
    mx.eval(dec.parameters())

    latent = mx.array(_load("dv_decode_latent", (1, 128, LATENT_T, LATENT_H, LATENT_W), rng))

    padded = dec._resize_axis(dec.ensure_min_latent_shape(latent), 2,
                              latent.shape[2] + dec.natten_trailing_pad, "repeat_last")
    context = dec.context_volume(padded, tile=TILE)
    ghost = dec.natten_trailing_pad * dec.temporal_scale
    keep = min(context.shape[1], max(max(context.shape[1] - ghost, 1), dec.stage5_kernel[0]))
    context = dec._resize_axis(context, 1, keep, "repeat_last")
    mx.eval(context)
    out["decode_latent"] = latent
    out["decode_context"] = context

    B, F, H4, W4, _ = context.shape
    pix_shape = (B, dec.out_channels, F, H4 * dec.patch_size, W4 * dec.patch_size)
    noise = mx.array(_load("dv_decode_noise", pix_shape, rng))
    pixels = dec.decode(latent, noise=noise, tile=DECODE_TILE)
    mx.eval(pixels)
    out["decode_noise"] = noise
    out["decode_pixels"] = pixels
    print(f"  latent {latent.shape} -> context {context.shape} -> pixels {pixels.shape}")

    # A SECOND decode of the same latent from a DIFFERENT canvas. The Swift gate uses this
    # to prove the decoder is genuinely noise-driven: if a port ignored the canvas (or the
    # gate's injection silently failed), these two would be identical and the fixed-noise
    # comparison would be passing for the wrong reason.
    noise2 = mx.array(rng.standard_normal(pix_shape).astype(np.float32))
    pixels2 = dec.decode(latent, noise=noise2, tile=DECODE_TILE)
    mx.eval(pixels2)
    out["decode_noise_alt"] = noise2
    out["decode_pixels_alt"] = pixels2
    spread = float(mx.abs(pixels2 - pixels).max())
    print(f"  canvas sensitivity: maxAbs(pixels_alt - pixels) = {spread:.4f} (must be >> 0)")

    # torch reference tensors ride along for reporting where the bank has them.
    for name, key in (("nablock_out", "dv_nablock_out"),
                      ("upsample_out", "dv_upsample_out"),
                      ("diffblock_out", "dv_diffblock_out"),
                      ("stage5_out", "dv_stage5_out"),
                      ("decode_context", "dv_decode_context"),
                      ("decode_pixels", "dv_decode_pixels")):
        p = REF_BANK / f"{key}.npy"
        if p.exists():
            ref = np.load(p)
            if tuple(ref.shape) == tuple(out[name].shape):
                out[f"torch_{name}"] = mx.array(ref.astype(np.float32))

    mx.save_safetensors(str(OUT_DIR / "io.safetensors"),
                        {k: v.astype(mx.float32) for k, v in out.items()})
    print("wrote", OUT_DIR / "io.safetensors")
    print("  keys:", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
