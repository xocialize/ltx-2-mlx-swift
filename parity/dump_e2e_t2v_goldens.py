#!/usr/bin/env python
"""End-to-end one-stage t2v fixture (real weights): noise → frames.

The capstone gate. Uses the real distilled transformer (bf16) + real video VAE
decoder, with the captured fox-prompt text embeds (skips re-running Gemma at
runtime — already gated separately). Runs the full distilled Euler denoise over
DISTILLED_SIGMAS, unpatchifies, and VAE-decodes to pixels. Dumps the shared
inputs + the final pixels; Swift reproduces and compares. Also saves a PNG frame.

Run in the oracle uv env:
    cd ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx && \
        uv run python ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/dump_e2e_t2v_goldens.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np

from ltx_core_mlx.components.patchifiers import VideoLatentPatchifier
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import load_split_safetensors
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.samplers import denoise_loop

MODEL_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
TE = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/text_encode/goldens.safetensors")
OUT = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/e2e_t2v")

F_LAT, H_LAT, W_LAT, FPS = 2, 8, 8, 24.0  # → 9 frames, 256×256, Nv=128


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = LTXModelConfig.from_checkpoint_dir(MODEL_DIR)
    model = LTXModel(cfg)
    model.load_weights(list(load_split_safetensors(MODEL_DIR / "transformer-distilled.safetensors", prefix="transformer.").items()))
    mx.eval(model.parameters())
    x0m = X0Model(model)

    dec = VideoDecoder()
    wd = {k: v.astype(mx.float32) for k, v in load_split_safetensors(MODEL_DIR / "vae_decoder.safetensors", prefix="vae_decoder.").items()}
    dec.load_weights(list(wd.items()))
    mx.eval(dec.parameters())

    te = mx.load(str(TE))
    video_text, audio_text = te["video_embeds"], te["audio_embeds"]

    num_frames = F_LAT * 8 - 7
    Nv = F_LAT * H_LAT * W_LAT
    audio_T = compute_audio_token_count(num_frames, frame_rate=FPS)

    mx.random.seed(11)
    video_latent = mx.random.normal((1, Nv, 128)).astype(mx.float32)
    audio_latent = mx.random.normal((1, audio_T, 128)).astype(mx.float32)
    video_positions = compute_video_positions(F_LAT, H_LAT, W_LAT, frame_rate=FPS)
    audio_positions = compute_audio_positions(audio_T)

    vstate = LatentState(latent=video_latent, clean_latent=mx.zeros_like(video_latent), denoise_mask=mx.ones((1, Nv, 1)), positions=video_positions)
    astate = LatentState(latent=audio_latent, clean_latent=mx.zeros_like(audio_latent), denoise_mask=mx.ones((1, audio_T, 1)), positions=audio_positions)

    # Diagnostic: LTX_E2E_STEPS limits the schedule (1-step isolates a single DiT
    # forward to separate wiring bugs from bf16 trajectory amplification over 8 steps).
    nsteps = os.environ.get("LTX_E2E_STEPS")
    sigmas = DISTILLED_SIGMAS if nsteps is None else (DISTILLED_SIGMAS[: int(nsteps)] + [0.0])
    out = denoise_loop(x0m, vstate, astate, video_text, audio_text, sigmas=sigmas, show_progress=True)

    patch = VideoLatentPatchifier()
    video_spatial = patch.unpatchify(out.video_latent, (F_LAT, H_LAT, W_LAT))  # (1,128,2,8,8)
    pixels = dec.decode(video_spatial)  # (1,3,9,256,256)
    mx.eval(pixels)

    io = {
        "video_latent": video_latent, "audio_latent": audio_latent,
        "video_text": video_text, "audio_text": audio_text,
        "video_positions": video_positions, "audio_positions": audio_positions,
        "sigmas": mx.array(sigmas).astype(mx.float32),
        "pixels": pixels.astype(mx.float32),
    }
    mx.save_safetensors(str(OUT / "io.safetensors"), io)

    # Save a visible frame (middle).
    frame = np.array(mx.clip(pixels[0, :, num_frames // 2], -1, 1).transpose(1, 2, 0))  # (H,W,3)
    img = ((frame + 1.0) * 127.5).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(img).save(str(OUT / "frame.png"))
        print("saved frame.png")
    except Exception as e:
        print("PNG save skipped:", e)

    print("wrote", OUT)
    print("pixels", pixels.shape, "range [%.3f, %.3f]" % (float(pixels.min()), float(pixels.max())),
          "Nv=%d audio_T=%d steps=%d" % (Nv, audio_T, len(DISTILLED_SIGMAS) - 1))


if __name__ == "__main__":
    main()
