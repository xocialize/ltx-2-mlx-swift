#!/usr/bin/env python
"""PrunaVAED video-decode parity fixture: decode a small latent → pixels (fp32).

Mirrors ``dump_vae_decode_goldens.py`` but for the pruned decoder, which adds two
channel-adapter blocks the stock decoder does not have. Reuses the stock fixture's latent
so the two decodes are directly comparable.

The MLX output this writes is itself cross-checked against PyTorch+diffusers by
``dump_vae_decode_pruna_reference.py`` (cosine 0.9999959) — that reference is the
independent ground truth; this file is the Swift gate's input/output pair.

Run in the oracle uv env::

    cd LTX_DEV/ltx-2-mlx && uv run python ../ltx-2-mlx-swift/parity/dump_vae_decode_pruna_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.video_vae.video_vae import DecoderLadder, VideoDecoder
from ltx_core_mlx.utils.weights import load_split_safetensors

MODEL_DIR = Path("/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx")
PARITY = Path(__file__).resolve().parent
STOCK_GOLDEN = PARITY / "goldens" / "vae_decode" / "io.safetensors"
OUT = PARITY / "goldens" / "vae_decode_pruna"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    w = load_split_safetensors(MODEL_DIR / "vae_decoder_pruna.safetensors", prefix="vae_decoder.")
    w = {k: v.astype(mx.float32) for k, v in w.items()}

    ladder = DecoderLadder.from_weights(w)
    dec = VideoDecoder(ladder=ladder)  # causal=False (LTX-2.3 default), zeros spatial pad
    dec.load_weights(list(w.items()))
    mx.eval(dec.parameters())

    latent = mx.load(str(STOCK_GOLDEN))["latent"].astype(mx.float32)
    pixels = dec.decode(latent)
    mx.eval(pixels)

    mx.save_safetensors(str(OUT / "io.safetensors"), {"latent": latent, "pixels": pixels})
    print("wrote", OUT)
    print("ladder res stages:", [c for c, _ in ladder.res_stages])
    print("adapters at up_blocks:", [2 * i + 1 for i, (_, _, a) in enumerate(ladder.upsamples) if a is not None])
    print(
        "latent", latent.shape, "→ pixels", pixels.shape,
        "mean=%.5f std=%.5f min=%.3f max=%.3f"
        % (float(pixels.mean()), float(pixels.std()), float(pixels.min()), float(pixels.max())),
    )


if __name__ == "__main__":
    main()
