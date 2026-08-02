#!/usr/bin/env python
"""Independent ground truth for the PrunaVAED decoder: PyTorch + diffusers.

The MLX decoder and its ltx-core weight conversion were both written from the same
reading of Pruna's ``patch_diffusers.py``. Gating one against the other would only prove
they share a misreading -- in particular the new channel-adapter block (``norm3`` +
1x1x1 ``conv_shortcut``) has no stock counterpart to fall back on. So this runs Pruna's
*actual* reference implementation and dumps its output for the MLX side to match.

Runs in the throwaway reference venv, NOT the oracle env::

    cd LTX_DEV/.work/pruna && .venv-ref/bin/python \
        ../../ltx-2-mlx-swift/parity/dump_vae_decode_pruna_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

WORK = Path(__file__).resolve().parents[2] / ".work" / "pruna"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "vae_decode" / "io.safetensors"
OUT = Path(__file__).resolve().parent / "goldens" / "vae_decode_pruna" / "reference.safetensors"


def main() -> None:
    sys.path.insert(0, str(WORK))
    from patch_diffusers import patch_pruna_ltx2_decoder

    from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video

    patch_pruna_ltx2_decoder()

    vae = AutoencoderKLLTX2Video.from_pretrained(WORK / "vae", torch_dtype=torch.float32)
    vae.eval()

    # Reuse the stock vae-decode golden's latent so the Pruna and stock decodes are
    # directly comparable, and so the Swift gate can share one input fixture.
    latent = load_file(str(GOLDEN))["latent"].to(torch.float32)  # (1, 128, F, H, W)

    # Denormalize exactly as the MLX decoder does internally: x * std + mean.
    mean = vae.latents_mean.reshape(1, -1, 1, 1, 1).to(torch.float32)
    std = vae.latents_std.reshape(1, -1, 1, 1, 1).to(torch.float32)
    x = latent * std + mean

    with torch.no_grad():
        pixels = vae.decoder(x, temb=None, causal=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({"latent": latent, "pixels": pixels.contiguous()}, str(OUT))
    print(f"wrote {OUT}")
    print(
        f"latent {tuple(latent.shape)} -> pixels {tuple(pixels.shape)} "
        f"mean={pixels.mean():+.5f} std={pixels.std():.5f} "
        f"min={pixels.min():.3f} max={pixels.max():.3f}"
    )


if __name__ == "__main__":
    main()
