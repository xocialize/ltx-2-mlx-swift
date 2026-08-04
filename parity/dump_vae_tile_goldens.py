#!/usr/bin/env python
"""Video VAE spatial-tiling parity fixture (BLOCKSTREAM-EXPANSION-EVAL §3).

Decodes one latent twice: whole-frame (the exact reference) and through the
upstream tiler with a FORCED SpatialTilingConfig — upstream's auto-tiler only
ever emits temporal configs (video_vae.py `_compute_decode_tiling`), so forcing
the spatial config here is the only way to exercise `tiling.py`'s spatial path.
The Swift `--vae-tile-gate` compares its whole decode against `pixels_whole`
(cross-binding parity ≥0.9999) and reports the oracle's own blend-tiled output
(`pixels_tiled`) as context — Swift ships halo+crop, not upstream's blend, so
the two tiled outputs are intentionally different algorithms.

Run in the oracle uv env (page the weights in first):
    cd <LTX_DEV>/ltx-2-mlx && uv run python ../ltx-2-mlx-swift/parity/dump_vae_tile_goldens.py
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
from ltx_core_mlx.utils.weights import load_split_safetensors

MODEL_DIR = Path("/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx")
OUT = Path(__file__).resolve().parent / "goldens" / "vae_tile"
F, H, W = 3, 24, 40  # latent → 17 frames, 768×1280 px — big enough for real spatial tiles


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dec = VideoDecoder()  # causal=False (LTX-2.3 default), zeros spatial pad
    w = load_split_safetensors(MODEL_DIR / "vae_decoder.safetensors", prefix="vae_decoder.")
    w = {k: v.astype(mx.float32) for k, v in w.items()}
    dec.load_weights(list(w.items()))
    mx.eval(dec.parameters())

    mx.random.seed(11)
    latent = mx.random.normal((1, 128, F, H, W)).astype(mx.float32)
    mx.eval(latent)

    whole = dec.decode(latent)
    mx.eval(whole)

    # Forced spatial config: 512 px tiles / 64 px (2-latent-cell) overlap → 3×3 tiles at 768×1280.
    # A whole-clip temporal config rides along because upstream's tiled_decode assumes the temporal
    # mapper exists (spatial-only crashes on the default slice(0, None) — the spatial path really is
    # unexercised upstream); at 24 ≥ clip frames it degenerates to a single full-span interval.
    cfg = TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(tile_size_in_frames=24, tile_overlap_in_frames=8),
    )
    chunks = list(dec.tiled_decode(latent, cfg))
    tiled = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=2)
    mx.eval(tiled)
    assert tiled.shape == whole.shape, (tiled.shape, whole.shape)

    # Upstream's own seam error (blend-tiled vs whole) — the context number the Swift gate prints.
    d = (tiled - whole).astype(mx.float32)
    cos = float((d.size and (mx.sum(tiled * whole) / (mx.sqrt(mx.sum(tiled * tiled)) * mx.sqrt(mx.sum(whole * whole))))) or 0)
    mse = float(mx.mean(d * d))
    psnr = 99.0 if mse <= 1e-12 else 10 * math.log10(4.0 / mse)
    print(f"oracle blend-tiled vs whole: cosine={cos:.6f}  global PSNR={psnr:.1f} dB  maxAbs={float(mx.abs(d).max()):.3e}")

    mx.save_safetensors(str(OUT / "io.safetensors"),
                        {"latent": latent, "pixels_whole": whole, "pixels_tiled": tiled})
    print("wrote", OUT / "io.safetensors")
    print("latent", latent.shape, "→ pixels", whole.shape)


if __name__ == "__main__":
    main()
