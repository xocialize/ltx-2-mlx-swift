#!/usr/bin/env python
"""x1.5-spatial + x2-temporal upsampler parity fixtures (fp32).

Exercises LatentUpsampler.from_config on each checkpoint's sidecar config — the same
construction path the Swift port's sidecar-driven `Upsampler.load(path:)` mirrors.

Run in the oracle uv env (page the weights in first):
    cd <LTX_DEV>/ltx-2-mlx && uv run python ../ltx-2-mlx-swift/parity/dump_upsampler_variants_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.utils.weights import load_split_safetensors

MODEL_DIR = Path("/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx")
GOLDENS = Path(__file__).resolve().parent / "goldens"

CASES = [
    # (checkpoint stem, goldens dir, input F/H/W)
    ("spatial_upscaler_x1_5_v1_0", "upsampler_x1_5", (2, 4, 4)),   # H,W 4→6 (×1.5)
    ("temporal_upscaler_x2_v1_0", "upsampler_temporal", (3, 4, 4)),  # F 3→5 (2F−1)
]


def main() -> None:
    for stem, out_name, (F, H, W) in CASES:
        cfg = json.loads((MODEL_DIR / f"{stem}_config.json").read_text())["config"]
        up = LatentUpsampler.from_config(cfg)

        raw = load_split_safetensors(MODEL_DIR / f"{stem}.safetensors")
        prefix = f"{stem}."
        if raw and all(k.startswith(prefix) for k in raw):
            raw = {k[len(prefix):]: v for k, v in raw.items()}
        raw = {k: v.astype(mx.float32) for k, v in raw.items()}
        up.load_weights(list(raw.items()))
        mx.eval(up.parameters())

        mx.random.seed(31)
        latent = mx.random.normal((1, 128, F, H, W)).astype(mx.float32)
        out = up(latent)
        mx.eval(out)

        out_dir = GOLDENS / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(out_dir / "io.safetensors"), {"latent": latent, "out": out})
        print(f"{stem}: {tuple(latent.shape)} -> {tuple(out.shape)}  std={float(out.std()):.5f}")


if __name__ == "__main__":
    main()
