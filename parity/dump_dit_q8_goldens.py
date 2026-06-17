#!/usr/bin/env python
"""q8 DiT parity fixture: int8-quantized transformer forward (bf16 activations).

Loads the real q8 distilled transformer (transformer-blocks Linears int8, group_size 64;
everything else bf16) into the oracle LTXModel via apply_quantization, runs ONE forward on
the SAME inputs as the dit_full fixture, and dumps the q8 velocities. The Swift q8 DiT loads
the same q8 weights and compares.

Run in the oracle uv env:
    cd ~/Development/mlxengine-video/LTX_DEV/ltx-2-mlx && \
        uv run python ../ltx-2-mlx-swift/parity/dump_dit_q8_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors

Q8_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx-q8")
IO = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/dit_full/io.safetensors")
OUT = Path("/Users/dustinnielson/Development/mlxengine-video/LTX_DEV/ltx-2-mlx-swift/parity/goldens/dit_q8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = LTXModelConfig.from_checkpoint_dir(Q8_DIR)
    print("config: layers=%d video_dim=%d audio_dim=%d" % (cfg.num_layers, cfg.video_dim, cfg.audio_dim))
    model = LTXModel(cfg)
    w = load_split_safetensors(Q8_DIR / "transformer-distilled.safetensors", prefix="transformer.")
    apply_quantization(model, w, group_size=64)  # nn.quantize the Linears that carry scales/biases
    model.load_weights(list(w.items()))
    mx.eval(model.parameters())

    io = mx.load(str(IO))  # reuse the dit_full inputs (Nv=192, Na=16) + real text embeds + positions
    vv, av = model(
        video_latent=io["video_latent"], audio_latent=io["audio_latent"], timestep=io["sigma"],
        video_text_embeds=io["video_text"], audio_text_embeds=io["audio_text"],
        video_positions=io["video_positions"], audio_positions=io["audio_positions"])
    mx.eval(vv, av)

    mx.save_safetensors(str(OUT / "io.safetensors"), {"video_v": vv.astype(mx.float32), "audio_v": av.astype(mx.float32)})
    print("wrote", OUT)
    print("video_v", vv.shape, "std=%.5f" % float(vv.std()), " audio_v", av.shape, "std=%.5f" % float(av.std()))


if __name__ == "__main__":
    main()
