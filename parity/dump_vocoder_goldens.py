#!/usr/bin/env python
"""Vocoder+BWE parity fixture: mel → 48kHz stereo waveform (fp32).

Run in the oracle uv env:
    cd ~/Development/ltx-2-mlx && \
        uv run python ~/Development/ltx-2-mlx-swift/parity/dump_vocoder_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
from ltx_core_mlx.utils.weights import load_split_safetensors

MODEL_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
OUT = Path("/Users/dustinnielson/Development/ltx-2-mlx-swift/parity/goldens/vocoder")
T = 16  # mel frames


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    voc = VocoderWithBWE()
    w = load_split_safetensors(MODEL_DIR / "vocoder.safetensors", prefix="vocoder.")
    voc.load_weights(list(w.items()))
    voc.upcast_weights_to_fp32()
    mx.eval(voc.parameters())

    mx.random.seed(13)
    mel = mx.random.normal((1, 2, T, 64)).astype(mx.float32)
    wav = voc(mel)  # (1, 2, T_48k)
    mx.eval(wav)

    mx.save_safetensors(str(OUT / "io.safetensors"), {"mel": mel, "wav": wav})
    print("wrote", OUT)
    print("mel", mel.shape, "→ wav", wav.shape,
          "mean=%.5f std=%.5f min=%.3f max=%.3f" % (float(wav.mean()), float(wav.std()), float(wav.min()), float(wav.max())))


if __name__ == "__main__":
    main()
