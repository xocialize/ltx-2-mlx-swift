#!/usr/bin/env python
"""Composed audio-decode parity: audio latent tokens → 48kHz waveform (fp32).

Validates the LTX2Pipeline.decodeAudio composition: AudioPatchifier.unpatchify
((1,T,128)→(1,8,T,16)) → AudioVAEDecoder → VocoderWithBWE.

Run in the oracle uv env:
    cd ~/Development/ltx-2-mlx && \
        uv run python ~/Development/ltx-2-mlx-swift/parity/dump_audio_decode_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import AudioPatchifier
from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

MODEL_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
OUT = Path("/Users/dustinnielson/Development/ltx-2-mlx-swift/parity/goldens/audio_decode")
T = 16


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Audio VAE decoder
    dec = AudioVAEDecoder()
    w = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.decoder.")
    allA = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.")
    for k, v in allA.items():
        if k.startswith("per_channel_statistics."):
            w[k] = v
    dec.load_weights(list(remap_audio_vae_keys(w).items()))
    # Vocoder
    voc = VocoderWithBWE()
    voc.load_weights(list(load_split_safetensors(MODEL_DIR / "vocoder.safetensors", prefix="vocoder.").items()))
    voc.upcast_weights_to_fp32()
    mx.eval(dec.parameters(), voc.parameters())

    mx.random.seed(21)
    tokens = mx.random.normal((1, T, 128)).astype(mx.float32)
    audio_latent = AudioPatchifier().unpatchify(tokens)  # (1,8,T,16)
    mel = dec.decode(audio_latent)
    wav = voc(mel)
    mx.eval(wav)

    mx.save_safetensors(str(OUT / "io.safetensors"), {"tokens": tokens, "wav": wav})
    print("wrote", OUT)
    print("tokens", tokens.shape, "→ wav", wav.shape, "std=%.5f" % float(wav.std()))


if __name__ == "__main__":
    main()
