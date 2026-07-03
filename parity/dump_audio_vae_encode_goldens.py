#!/usr/bin/env python
"""Audio VAE encode parity fixture: waveform → mel → latent (+ LipDub patchify) (fp32).

Run in the oracle uv env:
    cd ~/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx && \
        uv run python ~/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/dump_audio_vae_encode_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import AudioPatchifier
from ltx_core_mlx.model.audio_vae import AudioProcessor, AudioVAEEncoder, encode_audio
from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys
from ltx_pipelines_mlx.lipdub import patchify_lipdub_audio_reference_latent

MODEL_DIR = Path("/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
OUT = Path(
    "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/audio_vae_encode"
)
SAMPLE_RATE = 16000
SAMPLES = 16000  # 1 s stereo → mel T'=101 → latent T=26


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    enc = AudioVAEEncoder()
    # Faithful oracle load (utils/blocks.py _AudioConditionerBlock): encoder convs under
    # "audio_vae.encoder.", per-channel stats under "audio_vae." (outside encoder), then
    # remap _mean_of_means → mean_of_means. fp32 for the apples-to-apples golden.
    w = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.encoder.")
    all_audio = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.")
    for k, v in all_audio.items():
        if k.startswith("per_channel_statistics."):
            w[k] = v
    w = remap_audio_vae_keys(w)
    w = {k: v.astype(mx.float32) for k, v in w.items()}
    enc.load_weights(list(w.items()))
    mx.eval(enc.parameters())

    processor = AudioProcessor(sample_rate=SAMPLE_RATE)

    mx.random.seed(11)
    # Seeded stereo "waveform" (1, 2, S) — the mel's log(max(., 1e-5)) clamp makes the
    # distribution irrelevant for parity; a scaled normal keeps most bins above the clamp.
    waveform = (mx.random.normal((1, 2, SAMPLES)) * 0.1).astype(mx.float32)

    mel = processor.waveform_to_mel(waveform)  # (1, 2, T', 64)
    latent = encode_audio(waveform, SAMPLE_RATE, enc, processor)  # (1, 8, T, 16)
    mx.eval(mel, latent)

    # LipDub append helper: patchified tokens + NEGATIVE-time positions.
    ref_tokens, ref_positions = patchify_lipdub_audio_reference_latent(
        latent, AudioPatchifier(), negative_positions=True
    )
    mx.eval(ref_tokens, ref_positions)

    mx.save_safetensors(
        str(OUT / "io.safetensors"),
        {
            "waveform": waveform,
            "mel": mel.astype(mx.float32),
            "latent": latent.astype(mx.float32),
            "ref_tokens": ref_tokens.astype(mx.float32),
            "ref_positions": ref_positions.astype(mx.float32),
            # Diagnostics: Swift computes its own filterbank/window — compare directly.
            "mel_basis": processor.mel_basis.astype(mx.float32),
            "window": processor.window.astype(mx.float32),
        },
    )
    print("wrote", OUT)
    print(
        "waveform", waveform.shape, "→ mel", mel.shape, "→ latent", latent.shape,
        "mean=%.5f std=%.5f" % (float(latent.mean()), float(latent.std())),
    )
    print("ref_tokens", ref_tokens.shape, "ref_positions", ref_positions.shape,
          "pos range [%.4f, %.4f]" % (float(ref_positions.min()), float(ref_positions.max())))


if __name__ == "__main__":
    main()
