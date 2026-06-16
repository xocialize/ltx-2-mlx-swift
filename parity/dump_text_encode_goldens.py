#!/usr/bin/env python
"""Dump Gemma-seam + connector goldens from the ltx-2-mlx oracle.

Parity fixture for the Swift port's FIRST vertical slice (Gemma text-encode
→ connector → video/audio embeds). Drives the SAME code path the pipeline
uses (utils/blocks.PromptEncoder) so there is no divergence between this
fixture and production inference.

Run inside the oracle uv env:
    cd ~/Development/ltx-2-mlx && \
        uv run python ~/Development/ltx-2-mlx-swift/parity/dump_text_encode_goldens.py

Outputs .npy goldens under ../goldens/text_encode/:
    token_ids.npy            (1, 1024)        int
    attention_mask.npy       (1, 1024)        int  (1=valid, 0=left-pad)
    gemma_hidden_NN.npy      (1, 1024, 3840)  fp32  x49  (embed + 48 layers)
    video_embeds.npy         (1, 1024, 4096)  fp32
    audio_embeds.npy         (1, 1024, 2048)  fp32
    meta.json                prompt + shapes + dtypes + mlx version
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
from ltx_core_mlx.utils.weights import load_split_safetensors

# --- Fixed inputs (keep stable across reruns; the Swift parity test pins these) ---
GEMMA_DIR = os.environ.get("GEMMA_DIR", "/Volumes/DEV_ARCHIVE/models/mlx-community/gemma-3-12b-it-4bit")
LTX_DIR = os.environ.get("LTX_DIR", "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx")
PROMPT = "A red fox trotting through a snowy forest at dawn, cinematic lighting, shallow depth of field."
MAX_LENGTH = 1024
OUT_DIR = Path(__file__).resolve().parent / "goldens" / "text_encode"

# Per-layer eval keeps each Metal command buffer under the watchdog (oracle default).
os.environ.setdefault("LTX2_GEMMA_EVAL_EVERY", "1")


def _np(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Gemma: tokenize + extract ALL 49 hidden states (embed + 48 layers)
    te = GemmaLanguageModel()
    te.load(GEMMA_DIR)
    token_ids, attention_mask = te.tokenize(PROMPT, MAX_LENGTH)
    states = te.get_all_hidden_states(token_ids, attention_mask=attention_mask)
    mx.eval(states)

    # 2. Connector: dual projection + 8-block 1D transformers → video/audio embeds
    fe = GemmaFeaturesExtractorV2()
    connector_weights = load_split_safetensors(Path(LTX_DIR) / "connector.safetensors", prefix="connector.")
    fe.connector.load_weights(list(connector_weights.items()))
    video_embeds, audio_embeds = fe(states, attention_mask=attention_mask)
    mx.eval(video_embeds, audio_embeds)

    # 3. Save goldens
    np.save(OUT_DIR / "token_ids.npy", np.array(token_ids))
    np.save(OUT_DIR / "attention_mask.npy", np.array(attention_mask))
    for i, s in enumerate(states):
        np.save(OUT_DIR / f"gemma_hidden_{i:02d}.npy", _np(s))
    np.save(OUT_DIR / "video_embeds.npy", _np(video_embeds))
    np.save(OUT_DIR / "audio_embeds.npy", _np(audio_embeds))

    meta = {
        "prompt": PROMPT,
        "max_length": MAX_LENGTH,
        "gemma_dir": GEMMA_DIR,
        "ltx_dir": LTX_DIR,
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "num_hidden_states": len(states),
        "shapes": {
            "token_ids": list(token_ids.shape),
            "attention_mask": list(attention_mask.shape),
            "gemma_hidden": list(states[0].shape),
            "video_embeds": list(video_embeds.shape),
            "audio_embeds": list(audio_embeds.shape),
        },
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print("Wrote goldens to", OUT_DIR)
    print(json.dumps(meta["shapes"], indent=2))
    print("video_embeds stats: mean=%.5f std=%.5f" % (float(video_embeds.mean()), float(video_embeds.std())))
    print("audio_embeds stats: mean=%.5f std=%.5f" % (float(audio_embeds.mean()), float(audio_embeds.std())))


if __name__ == "__main__":
    main()
