#!/usr/bin/env python
"""LTX-2.5 Gemma-4 49-state encoder fixture.

Dumps token ids, attention mask, and all 49 hidden states from the oracle's
``Gemma4LanguageModel`` so the Swift tap can be gated per state.

⚠️ THREE deltas from the Gemma-3 fixture, each a silent divergence if the Swift
tap copies the 2.3 code:
  * embed scale is a plain Float (2.3 pre-rounds it to bfloat16),
  * the FINAL NORM lands on the last state only (HF output_hidden_states),
  * the norm convention flips — Gemma-3 is rmsNorm(x, 1 + w), Gemma-4 has no shift.
The per-state comparison localises which of these broke: a wrong embed scale moves
state 00, a wrong final norm moves ONLY state 48, and a wrong norm convention inside
the stack moves everything from 01 on.

Loads the 23.8 GB bf16 encoder — run on a machine with headroom.

    cd .../ltx-2-mlx && uv run python .../ltx-2-mlx-swift/parity/dump_gemma4_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import resolve_text_encoder

MODEL_DIR = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
OUT_DIR = Path(__file__).resolve().parent / "goldens" / "gemma4"
PROMPT = (
    "A lone lighthouse on a rocky headland during a storm, waves exploding against "
    "the rocks, the beam sweeping through sheets of rain."
)
MAX_LENGTH = 1024


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enc = resolve_text_encoder(MODEL_DIR)
    print(f"encoder: {type(enc).__name__} — loading (23.8 GB bf16)...")
    enc.load()

    token_ids, attention_mask = enc.tokenize(PROMPT, max_length=MAX_LENGTH)
    states = enc.get_all_hidden_states(token_ids, attention_mask)
    mx.eval(token_ids, attention_mask, *states)
    print(f"states: {len(states)}  shape {tuple(states[0].shape)}")

    n_valid = int(mx.sum(attention_mask).item())
    out = {
        "token_ids": token_ids.astype(mx.int32),
        "attention_mask": attention_mask.astype(mx.int32),
    }
    for i, st in enumerate(states):
        out[f"gemma_hidden_{i:02d}"] = st.astype(mx.float32)
    mx.save_safetensors(str(OUT_DIR / "goldens.safetensors"), out)

    # Prompt ships alongside so the Swift side tokenizes byte-identical input.
    (OUT_DIR / "meta.json").write_text(
        json.dumps({"prompt": PROMPT, "max_length": MAX_LENGTH,
                    "n_valid": n_valid, "n_states": len(states)}, indent=1)
    )
    print(f"wrote {OUT_DIR}  (n_valid={n_valid})")
    # Sanity the fixture can discriminate: state 48 must differ from state 47,
    # otherwise a missing final norm would be invisible.
    d = float(mx.abs(states[-1] - states[-2]).max())
    print(f"  |state48 - state47| maxAbs = {d:.4f} (final-norm discrimination; must be >> 0)")


if __name__ == "__main__":
    main()
