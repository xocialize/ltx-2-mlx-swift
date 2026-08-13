#!/usr/bin/env python
"""Dump TOKENIZATION goldens for `GemmaEncoder.tokenize` (`--gemma-tokenizer-gate`).

Why this exists: `--gemma-gate` reads `token_ids`/`attention_mask` FROM the golden
file and feeds them straight to the 49-state forward, so the Swift port's own
`GemmaEncoder.tokenize` is exercised by NO gate on the board. A text encoder can
gate at cosine 1.0 while being fed wrongly-tokenized input in production. This
fixture closes that hole: it pins the oracle's exact integer output for
`GemmaLanguageModel.tokenize` (base_encoder.py) across the cases where the two
implementations could plausibly diverge.

Cases (each must be reproduced EXACTLY, integer equality — not cosine):
    00 empty            "" — BOS only; the whole sequence is padding
    01 short            a handful of tokens; >99% padding
    02 canonical        the realistic prompt the other text goldens use
    03 unicode/emoji    non-BMP emoji, CJK, combining accents — byte-level BPE surface
    04 whitespace       leading/trailing spaces + newlines — Python .strip() vs
                        Swift .trimmingCharacters(in: .whitespacesAndNewlines)
    05 over-length      > max_length tokens → exercises truncate-to-LAST-1024
                        (which also DROPS the BOS, since BOS is at position 0)
    06 exactly-1024     the no-padding / no-truncation boundary

Run inside the oracle env (tokenizer only — no model weights are loaded):
    cd .../ltx-2-mlx && .venv/bin/python ../ltx-2-mlx-swift/parity/dump_gemma_tokenizer_goldens.py

Outputs under parity/goldens/gemma_tokenizer/:
    goldens.safetensors   caseNN_token_ids (1,1024) int32, caseNN_attention_mask (1,1024) int32,
                          caseNN_raw_ids (n,) int32   [pre-pad, post-truncate]
    prompts.json          the EXACT prompt strings (UTF-8) + tokenizer special ids + per-case meta
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

GEMMA_DIR = os.environ.get("GEMMA_DIR", "/Volumes/Satechi/Models/mlx-community/gemma-3-12b-it-4bit")
MAX_LENGTH = 1024
OUT_DIR = Path(__file__).resolve().parent / "goldens" / "gemma_tokenizer"

CANONICAL = "A red fox trotting through a snowy forest at dawn, cinematic lighting, shallow depth of field."
UNICODE = "café 🦊 雪の森 — naïve señor, 日本語テスト 🎬✨ Ωmega ﬁ ligature, é combining"
WHITESPACE = "  \n\t  a red fox in the snow  \n\n  "


def _build_over_length(tokenizer, target: int) -> str:
    """A deterministic prompt whose token count exceeds `target`."""
    unit = "segment {i} shows a red fox trotting through a snowy forest at dawn. "
    n = 1
    while True:
        text = "".join(unit.format(i=i) for i in range(n))
        if len(tokenizer.encode(text.strip())) > target:
            return text
        n *= 2


def _build_exact_length(tokenizer, target: int) -> str:
    """A deterministic prompt tokenizing to EXACTLY `target` tokens (boundary case).

    Grows a word at a time, then trims from the front; asserts the final count.
    """
    word = "fox "
    text = word * target
    ids = tokenizer.encode(text.strip())
    # One token per "fox" plus BOS overshoots; drop words until the count lands on target.
    while len(ids) > target:
        text = text[: -len(word) * max(1, (len(ids) - target) // 2 or 1)]
        ids = tokenizer.encode(text.strip())
    while len(ids) < target:
        text = text + word
        ids = tokenizer.encode(text.strip())
    assert len(ids) == target, f"could not hit {target} exactly (got {len(ids)})"
    return text


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The tokenizer alone — same object `GemmaLanguageModel.load` installs, without the 12B weights.
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    tokenizer = load_tokenizer(Path(GEMMA_DIR))
    te = GemmaLanguageModel()
    te._tokenizer = tokenizer  # tokenize() only touches the tokenizer

    cases = [
        ("empty", ""),
        ("short", "a red fox"),
        ("canonical", CANONICAL),
        ("unicode_emoji", UNICODE),
        ("whitespace", WHITESPACE),
        ("over_length", _build_over_length(tokenizer, MAX_LENGTH)),
        ("exactly_max", _build_exact_length(tokenizer, MAX_LENGTH)),
    ]

    packed: dict[str, mx.array] = {}
    meta_cases = []
    for idx, (name, prompt) in enumerate(cases):
        token_ids, attention_mask = te.tokenize(prompt, MAX_LENGTH)
        raw = tokenizer.encode(prompt.strip())
        truncated = len(raw) > MAX_LENGTH
        kept = raw[-MAX_LENGTH:] if truncated else raw
        n_valid = int(mx.sum(attention_mask).item())
        assert n_valid == len(kept), f"{name}: mask says {n_valid} valid, kept {len(kept)}"

        packed[f"case{idx:02d}_token_ids"] = token_ids.astype(mx.int32)
        packed[f"case{idx:02d}_attention_mask"] = attention_mask.astype(mx.int32)
        packed[f"case{idx:02d}_raw_ids"] = mx.array(kept, dtype=mx.int32)

        meta_cases.append(
            {
                "index": idx,
                "name": name,
                "prompt": prompt,
                "raw_token_count": len(raw),
                "kept_token_count": len(kept),
                "truncated": truncated,
                "n_valid": n_valid,
                "n_pad": MAX_LENGTH - n_valid,
                "first_kept_id": int(kept[0]) if kept else None,
                "starts_with_bos": bool(kept and kept[0] == tokenizer.bos_token_id),
            }
        )
        print(
            f"[tok-goldens] {idx:02d} {name:<14} raw={len(raw):>5} kept={len(kept):>5} "
            f"valid={n_valid:>5} pad={MAX_LENGTH - n_valid:>5} "
            f"truncated={truncated} bos={meta_cases[-1]['starts_with_bos']}"
        )

    mx.save_safetensors(str(OUT_DIR / "goldens.safetensors"), packed)

    meta = {
        "max_length": MAX_LENGTH,
        "gemma_dir": GEMMA_DIR,
        "mlx_version": getattr(mx, "__version__", "unknown"),
        # The oracle pads with pad_token_id; the Swift port pads with unknownTokenId.
        # Both are recorded so the gate can state the divergence explicitly instead of
        # hiding it behind a tolerance. See probe_pad_token_effect.py for the verdict.
        "pad_token_id": int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0),
        "unk_token_id": int(tokenizer.unk_token_id) if tokenizer.unk_token_id is not None else None,
        "bos_token_id": int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else None,
        "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        "cases": meta_cases,
    }
    (OUT_DIR / "prompts.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[tok-goldens] wrote {OUT_DIR}/goldens.safetensors + prompts.json")
    print(f"[tok-goldens] pad={meta['pad_token_id']} unk={meta['unk_token_id']} bos={meta['bos_token_id']}")


if __name__ == "__main__":
    main()
