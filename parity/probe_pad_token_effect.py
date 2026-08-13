#!/usr/bin/env python
"""Does the LEFT-PAD TOKEN ID change the text-encoder output? (empirical, not argued)

The Swift port pads with `<unk>` (id 3, `tk.unknownTokenId`); the oracle pads with
`<pad>` (id 0, `pad_token_id`). Both tokenizers share the id layout
(0=<pad>, 1=<eos>, 2=<bos>, 3=<unk>), so the *valid* tokens are identical — only
the ~1000 left-pad slots differ.

There is a plausible two-part argument that this cannot matter:
  (a) the connector's `_replace_padding_with_registers` keeps only
      `hidden_states[b, seq_len - n_valid:, :]` and fills the rest with learned
      registers, and `GemmaFeaturesExtractorV2` zeroes padded positions before the
      projection — padded positions are discarded twice over;
  (b) the combined causal+padding mask (-1e9 on pad KEYS for every query) should
      stop valid tokens attending to pad positions inside Gemma.
This probe MEASURES both instead of trusting either.

METHOD NOTE — the regime is the whole experiment. A near-1024-token prompt has
almost no padding and would show ~0 difference no matter what the pad token does;
that regime cannot discriminate. So this uses a DELIBERATELY TINY prompt
(a handful of tokens, >99% of the 1024 slots are padding) — if pad ids matter at
all, this is where it shows.

Three measurements, two of which are controls that MUST be non-zero — if they come
back 0 the probe is broken, not the port:

  M1  Gemma hidden states at PAD positions, pad0 vs pad3
      → CONTROL. Must differ. Proves the two runs really did see different input
        and that the 12B forward is sensitive to it.
  M2  Gemma hidden states at VALID positions, pad0 vs pad3
      → tests argument (b): the attention mask's isolation of pad keys.
  M3  FINAL connector outputs (video_embeds, audio_embeds — what the DiT consumes)
      → the verdict. This is the thing that would be a live 2.3 bug.
  M4  FINAL connector outputs with the mask REPLACED BY ALL-ONES (padding declared
      valid, so nothing is discarded)
      → POISON CONTROL. Must differ. Proves M3's comparison is capable of
        detecting a pad-id difference at all.

Run (weights should be page-warm first — see the prewarm in the command below):
    cd .../ltx-2-mlx && .venv/bin/python ../ltx-2-mlx-swift/parity/probe_pad_token_effect.py
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

GEMMA_DIR = os.environ.get("GEMMA_DIR", "/Volumes/Satechi/Models/mlx-community/gemma-3-12b-it-4bit")
LTX_DIR = os.environ.get("LTX_DIR", "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx")
MAX_LENGTH = 1024

# DELIBERATELY TINY — see the METHOD NOTE above. ~4 tokens valid, ~1020 padded.
PROMPT = "a red fox"

OUT = Path(__file__).resolve().parent / "goldens" / "pad_token_probe"

os.environ.setdefault("LTX2_GEMMA_EVAL_EVERY", "1")


def stats(a: mx.array, b: mx.array) -> dict:
    """maxAbs + cosine between two arrays, computed in fp32."""
    af = np.array(a.astype(mx.float32)).reshape(-1)
    bf = np.array(b.astype(mx.float32)).reshape(-1)
    max_abs = float(np.max(np.abs(af - bf))) if af.size else 0.0
    na, nb = float(np.linalg.norm(af)), float(np.linalg.norm(bf))
    cos = float(np.dot(af, bf) / (na * nb)) if na > 0 and nb > 0 else 1.0
    n_diff = int(np.count_nonzero(af != bf))
    return {"maxAbs": max_abs, "cosine": cos, "n_differing_elements": n_diff, "n_elements": int(af.size)}


def fmt(name: str, s: dict, expect: str) -> str:
    return (
        f"  {name:<46} maxAbs={s['maxAbs']:.6e}  cosine={s['cosine']:.9f}  "
        f"differing={s['n_differing_elements']}/{s['n_elements']}   [{expect}]"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"prompt": PROMPT, "max_length": MAX_LENGTH, "gemma_dir": GEMMA_DIR, "ltx_dir": LTX_DIR}

    # ---- 1. Build the two token sequences: identical valid tokens, different pad id ----
    te = GemmaLanguageModel()
    te.load(GEMMA_DIR)

    tok = te._tokenizer
    pad_id_oracle = tok.pad_token_id if tok.pad_token_id is not None else 0
    unk_id_swift = tok.unk_token_id if getattr(tok, "unk_token_id", None) is not None else 3
    print(f"[probe] tokenizer: pad_token_id={pad_id_oracle} ({tok.pad_token!r})  "
          f"unk_token_id={unk_id_swift} ({getattr(tok, 'unk_token', None)!r})")

    ids_pad0, mask = te.tokenize(PROMPT, MAX_LENGTH)  # oracle path → pad id 0
    n_valid = int(mx.sum(mask).item())
    n_pad = MAX_LENGTH - n_valid
    print(f"[probe] prompt={PROMPT!r}  valid tokens={n_valid}  padded slots={n_pad} "
          f"({100.0 * n_pad / MAX_LENGTH:.1f}% of the sequence)")
    if n_pad < 0.9 * MAX_LENGTH:
        raise SystemExit(
            f"REFUSING TO RUN: only {n_pad}/{MAX_LENGTH} slots are padding. This regime cannot "
            "discriminate — pick a shorter prompt (see the METHOD NOTE in this file's docstring)."
        )

    # Swift path → same valid tokens, <unk>(3) in the pad slots.
    ids_pad3 = mx.where(mask == 1, ids_pad0, mx.array(unk_id_swift, dtype=ids_pad0.dtype))
    assert int(mx.sum(ids_pad0 != ids_pad3).item()) == n_pad, "pad-swap did not touch exactly the pad slots"
    print(f"[probe] token sequences differ in exactly {n_pad} positions (all padding) ✓")

    report["n_valid"] = n_valid
    report["n_pad"] = n_pad
    report["pad_id_oracle"] = int(pad_id_oracle)
    report["pad_id_swift"] = int(unk_id_swift)

    # ---- 2. Gemma forward, both variants ----
    print("[probe] running Gemma 49-state forward (pad=0) ...")
    states0 = te.get_all_hidden_states(ids_pad0, attention_mask=mask)
    mx.eval(states0)
    print("[probe] running Gemma 49-state forward (pad=3) ...")
    states3 = te.get_all_hidden_states(ids_pad3, attention_mask=mask)
    mx.eval(states3)

    # M1 (control) / M2 — split each state into pad rows and valid rows.
    # Left-padded, so slots [0 : n_pad) are padding and [n_pad : ) are valid.
    #
    # NOTE: seed the "worst" from layer 0 rather than from a zero sentinel. With a
    # `maxAbs > worst` update rule and an all-zero result, a sentinel never gets
    # replaced and the row prints `differing=0/0` — indistinguishable from having
    # measured an EMPTY slice. The all-zero case is the interesting one here, so the
    # report has to show the real element count that was compared.
    worst_valid = worst_pad = None
    worst_valid_layer = worst_pad_layer = -1
    layers_valid_differing = 0
    for i, (a, b) in enumerate(zip(states0, states3)):
        sv = stats(a[:, n_pad:, :], b[:, n_pad:, :])
        sp = stats(a[:, :n_pad, :], b[:, :n_pad, :])
        assert sv["n_elements"] == n_valid * a.shape[-1], "valid slice is empty/wrong — probe is broken"
        assert sp["n_elements"] == n_pad * a.shape[-1], "pad slice is empty/wrong — probe is broken"
        if sv["n_differing_elements"]:
            layers_valid_differing += 1
        key = lambda s: (s["maxAbs"], s["n_differing_elements"])  # noqa: E731
        if worst_valid is None or key(sv) > key(worst_valid):
            worst_valid, worst_valid_layer = sv, i
        if worst_pad is None or key(sp) > key(worst_pad):
            worst_pad, worst_pad_layer = sp, i
    print()
    print(f"[probe] --- Gemma hidden states (49 layers), pad0 vs pad3 "
          f"[{n_valid} valid rows, {n_pad} pad rows, {states0[0].shape[-1]} dims] ---")
    print(fmt(f"M1 PAD positions   (worst = layer {worst_pad_layer})", worst_pad, "CONTROL — must be NON-ZERO"))
    print(fmt(f"M2 VALID positions (worst = layer {worst_valid_layer})", worst_valid, "expect ZERO"))
    print(f"  {'':<46} layers with ANY differing valid element: {layers_valid_differing}/{len(states0)}")
    report["M1_gemma_pad_positions_worst"] = {**worst_pad, "layer": worst_pad_layer}
    report["M2_gemma_valid_positions_worst"] = {**worst_valid, "layer": worst_valid_layer}
    report["M2_layers_with_differing_valid_elements"] = layers_valid_differing

    del te
    mx.clear_cache()

    # ---- 3. Connector forward, both variants (this is what the DiT actually consumes) ----
    print("\n[probe] loading connector ...")
    fe = GemmaFeaturesExtractorV2()
    connector_weights = load_split_safetensors(Path(LTX_DIR) / "connector.safetensors", prefix="connector.")
    fe.connector.load_weights(list(connector_weights.items()))
    del connector_weights

    print("[probe] connector forward (pad=0, real mask) ...")
    v0, a0 = fe(states0, attention_mask=mask)
    mx.eval(v0, a0)
    print("[probe] connector forward (pad=3, real mask) ...")
    v3, a3 = fe(states3, attention_mask=mask)
    mx.eval(v3, a3)

    m3_v, m3_a = stats(v0, v3), stats(a0, a3)
    del v0, a0, v3, a3
    mx.clear_cache()

    # ---- 4. POISON CONTROL: declare every position valid, so nothing is discarded ----
    ones = mx.ones_like(mask)
    print("[probe] POISON CONTROL: connector forward with all-ones mask (pad=0) ...")
    pv0, pa0 = fe(states0, attention_mask=ones)
    mx.eval(pv0, pa0)
    print("[probe] POISON CONTROL: connector forward with all-ones mask (pad=3) ...")
    pv3, pa3 = fe(states3, attention_mask=ones)
    mx.eval(pv3, pa3)

    m4_v, m4_a = stats(pv0, pv3), stats(pa0, pa3)

    print()
    print("[probe] --- FINAL connector outputs (what the DiT consumes), pad0 vs pad3 ---")
    print(fmt("M3 video_embeds  (real mask)", m3_v, "VERDICT — expect ZERO"))
    print(fmt("M3 audio_embeds  (real mask)", m3_a, "VERDICT — expect ZERO"))
    print(fmt("M4 video_embeds  (all-ones mask)", m4_v, "POISON CONTROL — must be NON-ZERO"))
    print(fmt("M4 audio_embeds  (all-ones mask)", m4_a, "POISON CONTROL — must be NON-ZERO"))
    report["M3_video_real_mask"] = m3_v
    report["M3_audio_real_mask"] = m3_a
    report["M4_video_allones_mask"] = m4_v
    report["M4_audio_allones_mask"] = m4_a

    # ---- 5. Verdict ----
    controls_live = worst_pad["n_differing_elements"] > 0 and m4_v["n_differing_elements"] > 0
    benign = m3_v["n_differing_elements"] == 0 and m3_a["n_differing_elements"] == 0
    print()
    if not controls_live:
        verdict = "INCONCLUSIVE — a control came back identical; the probe cannot discriminate"
    elif benign:
        verdict = "BENIGN — pad token id is bit-irrelevant to the DiT-facing embeds"
    else:
        verdict = "LIVE BUG — pad token id changes the DiT-facing embeds; Swift must pad with 0"
    print(f"[probe] VERDICT: {verdict}")
    report["controls_live"] = controls_live
    report["verdict"] = verdict

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[probe] wrote {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
