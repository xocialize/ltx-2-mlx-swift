"""Per-layer SDPA comparison along the real (golden-path) trajectory.

Usage: python -I per_layer.py <goldens.safetensors> <gemma_dir> <out_dir>
Runs the 49-state tap with the NON-d-split path (chunked 512-row queries, which reproduces the
goldens bit-for-bit), and at every layer also runs the fused d-split kernel on the SAME q/k/v/mask.
Reports per layer: fused-vs-chunked agreement on valid and padded rows, and the padded rows'
scaled-score spread. Saves the inputs of the worst layer.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

goldens_path, gemma_dir, out_dir = sys.argv[1:4]
SAVE_LAYER = int(sys.argv[4]) if len(sys.argv) > 4 else -1
out = Path(out_dir)
G = mx.load(goldens_path)
token_ids = G["token_ids"].astype(mx.int32)
attn = G["attention_mask"]
T = token_ids.shape[1]
valid = np.array(attn[0]).astype(bool)

_sdpa = mx.fast.scaled_dot_product_attention
rows = []


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def sdpa(q, k, v, *, scale, mask=None, **kw):
    chunked = mx.concatenate(
        [_sdpa(q[:, :, s:s + 512], k, v, scale=scale, mask=mask[:, :, s:s + 512], **kw) for s in range(0, T, 512)],
        axis=2)
    fused = _sdpa(q, k, v, scale=scale, mask=mask, **kw)
    kr = mx.repeat(k, q.shape[1] // k.shape[1], axis=1)
    s_ = (q.astype(mx.float32) @ kr.astype(mx.float32).transpose(0, 1, 3, 2)) * scale
    pad_rows = mx.array(np.where(~valid)[0])
    sp = s_[:, :, pad_rows]
    spread = sp.max(-1) - sp.min(-1)
    mx.eval(chunked, fused, spread)
    C = np.array(chunked.astype(mx.float32))[0]
    F = np.array(fused.astype(mx.float32))[0]
    sp_np = np.array(spread)
    i = len(rows)
    rows.append(dict(
        layer=i,
        cos_valid=cos(F[:, valid], C[:, valid]), maxabs_valid=float(np.abs(F[:, valid] - C[:, valid]).max()),
        cos_pad=cos(F[:, ~valid], C[:, ~valid]), maxabs_pad=float(np.abs(F[:, ~valid] - C[:, ~valid]).max()),
        pad_score_spread_median=float(np.median(sp_np)), pad_score_spread_max=float(sp_np.max()),
        pad_absmax_out=float(np.abs(C[:, ~valid]).max()),
    ))
    if i == SAVE_LAYER:
        mx.save_safetensors(str(out / f"layer{i:02d}_sdpa_inputs.safetensors"),
                            dict(q=q, k=k, v=v, mask=mask, o_prebuilt_fused=fused, o_chunked512=chunked,
                                 attention_mask=attn.astype(mx.int32)),
                            metadata=dict(scale=repr(scale), mlx=mx.__version__, layer=str(i),
                                          note="LTX Gemma-3 12B 4-bit tap, golden-path inputs, bf16"))
    return chunked


mx.fast.scaled_dot_product_attention = sdpa
model, _ = load(gemma_dir)
lm = model
for attr in ("model", "language_model", "model"):
    if hasattr(lm, attr):
        lm = getattr(lm, attr)
    if hasattr(lm, "embed_tokens"):
        break

causal = mx.triu(mx.full((T, T), -1e9, dtype=mx.bfloat16), k=1)
mask = causal[None, None] + (1 - attn[:, None, None, :].astype(mx.bfloat16)) * -1e9
h = lm.embed_tokens(token_ids)
h = h * mx.array(h.shape[-1] ** 0.5, mx.bfloat16).astype(h.dtype)
for layer in lm.layers:
    h = layer(h, mask=mask, cache=None)
    mx.eval(h)

print(f"{'L':>3} {'valid cos':>10} {'valid max':>10} {'pad cos':>10} {'pad max':>10} {'spread med':>10} {'spread max':>10}")
for r in rows:
    print(f"{r['layer']:3d} {r['cos_valid']:10.6f} {r['maxabs_valid']:10.3f} {r['cos_pad']:10.6f} {r['maxabs_pad']:10.3f} "
          f"{r['pad_score_spread_median']:10.1f} {r['pad_score_spread_max']:10.1f}")
worst = min(rows, key=lambda r: r["cos_pad"])
print("worst padded-row layer:", worst["layer"], f"cos_pad={worst['cos_pad']:.6f}")
(out / "per_layer.json").write_text(json.dumps(rows, indent=1))
