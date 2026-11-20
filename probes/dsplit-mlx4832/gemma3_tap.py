"""LTX Gemma-3 49-state tap under pip mlx (prebuilt metallib), d-split on vs forced off.

Usage: python -I gemma3_tap.py <goldens.safetensors> <gemma_dir> <out_dir> <mode>
  mode = fused    -> stock mx.fast.scaled_dot_product_attention (T=1024 takes d-split on M5)
  mode = chunked  -> same op, but queries split into 512-row chunks (below the d-split threshold)

Mirrors ltx_core_mlx GemmaLanguageModel.get_all_hidden_states: one uniform combined
causal+padding bf16 mask (-1e9) on every layer, per-layer eval.
Saves layer-0 SDPA inputs/outputs (fused mode) and prints per-state cosine vs goldens,
split into valid (prompt) positions and padded positions.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

goldens_path, gemma_dir, out_dir, mode = sys.argv[1:5]
out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)

G = mx.load(goldens_path)
token_ids = G["token_ids"].astype(mx.int32)
attn = G["attention_mask"]
T = token_ids.shape[1]
valid = np.array(attn[0]).astype(bool)
print(f"[{mode}] mlx {mx.__version__}  T={T}  valid tokens={valid.sum()}  padded={(~valid).sum()}")

_sdpa = mx.fast.scaled_dot_product_attention
captured = {}


def sdpa(q, k, v, *, scale, mask=None, **kw):
    if not captured:
        captured.update(q=q, k=k, v=v, mask=mask, scale=scale)
    if mode == "chunked" and q.shape[2] >= 1024:
        outs = []
        for s in range(0, q.shape[2], 512):
            m = mask[:, :, s : s + 512, :] if isinstance(mask, mx.array) else mask
            outs.append(_sdpa(q[:, :, s : s + 512], k, v, scale=scale, mask=m, **kw))
        return mx.concatenate(outs, axis=2)
    return _sdpa(q, k, v, scale=scale, mask=mask, **kw)


mx.fast.scaled_dot_product_attention = sdpa

model, _ = load(gemma_dir)
lm = model
for attr in ("model", "language_model", "model"):  # same walk as base_encoder.py
    if hasattr(lm, attr):
        lm = getattr(lm, attr)
    if hasattr(lm, "embed_tokens"):
        break

causal = mx.triu(mx.full((T, T), -1e9, dtype=mx.bfloat16), k=1)
pad = (1 - attn[:, None, None, :].astype(mx.bfloat16)) * -1e9
mask = causal[None, None] + pad

h = lm.embed_tokens(token_ids)
h = h * mx.array(h.shape[-1] ** 0.5, mx.bfloat16).astype(h.dtype)
states = [h]
for layer in lm.layers:
    h = layer(h, mask=mask, cache=None)
    mx.eval(h)
    states.append(h)


def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


rows = []
for i, s in enumerate(states):
    a = np.array(s[0].astype(mx.float32))
    g = np.array(G[f"gemma_hidden_{i:02d}"][0].astype(mx.float32))
    rows.append(
        dict(
            layer=i,
            cos_all=cos(a, g),
            cos_valid=cos(a[valid], g[valid]),
            cos_pad=cos(a[~valid], g[~valid]),
            maxabs_valid=float(np.abs(a[valid] - g[valid]).max()),
            maxabs_pad=float(np.abs(a[~valid] - g[~valid]).max()),
        )
    )
for r in rows[:3] + rows[-1:]:
    print("[{m}] layer {layer:02d} cos all={cos_all:.6f} valid={cos_valid:.6f} pad={cos_pad:.6f}  "
          "maxAbs valid={maxabs_valid:.4f} pad={maxabs_pad:.4f}".format(m=mode, **r))
for key in ("cos_all", "cos_valid", "cos_pad"):
    v = [r[key] for r in rows]
    print(f"[{mode}] {key}: mean={np.mean(v):.6f} worst={np.min(v):.6f} (layer {int(np.argmin(v))})")
(out / f"tap_{mode}.json").write_text(json.dumps(dict(mlx=mx.__version__, rows=rows), indent=1))

if mode == "fused":
    q, k, v, m, sc = (captured[x] for x in ("q", "k", "v", "mask", "scale"))
    o_fused = _sdpa(q, k, v, scale=sc, mask=m)
    mx.save_safetensors(
        str(out / "layer0_sdpa_inputs.safetensors"),
        dict(q=q, k=k, v=v, mask=m, o_prebuilt_fused=o_fused, attention_mask=attn.astype(mx.int32)),
        metadata=dict(scale=repr(sc), mlx=mx.__version__, note="LTX Gemma-3 12B 4-bit, layer 0, bf16"),
    )
    print(f"[fused] saved layer-0 q{q.shape} k{k.shape} v{v.shape} mask{m.shape} {m.dtype} scale={sc}")
