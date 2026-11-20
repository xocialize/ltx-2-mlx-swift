"""Layer-0 SDPA check on the saved LTX Gemma-3 tensors.

Usage: python -I layer0_check.py <layer0_sdpa_inputs.safetensors>
Compares: fused (d-split on M5), chunked 512-row (non-d-split), and float64 references,
separately for valid query rows and fully masked (left-padded) query rows.
"""
import sys

import mlx.core as mx
import numpy as np

d = mx.load(sys.argv[1])
q, k, v, mask = d["q"], d["k"], d["v"], d["mask"]
valid = np.array(d["attention_mask"][0]).astype(bool)
scale = 1 / 16
B, H, T, D = q.shape
Hk = k.shape[1]
print(f"mlx {mx.__version__}  q{q.shape} k{k.shape} mask{mask.shape} {mask.dtype}  valid={valid.sum()}")

fused = _ = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
chunks = [mx.fast.scaled_dot_product_attention(q[:, :, s:s + 512], k, v, scale=scale, mask=mask[:, :, s:s + 512])
          for s in range(0, T, 512)]
chunked = mx.concatenate(chunks, axis=2)
# MLX's own unfused composition, the same ops the fallback runs (bf16 matmul, precise softmax)
kr = mx.repeat(k, H // Hk, axis=1)
vr = mx.repeat(v, H // Hk, axis=1)
unfused = mx.softmax((q * scale) @ kr.transpose(0, 1, 3, 2) + mask, axis=-1, precise=True) @ vr
mx.eval(fused, chunked, unfused)

Q = np.array(q.astype(mx.float32), dtype=np.float64)
K = np.repeat(np.array(k.astype(mx.float32), dtype=np.float64), H // Hk, axis=1)
V = np.repeat(np.array(v.astype(mx.float32), dtype=np.float64), H // Hk, axis=1)
M = np.array(mask.astype(mx.float32), dtype=np.float64)
S = (Q @ K.transpose(0, 1, 3, 2)) * scale


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


ref64 = softmax(S + M) @ V  # exact float64: masked rows keep their real score differences
# "uniform" reading of a fully masked row: every key whose mask equals the row max counts equally
Mrow = M + 0 * S
eq = (Mrow == Mrow.max(-1, keepdims=True)).astype(np.float64)
uniform = (eq / eq.sum(-1, keepdims=True)) @ V

s_pad = S[..., ~valid, :]
print(f"scaled scores, padded rows: |s| max {np.abs(s_pad).max():.1f}, per-row spread "
      f"median {np.median(s_pad.max(-1) - s_pad.min(-1)):.1f}, max {(s_pad.max(-1) - s_pad.min(-1)).max():.1f}")
print("  (fp32 spacing at 1e9 is 64, at 2e9 is 128: a spread above that survives '-1e9 + s' in fp32)")

arms = dict(fused_dsplit=fused, chunked512=chunked, mlx_unfused=unfused)
refs = dict(ref64_exact=ref64, ref_uniform=uniform)


def cmp(a, b, rows):
    a = a[:, :, rows].ravel()
    b = b[:, :, rows].ravel()
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos, np.abs(a - b).max()


print(f"{'arm':14s} {'vs':12s} {'valid cos':>12s} {'valid maxAbs':>13s} {'pad cos':>10s} {'pad maxAbs':>11s}")
for an, a in arms.items():
    A = np.array(a.astype(mx.float32), dtype=np.float64)
    for rn, r in list(refs.items()) + [("chunked512", np.array(chunked.astype(mx.float32), dtype=np.float64))]:
        if rn == an:
            continue
        cv, mv = cmp(A, r, valid)
        cp, mp = cmp(A, r, ~valid)
        print(f"{an:14s} {rn:12s} {cv:12.6f} {mv:13.5f} {cp:10.6f} {mp:11.5f}")
pre = np.array(d["o_prebuilt_fused"].astype(mx.float32))
print("saved o_prebuilt_fused == fused now:", bool(np.array_equal(pre, np.array(fused.astype(mx.float32)))))
