"""fp32 emulation of how each path adds a finite -1e9 mask, on saved SDPA inputs.

Usage: python -I emulate.py <layerNN_sdpa_inputs.safetensors>
  unfused (fallback):  x = fp32(s*scale) + fp32(mask);          softmax(x)
  NAX steel kernels:   x = fp32(s*scale*log2e) + log2e*fp32(mask); exp2-softmax(x)
In a fully masked row every logit sits near -1e9 (-1.44e9 in log2 units), where the fp32 grid is
64 (128), so score differences are quantized to that grid and only the keys that round up survive.
"""
import sys

import mlx.core as mx
import numpy as np

d = mx.load(sys.argv[1])
q, k, v, mask = d["q"], d["k"], d["v"], d["mask"]
valid = np.array(d["attention_mask"][0]).astype(bool)
scale = np.float32(1 / 16)
H, Hk = q.shape[1], k.shape[1]
LOG2E = np.float32(1.44269504089)

Q = np.array(q.astype(mx.float32))[0]
K = np.repeat(np.array(k.astype(mx.float32))[0], H // Hk, axis=0)
V = np.repeat(np.array(v.astype(mx.float32))[0], H // Hk, axis=0)
M = np.array(mask.astype(mx.float32))[0, 0]  # (T, T), exact bf16 values
pad = np.where(~valid)[0]
S = np.einsum("hqd,hkd->hqk", Q[:, pad].astype(np.float64), K.astype(np.float64)).astype(np.float32)
Mp = M[pad][None]


def softmax_rows(x, base_e=True):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x.astype(np.float64)) if base_e else np.exp2(x.astype(np.float64))
    return e / e.sum(-1, keepdims=True)


x_unf = (S * scale).astype(np.float32) + Mp.astype(np.float32)
x_nax = (S * (scale * LOG2E)).astype(np.float32) + (LOG2E * Mp).astype(np.float32)
P_unf = softmax_rows(x_unf, True)
P_nax = softmax_rows(x_nax, False)
O_unf = np.einsum("hqk,hkd->hqd", P_unf, V)
O_nax = np.einsum("hqk,hkd->hqd", P_nax, V)

fused = np.array(d["o_prebuilt_fused"].astype(mx.float32))[0][:, pad]
chunk = np.array(d["o_chunked512"].astype(mx.float32))[0][:, pad]


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


print(f"padded rows: {len(pad)} x {H} heads")
print(f"  emulated unfused vs chunked512 (MLX):  cos {cos(O_unf, chunk):.6f}  maxAbs {np.abs(O_unf - chunk).max():.4f}")
print(f"  emulated NAX     vs fused d-split:     cos {cos(O_nax, fused):.6f}  maxAbs {np.abs(O_nax - fused).max():.4f}")
print(f"  emulated unfused vs emulated NAX:      cos {cos(O_unf, O_nax):.6f}")
print(f"  MLX chunked512   vs MLX fused:         cos {cos(chunk, fused):.6f}")

# How many keys survive (carry the row max) under each rounding, per (head,row)
surv_unf = (x_unf == x_unf.max(-1, keepdims=True)).sum(-1)
surv_nax = (x_nax == x_nax.max(-1, keepdims=True)).sum(-1)
diff = surv_unf != surv_nax
print(f"  (head,row) pairs whose surviving-key set size differs: {diff.sum()} / {diff.size}")
i = np.argwhere(diff)[0] if diff.any() else None
if i is not None:
    h, r = i
    print(f"  e.g. head {h}, row {pad[r]}: unfused keeps {surv_unf[h, r]} keys, NAX keeps {surv_nax[h, r]}; "
          f"max scaled score {float((S[h, r] * scale).max()):.2f}")

# --- The unfused path does its arithmetic in bf16: q*scale, the matmul output and +mask are all
# bf16, where the spacing at 1e9 is 2^22, so every score is lost and a fully masked row becomes a
# plain average over the keys that share the row's largest mask value.
eq = (Mp == Mp.max(-1, keepdims=True)).astype(np.float64)
O_uniform = np.einsum("hqk,hkd->hqd", np.broadcast_to(eq / eq.sum(-1, keepdims=True), P_unf.shape), V)
print(f"  uniform-over-mask-max vs chunked512:   cos {cos(O_uniform, chunk):.6f}  maxAbs {np.abs(O_uniform - chunk).max():.4f}")
print(f"  uniform-over-mask-max vs fused d-split: cos {cos(O_uniform, fused):.6f}")

# --- Conditioning of the NAX rounding: recompute scores with fp32 accumulation instead of fp64.
S32 = np.einsum("hqd,hkd->hqk", Q[:, pad], K).astype(np.float32)
x_nax32 = (S32 * (scale * LOG2E)).astype(np.float32) + (LOG2E * Mp).astype(np.float32)
O_nax32 = np.einsum("hqk,hkd->hqd", softmax_rows(x_nax32, False), V)
print(f"  NAX emulation, fp64- vs fp32-accumulated scores: cos {cos(O_nax, O_nax32):.6f}  "
      f"(keys flipped across the grid: {(x_nax != x_nax32).sum()})")

# --- Exact math: a constant offset cancels in softmax, so the true answer for these rows is the
# softmax of the real scores over the -1e9 class of keys.
x64 = S.astype(np.float64) * float(scale) + Mp.astype(np.float64)
O_exact = np.einsum("hqk,hkd->hqd", softmax_rows(x64, True), V)
print(f"  exact fp64 vs chunked512 (uniform):     cos {cos(O_exact, chunk):.6f}")
print(f"  exact fp64 vs fused d-split:            cos {cos(O_exact, fused):.6f}")
