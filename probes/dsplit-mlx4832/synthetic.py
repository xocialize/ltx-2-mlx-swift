"""Minimal synthetic repro: fully masked rows under a finite additive bf16 mask.

Usage: python -I synthetic.py
LTX-style mask: causal -1e9 + left padding -1e9 (bf16), 21 valid tokens at the end of T=1024,
so 1003 query rows are fully masked. For a range of score magnitudes, compare the fused SDPA
(whatever kernel MLX picks) against the uniform average over the row's top mask class, which is
what MLX's unfused path returns for such rows, on valid and padded rows separately.
"""
import mlx.core as mx
import numpy as np

mx.random.seed(0)
T, n_valid, B = 1024, 21, 1
attn = mx.array([[0] * (T - n_valid) + [1] * n_valid])
valid = np.array(attn[0]).astype(bool)
causal = mx.triu(mx.full((T, T), -1e9, dtype=mx.bfloat16), k=1)
mask = causal[None, None] + (1 - attn[:, None, None, :].astype(mx.bfloat16)) * -1e9
Mp = np.array(mask.astype(mx.float32))[0, 0]


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def chunked(q, k, v, scale):
    return mx.concatenate([mx.fast.scaled_dot_product_attention(q[:, :, s:s + 256], k, v, scale=scale,
                                                                mask=mask[:, :, s:s + 256])
                           for s in range(0, T, 256)], axis=2)


print(f"mlx {mx.__version__}  {mx.device_info()['device_name']}")
print(f"{'D':>4} {'Hq/Hk':>6} {'score max':>9} | {'fused: valid vs unfused':>23} {'pad vs uniform':>15} | {'qL=256 chunks: pad vs uniform':>29}")
for D, Hq, Hk in ((128, 16, 8), (256, 16, 8)):
    scale = D ** -0.5
    for amp in (1.0, 2.0, 3.0, 4.0):
        q = (mx.random.normal((B, Hq, T, D)) * amp).astype(mx.bfloat16)
        k = (mx.random.normal((B, Hk, T, D)) * amp).astype(mx.bfloat16)
        v = mx.random.normal((B, Hk, T, D)).astype(mx.bfloat16)
        fused = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        ch = chunked(q, k, v, scale)
        kr = mx.repeat(k, Hq // Hk, axis=1)
        vr = mx.repeat(v, Hq // Hk, axis=1)
        unf = mx.softmax((q * scale) @ kr.transpose(0, 1, 3, 2) + mask, axis=-1, precise=True) @ vr
        smax = mx.max(mx.abs((q.astype(mx.float32) * scale) @ kr.astype(mx.float32).transpose(0, 1, 3, 2)))
        mx.eval(fused, ch, unf, smax)
        V = np.array(vr.astype(mx.float32))[0]
        eq = (Mp == Mp.max(-1, keepdims=True)).astype(np.float64)
        uni = np.einsum("qk,hkd->hqd", eq / eq.sum(-1, keepdims=True), V)
        F = np.array(fused.astype(mx.float32))[0]
        C = np.array(ch.astype(mx.float32))[0]
        U = np.array(unf.astype(mx.float32))[0]
        print(f"{D:4d} {Hq:>3}/{Hk:<2} {float(smax):9.1f} | {cos(F[:, valid], U[:, valid]):23.6f} "
              f"{cos(F[:, ~valid], uni[:, ~valid]):15.6f} | {cos(C[:, ~valid], uni[:, ~valid]):29.6f}")
