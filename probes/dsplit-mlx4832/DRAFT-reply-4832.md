@bazyli Thanks. Those two asks settled it, and I need to correct my earlier report. The difference isn't caused by JIT or by mlx-swift, and I don't think #4903 is wrong. It's confined to **fully masked query rows** (left padding under an additive `-1e9` mask), and the prebuilt metallib shows it too.

**Setup:** M5 Max, macOS 27.3, pip `mlx==0.32.4` / `mlx-metal==0.32.4` (prebuilt metallib, no JIT), `mlx-lm==0.32.2`. Same Gemma 3 12B 4-bit encoder tap: T = 1024, 21 real tokens left-padded with 1003 pad tokens, and the bf16 mask `triu(-1e9) + (1 - attn) * -1e9` applied on every layer.

| 49 states vs our goldens | mean cosine | valid rows | padded rows |
|---|---|---|---|
| fused (takes d-split) | 0.929 | 0.999998 | 0.917 |
| same run, queries split into 512-row chunks (no d-split) | 1.000000 (bit-identical) | 1.000000 | 1.000000 |

The prebuilt kernel gives about the same number we got through JIT + #4903 (0.925). Real tokens are fine.

**Per layer:** I ran both paths on identical q/k/v/mask. Valid rows agree at every layer (cos ≥ 0.99977). Padded rows agree exactly in most layers and split in a few: layer 4 0.933, layer 28 0.949, layer 2 0.973.

**Why:** in a fully masked row every logit is about `-1e9 + s`.
- The unfused path computes `q*scale`, the matmul and `+ mask` in bf16. Near 1e9 the bf16 spacing is 2^22, so `s` is lost and the row becomes a plain average over the keys tied at the row's largest mask value. On layer 4 that uniform average matches the chunked output at cos 0.999997.
- The steel/NAX kernels compute `S*scale*log2e + log2e*mask` in fp32. Near `log2e * bf16(-1e9) = -1.44e9` the fp32 spacing is 128, so scaled scores below 64/log2e ≈ 44.4 round away (the row stays uniform, as with your synthetic inputs at scale 1/16 and our layer 0). Above that, the keys that round up take the whole row.
- Neither matches the exact fp64 softmax of those rows (cos 0.56 and 0.61 on layer 4). Under a finite mask, a fully masked row has no well-defined output.

**It isn't specific to d-split.** Synthetic, no weights (script below): with head dim 128 the regular fused kernel leaves the uniform answer the same way once scores pass ~44, at qL 1024 and at qL 256. With head dim 256 only the qL ≥ 1024 route does, because shorter queries go unfused:

```
   D  Hq/Hk score max | fused: valid vs unfused  pad vs uniform | qL=256 chunks: pad vs uniform
 128  16/8       23.0 |                0.999990        0.999999 |                      0.999999
 128  16/8       49.6 |                0.999978        0.174420 |                      0.174420
 256  16/8       21.0 |                0.999990        0.999999 |                      0.999997
 256  16/8       49.9 |                0.999987        0.172957 |                      0.999997
```

Until v0.32.4, head dim 256 with qL ≥ 1024 and an array mask went unfused, so our goldens encode that convention.

For our pipeline this is harmless: padded positions are zeroed and replaced by learned registers before they're used, and the real tokens match. So please treat my earlier "wrong attention" note as withdrawn. #4903 looks correct to me. Whether fully masked rows should give the same output across paths is a separate question. I'm happy to open an issue for it if that's useful. I can also share the saved layer-0 and layer-4 q/k/v/mask tensors if you still want them.

<details><summary>synthetic repro</summary>

```python
import mlx.core as mx
import numpy as np

mx.random.seed(0)
T, n_valid = 1024, 21
attn = mx.array([[0] * (T - n_valid) + [1] * n_valid])
valid = np.array(attn[0]).astype(bool)
causal = mx.triu(mx.full((T, T), -1e9, dtype=mx.bfloat16), k=1)
mask = causal[None, None] + (1 - attn[:, None, None, :].astype(mx.bfloat16)) * -1e9
Mp = np.array(mask.astype(mx.float32))[0, 0]

def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

for D in (128, 256):
    scale = D ** -0.5
    for amp in (2.0, 3.0):
        q = (mx.random.normal((1, 16, T, D)) * amp).astype(mx.bfloat16)
        k = (mx.random.normal((1, 8, T, D)) * amp).astype(mx.bfloat16)
        v = mx.random.normal((1, 8, T, D)).astype(mx.bfloat16)
        fused = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        V = np.repeat(np.array(v.astype(mx.float32))[0], 2, axis=0)
        eq = (Mp == Mp.max(-1, keepdims=True)).astype(np.float64)
        uniform = np.einsum("qk,hkd->hqd", eq / eq.sum(-1, keepdims=True), V)
        F = np.array(fused.astype(mx.float32))[0]
        print(D, amp, "padded rows, fused vs uniform:", round(cos(F[:, ~valid], uniform[:, ~valid]), 6))
```
</details>
