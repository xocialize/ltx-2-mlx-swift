// ICAttentionMask.swift — pixel-masked IC conditioning: the (B,N,N) machinery `ReferenceConditioning`
// deferred. Three upstream pieces, ported 1:1 from Lightricks/LTX-2 (ltx-core / ltx-pipelines @ 9ec55f9):
//
//   1. `iclora_utils.downsample_mask_video_to_latent` — a pixel-space mask video (F_pix, H, W) in
//      [0,1] → one weight per reference token: `area` interpolation per frame, then CAUSAL temporal
//      downsampling (frame 0 alone — the causal VAE's first latent frame covers one pixel frame —
//      then means over groups of t = (F_pix−1)/(F_lat−1)), flattened in (f, h, w) token order.
//   2. `mask_utils.build_attention_mask` — the multiplicative block mask over target ++ reference
//      groups: target↔target 1, target↔group_g = cross_g (both directions), group_g↔group_g 1,
//      group↔other group 0.
//   3. The multiplicative→ADDITIVE conversion the transformer consumes. Two conventions exist in
//      the wild and they DISAGREE on fractional weights (they agree exactly on 0 and 1):
//        .ltxCore    `transformer_args._prepare_self_attention_mask`: log(m), m ≤ 0 → finfo.min.
//                    m = 0.75 costs only log 0.75 = −0.29.
//        .diffusers  `transformer_ltx2.py`: (1 − m) · −10000. m = 0.75 → −2500, i.e. masked.
//      The ltx-community outpaint Space (the published reference usage for the In-Outpainting
//      IC-LoRA) runs diffusers. Fractional weights only arise at latent cells that straddle the
//      mask edge, so the choice moves exactly those tokens — an A/B axis, not a correctness one.
//
// ⚠️ The MLX oracle (`ltx-2-mlx`, `samplers.py` → `model.py` → `mx.fast.scaled_dot_product_attention`)
// passes the MULTIPLICATIVE [0,1] mask straight in as an additive bias — a +1 nudge, not a mask.
// It is not a parity reference for this path; `dump_ic_mask_goldens.py` converts upstream's way
// before handing the oracle DiT an additive bias.

import MLX

public enum ICAttentionMask {

    /// Multiplicative → additive conversion (see file header).
    public enum Convention: String, Sendable {
        case ltxCore
        case diffusers
    }

    /// Pixel mask (F_pix, H_pix, W_pix) in [0,1] → (Nr,) per-reference-token weights in (f, h, w)
    /// order, for a reference encoded to (latentFrames, latentHeight, latentWidth).
    ///
    /// Spatial `area` interpolation is a block mean when the pixel/latent ratio is integral, which
    /// every pipeline canvas is (pixels = 32 × latent cells). A non-integral ratio would need
    /// PyTorch's adaptive-pool window rule and is refused rather than approximated.
    public static func latentCrossMask(pixelMask: MLXArray, latentFrames fL: Int,
                                       latentHeight hL: Int, latentWidth wL: Int) -> MLXArray {
        precondition(pixelMask.ndim == 3, "pixel mask must be (F, H, W), got \(pixelMask.shape)")
        let fP = pixelMask.dim(0), hP = pixelMask.dim(1), wP = pixelMask.dim(2)
        precondition(hP % hL == 0 && wP % wL == 0,
                     "area downsample needs an integral ratio: \(hP)x\(wP) → \(hL)x\(wL)")
        let bh = hP / hL, bw = wP / wL
        let spatial = pixelMask.asType(.float32)
            .reshaped(fP, hL, bh, wL, bw).mean(axes: [2, 4])               // (F_pix, hL, wL)
        let first = spatial[0 ..< 1]
        let latent: MLXArray
        if fP > 1 && fL > 1 {
            precondition((fP - 1) % (fL - 1) == 0,
                         "pixel frames (\(fP)) not compatible with latent frames (\(fL))")
            let t = (fP - 1) / (fL - 1)
            let rest = spatial[1...].reshaped(fL - 1, t, hL, wL).mean(axis: 1)
            latent = concatenated([first, rest], axis: 0)
        } else {
            latent = first
        }
        return latent.reshaped(-1)
    }

    /// The multiplicative block mask (N, N), N = numNoisy + Σ groups, for reference groups appended
    /// in order after the target tokens (no keyframe tokens between — the IC layout `ICVideoState`
    /// builds). Each `crossMasks[g]` is (n_g,) in [0,1]; pass ones for an unmasked group.
    public static func multiplicativeMask(numNoisy: Int, crossMasks: [MLXArray]) -> MLXArray {
        let sizes = crossMasks.map { $0.dim(0) }
        var noisyRow = [MLXArray.ones([numNoisy, numNoisy], dtype: .float32)]
        for c in crossMasks {
            noisyRow.append(broadcast(c.asType(.float32).reshaped(1, -1), to: [numNoisy, c.dim(0)]))
        }
        var rows = [concatenated(noisyRow, axis: 1)]
        for (g, c) in crossMasks.enumerated() {
            let n = sizes[g]
            var row = [broadcast(c.asType(.float32).reshaped(-1, 1), to: [n, numNoisy])]
            for (h, m) in sizes.enumerated() {
                row.append(h == g ? MLXArray.ones([n, m], dtype: .float32)
                                  : MLXArray.zeros([n, m], dtype: .float32))
            }
            rows.append(concatenated(row, axis: 1))
        }
        return concatenated(rows, axis: 0)
    }

    /// Multiplicative mask → additive bias (1, 1, N, N) in `dtype`, ready for
    /// `DiT.withVideoSelfAttentionBias`.
    public static func additiveBias(_ mult: MLXArray, convention: Convention, dtype: DType) -> MLXArray {
        let m = mult.asType(.float32)
        let bias: MLXArray
        switch convention {
        case .ltxCore:
            // finfo(dtype).min for the masked entries, as upstream. bf16's largest finite magnitude
            // is exactly representable in fp32; fp32's own is used for the fp32 gate.
            let floor: Float = dtype == .bfloat16 ? -3.3895313892515355e38
                : (dtype == .float16 ? -65504 : -Float.greatestFiniteMagnitude)
            let positive = MLX.greater(m, MLXArray(Float(0)))
            let safe = MLX.maximum(m, MLXArray(Float.leastNormalMagnitude))
            bias = MLX.which(positive, MLX.log(safe), MLXArray(floor))
        case .diffusers:
            bias = (1.0 - m) * Float(-10000)
        }
        return bias.reshaped(1, 1, m.dim(0), m.dim(1)).asType(dtype)
    }

    /// Convenience: the whole chain for ONE masked reference group (the outpaint/inpaint case).
    /// `strength` is upstream's `conditioning_attention_strength`, multiplied into the weights.
    public static func selfAttentionBias(numNoisy: Int, referenceCrossMask: MLXArray,
                                         strength: Float = 1.0, convention: Convention,
                                         dtype: DType) -> MLXArray {
        let cross = strength == 1 ? referenceCrossMask : referenceCrossMask * strength
        return additiveBias(multiplicativeMask(numNoisy: numNoisy, crossMasks: [cross]),
                            convention: convention, dtype: dtype)
    }
}
