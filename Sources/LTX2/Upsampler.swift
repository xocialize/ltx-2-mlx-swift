// Upsampler.swift — LTX-2.3 neural latent upsampler (all three shipped variants).
//
// 1:1 functional port of ltx_core_mlx/model/upsampler/model.py (LatentUpsampler).
// The shell is shared — Conv3d init → 4 ResBlocks → VARIANT → 4 ResBlocks → Conv3d,
// GroupNorm(32, pytorch_compatible), fp32 — and only the middle differs:
//   spatial_x2   per-frame Conv2d(mid→4·mid) + PixelShuffle2D(2)            → (B,C,F,2H,2W)
//   spatial_x1_5 per-frame Conv2d(mid→9·mid) + PixelShuffle2D(3) + binomial
//                BlurDownsample(stride 2) — the rational 3/2 resampler       → (B,C,F,1.5H,1.5W)
//   temporal_x2  Conv3d(mid→2·mid) + PixelShuffle3D(temporal 2), then DROP
//                THE FIRST FRAME (frame 0 encodes one pixel frame in the VAE) → (B,C,2F−1,H,W)
//
// Variant selection is CONFIG-DRIVEN (GAP-ANALYSIS #5/#6): `load(path:)` reads the sidecar
// `<stem>_config.json` shipped next to each checkpoint (spatial_upsample / temporal_upsample /
// spatial_scale / rational_resampler — the oracle's `from_config` fields); when no sidecar
// exists, the variant is inferred from the weight keys, which are disjoint by construction
// (rational → `upsampler.blur_down.kernel`; temporal → 5-D `upsampler.0.weight`; x2 → 4-D).
// The hardcoded-x2 era is GAP-ANALYSIS's "oracle can load 3, we have 1" row — closed.

import Foundation
import MLX
import MLXNN

public struct Upsampler {
    public enum Variant: String, Sendable {
        case spatialX2 = "spatial_x2"
        case spatialX1_5 = "spatial_x1_5"
        case temporalX2 = "temporal_x2"
    }

    let w: [String: MLXArray]
    public let variant: Variant

    public init(weights: [String: MLXArray], variant: Variant? = nil) {
        // Keys are stem-prefixed (e.g. "spatial_upscaler_x2_v1_1.initial_conv.weight");
        // strip a leading "*upscaler*." segment so lookups are bare. fp32.
        var m: [String: MLXArray] = [:]
        for (k, v) in weights {
            var key = k
            if let dot = k.firstIndex(of: "."), k[..<dot].contains("upscaler") {
                key = String(k[k.index(after: dot)...])
            }
            m[key] = v.asType(.float32)
        }
        self.w = m
        // Weight-key inference fallback (keys are disjoint across variants by construction).
        if let variant {
            self.variant = variant
        } else if m["upsampler.blur_down.kernel"] != nil || m["upsampler.conv.weight"] != nil {
            self.variant = .spatialX1_5
        } else if let u0 = m["upsampler.0.weight"], u0.ndim == 5 {
            self.variant = .temporalX2
        } else {
            self.variant = .spatialX2
        }
    }

    /// Load a checkpoint, resolving the variant from its sidecar `<stem>_config.json` when
    /// present (the oracle's `from_config` contract), else from the weight keys.
    public static func load(path: URL) throws -> Upsampler {
        var variant: Variant?
        let stem = path.deletingPathExtension().lastPathComponent
        let sidecar = path.deletingLastPathComponent().appendingPathComponent(stem + "_config.json")
        if let data = try? Data(contentsOf: sidecar),
           let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let cfg = root["config"] as? [String: Any] {
            let spatial = cfg["spatial_upsample"] as? Bool ?? true
            let temporal = cfg["temporal_upsample"] as? Bool ?? false
            let scale = (cfg["spatial_scale"] as? Double) ?? 2.0
            if temporal { variant = .temporalX2 }
            else if spatial && abs(scale - 1.5) < 0.01 { variant = .spatialX1_5 }
            else if spatial { variant = .spatialX2 }
        }
        return Upsampler(weights: try MLX.loadArrays(url: path), variant: variant)
    }

    /// latent (B, C, F, H, W) → upsampled latent (channels-first); output geometry per `variant`.
    public func callAsFunction(_ latent: MLXArray) -> MLXArray {
        var x = latent.asType(.float32).transposed(0, 2, 3, 4, 1)  // BFHWC

        x = silu(groupNorm(conv3dB(x, "initial_conv"), "initial_norm"))
        for i in 0 ..< 4 { x = resBlock(x, "res_blocks.\(i)") }

        x = applyUpsampler(x)
        if variant == .temporalX2 {
            // Remove the first frame after temporal upsample — frame 0 encodes ONE pixel
            // frame in the causal VAE, so its doubled twin is synthetic (oracle model.py:348).
            x = x[0..., 1..., 0..., 0..., 0...]
        }

        for i in 0 ..< 4 { x = resBlock(x, "post_upsample_res_blocks.\(i)") }
        x = conv3dB(x, "final_conv")
        return x.transposed(0, 4, 1, 2, 3)  // BCFHW
    }

    /// The variant-specific middle (oracle `_apply_upsampler`). Input/output BDHWC.
    private func applyUpsampler(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), D = x.dim(1), H = x.dim(2), W = x.dim(3), C = x.dim(4)
        switch variant {
        case .spatialX2:
            var f = x.reshaped(B * D, H, W, C)
            f = conv2dB(f, "upsampler.0")
            f = pixelShuffle2d(f, factor: 2)
            return f.reshaped(B, D, H * 2, W * 2, f.dim(-1))
        case .spatialX1_5:
            // Rational 3/2: Conv2d(mid→9·mid) → PixelShuffle2D(3) → BlurDownsample(stride 2).
            var f = x.reshaped(B * D, H, W, C)
            f = conv2dB(f, "upsampler.conv")
            f = pixelShuffle2d(f, factor: 3)
            f = blurDownsample(f, stride: 2)
            let H2 = f.dim(1), W2 = f.dim(2)
            return f.reshaped(B, D, H2, W2, f.dim(-1))
        case .temporalX2:
            let y = conv3dB(x, "upsampler.0")
            return pixelShuffle3d(y, spatialFactor: 1, temporalFactor: 2)
        }
    }

    /// Depthwise blur-then-downsample (BHWC). Kernel from the checkpoint
    /// (`upsampler.blur_down.kernel`, OHWI (1,K,K,1) binomial); channels processed
    /// independently via a (B·C, H, W, 1) reshape — mirrors the oracle exactly.
    private func blurDownsample(_ x: MLXArray, stride: Int) -> MLXArray {
        guard stride > 1 else { return x }
        let kernel = w["upsampler.blur_down.kernel"] ?? Self.binomialKernel(size: 5)
        let B = x.dim(0), H = x.dim(1), W = x.dim(2), C = x.dim(3)
        let K = kernel.dim(1), pad = K / 2
        var y = x.transposed(0, 3, 1, 2).reshaped(B * C, H, W, 1)
        y = MLX.padded(y, widths: [IntOrPair(0), IntOrPair(pad), IntOrPair(pad), IntOrPair(0)])
        y = MLX.conv2d(y, kernel, stride: IntOrPair(stride))
        let H2 = y.dim(1), W2 = y.dim(2)
        return y.reshaped(B, C, H2, W2).transposed(0, 2, 3, 1)
    }

    /// Deterministic 5-tap binomial kernel, only used when a checkpoint omits the buffer.
    private static func binomialKernel(size: Int) -> MLXArray {
        func comb(_ n: Int, _ r: Int) -> Float {
            var out = 1.0
            for i in 0 ..< r { out *= Double(n - i) / Double(i + 1) }
            return Float(out)
        }
        let k = MLXArray((0 ..< size).map { comb(size - 1, $0) })
        var k2 = k.expandedDimensions(axis: 1) * k.expandedDimensions(axis: 0)
        k2 = k2 / MLX.sum(k2)
        return k2.reshaped(1, size, size, 1)
    }

    /// 3D pixel shuffle (BDHWC): (B,D,H,W,C·tf·sf²) → (B,D·tf,H·sf,W·sf,C).
    /// ⚠️ Split/transpose order mirrors the oracle VERBATIM (channel-split order is
    /// load-bearing — the parent CLAUDE.md checkerboard lesson): reshape to
    /// (B,D,H,W,C,tf,sf,sf) then transpose(0,1,5,2,6,3,7,4).
    private func pixelShuffle3d(_ x: MLXArray, spatialFactor sf: Int, temporalFactor tf: Int) -> MLXArray {
        let B = x.dim(0), D = x.dim(1), H = x.dim(2), W = x.dim(3), Ct = x.dim(4)
        let C = Ct / (sf * sf * tf)
        var y = x.reshaped(B, D, H, W, C, tf, sf, sf)
        y = y.transposed(0, 1, 5, 2, 6, 3, 7, 4)
        return y.reshaped(B, D * tf, H * sf, W * sf, C)
    }

    // MARK: - blocks

    /// ResBlock: conv1 → norm1 → silu → conv2 → norm2 → silu(x + residual). Conv3d.
    private func resBlock(_ x0: MLXArray, _ p: String) -> MLXArray {
        var x = conv3dB(x0, "\(p).conv1")
        x = silu(groupNorm(x, "\(p).norm1"))
        x = conv3dB(x, "\(p).conv2")
        x = groupNorm(x, "\(p).norm2")
        return silu(x + x0)
    }

    private func conv3dB(_ x: MLXArray, _ p: String) -> MLXArray {
        MLX.conv3d(x, w["\(p).weight"]!, stride: 1, padding: 1) + w["\(p).bias"]!
    }

    private func conv2dB(_ x: MLXArray, _ p: String) -> MLXArray {
        MLX.conv2d(x, w["\(p).weight"]!, stride: 1, padding: 1) + w["\(p).bias"]!
    }

    /// GroupNorm (pytorch-compatible): split C into G contiguous groups, normalize each
    /// group over (spatial × C/G), then per-channel affine. x is channels-last (B, …, C).
    private func groupNorm(_ x: MLXArray, _ p: String, groups G: Int = 32, eps: Float = 1e-5) -> MLXArray {
        let shape = x.shape
        let B = shape[0], C = shape[shape.count - 1]
        let N = x.size / (B * C)              // product of spatial dims
        let cg = C / G
        // (B, N, C) → (B, N, G, cg); normalize over (N, cg) per (B, G)
        let g = x.reshaped(B, N, G, cg)
        let mean = MLX.mean(g, axes: [1, 3], keepDims: true)
        let v = MLX.mean((g - mean) * (g - mean), axes: [1, 3], keepDims: true)
        let normed = ((g - mean) * MLX.rsqrt(v + eps)).reshaped(B, N, C)
        let weight = w["\(p).weight"]!, bias = w["\(p).bias"]!
        let out = normed * weight + bias
        return out.reshaped(shape)
    }

    /// 2D pixel shuffle (BHWC): (B,H,W,C·f²) → (B,H·f,W·f,C). Split order (c, p1, p2).
    private func pixelShuffle2d(_ x: MLXArray, factor f: Int) -> MLXArray {
        let B = x.dim(0), H = x.dim(1), W = x.dim(2), Ct = x.dim(3)
        let C = Ct / (f * f)
        var y = x.reshaped(B, H, W, C, f, f)
        y = y.transposed(0, 1, 4, 2, 5, 3)   // (B,H,p1,W,p2,C)
        return y.reshaped(B, H * f, W * f, C)
    }
}
