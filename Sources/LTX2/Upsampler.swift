// Upsampler.swift — LTX-2.3 neural latent upsampler (spatial_x2 variant).
//
// 1:1 functional port of ltx_core_mlx/model/upsampler/model.py (LatentUpsampler,
// spatial_x2). The two-stage distilled pipeline uses this to lift the half-res
// stage-1 latent 2× spatially (in un-normalized latent space) before the stage-2
// refine. Conv3d init → 4 ResBlocks → per-frame Conv2d + PixelShuffle2D(2) →
// 4 ResBlocks → Conv3d. GroupNorm(32, pytorch_compatible) — first GroupNorm in
// the port (the VAEs use PixelNorm). fp32.

import Foundation
import MLX
import MLXNN

public struct Upsampler {
    let w: [String: MLXArray]

    public init(weights: [String: MLXArray]) {
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
    }

    public static func load(path: URL) throws -> Upsampler {
        Upsampler(weights: try MLX.loadArrays(url: path))
    }

    /// latent (B, C, F, H, W) → 2×-spatial latent (B, C, F, 2H, 2W) (channels-first).
    public func callAsFunction(_ latent: MLXArray) -> MLXArray {
        var x = latent.asType(.float32).transposed(0, 2, 3, 4, 1)  // BFHWC

        x = silu(groupNorm(conv3dB(x, "initial_conv"), "initial_norm"))
        for i in 0 ..< 4 { x = resBlock(x, "res_blocks.\(i)") }

        // spatial_x2 upsampler: per-frame Conv2d (mid→4·mid) + PixelShuffle2D(2)
        let B = x.dim(0), D = x.dim(1), H = x.dim(2), W = x.dim(3), C = x.dim(4)
        var f = x.reshaped(B * D, H, W, C)
        f = conv2dB(f, "upsampler.0")
        f = pixelShuffle2d(f, factor: 2)
        x = f.reshaped(B, D, H * 2, W * 2, f.dim(-1))

        for i in 0 ..< 4 { x = resBlock(x, "post_upsample_res_blocks.\(i)") }
        x = conv3dB(x, "final_conv")
        return x.transposed(0, 4, 1, 2, 3)  // BCFHW
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
