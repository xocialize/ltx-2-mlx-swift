// VideoVAE.swift — LTX-2.3 video VAE decoder (latent → pixels).
//
// 1:1 functional port of ltx_core_mlx/model/video_vae/{video_vae,convolution,
// resnet,sampling,normalization,ops}.py (decode path). 128-ch latent, 8× temporal
// / 32× spatial, pixel-shuffle upsampling, PixelNorm (parameterless), Conv3d.
// Decoder is NON-causal (LTX-2.3: symmetric replicate temporal pad, zeros spatial).
// See docs/ltx-vae-vs-wan-vae.md. Encoder + streaming tiling are follow-ups.

import Foundation
import MLX
import MLXNN

public struct VideoVAEDecoder {
    let w: [String: MLXArray]

    // up_blocks: ResStage block counts at even indices; DepthToSpace at odd.
    static let resStageBlocks: [Int: Int] = [0: 2, 2: 2, 4: 4, 6: 6, 8: 4]
    // (spatial_factor, temporal_factor) for DepthToSpace at odd indices 1,3,5,7.
    static let upsampleConfig: [(Int, Int)] = [(2, 2), (2, 2), (1, 2), (2, 1)]

    public init(weights: [String: MLXArray]) {
        var m: [String: MLXArray] = [:]
        for (k, v) in weights {
            let key = k.hasPrefix("vae_decoder.") ? String(k.dropFirst("vae_decoder.".count)) : k
            m[key] = v.asType(.float32)
        }
        self.w = m
    }

    public static func load(path: URL) throws -> VideoVAEDecoder {
        VideoVAEDecoder(weights: try MLX.loadArrays(url: path))
    }

    /// latent: (B, C=128, F, H, W) PyTorch layout → pixels (B, 3, F*8-7, H*32, W*32) in [-1,1].
    public func decode(_ latent: MLXArray) -> MLXArray {
        var x = latent.asType(.float32).transposed(0, 2, 3, 4, 1)  // BFHWC
        x = denormalize(x)
        x = conv3dBlock(x, "conv_in")

        var upIdx = 0
        for i in 0 ..< 9 {
            if i % 2 == 0 {
                x = resStage(x, "up_blocks.\(i)", VideoVAEDecoder.resStageBlocks[i]!)
            } else {
                x = conv3dBlock(x, "up_blocks.\(i).conv")
                let (sf, tf) = VideoVAEDecoder.upsampleConfig[upIdx]
                x = pixelShuffle3d(x, spatialFactor: sf, temporalFactor: tf)
                if tf > 1 { x = x[0..., 1...] }  // drop first frame after temporal upsample
                upIdx += 1
                eval(x)  // materialize between upsample stages (memory)
            }
        }

        x = conv3dBlock(silu(pixelNorm(x)), "conv_out")
        x = unpatchifySpatial(x, patchSize: 4)
        return x.transposed(0, 4, 1, 2, 3)  // BCFHW
    }

    // MARK: - blocks

    private func resStage(_ x0: MLXArray, _ prefix: String, _ n: Int) -> MLXArray {
        var x = x0
        for j in 0 ..< n {
            let p = "\(prefix).res_blocks.\(j)"
            let residual = x
            x = conv3dBlock(silu(pixelNorm(x)), "\(p).conv1")
            x = conv3dBlock(silu(pixelNorm(x)), "\(p).conv2")
            x = x + residual
        }
        return x
    }

    /// Non-causal Conv3dBlock: symmetric replicate temporal pad + zeros spatial pad, then conv.
    private func conv3dBlock(_ x0: MLXArray, _ prefix: String) -> MLXArray {
        let tk = 3
        let tpad = (tk - 1) / 2  // 1
        var x = x0
        if tpad > 0 {
            let first = MLX.repeated(x[0..., 0 ..< 1], count: tpad, axis: 1)
            let last = MLX.repeated(x[0..., (x.dim(1) - 1)...], count: tpad, axis: 1)
            x = MLX.concatenated([first, x, last], axis: 1)
        }
        // zeros spatial pad (H,W) = 1
        x = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair(1), IntOrPair(1), IntOrPair(0)])
        var y = conv3d(x, w["\(prefix).conv.weight"]!, stride: 1, padding: 0)
        y = y + w["\(prefix).conv.bias"]!
        return y
    }

    private func denormalize(_ x: MLXArray) -> MLXArray {
        let mean = w["per_channel_statistics.mean"]!.reshaped(1, 1, 1, 1, -1)
        let std = w["per_channel_statistics.std"]!.reshaped(1, 1, 1, 1, -1)
        return x * std + mean
    }

    private func pixelNorm(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
        let v = MLX.mean(x * x, axis: -1, keepDims: true)
        return x * MLX.rsqrt(v + eps)
    }

    // MARK: - pixel-shuffle helpers

    /// depth-to-space: (B,D,H,W, C·tf·sf²) → (B, D·tf, H·sf, W·sf, C). Split order (c, t, h, w).
    private func pixelShuffle3d(_ x: MLXArray, spatialFactor sf: Int, temporalFactor tf: Int) -> MLXArray {
        let B = x.dim(0), D = x.dim(1), H = x.dim(2), W = x.dim(3), Ct = x.dim(4)
        let C = Ct / (sf * sf * tf)
        var y = x.reshaped(B, D, H, W, C, tf, sf, sf)
        y = y.transposed(0, 1, 5, 2, 6, 3, 7, 4)
        return y.reshaped(B, D * tf, H * sf, W * sf, C)
    }

    /// final spatial unpatchify: (B,F,H,W, C·ps²) → (B,F, H·ps, W·ps, C). Split order (c, r=W, q=H).
    private func unpatchifySpatial(_ x: MLXArray, patchSize ps: Int) -> MLXArray {
        let B = x.dim(0), F = x.dim(1), H = x.dim(2), W = x.dim(3), Ct = x.dim(4)
        let C = Ct / (ps * ps)
        var y = x.reshaped(B, F, H, W, C, ps, ps)   // (B,F,H,W,C,r_W,q_H)
        y = y.transposed(0, 1, 2, 6, 3, 5, 4)        // (B,F,H,q_H,W,r_W,C)
        return y.reshaped(B, F, H * ps, W * ps, C)
    }
}
