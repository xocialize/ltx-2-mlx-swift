// AudioVAE.swift — LTX-2.3 audio VAE decoder (audio latent → mel spectrogram).
//
// 1:1 functional port of ltx_core_mlx/model/audio_vae/audio_vae.py (decode path,
// distilled config = NO attention). Conv2d-based: the (B,8,T,16) latent is a 2D
// spatial tensor (NHWC (B,T,16,8)); the net upsamples frequency 16→64 keeping
// time T, → mel (B,2,T',64). Causal height (time) padding, PixelNorm, nearest
// upsample + drop-first-row. fp32.

import Foundation
import MLX
import MLXNN

public struct AudioVAEDecoder {
    let w: [String: MLXArray]

    public init(weights: [String: MLXArray]) {
        // Decoder convs live under "audio_vae.decoder."; per-channel stats under
        // "audio_vae." (outside decoder). Strip accordingly so lookups are bare
        // (conv_in.conv.weight / per_channel_statistics._mean_of_means).
        var m: [String: MLXArray] = [:]
        for (k, v) in weights {
            let key: String
            if k.hasPrefix("audio_vae.decoder.") { key = String(k.dropFirst("audio_vae.decoder.".count)) }
            else if k.hasPrefix("audio_vae.") { key = String(k.dropFirst("audio_vae.".count)) }
            else { key = k }
            m[key] = v.asType(.float32)
        }
        self.w = m
    }

    public static func load(path: URL) throws -> AudioVAEDecoder {
        AudioVAEDecoder(weights: try MLX.loadArrays(url: path))
    }

    /// latent (B, 8, T, 16) → mel (B, 2, T', 64).
    public func decode(_ latent: MLXArray) -> MLXArray {
        let B = latent.dim(0), C1 = latent.dim(1), T = latent.dim(2), C2 = latent.dim(3)
        // (B,8,T,16) → (B,T,8,16) → flatten (B,T,128) for denorm
        var xFlat = latent.asType(.float32).transposed(0, 2, 1, 3).reshaped(B, T, C1 * C2)
        let mean = w["per_channel_statistics._mean_of_means"]!.reshaped(1, 1, -1)
        let std = w["per_channel_statistics._std_of_means"]!.reshaped(1, 1, -1)
        xFlat = xFlat * std + mean
        // (B,T,128) → (B,T,8,16) → (B,T,16,8) NHWC: H=T(time), W=16(freq), C=8
        var x = xFlat.reshaped(B, T, C1, C2).transposed(0, 1, 3, 2)

        x = convCausal(x, "conv_in")                 // 8 → 512
        x = resBlock(x, "mid.block_1", shortcut: false)
        x = resBlock(x, "mid.block_2", shortcut: false)
        // up runs in REVERSE index order: up.2 (512→512,↑), up.1 (512→256,↑), up.0 (256→128)
        x = upBlock(x, "up.2", nBlocks: 3, upsample: true, shortcutFirst: false)
        x = upBlock(x, "up.1", nBlocks: 3, upsample: true, shortcutFirst: true)
        x = upBlock(x, "up.0", nBlocks: 3, upsample: false, shortcutFirst: true)
        x = silu(pixelNorm(x))
        x = convCausal(x, "conv_out")                // 128 → 2
        return x.transposed(0, 3, 1, 2)              // (B,2,T',64)
    }

    // MARK: - blocks

    private func upBlock(_ x0: MLXArray, _ prefix: String, nBlocks: Int, upsample: Bool, shortcutFirst: Bool) -> MLXArray {
        var x = x0
        for j in 0 ..< nBlocks {
            x = resBlock(x, "\(prefix).block.\(j)", shortcut: shortcutFirst && j == 0)
        }
        if upsample {
            // nearest 2× on H and W, conv (causal), drop first row
            x = MLX.repeated(x, count: 2, axis: 1)
            x = MLX.repeated(x, count: 2, axis: 2)
            x = convCausal(x, "\(prefix).upsample.conv")
            x = x[0..., 1...]
        }
        return x
    }

    private func resBlock(_ x0: MLXArray, _ prefix: String, shortcut: Bool) -> MLXArray {
        var x = silu(pixelNorm(x0))
        x = convCausal(x, "\(prefix).conv1")
        x = silu(pixelNorm(x))
        x = convCausal(x, "\(prefix).conv2")
        let residual = shortcut ? conv1x1(x0, "\(prefix).nin_shortcut") : x0
        return x + residual
    }

    /// Causal Conv2d (kernel 3): pad top (ks-1) on H, symmetric on W; conv.
    private func convCausal(_ x0: MLXArray, _ prefix: String) -> MLXArray {
        let ks = 3
        let x = MLX.padded(x0, widths: [IntOrPair(0), IntOrPair((ks - 1, 0)), IntOrPair((1, 1)), IntOrPair(0)])
        var y = conv2d(x, w["\(prefix).conv.weight"]!, stride: 1, padding: 0)
        y = y + w["\(prefix).conv.bias"]!
        return y
    }

    /// 1×1 Conv2d (nin_shortcut), no padding.
    private func conv1x1(_ x: MLXArray, _ prefix: String) -> MLXArray {
        conv2d(x, w["\(prefix).conv.weight"]!, stride: 1, padding: 0) + w["\(prefix).conv.bias"]!
    }

    private func pixelNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let v = MLX.mean(x * x, axis: -1, keepDims: true)
        return x * MLX.rsqrt(v + eps)
    }
}
