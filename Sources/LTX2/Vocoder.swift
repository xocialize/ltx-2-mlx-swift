// Vocoder.swift — BigVGAN v2 vocoder + BWE: mel → 48kHz stereo waveform.
//
// 1:1 functional port of ltx_core_mlx/model/audio_vae/{vocoder,bwe}.py
// (VocoderWithBWE.__call__). Base BigVGAN (mel→16kHz) + Hann-sinc 3× resampler
// + MelSTFT + BWE-generator BigVGAN → 48kHz. fp32 throughout (the oracle upcasts:
// bf16 compounds over ~108 sequential convs). SnakeBeta (log-scale exp(α)/exp(β)),
// anti-aliased Activation1d (2× up → act → 2× down), 18 AMPBlock1 per BigVGAN.

import Foundation
import MLX

public struct Vocoder {
    let w: [String: MLXArray]

    struct BigVGANConfig {
        let upsampleRates: [Int]
        let resblockKernels: [Int]
        let resblockDilations: [[Int]]
        let applyFinalActivation: Bool
    }
    static let baseCfg = BigVGANConfig(
        upsampleRates: [5, 2, 2, 2, 2, 2], resblockKernels: [3, 7, 11],
        resblockDilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]], applyFinalActivation: true)
    static let bweCfg = BigVGANConfig(
        upsampleRates: [6, 5, 2, 2, 2], resblockKernels: [3, 7, 11],
        resblockDilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]], applyFinalActivation: false)

    public init(weights: [String: MLXArray]) {
        var m: [String: MLXArray] = [:]
        for (k, v) in weights {
            let key = k.hasPrefix("vocoder.") ? String(k.dropFirst("vocoder.".count)) : k
            m[key] = v.asType(.float32)  // fp32 (oracle upcast_weights_to_fp32)
        }
        self.w = m
    }

    public static func load(path: URL) throws -> Vocoder {
        Vocoder(weights: try MLX.loadArrays(url: path))
    }

    /// mel (B, 2, T, 64) → 48kHz stereo waveform (B, 2, T_48k).
    public func callAsFunction(_ mel0: MLXArray) -> MLXArray {
        let mel = mel0.asType(.float32)
        let B = mel.dim(0), C = mel.dim(1), T = mel.dim(2), M = mel.dim(3)

        // 1. base vocoder (concatenate stereo channels): (B,2,T,64)→(B,T,128)
        var melConcat = mel.transposed(0, 1, 3, 2).reshaped(B, C * M, T).transposed(0, 2, 1)
        var wav16 = bigVGAN(melConcat, prefix: "", cfg: Vocoder.baseCfg)  // (B, T16, 2)
        wav16 = wav16.transposed(0, 2, 1)                                  // (B, 2, T16)
        let length16 = wav16.dim(-1)
        let outputLength = length16 * 3

        // pad to multiple of mel_stft hop (80)
        let hop = 80
        let rem = length16 % hop
        if rem != 0 { wav16 = MLX.padded(wav16, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((0, hop - rem))]) }

        // 2. mel of vocoder output → BWE generator residual
        let flat = wav16.reshaped(B * C, wav16.dim(-1))                    // (B*2, T)
        let bweMel = melSTFT(flat)                                         // (B*2, T', 64)
        let tFrames = bweMel.dim(1)
        let bweMelR = bweMel.reshaped(B, C, tFrames, M)                    // (B,2,T',64)
        var bweConcat = bweMelR.transposed(0, 1, 3, 2).reshaped(B, C * M, tFrames).transposed(0, 2, 1)
        var residual = bigVGAN(bweConcat, prefix: "bwe_generator.", cfg: Vocoder.bweCfg)  // (B, Tbwe, 2)
        residual = residual.transposed(0, 2, 1)                            // (B, 2, Tbwe)

        // 3. resample base output to 48kHz (per channel)
        var skipCh: [MLXArray] = []
        for c in 0 ..< C { skipCh.append(hannSinc3x(wav16[0..., c, 0...])) }
        let skip = MLX.stacked(skipCh, axis: 1)                            // (B, 2, T48)

        // 4. add residual + clip
        let minLen = min(skip.dim(-1), residual.dim(-1))
        var out = skip[0..., 0..., 0 ..< minLen] + residual[0..., 0..., 0 ..< minLen]
        out = MLX.clip(out, min: -1.0, max: 1.0)[0..., 0..., 0 ..< outputLength]
        _ = melConcat; _ = bweConcat
        return out
    }

    // MARK: - BigVGAN

    private func bigVGAN(_ mel: MLXArray, prefix p: String, cfg: BigVGANConfig) -> MLXArray {
        var x = conv1dB(mel, "\(p)conv_pre", stride: 1, padding: 3)
        let numKernels = cfg.resblockKernels.count
        for i in 0 ..< cfg.upsampleRates.count {
            x = convT1dB(x, "\(p)ups.\(i)", stride: cfg.upsampleRates[i],
                         padding: (kernelFor(x: x, p: "\(p)ups.\(i)") - cfg.upsampleRates[i]) / 2)
            var xs: MLXArray? = nil
            for j in 0 ..< numKernels {
                let idx = i * numKernels + j
                let r = ampBlock(x, "\(p)resblocks.\(idx)", kernel: cfg.resblockKernels[j], dilations: cfg.resblockDilations[j])
                xs = (xs == nil) ? r : xs! + r
            }
            x = xs! / Float(numKernels)
        }
        x = activation1d(x, "\(p)act_post")
        x = conv1dB(x, "\(p)conv_post", stride: 1, padding: 3, hasBias: false)
        return cfg.applyFinalActivation ? MLX.tanh(x) : x
    }

    private func ampBlock(_ x0: MLXArray, _ p: String, kernel: Int, dilations: [Int]) -> MLXArray {
        var x = x0
        for j in 0 ..< dilations.count {
            let residual = x
            x = activation1d(x, "\(p).acts1.\(j)")
            let pad1 = (kernel * dilations[j] - dilations[j]) / 2
            x = conv1dB(x, "\(p).convs1.\(j)", stride: 1, padding: pad1, dilation: dilations[j])
            x = activation1d(x, "\(p).acts2.\(j)")
            x = conv1dB(x, "\(p).convs2.\(j)", stride: 1, padding: kernel / 2)
            x = x + residual
        }
        return x
    }

    // MARK: - Activation1d / SnakeBeta / anti-alias resample

    private func activation1d(_ x: MLXArray, _ p: String) -> MLXArray {
        var y = upSample1d(x, "\(p).upsample.filter")
        y = snakeBeta(y, "\(p).act")
        return downSample1d(y, "\(p).downsample.lowpass.filter")
    }

    private func snakeBeta(_ x: MLXArray, _ p: String) -> MLXArray {
        let alpha = MLX.exp(w["\(p).alpha"]!).reshaped(1, 1, -1)
        let beta = MLX.exp(w["\(p).beta"]!).reshaped(1, 1, -1)
        let s = MLX.sin(alpha * x)
        return x + (1.0 / (beta + 1e-9)) * (s * s)
    }

    /// 2× upsample with anti-alias filter. x (B,T,C) → (B,2T,C).
    private func upSample1d(_ x: MLXArray, _ filterKey: String) -> MLXArray {
        let B = x.dim(0), T = x.dim(1), C = x.dim(2)
        let filt = w[filterKey]!                      // (1, K, 1)
        let K = filt.dim(1)
        // zero-insert (even idx = x): (B,T,C) → (B,T,2,C) → (B,2T,C)
        let z = MLXArray.zeros(like: x)
        var xUp = MLX.stacked([x, z], axis: 2).reshaped(B, T * 2, C)
        xUp = xUp.transposed(0, 2, 1).reshaped(B * C, T * 2, 1)
        let pad = K / 2
        let left = MLX.repeated(xUp[0..., 0 ..< 1, 0...], count: pad, axis: 1)
        let right = MLX.repeated(xUp[0..., (xUp.dim(1) - 1)..., 0...], count: pad - 1, axis: 1)
        xUp = MLX.concatenated([left, xUp, right], axis: 1)
        var out = conv1d(xUp, filt, stride: 1, padding: 0)
        let tOut = out.dim(1)
        out = out.reshaped(B, C, tOut).transposed(0, 2, 1) * 2.0
        return out
    }

    /// 2× downsample with low-pass filter. x (B,T,C) → (B,T//2,C).
    private func downSample1d(_ x: MLXArray, _ filterKey: String) -> MLXArray {
        let B = x.dim(0), T = x.dim(1), C = x.dim(2)
        let filt = w[filterKey]!
        let K = filt.dim(1)
        var y = x.transposed(0, 2, 1).reshaped(B * C, T, 1)
        let even = K % 2 == 0 ? 1 : 0
        let padLeft = K / 2 - even, padRight = K / 2
        let left = MLX.repeated(y[0..., 0 ..< 1, 0...], count: padLeft, axis: 1)
        let right = MLX.repeated(y[0..., (y.dim(1) - 1)..., 0...], count: padRight, axis: 1)
        y = MLX.concatenated([left, y, right], axis: 1)
        y = conv1d(y, filt, stride: 2, padding: 0)
        let tOut = y.dim(1)
        return y.reshaped(B, C, tOut).transposed(0, 2, 1)
    }

    // MARK: - Hann-sinc 3× resampler (no weights; kernel built here)

    private func hannSinc3x(_ x: MLXArray) -> MLXArray {
        let ratio = 3, lpfw = 6, rolloff: Float = 0.99
        let width = Int(ceil(Double(lpfw) / Double(rolloff)))   // 7
        let kernelSize = 2 * width * ratio + 1                  // 43
        // Hann-windowed sinc kernel
        var k = [Float](repeating: 0, count: kernelSize)
        for i in 0 ..< kernelSize {
            let timeAxis = (Double(i) / Double(ratio) - Double(width)) * Double(rolloff)
            let tc = max(-Double(lpfw), min(Double(lpfw), timeAxis))
            let window = pow(cos(tc * Double.pi / Double(lpfw) / 2), 2)
            let sinc = timeAxis == 0 ? 1.0 : sin(Double.pi * timeAxis) / (Double.pi * timeAxis)
            k[i] = Float(sinc * window * Double(rolloff) / Double(ratio))
        }
        let kernel = MLXArray(k).reshaped(kernelSize, 1)        // (K, 1)

        let B = x.dim(0), T = x.dim(1)
        let pad = width
        let first = MLX.repeated(x[0..., 0 ..< 1], count: pad, axis: 1)
        let last = MLX.repeated(x[0..., (T - 1)...], count: pad, axis: 1)
        let xPadded = MLX.concatenated([first, x, last], axis: 1)  // (B, T+2pad)
        let tPad = xPadded.dim(1)
        let ziLen = (tPad - 1) * ratio + 1
        // zero-insert with stride=ratio: (B, tPad) → (B, tPad, ratio) [first col = x] → (B, tPad*ratio) → trim
        let zerosTail = MLXArray.zeros([B, tPad, ratio - 1])
        let interleaved = MLX.concatenated([xPadded.reshaped(B, tPad, 1), zerosTail], axis: 2).reshaped(B, tPad * ratio)
        var up = interleaved[0..., 0 ..< ziLen]
        let K = kernelSize
        up = MLX.padded(up.reshaped(B, ziLen, 1), widths: [IntOrPair(0), IntOrPair((K - 1, K - 1)), IntOrPair(0)])
        var result = conv1d(up, kernel.reshaped(1, K, 1), stride: 1, padding: 0).reshaped(B, -1)
        result = result * Float(ratio)
        let padLeft = 2 * width * ratio, padRight = K - ratio
        result = result[0..., padLeft ..< (result.dim(1) - padRight)]
        return result[0..., 0 ..< (T * ratio)]
    }

    // MARK: - MelSTFT

    private func melSTFT(_ waveform: MLXArray) -> MLXArray {
        let nFFT = 512, hop = 80, nBins = nFFT / 2 + 1
        var x = waveform.expandedDimensions(axis: -1)           // (B, T, 1)
        let leftPad = max(0, nFFT - hop)                        // 432
        x = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((leftPad, 0)), IntOrPair(0)])
        let basis = w["mel_stft.stft_fn.forward_basis"]!        // (514, 512, 1)
        let stft = conv1d(x, basis, stride: hop, padding: 0)    // (B, T', 514)
        let real = stft[0..., 0..., 0 ..< nBins]
        let imag = stft[0..., 0..., nBins ..< (2 * nBins)]
        let mag = MLX.sqrt(real * real + imag * imag + 1e-9)
        let mel = mag.matmul(w["mel_stft.mel_basis"]!.transposed())  // (B, T', 64)
        return MLX.log(MLX.maximum(mel, 1e-5))
    }

    // MARK: - conv primitives

    private func kernelFor(x: MLXArray, p: String) -> Int { w["\(p).weight"]!.dim(1) }

    private func conv1d(_ x: MLXArray, _ weight: MLXArray, stride: Int, padding: Int, dilation: Int = 1) -> MLXArray {
        MLX.conv1d(x, weight, stride: stride, padding: padding, dilation: dilation)
    }

    private func conv1dB(_ x: MLXArray, _ p: String, stride: Int, padding: Int, dilation: Int = 1, hasBias: Bool = true) -> MLXArray {
        var y = MLX.conv1d(x, w["\(p).weight"]!, stride: stride, padding: padding, dilation: dilation)
        if hasBias, let b = w["\(p).bias"] { y = y + b }
        return y
    }

    private func convT1dB(_ x: MLXArray, _ p: String, stride: Int, padding: Int) -> MLXArray {
        var y = MLX.convTransposed1d(x, w["\(p).weight"]!, stride: stride, padding: padding)
        if let b = w["\(p).bias"] { y = y + b }
        return y
    }
}
