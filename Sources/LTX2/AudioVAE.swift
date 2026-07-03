// AudioVAE.swift — LTX-2.3 audio VAE decoder (audio latent → mel spectrogram)
// + encoder (waveform → mel → audio latent; the LipDub reference-audio path).
//
// 1:1 functional port of ltx_core_mlx/model/audio_vae/audio_vae.py (decode path,
// distilled config = NO attention) + encoder.py/processor.py (encode path).
// Conv2d-based: the (B,8,T,16) latent is a 2D spatial tensor (NHWC (B,T,16,8));
// the decoder upsamples frequency 16→64 keeping time T, → mel (B,2,T',64); the
// encoder mirrors it back down (mel → latent). Causal height (time) padding,
// PixelNorm, nearest upsample + drop-first-row. fp32.

import Foundation
import MLX
import MLXFast
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

    /// Oracle pixel_norm = mx.fast.rms_norm(weight=None): use the SAME fast kernel
    /// (weight=1 is numerically exact) — the composed mean/rsqrt form drifts ~1e-5
    /// per layer, which the encoder's (std + 1e-8) latent normalization amplifies
    /// (latent maxAbs 0.19 composed vs ~1e-4 fast-kernel).
    private func pixelNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)]), eps: eps)
    }
}

// MARK: - Encoder mel processor


/// Waveform → mel spectrogram for the audio VAE ENCODER (LipDub reference audio).
///
/// 1:1 port of ltx_core_mlx/model/audio_vae/processor.py (AudioProcessor):
/// torchaudio-style MelSpectrogram — nFFT 1024, hop 160, 64 Slaney-scale mels with
/// Slaney area norm, periodic Hann window, center=True reflect padding, power=1.0
/// (magnitude), log(max(mel, 1e-5)). NOT the vocoder's BWE MelSTFT (Vocoder.swift:
/// nFFT 512, hop 80, basis loaded from weights) — different parameters, computed
/// filterbank; keep them separate.
public struct AudioMelProcessor {
    public let nFFT: Int
    public let hop: Int
    public let nMels: Int
    public let melBasis: MLXArray  // (nMels, nFFT/2+1) fp32
    public let window: MLXArray    // (nFFT,) fp32

    public init(sampleRate: Double = 16000, nFFT: Int = 1024, hop: Int = 160, nMels: Int = 64,
                fMin: Double = 0, fMax: Double? = nil) {
        self.nFFT = nFFT
        self.hop = hop
        self.nMels = nMels
        self.melBasis = Self.slaneyMelFilterbank(
            sampleRate: sampleRate, nFFT: nFFT, nMels: nMels,
            fMin: fMin, fMax: fMax ?? sampleRate / 2.0)
        // np.hanning(nFFT+1)[:-1] — periodic Hann.
        let win = (0 ..< nFFT).map { Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double($0) / Double(nFFT))) }
        self.window = MLXArray(win)
    }

    /// Slaney-scale mel filterbank with Slaney area normalization (float64 math →
    /// fp32, matching the oracle's numpy build). Shape (nMels, nFFT/2+1).
    static func slaneyMelFilterbank(sampleRate: Double, nFFT: Int, nMels: Int,
                                    fMin: Double, fMax: Double) -> MLXArray {
        // Slaney mel scale: linear below 1000 Hz, logarithmic above.
        func hzToMel(_ f: Double) -> Double {
            f < 1000.0 ? 3.0 * f / 200.0 : 15.0 + 27.0 * log(max(f, 1e-10) / 1000.0) / log(6.4)
        }
        func melToHz(_ m: Double) -> Double {
            m < 15.0 ? 200.0 * m / 3.0 : 1000.0 * exp((m - 15.0) * log(6.4) / 27.0)
        }
        let nFreqs = nFFT / 2 + 1
        let melMin = hzToMel(fMin), melMax = hzToMel(fMax)
        let hzPoints = (0 ..< nMels + 2).map {
            melToHz(melMin + (melMax - melMin) * Double($0) / Double(nMels + 1))
        }
        let fftFreqs = (0 ..< nFreqs).map { Double($0) * (sampleRate / 2.0) / Double(nFreqs - 1) }
        var fb = [Float](repeating: 0, count: nMels * nFreqs)
        for i in 0 ..< nMels {
            let lower = hzPoints[i], center = hzPoints[i + 1], upper = hzPoints[i + 2]
            let enorm = 2.0 / (upper - lower)  // Slaney area normalization
            for j in 0 ..< nFreqs {
                let f = fftFreqs[j]
                var v = 0.0
                if center > lower, f <= center { v += max(0.0, (f - lower) / (center - lower)) }
                if upper > center, f > center { v += max(0.0, (upper - f) / (upper - center)) }
                fb[i * nFreqs + j] = Float(v * enorm)
            }
        }
        return MLXArray(fb, [nMels, nFreqs])
    }

    /// waveform (B, C, T) or (B, T) → log-mel (B, C, T', nMels) or (B, T', nMels).
    ///
    /// Per-channel 2D loop like the oracle — batching the channels into one 3D
    /// rfft/matmul picks different kernel splits (≈7e-7 log-mel drift) that the
    /// 10-conv encoder stack then amplifies to ~0.19 latent maxAbs; the loop is
    /// bit-exact (B·C is 2, cost irrelevant).
    public func waveformToMel(_ waveform0: MLXArray) -> MLXArray {
        var waveform = waveform0.asType(.float32)
        let squeezeChannel = waveform.ndim == 2
        if squeezeChannel { waveform = waveform.expandedDimensions(axis: 1) }
        let B = waveform.dim(0), C = waveform.dim(1), T = waveform.dim(2)

        // center=True reflect padding: left = x[1...pad] reversed, right = x[T-1-pad ..< T-1] reversed.
        let pad = nFFT / 2
        let leftIdx = MLXArray((1 ... pad).reversed().map { Int32($0) })
        let rightIdx = MLXArray(((T - 1 - pad) ..< (T - 1)).reversed().map { Int32($0) })
        let numFrames = (T + 2 * pad - nFFT) / hop + 1
        var frameIdx = [Int32](); frameIdx.reserveCapacity(numFrames * nFFT)
        for f in 0 ..< numFrames { for j in 0 ..< nFFT { frameIdx.append(Int32(f * hop + j)) } }
        let frameIndices = MLXArray(frameIdx, [numFrames, nFFT])

        var mels: [MLXArray] = []
        for b in 0 ..< B {
            var channelMels: [MLXArray] = []
            for c in 0 ..< C {
                let signal = waveform[b, c]                                          // (T,)
                let padded = MLX.concatenated(
                    [MLX.take(signal, leftIdx), signal, MLX.take(signal, rightIdx)], axis: -1)
                let frames = MLX.take(padded, frameIndices) * window                 // (T', nFFT)
                let mag = MLX.abs(MLX.rfft(frames, axis: -1))                        // (T', nFFT/2+1), power=1.0
                var mel = mag.matmul(melBasis.transposed())                          // (T', nMels)
                mel = MLX.log(MLX.maximum(mel, 1e-5))
                channelMels.append(mel)
            }
            mels.append(MLX.stacked(channelMels, axis: 0))
        }
        var out = MLX.stacked(mels, axis: 0)                                         // (B, C, T', nMels)
        if squeezeChannel { out = out[0..., 0] }
        return out
    }
}

// MARK: - Encoder

/// LTX-2.3 audio VAE ENCODER (mel spectrogram → audio latent) — the LipDub
/// reference-audio path (IC-LORA-PLAN P3b).
///
/// 1:1 functional port of ltx_core_mlx/model/audio_vae/encoder.py: mel (B,2,T',64)
/// → NHWC (B,T',64,2) → conv_in(2→128) → down.0 (128, ↓) → down.1 (→256, ↓) →
/// down.2 (→512) → mid (2 ResBlocks) → PixelNorm/SiLU → conv_out(512→16, double_z)
/// → keep mean 8 channels → per-channel normalize → latent (B,8,T,16). Causal
/// height (time) padding throughout. fp32.
public struct AudioVAEEncoder {
    let w: [String: MLXArray]

    public init(weights: [String: MLXArray]) {
        // Encoder convs live under "audio_vae.encoder."; per-channel stats under
        // "audio_vae." (outside encoder/decoder) — same two-prefix strip as the decoder.
        var m: [String: MLXArray] = [:]
        for (k, v) in weights {
            let key: String
            if k.hasPrefix("audio_vae.encoder.") { key = String(k.dropFirst("audio_vae.encoder.".count)) }
            else if k.hasPrefix("audio_vae.") { key = String(k.dropFirst("audio_vae.".count)) }
            else { key = k }
            m[key] = v.asType(.float32)
        }
        self.w = m
    }

    public static func load(path: URL) throws -> AudioVAEEncoder {
        AudioVAEEncoder(weights: try MLX.loadArrays(url: path))
    }

    /// Oracle `encode_audio`: waveform (B, 2, T) @16 kHz → mel → latent (B, 8, T', 16).
    public func encode(waveform: MLXArray, processor: AudioMelProcessor = AudioMelProcessor()) -> MLXArray {
        encode(processor.waveformToMel(waveform))
    }

    /// mel (B, 2, T', 64) → normalized latent (B, 8, T, 16).
    public func encode(_ mel: MLXArray) -> MLXArray {
        // NHWC: (B, T', 64, 2) — H=T'(time), W=64(freq), C=2
        var x = mel.asType(.float32).transposed(0, 2, 3, 1)

        x = convCausal(x, "conv_in")                                     // 2 → 128
        x = downBlock(x, "down.0", nBlocks: 2, downsample: true, shortcutFirst: false)   // 128, freq 64→32
        x = downBlock(x, "down.1", nBlocks: 2, downsample: true, shortcutFirst: true)    // →256, freq 32→16
        x = downBlock(x, "down.2", nBlocks: 2, downsample: false, shortcutFirst: true)   // →512
        x = resBlock(x, "mid.block_1", shortcut: false)
        x = resBlock(x, "mid.block_2", shortcut: false)
        x = silu(pixelNorm(x))
        x = convCausal(x, "conv_out")                                    // 512 → 16 (double_z)

        // Keep the 8 mean channels, discard logvar; flatten (c,f)-ordered for normalization.
        let B = x.dim(0), T = x.dim(1), F = x.dim(2)
        x = x[0..., 0..., 0..., 0 ..< 8]                                 // (B, T, 16, 8)
        var xFlat = x.transposed(0, 1, 3, 2).reshaped(B, T, 8 * F)      // (B, T, 128)
        let mean = w["per_channel_statistics._mean_of_means"]!.reshaped(1, 1, -1)
        let std = w["per_channel_statistics._std_of_means"]!.reshaped(1, 1, -1)
        xFlat = (xFlat - mean) / (std + 1e-8)
        return xFlat.reshaped(B, T, 8, F).transposed(0, 2, 1, 3)        // (B, 8, T, 16)
    }

    // MARK: - blocks (mirrors AudioVAEDecoder's helpers; encoder-only downsample added)

    private func downBlock(_ x0: MLXArray, _ prefix: String, nBlocks: Int, downsample: Bool, shortcutFirst: Bool) -> MLXArray {
        var x = x0
        for j in 0 ..< nBlocks {
            x = resBlock(x, "\(prefix).block.\(j)", shortcut: shortcutFirst && j == 0)
        }
        if downsample {
            // Causal stride-2 conv: pad (2,0) on H (time), (0,1) on W (freq).
            // NOTE: key is "\(prefix).downsample.conv" — direct Conv2d, no ".conv.conv".
            x = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((2, 0)), IntOrPair((0, 1)), IntOrPair(0)])
            x = conv2d(x, w["\(prefix).downsample.conv.weight"]!, stride: 2, padding: 0)
                + w["\(prefix).downsample.conv.bias"]!
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

    /// Oracle pixel_norm = mx.fast.rms_norm(weight=None): use the SAME fast kernel
    /// (weight=1 is numerically exact) — the composed mean/rsqrt form drifts ~1e-5
    /// per layer, which the encoder's (std + 1e-8) latent normalization amplifies
    /// (latent maxAbs 0.19 composed vs ~1e-4 fast-kernel).
    private func pixelNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)]), eps: eps)
    }
}
