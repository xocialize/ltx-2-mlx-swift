// Retake.swift — local retake/extend on the DISTILLED checkpoint (AB-R-0133 piece 2).
//
// Port of LTX Desktop's SHIPPING retake (`services/retake_pipeline/ltx_retake_pipeline.py`,
// distilled branch), NOT the dev+CFG oracle in ltx-2-mlx — the vendor's product runs distilled
// (SimpleDenoiser, DISTILLED_SIGMA_VALUES byte-identical to ours) and the dev branch is dead code
// there. Mechanism: encode the source (tiled), then `runConditioned` with a TEMPORAL-SPAN denoise
// mask — 1 inside [start, end) = regenerate, 0 outside = hold clean — per modality. A modality
// with an all-zero mask is FROZEN: re-blended to clean every step, so cross-attention sees the
// true source while nothing of it is regenerated.

import Foundation
import MLX
import MLXRandom

public enum RetakeMode: String, Sendable {
    case replaceVideo = "replace_video"            // regenerate video in the span; audio frozen
    case replaceAudio = "replace_audio"            // regenerate audio in the span; video frozen
    case replaceAudioAndVideo = "replace_audio_and_video"
}

public enum RetakeMask {
    /// Per-token video span mask (1, F·H·W, 1): 1 where the token's START time ∈ [start, end).
    /// Token start times reproduce `Positions.video`'s fStarts/fps exactly — the oracle's
    /// TemporalRegionMask keys on the pixel-frame START boundary, not the midpoint.
    public static func videoSpan(F: Int, H: Int, W: Int, fps: Float,
                                 start: Float, end: Float) -> MLXArray {
        let idx = MLXArray(0 ..< F).asType(.float32)
        let fStarts = MLX.maximum(idx * 8 + 1 - 8, 0.0) / fps                    // (F,) seconds
        let inSpan = MLX.logicalAnd(fStarts .>= start, fStarts .< end).asType(.float32)
        return MLX.broadcast(inSpan.reshaped(1, F, 1, 1), to: [1, F, H * W, 1])
            .reshaped(1, F * H * W, 1)
    }

    /// Per-token audio span mask (1, T, 1) on the same start-boundary rule (audio coords are
    /// already seconds — `Positions.audio`'s `starts`).
    public static func audioSpan(tokens T: Int, start: Float, end: Float) -> MLXArray {
        let idx = MLXArray(0 ..< T).asType(.float32)
        let starts = MLX.maximum(idx * Positions.audioDownsampleFactor + 1 - Positions.audioDownsampleFactor, 0.0)
            * Positions.audioHopLength / Positions.audioSampleRate
        let inSpan = MLX.logicalAnd(starts .>= start, starts .< end).asType(.float32)
        return inSpan.reshaped(1, T, 1)
    }
}

extension LTX2Pipeline {
    /// Retake: regenerate `[start, start+duration)` of an already-encoded source, holding
    /// everything outside the span (and any frozen modality) exactly.
    ///
    /// Single-stage at the SOURCE's own geometry — the vendor's retake never upsamples; dims are
    /// the caller's job and must snap DOWN to /32 (their `video_resolution.py` rule; a retake
    /// that upscales the user's footage is a bug).
    ///
    /// - Parameters:
    ///   - sourcePixels: (1,3,F,H,W) in [-1,1], F = 8k+1, H/W multiples of 32.
    ///   - sourceAudioWaveform: (1,2,S) 16 kHz stereo in [-1,1], or nil (video-only source —
    ///     forces `.replaceAudioAndVideo`-style audio regeneration inside the span only).
    ///   - extendFeather: widen the regenerated span this many seconds INTO the kept source so
    ///     the frozen→generated boundary blends (the vendor's `_EXTEND_MASK_DELTA_SECONDS`).
    ///     0 for plain retake; 0.5 for extend.
    public func retake(
        prompt: String, sourcePixels: MLXArray, sourceAudioWaveform: MLXArray?,
        fps: Double, start: Double, duration: Double, mode: RetakeMode,
        seed: UInt64? = nil, extendFeather: Double = 0,
        encodeTiling: EncodeTiling = .vendorDefault,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> Output {
        let f = sourcePixels.dim(2), h = sourcePixels.dim(3), w = sourcePixels.dim(4)
        precondition((f - 1) % 8 == 0 && h % 32 == 0 && w % 32 == 0,
                     "retake source must be 8k+1 frames and /32 spatial (snap DOWN, never up)")
        let fLat = (f - 1) / 8 + 1, hLat = h / 32, wLat = w / 32
        let nv = fLat * hLat * wLat
        let audioT = Positions.audioTokenCount(numFrames: f, fps: fps)
        let spanStart = Float(max(0, start - extendFeather))
        let spanEnd = Float(start + duration)

        // 1. Text encode (sequential, self-evicting — the t2v discipline).
        let (videoEmbeds, audioEmbeds) = try await encodePrompt(prompt)

        // 2. Encode the source: video TILED (the whole reason encodeTiled exists — an HD source
        //    untiled is exactly the OOM the vendor's fork patched), audio via the audio VAE.
        LTX2Progress.report(.encode)
        try ensureVAEEncoder()
        let srcLatent = vaeEncoder!.encodeTiled(sourcePixels, tiling: encodeTiling)
        let cleanVideo = LTX2Pipeline.patchify(srcLatent)               // (1, nv, 128)
        eval(cleanVideo)
        dropUpscaler()

        var cleanAudio: MLXArray? = nil
        if let wave = sourceAudioWaveform {
            let enc = try AudioVAEEncoder.load(path: ltxDir.appending(path: "audio_vae.safetensors"))
            let aLat = enc.encode(waveform: wave)                       // (1,8,T',16)
            var (tok, _) = Positions.patchifyLipdubAudioReference(aLat, negativePositions: false)
            // Align T' → audioT (crop or zero-pad the tail) — the source track and the token
            // grid disagree by ≤1 token from rounding.
            let tHave = tok.dim(1)
            if tHave > audioT { tok = tok[0..., 0 ..< audioT] }
            else if tHave < audioT {
                tok = MLX.concatenated([tok, MLXArray.zeros([1, audioT - tHave, 128])], axis: 1)
            }
            cleanAudio = tok
            eval(cleanAudio!)
            Memory.clearCache()
        }

        // 3. Span masks per mode. An all-zero mask FREEZES the modality (held clean every step);
        //    a modality with no clean source (nil audio) regenerates over the span like the span
        //    tokens of the other stream.
        let vSpan = RetakeMask.videoSpan(F: fLat, H: hLat, W: wLat, fps: Float(fps),
                                         start: spanStart, end: spanEnd)
        let aSpan = RetakeMask.audioSpan(tokens: audioT, start: spanStart, end: spanEnd)
        let zerosV = MLXArray.zeros([1, nv, 1]), zerosA = MLXArray.zeros([1, audioT, 1])
        let videoMask = (mode == .replaceAudio) ? zerosV : vSpan
        let audioMask: MLXArray = {
            guard cleanAudio != nil else { return aSpan }   // no source audio → regen span only
            return (mode == .replaceVideo) ? zerosA : aSpan
        }()

        // 4. Noise + conditioned single-stage denoise (distilled sigmas).
        //
        // ⚠️ This comment used to read "mirroring i2v: no keyframes mask — TemporalRegionMask
        // touches only the denoise mask, and our shipping one-stage i2v does the same". The first
        // clause is true and the conclusion did not follow (AB-T-0090). `TemporalRegionMask` adds
        // no keyframe SLOTS, but the oracle keeps slots and the first-latent-frame mask strictly
        // separate: `create_initial_state` marks the first frame unconditionally, before any
        // conditioning is applied. Citing i2v as precedent propagated one omission into four
        // pipelines — t2v, i2v, icT2V and here.
        if let seed { MLXRandom.seed(seed) }
        let videoLatent = MLXRandom.normal([1, nv, 128])
        let audioLatent = MLXRandom.normal([1, audioT, 128])
        _ = try ensureDiT()
        armStreamingGate(largestStageTokens: nv + audioT)
        let kfMask = isLTX25
            ? Self.firstLatentFrameKeyframesMask(totalTokens: nv, tokensPerLatentFrame: hLat * wLat)
            : nil
        let (vfinal, afinalOpt) = try DenoiseLoop.runConditioned(
            dit: try ensureDiT(), videoLatent0: videoLatent, audioLatent0: audioLatent,
            sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: Positions.video(F: fLat, H: hLat, W: wLat, fps: Float(fps)),
            audioPositions: Positions.audio(tokens: audioT),
            videoCleanLatent: cleanVideo, videoDenoiseMask: videoMask,
            audioCleanLatent: cleanAudio ?? MLXArray.zeros([1, audioT, 128]),
            audioDenoiseMask: audioMask,
            keyframesMask: kfMask)
        let afinal = afinalOpt!
        eval(vfinal, afinal)
        quiesceStreaming()
        dropDiTIfSequential()

        // 5. Decode (the existing tiled/chunked decode path).
        let vspatial = vfinal.reshaped(1, fLat, hLat, wLat, 128).transposed(0, 4, 1, 2, 3)
        try ensureDecoder()
        let pixels = try decodePixels(vspatial)
        // ⚠️ For .replaceVideo the caller should MUX THE SOURCE TRACK rather than this decode —
        // a frozen latent round-trips the audio VAE and is near- but not bit-identical. Returned
        // anyway so callers without the source track still get audio.
        let waveform = decodeAudio(afinal)
        eval(pixels); if let waveform { eval(waveform) }
        dropDecoder()
        return Output(video: pixels, audio: waveform)
    }

    /// Extend: grow the clip by `addSeconds` at the end. The source occupies the head; the new
    /// tail is pure generation, with the vendor's 0.5 s feather widening the span INTO the source
    /// so the boundary blends instead of cutting.
    public func extend(
        prompt: String, sourcePixels: MLXArray, sourceAudioWaveform: MLXArray?,
        fps: Double, addSeconds: Double, seed: UInt64? = nil,
        encodeTiling: EncodeTiling = .vendorDefault,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> Output {
        let srcF = sourcePixels.dim(2)
        let addFrames = Int((addSeconds * fps).rounded())
        let totalF = srcF + addFrames - (srcF + addFrames - 1) % 8   // snap to 8k+1
        let padF = totalF - srcF
        precondition(padF > 0, "extend needs at least one 8-frame latent step of new content")
        // Pad the source with repeats of its last frame — placeholder pixels for the region the
        // span mask fully regenerates; only their ENCODED values under the feather matter, and
        // the feather zone is genuine source.
        let lastFrame = sourcePixels[0..., 0..., (srcF - 1) ..< srcF]
        let pad = MLX.broadcast(lastFrame, to: [1, 3, padF, sourcePixels.dim(3), sourcePixels.dim(4)])
        let padded = MLX.concatenated([sourcePixels, pad], axis: 2)
        let srcSeconds = Double(srcF) / fps
        return try await retake(
            prompt: prompt, sourcePixels: padded, sourceAudioWaveform: sourceAudioWaveform,
            fps: fps, start: srcSeconds, duration: Double(padF) / fps + 1,   // span to the end
            mode: .replaceAudioAndVideo, seed: seed,
            extendFeather: 0.5, encodeTiling: encodeTiling)
    }
}
