// AudioToVideo.swift — a2v: generate video FROM a supplied audio track (AB-D-0044 follow-on).
//
// The last LTX-Desktop local-feature gap. Both the shipping Desktop pipeline
// (`services/a2v_pipeline/distilled_a2v_pipeline.py`) and the first-party reference
// (`ltx_pipelines/a2vid_two_stage.py`) agree on the contract, and both run it DISTILLED:
//
//   stage 1  video  ModalitySpec(context=video_ctx, conditionings=[...])            ← pure noise
//            audio  ModalitySpec(frozen=True, noise_scale=0.0, initial_latent=enc)  ← held exact
//            sigmas DISTILLED_SIGMA_VALUES, at HALF resolution
//   upsample video latent 2x
//   stage 2  video  ModalitySpec(noise_scale=stage2[0], initial_latent=upscaled)
//            audio  ModalitySpec(frozen=True, noise_scale=0.0, initial_latent=enc)  ← the SAME
//                                                                                     encoding
//            sigmas STAGE_2_DISTILLED_SIGMA_VALUES, at full resolution
//   decode video; return the ORIGINAL waveform, not the VAE round-trip.
//
// Two details are easy to get wrong and are load-bearing:
//
//  1. `noise_scale=0.0` on audio — NOT retake's default of 1.0. `GaussianNoiser` computes
//     `lerp(initial_latent, noise, noise_scale)`, so 0.0 leaves the encoded latent untouched
//     where 1.0 would replace it with pure noise. We reproduce it by handing `runConditioned`
//     the encoded latent as BOTH the initial and the clean audio latent under an all-zero mask.
//
//  2. `frozen=True` ⇒ the audio modality's SCALAR timestep is zero, not the step sigma
//     (`ltx_pipelines/utils/helpers.py:478`). That is not free from the mask — the mask is
//     per-token and the AV-cross gate + prompt AdaLN are not. `runConditioned` infers it from
//     the all-zero mask (AB-R-0139); without that fix every a2v forward would mis-condition the
//     frozen audio branch, and those hidden states are exactly what the video branch attends to
//     through a2v cross-attention. This pipeline is the reason that fix went in first.

import Foundation
import MLX
import MLXProfiling
import MLXRandom

extension LTX2Pipeline {

    /// Encode a source waveform to the frozen audio latent, aligned to the run's token grid.
    /// Mirrors Desktop's pad/truncate against `AudioLatentShape.from_duration(...)`: the track
    /// and the token grid disagree by a token or two from rounding, and a SHORT track is
    /// zero-padded rather than rejected (Desktop pads too — a 3 s clip over 5 s of video is a
    /// legal request, and the tail simply carries no audio conditioning).
    func encodeFrozenAudio(waveform: MLXArray, audioT: Int) throws -> MLXArray {
        let enc = try AudioVAEEncoder.load(path: ltxDir.appending(path: "audio_vae.safetensors"))
        let aLat = enc.encode(waveform: waveform)                      // (1,8,T',16)
        var (tok, _) = Positions.patchifyLipdubAudioReference(aLat, negativePositions: false)
        let tHave = tok.dim(1)
        if tHave > audioT { tok = tok[0..., 0 ..< audioT] }
        else if tHave < audioT {
            tok = MLX.concatenated([tok, MLXArray.zeros([1, audioT - tHave, 128])], axis: 1)
        }
        eval(tok)
        Memory.clearCache()
        return tok
    }

    /// a2v — two-stage distilled audio-to-video with the audio track held frozen throughout.
    ///
    /// - Parameters:
    ///   - audioWaveform: (1, 2, S) 16 kHz stereo in [-1,1] — what `AudioInput.referenceWaveform`
    ///     returns. This track is what the video is generated AGAINST, and it is also what the
    ///     caller should mux: `Output.audio` is the original waveform, not a VAE round-trip.
    ///   - numFrames: pixel frames, 8k+1. Together with `fps` this fixes the audio token grid,
    ///     so a track longer than the video is truncated and a shorter one zero-padded.
    /// `keyframes` / `initImage` condition the VIDEO while the audio stays frozen (AB-T-0096).
    /// Both the vendor CLI (`--image PATH FRAME_IDX STRENGTH`, repeatable) and Desktop's
    /// `DistilledA2VPipeline` (`images=[...]`, applied at BOTH stages) accept these on a2v; we
    /// previously dropped them SILENTLY, which is worse than not having them.
    public func audioToVideo(
        prompt: String, audioWaveform: MLXArray,
        height: Int = 512, width: Int = 704, numFrames: Int = 9,
        fps: Double = 24, seed: UInt64? = nil,
        keyframes: [KeyframeRequest] = [],
        initImage: ((_ width: Int, _ height: Int) throws -> MLXArray)? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> Output {
        for k in keyframes where k.frameIdx <= 0 {
            throw TwoStageError.badGeometry(
                "keyframe frameIdx must be > 0 (got \(k.frameIdx)); frame 0 replaces the latent "
                + "and rides initImage, per the oracle's combined_image_conditionings split")
        }
        // Single-stage fallback mirrors `t2vTwoStage`: without an encoder+upsampler there is no
        // stage 2 to run, and a one-stage a2v is still a correct (lower-detail) a2v.
        guard supportsTwoStage else {   // the existing public accessor: hasEncoder && hasUpsampler
            return try await audioToVideoOneStage(
                prompt: prompt, audioWaveform: audioWaveform, height: height, width: width,
                numFrames: numFrames, fps: fps, seed: seed)
        }

        let geo = try resolveTwoStageGeometry(height: height, width: width,
                                              numFrames: numFrames, fps: fps)
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)
        let s2 = Positions.stage2Sigmas
        let sigma0 = s2[0]
        MLXProfiler.shared.beginRun(String(format:
            "a2v %dx%d %df fps=%.0f | %@ | fLat=%d→%d nv1=%d nv2=%d audioT=%d | steps s1=%d s2=%d",
            width, height, numFrames, fps, geo.variant.rawValue, geo.fLat1, geo.fLat2,
            geo.nv1, geo.nv2, audioT, Positions.distilledSigmas.count - 1, s2.count - 1))

        // 1. Text encode (positive only — distilled, no negative context).
        let (videoEmbeds, audioEmbeds) = try await encodePrompt(prompt)

        // 2. Encode the driving audio ONCE and reuse it in both stages, exactly as the vendor
        //    does (it passes `encoded_audio_latent` to stage 2, NOT stage 1's audio state).
        LTX2Progress.report(.encode)
        let frozenAudio = try encodeFrozenAudio(waveform: audioWaveform, audioT: audioT)
        let zeroAudioMask = MLXArray.zeros([1, audioT, 1])   // all-zero ⇒ frozen (scalar σ = 0)

        // --- Stage 1: half-resolution video from pure noise, audio frozen ---
        if let seed { MLXRandom.seed(seed) }
        let v1 = MLXRandom.normal([1, geo.nv1, 128])
        _ = try ensureDiT()
        armStreamingGate(largestStageTokens: geo.nv2 + audioT)
        let kfMask1 = isLTX25
            ? Self.firstLatentFrameKeyframesMask(totalTokens: v1.dim(1),
                                                 tokensPerLatentFrame: geo.hLat1 * geo.wLat1)
            : nil
        let vPos1 = Positions.video(F: geo.fLat1, H: geo.hLat1, W: geo.wLat1, fps: Float(geo.fps1))
        let aPos = Positions.audio(tokens: audioT)
        let conditioned = !keyframes.isEmpty || initImage != nil
        let v1f: MLXArray
        if conditioned {
            let items1 = try keyframeItems(keyframes, hLat: geo.hLat1, wLat: geo.wLat1, fps: Float(geo.fps1))
            let base1 = try initImage.map {
                try frame0Conditioning($0, hLat: geo.hLat1, wLat: geo.wLat1, nv: geo.nv1, dtype: v1.dtype)
            }
            v1f = try runKeyframedStage(
                videoLatent: v1, audioLatent: frozenAudio, sigmas: Positions.distilledSigmas,
                videoText: videoEmbeds, audioText: audioEmbeds,
                videoPositions: vPos1, audioPositions: aPos,
                keyframesMask: kfMask1, items: items1,
                baseClean: base1?.clean, baseMask: base1?.mask,
                audioCleanLatent: frozenAudio, audioDenoiseMask: zeroAudioMask,
                ancestralEta: 0, ancestralNoiseSeed: 0,
                label: "a2v-s1-", stage: 1, totalStages: 2).video
        } else {
        let (v1fPlain, _) = try DenoiseLoop.runConditioned(
            dit: try ensureDiT(), videoLatent0: v1, audioLatent0: frozenAudio,
            sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: vPos1,
            audioPositions: aPos,
            audioCleanLatent: frozenAudio, audioDenoiseMask: zeroAudioMask,
            keyframesMask: kfMask1,
            // ⚠️ NO ancestral sampler here, deliberately — do not "restore" it by analogy with
            // `t2vTwoStage`. Ancestral stage-1 is a DISTILLED-T2V behaviour, opted into per
            // checkpoint by `should_use_ancestral_sampler` (first-party `distilled.py:76`).
            // Neither a2v pipeline passes it: the first-party `a2vid_two_stage.py` has no
            // stepper/loop override at all, and Desktop's `DistilledA2VPipeline` calls
            // `self.stage(...)` without the `distilled_stage_sampler_kwargs` its own retake
            // pipeline uses — both therefore take DiffusionStage's deterministic Euler defaults.
            // (Our retake already matched this; a2v had inherited t2v's eta=1.0 by copy.)
            label: "a2v-s1-", stage: 1, totalStages: 2)
        v1f = v1fPlain
        }
        eval(v1f)

        // --- Upscale in un-normalized latent space ---
        LTX2Progress.report(.upsample)
        let upSpan = MLXProfiler.shared.begin("upscale", "vae-enc+upsampler",
                                              note: "\(geo.variant.rawValue) stage1→stage2 latent")
        try ensureVAEEncoder(); try ensureUpsampler()
        let v1spatial = v1f.reshaped(1, geo.fLat1, geo.hLat1, geo.wLat1, 128).transposed(0, 4, 1, 2, 3)
        let upscaled = vaeEncoder!.normalizeLatent(upsampler!(vaeEncoder!.denormalizeLatent(v1spatial)))
        eval(upscaled)
        MLXProfiler.shared.end(upSpan)
        dropUpscaler()
        guard upscaled.dim(2) == geo.fLat2, upscaled.dim(3) == geo.hLat2,
              upscaled.dim(4) == geo.wLat2 else {
            throw TwoStageError.badGeometry(
                "upsampler(\(geo.variant.rawValue)) produced latent \(upscaled.shape), expected "
                + "(1, 128, \(geo.fLat2), \(geo.hLat2), \(geo.wLat2)) for target "
                + "\(width)x\(height)x\(numFrames)f")
        }
        let v2tokens = LTX2Pipeline.patchify(upscaled)

        // --- Stage 2: full resolution refine, audio still frozen at the SAME encoding ---
        let v2init = LTX2Pipeline.noiseInit(clean: v2tokens, sigma: sigma0,
                                            shape: v2tokens.shape, seed: seed.map { $0 &+ 2 })
        let kfMask2 = isLTX25
            ? Self.firstLatentFrameKeyframesMask(totalTokens: v2init.dim(1),
                                                 tokensPerLatentFrame: geo.hLat2 * geo.wLat2)
            : nil
        let vPos2 = Positions.video(F: geo.fLat2, H: geo.hLat2, W: geo.wLat2, fps: Float(fps))
        let v2f: MLXArray
        if conditioned {
            // Applied in BOTH stages, re-encoded at stage 2's geometry — Desktop does the same.
            let items2 = try keyframeItems(keyframes, hLat: geo.hLat2, wLat: geo.wLat2, fps: Float(fps))
            let base2 = try initImage.map {
                try frame0Conditioning($0, hLat: geo.hLat2, wLat: geo.wLat2, nv: geo.nv2, dtype: v2init.dtype)
            }
            dropUpscaler()
            v2f = try runKeyframedStage(
                videoLatent: v2init, audioLatent: frozenAudio, sigmas: s2,
                videoText: videoEmbeds, audioText: audioEmbeds,
                videoPositions: vPos2, audioPositions: aPos,
                keyframesMask: kfMask2, items: items2,
                baseClean: base2?.clean, baseMask: base2?.mask,
                audioCleanLatent: frozenAudio, audioDenoiseMask: zeroAudioMask,
                ancestralEta: 0, ancestralNoiseSeed: 0,
                label: "a2v-s2-", stage: 2, totalStages: 2).video
        } else {
            let (v, _) = try DenoiseLoop.runConditioned(
                dit: try ensureDiT(), videoLatent0: v2init, audioLatent0: frozenAudio, sigmas: s2,
                videoText: videoEmbeds, audioText: audioEmbeds,
                videoPositions: vPos2, audioPositions: aPos,
                audioCleanLatent: frozenAudio, audioDenoiseMask: zeroAudioMask,
                keyframesMask: kfMask2,
                label: "a2v-s2-", stage: 2, totalStages: 2)
            v2f = v
        }
        eval(v2f)
        quiesceStreaming()
        dropDiTIfSequential()

        // 3. Decode video. Audio comes back as the ORIGINAL waveform: the frozen latent would
        //    round-trip the audio VAE and land near- but not bit-identical, and a2v's whole
        //    premise is that the supplied track is the ground truth ("return original audio
        //    (not VAE-decoded) for fidelity" — Desktop's own comment).
        let vspatial = v2f.reshaped(1, geo.fLat2, geo.hLat2, geo.wLat2, 128).transposed(0, 4, 1, 2, 3)
        try ensureDecoder()
        let pixels = try decodePixels(vspatial)
        eval(pixels)
        dropDecoder()
        return Output(video: pixels, audio: audioWaveform)
    }

    /// One-stage a2v — the no-upsampler fallback. Same frozen-audio contract, full resolution
    /// in a single distilled pass.
    func audioToVideoOneStage(
        prompt: String, audioWaveform: MLXArray,
        height: Int, width: Int, numFrames: Int, fps: Double, seed: UInt64?,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> Output {
        guard height % 32 == 0, width % 32 == 0 else {
            throw TwoStageError.badGeometry("a2v target \(width)x\(height) must be divisible by 32")
        }
        let fLat = (numFrames + 7) / 8, hLat = height / 32, wLat = width / 32
        let nv = fLat * hLat * wLat
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)
        MLXProfiler.shared.beginRun(String(format:
            "a2v-1stage %dx%d %df fps=%.0f | nv=%d audioT=%d | steps=%d",
            width, height, numFrames, fps, nv, audioT, Positions.distilledSigmas.count - 1))

        let (videoEmbeds, audioEmbeds) = try await encodePrompt(prompt)
        LTX2Progress.report(.encode)
        let frozenAudio = try encodeFrozenAudio(waveform: audioWaveform, audioT: audioT)

        if let seed { MLXRandom.seed(seed) }
        let v0 = MLXRandom.normal([1, nv, 128])
        _ = try ensureDiT()
        armStreamingGate(largestStageTokens: nv + audioT)
        // Same first-latent-frame keyframes mask the two-stage arm uses. The oracle populates it
        // in `create_initial_state` for EVERY modality state — "the reference implementation marks
        // it unconditionally -- independently of whether any keyframe slots exist"
        // (`ltx_core/tools.py:186`) — so a generation path must not skip it just because it has no
        // user keyframes.
        let kfMask = isLTX25
            ? Self.firstLatentFrameKeyframesMask(totalTokens: v0.dim(1),
                                                 tokensPerLatentFrame: hLat * wLat)
            : nil
        let (vf, _) = try DenoiseLoop.runConditioned(
            dit: try ensureDiT(), videoLatent0: v0, audioLatent0: frozenAudio,
            sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: Positions.video(F: fLat, H: hLat, W: wLat, fps: Float(fps)),
            audioPositions: Positions.audio(tokens: audioT),
            audioCleanLatent: frozenAudio,
            audioDenoiseMask: MLXArray.zeros([1, audioT, 1]),
            keyframesMask: kfMask,
            label: "a2v-", stage: 1, totalStages: 1)
        eval(vf)
        quiesceStreaming()
        dropDiTIfSequential()

        let vspatial = vf.reshaped(1, fLat, hLat, wLat, 128).transposed(0, 4, 1, 2, 3)
        try ensureDecoder()
        let pixels = try decodePixels(vspatial)
        eval(pixels)
        dropDecoder()
        return Output(video: pixels, audio: audioWaveform)
    }
}
