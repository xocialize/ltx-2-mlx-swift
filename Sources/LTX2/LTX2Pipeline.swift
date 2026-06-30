// LTX2Pipeline.swift — one-stage distilled t2v: prompt → frames.
//
// Assembles the parity-validated pieces: GemmaEncoder (49-state extract) →
// Connector (video/audio embeds) → noised latents → DenoiseLoop (distilled
// Euler) → unpatchify → VideoVAEDecoder → pixels. Video-only for now (audio
// latent is denoised jointly but not decoded — Audio VAE/vocoder are a follow-up).
// Single-stage at target resolution (no two-stage upsampler yet).

import Foundation
import MLX
import MLXRandom

public final class LTX2Pipeline {
    let gemma: GemmaEncoder
    let connector: Connector
    let dit: DiT
    let vae: VideoVAEDecoder
    let audioVAE: AudioVAEDecoder?
    let vocoder: Vocoder?
    let vaeEncoder: VideoVAEEncoder?   // for two-stage upscale denorm/renorm stats
    let upsampler: Upsampler?          // spatial_x2

    init(gemma: GemmaEncoder, connector: Connector, dit: DiT, vae: VideoVAEDecoder,
         audioVAE: AudioVAEDecoder?, vocoder: Vocoder?,
         vaeEncoder: VideoVAEEncoder?, upsampler: Upsampler?) {
        self.gemma = gemma
        self.connector = connector
        self.dit = dit
        self.vae = vae
        self.audioVAE = audioVAE
        self.vocoder = vocoder
        self.vaeEncoder = vaeEncoder
        self.upsampler = upsampler
    }

    /// True when the two-stage (half-res → upsample → refine) path is available.
    public var supportsTwoStage: Bool { vaeEncoder != nil && upsampler != nil }

    // MARK: - Runtime LoRA (extend, not swap) — public seam for the engine wrapper

    /// Make `loras` the active runtime-LoRA set on the resident DiT (replaces any current set;
    /// empty array detaches). Hot-swap: no base reload, only the small low-rank factors change.
    public func setLoRAs(_ loras: [(url: URL, strength: Float)]) throws {
        try LTX2LoRA.apply(loras, to: dit)
    }

    /// Restore the pristine base (drop all runtime LoRAs).
    public func clearLoRAs() { LTX2LoRA.detach(dit) }

    /// Number of currently-adapted DiT targets (0 = pristine base).
    public var activeLoRATargets: Int { dit.loraTargetCount }

    /// Output of a t2v run: video pixels (1,3,F,H,W) and optional 48kHz stereo audio (1,2,T).
    public struct Output {
        public let video: MLXArray
        public let audio: MLXArray?
    }

    /// Load all components. `ltxDir` holds connector/transformer-distilled/vae_decoder
    /// (+ optional audio_vae/vocoder) safetensors; `gemmaDir` is the Gemma-3 weights dir.
    /// Audio decode is enabled when both audio_vae.safetensors and vocoder.safetensors exist.
    public static func load(ltxDir: URL, gemmaDir: URL, transformerPath: URL? = nil) async throws -> LTX2Pipeline {
        let gemma = try await GemmaEncoder.load(directory: gemmaDir)
        let connector = try Connector.load(connectorPath: ltxDir.appending(path: "connector.safetensors"))
        // transformerPath override → quantized checkpoint (q8/q4); DiT auto-detects quant.
        let ditPath = transformerPath ?? ltxDir.appending(path: "transformer-distilled.safetensors")
        let dit = try DiT.load(weightsPath: ditPath, config: DiTConfig(), computeDtype: .bfloat16)
        let vae = try VideoVAEDecoder.load(path: ltxDir.appending(path: "vae_decoder.safetensors"))
        let audioPath = ltxDir.appending(path: "audio_vae.safetensors")
        let vocPath = ltxDir.appending(path: "vocoder.safetensors")
        let fm = FileManager.default
        let audioVAE = fm.fileExists(atPath: audioPath.path) ? try AudioVAEDecoder.load(path: audioPath) : nil
        let vocoder = fm.fileExists(atPath: vocPath.path) ? try Vocoder.load(path: vocPath) : nil
        let encPath = ltxDir.appending(path: "vae_encoder.safetensors")
        let upPath = ltxDir.appending(path: "spatial_upscaler_x2_v1_1.safetensors")
        let vaeEncoder = fm.fileExists(atPath: encPath.path) ? try VideoVAEEncoder.load(path: encPath) : nil
        let upsampler = fm.fileExists(atPath: upPath.path) ? try Upsampler.load(path: upPath) : nil
        return LTX2Pipeline(gemma: gemma, connector: connector, dit: dit, vae: vae,
                            audioVAE: audioVAE, vocoder: vocoder,
                            vaeEncoder: vaeEncoder, upsampler: upsampler)
    }

    /// Flow-matching noised init: noise·σ + clean·(1-σ). Stage-1 (clean=0,σ=1)=noise;
    /// stage-2 starts from the upscaled latent at σ₀.
    static func noiseInit(clean: MLXArray, sigma: Float, shape: [Int], seed: UInt64?) -> MLXArray {
        if let seed { MLXRandom.seed(seed) }
        let noise = MLXRandom.normal(shape)
        return noise * sigma + clean * (1.0 - sigma)
    }

    /// Re-patchify a (1,128,F,H,W) latent → tokens (1, F·H·W, 128).
    static func patchify(_ latent: MLXArray) -> MLXArray {
        let B = latent.dim(0), C = latent.dim(1), F = latent.dim(2), H = latent.dim(3), W = latent.dim(4)
        return latent.transposed(0, 2, 3, 4, 1).reshaped(B, F * H * W, C)
    }

    /// Decode a denoised audio latent (1, T, 128) → 48kHz stereo waveform (1, 2, T_48k).
    /// AudioPatchifier.unpatchify: (B,T,128) → (B,8,T,16), then Audio VAE → mel → vocoder+BWE.
    func decodeAudio(_ audioTokens: MLXArray) -> MLXArray? {
        guard let audioVAE, let vocoder else { return nil }
        let B = audioTokens.dim(0), T = audioTokens.dim(1)
        let audioLatent = audioTokens.reshaped(B, T, 8, 16).transposed(0, 2, 1, 3)  // (B,8,T,16)
        let mel = audioVAE.decode(audioLatent)                                       // (B,2,T',64)
        return vocoder(mel)                                                          // (B,2,T_48k)
    }

    /// Text-to-video(+audio). Returns video pixels (1,3,F,H,W) in [-1,1] (channels-first)
    /// and optional 48kHz stereo audio.
    public func t2v(
        prompt: String, height: Int = 256, width: Int = 256, numFrames: Int = 9,
        fps: Double = 24, seed: UInt64? = nil
    ) -> Output {
        // 1. Text encode (Gemma → connector)
        let (ids, mask) = gemma.tokenize(prompt)
        let states = gemma.allHiddenStates(tokenIds: ids, attentionMask: mask)
        let (videoEmbeds, audioEmbeds) = connector(hiddenStates: states, mask: mask)

        // 2. Latent geometry
        let fLat = (numFrames + 7) / 8, hLat = height / 32, wLat = width / 32
        let nv = fLat * hLat * wLat
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)

        // 3. Noised init (t2v starts from pure noise at σ_max)
        if let seed { MLXRandom.seed(seed) }
        let videoLatent = MLXRandom.normal([1, nv, 128])
        let audioLatent = MLXRandom.normal([1, audioT, 128])
        let videoPositions = Positions.video(F: fLat, H: hLat, W: wLat, fps: Float(fps))
        let audioPositions = Positions.audio(tokens: audioT)

        // 4. Distilled Euler denoise (joint video + audio)
        let (vfinal, afinal) = DenoiseLoop.run(
            dit: dit, videoLatent0: videoLatent, audioLatent0: audioLatent, sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: videoPositions, audioPositions: audioPositions)

        // 5. Video: unpatchify → (1, 128, F, H, W) → VAE decode → pixels
        let vspatial = vfinal.reshaped(1, fLat, hLat, wLat, 128).transposed(0, 4, 1, 2, 3)
        let pixels = vae.decode(vspatial)

        // 6. Audio: decode the jointly-denoised audio latent (if audio components loaded)
        let waveform = decodeAudio(afinal)
        return Output(video: pixels, audio: waveform)
    }

    /// Image-to-video (first-frame conditioning, frame_idx=0). `initFrame` is (1,3,1,H,W) in
    /// [-1,1] at the target resolution. VAE-encodes it to the frame-0 latent and holds those
    /// tokens clean — per-token timestep 0 in the DiT + a per-step clean re-blend — while the
    /// rest denoise from noise. `strength` 1.0 = fully condition on the frame; <1.0 lets the
    /// frame be partially re-noised (denoise_mask = 1-strength). One-stage distilled; requires
    /// the VAE encoder (falls back to t2v if absent). Keyframe (frame_idx>0) + two-stage i2v
    /// are follow-ups.
    public func i2v(
        prompt: String, initFrame: MLXArray, height: Int = 512, width: Int = 704, numFrames: Int = 9,
        fps: Double = 24, seed: UInt64? = nil, strength: Float = 1.0
    ) -> Output {
        guard let vaeEncoder else {
            return t2v(prompt: prompt, height: height, width: width, numFrames: numFrames, fps: fps, seed: seed)
        }
        // 1. Text encode
        let (ids, mask) = gemma.tokenize(prompt)
        let states = gemma.allHiddenStates(tokenIds: ids, attentionMask: mask)
        let (videoEmbeds, audioEmbeds) = connector(hiddenStates: states, mask: mask)

        // 2. Latent geometry
        let fLat = (numFrames + 7) / 8, hLat = height / 32, wLat = width / 32
        let frame0 = hLat * wLat, nv = fLat * frame0
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)

        // 3. Encode the init frame → frame-0 clean latent tokens (1, frame0, 128), normalized.
        let refLatent = vaeEncoder.encode(initFrame)            // (1,128,1,hLat,wLat)
        let refTokens = LTX2Pipeline.patchify(refLatent)        // (1, frame0, 128)

        // 4. Full-size clean latent (frame 0 = ref, rest 0) + denoise mask (1-strength at frame 0).
        let cleanVideo = MLX.concatenated([refTokens, MLXArray.zeros([1, nv - frame0, 128])], axis: 1)
        let maskHead = MLXArray.zeros([1, frame0, 1]) + (1.0 - strength)   // conditioned tokens
        let videoMask = MLX.concatenated([maskHead, MLXArray.ones([1, nv - frame0, 1])], axis: 1)

        // 5. Noised init + conditioned denoise (audio is unconditioned → scalar path).
        if let seed { MLXRandom.seed(seed) }
        let videoLatent = MLXRandom.normal([1, nv, 128])
        let audioLatent = MLXRandom.normal([1, audioT, 128])
        let (vfinal, afinal) = DenoiseLoop.runConditioned(
            dit: dit, videoLatent0: videoLatent, audioLatent0: audioLatent, sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: Positions.video(F: fLat, H: hLat, W: wLat, fps: Float(fps)),
            audioPositions: Positions.audio(tokens: audioT),
            videoCleanLatent: cleanVideo, videoDenoiseMask: videoMask)

        // 6. Decode
        let vspatial = vfinal.reshaped(1, fLat, hLat, wLat, 128).transposed(0, 4, 1, 2, 3)
        return Output(video: vae.decode(vspatial), audio: decodeAudio(afinal))
    }

    /// Two-stage distilled t2v (the `generate --distilled` flow): stage-1 denoise at
    /// HALF resolution → unpatchify → encoder-stats denorm → upsampler 2× → renorm →
    /// re-patchify → stage-2 refine at full resolution. Requires `supportsTwoStage`.
    public func t2vTwoStage(
        prompt: String, height: Int = 512, width: Int = 704, numFrames: Int = 9,
        fps: Double = 24, seed: UInt64? = nil
    ) -> Output {
        guard let vaeEncoder, let upsampler else { return t2v(prompt: prompt, height: height, width: width, numFrames: numFrames, fps: fps, seed: seed) }

        let (ids, mask) = gemma.tokenize(prompt)
        let states = gemma.allHiddenStates(tokenIds: ids, attentionMask: mask)
        let (videoEmbeds, audioEmbeds) = connector(hiddenStates: states, mask: mask)

        let fLat = (numFrames + 7) / 8
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)
        let s2 = Positions.stage2Sigmas
        let sigma0 = s2[0]

        // --- Stage 1: half resolution ---
        let hHalf = height / 2, wHalf = width / 2
        let hLat1 = hHalf / 32, wLat1 = wHalf / 32, nv1 = fLat * hLat1 * wLat1
        let v1 = LTX2Pipeline.noiseInit(clean: MLXArray.zeros([1, nv1, 128]), sigma: 1.0, shape: [1, nv1, 128], seed: seed)
        let a1 = LTX2Pipeline.noiseInit(clean: MLXArray.zeros([1, audioT, 128]), sigma: 1.0, shape: [1, audioT, 128], seed: seed.map { $0 &+ 1 })
        let (v1f, a1f) = DenoiseLoop.run(
            dit: dit, videoLatent0: v1, audioLatent0: a1, sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: Positions.video(F: fLat, H: hLat1, W: wLat1, fps: Float(fps)),
            audioPositions: Positions.audio(tokens: audioT))

        // --- Upscale 2× in un-normalized latent space ---
        let v1spatial = v1f.reshaped(1, fLat, hLat1, wLat1, 128).transposed(0, 4, 1, 2, 3)  // (1,128,F,h1,w1)
        let upscaled = vaeEncoder.normalizeLatent(upsampler(vaeEncoder.denormalizeLatent(v1spatial)))  // (1,128,F,2h1,2w1)
        eval(upscaled)
        let hLat2 = hLat1 * 2, wLat2 = wLat1 * 2
        let v2tokens = LTX2Pipeline.patchify(upscaled)  // (1, F*2h1*2w1, 128)

        // --- Stage 2: full resolution refine (init = noise·σ₀ + upscaled·(1-σ₀)) ---
        let v2init = LTX2Pipeline.noiseInit(clean: v2tokens, sigma: sigma0, shape: v2tokens.shape, seed: seed.map { $0 &+ 2 })
        let a2init = LTX2Pipeline.noiseInit(clean: a1f, sigma: sigma0, shape: a1f.shape, seed: seed.map { $0 &+ 2 })
        let (v2f, a2f) = DenoiseLoop.run(
            dit: dit, videoLatent0: v2init, audioLatent0: a2init, sigmas: s2,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: Positions.video(F: fLat, H: hLat2, W: wLat2, fps: Float(fps)),
            audioPositions: Positions.audio(tokens: audioT))

        let vspatial = v2f.reshaped(1, fLat, hLat2, wLat2, 128).transposed(0, 4, 1, 2, 3)
        return Output(video: vae.decode(vspatial), audio: decodeAudio(a2f))
    }
}
