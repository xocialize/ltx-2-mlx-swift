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

    init(gemma: GemmaEncoder, connector: Connector, dit: DiT, vae: VideoVAEDecoder,
         audioVAE: AudioVAEDecoder?, vocoder: Vocoder?) {
        self.gemma = gemma
        self.connector = connector
        self.dit = dit
        self.vae = vae
        self.audioVAE = audioVAE
        self.vocoder = vocoder
    }

    /// Output of a t2v run: video pixels (1,3,F,H,W) and optional 48kHz stereo audio (1,2,T).
    public struct Output {
        public let video: MLXArray
        public let audio: MLXArray?
    }

    /// Load all components. `ltxDir` holds connector/transformer-distilled/vae_decoder
    /// (+ optional audio_vae/vocoder) safetensors; `gemmaDir` is the Gemma-3 weights dir.
    /// Audio decode is enabled when both audio_vae.safetensors and vocoder.safetensors exist.
    public static func load(ltxDir: URL, gemmaDir: URL) async throws -> LTX2Pipeline {
        let gemma = try await GemmaEncoder.load(directory: gemmaDir)
        let connector = try Connector.load(connectorPath: ltxDir.appending(path: "connector.safetensors"))
        let dit = try DiT.load(weightsPath: ltxDir.appending(path: "transformer-distilled.safetensors"),
                               config: DiTConfig(), computeDtype: .bfloat16)
        let vae = try VideoVAEDecoder.load(path: ltxDir.appending(path: "vae_decoder.safetensors"))
        let audioPath = ltxDir.appending(path: "audio_vae.safetensors")
        let vocPath = ltxDir.appending(path: "vocoder.safetensors")
        let fm = FileManager.default
        let audioVAE = fm.fileExists(atPath: audioPath.path) ? try AudioVAEDecoder.load(path: audioPath) : nil
        let vocoder = fm.fileExists(atPath: vocPath.path) ? try Vocoder.load(path: vocPath) : nil
        return LTX2Pipeline(gemma: gemma, connector: connector, dit: dit, vae: vae,
                            audioVAE: audioVAE, vocoder: vocoder)
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
}
