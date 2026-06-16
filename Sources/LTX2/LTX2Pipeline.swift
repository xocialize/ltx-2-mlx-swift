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

    init(gemma: GemmaEncoder, connector: Connector, dit: DiT, vae: VideoVAEDecoder) {
        self.gemma = gemma
        self.connector = connector
        self.dit = dit
        self.vae = vae
    }

    /// Load all components. `ltxDir` holds connector/transformer-distilled/vae_decoder
    /// safetensors; `gemmaDir` is the Gemma-3 MLX weights directory.
    public static func load(ltxDir: URL, gemmaDir: URL) async throws -> LTX2Pipeline {
        let gemma = try await GemmaEncoder.load(directory: gemmaDir)
        let connector = try Connector.load(connectorPath: ltxDir.appending(path: "connector.safetensors"))
        let dit = try DiT.load(weightsPath: ltxDir.appending(path: "transformer-distilled.safetensors"),
                               config: DiTConfig(), computeDtype: .bfloat16)
        let vae = try VideoVAEDecoder.load(path: ltxDir.appending(path: "vae_decoder.safetensors"))
        return LTX2Pipeline(gemma: gemma, connector: connector, dit: dit, vae: vae)
    }

    /// Text-to-video. Returns pixels (1, 3, F_out, H, W) in [-1, 1] (channels-first).
    /// `onStep` is invoked per denoise step for cancellation.
    public func t2v(
        prompt: String, height: Int = 256, width: Int = 256, numFrames: Int = 9,
        fps: Double = 24, seed: UInt64? = nil
    ) -> MLXArray {
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

        // 4. Distilled Euler denoise
        let (vfinal, _) = DenoiseLoop.run(
            dit: dit, videoLatent0: videoLatent, audioLatent0: audioLatent, sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: videoPositions, audioPositions: audioPositions)

        // 5. Unpatchify → (1, 128, F, H, W) → VAE decode → pixels
        let vspatial = vfinal.reshaped(1, fLat, hLat, wLat, 128).transposed(0, 4, 1, 2, 3)
        return vae.decode(vspatial)
    }
}
