// DenoiseLoop.swift — distilled Euler denoising (t2v + i2v conditioning).
//
// 1:1 port of ltx_pipelines_mlx/utils/samplers.denoise_loop for the distilled
// path: X0Model (x0 = x − σ·v) + euler_step (x + (σ_next−σ)(x−x0)/σ).
//  • t2v: uniform mask → scalar timestep, apply_denoise_mask is identity (`run`).
//  • i2v: a non-uniform denoise mask (conditioned tokens = 0) drives per-token
//    timesteps (σ_token = mask·σ) AND a per-step clean-latent re-blend so the
//    conditioned tokens stay exactly clean (`runConditioned`).
// CFG/STG/res2s + keyframe (frame_idx>0) conditioning are follow-ups.

import Foundation
import MLX
import MLXProfiling

public enum DenoiseLoop {

    /// X0Model: predict clean x0 from the DiT velocity. x0 = x_t − σ·v (fp32). When per-token
    /// `videoTimesteps`/`audioTimesteps` (B,N) are given, σ is per-token (B,N,1) and they also
    /// drive the DiT's per-token AdaLN path.
    static func x0(
        _ dit: DiT, videoLatent: MLXArray, audioLatent: MLXArray, sigma: Float,
        videoText: MLXArray?, audioText: MLXArray?, videoPositions: MLXArray, audioPositions: MLXArray,
        videoTimesteps: MLXArray? = nil, audioTimesteps: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let (vv, av) = dit(
            videoLatent: videoLatent, audioLatent: audioLatent, sigma: MLXArray([sigma]),
            videoText: videoText, audioText: audioText,
            videoPositions: videoPositions, audioPositions: audioPositions,
            videoTimesteps: videoTimesteps, audioTimesteps: audioTimesteps)
        let vSigma = videoTimesteps.map { $0.asType(.float32).expandedDimensions(axis: -1) }  // (B,N,1)
        let aSigma = audioTimesteps.map { $0.asType(.float32).expandedDimensions(axis: -1) }
        let vx0 = videoLatent.asType(.float32) - (vSigma ?? MLXArray(sigma)) * vv.asType(.float32)
        let ax0 = audioLatent.asType(.float32) - (aSigma ?? MLXArray(sigma)) * av.asType(.float32)
        return (vx0, ax0)
    }

    /// Blend a predicted x0 with the clean conditioned latent. mask (B,N,1): 1 = denoise, 0 = keep.
    static func applyDenoiseMask(_ x0: MLXArray, clean: MLXArray, mask: MLXArray) -> MLXArray {
        x0 * mask + clean.asType(.float32) * (1.0 - mask)
    }

    /// euler_step on an x0-prediction model. σ==0 → already clean.
    static func eulerStep(_ x: MLXArray, _ x0: MLXArray, sigma: Float, sigmaNext: Float) -> MLXArray {
        if sigma == 0 { return x0 }
        return x + (sigmaNext - sigma) * (x - x0) / sigma
    }

    /// Distilled t2v Euler loop. `sigmas` includes the terminal 0.0; pairs are consecutive.
    public static func run(
        dit: DiT, videoLatent0: MLXArray, audioLatent0: MLXArray, sigmas: [Float],
        videoText: MLXArray?, audioText: MLXArray?, videoPositions: MLXArray, audioPositions: MLXArray,
        label: String = ""
    ) -> (video: MLXArray, audio: MLXArray) {
        var vx = videoLatent0, ax = audioLatent0
        let vN = vx.dim(1), aN = ax.dim(1)   // token counts (static shapes — no eval)
        for i in 0 ..< (sigmas.count - 1) {
            let span = MLXProfiler.shared.begin("denoise", "\(label)step\(i)",
                note: String(format: "vN=%d aN=%d σ=%.3f", vN, aN, sigmas[i]))
            let sigma = sigmas[i], sigmaNext = sigmas[i + 1]
            let (vx0, ax0) = x0(dit, videoLatent: vx, audioLatent: ax, sigma: sigma,
                                videoText: videoText, audioText: audioText,
                                videoPositions: videoPositions, audioPositions: audioPositions)
            vx = eulerStep(vx, vx0, sigma: sigma, sigmaNext: sigmaNext)
            ax = eulerStep(ax, ax0, sigma: sigma, sigmaNext: sigmaNext)
            eval(vx, ax)
            MLXProfiler.shared.end(span)
        }
        return (vx, ax)
    }

    /// Distilled i2v Euler loop with optional per-modality conditioning. A `denoiseMask` (B,N,1)
    /// with zeros at conditioned tokens drives the per-token timestep path; `cleanLatent` (B,N,C)
    /// is injected into the initial state and re-blended after each x0 prediction so those tokens
    /// stay exactly clean. Pass a mask ONLY when non-uniform (oracle `_is_uniform_mask`); nil ⇒
    /// that modality follows the scalar t2v path. Audio is usually unconditioned (nil) for i2v.
    public static func runConditioned(
        dit: DiT, videoLatent0: MLXArray, audioLatent0: MLXArray, sigmas: [Float],
        videoText: MLXArray?, audioText: MLXArray?, videoPositions: MLXArray, audioPositions: MLXArray,
        videoCleanLatent: MLXArray? = nil, videoDenoiseMask: MLXArray? = nil,
        audioCleanLatent: MLXArray? = nil, audioDenoiseMask: MLXArray? = nil,
        label: String = ""
    ) -> (video: MLXArray, audio: MLXArray) {
        var vx = videoLatent0.asType(.float32), ax = audioLatent0.asType(.float32)
        // Inject clean conditioned latents into the initial noised state.
        if let clean = videoCleanLatent, let m = videoDenoiseMask { vx = applyDenoiseMask(vx, clean: clean, mask: m) }
        if let clean = audioCleanLatent, let m = audioDenoiseMask { ax = applyDenoiseMask(ax, clean: clean, mask: m) }
        let vN = vx.dim(1), aN = ax.dim(1)
        for i in 0 ..< (sigmas.count - 1) {
            let span = MLXProfiler.shared.begin("denoise", "\(label)step\(i)",
                note: String(format: "vN=%d aN=%d σ=%.3f", vN, aN, sigmas[i]))
            let sigma = sigmas[i], sigmaNext = sigmas[i + 1]
            let vts = videoDenoiseMask.map { ($0 * sigma).squeezed(axis: -1) }   // (B,N) per-token σ
            let ats = audioDenoiseMask.map { ($0 * sigma).squeezed(axis: -1) }
            var (vx0, ax0) = x0(dit, videoLatent: vx, audioLatent: ax, sigma: sigma,
                                videoText: videoText, audioText: audioText,
                                videoPositions: videoPositions, audioPositions: audioPositions,
                                videoTimesteps: vts, audioTimesteps: ats)
            if let clean = videoCleanLatent, let m = videoDenoiseMask { vx0 = applyDenoiseMask(vx0, clean: clean, mask: m) }
            if let clean = audioCleanLatent, let m = audioDenoiseMask { ax0 = applyDenoiseMask(ax0, clean: clean, mask: m) }
            vx = eulerStep(vx, vx0, sigma: sigma, sigmaNext: sigmaNext)
            ax = eulerStep(ax, ax0, sigma: sigma, sigmaNext: sigmaNext)
            eval(vx, ax)
            MLXProfiler.shared.end(span)
        }
        return (vx, ax)
    }
}
