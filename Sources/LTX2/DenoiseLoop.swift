// DenoiseLoop.swift — distilled Euler denoising (t2v, uniform mask, no CFG).
//
// 1:1 port of ltx_pipelines_mlx/utils/samplers.denoise_loop for the distilled
// path: X0Model (x0 = x − σ·v) + euler_step (x + (σ_next−σ)(x−x0)/σ). Uniform
// denoise mask (full t2v) → apply_denoise_mask is identity, no per-token
// timesteps. Conditioning masks (i2v/retake) + CFG/STG/res2s are follow-ups.

import Foundation
import MLX

public enum DenoiseLoop {

    /// X0Model: predict clean x0 from the DiT velocity. x0 = x_t − σ·v (fp32).
    static func x0(
        _ dit: DiT, videoLatent: MLXArray, audioLatent: MLXArray, sigma: Float,
        videoText: MLXArray?, audioText: MLXArray?, videoPositions: MLXArray, audioPositions: MLXArray
    ) -> (MLXArray, MLXArray) {
        let (vv, av) = dit(
            videoLatent: videoLatent, audioLatent: audioLatent, sigma: MLXArray([sigma]),
            videoText: videoText, audioText: audioText,
            videoPositions: videoPositions, audioPositions: audioPositions)
        let vx0 = videoLatent.asType(.float32) - sigma * vv.asType(.float32)
        let ax0 = audioLatent.asType(.float32) - sigma * av.asType(.float32)
        return (vx0, ax0)
    }

    /// euler_step on an x0-prediction model. σ==0 → already clean.
    static func eulerStep(_ x: MLXArray, _ x0: MLXArray, sigma: Float, sigmaNext: Float) -> MLXArray {
        if sigma == 0 { return x0 }
        return x + (sigmaNext - sigma) * (x - x0) / sigma
    }

    /// Distilled t2v Euler loop. `sigmas` includes the terminal 0.0; pairs are consecutive.
    public static func run(
        dit: DiT, videoLatent0: MLXArray, audioLatent0: MLXArray, sigmas: [Float],
        videoText: MLXArray?, audioText: MLXArray?, videoPositions: MLXArray, audioPositions: MLXArray
    ) -> (video: MLXArray, audio: MLXArray) {
        var vx = videoLatent0, ax = audioLatent0
        for i in 0 ..< (sigmas.count - 1) {
            let sigma = sigmas[i], sigmaNext = sigmas[i + 1]
            let (vx0, ax0) = x0(dit, videoLatent: vx, audioLatent: ax, sigma: sigma,
                                videoText: videoText, audioText: audioText,
                                videoPositions: videoPositions, audioPositions: audioPositions)
            vx = eulerStep(vx, vx0, sigma: sigma, sigmaNext: sigmaNext)
            ax = eulerStep(ax, ax0, sigma: sigma, sigmaNext: sigmaNext)
            eval(vx, ax)
        }
        return (vx, ax)
    }
}
