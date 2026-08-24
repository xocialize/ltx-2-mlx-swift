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

    /// SPEED-PLAN S3 ceiling probe (`LTX_STEP_DELTA=1`): logs the step-to-step cosine of the DiT
    /// input latent and of the x0 predictions. TeaCache-style step caching is only worth building
    /// if consecutive steps are highly redundant — the plan's kill rule: video cosine already
    /// < ~0.95 across the 8 distilled steps ⇒ do not build S3.
    static let logStepDeltas = ProcessInfo.processInfo.environment["LTX_STEP_DELTA"] == "1"

    static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let af = a.asType(.float32).flattened(), bf = b.asType(.float32).flattened()
        let n = (af * bf).sum()
        let d = MLX.sqrt((af * af).sum()) * MLX.sqrt((bf * bf).sum())
        return (n / d).item(Float.self)
    }

    /// X0Model: predict clean x0 from the DiT velocity. x0 = x_t − σ·v (fp32). When per-token
    /// `videoTimesteps`/`audioTimesteps` (B,N) are given, σ is per-token (B,N,1) and they also
    /// drive the DiT's per-token AdaLN path.
    ///
    /// `audioLatent == nil` ⇒ the audio-free forward (oracle `run_ax`); the audio x0 is nil too.
    /// `videoFrozen`/`audioFrozen` forward the oracle's frozen-modality rule
    /// (`utils/helpers.py:478`): a frozen modality's SCALAR timestep is zero, not the step sigma.
    /// The per-token timesteps are already zero through its all-zero denoise mask; this covers the
    /// two scalar consumers that mask cannot reach (that side's AV-cross gate + prompt AdaLN).
    static func x0(
        _ dit: any LTXDenoiser, videoLatent: MLXArray, audioLatent: MLXArray?, sigma: Float,
        videoText: MLXArray?, audioText: MLXArray?, videoPositions: MLXArray, audioPositions: MLXArray?,
        videoTimesteps: MLXArray? = nil, audioTimesteps: MLXArray? = nil,
        keyframesMask: MLXArray? = nil,
        videoFrozen: Bool = false, audioFrozen: Bool = false
    ) -> (MLXArray, MLXArray?) {
        let (vv, av) = dit(
            videoLatent: videoLatent, audioLatent: audioLatent, sigma: MLXArray([sigma]),
            videoText: videoText, audioText: audioText,
            videoPositions: videoPositions, audioPositions: audioPositions,
            videoTimesteps: videoTimesteps, audioTimesteps: audioTimesteps,
            keyframesMask: keyframesMask,
            videoSigma: videoFrozen ? MLXArray([Float(0)]) : nil,
            audioSigma: audioFrozen ? MLXArray([Float(0)]) : nil)
        let vSigma = videoTimesteps.map { $0.asType(.float32).expandedDimensions(axis: -1) }  // (B,N,1)
        let aSigma = audioTimesteps.map { $0.asType(.float32).expandedDimensions(axis: -1) }
        let vx0 = videoLatent.asType(.float32) - (vSigma ?? MLXArray(sigma)) * vv.asType(.float32)
        let ax0 = zip2(audioLatent, av).map { a, v in
            a.asType(.float32) - (aSigma ?? MLXArray(sigma)) * v.asType(.float32)
        }
        return (vx0, ax0)
    }

    /// Pair two optionals, non-nil only when both are.
    static func zip2(_ a: MLXArray?, _ b: MLXArray?) -> (MLXArray, MLXArray)? {
        guard let a, let b else { return nil }
        return (a, b)
    }

    /// Blend a predicted x0 with the clean conditioned latent. mask (B,N,1): 1 = denoise, 0 = keep.
    static func applyDenoiseMask(_ x0: MLXArray, clean: MLXArray, mask: MLXArray) -> MLXArray {
        x0 * mask + clean.asType(.float32) * (1.0 - mask)
    }

    /// Ancestral (SDE) Euler step in the RECTIFIED-FLOW parameterisation — LTX-2.5 stage 1.
    ///
    /// Exact port of `euler_ancestral_step` (oracle samplers.py), itself a port of upstream
    /// `EulerAncestralDiffusionStep.step`: a deterministic Euler step down to an intermediate
    /// `sigmaDown`, then a variance-preserving renoise up to `sigmaNext`.
    ///
    /// ⚠️ `alpha = 1 - sigma`, deliberately NOT the DDIM helper — rectified flow, not
    /// variance-preserving diffusion. Using the DDIM alpha here yields a plausible-looking
    /// trajectory that drifts.
    /// ⚠️ Math in fp32, result cast back to `x.dtype`.
    /// ⚠️ `eta == 0` reduces to the plain Euler step, but the variance-preserving rescale is
    /// still skipped only because the noise term vanishes — upstream gates the noise DRAW on
    /// eta alone, so keep the branch shape identical.
    public static func eulerAncestralStep(
        _ x: MLXArray, _ x0: MLXArray, sigma: Float, sigmaNext: Float,
        noise: MLXArray?, eta: Float = 1.0, sNoise: Float = 1.0
    ) -> MLXArray {
        if sigmaNext == 0 { return x0.asType(x.dtype) }
        let xf = x.asType(.float32), x0f = x0.asType(.float32)

        let downstepRatio = 1.0 + (sigmaNext / sigma - 1.0) * eta
        let sigmaDown = sigmaNext * downstepRatio
        let sigmaDownRatio = sigmaDown / sigma
        var xNext = sigmaDownRatio * xf + (1.0 - sigmaDownRatio) * x0f

        if eta > 0 {
            guard let noise else {
                fatalError("eulerAncestralStep requires a noise tensor when eta > 0")
            }
            let alphaNext = 1.0 - sigmaNext
            let alphaDown = 1.0 - sigmaDown
            let renoiseSq = sigmaNext * sigmaNext
                - sigmaDown * sigmaDown * alphaNext * alphaNext / (alphaDown * alphaDown)
            let renoiseCoeff = Foundation.sqrt(Swift.max(renoiseSq, 0.0))
            xNext = (alphaNext / alphaDown) * xNext + noise.asType(.float32) * sNoise * renoiseCoeff
        }
        return xNext.asType(x.dtype)
    }

    /// euler_step on an x0-prediction model. σ==0 → already clean.
    public static func eulerStep(_ x: MLXArray, _ x0: MLXArray, sigma: Float, sigmaNext: Float) -> MLXArray {
        if sigma == 0 { return x0 }
        return x + (sigmaNext - sigma) * (x - x0) / sigma
    }

    /// Distilled t2v Euler loop. `sigmas` includes the terminal 0.0; pairs are consecutive.
    /// Throws `CancellationError` between steps (MVP-READINESS M3) — a consumer Cancel stops the
    /// run at the next step boundary (≤ one step's latency) instead of after the full loop.
    /// Reports each step to `LTX2Progress` (V2 run-phase plane); `stage`/`totalStages` tag the
    /// two-stage passes (s1 = 1/2, s2 = 2/2) so a consumer can render "pass 2/2 · step 3/8".
    /// ⚠️ `keyframesMask` and `ancestralEta` are the LTX-2.5 deltas. They MUST be threaded
    /// from here, not just implemented on `DiT`/`eulerAncestralStep`: an earlier revision
    /// wired them only as far as `x0` and the gates, so `--dit-tiny-kf25-gate` and
    /// `--ancestral-step-gate` were green while every production run took the `nil` default
    /// and plain Euler. A defaulted-nil argument turns "forgot to wire it" into a
    /// compile-clean no-op.
    /// ⚠️ `audioLatent0 == nil` is the AUDIO-FREE loop (oracle `run_a`), used by the DFR temporal
    /// rounds — every tile there denoises video only. The ancestral key stream then SKIPS the audio
    /// split, so it advances exactly as a video-only stage's would; the audio-present stream is
    /// byte-identical to before.
    public static func run(
        dit: any LTXDenoiser, videoLatent0: MLXArray, audioLatent0: MLXArray?, sigmas: [Float],
        videoText: MLXArray?, audioText: MLXArray?, videoPositions: MLXArray, audioPositions: MLXArray?,
        keyframesMask: MLXArray? = nil,
        ancestralEta: Float = 0, ancestralSNoise: Float = 1, ancestralNoiseSeed: UInt64 = 0,
        label: String = "", stage: Int? = nil, totalStages: Int? = nil
    ) throws -> (video: MLXArray, audio: MLXArray?) {
        var ancestralKey = MLXRandom.key(ancestralNoiseSeed)
        var vx = videoLatent0, ax = audioLatent0
        let vN = vx.dim(1), aN = ax?.dim(1) ?? 0   // token counts (static shapes — no eval)
        var prevIn: MLXArray?, prevVX0: MLXArray?, prevAX0: MLXArray?
        for i in 0 ..< (sigmas.count - 1) {
            try Task.checkCancellation()
            LTX2Progress.report(.denoise, step: i + 1, totalSteps: sigmas.count - 1,
                                stage: stage, totalStages: totalStages)
            let span = MLXProfiler.shared.begin("denoise", "\(label)step\(i)",
                note: String(format: "vN=%d aN=%d σ=%.3f", vN, aN, sigmas[i]))
            let sigma = sigmas[i], sigmaNext = sigmas[i + 1]
            let vxIn = vx
            let (vx0, ax0) = x0(dit, videoLatent: vx, audioLatent: ax, sigma: sigma,
                                videoText: videoText, audioText: audioText,
                                videoPositions: videoPositions, audioPositions: audioPositions,
                                keyframesMask: keyframesMask)
            if logStepDeltas {
                if let pIn = prevIn, let pV = prevVX0 {
                    let aCos = zip2(prevAX0, ax0).map { cosine($0, $1) } ?? Float.nan
                    print(String(format: "[STEP-DELTA] %@step%d σ=%.3f  video in-cos=%.4f x0-cos=%.4f  audio x0-cos=%.4f",
                                 label, i, sigma, cosine(pIn, vxIn), cosine(pV, vx0), aCos))
                }
                prevIn = vxIn; prevVX0 = vx0; prevAX0 = ax0
            }
            if ancestralEta > 0 && sigmaNext != 0 {
                // Video noise is drawn before audio, matching the oracle's split order. The audio
                // split is SKIPPED (not merely unused) when there is no audio.
                let (k1, vKey) = MLXRandom.split(key: ancestralKey)
                ancestralKey = k1
                var aKey: MLXArray?
                if ax != nil {
                    let (k2, ak) = MLXRandom.split(key: k1)
                    ancestralKey = k2; aKey = ak
                }
                vx = eulerAncestralStep(vx, vx0, sigma: sigma, sigmaNext: sigmaNext,
                                        noise: MLXRandom.normal(vx.shape, key: vKey),
                                        eta: ancestralEta, sNoise: ancestralSNoise)
                if let a = ax, let a0 = ax0, let ak = aKey {
                    ax = eulerAncestralStep(a, a0, sigma: sigma, sigmaNext: sigmaNext,
                                            noise: MLXRandom.normal(a.shape, key: ak),
                                            eta: ancestralEta, sNoise: ancestralSNoise)
                }
            } else {
                vx = eulerStep(vx, vx0, sigma: sigma, sigmaNext: sigmaNext)
                if let a = ax, let a0 = ax0 { ax = eulerStep(a, a0, sigma: sigma, sigmaNext: sigmaNext) }
            }
            eval([vx, ax].compactMap { $0 })
            MLXProfiler.shared.end(span)
        }
        return (vx, ax)
    }

    /// Distilled i2v Euler loop with optional per-modality conditioning. A `denoiseMask` (B,N,1)
    /// with zeros at conditioned tokens drives the per-token timestep path; `cleanLatent` (B,N,C)
    /// is injected into the initial state and re-blended after each x0 prediction so those tokens
    /// stay exactly clean. Pass a mask ONLY when non-uniform (oracle `_is_uniform_mask`); nil ⇒
    /// that modality follows the scalar t2v path. Audio is usually unconditioned (nil) for i2v.
    /// ⚠️ `keyframesMask` and `ancestralEta` are the LTX-2.5 deltas. They MUST be threaded
    /// from here, not just implemented on `DiT`/`eulerAncestralStep`: an earlier revision
    /// wired them only as far as `x0` and the gates, so `--dit-tiny-kf25-gate` and
    /// `--ancestral-step-gate` were green while every production run took the `nil` default
    /// and plain Euler. A defaulted-nil argument turns "forgot to wire it" into a
    /// compile-clean no-op.
    /// ⚠️ `audioLatent0 == nil` is the AUDIO-FREE loop (oracle `run_a`) — see `run`.
    public static func runConditioned(
        dit: any LTXDenoiser, videoLatent0: MLXArray, audioLatent0: MLXArray?, sigmas: [Float],
        videoText: MLXArray?, audioText: MLXArray?, videoPositions: MLXArray, audioPositions: MLXArray?,
        videoCleanLatent: MLXArray? = nil, videoDenoiseMask: MLXArray? = nil,
        audioCleanLatent: MLXArray? = nil, audioDenoiseMask: MLXArray? = nil,
        keyframesMask: MLXArray? = nil,
        ancestralEta: Float = 0, ancestralSNoise: Float = 1, ancestralNoiseSeed: UInt64 = 0,
        label: String = "", stage: Int? = nil, totalStages: Int? = nil
    ) throws -> (video: MLXArray, audio: MLXArray?) {
        var ancestralKey = MLXRandom.key(ancestralNoiseSeed)
        var vx = videoLatent0.asType(.float32), ax = audioLatent0?.asType(.float32)
        // Inject clean conditioned latents into the initial noised state.
        if let clean = videoCleanLatent, let m = videoDenoiseMask { vx = applyDenoiseMask(vx, clean: clean, mask: m) }
        if let a = ax, let clean = audioCleanLatent, let m = audioDenoiseMask { ax = applyDenoiseMask(a, clean: clean, mask: m) }
        let vN = vx.dim(1), aN = ax?.dim(1) ?? 0
        // FROZEN-MODALITY detection, once (the masks don't change across steps). The oracle sets
        // `denoise_mask = zeros_like(...)` exactly when `ModalitySpec.frozen`, so an all-zero mask
        // IS the frozen flag — no new parameter needed, and every caller (retake's frozen
        // modality, i2v, icT2V) inherits the rule. A nil mask means "denoise everything" ⇒ live
        // sigma; a partially-zero mask (i2v conditioning) is not frozen either.
        func isFrozen(_ mask: MLXArray?) -> Bool {
            guard let mask else { return false }
            return MLX.max(MLX.abs(mask)).item(Float.self) == 0
        }
        let videoFrozen = isFrozen(videoDenoiseMask)
        let audioFrozen = ax == nil ? false : isFrozen(audioDenoiseMask)
        var prevIn: MLXArray?, prevVX0: MLXArray?, prevAX0: MLXArray?
        for i in 0 ..< (sigmas.count - 1) {
            try Task.checkCancellation()   // MVP-READINESS M3: per-step cancel point
            LTX2Progress.report(.denoise, step: i + 1, totalSteps: sigmas.count - 1,
                                stage: stage, totalStages: totalStages)
            let span = MLXProfiler.shared.begin("denoise", "\(label)step\(i)",
                note: String(format: "vN=%d aN=%d σ=%.3f", vN, aN, sigmas[i]))
            let sigma = sigmas[i], sigmaNext = sigmas[i + 1]
            let vxIn = vx
            let vts = videoDenoiseMask.map { ($0 * sigma).squeezed(axis: -1) }   // (B,N) per-token σ
            let ats = ax == nil ? nil : audioDenoiseMask.map { ($0 * sigma).squeezed(axis: -1) }
            var (vx0, ax0) = x0(dit, videoLatent: vx, audioLatent: ax, sigma: sigma,
                                videoText: videoText, audioText: audioText,
                                videoPositions: videoPositions, audioPositions: audioPositions,
                                videoTimesteps: vts, audioTimesteps: ats,
                                keyframesMask: keyframesMask,
                                videoFrozen: videoFrozen, audioFrozen: audioFrozen)
            if logStepDeltas {
                if let pIn = prevIn, let pV = prevVX0 {
                    let aCos = zip2(prevAX0, ax0).map { cosine($0, $1) } ?? Float.nan
                    print(String(format: "[STEP-DELTA] %@step%d σ=%.3f  video in-cos=%.4f x0-cos=%.4f  audio x0-cos=%.4f",
                                 label, i, sigma, cosine(pIn, vxIn), cosine(pV, vx0), aCos))
                }
                prevIn = vxIn; prevVX0 = vx0; prevAX0 = ax0
            }
            if let clean = videoCleanLatent, let m = videoDenoiseMask { vx0 = applyDenoiseMask(vx0, clean: clean, mask: m) }
            if let a0 = ax0, let clean = audioCleanLatent, let m = audioDenoiseMask { ax0 = applyDenoiseMask(a0, clean: clean, mask: m) }
            if ancestralEta > 0 && sigmaNext != 0 {
                // Video noise is drawn before audio, matching the oracle's split order. The audio
                // split is SKIPPED (not merely unused) when there is no audio.
                let (k1, vKey) = MLXRandom.split(key: ancestralKey)
                ancestralKey = k1
                var aKey: MLXArray?
                if ax != nil {
                    let (k2, ak) = MLXRandom.split(key: k1)
                    ancestralKey = k2; aKey = ak
                }
                vx = eulerAncestralStep(vx, vx0, sigma: sigma, sigmaNext: sigmaNext,
                                        noise: MLXRandom.normal(vx.shape, key: vKey),
                                        eta: ancestralEta, sNoise: ancestralSNoise)
                if let a = ax, let a0 = ax0, let ak = aKey {
                    ax = eulerAncestralStep(a, a0, sigma: sigma, sigmaNext: sigmaNext,
                                            noise: MLXRandom.normal(a.shape, key: ak),
                                            eta: ancestralEta, sNoise: ancestralSNoise)
                }
                // ⚠️ Re-apply conditioning AFTER the noise injection (oracle `post_process_latent`,
                // samplers.py:279-283). The renoise is added to EVERY token, conditioned ones
                // included, so without this a (1−strength)-masked anchor keyframe drifts off its
                // reference by exactly the injected noise. Unreachable before DFR — the two
                // `runConditioned` callers (i2v, icT2V) both run plain Euler — which is precisely
                // why it went missing.
                if let clean = videoCleanLatent, let m = videoDenoiseMask { vx = applyDenoiseMask(vx, clean: clean, mask: m) }
                if let a = ax, let clean = audioCleanLatent, let m = audioDenoiseMask { ax = applyDenoiseMask(a, clean: clean, mask: m) }
            } else {
                vx = eulerStep(vx, vx0, sigma: sigma, sigmaNext: sigmaNext)
                if let a = ax, let a0 = ax0 { ax = eulerStep(a, a0, sigma: sigma, sigmaNext: sigmaNext) }
            }
            eval([vx, ax].compactMap { $0 })
            MLXProfiler.shared.end(span)
        }
        return (vx, ax)
    }
}
