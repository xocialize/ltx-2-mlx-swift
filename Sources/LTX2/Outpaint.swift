// Outpaint.swift — IC in/outpainting: a reference video whose fill region is blanked, an attention
// mask that hides that region's reference tokens from the generated ones, and the
// In-Outpainting IC-LoRA (Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting) doing the filling.
//
// Composition = the ltx-community outpaint Space (`LTX2InContextPipeline` + `conditioning_attention_mask`,
// 2026-10), the published reference usage for that adapter:
//   stage 1  CANVAS resolution, LoRA applied, the canvas-padded source as an IC reference (strength
//            1, downscale 1), the pixel mask's cross weights as a video self-attention bias, the
//            distilled sigmas with plain Euler (`icT2V`'s convention).
//   stage 2  (optional) x2 latent upsample → refine at 2× on the BARE distilled model: LoRA
//            suspended, no reference, no mask, stage-2 sigmas, init = noise·σ₀ + upscaled·(1−σ₀).
//
// ⚠️ MEASURED 2026-10-04 (anime, Wistoria cropback): the Space's stage 2 re-noises to σ₀ = 0.909 and
// the bare model then REINVENTS the frame — a different character, different set dressing — so the
// refined margins no longer continue the original once it is pasted back (ghosted pillar at the
// seam). `stage2Anchor` is OUR composition for that: the source-region tokens are pinned CLEAN to
// the VAE-encoded ORIGINAL at output resolution (denoise mask 0, per-token timestep 0, re-blended
// every step — the i2v mechanism over space instead of time) and only the margin tokens refine,
// attending to the true centre at full resolution. Same token count as the bare stage 2.
// The WHOLE canvas is generated — the source region included — and comes back decoded. Putting the
// original pixels back over the source region is the caller's job: a pixel-exact composite does not
// belong inside a latent pipeline whose VAE round-trip would soften it.
//
// Upstream ltx-core runs the same mask machinery from `ic_lora.py --conditioning-attention-mask`;
// the Space differs only in the additive conversion (see `ICAttentionMask.Convention`).

import Foundation
import MLX
import MLXProfiling
import MLXRandom

extension LTX2Pipeline {

    /// IC outpaint/inpaint. `reference` is the canvas-sized source with the fill region blanked
    /// (`encodeReference` at the canvas resolution). `referenceCrossMask` (Nr,) holds per-reference-
    /// token weights from `ICAttentionMask.latentCrossMask` — 1 where the source is real, 0 over the
    /// fill region; nil runs the adapter with no mask (the black-sentinel adapters' usage).
    /// `height`/`width` are the stage-1 CANVAS; with `twoStage` the output is 2× that.
    /// `stage2Anchor` (two-stage only): `pixels` (1,3,F,2H,2W) in [-1,1] — the ORIGINAL source at
    /// output resolution inside the output canvas — and `keep` (2H,2W) or (F,2H,2W) in [0,1], 1 over
    /// the source region. Latent cells the keep mask covers COMPLETELY are pinned to the encoded
    /// original; every other cell (margins, and any boundary cell the mask only partly covers) refines.
    public func icOutpaint(
        prompt: String, reference: ReferenceConditioning,
        referenceCrossMask: MLXArray?, convention: ICAttentionMask.Convention = .diffusers,
        attentionStrength: Float = 1.0,
        height: Int, width: Int, numFrames: Int, fps: Double = 24,
        seed: UInt64? = nil, twoStage: Bool = true,
        stage2Anchor: (pixels: MLXArray, keep: MLXArray)? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> Output {
        guard height % 32 == 0, width % 32 == 0 else {
            throw TwoStageError.badGeometry("outpaint canvas \(width)x\(height) must be divisible by 32")
        }
        let fLat = (numFrames + 7) / 8, hLat = height / 32, wLat = width / 32
        let nv = fLat * hLat * wLat
        let nr = reference.tokens.dim(1)
        // A reference on a different grid would be positioned by RoPE, not aligned token for token;
        // the mask weights are per reference token, so the two must describe the same grid.
        if let m = referenceCrossMask {
            guard m.ndim == 1, m.dim(0) == nr else {
                throw TwoStageError.badGeometry("cross mask \(m.shape) does not match \(nr) reference tokens")
            }
        }
        // Validate stage 2 BEFORE any heavy phase.
        let geo: TwoStageGeometry? = twoStage
            ? try resolveTwoStageGeometry(height: height * 2, width: width * 2, numFrames: numFrames, fps: fps)
            : nil
        if let geo {
            guard geo.variant == .spatialX2, geo.hLat1 == hLat, geo.wLat1 == wLat, geo.fLat1 == fLat else {
                throw TwoStageError.badGeometry(
                    "outpaint stage 2 needs the x2 spatial upsampler on the \(width)x\(height) canvas grid, got \(geo.variant.rawValue)")
            }
        }
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)
        MLXProfiler.shared.beginRun(String(format:
            "icOutpaint %dx%d %df fps=%.0f | nv=%d +ref=%d audioT=%d | mask=%@ | two-stage=%@",
            width, height, numFrames, fps, nv, nr, audioT,
            referenceCrossMask == nil ? "none" : convention.rawValue, twoStage ? "x2" : "off"))

        // LTX_OP_PHASES=1: wall clock per phase, read at the evals that already bound each phase
        // (no extra syncs, unlike MLX_PROFILE, which breaks fusion and inflates what it measures).
        let phases = ProcessInfo.processInfo.environment["LTX_OP_PHASES"] == "1"
        var phaseT = Date()
        func phase(_ name: String) {
            guard phases else { return }
            FileHandle.standardError.write(Data(String(format: "[op-phase] %-22@ %7.1fs  phys %.1f GB\n",
                name as NSString, Date().timeIntervalSince(phaseT), Double(physFootprintBytes()) / 1e9).utf8))
            phaseT = Date()
        }

        let (videoEmbeds, audioEmbeds) = try await encodePrompt(prompt)
        phase("encode-prompt")

        // --- Stage 1: masked IC at the canvas ---
        if let seed { MLXRandom.seed(seed) }
        let videoLatent = MLXRandom.normal([1, nv, 128])
        let audioLatent = MLXRandom.normal([1, audioT, 128])
        let state = ICVideoState.build(
            targetLatent: videoLatent,
            targetPositions: Positions.video(F: fLat, H: hLat, W: wLat, fps: Float(fps)),
            references: [reference])
        let kfMask = isLTX25
            ? Self.firstLatentFrameKeyframesMask(totalTokens: state.latent.dim(1),
                                                 tokensPerLatentFrame: hLat * wLat)
            : nil
        let dit = try ensureDiT()
        armStreamingGate(largestStageTokens: geo.map { $0.nv2 + audioT } ?? (nv + nr + audioT))
        let bias = referenceCrossMask.map {
            ICAttentionMask.selfAttentionBias(numNoisy: nv, referenceCrossMask: $0,
                                              strength: attentionStrength, convention: convention,
                                              dtype: .bfloat16)
        }
        let (vfull, afull) = try dit.withVideoSelfAttentionBias(bias) {
            try DenoiseLoop.runConditioned(
                dit: dit, videoLatent0: state.latent, audioLatent0: audioLatent,
                sigmas: Positions.distilledSigmas,
                videoText: videoEmbeds, audioText: audioEmbeds,
                videoPositions: state.positions, audioPositions: Positions.audio(tokens: audioT),
                videoCleanLatent: state.clean, videoDenoiseMask: state.denoiseMask,
                keyframesMask: kfMask, label: "op-s1-",
                stage: 1, totalStages: twoStage ? 2 : 1)
        }
        let v1 = state.slice(vfull)
        let a1 = afull!   // audio always supplied
        eval(v1, a1)
        phase("stage1-denoise")

        var vfinal = v1, afinal = a1
        var fOut = fLat, hOut = hLat, wOut = wLat
        if let geo {
            // --- x2 upsample (encoder stats + upsampler loaded only here) ---
            LTX2Progress.report(.upsample)
            try ensureVAEEncoder(); try ensureUpsampler()
            let v1spatial = v1.reshaped(1, fLat, hLat, wLat, 128).transposed(0, 4, 1, 2, 3)
            let upscaled = vaeEncoder!.normalizeLatent(upsampler!(vaeEncoder!.denormalizeLatent(v1spatial)))
            eval(upscaled)
            dropUpscaler()
            phase("upsample")
            guard upscaled.dim(2) == geo.fLat2, upscaled.dim(3) == geo.hLat2, upscaled.dim(4) == geo.wLat2 else {
                throw TwoStageError.badGeometry("upsampler produced \(upscaled.shape), expected grid "
                    + "(\(geo.fLat2), \(geo.hLat2), \(geo.wLat2))")
            }
            let v2tokens = LTX2Pipeline.patchify(upscaled)
            // --- Stage 2: bare distilled refine at 2× (no adapter, no reference, no mask) ---
            let s2 = Positions.stage2Sigmas
            let v2init = LTX2Pipeline.noiseInit(clean: v2tokens, sigma: s2[0], shape: v2tokens.shape,
                                                seed: seed.map { $0 &+ 2 })
            let a2init = LTX2Pipeline.noiseInit(clean: a1, sigma: s2[0], shape: a1.shape,
                                                seed: seed.map { $0 &+ 2 })
            let kfMask2 = isLTX25
                ? Self.firstLatentFrameKeyframesMask(totalTokens: v2init.dim(1),
                                                     tokensPerLatentFrame: geo.hLat2 * geo.wLat2)
                : nil
            let vPos2 = Positions.video(F: geo.fLat2, H: geo.hLat2, W: geo.wLat2, fps: Float(fps))
            // Anchored stage 2: encode the original at output resolution, pin fully-covered cells.
            var anchorClean: MLXArray? = nil, anchorMask: MLXArray? = nil
            if let anchor = stage2Anchor {
                let f2 = numFrames, h2 = height * 2, w2 = width * 2
                guard anchor.pixels.shape == [1, 3, f2, h2, w2] else {
                    throw TwoStageError.badGeometry("stage-2 anchor \(anchor.pixels.shape) must be [1, 3, \(f2), \(h2), \(w2)]")
                }
                var keep = anchor.keep
                if keep.ndim == 2 { keep = broadcast(keep.expandedDimensions(axis: 0), to: [f2, h2, w2]) }
                guard keep.shape == [f2, h2, w2] else {
                    throw TwoStageError.badGeometry("stage-2 keep mask \(anchor.keep.shape) must be (H,W) or (F,H,W) at \(w2)x\(h2)")
                }
                // The anchor's MARGINS are stage 1's own output, decoded and 2× upsampled. The VAE
                // encoder's receptive field reaches across the boundary into the pinned edge cells, and
                // whatever fills the margins there is what stage 2 continues: a black fill came back as
                // a dark band, a MIRRORED fill as mirrored faces (both measured 2026-10-04). The content
                // stage 2 is refining anyway is the one fill that adds nothing of its own.
                try ensureDecoder()
                let s1px = try decodePixels(v1spatial).asType(.float32)          // (1,3,F,H,W) in [-1,1]
                dropDecoder()
                phase("anchor-s1-decode")
                let k5 = keep.reshaped(1, 1, f2, h2, w2).asType(.float32)
                let composed = Self.upsample2x(s1px) * (1 - k5) + anchor.pixels.asType(.float32) * k5
                eval(composed)
                try ensureVAEEncoder()
                let lat = vaeEncoder!.encode(composed)                                // normalized, (1,128,F2,h2,w2)
                anchorClean = LTX2Pipeline.patchify(lat)
                let cover = ICAttentionMask.latentCrossMask(pixelMask: keep, latentFrames: geo.fLat2,
                                                            latentHeight: geo.hLat2, latentWidth: geo.wLat2)
                // pinned (0) only where the keep mask covers the whole cell; refine (1) elsewhere
                anchorMask = MLX.which(MLX.greaterEqual(cover, MLXArray(Float(1))), MLXArray(Float(0)), MLXArray(Float(1)))
                    .reshaped(1, -1, 1)
                eval(anchorClean!, anchorMask!)
                dropUpscaler()
                phase("anchor-encode")
            }
            let (v2, a2) = try withLoRAsSuspended {
                if let clean = anchorClean, let mask = anchorMask {
                    return try DenoiseLoop.runConditioned(
                        dit: try ensureDiT(), videoLatent0: v2init, audioLatent0: a2init, sigmas: s2,
                        videoText: videoEmbeds, audioText: audioEmbeds,
                        videoPositions: vPos2, audioPositions: Positions.audio(tokens: audioT),
                        videoCleanLatent: clean, videoDenoiseMask: mask,
                        keyframesMask: kfMask2, label: "op-s2a-", stage: 2, totalStages: 2)
                }
                return try DenoiseLoop.run(
                    dit: try ensureDiT(), videoLatent0: v2init, audioLatent0: a2init, sigmas: s2,
                    videoText: videoEmbeds, audioText: audioEmbeds,
                    videoPositions: vPos2, audioPositions: Positions.audio(tokens: audioT),
                    keyframesMask: kfMask2, label: "op-s2-", stage: 2, totalStages: 2)
            }
            eval(v2, a2!)
            phase("stage2-denoise")
            vfinal = v2; afinal = a2!
            fOut = geo.fLat2; hOut = geo.hLat2; wOut = geo.wLat2
        }
        quiesceStreaming()
        dropDiTIfSequential()

        let vspatial = vfinal.reshaped(1, fOut, hOut, wOut, 128).transposed(0, 4, 1, 2, 3)
        try ensureDecoder()
        let waveform = decodeAudio(afinal)
        if let waveform { eval(waveform) }
        let pixels = try decodePixels(vspatial)
        eval(pixels)
        dropDecoder()
        phase("final-decode")
        MLXProfiler.shared.endRun()
        return Output(video: pixels, audio: waveform)
    }

    /// 2× bilinear on the last two axes of a 5-D (B,C,F,H,W) tensor — half-pixel centres, edge
    /// clamped (PyTorch `align_corners=False`). Only feeds the stage-2 anchor's MARGINS, which are
    /// refined, never pinned.
    public static func upsample2x(_ x: MLXArray) -> MLXArray {
        func lastAxis(_ x: MLXArray) -> MLXArray {
            let n = x.dim(4)
            let prev = concatenated([x[0..., 0..., 0..., 0..., 0 ..< 1], x[0..., 0..., 0..., 0..., 0 ..< (n - 1)]], axis: 4)
            let next = concatenated([x[0..., 0..., 0..., 0..., 1 ..< n], x[0..., 0..., 0..., 0..., (n - 1) ..< n]], axis: 4)
            var shape = x.shape
            shape[4] = n * 2
            return stacked([0.75 * x + 0.25 * prev, 0.75 * x + 0.25 * next], axis: 5).reshaped(shape)
        }
        return lastAxis(lastAxis(x).swappedAxes(3, 4)).swappedAxes(3, 4)
    }
}
