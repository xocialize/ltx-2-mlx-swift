// DFRPipeline.swift — Diffusion Fidelity Rendering: keyframe-slot base, spatial detailing,
// tiled temporal rounds.
//
// Swift sibling of `ltx_pipelines_mlx/dfr.py`, itself a port of upstream `dfr_pipeline.py`
// (LTX-2.5). This is the pipeline the "native multishot generation" and "diffusion fidelity
// rendering" claims describe. The canvas geometry lives in `DFRLayout` (gated bit-exactly by
// `--dfr-layout-gate`); this file is the orchestration over it, and `--dfr-gate` covers that.
//
// Shape of the run:
//
//   Stage 1  half-res base + generated keyframe SLOTS on an x8-border segment grid.
//            The half-res result is reserved as the detailing reference.
//   Stage 2  full res. Video and slots are latent-upsampled; slots are re-appended SEEDED
//            with their upsampled stage-1 content (not re-rolled from noise) so the detailing
//            pass refines them instead of inventing new ones.
//   Rounds   optional temporal x2 densification, `2**round` keyframe-seam tiles per round, each
//            tile inventing mid-segment slots and denoising with ancestral Euler, then stitched
//            on the seam latents. Tiles denoise VIDEO-ONLY (`audioLatent: nil`), matching
//            upstream — audio comes from stage 2 and nothing after refines it.
//
// ⚠️ Rounds are a QUALITY/parity feature, NOT an optimization. The Python measurement has them
// costing x2.767 (r=1) / x3.542 (r=2) versus native generation at the same delivered frame count,
// and an operator judged NATIVE generation better (ltx-2-mlx/docs/claims/C2-dfr.md). Do not
// present or default them as a speed lever.
//
// ⚠️ Conditioning fps is capped at `maxConditioningFPS` INDEPENDENTLY of playback fps. RoPE time
// is `pixelFrame / fps`, so a 120 fps time base halves every token's temporal span versus the
// trained distribution and the model can no longer lay out 8 pixel frames inside one latent token
// — it decodes as a motion spike at each latent border followed by a stall. Playback fps is used
// for DECODING only, and is `fps * 2**rounds`: muxing the delivered frames at the base rate
// writes a 2x/4x slow-motion clip with the right pictures (a real bug, shipped and fixed).

import Foundation
import MLX
import MLXProfiling
import MLXRandom

/// Anchor keyframes carried between temporal rounds, pinned just short of fully clean so a tile
/// can still settle its seam frame. Ours (upstream has no equivalent knob).
private let anchorKeyframeStrength: Float = 0.95
private let temporalAncestralEta: Float = 0.5
/// Internal, not private: `DFRPlan` applies the cap and `--dfr-gate` asserts it.
let maxConditioningFPS: Double = 60.0

/// What the canvas resolver decided, reported so a caller can explain a frame count it did not ask
/// for (`resolveCanvas` pads `numFrames - 1` up to a whole number of segments).
public struct DFRCanvas: Equatable {
    public let segment: Int
    public let canvasFrames: Int
    public let requestedFrames: Int
    public let slotPositions: [Int]
}

/// DFR output in LATENT form — `LTX2Pipeline.dfr` decodes it. Split like the oracle's
/// `generate_dfr` / `generate_and_save` so a harness can assert the orchestration without paying
/// for a decode.
public struct DFRLatents {
    /// (1, 128, T, H, W), already trimmed to the delivered clip.
    public let video: MLXArray
    /// (1, T_a, 128) TOKEN form (what `decodeAudio` consumes), trimmed to the delivered clip.
    public let audio: MLXArray?
    /// ⚠️ The rate to DECODE and MUX at: `fps * 2**rounds`. Not the conditioning fps.
    public let playbackFPS: Double
    /// Conditioning fps actually used by the last round (capped at `maxConditioningFPS`).
    public let conditioningFPS: Double
    /// Pixel-frame indices of the generated keyframes INSIDE the delivered clip.
    public let generatedKeyframePositions: [Int]
    /// One (1, 128, 1, H, W) latent per reported position — decode these one at a time.
    public let generatedKeyframeLatents: [MLXArray]
    public let canvas: DFRCanvas
    /// Delivered pixel frames: `(requested - 1) * 2**rounds + 1`.
    public let deliveredFrames: Int
    public let latentHeight: Int
    public let latentWidth: Int
}

public enum DFRError: Error, CustomStringConvertible {
    case badRounds(Int)
    case notKeyframeCapable
    case canvasTooShort(target: Int, canvas: Int)
    case stitchLength(got: Int, want: Int)
    case missingAnchor([Int])
    public var description: String {
        switch self {
        case .badRounds(let r): "temporalUpsampleRounds must be 0, 1, or 2, got \(r)"
        case .notKeyframeCapable:
            "DFR requires a generated-keyframe checkpoint (transformer `keyframes_abs_pos_embedding`); "
            + "this model has none, so the slots would be denoised as unmarked tokens and the extra "
            + "compute would be wasted"
        case .canvasTooShort(let t, let c): "target \(t) frames exceeds the generated canvas \(c)"
        case .stitchLength(let g, let w): "stitched latent T=\(g) != expected \(w)"
        case .missingAnchor(let p): "anchor seams \(p) missing from the carry-forward bag"
        }
    }
}

// MARK: - The composed denoise state

/// Base tokens plus whatever the conditionings appended, in ONE bag — the Swift stand-in for the
/// oracle's `LatentState` threaded through `state_with_conditionings`.
///
/// Assembly order is load-bearing and mirrors the oracle's conditioning list: anchors (clean,
/// `1−strength`-masked) first, generated slots last, so the slot layout's `firstToken` lands past
/// every appended anchor and `extract` reads the right span back.
/// Public because `--dfr-gate` drives THIS assembler rather than a copy of it: a harness that
/// re-derives the composition would gate the harness, not the pipeline.
public struct DFRState {
    public var latent: MLXArray
    public var clean: MLXArray
    public var denoiseMask: MLXArray
    public var positions: MLXArray
    public var keyframesMask: MLXArray?
    public var slotLayout: GeneratedKeyframeLayout?
    /// Base (generation-region) token count — the slice bound after denoising.
    public let baseTokens: Int

    /// Whether every token is fully denoised. The oracle's `_is_uniform_mask` decides the same
    /// thing, and it decides which loop runs: uniform ⇒ scalar timesteps, non-uniform ⇒ per-token
    /// timesteps plus the clean re-blend.
    public var isUniformMask: Bool { MLX.all(MLX.equal(denoiseMask, MLXArray(Float(1)))).item(Bool.self) }

    public func slice(_ denoised: MLXArray) -> MLXArray { denoised[0..., ..<baseTokens, 0...] }
}

extension LTX2Pipeline {

    // MARK: - helpers

    /// Denormalize → neural upsample → re-normalize, the invariant this repo learned the hard way.
    ///
    /// The upsamplers operate in UN-normalized latent space. Skipping the round-trip produces grid
    /// artefacts, not an error.
    func upsampleLatent(_ latentBCFHW: MLXArray, _ up: Upsampler) -> MLXArray {
        vaeEncoder!.normalizeLatent(up(vaeEncoder!.denormalizeLatent(latentBCFHW)))
    }

    /// Frame-0 marker UNION the state's own slot marks.
    ///
    /// ⚠️ The frame-0 helper builds a FRESH mask over the whole sequence; assigning it would unmark
    /// every slot and silently disable the embedding that makes a slot a keyframe. The oracle takes
    /// an element-wise max for exactly this reason (`_keyframes_mask`).
    func dfrKeyframesMask(_ state: DFRState, tokensPerFrame: Int) -> MLXArray? {
        guard let dit = try? ensureDiT(), dit.weight("keyframes_abs_pos_embedding") != nil else { return nil }
        return Self.dfrKeyframesMask(state, tokensPerFrame: tokensPerFrame)
    }

    /// The union itself, without the checkpoint-capability probe — the part `--dfr-gate` asserts.
    public static func dfrKeyframesMask(_ state: DFRState, tokensPerFrame: Int) -> MLXArray {
        var mask = firstLatentFrameKeyframesMask(totalTokens: state.latent.dim(1),
                                                 tokensPerLatentFrame: tokensPerFrame,
                                                 batch: state.latent.dim(0))
        if let own = state.keyframesMask { mask = MLX.maximum(mask, own.asType(mask.dtype)) }
        return mask
    }

    /// Build a noised base state, then apply anchors and slots in the oracle's order.
    ///
    /// `initialLatent == nil` starts from zeros (stage 1); otherwise the clean reference is that
    /// latent and the base is `noise·σ + initial·(1−σ)` — the `legacy_scalar_blend` flow, i.e. the
    /// noise blend runs BEFORE conditionings, which is why appended slot tokens must arrive
    /// already-noised (see `KeyframeSlots.append`).
    public static func dfrState(
        baseTokens n: Int, channels: Int = 128, initialLatent: MLXArray?, sigma: Float, seed: UInt64,
        positions: MLXArray, H: Int, W: Int,
        anchors: [KeyframeConditioning] = [],
        slotPositions: [Int] = [], slotFPS: Float = 24, slotSigma: Float = 1,
        slotNoiseSeed: UInt64 = 0, slotInitials: MLXArray? = nil,
        references: [ReferenceConditioning] = []
    ) -> DFRState {
        let clean = initialLatent ?? MLXArray.zeros([1, n, channels])
        let latent = noiseInit(clean: clean, sigma: sigma, shape: [1, n, channels], seed: seed)
        var state = DFRState(latent: latent, clean: clean,
                             denoiseMask: MLXArray.ones([1, n, 1]).asType(latent.dtype),
                             positions: positions.asType(.float32), keyframesMask: nil,
                             slotLayout: nil, baseTokens: n)

        if !anchors.isEmpty {
            let a = KeyframeConditionedState.append(
                latent: state.latent, clean: state.clean, denoiseMask: state.denoiseMask,
                positions: state.positions, keyframesMask: state.keyframesMask, items: anchors)
            state.latent = a.latent; state.clean = a.clean
            state.denoiseMask = a.denoiseMask; state.positions = a.positions
            state.keyframesMask = a.keyframesMask
        }
        if !slotPositions.isEmpty {
            let s = KeyframeSlots.append(
                latent: state.latent, clean: state.clean, denoiseMask: state.denoiseMask,
                positions: state.positions, keyframesMask: state.keyframesMask,
                pixelFrameIndices: slotPositions, H: H, W: W, fps: slotFPS,
                sigma: slotSigma, noiseSeed: slotNoiseSeed, initialKeyframes: slotInitials)
            state.latent = s.latent; state.clean = s.clean
            state.denoiseMask = s.denoiseMask; state.positions = s.positions
            state.keyframesMask = s.keyframesMask; state.slotLayout = s.layout
        }
        if !references.isEmpty {
            let r = ICVideoState.build(targetLatent: state.latent, targetPositions: state.positions,
                                       targetClean: state.clean, targetDenoiseMask: state.denoiseMask,
                                       references: references)
            state.latent = r.latent; state.clean = r.clean
            state.denoiseMask = r.denoiseMask; state.positions = r.positions
            // Reference tokens are ordinary conditioning: NOT marked as generated keyframes.
            state.keyframesMask = state.keyframesMask.map {
                concatenated([$0, MLXArray.zeros([1, r.latent.dim(1) - $0.dim(1), 1]).asType($0.dtype)],
                             axis: 1)
            }
        }
        return state
    }

    /// Drive the right loop for the state's mask, mirroring the oracle's `_is_uniform_mask` branch.
    func dfrDenoise(
        _ state: DFRState, audioState: DFRState?, sigmas: [Float],
        videoText: MLXArray?, audioText: MLXArray?,
        keyframesMask: MLXArray?, ancestralEta: Float = 0, ancestralNoiseSeed: UInt64 = 0,
        label: String, stage: Int? = nil, totalStages: Int? = nil
    ) throws -> (video: MLXArray, audio: MLXArray?) {
        let uniform = state.isUniformMask && (audioState?.isUniformMask ?? true)
        if uniform {
            return try DenoiseLoop.run(
                dit: try ensureDiT(), videoLatent0: state.latent, audioLatent0: audioState?.latent,
                sigmas: sigmas, videoText: videoText, audioText: audioText,
                videoPositions: state.positions, audioPositions: audioState?.positions,
                keyframesMask: keyframesMask, ancestralEta: ancestralEta,
                ancestralNoiseSeed: ancestralNoiseSeed,
                label: label, stage: stage, totalStages: totalStages)
        }
        return try DenoiseLoop.runConditioned(
            dit: try ensureDiT(), videoLatent0: state.latent, audioLatent0: audioState?.latent,
            sigmas: sigmas, videoText: videoText, audioText: audioText,
            videoPositions: state.positions, audioPositions: audioState?.positions,
            videoCleanLatent: state.isUniformMask ? nil : state.clean,
            videoDenoiseMask: state.isUniformMask ? nil : state.denoiseMask,
            audioCleanLatent: (audioState?.isUniformMask ?? true) ? nil : audioState?.clean,
            audioDenoiseMask: (audioState?.isUniformMask ?? true) ? nil : audioState?.denoiseMask,
            keyframesMask: keyframesMask, ancestralEta: ancestralEta,
            ancestralNoiseSeed: ancestralNoiseSeed,
            label: label, stage: stage, totalStages: totalStages)
    }

    /// Tokens (1, F·H·W, C) → latent (1, C, F, H, W). Inverse of `LTX2Pipeline.patchify`.
    static func unpatchify(_ tokens: MLXArray, F: Int, H: Int, W: Int) -> MLXArray {
        let B = tokens.dim(0), C = tokens.dim(2)
        return tokens.reshaped(B, F, H, W, C).transposed(0, 4, 1, 2, 3)
    }

    /// Floor both dimensions to the grid a two-stage pipeline needs (oracle `snap_output_dimensions`
    /// with `two_stage=True`): a multiple of 64, so the half-res stage-1 grid is itself a multiple
    /// of 32. A pure floor — never rounds up, never raises, idempotent.
    public static func snapDFROutputDimensions(height: Int, width: Int) -> (height: Int, width: Int) {
        ((height / 64) * 64, (width / 64) * 64)
    }

    // MARK: - generation

    /// Run DFR and return the latents. Mirrors the oracle's `generate_dfr`.
    ///
    /// `detailingReference` appends the reserved half-res latent as an IC reference in stage 2.
    /// **Defaults to false, matching upstream**: upstream appends it only when a detailing IC-LoRA
    /// is loaded, and that LoRA has no default. We do not host one, so the reference would be out of
    /// distribution — the adapter that teaches the model to read those tokens is exactly what is
    /// missing. Kept as a flag so it can be exercised once a detailing LoRA exists.
    public func generateDFR(
        prompt: String, height: Int = 512, width: Int = 768, numFrames: Int = 97,
        fps: Double = 24, seed: UInt64 = 42,
        temporalUpsampleRounds rounds: Int = 0,
        detailingReference: Bool = false,
        stage1Steps: Int? = nil, stage2Steps: Int? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> DFRLatents {
        // Every frame/fps/trim number is resolved BEFORE any weight loads, so a bad request costs
        // milliseconds rather than a stage-1 denoise — and so `--dfr-gate` asserts against the very
        // object the run consumes rather than a re-derivation of it.
        let plan = try DFRPlan.resolve(requestedFrames: numFrames, height: height, width: width,
                                       fps: fps, rounds: rounds)

        // ⚠️ ORDER IS LOAD-BEARING: encode the prompt and free the encoder BEFORE the DiT is needed.
        // The 2.5 text encoder is a bf16 12B (~24 GB); carrying it under the 38 GB transformer
        // nearly doubles peak. `encodePrompt` already drops the DiT first under the sequential
        // policy and frees Gemma + connector after; keep this call ahead of every `ensureDiT()`.
        let (videoEmbeds, audioEmbeds) = try await encodePrompt(prompt)

        // The capability is read off the CHECKPOINT, never a version string: without the learned
        // embedding a slot is denoised as an unmarked token and the extra compute is wasted.
        guard try ensureDiT().weight("keyframes_abs_pos_embedding") != nil else {
            throw DFRError.notKeyframeCapable
        }

        let slotPixelPositions = plan.canvas.slotPositions
        let sigmas1 = stage1Steps.map { Array(Positions.distilledSigmas.prefix($0 + 1)) }
            ?? Positions.distilledSigmas
        let sigmas2 = stage2Steps.map { Array(Positions.stage2Sigmas.prefix($0 + 1)) }
            ?? Positions.stage2Sigmas

        let F = plan.F, H1 = plan.H1, W1 = plan.W1, H2 = plan.H2, W2 = plan.W2
        let nAudio = plan.audioTokens
        let nBase1 = plan.stage1Tokens, nBase2 = plan.stage2Tokens

        MLXProfiler.shared.beginRun(String(format:
            "dfr %dx%d %df→canvas %df (segment %d, %d slots) fps=%.0f rounds=%d | nv1=%d nv2=%d audioT=%d",
            plan.width, plan.height, plan.requestedFrames, plan.canvas.canvasFrames,
            plan.canvas.segment, slotPixelPositions.count, fps, rounds, nBase1, nBase2, nAudio))
        defer { MLXProfiler.shared.endRun() }

        // --- Stage 1: half-res base + keyframe slots ------------------------------------------
        let state1 = Self.dfrState(
            baseTokens: nBase1, initialLatent: nil, sigma: 1.0, seed: seed,
            positions: Positions.video(F: F, H: H1, W: W1, fps: Float(fps)), H: H1, W: W1,
            slotPositions: slotPixelPositions, slotFPS: Float(fps), slotSigma: 1.0,
            slotNoiseSeed: seed &+ 20_000)
        let audio1 = Self.dfrState(
            baseTokens: nAudio, initialLatent: nil, sigma: 1.0, seed: seed &+ 1,
            positions: Positions.audio(tokens: nAudio), H: H1, W: W1)
        armStreamingGate(largestStageTokens: nBase2 + nAudio)   // stage 2 is the largest forward
        let s1Span = MLXProfiler.shared.begin("denoise", "dfr-stage1",
                                              note: "half-res + \(slotPixelPositions.count) slots")
        let (v1, a1Opt) = try dfrDenoise(
            state1, audioState: audio1, sigmas: sigmas1,
            videoText: videoEmbeds, audioText: audioEmbeds,
            keyframesMask: dfrKeyframesMask(state1, tokensPerFrame: H1 * W1),
            ancestralEta: 1.0, ancestralNoiseSeed: seed &+ 10_000,
            label: "dfr-s1-", stage: 1, totalStages: 2 + rounds)
        let a1 = a1Opt!   // audio supplied at stage 1 ⇒ audio returned
        eval(v1, a1)
        MLXProfiler.shared.end(s1Span)

        let slotsHalf = state1.slotLayout!.extract(v1, H: H1, W: W1)
        let reservedHalf = Self.unpatchify(v1[0..., ..<nBase1, 0...], F: F, H: H1, W: W1)

        // --- Stage 2: spatial detailing ---------------------------------------------------------
        LTX2Progress.report(.upsample)
        let upSpan = MLXProfiler.shared.begin("upscale", "dfr-stage2", note: "video + \(slotsHalf.count) slots")
        try ensureVAEEncoder(); try ensureUpsampler()
        let upscaledVideo = upsampleLatent(reservedHalf, upsampler!)
        // The slots go through the SAME spatial upsampler as a stacked (B,C,K,H,W) — the one-frame-
        // at-a-time constraint is on DECODING, not on the upsampler (upstream does this too).
        let upsampledSlots = upsampleLatent(concatenated(slotsHalf, axis: 2), upsampler!)
        eval(upscaledVideo, upsampledSlots)
        MLXProfiler.shared.end(upSpan)
        dropUpscaler()

        var references: [ReferenceConditioning] = []
        if detailingReference {
            references.append(ReferenceConditioning(
                tokens: Self.patchify(reservedHalf),
                // Half-res grid positions; downscaleFactor scales the SPATIAL axes into the
                // full-res frame, leaving time alone.
                positions: Positions.video(F: F, H: H1, W: W1, fps: Float(fps)),
                downscaleFactor: 2, strength: 1.0))
        }
        let state2 = Self.dfrState(
            baseTokens: nBase2, initialLatent: Self.patchify(upscaledVideo), sigma: sigmas2[0],
            seed: seed &+ 2, positions: Positions.video(F: F, H: H2, W: W2, fps: Float(fps)),
            H: H2, W: W2,
            slotPositions: slotPixelPositions, slotFPS: Float(fps), slotSigma: sigmas2[0],
            slotNoiseSeed: seed &+ 30_000, slotInitials: upsampledSlots,
            references: references)
        let audio2 = Self.dfrState(
            baseTokens: nAudio, initialLatent: a1, sigma: sigmas2[0], seed: seed &+ 3,
            positions: Positions.audio(tokens: nAudio), H: H2, W: W2)
        let s2Span = MLXProfiler.shared.begin("denoise", "dfr-stage2", note: "full-res detailing")
        let (v2, a2) = try dfrDenoise(
            state2, audioState: audio2, sigmas: sigmas2,
            videoText: videoEmbeds, audioText: audioEmbeds,
            keyframesMask: dfrKeyframesMask(state2, tokensPerFrame: H2 * W2),
            label: "dfr-s2-", stage: 2, totalStages: 2 + rounds)
        eval([v2, a2].compactMap { $0 })
        MLXProfiler.shared.end(s2Span)

        var carryPositions = slotPixelPositions
        var carryKeyframes = concatenated(state2.slotLayout!.extract(v2, H: H2, W: W2), axis: 2)
        var videoLatent = Self.unpatchify(v2[0..., ..<nBase2, 0...], F: F, H: H2, W: W2)
        var audioTokens = a2

        // --- Temporal rounds ---------------------------------------------------------------------
        for roundPlan in plan.roundPlans {
            let round = roundPlan.index
            try ensureVAEEncoder(); try ensureTemporalUpsampler()
            videoLatent = upsampleLatent(videoLatent, temporalUpsampler!)
            eval(videoLatent)
            dropUpscaler()

            // Canvas + capped conditioning fps come from the plan — see `DFRPlan` for why the cap
            // is independent of playback fps.
            let canvasFrames = roundPlan.canvasFrames
            let condFPS = roundPlan.conditioningFPS
            let seamPositions = carryPositions.map { 2 * $0 }
            let anchorKeyframes = carryKeyframes
            var seamToIndex: [Int: Int] = [:]
            for (i, s) in seamPositions.enumerated() { seamToIndex[s] = i }
            let tiles = try DFRLayout.tileRanges(seamPositions: seamPositions,
                                                 numFrames: canvasFrames, numTiles: roundPlan.requestedTiles)
            let temporalSigmas = Array(Positions.distilledSigmas.dropFirst(4))
            // Re-arm the streaming gate for this round: a tile carries its own window's tokens
            // (video only) plus its appended anchors and slots, which can exceed stage 2's count
            // once the lead-in is included. Arming with the LARGEST forward the round will make
            // keeps the gate's threshold semantics intact (see `armStreamingGate`).
            armStreamingGate(largestStageTokens: Swift.max(
                nBase2 + nAudio,
                tiles.map { t in
                    ((t.latentEndExclusive - t.latentStart) + t.anchorKFGlobal.count
                        + t.slotKFGlobal.count) * H2 * W2
                }.max() ?? 0))

            var tileLatents: [MLXArray] = []
            var slotPositionsR: [Int] = []
            var slotSlices: [MLXArray] = []

            for (tileIndex, tile) in tiles.enumerated() {
                let Ft = tile.latentEndExclusive - tile.latentStart
                let localFrames = (Ft - 1) * DFRLayout.temporalScale + 1
                let tileVideo = videoLatent[0..., 0..., tile.latentStart ..< tile.latentEndExclusive, 0..., 0...]

                // Every seam in the window is a hard keyframe, including the one at local frame 0.
                // ⚠️ Conditioning is TILE-LOCAL: the global seam is remapped before it is pinned,
                // or a non-first tile would anchor the wrong frame onto its seam.
                let anchorGlobal = tile.anchorKFGlobal
                let missing = anchorGlobal.filter { seamToIndex[$0] == nil }
                guard missing.isEmpty else { throw DFRError.missingAnchor(missing) }
                let anchorLocal = DFRLayout.remapPositionsToLocal(anchorGlobal, pixelStart: tile.pixelStart)
                let anchors = zip(anchorGlobal, anchorLocal).map { global, local in
                    let i = seamToIndex[global]!
                    return KeyframeConditioning(
                        frameIdx: local,
                        latent: Self.patchify(anchorKeyframes[0..., 0..., i ..< (i + 1), 0..., 0...]),
                        H: H2, W: W2, fps: Float(condFPS), strength: anchorKeyframeStrength)
                }

                let slotGlobal = tile.slotKFGlobal
                let slotLocal = DFRLayout.remapPositionsToLocal(slotGlobal, pixelStart: tile.pixelStart)
                let tileState = Self.dfrState(
                    baseTokens: Ft * H2 * W2, initialLatent: Self.patchify(tileVideo),
                    sigma: temporalSigmas[0],
                    seed: seed &+ 5_000 &+ UInt64(100 * round + tileIndex),
                    positions: Positions.video(F: Ft, H: H2, W: W2, fps: Float(condFPS)),
                    H: H2, W: W2, anchors: anchors,
                    slotPositions: slotLocal, slotFPS: Float(condFPS), slotSigma: temporalSigmas[0],
                    slotNoiseSeed: seed &+ 40_000 &+ UInt64(100 * round + tileIndex),
                    slotInitials: slotLocal.isEmpty ? nil
                        : DFRLayout.slotInitialsFromVideo(tileVideo, positions: slotLocal))

                let tSpan = MLXProfiler.shared.begin(
                    "denoise", "dfr-r\(round)-t\(tileIndex + 1)",
                    note: "\(localFrames)f  anchors=\(anchors.count) slots=\(slotLocal.count)")
                // ⚠️ VIDEO-ONLY (audioState nil, oracle `audio_state=None`): audio comes from
                // stage 2 and nothing after refines it. The video genuinely differs from an
                // audio-present forward — it loses its A2V cross-attention residual — which is
                // upstream's behaviour, not an approximation. Ancestral seeds MUST differ per
                // tile: tiles are positionally identical, so a shared seed would inject
                // byte-identical noise into every one of them.
                let (tileOut, _) = try dfrDenoise(
                    tileState, audioState: nil, sigmas: temporalSigmas,
                    videoText: videoEmbeds, audioText: audioEmbeds,
                    keyframesMask: dfrKeyframesMask(tileState, tokensPerFrame: H2 * W2),
                    ancestralEta: temporalAncestralEta,
                    ancestralNoiseSeed: seed &+ UInt64(1_000 * round + tileIndex),
                    label: "dfr-r\(round)t\(tileIndex)-", stage: 2 + round, totalStages: 2 + rounds)
                eval(tileOut)
                MLXProfiler.shared.end(tSpan)

                tileLatents.append(Self.unpatchify(tileOut[0..., ..<(Ft * H2 * W2), 0...],
                                                   F: Ft, H: H2, W: W2))
                if let layout = tileState.slotLayout {
                    slotPositionsR.append(contentsOf: slotGlobal)
                    slotSlices.append(concatenated(layout.extract(tileOut, H: H2, W: W2), axis: 2))
                }
                Memory.clearCache()
            }

            videoLatent = try DFRLayout.stitchTileLatents(tileLatents, ranges: tiles)
            let expectedT = (canvasFrames - 1) / DFRLayout.temporalScale + 1
            guard videoLatent.dim(2) == expectedT else {
                throw DFRError.stitchLength(got: videoLatent.dim(2), want: expectedT)
            }

            var slotLatents = slotSlices.isEmpty ? nil : concatenated(slotSlices, axis: 2)
            if !slotPositionsR.isEmpty, let sl = slotLatents {
                // Lead-in segments repeat the previous tile's slots; the earlier tile's wins.
                var firstIndex: [Int: Int] = [:]
                for (i, p) in slotPositionsR.enumerated() where firstIndex[p] == nil { firstIndex[p] = i }
                slotPositionsR = firstIndex.keys.sorted()
                slotLatents = concatenated(slotPositionsR.map {
                    sl[0..., 0..., firstIndex[$0]! ..< (firstIndex[$0]! + 1), 0..., 0...]
                }, axis: 2)
            }
            (carryPositions, carryKeyframes) = try DFRLayout.mergeCarryForwardKeyframes(
                anchorPositions: seamPositions, anchorLatents: anchorKeyframes,
                slotPositions: slotPositionsR, slotLatents: slotLatents)
            eval(videoLatent, carryKeyframes)
        }
        dropTemporalUpsampler()

        // The canvas may have padded its tail, and each round maps N -> 2(N-1)+1; the plan carries
        // the resulting contract and BOTH trims (see `DFRPlan` for the two bugs they close).
        if plan.needsTrim {
            videoLatent = videoLatent[0..., 0..., ..<plan.keepLatentFrames, 0..., 0...]
            if let a = audioTokens, plan.audioKeepTokens < a.dim(1) {
                audioTokens = a[0..., ..<plan.audioKeepTokens, 0...]
            }
        }

        let kept = plan.keptKeyframeIndices(carryPositions)
        eval([videoLatent, audioTokens].compactMap { $0 })

        return DFRLatents(
            video: videoLatent, audio: audioTokens,
            // ⚠️ PLAYBACK fps is the base rate times 2**rounds. Each round doubles the temporal
            // sample count for the SAME motion, so muxing at the base rate writes a 2x/4x
            // slow-motion clip with the right frames — wrong duration, and silently so.
            playbackFPS: plan.playbackFPS,
            conditioningFPS: plan.conditioningFPS,
            generatedKeyframePositions: kept.map { carryPositions[$0] },
            generatedKeyframeLatents: kept.map { carryKeyframes[0..., 0..., $0 ..< ($0 + 1), 0..., 0...] },
            canvas: DFRCanvas(segment: plan.canvas.segment, canvasFrames: plan.finalCanvasFrames,
                              requestedFrames: plan.requestedFrames, slotPositions: slotPixelPositions),
            deliveredFrames: plan.deliveredFrames, latentHeight: H2, latentWidth: W2)
    }

    /// Run DFR and decode. Mirrors the oracle's `generate_and_save` minus the file write.
    ///
    /// ⚠️ The caller MUST mux at `DFRLatents.playbackFPS` (returned alongside), not the requested
    /// `fps` — see the file header.
    public func dfr(
        prompt: String, height: Int = 512, width: Int = 768, numFrames: Int = 97,
        fps: Double = 24, seed: UInt64 = 42,
        temporalUpsampleRounds rounds: Int = 0,
        detailingReference: Bool = false,
        stage1Steps: Int? = nil, stage2Steps: Int? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> (output: Output, latents: DFRLatents) {
        let latents = try await generateDFR(
            prompt: prompt, height: height, width: width, numFrames: numFrames, fps: fps, seed: seed,
            temporalUpsampleRounds: rounds, detailingReference: detailingReference,
            stage1Steps: stage1Steps, stage2Steps: stage2Steps)
        quiesceStreaming()
        // Decoding with the transformer still resident stacks ~38 GB of weights under the decoder
        // and roughly doubles peak — a first C2 measurement that skipped this reported +31 GB for
        // DFR and the number was entirely this, not the feature.
        dropDiTIfSequential()
        try ensureDecoder()
        let waveform = latents.audio.flatMap { decodeAudio($0) }
        if let waveform { eval(waveform) }
        let decSpan = MLXProfiler.shared.begin("vae-decode", "video",
                                               note: "\(latents.deliveredFrames)f dfr")
        let pixels = try decodePixels(latents.video)
        eval(pixels)
        MLXProfiler.shared.end(decSpan)
        dropDecoder()
        return (Output(video: pixels, audio: waveform), latents)
    }
}
