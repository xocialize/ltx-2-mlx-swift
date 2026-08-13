// KeyframeSlots.swift — LTX-2.5 generated keyframe slots (the multishot surface).
//
// Swift sibling of the oracle's `VideoGeneratedKeyframeSlots`. A *slot* is a block of
// appended tokens the model DENOISES — unlike ordinary keyframe conditioning, which appends
// CLEAN reference tokens — each asking the model to invent one standalone keyframe at a
// chosen pixel-frame index.
//
// Two things distinguish a slot from a regular latent frame, both load-bearing:
//   * RoPE span is forced to exactly [t, t+1) before the /fps division (our convention is the
//     midpoint of upstream's [start, end) bounds, so (t + 0.5) / fps). causal_fix must NOT
//     also be applied — it would collapse frame 0 a second time.
//   * keyframesMask is 1 on slot tokens, so they receive the DiT's learned
//     keyframes_abs_pos_embedding. Ordinary keyframe conditioning is deliberately NOT marked.
//
// ⚠️ Slot tokens are appended ALREADY NOISED as lerp(initial, noise, sigma). The oracle
// appends zeros and lets its noiser fill them, but our stage-1 callsites noise BEFORE
// conditionings run, so an appended zero block would stay zero and never generate. At
// sigma 1.0 this is pure noise (identical to upstream); at the small stage-2 sigma the
// seed content SURVIVES, which is what makes seeded re-append work.
//
// ⚠️ Read generated keyframes back ONE FRAME AT A TIME. A K-frame causal DECODE would blend
// slots that were never temporally adjacent (oracle types.py). `extract` returns one latent
// per slot for that reason — the constraint is on decoding, not on the latent upsampler,
// which upstream feeds a stacked (B,C,K,H,W).

import Foundation
import MLX
import MLXRandom

/// Where the slot tokens live, so they can be read back after denoising.
public struct GeneratedKeyframeLayout {
    public let pixelFrameIndices: [Int]
    public let tokensPerKeyframe: Int
    public let firstToken: Int

    public func slotSpan(_ i: Int) -> Range<Int> {
        let start = firstToken + i * tokensPerKeyframe
        return start ..< (start + tokensPerKeyframe)
    }
}

/// A latent state extended with generated keyframe slots.
public struct KeyframeSlotState {
    public let latent: MLXArray          // (B, N + K*H*W, C)
    public let clean: MLXArray
    public let denoiseMask: MLXArray     // 1 on slots — they are DENOISED
    public let positions: MLXArray
    public let keyframesMask: MLXArray   // 1 on slots — they carry the learned marker
    public let layout: GeneratedKeyframeLayout
    public let targetTokens: Int

    /// Drop the appended slot tokens from a denoised latent.
    public func slice(_ denoised: MLXArray) -> MLXArray {
        denoised[0..., ..<targetTokens, 0...]
    }

    /// Pull each denoised slot out as its own (B, C, 1, H, W) latent — one per slot,
    /// deliberately, so a caller cannot decode them as a K-frame clip.
    public func extract(_ denoised: MLXArray, H: Int, W: Int) -> [MLXArray] {
        (0 ..< layout.pixelFrameIndices.count).map { i in
            let span = layout.slotSpan(i)
            let toks = denoised[0..., span, 0...]
            let B = toks.dim(0), C = toks.dim(2)
            return toks.reshaped(B, H, W, C).transposed(0, 3, 1, 2).expandedDimensions(axis: 2)
        }
    }
}

public enum KeyframeSlots {

    /// Positions for one slot: full spatial grid, temporal span exactly [t, t+1) then /fps.
    public static func slotPositions(pixelFrame: Int, H: Int, W: Int, fps: Float) -> MLXArray {
        let base = Positions.video(F: 1, H: H, W: W, fps: fps)   // (1, H*W, 3)
        let tMid = (Float(pixelFrame) + 0.5) / fps
        let spatial = base[0..., 0..., 1...]
        let temporal = MLXArray.full([base.dim(0), base.dim(1), 1], values: MLXArray(tMid))
        return concatenated([temporal, spatial], axis: 2)
    }

    /// Append one denoised slot per requested pixel frame.
    ///
    /// - Parameters:
    ///   - sigma: noise level for the slot block. 1.0 at stage 1 (pure noise); the stage-2
    ///     start sigma when `initialKeyframes` seeds them.
    ///   - initialKeyframes: optional (B, C, K, H, W) seed content, e.g. the upsampled
    ///     stage-1 slots. Survives at small sigma; fully replaced at sigma 1.0.
    public static func append(
        latent: MLXArray,
        clean: MLXArray,
        denoiseMask: MLXArray,
        positions: MLXArray,
        keyframesMask: MLXArray?,
        pixelFrameIndices: [Int],
        H: Int, W: Int, fps: Float,
        sigma: Float = 1.0,
        noiseSeed: UInt64 = 0,
        initialKeyframes: MLXArray? = nil
    ) -> KeyframeSlotState {
        precondition(!pixelFrameIndices.isEmpty, "pixelFrameIndices must be non-empty")
        precondition(pixelFrameIndices.allSatisfy { $0 >= 0 }, "pixelFrameIndices must be non-negative")
        precondition(zip(pixelFrameIndices, pixelFrameIndices.dropFirst()).allSatisfy { $0 < $1 },
                     "pixelFrameIndices must be strictly increasing")
        if let seed = initialKeyframes {
            precondition(seed.ndim == 5, "initialKeyframes must be (B, C, K, H, W)")
            precondition(seed.dim(2) == pixelFrameIndices.count,
                         "initialKeyframes K must match pixelFrameIndices count")
        }

        let B = latent.dim(0), N = latent.dim(1), C = latent.dim(2)
        let tokensPer = H * W
        let nNew = tokensPer * pixelFrameIndices.count

        // Seed content, patchified to tokens; zeros when unseeded.
        var initial = MLXArray.zeros([B, nNew, C]).asType(latent.dtype)
        if let seed = initialKeyframes {
            let k = seed.dim(2)
            initial = seed.transposed(0, 2, 3, 4, 1).reshaped(B, k * H * W, C).asType(latent.dtype)
        }
        let noise = MLXRandom.normal([B, nNew, C], key: MLXRandom.key(noiseSeed)).asType(latent.dtype)
        let slotBlock = initial * (1.0 - sigma) + noise * sigma

        let slotPos = concatenated(
            pixelFrameIndices.map { slotPositions(pixelFrame: $0, H: H, W: W, fps: fps) }, axis: 1)
        let slotPosB = slotPos.dim(0) == B ? slotPos : MLX.broadcast(slotPos, to: [B, nNew, slotPos.dim(2)])

        let baseKf = keyframesMask ?? MLXArray.zeros([B, N, 1]).asType(latent.dtype)

        return KeyframeSlotState(
            latent: concatenated([latent, slotBlock], axis: 1),
            clean: concatenated([clean, MLXArray.zeros([B, nNew, C]).asType(clean.dtype)], axis: 1),
            denoiseMask: concatenated([denoiseMask, MLXArray.ones([B, nNew, 1]).asType(denoiseMask.dtype)], axis: 1),
            positions: concatenated([positions.asType(.float32), slotPosB], axis: 1),
            keyframesMask: concatenated([baseKf, MLXArray.ones([B, nNew, 1]).asType(baseKf.dtype)], axis: 1),
            layout: GeneratedKeyframeLayout(pixelFrameIndices: pixelFrameIndices,
                                            tokensPerKeyframe: tokensPer, firstToken: N),
            targetTokens: N)
    }

    /// Interior pixel-frame positions for `k` evenly spaced generated keyframes.
    ///
    /// ⚠️ Upstream is `torch.linspace(0, num_frames - 1, k + 2).round()`, and torch.round is
    /// round-half-to-EVEN on a float32 grid. Do not "simplify" to Int(x + 0.5) — it disagrees
    /// on every tie.
    public static func evenlySpacedPositions(_ k: Int, numFrames: Int) -> [Int] {
        precondition(k >= 0, "k must be non-negative")
        if k == 0 { return [] }
        precondition(numFrames >= k + 2, "need at least k + 2 frames")
        let grid = MLXArray.linspace(Float(0), Float(numFrames - 1), count: k + 2).asType(.float32)
        let rounded = MLX.round(grid).asType(.int32).asArray(Int32.self).map(Int.init)
        return Array(rounded.dropFirst().dropLast())
    }
}
