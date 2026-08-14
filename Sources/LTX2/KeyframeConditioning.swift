// KeyframeConditioning.swift — condition on a CLEAN keyframe at a chosen pixel frame.
//
// 1:1 port of the oracle's `VideoConditionByKeyframeIndex`
// (ltx_core_mlx/conditioning/types/keyframe_cond.py). One item per keyframe, applied in order;
// each appends its (B, H·W, C) latent as clean context with denoise mask `1 − strength` and its
// own single-frame RoPE positions.
//
// Contrast with `KeyframeSlots`, which appends DENOISED tokens the model fills in:
//   * a slot is marked in `keyframesMask` (it gets the learned keyframes embedding);
//     an ordinary keyframe conditioning is deliberately NOT marked;
//   * a slot's mask is 1 (generate); a keyframe's is `1 − strength` (preserve).
//
// ⚠️ `frameIdx` is TILE-LOCAL wherever the state is a tile: 0 means "the first frame of this
// window", not "the first frame of the clip". DFR remaps every global seam through
// `DFRLayout.remapPositionsToLocal` before building these — pinning a global index onto a
// mid-canvas tile would anchor the wrong frame, silently.
//
// ⚠️ Attention mask: the oracle calls `update_attention_mask(attention_mask: None, …)`, which
// returns nil when the state carries no mask yet — so the whole DFR path runs with
// `attention_mask == None` and the conditioning strength lives ENTIRELY in the denoise mask.
// That is why this port needs no (B, N, N) mask machinery; strength < 1 is not an attention
// weight here. (Ported reference: mask_utils.update_attention_mask:139-141.)

import Foundation
import MLX

/// One clean-keyframe conditioning item.
public struct KeyframeConditioning {
    /// Pixel-frame index this keyframe pins, LOCAL to the state being conditioned.
    public let frameIdx: Int
    /// Clean latent tokens (B, H·W, C) — VAE-space, normalized, already patchified.
    public let latent: MLXArray
    /// 1.0 = fully preserved (denoise mask 0); < 1.0 leaves the token partially free.
    public let strength: Float
    /// Positions (1, H·W, 3) at the target latent grid.
    public let positions: MLXArray

    public init(frameIdx: Int, latent: MLXArray, H: Int, W: Int, fps: Float,
                strength: Float = 1.0, numPixelFrames: Int = 1) {
        self.frameIdx = frameIdx
        self.latent = latent
        self.strength = strength
        self.positions = Self.keyframePositions(frameIdx: frameIdx, H: H, W: W, fps: fps,
                                                numPixelFrames: numPixelFrames)
    }

    /// (1, H·W, 3) positions for a single keyframe: single-frame temporal coords offset by
    /// `frameIdx` BEFORE the /fps division, spatial pixel midpoints as usual.
    ///
    /// The oracle's branch structure is kept verbatim even though both arms collapse to the same
    /// number at `numPixelFrames == 1` (the default and DFR's only case): causal fix gives
    /// `[0, 1)` at frame 0 and `[0, 8)` elsewhere, then a single-pixel-frame keyframe narrows the
    /// span back to `[0, 1)` either way. Flattening it would silently change behaviour the day
    /// someone passes a multi-frame keyframe.
    public static func keyframePositions(frameIdx: Int, H: Int, W: Int, fps: Float,
                                         numPixelFrames: Int = 1) -> MLXArray {
        let tStart: Float = 0
        var tEnd: Float = frameIdx == 0 ? 1 : Positions.videoTemporalScale
        if numPixelFrames == 1 { tEnd = tStart + 1 }
        let tMid = ((tStart + tEnd) / 2 + Float(frameIdx)) / fps

        let s = Positions.videoSpatialScale
        let hMids = MLXArray(0 ..< H).asType(.float32) * s + s / 2
        let wMids = MLXArray(0 ..< W).asType(.float32) * s + s / 2
        let hGrid = MLX.broadcast(hMids.reshaped(H, 1), to: [H, W])
        let wGrid = MLX.broadcast(wMids.reshaped(1, W), to: [H, W])
        let tGrid = MLXArray.full([H, W], values: MLXArray(tMid))
        return MLX.stacked([tGrid, hGrid, wGrid], axis: -1).reshaped(1, H * W, 3).asType(.float32)
    }
}

/// A latent state extended with appended clean keyframe tokens.
///
/// Deliberately shaped like `ICVideoState` (same concat semantics, same `slice` contract) — the
/// oracle's keyframe and reference conditionings differ only in how they compute positions.
public struct KeyframeConditionedState {
    public let latent: MLXArray        // (B, N + K·H·W, C)
    public let clean: MLXArray
    public let denoiseMask: MLXArray   // (B, N + K·H·W, 1): 1−strength on the keyframe tokens
    public let positions: MLXArray
    public let keyframesMask: MLXArray?
    /// N — the slice bound after the denoise loop.
    public let targetTokens: Int

    public func slice(_ denoised: MLXArray) -> MLXArray { denoised[0..., ..<targetTokens, 0...] }

    /// Append each item in order, mirroring `state_with_conditionings`.
    ///
    /// `keyframesMask` is extended with ZEROS for the appended tokens: an ordinary keyframe is not
    /// a generated keyframe and must not receive the learned `keyframes_abs_pos_embedding`.
    public static func append(
        latent: MLXArray, clean: MLXArray, denoiseMask: MLXArray, positions: MLXArray,
        keyframesMask: MLXArray?, items: [KeyframeConditioning]
    ) -> KeyframeConditionedState {
        let B = latent.dim(0)
        let targetTokens = latent.dim(1)
        var l = latent, c = clean, m = denoiseMask
        var p = positions.asType(.float32), kf = keyframesMask
        for item in items {
            let toks = item.latent.asType(l.dtype)
            let n = toks.dim(1)
            l = concatenated([l, toks], axis: 1)
            c = concatenated([c, toks.asType(c.dtype)], axis: 1)
            m = concatenated([m, MLXArray.full([B, n, 1], values: MLXArray(1.0 - item.strength)).asType(m.dtype)],
                             axis: 1)
            let posB = item.positions.dim(0) == B
                ? item.positions
                : MLX.broadcast(item.positions, to: [B, n, item.positions.dim(2)])
            p = concatenated([p, posB.asType(.float32)], axis: 1)
            kf = kf.map { concatenated([$0, MLXArray.zeros([B, n, 1]).asType($0.dtype)], axis: 1) }
        }
        return KeyframeConditionedState(latent: l, clean: c, denoiseMask: m, positions: p,
                                        keyframesMask: kf, targetTokens: targetTokens)
    }
}
