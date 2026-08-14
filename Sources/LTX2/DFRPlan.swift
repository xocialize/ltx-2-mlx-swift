// DFRPlan.swift — every DFR frame/fps/trim number, resolved once, before any weight loads.
//
// This exists so the arithmetic that carries the expensive traps is a NAMED OBJECT the production
// path consumes, not inline code a gate has to re-derive. Re-deriving it in a harness would gate a
// copy of the pipeline rather than the pipeline; `generateDFR` builds one of these and reads
// nothing else, so a green `--dfr-gate` is a statement about the shipping path.
//
// The four numbers that have each been a real bug:
//   * `deliveredFrames` = (requested − 1)·2^rounds + 1. The canvas PADS (`resolveCanvas`), so the
//     generated clip is longer than the request and must be trimmed back.
//   * `audioKeepTokens`. Audio is generated for the PADDED canvas; trimmed video with untrimmed
//     audio outlasts the picture, and `FrameCodec` writes with no `-shortest`. Measured before the
//     trim existed: 9 requested frames pad to a 25-frame canvas → 1.04 s of audio over 0.375 s of
//     video.
//   * `playbackFPS` = fps·2^rounds. Each round doubles the temporal sample count for the SAME
//     motion, so muxing at the base rate writes a 2x/4x slow-motion clip with the right pictures.
//   * `conditioningFPS`, capped at 60 INDEPENDENTLY of playback. RoPE time is pixelFrame/fps, so a
//     120 fps time base halves every token's temporal span versus training and decodes as a motion
//     spike at each latent border followed by a stall.

import Foundation

public struct DFRPlan: Equatable {

    /// One temporal round's geometry.
    public struct Round: Equatable {
        public let index: Int              // 1-based
        public let canvasFrames: Int       // canvas AFTER this round's x2 densification
        public let conditioningFPS: Double // capped
        public let requestedTiles: Int     // 2^index (DFRLayout clamps to the segment count)
    }

    public let requestedFrames: Int
    public let rounds: Int
    /// Snapped output dimensions (multiples of 64 — two-stage needs the half-res grid on 32).
    public let height: Int, width: Int
    public let stage1Height: Int, stage1Width: Int
    /// Latent grids: F is shared, (H1,W1) half-res, (H2,W2) = 2x.
    public let F: Int, H1: Int, W1: Int, H2: Int, W2: Int
    public let canvas: DFRCanvas
    public let audioTokens: Int
    public let roundPlans: [Round]
    /// Canvas frames after the last round.
    public let finalCanvasFrames: Int
    public let deliveredFrames: Int
    /// Latent frames to keep when trimming the padded canvas back to `deliveredFrames`.
    public let keepLatentFrames: Int
    /// Audio tokens to keep. Equals `audioTokens` when no trim is needed.
    public let audioKeepTokens: Int
    public let baseFPS: Double
    public let playbackFPS: Double
    /// The last round's conditioning fps (== `baseFPS` when `rounds == 0`).
    public let conditioningFPS: Double

    public var stage1Tokens: Int { F * H1 * W1 }
    public var stage2Tokens: Int { F * H2 * W2 }
    /// True when the canvas padded past the request, i.e. a trim is required.
    public var needsTrim: Bool { deliveredFrames != finalCanvasFrames }

    /// Indices of `positions` naming a keyframe INSIDE the delivered clip.
    ///
    /// ⚠️ Carry-forward positions span the PADDED canvas, so an unfiltered list names stills that do
    /// not exist in the output — 9 requested frames would report a keyframe at pixel frame 24 of a
    /// 9-frame video.
    public func keptKeyframeIndices(_ positions: [Int]) -> [Int] {
        positions.indices.filter { positions[$0] < deliveredFrames }
    }

    public static func resolve(
        requestedFrames: Int, height: Int, width: Int, fps: Double, rounds: Int
    ) throws -> DFRPlan {
        guard (0 ... 2).contains(rounds) else { throw DFRError.badRounds(rounds) }
        let c = try DFRLayout.resolveCanvas(numFrames: requestedFrames)
        let (snapH, snapW) = LTX2Pipeline.snapDFROutputDimensions(height: height, width: width)
        let s1h = (snapH / 2 / 32) * 32, s1w = (snapW / 2 / 32) * 32
        let h1 = s1h / 32, w1 = s1w / 32
        let f = (c.padded + 7) / 8
        // Audio is sized on the PADDED canvas, which is exactly why the trim below exists.
        let nAudio = Positions.audioTokenCount(numFrames: c.padded, fps: fps)

        var canvasFrames = c.padded
        var condFPS = fps, currentFPS = fps
        var plans: [Round] = []
        for r in 1 ..< (rounds + 1) {
            canvasFrames = 2 * (canvasFrames - 1) + 1
            currentFPS *= 2
            condFPS = Swift.min(currentFPS, maxConditioningFPS)
            plans.append(Round(index: r, canvasFrames: canvasFrames,
                               conditioningFPS: condFPS, requestedTiles: 1 << r))
        }

        let delivered = (requestedFrames - 1) * (1 << rounds) + 1
        guard delivered <= canvasFrames else {
            throw DFRError.canvasTooShort(target: delivered, canvas: canvasFrames)
        }
        let keep = (delivered - 1) / DFRLayout.temporalScale + 1
        let audioKeep = delivered == canvasFrames
            ? nAudio
            : Swift.min(nAudio, Positions.audioTokenCount(numFrames: delivered, fps: fps))

        return DFRPlan(
            requestedFrames: requestedFrames, rounds: rounds,
            height: snapH, width: snapW, stage1Height: s1h, stage1Width: s1w,
            F: f, H1: h1, W1: w1, H2: h1 * 2, W2: w1 * 2,
            canvas: DFRCanvas(segment: c.segment, canvasFrames: c.padded,
                              requestedFrames: requestedFrames, slotPositions: c.positions),
            audioTokens: nAudio, roundPlans: plans,
            finalCanvasFrames: canvasFrames, deliveredFrames: delivered,
            keepLatentFrames: keep, audioKeepTokens: audioKeep,
            baseFPS: fps, playbackFPS: fps * Double(1 << rounds), conditioningFPS: condFPS)
    }
}
