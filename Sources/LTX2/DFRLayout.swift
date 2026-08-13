// DFRLayout.swift — LTX-2.5 DFR canvas geometry: segment grid, tile ranges, latent stitch.
//
// Swift sibling of `ltx_pipelines_mlx/dfr_layout.py`, itself a bit-exact port of upstream
// `dfr_layout.py`. All integer arithmetic, so it is gated with `==` and not a cosine.
//
// Three pieces of geometry are subtle enough to name:
//   * `chooseSegmentLength` picks 24 vs 32 by LEAST PADDING, ties to the LARGER.
//   * Frame 0 is deliberately NOT a keyframe anchor — it is already a single-pixel-frame
//     token under causal encoding — which is why `buildTile` filters boundaries on != 0
//     rather than by index.
//   * `dropLatentPrefix` hands over exactly AT the shared seam keyframe: the previous tile
//     keeps the seam latent and the next resumes strictly after it, so a non-first tile
//     drops its lead-in PLUS one.

import Foundation

public enum DFRLayout {
    public static let temporalScale = 8
    public static let segmentCandidates = [24, 32]
    /// Lead-in for non-first tiles, in canvas segments. A tile's local latent 0 is an *image*
    /// latent and its local latent 1 was denoised against that image; neither may be spliced
    /// into the mid-canvas stream. One segment puts both inside the discarded prefix while
    /// keeping the window start on a keyframe anchor.
    public static let tileLeadSegments = 1

    public struct TileRange: Equatable {
        public let pixelStart: Int, pixelEnd: Int
        public let latentStart: Int, latentEndExclusive: Int
        public let anchorKFGlobal: [Int]
        public let slotKFGlobal: [Int]
        public let dropLatentPrefix: Int
    }

    public enum LayoutError: Error, CustomStringConvertible {
        case badFrameCount(Int), badSeams(String), spanTooShort(Int)
        public var description: String {
            switch self {
            case .badFrameCount(let n): "num_frames must satisfy (n - 1) % 8 == 0, got \(n)"
            case .badSeams(let s): "invalid seam positions: \(s)"
            case .spanTooShort(let s): "segment span \(s) is under 2 latent frames"
            }
        }
    }

    /// Pick the segment length that pads least; ties keep the LARGER.
    public static func chooseSegmentLength(contentFrames: Int) -> Int {
        precondition(contentFrames >= 1)
        func pad(_ seg: Int) -> Int { (seg - contentFrames % seg) % seg }
        var best = segmentCandidates[0], bestPad = pad(best)
        for c in segmentCandidates.dropFirst() {
            let p = pad(c)
            if p < bestPad || (p == bestPad && c > best) { best = c; bestPad = p }
        }
        return best
    }

    /// Pad `(numFrames - 1)` up to a multiple of the segment length.
    /// Positions are `[S, 2S, …, N'-1]` — frame 0 excluded, terminal frame included.
    public static func resolveCanvas(numFrames: Int) throws -> (padded: Int, segment: Int, positions: [Int]) {
        guard numFrames >= 2, (numFrames - 1) % temporalScale == 0 else {
            throw LayoutError.badFrameCount(numFrames)
        }
        let content = numFrames - 1
        let segment = chooseSegmentLength(contentFrames: content)
        let contentPadded = content + (segment - content % segment) % segment
        let positions = (1 ... contentPadded / segment).map { segment * $0 }
        return (contentPadded + 1, segment, positions)
    }

    public static func pixelToLatentIndex(_ pixelFrame: Int) -> Int {
        precondition(pixelFrame >= 0)
        precondition(pixelFrame == 0 || pixelFrame % temporalScale == 0,
                     "pixel frame \(pixelFrame) is not on the x\(temporalScale) latent border")
        return pixelFrame / temporalScale
    }

    static func ownedSegmentCounts(_ nSegments: Int, _ numTiles: Int) -> [Int] {
        precondition(nSegments >= 1 && numTiles >= 1)
        let base = nSegments / numTiles, rem = nSegments % numTiles
        return (0 ..< numTiles).map { base + ($0 < rem ? 1 : 0) }
    }

    static func buildTile(_ boundaries: [Int], ownLo: Int, ownHi: Int, leadSegments: Int) -> TileRange {
        let windowLo = Swift.max(0, ownLo - leadSegments)
        let pixelStart = boundaries[windowLo], pixelEnd = boundaries[ownHi]
        let latentStart = pixelToLatentIndex(pixelStart)
        // Handover is exactly at the shared keyframe: the previous tile keeps the seam latent
        // (the 8-frame token ending on the KF mark) and this tile resumes strictly after it.
        var drop = pixelToLatentIndex(boundaries[ownLo]) - latentStart
        if ownLo > 0 { drop += 1 }
        return TileRange(
            pixelStart: pixelStart, pixelEnd: pixelEnd,
            latentStart: latentStart,
            latentEndExclusive: pixelToLatentIndex(pixelEnd) + 1,
            // Frame 0 is not a keyframe, so a zero boundary contributes no anchor.
            anchorKFGlobal: (windowLo ... ownHi).map { boundaries[$0] }.filter { $0 != 0 },
            slotKFGlobal: (windowLo ..< ownHi).map { (boundaries[$0] + boundaries[$0 + 1]) / 2 },
            dropLatentPrefix: drop)
    }

    /// Partition the canvas into `numTiles` keyframe-seam tiles, gapless, with a lead-in.
    /// `numTiles` is clamped to the segment count.
    public static func tileRanges(seamPositions: [Int], numFrames: Int, numTiles: Int,
                                  leadSegments: Int = tileLeadSegments) throws -> [TileRange] {
        guard numFrames >= 2 else { throw LayoutError.badFrameCount(numFrames) }
        guard let last = seamPositions.last, last == numFrames - 1 else {
            throw LayoutError.badSeams("last seam must be the terminal frame \(numFrames - 1)")
        }
        guard leadSegments >= 1 else { throw LayoutError.badSeams("leadSegments must be >= 1") }
        let boundaries = [0] + seamPositions
        for i in 1 ..< boundaries.count {
            let span = boundaries[i] - boundaries[i - 1]
            guard span > 0 else { throw LayoutError.badSeams("seams must be strictly increasing") }
            guard span % temporalScale == 0 else { throw LayoutError.badSeams("span \(span) not a multiple of \(temporalScale)") }
            guard span / temporalScale >= 2 else { throw LayoutError.spanTooShort(span) }
        }
        let nSegments = boundaries.count - 1
        var tiles: [TileRange] = []
        var ownLo = 0
        for (i, count) in ownedSegmentCounts(nSegments, Swift.min(numTiles, nSegments)).enumerated() {
            tiles.append(buildTile(boundaries, ownLo: ownLo, ownHi: ownLo + count,
                                   leadSegments: i > 0 ? leadSegments : 0))
            ownLo += count
        }
        return tiles
    }
}
