// ModalityTiling.swift — split the patchified VIDEO token sequence into spatial+temporal tiles,
// denoise each independently, blend back with trapezoidal weights.
//
// This is the DiT-side activation lever, distinct from every lever we already ship:
//   · BlockStreamer        caps WEIGHT memory
//   · decodeChunked        caps VAE decode activation, temporal
//   · decodeSpatialTiled   caps VAE decode activation, spatial
//   · THIS                 caps the DiT forward's O(N²) attention-scores activation
//
// Splitting N into N/k per tile drops peak attention memory by ~k². Port of the oracle's
// `components/modality_tiling.py`, which is itself a port of upstream `ltx_core.modality_tiling`.
// Design notes, hazards, and the measured SDPA-mask prerequisite: `TILING-PLAN.md`.
//
// ── Two deliberate departures from the oracle, both measured or reasoned in TILING-PLAN ──
//
// 1. **Everything loop-invariant is precomputed at construction.** The oracle recomputes each
//    tile's keep-mask O(T²) times *per forward* to derive conditioning-token blend weights, and
//    round-trips masks through NumPy (`np.flatnonzero`) because MLX lacks boolean indexing — a
//    GPU→host sync in the hot loop. Both depend only on `positions` + the tile layout, which are
//    fixed for the whole denoise. We compute them once, on the host, in `init`. Swift-MLX has no
//    NumPy escape hatch anyway, so this is simultaneously the port and the optimization.
//
// 2. **Generated tokens are SLICED, not gathered.** A tile is a contiguous block in (F,H,W), so
//    viewing the token axis as (F,H,W) turns the oracle's flat-index gather/scatter into a plain
//    range slice and a range-accumulate. No index arrays, no scatter, for the bulk of the work.
//    Only conditioning tokens (few, scattered) still need index gathering.

import Foundation
import MLX

// MARK: - Configuration

public struct DimensionTilingConfig: Sendable, Equatable {
    public var numTiles: Int
    public var overlap: Int
    public init(numTiles: Int = 1, overlap: Int = 0) {
        precondition(numTiles >= 1, "numTiles must be >= 1, got \(numTiles)")
        precondition(overlap >= 0, "overlap must be >= 0, got \(overlap)")
        self.numTiles = numTiles
        self.overlap = overlap
    }
}

/// Token-grid tiling layout for an (F, H, W) latent shape. Tile *counts*, not tile sizes —
/// the pixel-grid VAE tiler (`VideoVAEDecoder`) uses sizes and a different recombination rule.
public struct TileCountConfig: Sendable, Equatable {
    public var frames: DimensionTilingConfig
    public var height: DimensionTilingConfig
    public var width: DimensionTilingConfig

    public init(frames: DimensionTilingConfig = .init(),
                height: DimensionTilingConfig = .init(),
                width: DimensionTilingConfig = .init()) {
        self.frames = frames
        self.height = height
        self.width = width
    }

    /// `--tile-frames N --tile-spatial M --tile-overlap K` in the oracle's CLI terms.
    public init(tileFrames: Int, tileSpatial: Int, overlap: Int) {
        self.init(frames: .init(numTiles: tileFrames, overlap: overlap),
                  height: .init(numTiles: tileSpatial, overlap: overlap),
                  width: .init(numTiles: tileSpatial, overlap: overlap))
    }

    public var totalTiles: Int { frames.numTiles * height.numTiles * width.numTiles }
    /// 1×1×1 ⇒ tiling is a no-op and the tiler should not be installed at all.
    public var isIdentity: Bool { totalTiles == 1 }
}

// MARK: - Token-grid geometry (fresh; deliberately NOT shared with VideoVAEDecoder.tileBounds)

/// How one dimension is split into overlapping intervals, with per-interval blend ramps.
struct DimensionIntervals {
    var starts: [Int]
    var ends: [Int]
    var leftRamps: [Int]
    var rightRamps: [Int]
    var count: Int { starts.count }
}

enum TokenTileGeometry {
    static func single(_ length: Int) -> DimensionIntervals {
        DimensionIntervals(starts: [0], ends: [length], leftRamps: [0], rightRamps: [0])
    }

    /// Port of the oracle's `split_with_symmetric_overlaps` (`tiling.py:307-329`).
    static func splitWithSymmetricOverlaps(size: Int, overlap: Int, dimSize: Int) -> DimensionIntervals {
        if dimSize <= size { return single(dimSize) }
        let amount = (dimSize + size - 2 * overlap - 1) / (size - overlap)
        var starts: [Int] = [], ends: [Int] = []
        for i in 0 ..< amount {
            let s = i * (size - overlap)
            starts.append(s)
            ends.append(s + size)
        }
        ends[amount - 1] = dimSize
        let leftRamps = [0] + Array(repeating: overlap, count: amount - 1)
        let rightRamps = Array(repeating: overlap, count: amount - 1) + [0]
        return DimensionIntervals(starts: starts, ends: ends, leftRamps: leftRamps, rightRamps: rightRamps)
    }

    /// Port of the oracle's `split_by_count` (`tiling.py:662-714`): tile size is
    /// `(dim + overlap·(n−1)) / n`, and the first `remainder` tiles each absorb one extra unit.
    static func splitByCount(numTiles: Int, overlap: Int, dimSize: Int) throws -> DimensionIntervals {
        guard numTiles >= 1 else { throw ModalityTilingError.badConfig("numTiles must be >= 1") }
        guard numTiles <= dimSize else {
            throw ModalityTilingError.badConfig(
                "numTiles (\(numTiles)) exceeds dimension (\(dimSize)) — cannot give every tile ≥1 unit")
        }
        if numTiles == 1 { return single(dimSize) }

        let total = dimSize + overlap * (numTiles - 1)
        let tileSize = total / numTiles
        let remainder = total % numTiles
        guard tileSize > overlap else {
            throw ModalityTilingError.badConfig(
                "overlap (\(overlap)) must be < tile size (\(tileSize)) for \(numTiles) tiles over \(dimSize)")
        }

        let base = splitWithSymmetricOverlaps(size: tileSize, overlap: overlap, dimSize: dimSize - remainder)
        var starts: [Int] = [], ends: [Int] = []
        for i in 0 ..< base.count {
            let shift = min(i, remainder)
            let grow = i < remainder ? 1 : 0
            starts.append(base.starts[i] + shift)
            ends.append(base.ends[i] + shift + grow)
        }
        return DimensionIntervals(starts: starts, ends: ends,
                                  leftRamps: base.leftRamps, rightRamps: base.rightRamps)
    }

    /// Port of `compute_trapezoidal_mask_1d` (`tiling.py:19-56`). Linear fade-in/out at overlaps;
    /// the interior stays 1. Ramps are computed so that contributions across tiles sum to 1.
    static func trapezoidalMask1D(length: Int, rampLeft: Int, rampRight: Int) -> MLXArray {
        precondition(length > 0, "mask length must be positive")
        let rl = max(0, min(rampLeft, length))
        let rr = max(0, min(rampRight, length))
        var mask = MLXArray.ones([length])

        if rl > 0 {
            // linspace(0,1,rl+2)[:-1][1:] — rl interior points, excluding both endpoints.
            let fadeIn = MLX.linspace(Float(0), Float(1), count: rl + 2)[1 ..< (rl + 1)]
            mask = MLX.concatenated([mask[0 ..< rl] * fadeIn, mask[rl...]], axis: 0)
        }
        if rr > 0 {
            // linspace(1,0,rr+2)[1:-1] — rr interior points.
            let fadeOut = MLX.linspace(Float(1), Float(0), count: rr + 2)[1 ..< (rr + 1)]
            mask = MLX.concatenated([mask[0 ..< (length - rr)], mask[(length - rr)...] * fadeOut], axis: 0)
        }
        return MLX.clip(mask, min: 0, max: 1)
    }
}

public enum ModalityTilingError: Error, CustomStringConvertible {
    case badConfig(String)
    case shapeMismatch(String)
    public var description: String {
        switch self {
        case .badConfig(let s): return "modality tiling config: \(s)"
        case .shapeMismatch(let s): return "modality tiling shape: \(s)"
        }
    }
}

// MARK: - A single tile, fully precomputed

public struct TokenTile {
    /// Ranges into the (F, H, W) token grid — the generated tokens this tile owns.
    let f: Range<Int>, h: Range<Int>, w: Range<Int>
    /// Trapezoidal blend weights over this tile's generated tokens, flattened f-major → (nGen,).
    let blendMask: MLXArray
    /// Absolute indices of the conditioning tokens this tile keeps (nil when there are none).
    let condIndices: MLXArray?
    /// `1 / (number of tiles that keep it)` per kept conditioning token, so contributions sum to 1.
    let condWeights: MLXArray?
    let numGen: Int
    let numCond: Int

    public var tokenCount: Int { numGen + numCond }
}

// MARK: - The tiler

/// Slices token-level state into tiles and blends tile outputs back. Stateless after `init`;
/// every per-tile index, mask and weight is computed once here because all of it depends only on
/// the tile layout and the (fixed) positions — never on the latent, sigma, or step.
public struct VideoModalityTiler {
    public let tiles: [TokenTile]
    public let numGeneratedTokens: Int
    public let numTotalTokens: Int
    let latentF: Int, latentH: Int, latentW: Int

    /// - Parameters:
    ///   - tiling: tile counts + overlap per axis, in token-grid units.
    ///   - latentShape: the (F, H, W) token grid — patch size is 1, so this IS the latent grid.
    ///   - positions: `(B, numTotal, 3)` point coords. Needed ONLY to decide which conditioning
    ///     tokens each tile keeps; pass the same positions the DiT will receive. When there are no
    ///     conditioning tokens (`numTotal == F·H·W`) the values are never read.
    public init(tiling: TileCountConfig, latentShape: (F: Int, H: Int, W: Int), positions: MLXArray) throws {
        let (F, H, W) = latentShape
        latentF = F; latentH = H; latentW = W
        numGeneratedTokens = F * H * W

        guard positions.ndim == 3, positions.dim(2) == 3 else {
            throw ModalityTilingError.shapeMismatch("positions must be (B, T, 3), got \(positions.shape)")
        }
        numTotalTokens = positions.dim(1)
        guard numTotalTokens >= numGeneratedTokens else {
            throw ModalityTilingError.shapeMismatch(
                "positions carry \(numTotalTokens) tokens, fewer than the \(numGeneratedTokens) generated")
        }
        let numCondTotal = numTotalTokens - numGeneratedTokens

        let fInt = try TokenTileGeometry.splitByCount(
            numTiles: tiling.frames.numTiles, overlap: tiling.frames.overlap, dimSize: F)
        let hInt = try TokenTileGeometry.splitByCount(
            numTiles: tiling.height.numTiles, overlap: tiling.height.overlap, dimSize: H)
        let wInt = try TokenTileGeometry.splitByCount(
            numTiles: tiling.width.numTiles, overlap: tiling.width.overlap, dimSize: W)

        // Per-axis 1-D trapezoidal masks, built once per interval and reused across the product.
        let fMasks = (0 ..< fInt.count).map {
            TokenTileGeometry.trapezoidalMask1D(length: fInt.ends[$0] - fInt.starts[$0],
                                                rampLeft: fInt.leftRamps[$0], rampRight: fInt.rightRamps[$0])
        }
        let hMasks = (0 ..< hInt.count).map {
            TokenTileGeometry.trapezoidalMask1D(length: hInt.ends[$0] - hInt.starts[$0],
                                                rampLeft: hInt.leftRamps[$0], rampRight: hInt.rightRamps[$0])
        }
        let wMasks = (0 ..< wInt.count).map {
            TokenTileGeometry.trapezoidalMask1D(length: wInt.ends[$0] - wInt.starts[$0],
                                                rampLeft: wInt.leftRamps[$0], rampRight: wInt.rightRamps[$0])
        }

        // ── Conditioning-token keeps, computed ONCE on the host ───────────────────────────────
        // The oracle rebuilds these masks O(T²) times per forward and syncs through NumPy for the
        // bool→index conversion. Both are loop-invariant; doing it here removes the sync entirely.
        // Locals, not property reads: closures below would otherwise capture `self` before
        // `tiles` is assigned, and the host materialization must happen ONCE, not per tile.
        let nGen = numGeneratedTokens
        let gH = H, gW = W
        var condKeepPerTile: [[Int]] = []
        var tileLo: [[Float]] = [], tileHi: [[Float]] = []
        var condPosHost: [Float] = []
        var genPosHost: [Float] = []
        if numCondTotal > 0 {
            let p = positions.asType(.float32)
            eval(p)
            genPosHost = p[0, 0 ..< nGen, 0...].asArray(Float.self)              // (nGen*3,)
            condPosHost = p[0, nGen..., 0...].asArray(Float.self)                // (numCond*3,)
        }

        // Cartesian product in f-major → h → w order (matches `create_tiles`' itertools.product).
        var built: [(f: Range<Int>, h: Range<Int>, w: Range<Int>, mask: MLXArray)] = []
        for fi in 0 ..< fInt.count {
            for hi in 0 ..< hInt.count {
                for wi in 0 ..< wInt.count {
                    let fr = fInt.starts[fi] ..< fInt.ends[fi]
                    let hr = hInt.starts[hi] ..< hInt.ends[hi]
                    let wr = wInt.starts[wi] ..< wInt.ends[wi]
                    // Outer product of the three 1-D masks → (dF, dH, dW) → flattened.
                    let m = fMasks[fi].reshaped(-1, 1, 1)
                        * hMasks[hi].reshaped(1, -1, 1)
                        * wMasks[wi].reshaped(1, 1, -1)
                    built.append((fr, hr, wr, m.reshaped(-1)))

                    if numCondTotal > 0 {
                        // Tile extent in POSITION space, from the generated tokens it owns.
                        // Positions are ordered f-major/h/w-minor, identical to `patchify`.
                        var lo = [Float](repeating: .greatestFiniteMagnitude, count: 3)
                        var hi3 = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
                        for f in fr { for h in hr { for w in wr {
                            let t = (f * gH * gW + h * gW + w) * 3
                            for a in 0 ..< 3 {
                                lo[a] = min(lo[a], genPosHost[t + a])
                                hi3[a] = max(hi3[a], genPosHost[t + a])
                            }
                        }}}
                        var kept: [Int] = []
                        for c in 0 ..< numCondTotal {
                            let base = c * 3
                            // Kept when the point falls inside the tile extent on EVERY axis…
                            var inRange = true
                            for a in 0 ..< 3 where !(condPosHost[base + a] >= lo[a] && condPosHost[base + a] <= hi3[a]) {
                                inRange = false
                            }
                            // …or when it carries a negative time coord (reference tokens).
                            // ⚠️ Faithful to the oracle, but DEAD on our video path: `Positions.video`
                            // clamps time at 0 and the IC downscale never touches axis 0, so our video
                            // reference tokens are always ≥ 0. See TILING-PLAN hazard 2.
                            if inRange || condPosHost[base] < 0 { kept.append(nGen + c) }
                        }
                        condKeepPerTile.append(kept)
                        tileLo.append(lo); tileHi.append(hi3)
                    }
                }
            }
        }

        // How many tiles keep each conditioning token → its blend weight is 1/that.
        var keepCount = [Int](repeating: 0, count: max(numCondTotal, 1))
        for kept in condKeepPerTile {
            for absIdx in kept { keepCount[absIdx - nGen] += 1 }
        }

        // 🚨 COVERAGE GUARANTEE — every conditioning token must be kept by at least one tile.
        //
        // The point-in-range test compares a conditioning token's position against the [min,max]
        // extent of each tile's GENERATED tokens. Those extents are disjoint when overlap == 0, and
        // conditioning positions live on a different grid (IC references are the reference grid's
        // positions scaled by `downscaleFactor` on the spatial axes) — so a token can land in the
        // GAP between two tiles and be kept by NOBODY. It is then never written back: the blended
        // output holds zero there.
        //
        // ⚠️ The oracle has the identical test and does not guard this: its `cond_counts` would be
        // 0 and the weight `1.0 / 0` → **inf**. Its own validation runs at saturating overlap,
        // which is exactly the regime where the gap cannot occur, so the case is invisible there.
        // Caught here by the round-trip identity gate at `overlap: 0`.
        //
        // Fix: assign each orphan to the tile whose extent box is nearest (squared distance to the
        // clamped point), deterministically. Coverage is then total and every weight is finite.
        if numCondTotal > 0 {
            for c in 0 ..< numCondTotal where keepCount[c] == 0 {
                var bestTile = 0
                var bestDist = Float.greatestFiniteMagnitude
                for t in 0 ..< condKeepPerTile.count {
                    var d: Float = 0
                    for a in 0 ..< 3 {
                        let p = condPosHost[c * 3 + a]
                        let clamped = min(max(p, tileLo[t][a]), tileHi[t][a])
                        d += (p - clamped) * (p - clamped)
                    }
                    if d < bestDist { bestDist = d; bestTile = t }
                }
                condKeepPerTile[bestTile].append(nGen + c)
                keepCount[c] = 1
            }
        }

        var out: [TokenTile] = []
        out.reserveCapacity(built.count)
        for (i, b) in built.enumerated() {
            var condIdx: MLXArray?, condW: MLXArray?
            var nCond = 0
            if numCondTotal > 0 {
                let kept = condKeepPerTile[i]
                nCond = kept.count
                if nCond > 0 {
                    condIdx = MLXArray(kept.map { Int32($0) })
                    condW = MLXArray(kept.map { 1.0 / Float(max(keepCount[$0 - nGen], 1)) })
                }
            }
            out.append(TokenTile(f: b.f, h: b.h, w: b.w, blendMask: b.mask,
                                 condIndices: condIdx, condWeights: condW,
                                 numGen: b.f.count * b.h.count * b.w.count, numCond: nCond))
        }
        tiles = out
    }

    /// The largest per-forward token count any tile will produce. **Arm the block-streaming gate
    /// with this, not with the untiled count** — the kit measures per-forward tokens, so arming
    /// with the untiled total means no forward ever reaches the threshold and the gate verdict
    /// stays `.undecided` forever (TILING-PLAN hazard 1).
    public var largestTileTokenCount: Int { tiles.map(\.tokenCount).max() ?? 0 }

    /// Slice `(B, T, D)` token state down to one tile: generated tokens by RANGE (they are a
    /// contiguous block in (F,H,W)), then any kept conditioning tokens by index, concatenated in
    /// that order — matching the layout `blend` expects back.
    public func sliceTokens(_ x: MLXArray, tile: TokenTile) -> MLXArray {
        let B = x.dim(0), D = x.dim(2)
        let grid = x[0..., 0 ..< numGeneratedTokens, 0...].reshaped(B, latentF, latentH, latentW, D)
        let genPart = grid[0..., tile.f, tile.h, tile.w, 0...].reshaped(B, tile.numGen, D)
        guard let condIdx = tile.condIndices else { return genPart }
        let condPart = MLX.take(x, condIdx, axis: 1)
        return MLX.concatenated([genPart, condPart], axis: 1)
    }

    /// Accumulate one tile's output into the full token buffer with trapezoidal weights.
    /// Generated tokens go back by range-accumulate; conditioning tokens by weighted index add.
    public func blend(_ tileOutput: MLXArray, tile: TokenTile, into output: MLXArray) -> MLXArray {
        let B = tileOutput.dim(0), D = tileOutput.dim(2)
        var acc = output
        let genPart = tileOutput[0..., 0 ..< tile.numGen, 0...] * tile.blendMask.reshaped(1, -1, 1).asType(tileOutput.dtype)

        var grid = acc[0..., 0 ..< numGeneratedTokens, 0...].reshaped(B, latentF, latentH, latentW, D)
        grid[0..., tile.f, tile.h, tile.w, 0...] =
            grid[0..., tile.f, tile.h, tile.w, 0...] + genPart.reshaped(B, tile.f.count, tile.h.count, tile.w.count, D)
        acc[0..., 0 ..< numGeneratedTokens, 0...] = grid.reshaped(B, numGeneratedTokens, D)

        if let condIdx = tile.condIndices, let condW = tile.condWeights, tile.numCond > 0 {
            let condPart = tileOutput[0..., tile.numGen..., 0...] * condW.reshaped(1, -1, 1).asType(tileOutput.dtype)
            acc[0..., condIdx, 0...] = MLX.take(acc, condIdx, axis: 1) + condPart
        }
        return acc
    }
}
