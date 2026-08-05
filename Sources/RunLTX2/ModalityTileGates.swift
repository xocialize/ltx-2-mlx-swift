// ModalityTileGates.swift — gates for the DiT-side modality tiler (`Sources/LTX2/ModalityTiling.swift`).
//
// `--modality-tile-gate` — the round-trip identity property.
//
// Slice a token tensor into tiles and blend the slices straight back. The result must reconstruct
// the input EXACTLY, because the trapezoidal ramps of adjacent tiles are complementary
// (fadeOut[j] + fadeIn[j] = 1 by construction) and conditioning tokens are weighted 1/keepCount.
// This is a strong self-contained gate: it exercises tile coverage, the flat-index ↔ (F,H,W)
// mapping, the overlap geometry, the blend weights, and the conditioning keep/weight logic —
// with NO model, NO weights and NO goldens. Anything that breaks the geometry breaks this.
//
// It is deliberately the FIRST gate: the oracle's own validation is the same property
// ("overlap saturates the tile coverage, blend math averages back to identity", PSNR 228 dB).
// ⚠️ Per TILING-PLAN it is not a stress test of tile *boundaries* under a real model — that needs
// the DiT arm, which lands with `TiledDiT`.

import Foundation
import LTX2
import MLX
import MLXRandom

private struct TileCase {
    let name: String
    let F: Int, H: Int, W: Int
    let tf: Int, ts: Int, ov: Int
    let numCond: Int
}

func modalityTileGatesMain(args: [String], positional: [String]) async throws {
    guard args.contains("--modality-tile-gate") else { return }

    let D = 16
    let cases: [TileCase] = [
        .init(name: "identity 1×1×1",            F: 4, H: 6, W: 8, tf: 1, ts: 1, ov: 0, numCond: 0),
        .init(name: "temporal 2 tiles, ov 1",    F: 4, H: 6, W: 8, tf: 2, ts: 1, ov: 1, numCond: 0),
        .init(name: "spatial 2×2, ov 1",         F: 4, H: 6, W: 8, tf: 1, ts: 2, ov: 1, numCond: 0),
        .init(name: "spatial 2×2, ov 2",         F: 4, H: 6, W: 8, tf: 1, ts: 2, ov: 2, numCond: 0),
        .init(name: "full 2×2×2, ov 1",          F: 4, H: 6, W: 8, tf: 2, ts: 2, ov: 1, numCond: 0),
        .init(name: "3 temporal, ov 2",          F: 9, H: 6, W: 8, tf: 3, ts: 1, ov: 2, numCond: 0),
        .init(name: "odd grid 2×2, ov 1",        F: 3, H: 7, W: 5, tf: 1, ts: 2, ov: 1, numCond: 0),
        .init(name: "with 12 cond tokens",       F: 4, H: 6, W: 8, tf: 2, ts: 2, ov: 1, numCond: 12),
        .init(name: "cond + no overlap",         F: 4, H: 6, W: 8, tf: 2, ts: 2, ov: 0, numCond: 12),
    ]

    var allPass = true
    for c in cases {
        let nGen = c.F * c.H * c.W
        let nTotal = nGen + c.numCond
        MLXRandom.seed(UInt64(c.F &* 100 &+ c.H &* 10 &+ c.W))

        // Positions: the real generator for the generated tokens, so the conditioning
        // point-in-range test runs against production-shaped coordinates.
        let genPositions = Positions.video(F: c.F, H: c.H, W: c.W, fps: 24)
        var positions = genPositions
        if c.numCond > 0 {
            // Conditioning tokens scattered across the generated extent (the IC-reference shape:
            // appended at the END of the sequence, positive time — see TILING-PLAN hazard 2).
            let lo = genPositions.min(axis: 1, keepDims: true)
            let hi = genPositions.max(axis: 1, keepDims: true)
            let u = MLXRandom.uniform(low: 0.0, high: 1.0, [1, c.numCond, 3])
            let condPos = lo + u * (hi - lo)
            positions = MLX.concatenated([genPositions, condPos], axis: 1)
        }
        eval(positions)

        let cfg = TileCountConfig(tileFrames: c.tf, tileSpatial: c.ts, overlap: c.ov)
        let tiler: VideoModalityTiler
        do {
            tiler = try VideoModalityTiler(tiling: cfg, latentShape: (F: c.F, H: c.H, W: c.W),
                                           positions: positions)
        } catch {
            print("[modality-tile-gate] \(c.name): CONSTRUCT FAILED ❌ \(error)")
            allPass = false
            continue
        }

        let x = MLXRandom.normal([1, nTotal, D]).asType(.float32)
        eval(x)

        var out = MLXArray.zeros([1, nTotal, D]).asType(.float32)
        for tile in tiler.tiles {
            let part = tiler.sliceTokens(x, tile: tile)
            out = tiler.blend(part, tile: tile, into: out)
        }
        eval(out)

        let genErr = MLX.max(MLX.abs(out[0..., 0 ..< nGen, 0...] - x[0..., 0 ..< nGen, 0...])).item(Float.self)
        var condErr: Float = 0
        if c.numCond > 0 {
            condErr = MLX.max(MLX.abs(out[0..., nGen..., 0...] - x[0..., nGen..., 0...])).item(Float.self)
        }
        let cos = cosine(out, x)

        // fp32 accumulation over a handful of tiles: exact is the expectation, 1e-5 the guard.
        let pass = genErr < 1e-5 && condErr < 1e-5 && cos > 0.999999
        allPass = allPass && pass
        print(String(format: "[modality-tile-gate] %-24@ tiles=%2d  maxTile=%5d tok  gen|Δ|=%.2e cond|Δ|=%.2e cos=%.6f %@",
                     c.name as NSString, tiler.tiles.count, tiler.largestTileTokenCount,
                     Double(genErr), Double(condErr), cos, pass ? "✅" : "❌"))
    }

    // The tiler must refuse layouts it cannot honour rather than silently degrade.
    let pos = Positions.video(F: 2, H: 4, W: 4, fps: 24)
    var refused = false
    do {
        _ = try VideoModalityTiler(tiling: TileCountConfig(tileFrames: 4, tileSpatial: 1, overlap: 0),
                                   latentShape: (F: 2, H: 4, W: 4), positions: pos)
    } catch { refused = true }
    print("[modality-tile-gate] refuses more tiles than units: \(refused ? "✅" : "❌")")
    allPass = allPass && refused

    print(allPass ? "[modality-tile-gate] PASS ✅" : "[modality-tile-gate] FAIL ❌")
    if !allPass { exit(1) }
}
