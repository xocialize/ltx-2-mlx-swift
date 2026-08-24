// DecodeProgressGate.swift — `--decode-progress-gate` (AB-T-0079).
//
// THE DEAD ZONE. Decode used to report per-CHUNK only, and chunking engages only when
// `latentFrames > vaeChunkFrames + 2*halo`. At 1080p x 121f that is 16 > 18 — FALSE. So the phase
// went silent exactly at the geometry where AB-R-0118 measured the peak as DECODE-BOUND: in a
// stepper UI "Render frames" was the node most likely to look stuck and the least able to prove it
// wasn't.
//
// The root cause is worth restating because it is the transferable part: a FRAME-COUNT threshold
// was gating a phase whose cost is RESOLUTION-driven. Two unrelated axes, so the gap opened
// precisely at high-res/short-clip.
//
// The fix is NOT a new counter. AB-T-0079's constraint is explicit — `RunPhaseReport` carries no
// payload, so sub-steps must be real countable units, never synthesised ticks, because a UI renders
// a fake heartbeat as progress. The real units were already there: the SPATIAL TILES the decode
// runs anyway (2x2 = 4 windows at 1080p). Decode now counts decode WINDOWS = chunks x tiles.
//
// This gate runs the REAL decoder, because the claim is "reports fire", and arithmetic alone cannot
// establish that.

import Foundation
import MLX
import MLXRandom
import LTX2

/// Collects `.decode` reports in order. Lock-guarded because `LTX2Progress.Sink` is `@Sendable`.
private final class DecodeTally: @unchecked Sendable {
    private let lock = NSLock()
    private var _steps: [Int] = []
    private var _totals: [Int] = []
    var steps: [Int] { lock.lock(); defer { lock.unlock() }; return _steps }
    var totals: [Int] { lock.lock(); defer { lock.unlock() }; return _totals }
    var sink: LTX2Progress.Sink {
        { [self] e in
            guard e.phase == .decode else { return }
            lock.lock(); defer { lock.unlock() }
            if let s = e.step { _steps.append(s) }
            if let t = e.totalSteps { _totals.append(t) }
        }
    }
}

func decodeProgressGate(pruna: Bool) throws {
    var fails: [String] = []

    // ── 1. THE DEAD ZONE IS REAL, and the fix's units exist there. Pure arithmetic on the
    //       shipping rule, stated for the geometry the ticket names.
    let chunk = 8, halo = 5, sHalo = 5
    // 1920x1088 x 121f → latent grid 34x60, 16 latent frames.
    let hdFLat = (121 + 7) / 8, hdGridH = 1088 / 32, hdGridW = 1920 / 32
    let hdChunks = hdFLat > chunk + 2 * halo
    if hdChunks {
        fails.append("case 1: 1080p x 121f now chunks (fLat \(hdFLat) vs \(chunk + 2 * halo)) — the "
            + "premise of this gate changed; re-derive the dead zone before trusting it")
    }
    let hdAuto = (hdGridW > 4 * sHalo && hdGridH > 4 * sHalo) ? 2 : 1
    let hdTiles = VideoVAEDecoder.spatialTileCount(gridH: hdGridH, gridW: hdGridW,
                                                   tilesH: hdAuto, tilesW: hdAuto)
    if hdTiles < 2 {
        fails.append("case 1: 1080p decode has \(hdTiles) countable window(s) — nothing to report, "
            + "so the dead zone is not closed")
    }
    print("[decode-progress-gate] case 1 1080p×121f: chunks=\(hdChunks ? "yes" : "NO (dead zone)") "
        + "· grid \(hdGridH)×\(hdGridW) → \(hdTiles) decode windows")

    // 704x512 is the honest single-window case: grid 22x16, and 16 <= 4*5, so NO tiling.
    let sdTiles = VideoVAEDecoder.spatialTileCount(gridH: 512 / 32, gridW: 704 / 32,
                                                   tilesH: 1, tilesW: 1)
    if sdTiles != 1 {
        fails.append("case 1b: 704x512 should be exactly 1 window, got \(sdTiles)")
    }

    // ── 2. REPORTS ACTUALLY FIRE, on the real decoder. 24x40 latent grid tiles 2x2 (both axes
    //       exceed 4*halo), the same geometry `--vae-tile-gate` uses.
    let dec = try loadDecoder(pruna: pruna)
    MLXRandom.seed(11)
    let latent = MLXRandom.normal([1, 128, 3, 24, 40]).asType(.float32)
    eval(latent)

    let tiles = VideoVAEDecoder.spatialTileCount(gridH: 24, gridW: 40, tilesH: 2, tilesW: 2)
    let t2 = DecodeTally()
    LTX2Progress.$sink.withValue(t2.sink) {
        let px = dec.decodeSpatialTiled(latent, tilesH: 2, tilesW: 2, halo: 5,
                                        progressBase: 0, progressTotal: tiles)
        eval(px)
    }
    if t2.steps != Array(1 ... tiles) {
        fails.append("case 2: tiled decode reported steps \(t2.steps), expected 1...\(tiles) — one "
            + "report per finished decode window, in order")
    }
    if Set(t2.totals) != [tiles] {
        fails.append("case 2: denominator moved during the phase (\(Set(t2.totals).sorted())); a "
            + "stepper would jump")
    }
    print("[decode-progress-gate] case 2 tiled 2×2 on the REAL decoder: steps \(t2.steps) of \(tiles)")

    // ── 3. NO MANUFACTURED TICKS. Untiled is ONE real window, so it must report exactly 1/1 —
    //       not zero (the old dead zone) and not a synthesised stream.
    let t1 = DecodeTally()
    LTX2Progress.$sink.withValue(t1.sink) {
        let px = dec.decodeSpatialTiled(latent, tilesH: 1, tilesW: 1, halo: 5,
                                        progressBase: 0, progressTotal: 1)
        eval(px)
    }
    if t1.steps != [1] || t1.totals != [1] {
        fails.append("case 3: untiled decode reported steps \(t1.steps)/totals \(t1.totals), "
            + "expected exactly one honest 1/1")
    }
    print("[decode-progress-gate] case 3 untiled: \(t1.steps) of \(t1.totals) (one real window)")

    // ── 4. COMPOSED chunks x tiles: ONE denominator across the whole phase, strictly increasing.
    //       Two counters on one phase (per-chunk AND per-tile) would make a UI jump backwards.
    MLXRandom.seed(12)
    let longLatent = MLXRandom.normal([1, 128, 9, 24, 40]).asType(.float32)
    eval(longLatent)
    let cFrames = 4
    let expectedChunks = (9 + cFrames - 1) / cFrames
    let tc = DecodeTally()
    try LTX2Progress.$sink.withValue(tc.sink) {
        _ = try dec.decodeChunked(longLatent, chunkFrames: cFrames, halo: 1,
                                  spatialTilesH: 2, spatialTilesW: 2, spatialHalo: 5)
    }
    let expectedUnits = expectedChunks * tiles
    if tc.steps != Array(1 ... expectedUnits) {
        fails.append("case 4: composed decode reported \(tc.steps), expected 1...\(expectedUnits) "
            + "(\(expectedChunks) chunks × \(tiles) tiles) with no repeats or gaps")
    }
    if Set(tc.totals) != [expectedUnits] {
        fails.append("case 4: composed denominator is not constant (\(Set(tc.totals).sorted())) — "
            + "chunk and tile counters are both reporting")
    }
    print("[decode-progress-gate] case 4 composed \(expectedChunks) chunks × \(tiles) tiles: "
        + "\(tc.steps.count) reports 1...\(tc.steps.last ?? 0) of \(Set(tc.totals).sorted())")

    if fails.isEmpty {
        print("[decode-progress-gate] PASS ✅  4/4 — decode counts real decode WINDOWS "
            + "(chunks × tiles), fires at HD where it used to be silent, and never invents a tick")
        fflush(stdout)
    } else {
        for f in fails { print("[decode-progress-gate] FAIL — \(f)") }
        fflush(stdout); exit(1)
    }
}
