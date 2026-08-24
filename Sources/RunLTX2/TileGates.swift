// TileGates.swift — spatial-tiling gates for the LTX-2.3 video VAE decoder
// (BLOCKSTREAM-EXPANSION-EVAL §1.4c / §3: the 4K decode gate).
//
// Stage 0 (`--vae-rf-probe`): measure before designing. Two independent probes,
// the way wan-core verified the vae22 suffix RF "three ways":
//   1. Perturbation bounding box — perturb one latent column, decode, and read the
//      spatial support of the output delta at several thresholds. The >0 support is
//      the architectural RF; the decay across thresholds is what makes an
//      approximate halo viable (the temporal precedent: exact RF 13.5, shipped 5).
//   2. Halo-crop seam sweep — split the latent in two, decode each side with a
//      real-neighbour halo, hard-crop, stitch, and compare against the whole
//      decode (cosine, maxAbs, min per-column seam PSNR). The halo where maxAbs
//      hits exactly 0 is the empirical RF cliff. The far-interior maxAbs (>16
//      cells from the seam, beyond any claimed RF) doubles as the MLX
//      shape-invariance diagnostic: nonzero there means conv kernels differ by
//      input width and bit-exact tiling is unreachable regardless of halo.
//
// Weights live on Satechi (PCI-E) — NEVER point these at /Volumes/DEV_ARCHIVE
// (deleted; ISSUES.md I9).

import CryptoKit
import Foundation
import MLX
import MLXRandom
import LTX2
import MLXLTX2

let vaeModelsDir = ProcessInfo.processInfo.environment["LTX_MODELS_DIR"]
    ?? "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx"

func tileGatesMain(args: [String], positional: [String]) async throws {
    let pruna = args.contains("--pruna")
    let ints = positional.compactMap { Int($0) }
    // tiles may be "3" (n×n) or "2x3" (tilesH×tilesW)
    func tileSpec(_ s: String?) -> (Int, Int) {
        let parts = (s ?? "2").split(separator: "x").compactMap { Int($0) }
        let h = parts.first ?? 2
        return (h, parts.count > 1 ? parts[1] : h)
    }
    if args.contains("--vae-rf-probe") {
        try vaeRFProbe(pruna: pruna)
    } else if args.contains("--retake-mask-gate") {
        try retakeMaskGate()
    } else if args.contains("--frozen-sigma-gate") {
        try frozenSigmaGate()
    } else if args.contains("--a2v-contract-gate") {
        try a2vContractGate()
    } else if args.contains("--keyframes-reach-gate") {
        try keyframesReachGate()
    } else if args.contains("--decode-progress-gate") {
        try decodeProgressGate(pruna: pruna)
    } else if args.contains("--vae-encode-tile-probe") {
        try vaeEncodeTileProbe()
    } else if args.contains("--vae-encode-tile-gate") {
        try vaeEncodeTileGate()
    } else if args.contains("--vae-tile-gate") {
        try vaeTileGate(pruna: pruna)
    } else if args.contains("--vae-tile-bench") {
        let spec = positional.first { $0.contains("x") }
        let (th, tw) = tileSpec(spec ?? (ints.count > 3 ? String(ints[3]) : nil))
        try vaeTileBench(widthPx: ints.count > 0 ? ints[0] : 3840,
                         heightPx: ints.count > 1 ? ints[1] : 2176,
                         fLat: ints.count > 2 ? ints[2] : 1,
                         tilesH: th, tilesW: tw,
                         halo: ints.count > 4 && spec == nil ? ints[4] : (ints.count > 3 && spec != nil ? ints[3] : 5),
                         chunk: args.contains("--chunk") ? 8 : 0,
                         untiled: !args.contains("--no-untiled"),
                         pruna: pruna)
    } else if args.contains("--vae-mux-bench") {
        // ONE lane per process (the §1.4d ④ fresh-process discipline):
        //   RunLTX2 --vae-mux-bench <mat|stream> [wPx hPx fLat chunk] [tilesHxW]
        let lane = positional.contains("stream") ? "stream" : "mat"
        let (th, tw) = tileSpec(positional.first { $0.contains("x") } ?? "4x4")
        try await vaeMuxBench(lane: lane,
                              widthPx: ints.count > 0 ? ints[0] : 3840,
                              heightPx: ints.count > 1 ? ints[1] : 2176,
                              fLat: ints.count > 2 ? ints[2] : 15,
                              chunk: ints.count > 3 ? ints[3] : 8,
                              tilesH: th, tilesW: tw, halo: 5, pruna: pruna)
    }
}

// MARK: - --vae-mux-bench

/// The DECODE+ASSEMBLY+MUX phase receipt for the streamed decode→mux seam (GAP #8) at the
/// geometry it was built for: composed 4K decode, where §1.4d ④b measured 99.78 GB with
/// ~18 GB of "parts/assembly" overhead above the tile-window arithmetic. Full-4K DENOISE
/// cannot run on this box (~122k tokens), so any real 4K pipeline is decode-phase-bound —
/// a phase-level A/B here IS the composed-run answer (phase-bound-lever rule, pitfall #56).
///
/// `mat`    = decodeChunked (array) → full pixel volume → encodeMP4 (the default lane)
/// `stream` = decodeChunked(sink:) → MP4StreamWriter.appendSync per chunk (LTX_STREAM_MUX=1 lane)
/// Same decode math, same writer, same H.264 settings → the ENCODER INPUT is bit-identical
/// across lanes (LTX_MUX_PIXHASH=1 digests per-frame floats in append order — measured equal),
/// but the MP4s are NOT comparable byte- or pixel-wise (all measured at 1024×576 fLat5 noise):
///   · whole-file md5 differs even between identical runs of the SAME lane (container
///     creation timestamps);
///   · each lane is decoded-bit-deterministic run-to-run (mat↔mat and stream↔stream both
///     max|Δ|=0), yet mat↔stream decoded diverges (SSIM 0.96 on noise content) — VideoToolbox
///     rate control responds deterministically to append BATCHING (one big appendSync vs
///     per-chunk appends with decode gaps). Same pixels in, different QP schedule out.
/// So the lane-equivalence gate is the PIXHASH, never the MP4. Wall is informational; the
/// deliverable is peakΔ (run peak measurements with pixhash OFF — per-frame host copies
/// would pollute the footprint). Never under MLX_PROFILE.
func vaeMuxBench(lane: String, widthPx: Int, heightPx: Int, fLat: Int, chunk: Int,
                 tilesH: Int, tilesW: Int, halo: Int, pruna: Bool) async throws {
    let dec = try loadDecoder(pruna: pruna)
    let hLat = heightPx / 32, wLat = widthPx / 32
    let frames = 8 * fLat - 7
    MLXRandom.seed(7)
    let latent = MLXRandom.normal([1, 128, fLat, hLat, wLat]).asType(.float32)
    eval(latent)
    print("[vae-mux-bench] lane=\(lane) \(wLat * 32)×\(hLat * 32) fLat=\(fLat) (→\(frames)f) tiles=\(tilesH)×\(tilesW) halo=\(halo) chunk=\(chunk) decoder=\(pruna ? "pruna" : "stock")")
    Memory.clearCache()
    let base = physFootprintBytes()
    let sampler = PhysSampler(); sampler.start(); sampler.resetMax()
    let t = Date()
    // LTX_MUX_PIXHASH=1: per-frame digest of the float pixels entering the writer, in append
    // order — identical digests across lanes proves the encoder INPUT matches and any MP4
    // divergence is encoder-side (rate control vs append batching).
    let pixhash = ProcessInfo.processInfo.environment["LTX_MUX_PIXHASH"] == "1"
    var hasher = Insecure.MD5()
    func hashFrames(_ cl: MLXArray) {   // (1, f, H, W, 3) channels-last
        guard pixhash else { return }
        for i in 0 ..< cl.dim(1) {
            let vals: [Float] = cl[0, i, 0..., 0..., 0...].asArray(Float.self)
            vals.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        }
    }
    let mp4: Data
    if lane == "stream" {
        let writer = try MP4StreamWriter(width: widthPx, height: heightPx, fps: 24, expectAudio: false)
        try dec.decodeChunked(latent, chunkFrames: chunk, halo: 5,
                              spatialTilesH: tilesH, spatialTilesW: tilesW, spatialHalo: halo) { px in
            let cl = px.transposed(0, 2, 3, 4, 1)
            hashFrames(cl)
            try writer.appendSync(framesChunk: cl, totalFrames: frames)
        }
        mp4 = try await writer.finish()
    } else {
        let px = try dec.decodeChunked(latent, chunkFrames: chunk, halo: 5,
                                       spatialTilesH: tilesH, spatialTilesW: tilesW, spatialHalo: halo)
        eval(px)
        let cl = px.transposed(0, 2, 3, 4, 1)
        hashFrames(cl)
        mp4 = try await encodeMP4(frames: cl, fps: 24)
    }
    if pixhash {
        let d = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        print("[vae-mux-bench] pixhash=\(d)")
    }
    let wall = Date().timeIntervalSince(t)
    let peak = sampler.maxBytes(); sampler.stop()
    let md5 = Insecure.MD5.hash(data: mp4).map { String(format: "%02x", $0) }.joined()
    if let save = ProcessInfo.processInfo.environment["LTX_MUX_SAVE"], !save.isEmpty {
        try mp4.write(to: URL(fileURLWithPath: save))
    }
    print(String(format: "[vae-mux-bench] SUMMARY lane=%@ wall=%.1fs peakΔ=%.2f GB base=%.2f GB mp4=%.1f MB md5=%@",
                 lane as NSString, wall, gbOf(peak > base ? peak - base : 0), gbOf(base),
                 Double(mp4.count) / 1e6, md5 as NSString))
}

/// Min per-line PSNR in ±`band` px around each seam. `d` = |tiled − whole| (1,3,F,Hpx,Wpx) fp32.
/// Column seams reduce over (B,C,F,H) → per-column; row seams over (B,C,F,W) → per-row.
private func minSeamPSNR(_ d: MLXArray, rowSeamsPx: [Int], colSeamsPx: [Int], band: Int = 64) -> Float {
    var minP = Float(99)
    func scan(_ mse: [Float]) {
        for m in mse {
            let p = m <= 1e-12 ? Float(99) : 10 * log10(4.0 / m)
            if p < minP { minP = p }
        }
    }
    for c in colSeamsPx {
        let b = d[0..., 0..., 0..., 0..., max(0, c - band) ..< min(d.dim(4), c + band)]
        scan((b * b).mean(axes: [0, 1, 2, 3]).asArray(Float.self))
    }
    for r in rowSeamsPx {
        let b = d[0..., 0..., 0..., max(0, r - band) ..< min(d.dim(3), r + band), 0...]
        scan((b * b).mean(axes: [0, 1, 2, 4]).asArray(Float.self))
    }
    return minP
}

private func seamsPx(_ size: Int, _ n: Int) -> [Int] {
    VideoVAEDecoder.tileBounds(size, n).dropLast().map { $0.1 * 32 }
}

// MARK: - --vae-tile-gate

/// Parity gate: spatially tiled decode vs whole-frame decode (the EXACT in-process reference, the
/// `--vae-chunk-gate` doctrine) + the oracle golden when present. Arms:
///   1. halo 5 (default ship) 2×2 @ 24×40 latent — cosine ≥0.9999 + min seam PSNR ≥60 dB.
///   2. halo 16 — MEMCMP (bit-exact; the measured RF cliff, ⌈15.12⌉).
///   3. composed: temporal chunking × spatial tiling vs whole — same bars as arm 1.
///   4. goldens (stock only): Swift-whole vs oracle-whole ≥0.9999 (cross-binding doctrine), plus
///      the oracle's own blend-tiled output for context.
func vaeTileGate(pruna: Bool) throws {
    let dec = try loadDecoder(pruna: pruna)
    let fLat = 3, hLat = 24, wLat = 40
    let goldenURL = URL(fileURLWithPath: "\(goldensBase)/vae_tile/io.safetensors")
    let golden: [String: MLXArray]? = pruna ? nil : (try? MLX.loadArrays(url: goldenURL))
    let latent: MLXArray
    if let g = golden, let l = g["latent"] {
        latent = l.asType(.float32)
        print("[vae-tile-gate] latent from golden \(goldenURL.path)  \(latent.shape)")
    } else {
        MLXRandom.seed(11)
        latent = MLXRandom.normal([1, 128, fLat, hLat, wLat]).asType(.float32)
        print("[vae-tile-gate] random latent \(latent.shape)\(pruna ? "" : "  (no golden — run parity/dump_vae_tile_goldens.py for the oracle arm)")")
    }
    eval(latent)
    let H = latent.dim(3), W = latent.dim(4)

    let whole = dec.decode(latent)
    eval(whole)
    Memory.clearCache()
    var failed = false

    // Arm 1: default ship config.
    do {
        let tiled = dec.decodeSpatialTiled(latent, tilesH: 2, tilesW: 2, halo: 5)
        eval(tiled)
        guard tiled.shape == whole.shape else {
            print("[vae-tile-gate] FAIL ❌ shape \(tiled.shape) vs \(whole.shape)"); exit(1)
        }
        let d = MLX.abs(tiled - whole)
        let psnr = minSeamPSNR(d, rowSeamsPx: seamsPx(H, 2), colSeamsPx: seamsPx(W, 2))
        let cos = cosine(tiled, whole)
        let ok = cos >= 0.9999 && psnr >= 60
        print(String(format: "[vae-tile-gate] 2×2 halo 5: cosine=%.6f  maxAbs=%.3e  minSeamPSNR=%.1f dB  %@",
                     cos, d.max().item(Float.self), psnr, ok ? "PASS ✅" : "FAIL ❌"))
        failed = failed || !ok
        Memory.clearCache()
    }

    // Arm 2: bit-exact at the measured RF cliff.
    do {
        let tiled = dec.decodeSpatialTiled(latent, tilesH: 2, tilesW: 2, halo: 16)
        eval(tiled)
        let ok = bitEqual(tiled, whole)
        print("[vae-tile-gate] 2×2 halo 16 (RF cliff): memcmp \(ok ? "bit-exact PASS ✅" : "MISMATCH FAIL ❌")")
        failed = failed || !ok
        Memory.clearCache()
    }

    // Arm 3: temporal × spatial composition, on a latent long enough that the temporal halos
    // actually CLIP (F=15, chunk 4, tHalo 5 → interior windows [0,13) / [3,15) — real temporal
    // seams, not whole-clip windows), with spatial tiling inside every chunk.
    do {
        MLXRandom.seed(13)
        let f15 = 15
        let long = MLXRandom.normal([1, 128, f15, hLat, wLat]).asType(.float32)
        eval(long)
        let ref = dec.decode(long)
        eval(ref)
        Memory.clearCache()
        let tiled = try dec.decodeChunked(long, chunkFrames: 4, halo: 5,
                                          spatialTilesH: 2, spatialTilesW: 2, spatialHalo: 5)
        eval(tiled)
        guard tiled.shape == ref.shape else {
            print("[vae-tile-gate] FAIL ❌ composed shape \(tiled.shape) vs \(ref.shape)"); exit(1)
        }
        let d = MLX.abs(tiled - ref)
        // temporal seams: latent chunk boundary b → pixel frame 8b−7 (the chunk-gate convention)
        var tSeamMin = Float(99)
        for b in stride(from: 4, to: f15, by: 4) {
            let seam = 8 * b - 7
            for f in max(0, seam - 2) ... min(ref.dim(2) - 1, seam + 2) {
                let df = d[0..., 0..., f].asType(.float32)
                let mse = (df * df).mean().item(Float.self)
                let p = mse <= 1e-12 ? Float(99) : 10 * log10(4.0 / mse)
                if p < tSeamMin { tSeamMin = p }
            }
        }
        let sPSNR = minSeamPSNR(d, rowSeamsPx: seamsPx(H, 2), colSeamsPx: seamsPx(W, 2))
        let cos = cosine(tiled, ref)
        let ok = cos >= 0.9999 && sPSNR >= 60 && tSeamMin >= 60
        print(String(format: "[vae-tile-gate] composed F15 chunk4×(2×2): cosine=%.6f  spatialSeam=%.1f dB  temporalSeam=%.1f dB  %@",
                     cos, sPSNR, tSeamMin, ok ? "PASS ✅" : "FAIL ❌"))
        failed = failed || !ok
        Memory.clearCache()
    }

    // Arm 4: oracle golden (cross-binding parity for the whole decode; oracle blend for context).
    if let g = golden, let oWhole = g["pixels_whole"] {
        let cos = cosine(whole, oWhole.asType(.float32))
        let ok = cos >= 0.9999
        print(String(format: "[vae-tile-gate] swift-whole vs oracle-whole: cosine=%.6f  %@",
                     cos, ok ? "PASS ✅" : "FAIL ❌"))
        failed = failed || !ok
        if let oTiled = g["pixels_tiled"] {
            let oc = cosine(oTiled.asType(.float32), oWhole.asType(.float32))
            print(String(format: "[vae-tile-gate] (context) oracle blend-tiled vs oracle whole: cosine=%.6f", oc))
        }
    }
    print(failed ? "[vae-tile-gate] FAIL ❌" : "[vae-tile-gate] PASS ✅")
    if failed { exit(1) }
}

// MARK: - --vae-tile-bench

/// The 4K payoff receipt: whole-frame vs spatially tiled decode peak (PhysSampler pattern) at a
/// 4K-class geometry. NEVER under MLX_PROFILE=1 (per-stage evals break fusion and compress the
/// ratio — the PrunaVAED measurement trap).
func vaeTileBench(widthPx: Int, heightPx: Int, fLat: Int, tilesH: Int, tilesW: Int,
                  halo: Int, chunk: Int, untiled: Bool, pruna: Bool) throws {
    let dec = try loadDecoder(pruna: pruna)
    let hLat = heightPx / 32, wLat = widthPx / 32
    MLXRandom.seed(7)
    let latent = MLXRandom.normal([1, 128, fLat, hLat, wLat]).asType(.float32)
    eval(latent)
    print("[vae-tile-bench] \(wLat * 32)×\(hLat * 32) fLat=\(fLat) (→\(8 * fLat - 7) frames)  tiles=\(tilesH)×\(tilesW) halo=\(halo)\(chunk > 0 ? " chunk=\(chunk)" : "")")
    Memory.clearCache()
    let base = physFootprintBytes()
    let sampler = PhysSampler(); sampler.start()

    var whole: MLXArray?
    if untiled {
        sampler.resetMax()
        let t = Date()
        let w = dec.decode(latent); eval(w)
        let peak = sampler.maxBytes()
        print(String(format: "[vae-tile-bench] whole: %.1fs  peakΔ=%.2f GB", Date().timeIntervalSince(t),
                     gbOf(peak > base ? peak - base : 0)))
        whole = w
        Memory.clearCache()
    }

    sampler.resetMax()
    let t = Date()
    let tiled: MLXArray
    if chunk > 0, fLat > chunk {
        tiled = try dec.decodeChunked(latent, chunkFrames: chunk, halo: 5,
                                      spatialTilesH: tilesH, spatialTilesW: tilesW, spatialHalo: halo)
    } else {
        tiled = dec.decodeSpatialTiled(latent, tilesH: tilesH, tilesW: tilesW, halo: halo)
    }
    eval(tiled)
    let peak = sampler.maxBytes(); sampler.stop()
    print(String(format: "[vae-tile-bench] tiled: %.1fs  peakΔ=%.2f GB", Date().timeIntervalSince(t),
                 gbOf(peak > base ? peak - base : 0)))

    if let whole {
        let d = MLX.abs(tiled - whole)
        let psnr = minSeamPSNR(d, rowSeamsPx: seamsPx(hLat, tilesH), colSeamsPx: seamsPx(wLat, tilesW))
        print(String(format: "[vae-tile-bench] tiled vs whole: cosine=%.6f  maxAbs=%.3e  minSeamPSNR=%.1f dB",
                     cosine(tiled, whole), d.max().item(Float.self), psnr))
    }
}

func loadDecoder(pruna: Bool) throws -> VideoVAEDecoder {
    let file = pruna ? "vae_decoder_pruna.safetensors" : "vae_decoder.safetensors"
    let url = URL(fileURLWithPath: "\(vaeModelsDir)/\(file)")
    prewarmFiles([url])
    print("[tile] decoder=\(pruna ? "pruna" : "stock")  weights=\(url.path)")
    return try VideoVAEDecoder.load(path: url)
}

/// First/last index where `v > thr`, or nil if none.
private func support(_ v: [Float], _ thr: Float) -> (Int, Int)? {
    guard let first = v.firstIndex(where: { $0 > thr }),
          let last = v.lastIndex(where: { $0 > thr }) else { return nil }
    return (first, last)
}

func vaeRFProbe(pruna: Bool) throws {
    let dec = try loadDecoder(pruna: pruna)
    let fLat = 2, hLat = 40, wLat = 40
    MLXRandom.seed(7)
    let latent = MLXRandom.normal([1, 128, fLat, hLat, wLat]).asType(.float32)
    eval(latent)
    print("[rf-probe] latent 1×128×\(fLat)×\(hLat)×\(wLat) → pixels \(8 * fLat - 7)×\(hLat * 32)×\(wLat * 32)")

    let t0 = Date()
    let whole = dec.decode(latent)
    eval(whole)
    print(String(format: "[rf-probe] whole decode %.1fs  out=%@", Date().timeIntervalSince(t0),
                 "\(whole.shape)" as NSString))

    // ---- 1. perturbation probe: one latent COLUMN (all channels, all frames) at (cy,cx) ----
    let cy = hLat / 2, cx = wLat / 2
    let delta = MLXArray.zeros([1, 128, fLat, hLat, wLat])
    delta[0..., 0..., 0..., cy ... cy, cx ... cx] = MLXArray(Float(3.0))
    let pert = dec.decode(latent + delta)
    let dmap = MLX.abs(pert - whole).max(axes: [0, 1, 2])  // (H_px, W_px)
    let colMax = dmap.max(axis: 0).asArray(Float.self)     // per-output-column
    let rowMax = dmap.max(axis: 1).asArray(Float.self)     // per-output-row
    // Perturbed cell's own pixel footprint: [c·32, (c+1)·32).
    print("[rf-probe] perturb latent col (y=\(cy), x=\(cx)) by +3.0 → output-delta support:")
    for thr: Float in [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
        guard let (cf, cl) = support(colMax, thr), let (rf, rl) = support(rowMax, thr) else {
            print(String(format: "  thr>%.0e: no support", thr)); continue
        }
        let reachL = Double(cx * 32 - cf) / 32.0, reachR = Double(cl + 1 - (cx + 1) * 32) / 32.0
        let reachU = Double(cy * 32 - rf) / 32.0, reachD = Double(rl + 1 - (cy + 1) * 32) / 32.0
        print(String(format: "  thr>%.0e: W reach %.2f/%.2f cells  H reach %.2f/%.2f cells  (px cols [%d,%d] rows [%d,%d])",
                     thr, reachL, reachR, reachU, reachD, cf, cl, rf, rl))
    }

    // ---- 2. halo-crop seam sweep: two tiles split at W/2, hard crop, stitch ----
    let split = wLat / 2, seamPx = split * 32
    print("[rf-probe] halo-crop sweep (split W at \(split), seam px col \(seamPx)):")
    for h in [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18] {
        let t = Date()
        var l = dec.decode(latent[0..., 0..., 0..., 0..., 0 ..< (split + h)])
        l = l[0..., 0..., 0..., 0..., 0 ..< seamPx]
        var r = dec.decode(latent[0..., 0..., 0..., 0..., (split - h)...])
        r = r[0..., 0..., 0..., 0..., (h * 32)...]
        let stitched = MLX.concatenated([l, r], axis: 4)
        eval(stitched)
        let d = MLX.abs(stitched - whole)
        let ma = d.max().item(Float.self)
        let cos = cosine(stitched, whole)
        // min per-column PSNR in a ±64 px band around the seam
        let band = d[0..., 0..., 0..., 0..., (seamPx - 64) ..< (seamPx + 64)].asType(.float32)
        let mseCols = (band * band).mean(axes: [0, 1, 2, 3]).asArray(Float.self)
        let minPSNR = mseCols.map { $0 <= 1e-12 ? Float(99) : 10 * log10(4.0 / $0) }.min() ?? 99
        // far interior: >16 cells (512 px) from the seam — beyond any claimed RF
        let farL = d[0..., 0..., 0..., 0..., 0 ..< (seamPx - 512)].max().item(Float.self)
        let farR = d[0..., 0..., 0..., 0..., (seamPx + 512)...].max().item(Float.self)
        print(String(format: "  halo %2d: maxAbs=%.3e  cosine=%.6f  minSeamPSNR=%5.1f dB  farInterior=%.3e/%.3e  %.1fs",
                     h, ma, cos, minPSNR, farL, farR, Date().timeIntervalSince(t)))
        Memory.clearCache()
    }
}

// MARK: - --vae-encode-tile-probe (scale sweep: is parity tile-size-limited?)

func vaeEncodeTileProbe() throws {
    let tree = URL(fileURLWithPath: "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
    let enc = try VideoVAEEncoder.load(path: tree.appendingPathComponent("vae_encoder.safetensors"))
    MLXRandom.seed(0x2026_0823)
    func fixture(_ f: Int, _ h: Int, _ w: Int) -> MLXArray {
        let tt = MLXArray(0 ..< f).reshaped(1, 1, f, 1, 1).asType(.float32)
        let yy = MLXArray(0 ..< h).reshaped(1, 1, 1, h, 1).asType(.float32)
        let xx = MLXArray(0 ..< w).reshaped(1, 1, 1, 1, w).asType(.float32)
        let ph = MLXRandom.uniform(low: 0.0, high: Float.pi * 2, [3, 1, 1, 1]).reshaped(1, 3, 1, 1, 1)
        let px = MLX.sin(xx * 0.05 + yy * 0.031 + tt * 0.4 + ph) * 0.6
            + MLX.sin(xx * 0.011 - yy * 0.017 + ph) * 0.35
        eval(px); return px
    }
    // A: spatial-only at VENDOR proportions — 9f, 1088x1920 px = latent 2x34x60, tiles 24/2 → 2x3
    let a = fixture(9, 1088, 1920)
    let refA = enc.encode(a).asType(.float32); eval(refA)
    let tA = enc.encodeTiled(a, tiling: EncodeTiling(
        height: .init(tileSize: 24, overlap: 2), width: .init(tileSize: 24, overlap: 2))).asType(.float32)
    print(String(format: "[probe] SPATIAL vendor 768/64 @1088x1920: cosine=%.6f", cosine(tA, refA)))
    // B: spatial at HALF vendor tile (384px/64) same fixture → artifact fraction doubles
    let tB = enc.encodeTiled(a, tiling: EncodeTiling(
        height: .init(tileSize: 12, overlap: 2), width: .init(tileSize: 12, overlap: 2))).asType(.float32)
    print(String(format: "[probe] SPATIAL 384/64  @1088x1920: cosine=%.6f", cosine(tB, refA)))
    // C: temporal-only at vendor 80/24 (latent 10/3) — 121f, 256x384 px
    let c = fixture(121, 256, 384)
    let refC = enc.encode(c).asType(.float32); eval(refC)
    let tC = enc.encodeTiled(c, tiling: EncodeTiling(frames: .init(tileSize: 10, overlap: 3))).asType(.float32)
    print(String(format: "[probe] TEMPORAL vendor 80/24 @121f: cosine=%.6f", cosine(tC, refC)))
    // D: bigger spatial overlap at small tiles — 192px tiles, overlap 4 latent (128px)
    let d = fixture(33, 256, 384)
    let refD = enc.encode(d).asType(.float32); eval(refD)
    let tD = enc.encodeTiled(d, tiling: EncodeTiling(
        height: .init(tileSize: 6, overlap: 4), width: .init(tileSize: 8, overlap: 4))).asType(.float32)
    print(String(format: "[probe] SPATIAL small 192px ov128: cosine=%.6f", cosine(tD, refD)))
}

// MARK: - --vae-encode-tile-gate (Desktop-parity prerequisite, AB-R-013/// Does `encodeTiled` reproduce the untiled `encode` AT THE SHIPPING CONFIG?
///
/// 🔑 CALIBRATED BY THE SCALE PROBE (`--vae-encode-tile-probe`, 2026-08-23): parity is
/// TILE-SIZE-limited, exactly as the vendor's 64px-overlap minimum implies — the conv edge-artifact
/// zone is a fixed width, so its share of a tile shrinks as tiles grow:
///     768px tiles → 0.999198 · 384px → 0.997816 · 192px → 0.988027 · 192px but ov=128px → 0.999118
/// So the gate pins the VENDOR config at realistic geometry, not a synthetic small-tile layout —
/// a small-tile bar would either fail correct code or pass with a meaninglessly loose threshold.
///
/// ⚠️ TEMPORAL tiling is intrinsically lossier: 0.997107 at the vendor's own 80/24. Each temporal
/// tile restarts causality on its own first frame; the mask zeroes that frame and the weights
/// normalisation heals the ramp, but positions 1..ramp still blend causally-different latents.
/// The vendor SHIPS this — it is the same class of accepted approximation as halo-5 decode
/// (~74 dB). The temporal bar is therefore 0.995 (measured 0.9971 + margin), NOT 0.999, and the
/// difference is doctrine, not sloppiness: parity with the VENDOR's product, per AB-R-0133.
func vaeEncodeTileGate() throws {
    let tree = URL(fileURLWithPath: "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
    let enc = try VideoVAEEncoder.load(path: tree.appendingPathComponent("vae_encoder.safetensors"))
    MLXRandom.seed(0x2026_0823)
    // Smooth low-frequency content, NOT white noise — noise has no spatial coherence, so a seam
    // (a coherence defect) would be invisible in it and the poison arm could pass.
    func fixture(_ f: Int, _ h: Int, _ w: Int) -> MLXArray {
        let tt = MLXArray(0 ..< f).reshaped(1, 1, f, 1, 1).asType(.float32)
        let yy = MLXArray(0 ..< h).reshaped(1, 1, 1, h, 1).asType(.float32)
        let xx = MLXArray(0 ..< w).reshaped(1, 1, 1, 1, w).asType(.float32)
        let ph = MLXRandom.uniform(low: 0.0, high: Float.pi * 2, [3, 1, 1, 1]).reshaped(1, 3, 1, 1, 1)
        let px = MLX.sin(xx * 0.05 + yy * 0.031 + tt * 0.4 + ph) * 0.6
            + MLX.sin(xx * 0.011 - yy * 0.017 + ph) * 0.35
        eval(px); return px
    }

    // 1 — identity fast path: tiling larger than the grid must BE the untiled encode.
    let small = fixture(33, 256, 384)
    let refSmall = enc.encode(small).asType(.float32); eval(refSmall)
    let identity = enc.encodeTiled(small, tiling: EncodeTiling(
        frames: .init(tileSize: 32, overlap: 4), height: .init(tileSize: 32, overlap: 4),
        width: .init(tileSize: 32, overlap: 4)))
    let idMax = MLX.abs(identity.asType(.float32) - refSmall).max().item(Float.self)
    print(String(format: "[vae-encode-tile-gate] identity: maxAbs=%.6f", idMax))

    // 2 — SPATIAL at the vendor config and 1080p-class geometry (2×3 tiles of 768px, ov 64px).
    let hd = fixture(9, 1088, 1920)
    let refHD = enc.encode(hd).asType(.float32); eval(refHD)
    let spatial = enc.encodeTiled(hd, tiling: EncodeTiling(
        height: .init(tileSize: 24, overlap: 2), width: .init(tileSize: 24, overlap: 2))).asType(.float32)
    let cosSpatial = cosine(spatial, refHD)
    print(String(format: "[vae-encode-tile-gate] SPATIAL vendor 768/64 @1088×1920: cosine=%.6f", cosSpatial))

    // 3 — TEMPORAL at the vendor config (80f/24f = latent 10/3) across 121f.
    let long = fixture(121, 256, 384)
    let refLong = enc.encode(long).asType(.float32); eval(refLong)
    let temporal = enc.encodeTiled(long, tiling: EncodeTiling(
        frames: .init(tileSize: 10, overlap: 3))).asType(.float32)
    let cosTemporal = cosine(temporal, refLong)
    print(String(format: "[vae-encode-tile-gate] TEMPORAL vendor 80/24 @121f: cosine=%.6f", cosTemporal))

    // 4 — POISON: zero overlap. No ramps, no healing → seams. If this scores near the parity
    // arms the gate cannot see seams and every number above is meaningless (AB-L-0017).
    let latP = enc.encodeTiled(small, tiling: EncodeTiling(
        frames: .init(tileSize: 3, overlap: 0), height: .init(tileSize: 4, overlap: 0),
        width: .init(tileSize: 6, overlap: 0))).asType(.float32)
    let cosPoison = cosine(latP, refSmall)
    print(String(format: "[vae-encode-tile-gate] POISON ov0: cosine=%.6f", cosPoison))

    let pass = idMax == 0 && cosSpatial >= 0.999 && cosTemporal >= 0.995 && cosPoison < 0.95
    print(pass ? "[vae-encode-tile-gate] PASS ✅" : "[vae-encode-tile-gate] FAIL ❌")
    if !pass { exit(1) }
}

// MARK: - --retake-mask-gate (pure, no weights — AB-R-0133 piece 2)

/// Pins the span→mask derivation to the oracle's TemporalRegionMask semantics: a token is
/// regenerated iff its START time ∈ [start, end). Boundary indices are asserted EXACTLY —
/// hand-derived from the causal frame math — so an off-by-one (midpoint instead of start,
/// inclusive end, feather applied twice) fails on the specific index it corrupts.
func retakeMaskGate() throws {
    var failures: [String] = []
    func check(_ name: String, _ ok: Bool, _ detail: String) {
        print("[retake-mask-gate] \(ok ? "✅" : "❌") \(name) — \(detail)")
        if !ok { failures.append(name) }
    }

    // VIDEO: F=16 latent, H=W=2, fps 24, span [1.0, 3.0).
    // Latent frame k starts at pixel frame max(8k−7, 0) → seconds (8k−7)/24 for k≥1, 0 for k=0.
    // ≥1.0 ⇒ 8k−7 ≥ 24 ⇒ k ≥ 4;  <3.0 ⇒ 8k−7 < 72 ⇒ k ≤ 9.  So frames 4…9 → 6×4 = 24 tokens.
    let vm = RetakeMask.videoSpan(F: 16, H: 2, W: 2, fps: 24, start: 1.0, end: 3.0)
    let vFlat = vm.reshaped(16, 4)
    let perFrame = vFlat.sum(axis: 1)   // (16,) tokens set per latent frame
    eval(perFrame)
    let counts = (0 ..< 16).map { perFrame[$0].item(Float.self) }
    check("video span [1,3) selects latent frames 4…9 exactly",
          counts == [0,0,0,0, 4,4,4,4,4,4, 0,0,0,0,0,0],
          "perFrame=\(counts.map { Int($0) })")

    // AUDIO: T=100 (4 s at 25 tok/s), span [1.0, 3.0).
    // Token i starts at max(4i−3,0)·160/16000 s. ≥1.0 ⇒ i ≥ 26; <3.0 ⇒ i ≤ 75. Count 50.
    let am = RetakeMask.audioSpan(tokens: 100, start: 1.0, end: 3.0)
    eval(am)
    let aTotal = am.sum().item(Float.self)
    let aFirst = am[0, 26, 0].item(Float.self), aBefore = am[0, 25, 0].item(Float.self)
    let aLast = am[0, 75, 0].item(Float.self), aAfter = am[0, 76, 0].item(Float.self)
    check("audio span [1,3) selects tokens 26…75 exactly",
          aTotal == 50 && aFirst == 1 && aBefore == 0 && aLast == 1 && aAfter == 0,
          "total=\(Int(aTotal)) edges=(\(Int(aBefore)),\(Int(aFirst))…\(Int(aLast)),\(Int(aAfter)))")

    // Degenerate spans: whole clip / empty.
    let vAll = RetakeMask.videoSpan(F: 4, H: 1, W: 1, fps: 24, start: 0, end: 1e9)
    let vNone = RetakeMask.videoSpan(F: 4, H: 1, W: 1, fps: 24, start: 5, end: 5)
    eval(vAll, vNone)
    check("whole-span → all ones, empty span → all zeros",
          vAll.sum().item(Float.self) == 4 && vNone.sum().item(Float.self) == 0,
          "all=\(vAll.sum().item(Float.self)) none=\(vNone.sum().item(Float.self))")

    // Frame 0 is the causal image frame (start time 0): a span starting at 0 must include it,
    // a span starting anywhere >0 must not — the boundary the causal fix exists for.
    let v0 = RetakeMask.videoSpan(F: 4, H: 1, W: 1, fps: 24, start: 0, end: 0.01)
    let v1 = RetakeMask.videoSpan(F: 4, H: 1, W: 1, fps: 24, start: 0.001, end: 1)
    eval(v0, v1)
    check("causal frame 0: included iff start == 0",
          v0[0, 0, 0].item(Float.self) == 1 && v1[0, 0, 0].item(Float.self) == 0,
          "start0=\(v0[0, 0, 0].item(Float.self)) start>0=\(v1[0, 0, 0].item(Float.self))")

    print(failures.isEmpty ? "[retake-mask-gate] PASS ✅ (4/4)"
                           : "[retake-mask-gate] FAIL ❌ \(failures.count)")
    if !failures.isEmpty { exit(1) }
}
