// NA3D.swift — 3D neighborhood attention + its RoPE, for the LTX-2.5 DiffVAE decoder.
//
// 1:1 functional port of the oracle's `ltx_core_mlx/model/video_vae/na3d.py`, which is
// itself a port of upstream's *eager* NA fallback (`transformer/fallback_na/eager.py`)
// and `transformer/rope_math.py` — deliberately NOT natten / Triton / CuTe, none of
// which exist on Metal.
//
// ## The one fact that makes this simple
//
// The reference's window is SHIFT-CLAMPED, never truncated:
//
//     K     = min(kernel, length)          // kernel clamps to a short axis
//     start = clamp(i - K / 2, 0, length - K)
//     end   = start + K                    // ALWAYS a full-size window
//
// so every query attends to exactly `kt*kh*kw` keys, all in bounds. A gather
// formulation therefore needs no attention mask at all; the box formulation needs one
// only because it shares a contiguous key box across queries whose windows differ.
// Softmax is permutation-invariant over keys, so within-window key order is irrelevant
// as long as K and V are gathered in the same order.
//
// ## Traps encoded here (each one is a silent wrong-output bug)
//
//  * RoPE pairing is GPT-J INTERLEAVED (adjacent element pairs), not NeoX split-half.
//    Split-half produces plausible-looking garbage.
//  * The three axes use DIFFERENT frequency ladders: T over `d_t` dims, H and W over
//    `d_hw` each, with `d_t = (headDim / 4) / 2 * 2` and `d_hw = (headDim - d_t) / 2`
//    — for headDim 64 that is (16, 24, 24).
//  * Positions are plain 0-based ABSOLUTE coordinates of the volume, not
//    window-relative offsets.
//  * RoPE runs in fp32 then casts back.
//  * The `headDim ** -0.5` scale is folded into Q *before* the NA call, so the
//    attention itself runs at `scale = 1.0`.
//
// ## Backends
//
// `box` (default) gathers ONE contiguous key box per query tile — the tile plus a
// `kernel - 1` halo — and hands it to MLX's fused SDPA with a cached boolean window
// mask. `gather` copies each key into every window that contains it and scores with an
// explicit multiply-and-sum. Both are structurally correct; the oracle measured box at
// 9.5e-7 and gather at 2.2e-6 against a CPU-stream dense reference, and box ~6× faster
// end to end. Selected by `LTX2_NA_IMPL`, matching the oracle's env var.

import Foundation
import MLX
import MLXFast

public enum NA3D {

    // MARK: - Backend selection

    public enum Backend: String {
        case box
        case gather
    }

    /// `LTX2_NA_IMPL=box|gather`, defaulting to `box` — same variable and default as the oracle.
    public static var backend: Backend {
        Backend(rawValue: (ProcessInfo.processInfo.environment["LTX2_NA_IMPL"] ?? "box").lowercased())
            ?? .box
    }

    // MARK: - RoPE

    /// `(d_t, d_h, d_w)` — upstream `rope_math.default_rope_dim_split`.
    public static func defaultRopeDimSplit(_ headDim: Int) -> (Int, Int, Int) {
        let dT = (headDim / 4) / 2 * 2
        let dHW = (headDim - dT) / 2
        return (dT, dHW, dHW)
    }

    /// `1 / base ** (2j / dim)` for j in [0, dim/2).
    ///
    /// Built in Double to match upstream's float64 construction before its float32 cast — the
    /// exponent is where a Float ladder would drift.
    public static func ropeInvFreqs(_ dim: Int, base: Double = 10000.0) -> MLXArray {
        let f = stride(from: 0, to: dim, by: 2).map { Float(1.0 / pow(base, Double($0) / Double(dim))) }
        return MLXArray(f)
    }

    /// Rotate the last axis of `x` by `positions` (GPT-J interleaved pairs), in fp32.
    ///
    /// `positions` must broadcast against `x` minus its final (head-dim) axis.
    static func rotateAxis(_ x: MLXArray, _ positions: MLXArray, _ inv: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        var lead = xf.shape
        let d = lead.removeLast()

        let pairs = xf.reshaped(lead + [d / 2, 2])
        let xe = pairs[.ellipsis, 0]
        let xo = pairs[.ellipsis, 1]

        let ang = positions.asType(.float32).expandedDimensions(axis: -1) * inv  // (..., d/2)
        let c = MLX.cos(ang), s = MLX.sin(ang)
        let re = xe * c - xo * s
        let ro = xe * s + xo * c
        return MLX.stacked([re, ro], axis: -1).reshaped(lead + [d]).asType(x.dtype)
    }

    /// Per-axis RoPE on `x` of shape (B, T, H, W, NH, HD).
    ///
    /// Head-dim is partitioned `[0, d_t)` by T, `[d_t, d_t + d_h)` by H, the remainder by W,
    /// each rotated at its own absolute coordinate.
    public static func applyRope3D(_ x: MLXArray,
                                   invT: MLXArray, invH: MLXArray, invW: MLXArray,
                                   dimSplit: (Int, Int, Int)) -> MLXArray {
        let T = x.dim(1), H = x.dim(2), W = x.dim(3), HD = x.dim(5)
        let (dT, dH, dW) = dimSplit
        precondition(dT + dH + dW == HD, "rope split \(dimSplit) != head_dim \(HD)")

        let xt = x[0..., 0..., 0..., 0..., 0..., 0 ..< dT]
        let xh = x[0..., 0..., 0..., 0..., 0..., dT ..< (dT + dH)]
        let xw = x[0..., 0..., 0..., 0..., 0..., (dT + dH)...]

        let pt = MLXArray(0 ..< Int32(T)).reshaped(1, T, 1, 1, 1)
        let ph = MLXArray(0 ..< Int32(H)).reshaped(1, 1, H, 1, 1)
        let pw = MLXArray(0 ..< Int32(W)).reshaped(1, 1, 1, W, 1)

        return MLX.concatenated([rotateAxis(xt, pt, invT),
                                 rotateAxis(xh, ph, invH),
                                 rotateAxis(xw, pw, invW)], axis: -1)
    }

    // MARK: - Windows

    /// Per-index window start along one axis, plus the effective kernel.
    ///
    /// Exact port of the non-causal branch of upstream `eager._window_bounds`.
    public static func windowStarts(_ length: Int, _ kernel: Int) -> (starts: [Int], k: Int) {
        let k = min(kernel, length)
        let lo = length - k
        let half = k / 2
        return ((0 ..< length).map { min(max($0 - half, 0), lo) }, k)
    }

    // MARK: - Dispatch

    /// 3D neighborhood attention on channels-last volumes.
    ///
    /// - Parameters:
    ///   - q, k, v: (B, T, H, W, NH, HD) — already normed, RoPE'd, and with the head-dim
    ///     scale folded into Q.
    ///   - kernel: (kt, kh, kw); each axis clamps to its length.
    ///   - scale: kept at 1.0 by the caller (the scale is folded into Q).
    ///   - tile: query tiling. The box backend reads all three axes; the gather backend
    ///     reads (t, h) and always keeps W whole.
    public static func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                              kernel: (Int, Int, Int),
                              scale: Float = 1.0,
                              tile: (Int, Int, Int) = (2, 32, 96)) -> MLXArray {
        switch backend {
        case .box:
            return box(q, k, v, kernel: kernel, scale: scale, tile: tile)
        case .gather:
            return gather(q, k, v, kernel: kernel, scale: scale,
                          queryTileT: tile.0, queryTileH: tile.1)
        }
    }

    // MARK: - Box backend (default)

    /// Neighborhood attention via shared key boxes + MLX's fused SDPA.
    ///
    /// Where the gather path copies each key into every window that contains it (traffic
    /// `N * prod(kernel) * heads * head_dim`, invariant to tiling), this gathers one
    /// contiguous key box per query tile and lets the fused kernel consume it without ever
    /// materializing the score matrix. Queries then need a boolean mask restricting each to
    /// its true neighborhood.
    ///
    /// Two things make it pay, both measured on the oracle:
    ///
    ///  * MLX's fused SDPA accepts BOOLEAN masks and they are bit-identical to the additive
    ///    form, at a quarter of the bytes on the dominant term.
    ///  * The mask depends only on each query's window *relative to the box corner*, which is
    ///    identical for every interior tile — so a handful of masks are built once and reused
    ///    across all tiles. (This is what upstream's "group tiles by window geometry" does.)
    ///
    /// Modelled traffic at the DiffVAE's stage-5 size: ~135 GB vs ~1140 GB for gather.
    public static func box(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                           kernel: (Int, Int, Int),
                           scale: Float = 1.0,
                           tile: (Int, Int, Int) = (2, 32, 96)) -> MLXArray {
        let B = q.dim(0), T = q.dim(1), H = q.dim(2), W = q.dim(3)
        let NH = q.dim(4), HD = q.dim(5)

        let (startsT, kt) = windowStarts(T, kernel.0)
        let (startsH, kh) = windowStarts(H, kernel.1)
        let (startsW, kw) = windowStarts(W, kernel.2)
        let starts = [startsT, startsH, startsW]
        let ks = [kt, kh, kw]
        let dims = [T, H, W]
        let tiles = [max(1, min(tile.0, T)), max(1, min(tile.1, H)), max(1, min(tile.2, W))]

        let kf = k.reshaped(B, T * H * W, NH, HD).transposed(0, 2, 1, 3)
        let vf = v.reshaped(B, T * H * W, NH, HD).transposed(0, 2, 1, 3)

        var maskCache: [String: MLXArray] = [:]

        var tRows: [MLXArray] = []
        for t0 in stride(from: 0, to: T, by: tiles[0]) {
            let t1 = min(t0 + tiles[0], T)
            var hRows: [MLXArray] = []
            for h0 in stride(from: 0, to: H, by: tiles[1]) {
                let h1 = min(h0 + tiles[1], H)
                var wCols: [MLXArray] = []
                for w0 in stride(from: 0, to: W, by: tiles[2]) {
                    let w1 = min(w0 + tiles[2], W)

                    let begin = [t0, h0, w0], end = [t1, h1, w1]
                    let lo = (0 ..< 3).map { starts[$0][begin[$0]] }
                    let hi = (0 ..< 3).map { starts[$0][end[$0] - 1] + ks[$0] }
                    let ext = (0 ..< 3).map { hi[$0] - lo[$0] }
                    let rel = (0 ..< 3).map { a in (begin[a] ..< end[a]).map { starts[a][$0] - lo[a] } }

                    // Cache on window GEOMETRY, not tile position: every interior tile shares one.
                    let sig = "\(rel)|\(ext)"
                    let m: MLXArray
                    if let cached = maskCache[sig] {
                        m = cached
                    } else {
                        var axisMasks: [MLXArray] = []
                        for a in 0 ..< 3 {
                            let r = MLXArray(rel[a].map { Int32($0) }).reshaped(rel[a].count, 1)
                            let j = MLXArray(0 ..< Int32(ext[a])).reshaped(1, ext[a])
                            axisMasks.append(MLX.logicalAnd(j .>= r, j .< (r + Int32(ks[a]))))
                        }
                        let qShape = rel.map(\.count)
                        let m3 = MLX.logicalAnd(
                            MLX.logicalAnd(
                                axisMasks[0].reshaped(qShape[0], 1, 1, ext[0], 1, 1),
                                axisMasks[1].reshaped(1, qShape[1], 1, 1, ext[1], 1)),
                            axisMasks[2].reshaped(1, 1, qShape[2], 1, 1, ext[2]))
                        let built = m3.reshaped(qShape.reduce(1, *), ext.reduce(1, *))
                        eval(built)
                        maskCache[sig] = built
                        m = built
                    }

                    // key box -> flat token ids, in (t, h, w) order
                    let bt = MLXArray(Int32(lo[0]) ..< Int32(hi[0])).reshaped(ext[0], 1, 1) * Int32(H * W)
                    let bh = MLXArray(Int32(lo[1]) ..< Int32(hi[1])).reshaped(1, ext[1], 1) * Int32(W)
                    let bw = MLXArray(Int32(lo[2]) ..< Int32(hi[2])).reshaped(1, 1, ext[2])
                    let boxIdx = (bt + bh + bw).reshaped(-1)

                    let kb = MLX.take(kf, boxIdx, axis: 2)
                    let vb = MLX.take(vf, boxIdx, axis: 2)
                    let qt = q[0..., t0 ..< t1, h0 ..< h1, w0 ..< w1]
                        .reshaped(B, -1, NH, HD).transposed(0, 2, 1, 3)

                    var o = MLXFast.scaledDotProductAttention(
                        queries: qt, keys: kb, values: vb, scale: scale, mask: .array(m))
                    o = o.transposed(0, 2, 1, 3)
                        .reshaped(B, t1 - t0, h1 - h0, w1 - w0, NH, HD)
                        .asType(q.dtype)
                    eval(o)
                    wCols.append(o)
                }
                hRows.append(wCols.count == 1 ? wCols[0] : MLX.concatenated(wCols, axis: 3))
            }
            tRows.append(hRows.count == 1 ? hRows[0] : MLX.concatenated(hRows, axis: 2))
        }
        return tRows.count == 1 ? tRows[0] : MLX.concatenated(tRows, axis: 1)
    }

    // MARK: - Gather backend (precision reference)

    /// Mask-free 3D neighborhood attention: gather each query's window explicitly.
    ///
    /// Scores with an explicit multiply-and-sum over head_dim rather than a matmul, which is
    /// why it avoids the Apple-GPU fp32 matmul noise the dense reference suffers — at the
    /// cost of copying every key into every window containing it.
    ///
    /// `queryTileT` / `queryTileH` chunk the query axes purely to bound the gathered K/V
    /// working set; each query's window is gathered independently, so any tiling is bit-exact.
    /// Tiling is required on real volumes: the smallest legal latent (3,7,7) already yields a
    /// 17×56×56 stage-5 grid, which at 11³ keys per query would need ~72 GB untiled.
    public static func gather(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                              kernel: (Int, Int, Int),
                              scale: Float = 1.0,
                              queryTileT: Int? = nil,
                              queryTileH: Int? = nil) -> MLXArray {
        let B = q.dim(0), T = q.dim(1), H = q.dim(2), W = q.dim(3)
        let NH = q.dim(4), HD = q.dim(5)

        let (startsT, kt) = windowStarts(T, kernel.0)
        let (startsH, kh) = windowStarts(H, kernel.1)
        let (startsW, kw) = windowStarts(W, kernel.2)

        func axisIndex(_ starts: [Int], _ k: Int) -> MLXArray {
            MLXArray(starts.map { Int32($0) }).reshaped(starts.count, 1) + MLXArray(0 ..< Int32(k)).reshaped(1, k)
        }
        let it = axisIndex(startsT, kt)   // (T, kt)
        let ih = axisIndex(startsH, kh)   // (H, kh)
        let iw = axisIndex(startsW, kw)   // (W, kw)

        let tileT = max(1, min(queryTileT ?? T, T))
        let tileH = max(1, min(queryTileH ?? H, H))

        let kFlat = k.reshaped(B, T * H * W, NH, HD)
        let vFlat = v.reshaped(B, T * H * W, NH, HD)

        var tRows: [MLXArray] = []
        for t0 in stride(from: 0, to: T, by: tileT) {
            let t1 = min(t0 + tileT, T)
            let itS = it[t0 ..< t1]
            var hCols: [MLXArray] = []
            for h0 in stride(from: 0, to: H, by: tileH) {
                let h1 = min(h0 + tileH, H)
                let ihS = ih[h0 ..< h1]
                let tt = t1 - t0, th = h1 - h0

                // ONE flat gather. Three sequential per-axis takes materialize oversized
                // intermediates (the first spans the full H×W plane before narrowing) — the
                // oracle measured that as the dominant cost, and folding them into a single
                // flat token index was a 2.5× end-to-end win.
                let flat = (itS.reshaped(tt, 1, 1, kt, 1, 1) * Int32(H * W)
                            + ihS.reshaped(1, th, 1, 1, kh, 1) * Int32(W)
                            + iw.reshaped(1, 1, W, 1, 1, kw)).reshaped(-1)

                func gatherKV(_ src: MLXArray) -> MLXArray {
                    MLX.take(src, flat, axis: 1)
                        .reshaped(B, tt, th, W, kt * kh * kw, NH, HD)
                        .transposed(0, 1, 2, 3, 5, 4, 6)
                }
                let kk = gatherKV(kFlat)
                let vv = gatherKV(vFlat)
                let qq = q[0..., t0 ..< t1, h0 ..< h1]
                    .expandedDimensions(axis: -2).asType(.float32)

                let scores = (qq * kk.asType(.float32)).sum(axis: -1) * scale
                let probs = MLX.softmax(scores, axis: -1, precise: true)
                let out = (probs.expandedDimensions(axis: -1) * vv.asType(.float32)).sum(axis: -2)
                let piece = out.asType(q.dtype)
                eval(piece)
                hCols.append(piece)
            }
            tRows.append(hCols.count == 1 ? hCols[0] : MLX.concatenated(hCols, axis: 2))
        }
        return tRows.count == 1 ? tRows[0] : MLX.concatenated(tRows, axis: 1)
    }

    // MARK: - Dense reference (tiny volumes only)

    /// Obviously-correct O(N²) reference: dense scores + an explicit window mask.
    ///
    /// Independent of the two production paths (no shared index machinery beyond the window
    /// formula), so agreement is real evidence. Only affordable on tiny volumes.
    ///
    /// ⚠️ Run this on the CPU stream. It is matmul-based, and Apple-GPU fp32 matmul accumulates
    /// ~1e-3 here, which swamps the comparison and looks like a port bug: on the GPU stream it
    /// disagrees with ``gather(_:_:_:kernel:scale:queryTileT:queryTileH:)`` by ~5e-3, on the CPU
    /// stream the two agree to ~2e-6.
    public static func denseReference(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                                      kernel: (Int, Int, Int),
                                      scale: Float = 1.0) -> MLXArray {
        let B = q.dim(0), T = q.dim(1), H = q.dim(2), W = q.dim(3)
        let NH = q.dim(4), HD = q.dim(5)
        let N = T * H * W

        func visible(_ length: Int, _ kernel: Int) -> MLXArray {
            let (starts, k) = windowStarts(length, kernel)
            let s = MLXArray(starts.map { Int32($0) }).reshaped(length, 1)
            let j = MLXArray(0 ..< Int32(length)).reshaped(1, length)
            return MLX.logicalAnd(j .>= s, j .< (s + Int32(k)))
        }
        let vt = visible(T, kernel.0), vh = visible(H, kernel.1), vw = visible(W, kernel.2)
        let vis = MLX.logicalAnd(
            MLX.logicalAnd(vt.reshaped(T, 1, 1, T, 1, 1), vh.reshaped(1, H, 1, 1, H, 1)),
            vw.reshaped(1, 1, W, 1, 1, W)).reshaped(N, N)

        let qf = q.reshaped(B, N, NH, HD).transposed(0, 2, 1, 3).asType(.float32)
        let kfm = k.reshaped(B, N, NH, HD).transposed(0, 2, 1, 3).asType(.float32)
        let vfm = v.reshaped(B, N, NH, HD).transposed(0, 2, 1, 3).asType(.float32)

        var scores = qf.matmul(kfm.transposed(0, 1, 3, 2)) * scale
        scores = MLX.where(vis.reshaped(1, 1, N, N), scores, MLXArray(Float(-1e9)))
        let out = MLX.softmax(scores, axis: -1, precise: true).matmul(vfm)
        return out.transposed(0, 2, 1, 3).reshaped(B, T, H, W, NH, HD).asType(q.dtype)
    }
}
