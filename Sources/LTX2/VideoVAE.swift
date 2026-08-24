// VideoVAE.swift — LTX-2.3 video VAE decoder (latent → pixels).
//
// 1:1 functional port of ltx_core_mlx/model/video_vae/{video_vae,convolution,
// resnet,sampling,normalization,ops}.py (decode path). 128-ch latent, 8× temporal
// / 32× spatial, pixel-shuffle upsampling, PixelNorm (parameterless), Conv3d.
// Decoder is NON-causal (LTX-2.3: symmetric replicate temporal pad, zeros spatial).
// See docs/ltx-vae-vs-wan-vae.md. Encoder + streaming tiling are follow-ups.

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXProfiling

public struct VideoVAEDecoder {
    let w: [String: MLXArray]

    // up_blocks: ResStage block counts at even indices; DepthToSpace at odd.
    static let resStageBlocks: [Int: Int] = [0: 2, 2: 2, 4: 4, 6: 6, 8: 4]
    // (spatial_factor, temporal_factor) for DepthToSpace at odd indices 1,3,5,7.
    static let upsampleConfig: [(Int, Int)] = [(2, 2), (2, 2), (1, 2), (2, 1)]

    public init(weights: [String: MLXArray]) {
        var m: [String: MLXArray] = [:]
        for (k, v) in weights {
            let key = k.hasPrefix("vae_decoder.") ? String(k.dropFirst("vae_decoder.".count)) : k
            m[key] = v.asType(.float32)
        }
        self.w = m
    }

    public static func load(path: URL) throws -> VideoVAEDecoder {
        VideoVAEDecoder(weights: try MLX.loadArrays(url: path))
    }

    /// latent: (B, C=128, F, H, W) PyTorch layout → pixels (B, 3, F*8-7, H*32, W*32) in [-1,1].
    public func decode(_ latent: MLXArray) -> MLXArray {
        let prof = MLXProfiler.shared
        var x = latent.asType(.float32).transposed(0, 2, 3, 4, 1)  // BFHWC
        x = denormalize(x)
        x = conv3dBlock(x, "conv_in")

        var upIdx = 0
        for i in 0 ..< 9 {
            // Per-up-block attribution (LOW-TIER-PLAN T0). Timing is only honest at an eval, so
            // when profiling we also materialize the even (res) stages — profiler-off behavior is
            // unchanged (evals only between upsample stages, as before).
            let span = prof.begin("vae-decode", "up\(i)", note: "in=\(x.shape)")
            if i % 2 == 0 {
                x = resStage(x, "up_blocks.\(i)", VideoVAEDecoder.resStageBlocks[i]!)
                if prof.enabled { eval(x) }
            } else {
                // Pruned decoders (PrunaVAED) prefix the upsample with a channel adapter;
                // the stock decoder has none, so presence is keyed off the weights.
                if w["up_blocks.\(i).conv_in.norm3.weight"] != nil {
                    x = channelAdapter(x, "up_blocks.\(i).conv_in")
                }
                x = conv3dBlock(x, "up_blocks.\(i).conv")
                let (sf, tf) = VideoVAEDecoder.upsampleConfig[upIdx]
                x = pixelShuffle3d(x, spatialFactor: sf, temporalFactor: tf)
                if tf > 1 { x = x[0..., 1...] }  // drop first frame after temporal upsample
                upIdx += 1
                eval(x)  // materialize between upsample stages (memory)
            }
            prof.end(span)
        }

        let span = prof.begin("vae-decode", "out+unpatchify")
        x = conv3dBlock(silu(pixelNorm(x)), "conv_out")
        x = unpatchifySpatial(x, patchSize: 4)
        x = x.transposed(0, 4, 1, 2, 3)  // BCFHW
        if prof.enabled { eval(x) }
        prof.end(span)
        return x
    }

    /// Spatial halo+crop tiling (BLOCKSTREAM-EXPANSION-EVAL §3 — the 4K decode gate; validated by
    /// `RunLTX2 --vae-tile-gate`). The decoder is spatially LOCAL (channel-axis norms only, pure
    /// pixel-shuffle upsampling — no attention), so a tile decoded with a real-neighbour halo and
    /// hard-cropped matches the whole-frame decode inside the halo's reach: the vae22 method, no
    /// blend, no accumulation buffer. Measured (RF probe, 40×40 latent, stock weights, fp32):
    /// architectural spatial RF = 15.12 latent cells/side; halo 16 is BIT-EXACT (max|Δ|=0 — MLX
    /// convs are shape-invariant across window widths here); influence decays fast, so small halos
    /// are perceptually exact: halo 4 → 67.8 dB min seam PSNR, halo 5 → 74.0 dB (temporal chunking
    /// shipped at 66.2 dB). Default halo 5.
    ///
    /// ⟲ **"4K-only" is RETIRED (2026-08-23, AB-D-0041) — MEASURED at 1080p: −47% peak at 121f
    /// (93.68 → 49.30 GB) and −38% at 241f, perceptually cleared by the operator with SSIM 0.999193
    /// / PSNR 58.19 dB.** At 121f untiled is OVER budget and tiled is within by 40 GB, so here
    /// tiling is a PRECONDITION, not an optimisation.
    /// 🔑 The old note reasoned from the WINDOW-WIDTH fraction (a 2-tile split of a 60-cell axis
    /// still spans ~67% of it) and concluded "marginal". But peak tracks tile **AREA** — 2×2 gives
    /// ~53% of the grid's cells — so one-axis reasoning about a two-dimensional quantity
    /// understated the benefit by roughly the square.
    /// ⚠️ Scope of the clearance: **2×2 only, 1080p only, halo 5 only** (halo 16 is unavailable
    /// there — a 62-cell window against a 60-cell axis), one prompt, one seed. Costs ~9–18% wall,
    /// measured but not claimed (fresh boot, n=1).
    ///
    /// The ORIGINAL arithmetic below is still correct about what it measured — whether tiling pays
    /// for its halo OVERHEAD — which is a different question from how much peak it removes:
    /// at 704×512 the
    /// latent (22×16) is smaller than the bit-exact halo, and even at halo 5 tiling only pays once
    /// window ≪ grid — 1080p is marginal, 4K (120×68) is the target. Crop math is pure (no
    /// temporal analog of drop-first): latent window [w0, w0+win) → keep px [32·(a−w0), 32·(a−w0)
    /// + 32·(b−a)).
    ///
    /// Windows are UNIFORM (the MiniMax-H3 idea): every window is exactly min(L, maxTile+2·halo)
    /// cells, slid inward at the global edges with the crop offset absorbing the shift — one
    /// compiled graph shape across all tile positions instead of a ragged edge shape. The first
    /// window still starts at cell 0 (clamp), so global-edge zero-padding matches the whole-frame
    /// decode exactly.
    /// How many REAL decode windows `decodeSpatialTiled` will run for this grid and tile spec.
    /// Shared by the progress denominator so the count a UI is promised is the count the loop
    /// actually performs — `tileBounds` drops empty tiles, so `tilesH * tilesW` is not always it.
    public static func spatialTileCount(gridH: Int, gridW: Int, tilesH: Int, tilesW: Int) -> Int {
        let nh = max(1, tilesH), nw = max(1, tilesW)
        guard nh > 1 || nw > 1 else { return 1 }
        return tileBounds(gridH, nh).count * tileBounds(gridW, nw).count
    }

    /// `progressTotal > 0` opts into per-tile `.decode` reporting, numbered from `progressBase`.
    /// Each report corresponds to one FINISHED decode window — a real unit of work, not a
    /// synthesised tick (AB-T-0079: a fake heartbeat is worse than an honest indeterminate node,
    /// because a UI renders it as progress).
    public func decodeSpatialTiled(_ latent: MLXArray, tilesH: Int, tilesW: Int, halo: Int,
                                   progressBase: Int = 0, progressTotal: Int = 0) -> MLXArray {
        let nh = max(1, tilesH), nw = max(1, tilesW), h = max(0, halo)
        guard nh > 1 || nw > 1 else {
            let px = decode(latent)
            // Untiled is ONE real window. Reporting 1/1 is honest; inventing more is not.
            if progressTotal > 0 {
                LTX2Progress.report(.decode, step: progressBase + 1, totalSteps: progressTotal)
            }
            return px
        }
        let H = latent.dim(3), W = latent.dim(4)
        let rows = VideoVAEDecoder.tileBounds(H, nh), cols = VideoVAEDecoder.tileBounds(W, nw)
        let winH = min(H, (rows.map { $0.1 - $0.0 }.max() ?? H) + 2 * h)
        let winW = min(W, (cols.map { $0.1 - $0.0 }.max() ?? W) + 2 * h)
        let prof = MLXProfiler.shared
        var rowStrips: [MLXArray] = []
        var tileIndex = 0
        for (r0, r1) in rows {
            var pieces: [MLXArray] = []
            for (c0, c1) in cols {
                let wr0 = max(0, min(r0 - h, H - winH))
                let wc0 = max(0, min(c0 - h, W - winW))
                let span = prof.begin("vae-decode", "tile[\(r0),\(r1))×[\(c0),\(c1))",
                                      note: "window[\(wr0),\(wr0 + winH))×[\(wc0),\(wc0 + winW))")
                var px = decode(latent[0..., 0..., 0..., wr0 ..< (wr0 + winH), wc0 ..< (wc0 + winW)])
                let pr0 = 32 * (r0 - wr0), pc0 = 32 * (c0 - wc0)
                px = px[0..., 0..., 0..., pr0 ..< (pr0 + 32 * (r1 - r0)), pc0 ..< (pc0 + 32 * (c1 - c0))]
                eval(px)
                prof.end(span)
                pieces.append(px)
                tileIndex += 1
                if progressTotal > 0 {
                    LTX2Progress.report(.decode, step: progressBase + tileIndex, totalSteps: progressTotal)
                }
                Memory.clearCache()   // keep the pool tile-bound (the decode's dominant allocation)
            }
            let strip = pieces.count == 1 ? pieces[0] : MLX.concatenated(pieces, axis: 4)
            eval(strip)
            rowStrips.append(strip)
            Memory.clearCache()
        }
        return rowStrips.count == 1 ? rowStrips[0] : MLX.concatenated(rowStrips, axis: 3)
    }

    /// Near-equal contiguous partition of [0, size) into n tiles (drops empties) — the vae22
    /// `tileBounds22` pattern. Public so gates/callers can recover the kept-tile seam positions.
    public static func tileBounds(_ size: Int, _ n: Int) -> [(Int, Int)] {
        let base = size / n, rem = size % n
        var bounds: [(Int, Int)] = []
        var s = 0
        for i in 0 ..< n {
            let e = s + base + (i < rem ? 1 : 0)
            if e > s { bounds.append((s, e)) }
            s = e
        }
        return bounds
    }

    /// LOW-TIER-PLAN T1 seam (validated by `RunLTX2 --vae-chunk-gate`): decode the latent in
    /// TEMPORAL CHUNKS with a halo per side so the decode-stage peak is chunk-bound, not
    /// clip-length-bound. Frame math: F latent frames decode to `8F−7` pixel frames (three
    /// temporal-2× upsamples, each dropping its first frame: F→2F−1→4F−3→8F−7); the −7 is absorbed
    /// entirely at the CLIP start. For a chunk [a,b) decoded with left/right halos (hl,hr):
    ///   window length W = (b−a)+hl+hr → decode gives 8W−7 pixel frames;
    ///   keep [startTrim, len−endTrim) with startTrim = (a==0 ? 0 : 8·hl−7), endTrim = (b==F ? 0 : 8·hr)
    ///   → exactly 8(b−a) interior pixels (8b−7 for the first chunk) — totals 8F−7 across chunks.
    /// The halo absorbs the non-causal receptive field (k=3 convs at 4 temporal scales); its exact
    /// safe minimum is derived empirically against the gate (start ≥1; T0 default 4). Each chunk is
    /// eval'd and the pool cleared, so peak ≈ one chunk's decode.
    ///
    /// Composes with spatial tiling (outer temporal, inner spatial): each temporal window is decoded
    /// via `decodeSpatialTiled`, which returns exactly the shape `decode` would, so the temporal
    /// trim math is untouched, and every spatial window carries the full temporal halo so the two
    /// axes' halos never interact. `spatialTiles* = 1` is the pre-existing behaviour.
    public func decodeChunked(
        _ latent: MLXArray, chunkFrames: Int, halo: Int,
        spatialTilesH: Int = 1, spatialTilesW: Int = 1, spatialHalo: Int = 5
    ) throws -> MLXArray {
        var parts: [MLXArray] = []
        try decodeChunked(latent, chunkFrames: chunkFrames, halo: halo,
                                spatialTilesH: spatialTilesH, spatialTilesW: spatialTilesW,
                                spatialHalo: spatialHalo) { parts.append($0) }
        return parts.count == 1 ? parts[0] : MLX.concatenated(parts, axis: 2)
    }

    /// Sink form of `decodeChunked` — the GAP-ANALYSIS #8 seam. Identical chunk policy, halo and
    /// trim math; each trimmed pixel chunk `(1, 3, f, H, W)` is handed to `sink` in order instead
    /// of being accumulated, so the caller can stream frames to an encoder and the WHOLE pixel
    /// volume never materializes (the oracle's `decode_and_stream` shape). The array form above is
    /// this plus a collecting sink — one code path, so parity of one is parity of both.
    /// ⚠️ The chunk handed to `sink` is already eval'd; the pool is cleared AFTER the sink returns,
    /// so a sink may hold host copies but should not retain the MLXArray past its return.
    public func decodeChunked(
        _ latent: MLXArray, chunkFrames: Int, halo: Int,
        spatialTilesH: Int = 1, spatialTilesW: Int = 1, spatialHalo: Int = 5,
        sink: (MLXArray) throws -> Void
    ) throws {
        let F = latent.dim(2)
        let chunk = max(1, chunkFrames), h = max(1, halo)
        guard F > chunk else {
            let tiles = VideoVAEDecoder.spatialTileCount(gridH: latent.dim(3), gridW: latent.dim(4),
                                                         tilesH: spatialTilesH, tilesW: spatialTilesW)
            let px = decodeSpatialTiled(latent, tilesH: spatialTilesH, tilesW: spatialTilesW,
                                        halo: spatialHalo, progressBase: 0, progressTotal: tiles)
            eval(px)
            try sink(px)
            return
        }
        let prof = MLXProfiler.shared
        let totalChunks = (F + chunk - 1) / chunk
        // ONE denominator for the whole phase: chunks x tiles, counted from the innermost loop.
        // Reporting chunks here AND tiles inside would put two different totals on the same phase
        // and make a stepper jump backwards. Spatial dims are identical across chunks, so
        // tilesPerChunk is constant; when it is 1 this degrades exactly to the old per-chunk
        // cadence, so long-clip runs are unchanged (AB-T-0079).
        let tilesPerChunk = VideoVAEDecoder.spatialTileCount(gridH: latent.dim(3), gridW: latent.dim(4),
                                                             tilesH: spatialTilesH, tilesW: spatialTilesW)
        let totalUnits = totalChunks * tilesPerChunk
        var chunkIndex = 0
        var a = 0
        while a < F {
            try Task.checkCancellation()   // MVP-READINESS M3: per-chunk cancel point
            chunkIndex += 1
            let b = min(a + chunk, F)
            let ws = max(0, a - h), we = min(F, b + h)
            let hl = a - ws, hr = we - b
            let span = prof.begin("vae-decode", "chunk[\(a),\(b))", note: "window[\(ws),\(we)) halo(\(hl),\(hr))")
            var px = decodeSpatialTiled(latent[0..., 0..., ws ..< we],           // (1,3,8W−7,H,W)
                                        tilesH: spatialTilesH, tilesW: spatialTilesW, halo: spatialHalo,
                                        progressBase: (chunkIndex - 1) * tilesPerChunk,
                                        progressTotal: totalUnits)
            let startTrim = (a == 0) ? 0 : 8 * hl - 7
            let endTrim = (b == F) ? 0 : 8 * hr
            px = px[0..., 0..., startTrim ..< (px.dim(2) - endTrim)]
            eval(px)
            prof.end(span)
            try sink(px)
            Memory.clearCache()   // keep the pool chunk-bound
            a = b
        }
    }

    // MARK: - blocks

    private func resStage(_ x0: MLXArray, _ prefix: String, _ n: Int) -> MLXArray {
        var x = x0
        for j in 0 ..< n {
            let p = "\(prefix).res_blocks.\(j)"
            let residual = x
            x = conv3dBlock(silu(pixelNorm(x)), "\(p).conv1")
            x = conv3dBlock(silu(pixelNorm(x)), "\(p).conv2")
            x = x + residual
        }
        return x
    }

    /// Channel-changing residual block — PrunaVAED's `conv_in`, present only on pruned
    /// decoders. Pruna prunes inside each stage but keeps the wider skip between stages;
    /// this reconciles the two widths ahead of the upsample conv.
    ///
    /// Unlike every stock block, the residual path here is *learned*: an affine LayerNorm
    /// over channels (`norm3`, eps 1e-6 — mean-subtracting, so NOT `pixelNorm`) followed by
    /// a 1×1×1 conv, which takes no padding and so bypasses `conv3dBlock`.
    private func channelAdapter(_ x0: MLXArray, _ prefix: String) -> MLXArray {
        var h = conv3dBlock(silu(pixelNorm(x0)), "\(prefix).conv1")
        h = conv3dBlock(silu(pixelNorm(h)), "\(prefix).conv2")
        let r = MLXFast.layerNorm(
            x0,
            weight: w["\(prefix).norm3.weight"]!,
            bias: w["\(prefix).norm3.bias"]!,
            eps: 1e-6
        )
        return h + (conv3d(r, w["\(prefix).conv_shortcut.weight"]!, stride: 1, padding: 0)
            + w["\(prefix).conv_shortcut.bias"]!)
    }

    /// Non-causal Conv3dBlock: symmetric replicate temporal pad + zeros spatial pad, then conv.
    private func conv3dBlock(_ x0: MLXArray, _ prefix: String) -> MLXArray {
        let tk = 3
        let tpad = (tk - 1) / 2  // 1
        var x = x0
        if tpad > 0 {
            let first = MLX.repeated(x[0..., 0 ..< 1], count: tpad, axis: 1)
            let last = MLX.repeated(x[0..., (x.dim(1) - 1)...], count: tpad, axis: 1)
            x = MLX.concatenated([first, x, last], axis: 1)
        }
        // zeros spatial pad (H,W) = 1
        x = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair(1), IntOrPair(1), IntOrPair(0)])
        var y = conv3d(x, w["\(prefix).conv.weight"]!, stride: 1, padding: 0)
        y = y + w["\(prefix).conv.bias"]!
        return y
    }

    private func denormalize(_ x: MLXArray) -> MLXArray {
        let mean = w["per_channel_statistics.mean"]!.reshaped(1, 1, 1, 1, -1)
        let std = w["per_channel_statistics.std"]!.reshaped(1, 1, 1, 1, -1)
        return x * std + mean
    }

    private func pixelNorm(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
        // Affine-free RMS over the channel axis — fused kernel (fp32-internal) replaces the manual
        // square/mean/rsqrt chain.
        MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)]).asType(x.dtype), eps: eps)
    }

    // MARK: - pixel-shuffle helpers

    /// depth-to-space: (B,D,H,W, C·tf·sf²) → (B, D·tf, H·sf, W·sf, C). Split order (c, t, h, w).
    private func pixelShuffle3d(_ x: MLXArray, spatialFactor sf: Int, temporalFactor tf: Int) -> MLXArray {
        let B = x.dim(0), D = x.dim(1), H = x.dim(2), W = x.dim(3), Ct = x.dim(4)
        let C = Ct / (sf * sf * tf)
        var y = x.reshaped(B, D, H, W, C, tf, sf, sf)
        y = y.transposed(0, 1, 5, 2, 6, 3, 7, 4)
        return y.reshaped(B, D * tf, H * sf, W * sf, C)
    }

    /// final spatial unpatchify: (B,F,H,W, C·ps²) → (B,F, H·ps, W·ps, C). Split order (c, r=W, q=H).
    private func unpatchifySpatial(_ x: MLXArray, patchSize ps: Int) -> MLXArray {
        let B = x.dim(0), F = x.dim(1), H = x.dim(2), W = x.dim(3), Ct = x.dim(4)
        let C = Ct / (ps * ps)
        var y = x.reshaped(B, F, H, W, C, ps, ps)   // (B,F,H,W,C,r_W,q_H)
        y = y.transposed(0, 1, 2, 6, 3, 5, 4)        // (B,F,H,q_H,W,r_W,C)
        return y.reshaped(B, F, H * ps, W * ps, C)
    }
}

/// LTX-2.3 video VAE ENCODER (pixels → latent). CAUSAL throughout (unlike the
/// decoder): replicate first frame for temporal pad. patchify(4×4) → conv_in →
/// 9 down_blocks (ResStage / SpaceToDepthDownsample w/ group-mean skip) →
/// PixelNorm→SiLU→conv_out (→129) → first 128 ch → normalize.
public struct VideoVAEEncoder {
    let w: [String: MLXArray]

    static let resStageBlocks: [Int: Int] = [0: 4, 2: 6, 4: 4, 6: 2, 8: 2]
    // SpaceToDepthDownsample at odd indices: (in, out, stride)
    static let downConfig: [Int: (Int, Int, (Int, Int, Int))] = [
        1: (128, 256, (1, 2, 2)), 3: (256, 512, (2, 1, 1)),
        5: (512, 1024, (2, 2, 2)), 7: (1024, 1024, (2, 2, 2)),
    ]

    public init(weights: [String: MLXArray]) {
        var m: [String: MLXArray] = [:]
        for (k, v) in weights {
            let key = k.hasPrefix("vae_encoder.") ? String(k.dropFirst("vae_encoder.".count)) : k
            m[key] = v.asType(.float32)
        }
        self.w = m
    }

    public static func load(path: URL) throws -> VideoVAEEncoder {
        VideoVAEEncoder(weights: try MLX.loadArrays(url: path))
    }

    /// pixels (B,3,F,H,W) in [-1,1] → latent (B,128,F',H',W'), 8×temporal/32×spatial.
    public func encode(_ pixels: MLXArray) -> MLXArray {
        var x = pixels.asType(.float32).transposed(0, 2, 3, 4, 1)  // BFHWC
        x = patchifySpatial(x, patchSize: 4)                       // 3 → 48
        x = convCausal(x, "conv_in")
        for i in 0 ..< 9 {
            if i % 2 == 0 {
                x = resStage(x, "down_blocks.\(i)", VideoVAEEncoder.resStageBlocks[i]!)
            } else {
                let (inc, outc, st) = VideoVAEEncoder.downConfig[i]!
                x = spaceToDepthDownsample(x, "down_blocks.\(i)", inChannels: inc, outChannels: outc, stride: st)
                eval(x)
            }
        }
        x = convCausal(silu(pixelNorm(x)), "conv_out")  // 1024 → 129
        x = x[.ellipsis, 0 ..< 128]                      // mean channels
        x = normalize(x)
        return x.transposed(0, 4, 1, 2, 3)               // BCFHW
    }

    // MARK: - blocks

    private func resStage(_ x0: MLXArray, _ prefix: String, _ n: Int) -> MLXArray {
        var x = x0
        for j in 0 ..< n {
            let p = "\(prefix).res_blocks.\(j)"
            let residual = x
            x = convCausal(silu(pixelNorm(x)), "\(p).conv1")
            x = convCausal(silu(pixelNorm(x)), "\(p).conv2")
            x = x + residual
        }
        return x
    }

    private func spaceToDepthDownsample(_ x0: MLXArray, _ prefix: String, inChannels: Int, outChannels: Int, stride st: (Int, Int, Int)) -> MLXArray {
        var x = x0
        if st.0 == 2 {  // causal temporal: prepend first frame
            x = MLX.concatenated([x[0..., 0 ..< 1], x], axis: 1)
        }
        // skip: space-to-depth → group-mean to out_channels
        var xIn = spaceToDepth(x, stride: st)
        let groupSize = inChannels * st.0 * st.1 * st.2 / outChannels
        if groupSize > 1 {
            let B = xIn.dim(0), D = xIn.dim(1), H = xIn.dim(2), W = xIn.dim(3), Ct = xIn.dim(4)
            xIn = xIn.reshaped(B, D, H, W, Ct / groupSize, groupSize)
            xIn = MLX.mean(xIn, axis: -1)
        }
        // conv branch: conv (stride 1) → space-to-depth
        var xConv = convCausal(x, "\(prefix).conv")
        xConv = spaceToDepth(xConv, stride: st)
        return xConv + xIn
    }

    /// Causal Conv3dBlock: replicate first frame (k-1) at front, zeros spatial pad, conv.
    private func convCausal(_ x0: MLXArray, _ prefix: String) -> MLXArray {
        let tk = 3
        var x = x0
        let first = MLX.repeated(x[0..., 0 ..< 1], count: tk - 1, axis: 1)
        x = MLX.concatenated([first, x], axis: 1)
        x = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair(1), IntOrPair(1), IntOrPair(0)])
        var y = conv3d(x, w["\(prefix).conv.weight"]!, stride: 1, padding: 0)
        y = y + w["\(prefix).conv.bias"]!
        return y
    }

    /// Reverse per-channel normalization on a (B,C,F,H,W) latent → (B,C,F,H,W).
    /// Used by the two-stage upsampler: the neural upsampler operates in
    /// un-normalized latent space (denormalize → upsample → normalize).
    public func denormalizeLatent(_ latent: MLXArray) -> MLXArray {
        let x = latent.asType(.float32).transposed(0, 2, 3, 4, 1)  // NHWC
        let mean = w["per_channel_statistics._mean_of_means"]!.reshaped(1, 1, 1, 1, -1)
        let std = w["per_channel_statistics._std_of_means"]!.reshaped(1, 1, 1, 1, -1)
        return (x * std + mean).transposed(0, 4, 1, 2, 3)
    }

    /// Apply per-channel normalization on a (B,C,F,H,W) latent → (B,C,F,H,W).
    public func normalizeLatent(_ latent: MLXArray) -> MLXArray {
        let x = latent.asType(.float32).transposed(0, 2, 3, 4, 1)  // NHWC
        let mean = w["per_channel_statistics._mean_of_means"]!.reshaped(1, 1, 1, 1, -1)
        let std = w["per_channel_statistics._std_of_means"]!.reshaped(1, 1, 1, 1, -1)
        return ((x - mean) / std).transposed(0, 4, 1, 2, 3)
    }

    private func normalize(_ x: MLXArray) -> MLXArray {
        let mean = w["per_channel_statistics._mean_of_means"]!.reshaped(1, 1, 1, 1, -1)
        let std = w["per_channel_statistics._std_of_means"]!.reshaped(1, 1, 1, 1, -1)
        return (x - mean) / std
    }

    private func pixelNorm(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
        // Affine-free RMS over the channel axis — fused kernel (fp32-internal) replaces the manual
        // square/mean/rsqrt chain.
        MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)]).asType(x.dtype), eps: eps)
    }

    /// space-to-depth: (B,D,H,W,C) → (B, D/st, H/sh, W/sw, C·st·sh·sw). Order (c, t, h, w).
    private func spaceToDepth(_ x: MLXArray, stride st: (Int, Int, Int)) -> MLXArray {
        let B = x.dim(0), Df = x.dim(1), Hf = x.dim(2), Wf = x.dim(3), C = x.dim(4)
        let D = Df / st.0, H = Hf / st.1, W = Wf / st.2
        var y = x.reshaped(B, D, st.0, H, st.1, W, st.2, C)
        y = y.transposed(0, 1, 3, 5, 7, 2, 4, 6)   // (B,D,H,W,C,st,sh,sw)
        return y.reshaped(B, D, H, W, C * st.0 * st.1 * st.2)
    }

    /// spatial patchify: (B,F,H,W,C) → (B,F, H/ps, W/ps, C·ps²). Order (c, r=W, q=H).
    private func patchifySpatial(_ x: MLXArray, patchSize ps: Int) -> MLXArray {
        let B = x.dim(0), F = x.dim(1), H = x.dim(2), W = x.dim(3), C = x.dim(4)
        var y = x.reshaped(B, F, H / ps, ps, W / ps, ps, C)  // 0B,1F,2H/ps,3q_H,4W/ps,5r_W,6C
        y = y.transposed(0, 1, 2, 4, 6, 5, 3)                // (B,F,H/ps,W/ps,C,r_W,q_H)
        return y.reshaped(B, F, H / ps, W / ps, C * ps * ps)
    }
}

// MARK: - Tiled ENCODE (Desktop-parity prerequisite — AB-R-0133)
//
// Port of ltx_core `VideoEncoder.tiled_encode` + `prepare_tiles_for_encoding`: split on the
// LATENT grid, encode overlapping PIXEL tiles independently through the normal causal forward,
// then feather-blend the latent tiles with separable 1-D trapezoid masks and normalize by the
// accumulated weights.
//
// 🔑 WHY BLEND, NOT HALO-CROP (our decode idiom): the oracle blends, and retake conditions on
// latents encoded this way — matching the vendor keeps seam behaviour and numbers comparable.
// The causal boundary self-heals by construction: a temporal tile that starts mid-video re-runs
// the causal front-pad on ITS OWN first frame (wrong history), and the temporal mask's
// `leftStartsFromZero` zeroes exactly that first latent position, so the previous tile's ramp
// supplies it instead.
//
// ⚠️ Deviation from the oracle, deliberate: we ALWAYS accumulate a weights buffer and divide.
// The oracle skips it when masks are provably complementary (interior spatial ramps sum to 1);
// dividing by exactly-1.0 is a no-op, and this spares porting `masks_are_complementary`. The
// temporal causal ramps are NOT complementary, so the division is load-bearing there.

/// Per-axis tiling in LATENT units. `tileSize == 0` → untiled on that axis.
public struct EncodeTileAxis: Sendable, Equatable {
    public let tileSize: Int
    public let overlap: Int
    public init(tileSize: Int = 0, overlap: Int = 0) {
        self.tileSize = tileSize; self.overlap = overlap
    }
}

public struct EncodeTiling: Sendable, Equatable {
    public let frames: EncodeTileAxis   // latent frames (pixel frames = 1 + 8·(t−1))
    public let height: EncodeTileAxis   // latent rows   (× 32 px)
    public let width: EncodeTileAxis    // latent cols   (× 32 px)
    public init(frames: EncodeTileAxis = .init(), height: EncodeTileAxis = .init(),
                width: EncodeTileAxis = .init()) {
        self.frames = frames; self.height = height; self.width = width
    }
    /// The vendor default (`TileSizeConfig.default()`): 80f/24f · 768px/64px · 768px/64px,
    /// expressed on the latent grid (÷8 temporal, ÷32 spatial) = 10/3 · 24/2 · 24/2.
    /// ⚠️ Their guard: encode needs ≥16 frames / ≥64 px of overlap to bury the symmetric-pad
    /// edge artifacts — on the latent grid that is ≥2 everywhere. Keep overlaps ≥2.
    public static let vendorDefault = EncodeTiling(
        frames: .init(tileSize: 10, overlap: 3),
        height: .init(tileSize: 24, overlap: 2),
        width: .init(tileSize: 24, overlap: 2))
}

struct EncodeInterval { let start: Int, end: Int, leftRamp: Int, rightRamp: Int }

/// `split_by_size` — first tile ramps (0, ov), interior (ov, ov), last (ov, 0); the last tile
/// may be short. Untiled (or dim ≤ size) → one full-span interval with no ramps.
func encodeIntervals(dim: Int, axis: EncodeTileAxis) -> [EncodeInterval] {
    let size = axis.tileSize, ov = axis.overlap
    guard size > 0, dim > size else { return [EncodeInterval(start: 0, end: dim, leftRamp: 0, rightRamp: 0)] }
    precondition(ov >= 0 && ov < size, "overlap must satisfy 0 <= overlap < tileSize")
    let amount = (dim + size - 2 * ov - 1) / (size - ov)
    var out = [EncodeInterval(start: 0, end: size, leftRamp: 0, rightRamp: ov)]
    for i in 1 ..< max(1, amount - 1) {
        out.append(EncodeInterval(start: i * (size - ov), end: i * (size - ov) + size,
                                  leftRamp: ov, rightRamp: ov))
    }
    if amount > 1 {
        out.append(EncodeInterval(start: (amount - 1) * (size - ov), end: dim,
                                  leftRamp: ov, rightRamp: 0))
    }
    return out
}

/// `compute_trapezoidal_mask_1d` — linear fade over the ramps. Spatial ramps exclude both
/// endpoints (values k/(ov+1)), so opposing ramps sum to exactly 1; the temporal causal variant
/// (`leftStartsFromZero`) includes 0 (values k/ov), zeroing the tile's causally-wrong first
/// latent frame outright.
func trapezoidMask(length: Int, leftRamp: Int, rightRamp: Int, leftStartsFromZero: Bool) -> [Float] {
    var m = [Float](repeating: 1, count: length)
    let l = max(0, min(leftRamp, length)), r = max(0, min(rightRamp, length))
    if l > 0 {
        for k in 0 ..< l {
            m[k] = leftStartsFromZero ? Float(k) / Float(l) : Float(k + 1) / Float(l + 1)
        }
    }
    if r > 0 {
        for k in 0 ..< r { m[length - r + k] *= Float(r - k) / Float(r + 1) }
    }
    return m
}

extension VideoVAEEncoder {
    /// Tiled encode: pixels (B,3,F,H,W) in [-1,1] → latent (B,128,F',H',W'), numerically ≈ the
    /// untiled `encode` (interior tiles are bit-identical work; only the blend ramps differ).
    public func encodeTiled(_ pixels: MLXArray, tiling: EncodeTiling = .vendorDefault) -> MLXArray {
        let f = pixels.dim(2), h = pixels.dim(3), w = pixels.dim(4)
        precondition((f - 1) % 8 == 0 && h % 32 == 0 && w % 32 == 0,
                     "encodeTiled needs F=8k+1 and H,W multiples of 32 (got \(f)×\(h)×\(w))")
        let lt = (f - 1) / 8 + 1, lh = h / 32, lw = w / 32
        let tIv = encodeIntervals(dim: lt, axis: tiling.frames)
        let hIv = encodeIntervals(dim: lh, axis: tiling.height)
        let wIv = encodeIntervals(dim: lw, axis: tiling.width)
        if tIv.count == 1 && hIv.count == 1 && wIv.count == 1 { return encode(pixels) }

        let B = pixels.dim(0)
        var latentSum = MLXArray.zeros([B, 128, lt, lh, lw])
        var weightSum = MLXArray.zeros([B, 128, lt, lh, lw])
        for t in tIv { for y in hIv { for x in wIv {
            // latent interval → pixel in_coords (map_temporal_slice / map_spatial_slice)
            let pf0 = t.start * 8, pf1 = 1 + (t.end - 1) * 8
            let tile = pixels[0..., 0..., pf0 ..< pf1, (y.start * 32) ..< (y.end * 32),
                              (x.start * 32) ..< (x.end * 32)]
            var lat = encode(tile).asType(.float32)           // (B,128,t,h,w)
            let mt = MLXArray(trapezoidMask(length: t.end - t.start, leftRamp: t.leftRamp,
                                            rightRamp: t.rightRamp, leftStartsFromZero: true))
            let mh = MLXArray(trapezoidMask(length: y.end - y.start, leftRamp: y.leftRamp,
                                            rightRamp: y.rightRamp, leftStartsFromZero: false))
            let mw = MLXArray(trapezoidMask(length: x.end - x.start, leftRamp: x.leftRamp,
                                            rightRamp: x.rightRamp, leftStartsFromZero: false))
            let mask = mt.reshaped(1, 1, t.end - t.start, 1, 1)
                * mh.reshaped(1, 1, 1, y.end - y.start, 1)
                * mw.reshaped(1, 1, 1, 1, x.end - x.start)
            lat = lat * mask
            latentSum[0..., 0..., t.start ..< t.end, y.start ..< y.end, x.start ..< x.end] += lat
            weightSum[0..., 0..., t.start ..< t.end, y.start ..< y.end, x.start ..< x.end] += mask
            eval(latentSum, weightSum)                         // watchdog + free the tile graph
        } } }
        return latentSum / MLX.maximum(weightSum, MLXArray(Float(1e-8)))
    }
}
