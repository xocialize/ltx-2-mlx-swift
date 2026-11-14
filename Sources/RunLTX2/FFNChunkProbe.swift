// FFNChunkProbe.swift — `--ffn-chunk-probe`, the per-forward A/B for `DiT.ffChunking` (AB-T-0484).
//
// WHAT IT ANSWERS. Whether running the block feed-forward over token chunks (NVlabs/Sana
// `models/ltx25/RTX5090/memory.py` @ cf67a3a: 16,384-token chunks at >= 65,536 tokens) lowers the
// peak of ONE LTX-2.5 DiT forward at a stage-2 geometry, what it costs in step time, and whether
// the output is the unchunked one bit for bit. It also times the scalar-path timestep-embedding +
// AdaLN MLPs against the same forward (the "exact stage-2 AdaLN table" question).
//
// ONE PROCESS, ARMS INTERLEAVED. Every arm runs on the same loaded weights and the same inputs,
// round-robin with the order reversed on odd reps, so drift from any cause lands on every arm
// (AB-L-0339). `Memory.clearCache()` runs before each forward so one arm's pooled buffers are not
// the next arm's footprint. Per forward it reports:
//   - phys_footprint high-water (25 ms sampler) and its rise over the pre-forward footprint;
//   - mlx's own peak ACTIVE memory, reset per forward (exact, allocator-level; no pool);
//   - wall time;
//   - max|delta| against the first `off` forward, video and audio.
//
// SCOPE. A DiT forward, not a run: no encode, no decode. Whether a lower forward peak lowers a
// tier's WHOLE-RUN peak depends on which phase binds there — `--t2v-spot25` owns that.
//
// INPUTS are `--dit25-probe`'s: the real Gemma-4 golden through the real 2.5 connector, positions
// and first-latent-frame keyframes mask as `t2vTwoStage` builds them, unit-variance noise from a
// fixed key at the stage-2 entry sigma.
//
// usage: RunLTX2 --ffn-chunk-probe <W> <H> <F>
//   env: LTX_MODELS_ROOT=<dir>          (default /Volumes/Satechi/Models)
//        LTX_QUANT=bf16|int8            (default bf16; int8 = the -ditq8 tree)
//        LTX_STREAM_25=1|0              (default 0 = resident; 1 = granules, gate pinned)
//        LTX_FFN_ARMS=off,lazy,eval     (default all three; `off` must be present)
//        LTX_FFN_REPS=<n>               (default 2)
//        LTX_FFN_MIN_TOKENS / LTX_FFN_CHUNK_TOKENS   (probe-only; default 65536 / 16384)
//        LTX_FFN_WARM=0                 (skip the discarded warm forward at the measured geometry)
//        LTX_FFN_ADALN_ONLY=1           (time the AdaLN prelude and one forward, then stop)

import Foundation
import LTX2
import MLX
import MLXNN
import MLXRandom

private struct FFNArm {
    let name: String
    let chunking: FFChunking?
}

private func gb(_ bytes: Int) -> Double { Double(bytes) / 1_000_000_000.0 }
private func gb(_ bytes: UInt64) -> Double { Double(bytes) / 1_000_000_000.0 }

/// The scalar-path conditioning prelude, rebuilt from the checkpoint's own tensors with the ops
/// `DiT` uses (`timestepEmbedding` -> linear1 -> silu -> linear2 -> silu -> linear, per AdaLN
/// kind). All eight kinds the forward computes are included; upstream's table replaces two of
/// them, so this over-states what a table could remove.
private func adalnPrelude(_ dit: DiT, sigma: MLXArray) -> [MLXArray]? {
    let cfg = DiTConfig()
    func lin(_ x: MLXArray, _ prefix: String) -> MLXArray? {
        guard let w = dit.weight("\(prefix).weight"), dit.weight("\(prefix).scales") == nil else { return nil }
        var y = x.matmul(w.transposed())
        if let b = dit.weight("\(prefix).bias") { y = y + b }
        return y
    }
    func embedding(_ t: MLXArray) -> MLXArray {
        let half = cfg.timestepEmbeddingDim / 2
        let exponent = (-Foundation.log(Float(10000.0)) * MLXArray(0 ..< half).asType(.float32)) / Float(half)
        let args = t.asType(.float32).expandedDimensions(axis: -1) * MLX.exp(exponent)
        return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
    }
    let t = sigma.asType(.float32) * cfg.timestepScaleMultiplier
    let tEmb = embedding(t)
    let tEmbGate = embedding(t * (cfg.avCaTimestepScaleMultiplier / cfg.timestepScaleMultiplier))
    let kinds: [(String, MLXArray)] = [
        ("adaln_single", tEmb), ("av_ca_video_scale_shift_adaln_single", tEmb),
        ("av_ca_a2v_gate_adaln_single", tEmbGate), ("prompt_adaln_single", tEmb),
        ("audio_adaln_single", tEmb), ("av_ca_audio_scale_shift_adaln_single", tEmb),
        ("av_ca_v2a_gate_adaln_single", tEmbGate), ("audio_prompt_adaln_single", tEmb),
    ]
    var out: [MLXArray] = []
    for (prefix, emb) in kinds {
        let dt = dit.weight("\(prefix).emb.timestep_embedder.linear1.weight")?.dtype ?? .float32
        guard let h1 = lin(emb.asType(dt), "\(prefix).emb.timestep_embedder.linear1"),
              let embedded = lin(silu(h1), "\(prefix).emb.timestep_embedder.linear2"),
              let params = lin(silu(embedded), "\(prefix).linear")
        else { return nil }
        out.append(params)
        out.append(embedded)
    }
    return out
}

/// `--ffn-chunk-micro`: ONE feed-forward (block 0's real `ff` weights) on random hidden states,
/// whole vs chunked, per stage — which op (if any) changes its values with the chunk shape, and
/// which ROWS. No forward, so it isolates the feed-forward from 48 blocks of amplification.
///
/// usage: RunLTX2 --ffn-chunk-micro <N> <chunk>[,<chunk>...]   (env LTX_MODELS_ROOT, LTX_FFN_DTYPE=bf16|fp32)
func ffnChunkMicro(tokens n: Int, chunks: [Int]) throws {
    let env = ProcessInfo.processInfo.environment
    let modelsRoot = env["LTX_MODELS_ROOT"].flatMap { $0.isEmpty ? nil : $0 } ?? "/Volumes/Satechi/Models"
    let dtype: DType = env["LTX_FFN_DTYPE"] == "fp32" ? .float32 : .bfloat16
    let granule = URL(fileURLWithPath: "\(modelsRoot)/ltx-granules-25/bf16")
    // Block 0 only: read its tensors straight from the checkpoint header (lazy load, 4 arrays evaluated).
    let ckpt = URL(fileURLWithPath: "\(modelsRoot)/xocialize/ltx-2.5-mlx/transformer-distilled.safetensors")
    _ = granule
    let all = try MLX.loadArrays(url: ckpt)
    let p = "transformer.transformer_blocks.0.ff"
    guard let w1 = all["\(p).proj_in.weight"], let w2 = all["\(p).proj_out.weight"] else {
        print("[ffn-micro] FAIL ❌ block 0 ff weights not found"); exit(2)
    }
    let b1 = all["\(p).proj_in.bias"], b2 = all["\(p).proj_out.bias"]
    let W1 = w1.asType(dtype).transposed(), W2 = w2.asType(dtype).transposed()
    let B1 = b1?.asType(dtype), B2 = b2?.asType(dtype)
    MLXRandom.seed(0x2025_1114)
    let x = MLXRandom.normal([1, n, w1.dim(1)]).asType(dtype)
    eval(W1, W2, x)
    if let B1 { eval(B1) }
    if let B2 { eval(B2) }
    print("[ffn-micro] N=\(n) dtype=\(dtype) proj_in \(w1.shape) bias=\(b1 != nil) proj_out \(w2.shape) bias=\(b2 != nil)")

    func stage1(_ a: MLXArray) -> MLXArray { let y = a.matmul(W1); return B1.map { y + $0 } ?? y }
    func stage2(_ a: MLXArray) -> MLXArray { geluApproximate(a) }
    func stage3(_ a: MLXArray) -> MLXArray { let y = a.matmul(W2); return B2.map { y + $0 } ?? y }
    // `LTX_FFN_SPLIT=remainder` is upstream's torch.split (full chunks + a short tail) — the
    // poison arm: the tail lands on mlx's split-K kernel and must NOT read bit-identical when it
    // is under a third of the FF inner width. Default: `FFChunking.ranges`, what `DiT` runs.
    let remainderSplit = env["LTX_FFN_SPLIT"] == "remainder"
    func chunked(_ a: MLXArray, _ c: Int, _ f: (MLXArray) -> MLXArray) -> MLXArray {
        let total = a.dim(-2)
        var ranges: [Range<Int>] = []
        if remainderSplit {
            var s = 0
            while s < total { let e = min(s + c, total); ranges.append(s ..< e); s = e }
        } else {
            ranges = FFChunking(minTokens: 0, chunkTokens: c).ranges(tokens: total) ?? [0 ..< total]
        }
        return MLX.concatenated(ranges.map { f(a[.ellipsis, $0, 0...]) }, axis: -2)
    }
    print("[ffn-micro] split=\(remainderSplit ? "remainder (upstream torch.split)" : "equal (FFChunking.ranges)")")
    /// max|Δ| and how many token rows differ at all.
    func diff(_ a: MLXArray, _ b: MLXArray) -> (Float, Int) {
        let d = MLX.abs(a.asType(.float32) - b.asType(.float32))
        let rows = d.max(axis: -1).reshaped(-1)
        return (d.max().item(Float.self), MLX.greater(rows, MLXArray(Float(0))).asType(.int32).sum().item(Int.self))
    }

    if env["LTX_FFN_SWEEP"] == "1" {
        // Which row counts M give the SAME values for the same leading rows? Each M's first 256
        // rows are compared with the M=256 result and with the previous M's.
        let h = stage2(stage1(x)); eval(h)
        for (label, input, f) in [("proj_in ", x, stage1), ("proj_out", h, stage3)] as [(String, MLXArray, (MLXArray) -> MLXArray)] {
            var base: MLXArray? = nil, prev: MLXArray? = nil
            for m in chunks where m <= n {
                let y = f(input[.ellipsis, 0 ..< m, 0...])[.ellipsis, 0 ..< 256, 0...]; eval(y)
                let dBase = base.map { diff($0, y).0 } ?? 0
                let dPrev = prev.map { diff($0, y).0 } ?? 0
                print(String(format: "[ffn-micro] sweep %@ M=%7d  vs M=%d: max|Δ| %.3e   vs previous M: %.3e",
                             label as NSString, m, chunks[0], dBase, dPrev))
                if base == nil { base = y }
                prev = y
            }
        }
        return
    }
    let h1 = stage1(x); eval(h1)
    let h1b = stage1(x); eval(h1b)
    let (rep, repRows) = diff(h1, h1b)
    print(String(format: "[ffn-micro] control: proj_in whole vs itself  max|Δ| %.3e rows %d", rep, repRows))
    let g = stage2(h1); eval(g)
    let out = stage3(g); eval(out)
    for c in chunks {
        let c1 = chunked(x, c, stage1); eval(c1)
        let (d1, r1) = diff(h1, c1)
        let c2 = chunked(h1, c, stage2); eval(c2)
        let (d2, r2) = diff(g, c2)
        let c3 = chunked(g, c, stage3); eval(c3)
        let (d3, r3) = diff(out, c3)
        let full = chunked(x, c) { stage3(stage2(stage1($0))) }; eval(full)
        let (d4, r4) = diff(out, full)
        print(String(format: "[ffn-micro] chunk %6d  proj_in max|Δ| %.3e rows %d/%d · gelu %.3e rows %d · "
                     + "proj_out %.3e rows %d · whole FF %.3e rows %d",
                     c, d1, r1, n, d2, r2, d3, r3, d4, r4))
    }
}

func ffnChunkProbe(width: Int, height: Int, frames: Int) throws {
    let env = ProcessInfo.processInfo.environment
    let modelsRoot = env["LTX_MODELS_ROOT"].flatMap { $0.isEmpty ? nil : $0 } ?? "/Volumes/Satechi/Models"
    let quant = env["LTX_QUANT"] ?? "bf16"
    let streamed = env["LTX_STREAM_25"] == "1"
    let reps = env["LTX_FFN_REPS"].flatMap { Int($0) } ?? 2
    let minTokens = env["LTX_FFN_MIN_TOKENS"].flatMap { Int($0) } ?? FFChunking().minTokens
    let chunkTokens = env["LTX_FFN_CHUNK_TOKENS"].flatMap { Int($0) } ?? FFChunking().chunkTokens
    let warm = env["LTX_FFN_WARM"] != "0"
    let adalnOnly = env["LTX_FFN_ADALN_ONLY"] == "1"
    guard quant == "bf16" || quant == "int8" else {
        print("[ffn-probe] FAIL ❌ LTX_QUANT must be bf16 or int8"); exit(2)
    }

    let armNames = (env["LTX_FFN_ARMS"] ?? "off,lazy,eval").split(separator: ",").map(String.init)
    var arms: [FFNArm] = []
    for n in armNames {
        switch n {
        case "off": arms.append(FFNArm(name: n, chunking: nil))
        case "lazy": arms.append(FFNArm(name: n, chunking: FFChunking(
            minTokens: minTokens, chunkTokens: chunkTokens, evalPerChunk: false)))
        case "eval": arms.append(FFNArm(name: n, chunking: FFChunking(
            minTokens: minTokens, chunkTokens: chunkTokens, evalPerChunk: true)))
        default: print("[ffn-probe] FAIL ❌ unknown arm '\(n)' (off | lazy | eval)"); exit(2)
        }
    }
    guard arms.first?.name == "off" else {
        print("[ffn-probe] FAIL ❌ the first arm must be `off` (it is the reference)"); exit(2)
    }

    let base = "\(modelsRoot)/xocialize"
    let componentsTree = URL(fileURLWithPath: "\(base)/ltx-2.5-mlx")
    let ditTree = URL(fileURLWithPath: quant == "int8" ? "\(base)/ltx-2.5-mlx-ditq8" : "\(base)/ltx-2.5-mlx")
    let ditPath = ditTree.appendingPathComponent("transformer-distilled.safetensors")
    let granules = URL(fileURLWithPath: "\(modelsRoot)/ltx-granules-25/\(quant == "int8" ? "q8" : "bf16")")
    guard FileManager.default.fileExists(atPath: ditPath.path) else {
        print("[ffn-probe] FAIL ❌ missing \(ditPath.path)"); exit(2)
    }
    guard LTX2Pipeline.isLTX25(ltxDir: componentsTree) else {
        print("[ffn-probe] FAIL ❌ \(componentsTree.path) is not an LTX-2.5 tree"); exit(2)
    }

    let fps = 24.0
    let fLat = (frames - 1) / 8 + 1, hLat = height / 32, wLat = width / 32
    let videoTokens = fLat * hLat * wLat
    let audioTokens = Positions.audioTokenCount(numFrames: frames, fps: fps)
    let sigmaValue = Positions.stage2Sigmas[0]
    let probeRanges = FFChunking(minTokens: minTokens, chunkTokens: chunkTokens).ranges(tokens: videoTokens)
    let engages = probeRanges != nil
    print("[ffn-probe] geom=\(width)x\(height)x\(frames)f → latent \(fLat)x\(hLat)x\(wLat)"
        + "  videoTokens=\(videoTokens) audioTokens=\(audioTokens) sigma=\(sigmaValue)")
    print("[ffn-probe] quant=\(quant) lane=\(streamed ? "streamed (gate pinned)" : "resident")"
        + " arms=\(arms.map(\.name).joined(separator: ",")) reps=\(reps)"
        + " minTokens=\(minTokens) chunkTokens=\(chunkTokens)"
        + " → chunking \(engages ? "ENGAGES (\(probeRanges!.count) chunks of \(probeRanges!.map(\.count).min()!)–\(probeRanges!.map(\.count).max()!) tokens)" : "does NOT engage at this N")")

    // --- inputs (the connector is transient: built, evaluated, dropped) ---
    let videoText: MLXArray, audioText: MLXArray
    do {
        let goldens = try MLX.loadArrays(url: URL(fileURLWithPath: "\(goldensBase)/gemma4/goldens.safetensors"))
        guard let mask = goldens["attention_mask"] else {
            print("[ffn-probe] FAIL ❌ gemma4 goldens missing attention_mask"); exit(2)
        }
        var hidden: [MLXArray] = []
        for i in 0 ..< 49 {
            guard let h = goldens[String(format: "gemma_hidden_%02d", i)] else {
                print("[ffn-probe] FAIL ❌ gemma4 goldens missing state \(i)"); exit(2)
            }
            hidden.append(h)
        }
        let connector = try Connector.load(connectorPath: componentsTree.appendingPathComponent("connector.safetensors"))
        (videoText, audioText) = connector(hiddenStates: hidden, mask: mask)
        eval(videoText, audioText)
    }
    Memory.clearCache()

    MLXRandom.seed(0x2025_0814)
    let videoLatent = MLXRandom.normal([1, videoTokens, 128]).asType(.float32)
    let audioLatent = MLXRandom.normal([1, audioTokens, 128]).asType(.float32)
    let videoPositions = Positions.video(F: fLat, H: hLat, W: wLat, fps: Float(fps))
    let audioPositions = Positions.audio(tokens: audioTokens)
    let kfMask = LTX2Pipeline.firstLatentFrameKeyframesMask(
        totalTokens: videoTokens, tokensPerLatentFrame: hLat * wLat)
    let sigma = MLXArray([sigmaValue])
    eval(videoLatent, audioLatent, videoPositions, audioPositions, kfMask)

    // --- DiT ---
    let t0 = Date()
    var dit: DiT
    if streamed {
        guard FileManager.default.fileExists(atPath: granules.appendingPathComponent("manifest.json").path) else {
            print("[ffn-probe] FAIL ❌ no granule tree at \(granules.path)"); exit(2)
        }
        let streamer = try LTXBlockStreamer(
            granuleDir: granules, options: BlockStreamingOptions(gatePolicy: .forceStream))
        dit = try DiT(streaming: streamer, checkpoint: ditPath, config: DiTConfig(), computeDtype: .bfloat16)
    } else {
        prewarmFiles([ditPath])
        dit = try DiT.load(weightsPath: ditPath, config: DiTConfig(), computeDtype: .bfloat16)
    }
    dit.warmup()
    print(String(format: "[ffn-probe] DiT loaded + warmed in %.1fs · phys %.2f GB", Date().timeIntervalSince(t0),
                 gb(physFootprintBytes())))

    func forward(_ d: DiT) -> (MLXArray, MLXArray) {
        let (v, a) = d(
            videoLatent: videoLatent, audioLatent: audioLatent, sigma: sigma,
            videoText: videoText, audioText: audioText,
            videoPositions: videoPositions, audioPositions: audioPositions,
            keyframesMask: kfMask)
        let vf = v.asType(.float32), af = a!.asType(.float32)
        eval(vf, af)
        return (vf, af)
    }

    let sampler = PhysSampler()
    sampler.start()
    defer { sampler.stop() }

    // --- warm forward at the measured geometry (discarded): shape-specialised kernels + page-in ---
    var warmSeconds: Double? = nil
    if warm || adalnOnly {
        var d = dit; d.ffChunking = nil
        let w0 = Date()
        _ = forward(d)
        warmSeconds = Date().timeIntervalSince(w0)
        print(String(format: "[ffn-probe] warm forward (off, discarded) %.2fs", warmSeconds!))
    }

    // --- AdaLN prelude timing ---
    if let first = adalnPrelude(dit, sigma: sigma) {
        eval(first)
        var times: [Double] = []
        for i in 0 ..< 40 {
            // A different sigma each rep so nothing can be served from a cached result.
            let s = MLXArray([sigmaValue - Float(i) * 1e-4])
            eval(s)
            let a0 = Date()
            eval(adalnPrelude(dit, sigma: s)!)
            times.append(Date().timeIntervalSince(a0))
        }
        let kept = times.dropFirst(8).sorted()
        let median = kept[kept.count / 2]
        print(String(format: "[ffn-probe] ADALN prelude (2 embeddings + 8 AdaLN MLPs, scalar path): "
                     + "median %.3f ms  min %.3f ms  max %.3f ms  (n=%d)",
                     median * 1e3, kept.first! * 1e3, kept.last! * 1e3, kept.count))
        if let ws = warmSeconds {
            print(String(format: "[ffn-probe] ADALN share of the warm forward: %.4f%%  (%.3f ms / %.2f s)",
                         100 * median / ws, median * 1e3, ws))
        }
    } else {
        print("[ffn-probe] ADALN prelude: SKIPPED (AdaLN weights not plain in this checkpoint)")
    }
    if adalnOnly { return }

    // --- interleaved arms ---
    struct Row { let arm: String; let rep: Int; let seconds: Double; let physPeak, physBase: UInt64
                 let mlxPeak: Int; let dV, dA: Float }
    var rows: [Row] = []
    var refV: MLXArray? = nil, refA: MLXArray? = nil
    for rep in 0 ..< reps {
        let order = rep % 2 == 0 ? arms : [arms[0]] + arms.dropFirst().reversed()
        // `off` leads rep 0 (it is the reference); on odd reps it trails so position cancels.
        let seq = rep % 2 == 0 ? order : Array(order.dropFirst()) + [order[0]]
        for arm in seq {
            var d = dit; d.ffChunking = arm.chunking
            Memory.clearCache()
            Memory.peakMemory = 0
            let physBase = physFootprintBytes()
            _ = sampler.takePhaseMax()
            let f0 = Date()
            let (v, a) = forward(d)
            let seconds = Date().timeIntervalSince(f0)
            let physPeak = sampler.takePhaseMax()
            let mlxPeak = Memory.peakMemory
            var dV: Float = 0, dA: Float = 0
            if let rv = refV, let ra = refA {
                dV = maxAbs(v, rv); dA = maxAbs(a, ra)
            } else {
                refV = v; refA = a
            }
            rows.append(Row(arm: arm.name, rep: rep, seconds: seconds, physPeak: physPeak, physBase: physBase,
                            mlxPeak: mlxPeak, dV: dV, dA: dA))
            print(String(format: "[ffn-probe] rep %d arm %-4@  %8.2fs  phys peak %6.2f GB (base %6.2f, +%5.2f)"
                         + "  mlx peak active %6.2f GB  max|Δ| video %.3e audio %.3e",
                         rep, arm.name as NSString, seconds, gb(physPeak), gb(physBase),
                         gb(physPeak) - gb(physBase), gb(mlxPeak), dV, dA))
        }
    }

    // --- summary: worst peak and median time per arm ---
    print("[ffn-probe] SUMMARY geom=\(width)x\(height)x\(frames) videoTokens=\(videoTokens) quant=\(quant)"
        + " lane=\(streamed ? "streamed" : "resident") minTokens=\(minTokens) chunkTokens=\(chunkTokens)"
        + " engages=\(engages)")
    for arm in arms {
        let r = rows.filter { $0.arm == arm.name }
        let secs = r.map(\.seconds).sorted()
        let bit = r.allSatisfy { $0.dV == 0 && $0.dA == 0 }
        print(String(format: "[ffn-probe] SUMMARY arm %-4@  phys peak worst %6.2f GB  rise worst %5.2f GB"
                     + "  mlx peak active worst %6.2f GB  time median %8.2fs (min %.2f max %.2f)  bit-identical to off: %@",
                     arm.name as NSString, gb(r.map(\.physPeak).max()!),
                     r.map { gb($0.physPeak) - gb($0.physBase) }.max()!,
                     gb(r.map(\.mlxPeak).max()!), secs[secs.count / 2], secs.first!, secs.last!,
                     (bit ? "yes" : "NO") as NSString))
    }
}

// MARK: - `--ffn-chunk-gate`

/// The regression gate for `DiT.ffChunking`. Four parts, cheapest first:
///
///  1. `FFChunking.ranges` — pure: contiguous cover of 0..<n, no chunk above `chunkTokens`, none
///     below `chunkTokens / 2`, nil under `minTokens`. Swept over n.
///  2. WIRING on the tiny DiT fixture (no real weights): chunked forwards, lazy and per-chunk
///     eval, against the unchunked one. Tiny widths sit on other GEMM kernels, so the bar here is
///     cosine, not bits — it catches a mis-ordered or mis-sliced concatenation, which moves whole
///     tokens.
///  3. KERNEL REGIME at the real FF widths with synthetic weights (no checkpoint): one
///     feed-forward at 70,000 tokens must be BIT-IDENTICAL whole vs equal chunks. The poison arm
///     is upstream's remainder split, whose 4,464-row tail falls under a third of the inner
///     width: where mlx has the shape-dependent dispatch it MUST differ, which is what proves the
///     check can fail. A device without that dispatch reports the poison as not applicable.
///  4. REAL CODE PATH at real widths: block 0 of each present 2.5 checkpoint (bf16, int8) as a
///     one-layer `DiT`, 70,000 video tokens, off vs lazy vs eval — bit-identical. Skipped with a
///     note when no checkpoint is on this machine.
///
/// usage: RunLTX2 --ffn-chunk-gate      (env LTX_MODELS_ROOT)
func ffnChunkGate() throws {
    var fails: [String] = []
    func check(_ ok: Bool, _ what: String) {
        print("[ffn-chunk-gate] \(ok ? "✅" : "❌") \(what)")
        if !ok { fails.append(what) }
    }

    // --- 1. ranges ---
    do {
        let c = FFChunking()
        var ok = c.ranges(tokens: c.minTokens - 1) == nil && c.ranges(tokens: 0) == nil
        var smallest = Int.max, largest = 0
        var n = c.minTokens
        while n <= 400_000 {
            guard let r = c.ranges(tokens: n) else { ok = false; break }
            var cursor = 0
            for x in r {
                if x.lowerBound != cursor || x.isEmpty { ok = false }
                cursor = x.upperBound
                smallest = min(smallest, x.count); largest = max(largest, x.count)
            }
            if cursor != n { ok = false }
            n += 997
        }
        ok = ok && largest <= c.chunkTokens && smallest >= c.chunkTokens / 2
        check(ok, "ranges: contiguous cover, chunk sizes \(smallest)…\(largest) within "
            + "[\(c.chunkTokens / 2), \(c.chunkTokens)], nil below \(c.minTokens)")
        let r = c.ranges(tokens: 124_440)!
        check(r.count == 8 && r.allSatisfy { $0.count == 15_555 },
              "ranges: 124,440 tokens (1920x1088x481 stage 2) → 8 chunks of 15,555")
    }

    // --- 2. wiring on the tiny fixture ---
    do {
        let dir = "\(goldensBase)/dit_tiny"
        if FileManager.default.fileExists(atPath: "\(dir)/weights.safetensors") {
            let weights = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/weights.safetensors"))
            let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
            let base = DiT(weights: weights, config: tinyDiTConfig())
            let n = io["video_latent"]!.dim(1)
            func run(_ c: FFChunking?) -> (MLXArray, MLXArray) {
                var d = base; d.ffChunking = c
                let (v, a) = ditAV(d,
                    videoLatent: io["video_latent"]!, audioLatent: io["audio_latent"]!, sigma: io["sigma"]!,
                    videoText: io["video_text"], audioText: io["audio_text"],
                    videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!)
                eval(v, a)
                return (v, a)
            }
            let (v0, a0) = run(nil)
            let chunk = max(1, (n + 2) / 3)
            for evalPerChunk in [false, true] {
                let c = FFChunking(minTokens: 2, chunkTokens: chunk, evalPerChunk: evalPerChunk)
                let (v, a) = run(c)
                let vc = cosine(v, v0), ac = cosine(a, a0)
                check(c.ranges(tokens: n) != nil && vc >= 0.999999 && ac >= 0.999999,
                      String(format: "wiring (tiny DiT, %d tokens in %d chunks, %@): video cos %.8f max|Δ| %.2e · audio cos %.8f",
                             n, c.ranges(tokens: n)?.count ?? 0, (evalPerChunk ? "eval" : "lazy") as NSString,
                             vc, maxAbs(v, v0), ac))
            }
        } else {
            print("[ffn-chunk-gate] ⚠️ wiring: SKIPPED — no dit_tiny goldens at \(dir)")
        }
    }

    // --- 3. kernel regime at the real widths, synthetic weights ---
    do {
        let n = 70_000, dim = 4096, inner = 16_384
        MLXRandom.seed(0x2025_1114)
        let W1 = (MLXRandom.normal([dim, inner]) * 0.02).asType(.bfloat16)
        let W2 = (MLXRandom.normal([inner, dim]) * 0.02).asType(.bfloat16)
        let x = MLXRandom.normal([1, n, dim]).asType(.bfloat16)
        eval(W1, W2, x)
        func ff(_ a: MLXArray) -> MLXArray { geluApproximate(a.matmul(W1)).matmul(W2) }
        func over(_ ranges: [Range<Int>]) -> MLXArray {
            MLX.concatenated(ranges.map { r -> MLXArray in let y = ff(x[.ellipsis, r, 0...]); eval(y); return y }, axis: -2)
        }
        let whole = ff(x); eval(whole)
        let equal = over(FFChunking().ranges(tokens: n)!); eval(equal)
        let dEqual = maxAbs(whole, equal)
        check(dEqual == 0, String(format: "kernel regime: one FF at %d tokens, equal chunks vs whole  max|Δ| %.3e (must be 0)", n, dEqual))
        var tail: [Range<Int>] = []
        var s = 0
        while s < n { let e = min(s + 16_384, n); tail.append(s ..< e); s = e }
        let poison = over(tail); eval(poison)
        let dPoison = maxAbs(whole, poison)
        if dPoison != 0 {
            print(String(format: "[ffn-chunk-gate] ✅ poison: upstream's remainder split (tail %d rows) differs, max|Δ| %.3e — the check can fail",
                         tail.last!.count, dPoison))
        } else {
            print("[ffn-chunk-gate] ⚠️ poison: the remainder split did NOT differ on this device/mlx — no "
                + "shape-dependent GEMM dispatch here, so part 3 cannot fail on it (equal chunks are still exact)")
        }
    }
    Memory.clearCache()

    // --- 4. the real code path: block 0 of each present checkpoint as a one-layer DiT ---
    let modelsRoot = ProcessInfo.processInfo.environment["LTX_MODELS_ROOT"].flatMap { $0.isEmpty ? nil : $0 }
        ?? "/Volumes/Satechi/Models"
    var ranReal = false
    for (label, tree) in [("bf16", "ltx-2.5-mlx"), ("int8", "ltx-2.5-mlx-ditq8")] {
        let path = "\(modelsRoot)/xocialize/\(tree)/transformer-distilled.safetensors"
        guard FileManager.default.fileExists(atPath: path) else {
            print("[ffn-chunk-gate] ⚠️ real code path (\(label)): SKIPPED — no checkpoint at \(path)")
            continue
        }
        ranReal = true
        let all = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let kept = all.filter { !$0.key.contains("transformer_blocks.") || $0.key.contains("transformer_blocks.0.") }
        var cfg = DiTConfig(); cfg.numLayers = 1
        let base = DiT(weights: kept, config: cfg, computeDtype: .bfloat16)
        let F = 35, H = 40, W = 50, n = F * H * W, audioN = 64
        MLXRandom.seed(0x2025_1115)
        let vLat = MLXRandom.normal([1, n, 128]), aLat = MLXRandom.normal([1, audioN, 128])
        let vText = MLXRandom.normal([1, 64, cfg.videoDim]), aText = MLXRandom.normal([1, 64, cfg.audioDim])
        let vPos = Positions.video(F: F, H: H, W: W, fps: 24), aPos = Positions.audio(tokens: audioN)
        let kf = LTX2Pipeline.firstLatentFrameKeyframesMask(totalTokens: n, tokensPerLatentFrame: H * W)
        let sigma = MLXArray([Positions.stage2Sigmas[0]])
        eval(vLat, aLat, vText, aText, vPos, aPos, kf)
        func run(_ c: FFChunking?) -> (MLXArray, MLXArray) {
            var d = base; d.ffChunking = c
            let (v, a) = ditAV(d, videoLatent: vLat, audioLatent: aLat, sigma: sigma, videoText: vText,
                               audioText: aText, videoPositions: vPos, audioPositions: aPos, keyframesMask: kf)
            let vf = v.asType(.float32), af = a.asType(.float32)
            eval(vf, af)
            return (vf, af)
        }
        let (v0, a0) = run(nil)
        for evalPerChunk in [false, true] {
            let (v, a) = run(FFChunking(evalPerChunk: evalPerChunk))
            let dv = maxAbs(v, v0), da = maxAbs(a, a0)
            check(dv == 0 && da == 0,
                  String(format: "real code path (%@ block 0, %d tokens, %@): max|Δ| video %.3e audio %.3e (must be 0)",
                         label as NSString, n, (evalPerChunk ? "eval" : "lazy") as NSString, dv, da))
        }
        Memory.clearCache()
    }
    if !ranReal { print("[ffn-chunk-gate] ⚠️ part 4 did not run on this machine — parts 1–3 only") }

    print(fails.isEmpty ? "[ffn-chunk-gate] PASS ✅" : "[ffn-chunk-gate] FAIL ❌ (\(fails.count))")
    if !fails.isEmpty { exit(1) }
}
