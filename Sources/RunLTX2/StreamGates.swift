// StreamGates.swift — HV2 BlockStreamer gates for the LTX-2.3 DiT
// (STREAMING-PLAN.md; the wan-core RunWanStream receipts, LTX edition).
//
// Gate doctrine carried from the wan receipts verbatim:
//   - streamed vs resident is a MEMCMP gate, not a cosine gate — the streamer
//     must be bit-exact at the same computeDtype;
//   - a parity compare that cannot see a bad refill is not evidence → every
//     parity gate is paired with a poisoned-slot negative control;
//   - `.auto` ladder rungs run ONE RUNG PER PROCESS (the repeated-fallback
//     phys ratchet), receipts run bare (no `| tee` — it masks signal kills);
//   - parity gates run computeDtype == checkpoint precision (bf16 prod): the
//     resident path's cast is then a no-op, so both sides feed identical
//     weight bytes to identical kernels. (fp32-compute-over-bf16-weights is
//     NOT a streamed configuration.)

import Foundation
import MLX
import MLXRandom
import LTX2

// MARK: - helpers

/// Bitwise equality: shapes, dtype, and raw backing bytes.
func bitEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
    guard a.shape == b.shape, a.dtype == b.dtype else { return false }
    eval(a, b)
    let da = a.asData(access: .noCopyIfContiguous).data
    let db = b.asData(access: .noCopyIfContiguous).data
    return da == db
}

private func physFootprintGB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return 0 }
    return Double(info.phys_footprint) / 1_073_741_824
}

private struct QuantPaths {
    let checkpoint: URL
    let granules: URL

    init?(_ quant: String) {
        let repos = [
            "bf16": "ltx-2.3-mlx",
            "q8": "ltx-2.3-mlx-q8",
            "q4": "ltx-2.3-mlx-q4",
        ]
        guard let repo = repos[quant] else { return nil }
        checkpoint = URL(
            fileURLWithPath:
                "/Volumes/Satechi/Models/dgrauet/\(repo)/transformer-distilled.safetensors")
        granules = URL(fileURLWithPath: "/Volumes/Satechi/Models/ltx-granules/\(quant)")
    }
}

private struct DiTInputs {
    var videoLatent, audioLatent, sigma: MLXArray
    var videoText, audioText: MLXArray?
    var videoPositions, audioPositions: MLXArray

    static func fromGoldens(_ dir: String) throws -> DiTInputs {
        let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
        return DiTInputs(
            videoLatent: io["video_latent"]!, audioLatent: io["audio_latent"]!,
            sigma: io["sigma"]!, videoText: io["video_text"], audioText: io["audio_text"],
            videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!)
    }

    /// Token-axis tile to a target video N (positions tiled alongside — RoPE
    /// values don't matter for gate/footprint arms, only shapes drive cost).
    func tiled(toVideoN n: Int) -> DiTInputs {
        var out = self
        func tile(_ a: MLXArray, axis: Int, to target: Int) -> MLXArray {
            let cur = a.dim(axis)
            let reps = (target + cur - 1) / cur
            let cat = MLX.concatenated(Array(repeating: a, count: reps), axis: axis)
            return cat.split(indices: [target], axis: axis)[0]
        }
        out.videoLatent = tile(videoLatent, axis: 1, to: n)
        out.videoPositions = tile(videoPositions, axis: 1, to: n)  // (B, N, 3)
        return out
    }

    func run(_ dit: DiT) -> (MLXArray, MLXArray) {
        let (v, a) = dit(
            videoLatent: videoLatent, audioLatent: audioLatent, sigma: sigma,
            videoText: videoText, audioText: audioText,
            videoPositions: videoPositions, audioPositions: audioPositions)
        eval(v, a)
        return (v, a)
    }
}

// MARK: - tiny synthetic gate (CI-runnable, no real weights)

/// Builds granules from the dit_tiny golden weights, then proves on one tiny
/// DiT: streamed ≡ resident (memcmp), clean/clean determinism, poisoned slot
/// diverges, threshold defers the verdict, and a rigged `.auto` gate falls
/// back resident bit-exactly with the streamer detached.
func streamTinyGate(scratch: String) throws {
    let dir = "\(goldensBase)/dit_tiny"
    let weightsURL = URL(fileURLWithPath: "\(dir)/weights.safetensors")
    let weights = try MLX.loadArrays(url: weightsURL)
    let inputs = try DiTInputs.fromGoldens(dir)

    // The tiny dump's dtype decides computeDtype (parity gates run at
    // checkpoint precision — see header).
    let sample = weights.first { $0.key.hasSuffix(".weight") }!.value
    let computeDtype = sample.dtype
    let blockPrefix =
        weights.keys.contains { $0.hasPrefix("transformer.transformer_blocks.") }
        ? "transformer.transformer_blocks." : "transformer_blocks."
    print("[stream-tiny-gate] dtype=\(computeDtype) blockPrefix=\(blockPrefix)")

    let granuleDir = URL(fileURLWithPath: "\(scratch)/ltx-stream-tiny-granules")
    try? FileManager.default.removeItem(at: granuleDir)
    let result = try LTXGranuleLayout.write(
        safetensors: weightsURL, outputDir: granuleDir, blockPrefix: blockPrefix)
    let blockCount = result.manifest.blockCount
    print("[stream-tiny-gate] laid out \(blockCount) blocks, "
        + "\(result.manifest.blocks[0].tensors.count) tensors/block")
    let groupSize = 1  // numGroups must be even (slot parity): 2 blocks → 2 groups

    let resident = DiT(weights: weights, config: tinyDiTConfig(), computeDtype: computeDtype)
    let (rv, ra) = inputs.run(resident)

    // 1. parity + determinism (forceStream: the gate never interferes)
    let s1 = try LTXBlockStreamer(
        granuleDir: granuleDir,
        options: .init(groupSize: groupSize, gatePolicy: .forceStream, quiet: false))
    let streamed = try DiT(
        streaming: s1, checkpoint: weightsURL, config: tinyDiTConfig(),
        computeDtype: computeDtype)

    // Weight-dict diff: every key the resident DiT holds must be bit-equal in
    // the streamed dict AFTER one acquire cycle... but before any forward the
    // slots hold whatever the FIRST refills delivered. Compare per-key by
    // forcing one forward first is circular — instead compare the arrays the
    // dict maps for block keys against the source checkpoint values right
    // after the first streamed forward completes (slots then hold the LAST
    // groups' data, so only the final groups' keys are comparable; the full
    // proof is the forward memcmp below). Globals compare unconditionally.
    var globalDiff = 0
    for k in resident.weightKeys where !k.hasPrefix("transformer_blocks.") {
        if let a = resident.weight(k), let b = streamed.weight(k), !bitEqual(a, b) {
            if globalDiff < 5 { print("[stream-tiny-gate]   global mismatch: \(k)") }
            globalDiff += 1
        }
    }
    print(globalDiff == 0
        ? "[stream-tiny-gate] globals bit-equal (58) ✅"
        : "[stream-tiny-gate] \(globalDiff) global keys differ ❌")

    let (sv1, sa1) = inputs.run(streamed)
    let parity = bitEqual(sv1, rv) && bitEqual(sa1, ra)
    print(parity
        ? "[stream-tiny-gate] parity streamed≡resident (memcmp) ✅"
        : String(
            format: "[stream-tiny-gate] parity FAILED ❌ (video cos=%.6f maxAbs=%.3e · "
                + "audio cos=%.6f maxAbs=%.3e)",
            cosine(sv1, rv), maxAbs(sv1, rv), cosine(sa1, ra), maxAbs(sa1, ra)))
    // Post-forward block-weight audit: after the final group's eval the slots
    // still hold the LAST sweep's groups — compare every block key currently
    // resident in a slot against the source values.
    var blockDiff = 0
    for k in resident.weightKeys where k.hasPrefix("transformer_blocks.") {
        if let a = resident.weight(k), let b = streamed.weight(k), !bitEqual(a, b) {
            if blockDiff < 6 { print("[stream-tiny-gate]   block-key mismatch: \(k)") }
            blockDiff += 1
        }
    }
    print("[stream-tiny-gate] post-forward block-key mismatches: \(blockDiff)/172")
    let (sv2, sa2) = inputs.run(streamed)
    let determinism = bitEqual(sv2, sv1) && bitEqual(sa2, sa1)
    print(determinism
        ? "[stream-tiny-gate] clean/clean determinism ✅"
        : "[stream-tiny-gate] determinism FAILED ❌")

    // 2. poisoned-slot negative control — the compare must have teeth
    s1.armPoison(afterAcquires: 0, localBlock: 0, tensor: 0, bytes: 4096)
    let (pv, pa) = inputs.run(streamed)
    let poisonSeen = !(bitEqual(pv, sv1) && bitEqual(pa, sa1))
    print(poisonSeen
        ? "[stream-tiny-gate] poisoned slot diverges (control has teeth) ✅"
        : "[stream-tiny-gate] poison NOT observed — compare is blind ❌")
    s1.finish()

    // 3. threshold defers the verdict; rigged `.auto` falls back bit-exactly
    let s2 = try LTXBlockStreamer(
        granuleDir: granuleDir,
        options: .init(groupSize: groupSize, gatePolicy: .auto, gateMargin: .infinity))
    let autoDit = try DiT(
        streaming: s2, checkpoint: weightsURL, config: tinyDiTConfig(),
        computeDtype: computeDtype)
    s2.gateEvaluationThresholdTokens = .max  // stage-1 semantics: no verdict yet
    _ = inputs.run(autoDit)
    let deferred = s2.verdict == .undecided
    print(deferred
        ? "[stream-tiny-gate] below-threshold forward left verdict undecided ✅"
        : "[stream-tiny-gate] threshold did NOT defer (verdict \(s2.verdict)) ❌")
    s2.gateEvaluationThresholdTokens = 0  // stage 2 arrives
    let (fv1, fa1) = inputs.run(autoDit)  // gate evaluates → rigged margin → falls back
    let fell = s2.verdict == .fellBack && autoDit.blockStreamer == nil
    print(fell
        ? "[stream-tiny-gate] rigged .auto fell back + detached ✅"
        : "[stream-tiny-gate] fallback did NOT happen (verdict \(s2.verdict)) ❌")
    // The falling-back forward itself streamed (verdict applies to LATER
    // forwards) — its output must equal resident, and so must post-fallback.
    let fellExact = bitEqual(fv1, rv) && bitEqual(fa1, ra)
    let (fv2, fa2) = inputs.run(autoDit)  // resident path now
    let postExact = bitEqual(fv2, rv) && bitEqual(fa2, ra)
    print(fellExact && postExact
        ? "[stream-tiny-gate] fallback output-invisible (memcmp, during+after) ✅"
        : "[stream-tiny-gate] fallback changed output ❌")

    // 4. Manifest v2 arms: sidecar-only bind (no checkpoint), integrity, tamper.
    let v2 = result.manifest.globals != nil && result.manifest.blocks[0].sha256 != nil
    print(v2
        ? "[stream-tiny-gate] v2 manifest (globals sidecar + sha256) ✅"
        : "[stream-tiny-gate] manifest is not v2 ❌")
    let s3 = try LTXBlockStreamer(
        granuleDir: granuleDir,
        options: .init(groupSize: groupSize, gatePolicy: .forceStream, quiet: true))
    let ghostCkpt = URL(fileURLWithPath: "\(scratch)/nonexistent-checkpoint.safetensors")
    let sidecarDit = try DiT(
        streaming: s3, checkpoint: ghostCkpt, config: tinyDiTConfig(),
        computeDtype: computeDtype)
    let (gv, ga) = inputs.run(sidecarDit)
    let sidecarExact = bitEqual(gv, rv) && bitEqual(ga, ra)
    print(sidecarExact
        ? "[stream-tiny-gate] checkpoint-free bind (sidecar globals) ≡ resident ✅"
        : "[stream-tiny-gate] sidecar bind diverged ❌")
    var integrityOK = false
    do {
        try s3.core.verifyIntegrity()
        integrityOK = true
        print("[stream-tiny-gate] verifyIntegrity clean tree ✅")
    } catch {
        print("[stream-tiny-gate] verifyIntegrity FAILED on clean tree ❌ \(error)")
    }
    s3.finish()
    // Tamper LAST (corrupts the scratch tree): flip one byte mid-file.
    var tamperCaught = false
    let victim = granuleDir.appendingPathComponent(result.manifest.blocks[0].file)
    if let fh = FileHandle(forWritingAtPath: victim.path) {
        try fh.seek(toOffset: 64)
        try fh.write(contentsOf: Data([0xAB]))
        try fh.close()
        do {
            try s3.core.verifyIntegrity()
            print("[stream-tiny-gate] tampered tree NOT caught ❌")
        } catch {
            tamperCaught = true
            print("[stream-tiny-gate] tampered granule caught by verifyIntegrity ✅")
        }
    }

    let pass = parity && determinism && poisonSeen && deferred && fell && fellExact && postExact
        && v2 && sidecarExact && integrityOK && tamperCaught
    print(pass ? "[stream-tiny-gate] PASS ✅" : "[stream-tiny-gate] FAIL ❌")
    if !pass { exit(1) }
}

// MARK: - real-checkpoint parity (one quant per invocation)

/// Full-scale streamed ≡ resident memcmp on the real checkpoint (dit_full
/// golden inputs), plus determinism, poison control, and — bf16 — the absolute
/// golden cosine so self-consistency can't hide a shared wrong turn.
func streamParityGate(quant: String) throws {
    guard let paths = QuantPaths(quant) else {
        print("[stream-parity-gate] unknown quant '\(quant)'")
        exit(2)
    }
    let inputs = try DiTInputs.fromGoldens("\(goldensBase)/dit_full")
    let tokens = inputs.videoLatent.dim(1) + inputs.audioLatent.dim(1)
    print("[stream-parity-gate] quant=\(quant) tokens=\(tokens) phys=\(String(format: "%.2f", physFootprintGB())) GB")

    // Resident reference first (run TWICE — the intrinsic-stability control:
    // the quantized DiT paths are documented run-order-sensitive, dit-q8's
    // flaky gate / q4's 1.5e-4 band, so the acceptance bar must be calibrated
    // against what the RESIDENT path itself reproduces), then drop it before
    // the streamed arm.
    var rv: MLXArray, ra: MLXArray
    var residentSelfExact = true
    do {
        let resident = try DiT.load(
            weightsPath: paths.checkpoint, config: DiTConfig(), computeDtype: .bfloat16)
        (rv, ra) = inputs.run(resident)
        let (rv2, ra2) = inputs.run(resident)
        residentSelfExact = bitEqual(rv2, rv) && bitEqual(ra2, ra)
        print(String(
            format: "[stream-parity-gate] resident done, self-repeat %@ (cos %.6f) · phys=%.2f GB",
            residentSelfExact ? "bit-exact" : "NOT bit-exact", cosine(rv2, rv),
            physFootprintGB()))
    }
    Memory.clearCache()

    let streamer = try LTXBlockStreamer(
        granuleDir: paths.granules, options: .init(gatePolicy: .forceStream))
    let dit = try DiT(
        streaming: streamer, checkpoint: paths.checkpoint, config: DiTConfig(),
        computeDtype: .bfloat16)
    let (sv1, sa1) = inputs.run(dit)
    // Acceptance: memcmp for bf16 (documented exactly deterministic, receipted
    // repeatedly). Quantized checkpoints accept at the repo's OWN quant-gate
    // bar (cosine ≥ 0.999): the int8 path's nondeterminism is GRAPH-SHAPE
    // sensitive (the flaky --dit-q8-gate class — scheduling/reduction-order
    // dependent per CLAUDE.md), so a stable resident self-repeat does NOT
    // predict the streamed shape's draw, and memcmp can pass or fail by luck.
    // The poison control below keeps the compare honest at that bar.
    let exact = bitEqual(sv1, rv) && bitEqual(sa1, ra)
    let vCos = cosine(sv1, rv), aCos = cosine(sa1, ra)
    let parity = exact || (quant != "bf16" && vCos >= 0.999 && aCos >= 0.999)
    print(exact
        ? "[stream-parity-gate] streamed≡resident (memcmp) ✅"
        : String(
            format: "[stream-parity-gate] streamed vs resident: video cos=%.6f maxAbs=%.3e · "
                + "audio cos=%.6f maxAbs=%.3e → %@",
            vCos, maxAbs(sv1, rv), aCos, maxAbs(sa1, ra),
            parity ? "within the quant band ✅" : "FAILED ❌"))
    print(String(
        format: "[stream-parity-gate] S=%.2f GiB/s · step compute %.2fs stall %.2fs · "
            + "slots %.2f GiB · phys %.2f GB",
        streamer.measuredSGiBs, streamer.lastForwardComputeSeconds,
        streamer.lastForwardStallSeconds,
        Double(streamer.slotResidentBytes) / 1_073_741_824, physFootprintGB()))

    let (sv2, sa2) = inputs.run(dit)
    let determinism = bitEqual(sv2, sv1) && bitEqual(sa2, sa1)
    print(determinism
        ? "[stream-parity-gate] clean/clean determinism ✅"
        : "[stream-parity-gate] determinism FAILED ❌")

    streamer.armPoison(afterAcquires: 3, localBlock: 0, tensor: 2, bytes: 1 << 20)
    let (pv, pa) = inputs.run(dit)
    let poisonSeen = !(bitEqual(pv, sv1) && bitEqual(pa, sa1))
    print(poisonSeen
        ? "[stream-parity-gate] poisoned slot diverges ✅"
        : "[stream-parity-gate] poison NOT observed ❌")
    streamer.finish()

    var goldenOK = true
    if quant == "bf16" {
        let io = try MLX.loadArrays(
            url: URL(fileURLWithPath: "\(goldensBase)/dit_full/io.safetensors"))
        let vCos = cosine(sv1, io["video_v"]!)
        let aCos = cosine(sa1, io["audio_v"]!)
        goldenOK = vCos >= 0.999 && aCos >= 0.999
        print(String(
            format: "[stream-parity-gate] vs oracle golden: video %.6f audio %.6f %@",
            vCos, aCos, goldenOK ? "✅" : "❌"))
    }

    let pass = parity && determinism && poisonSeen && goldenOK
    print(pass ? "[stream-parity-gate] PASS ✅" : "[stream-parity-gate] FAIL ❌")
    if !pass { exit(1) }
}

// MARK: - .auto gate ladder rung (ONE rung per process — the phys ratchet)

/// Run one `.auto` rung at a chosen video-token count. Small N on this machine
/// must FALL BACK (output-invisibly vs a resident rerun); target-tier N must
/// STREAM with the report printed. The caller picks the rung; the gate decides
/// from its own in-regime measurements.
func streamAutoGate(quant: String, videoN: Int?) throws {
    guard let paths = QuantPaths(quant) else {
        print("[stream-auto-gate] unknown quant '\(quant)'")
        exit(2)
    }
    var inputs = try DiTInputs.fromGoldens("\(goldensBase)/dit_full")
    if let n = videoN { inputs = inputs.tiled(toVideoN: n) }
    let tokens = inputs.videoLatent.dim(0)
        * (inputs.videoLatent.dim(1) + inputs.audioLatent.dim(1))
    print("[stream-auto-gate] quant=\(quant) N=\(tokens) (one rung per process)")

    let streamer = try LTXBlockStreamer(
        granuleDir: paths.granules, options: .init(gatePolicy: .auto))
    let dit = try DiT(
        streaming: streamer, checkpoint: paths.checkpoint, config: DiTConfig(),
        computeDtype: .bfloat16)
    let (v1, a1) = inputs.run(dit)  // gate evaluates in-regime on this forward
    guard let report = streamer.gateReport else {
        print("[stream-auto-gate] no gate report ❌")
        exit(1)
    }
    print(String(
        format: "[stream-auto-gate] N=%d S=%.2f C=%.2f GiB/s N_min≈%d → %@ · "
            + "stall %.2fs/%.2fs",
        report.n, report.sGiBs, report.cGiBs, report.nMin,
        report.streaming ? "STREAM" : "FELL BACK",
        report.step1StallSeconds, report.step1ComputeSeconds))

    var pass = true
    if streamer.verdict == .fellBack {
        // Output-invisibility: the streamed-then-fallen run must equal a pure
        // resident forward.
        let (v2, a2) = inputs.run(dit)  // resident now
        let resident = try DiT.load(
            weightsPath: paths.checkpoint, config: DiTConfig(), computeDtype: .bfloat16)
        let (rv, ra) = inputs.run(resident)
        let exact = bitEqual(v1, rv) && bitEqual(a1, ra)
            && bitEqual(v2, rv) && bitEqual(a2, ra)
        print(exact
            ? "[stream-auto-gate] fallback output-invisible (memcmp) ✅"
            : "[stream-auto-gate] fallback changed output ❌")
        pass = exact
    } else {
        // Streaming: run 3 more steps for a stall figure.
        var stall = 0.0, compute = 0.0
        for _ in 0..<3 {
            _ = inputs.run(dit)
            stall += streamer.lastForwardStallSeconds
            compute += streamer.lastForwardComputeSeconds
        }
        print(String(
            format: "[stream-auto-gate] steady: stall %.3fs / compute %.2fs (%.1f%%) "
                + "over 3 steps · phys %.2f GB",
            stall, compute, compute > 0 ? 100 * stall / compute : 0, physFootprintGB()))
        streamer.finish()
    }
    print(pass ? "[stream-auto-gate] PASS ✅" : "[stream-auto-gate] FAIL ❌")
    if !pass { exit(1) }
}

// MARK: - budget arm (emulated small-machine admission)

/// Stream the DiT denoise phase under an emulated memory budget: cap
/// `Memory.memoryLimit` + `Memory.cacheLimit`, run N steps at a target token
/// count, report MLX peak + phys. DiT-phase scope ONLY (Gemma encode + VAE out
/// of scope — §2.4 eviction owns those) — NOT a `residentBytes` declaration.
func streamBudgetGate(quant: String, videoN: Int, budgetGB: Double, steps: Int) throws {
    guard let paths = QuantPaths(quant) else {
        print("[stream-budget-gate] unknown quant '\(quant)'")
        exit(2)
    }
    var inputs = try DiTInputs.fromGoldens("\(goldensBase)/dit_full")
    inputs = inputs.tiled(toVideoN: videoN)
    let budget = Int(budgetGB * 1_073_741_824)
    print(String(
        format: "[stream-budget-gate] quant=%@ videoN=%d budget=%.2f GB cache=2 GiB steps=%d",
        quant, videoN, budgetGB, steps))

    Memory.memoryLimit = budget
    Memory.cacheLimit = 2 << 30
    GPU.resetPeakMemory()  // NOT deprecated (NEUROSTREAM QW1 sweep) — Memory has no reset

    let streamer = try LTXBlockStreamer(
        granuleDir: paths.granules, options: .init(gatePolicy: .forceStream))
    let dit = try DiT(
        streaming: streamer, checkpoint: paths.checkpoint, config: DiTConfig(),
        computeDtype: .bfloat16)
    var wall = 0.0
    for i in 0..<steps {
        let t0 = CFAbsoluteTimeGetCurrent()
        _ = inputs.run(dit)
        let dt = CFAbsoluteTimeGetCurrent() - t0
        wall += dt
        print(String(
            format: "[stream-budget-gate] step %d/%d %.2fs (stall %.2fs) phys %.2f GB",
            i + 1, steps, dt, streamer.lastForwardStallSeconds, physFootprintGB()))
    }
    streamer.finish()
    let peakGB = Double(Memory.peakMemory) / 1_073_741_824
    print(String(
        format: "[stream-budget-gate] MLX peak %.2f GB · phys %.2f GB · slots %.2f GiB · "
            + "%.2fs/step avg",
        peakGB, physFootprintGB(),
        Double(streamer.slotResidentBytes) / 1_073_741_824, wall / Double(steps)))
    let pass = peakGB < budgetGB
    print(pass
        ? "[stream-budget-gate] PASS ✅ (peak under budget)"
        : "[stream-budget-gate] FAIL ❌ (peak above budget)")
    if !pass { exit(1) }
}
