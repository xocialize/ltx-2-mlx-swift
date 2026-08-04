// BenchE2E.swift — the end-to-end metrics harness (`RunLTX2 --bench-e2e`).
//
// Encodes the measurement doctrine this repo keeps re-learning by hand (SPEED-PLAN S8 traps,
// BLOCKSTREAM-EXPANSION-EVAL §1.3, PROFILING.md) as an executable protocol:
//
//   · prewarm weight files before every load (cold faults measure paging, not the arm)
//   · one excluded warmup generation per block (shape-specific kernel compile lands there)
//   · R measured runs per block, arms alternating in ABBA block order (thermal/session drift
//     cancels to first order; a sign flip between block sets = NOISE, not a finding)
//   · cooldown between blocks; thermal state recorded before/after every run
//   · phys_footprint high-water sampled at 25 ms (the peak is a transient inside denoise)
//   · MLX_PROFILE is force-disabled — the profiler's per-span evals break fusion and inflate
//     decode ~1.6×; profile for the *split*, use THIS harness for the *ratio*
//   · receipts: probes/bench_e2e_<label>_<stamp>.{md,json} with an env fingerprint, per-run
//     rows, per-arm stats, intra-arm reproducibility, cross-arm output cosine, and an explicit
//     MEASURABLE / NOISE verdict per arm pair
//
// Quality verdicts follow the determinism doctrine (../CLAUDE.md): same-weights arms are
// expected near-identical (streaming, cache caps, chunking are output-invisible levers);
// q4 / pruna arms legitimately DIVERGE the sample — their cosine is recorded, never gated.
//
// Usage:
//   RunLTX2 --bench-e2e --arm base:quant=bf16 --arm lean:quant=bf16,decoder=pruna \
//       [--blocks 2] [--runs 2] [--cooldown 45] [--size 704x512] [--frames 24] [--fps 24] \
//       [--two-stage] [--prompt "..."] [--seed 42] [--label pruna-ab] [--out probes]
//
// Arm spec: name:key=val,key=val…   keys:
//   quant=bf16|q8|q4        transformer checkpoint (default bf16)
//   decoder=stock|pruna     video VAE decoder variant (default stock)
//   cache=<GB>              Memory.cacheLimit for this arm (default: harness-start value)
//   env.KEY=VAL             arbitrary env for load-time levers (LTX_VAE_CHUNK, LTX_VAE_HALO,
//                           LTX_STREAM_GRANULES, …) — set for the arm's block, removed after

import Foundation
import MLX
import LTX2

// MARK: - Spec

/// Weight roots. Defaults match the fleet's archive layout; `LTX_BENCH_BASE` / `LTX_BENCH_GEMMA`
/// override them (same convention as the I9 worktree) — load-bearing since I9: the archive volume
/// is USB, and bf16's working set faulting from it inside live command buffers IS the watchdog
/// abort. Point LTX_BENCH_BASE at a fast-storage staging tree for bf16 arms.
func benchBase() -> String {
    ProcessInfo.processInfo.environment["LTX_BENCH_BASE"] ?? "/Volumes/Satechi/Models/dgrauet"
}
func benchGemma() -> String {
    ProcessInfo.processInfo.environment["LTX_BENCH_GEMMA"] ?? defaultGemma
}

struct BenchArm {
    let name: String
    let quant: String          // bf16 | q8 | q4
    let decoder: String        // stock | pruna
    let cacheGB: Int?
    let env: [String: String]

    var transformerPath: URL? {
        switch quant {
        case "q8", "int8": return URL(fileURLWithPath: "\(benchBase())/ltx-2.3-mlx-q8/transformer-distilled.safetensors")
        case "q4", "int4": return URL(fileURLWithPath: "\(benchBase())/ltx-2.3-mlx-q4/transformer-distilled.safetensors")
        default: return nil
        }
    }
    var vaeDecoderPath: URL? {
        decoder == "pruna"
            ? URL(fileURLWithPath: "\(benchBase())/ltx-2.3-mlx/vae_decoder_pruna.safetensors")
            : nil
    }
    /// Only bf16 same-config arms carry an "expect ≈1" cross-arm cosine: bf16 is the measured
    /// exactly-deterministic path (5-run spread 0). q8 is documented NONDETERMINISTIC (int8
    /// graph-order sensitivity, intra-arm spread ~3.5e-3 — the flaky-gate saga), so its cross-arm
    /// cosine is only meaningful against its own intra-arm spread; q4 and pruna diverge by design.
    func expectsIdenticalOutput(to ref: BenchArm) -> Bool {
        quant == ref.quant && decoder == ref.decoder && quant == "bf16"
    }

    static func parse(_ spec: String) throws -> BenchArm {
        guard let colon = spec.firstIndex(of: ":") else {
            return BenchArm(name: spec, quant: "bf16", decoder: "stock", cacheGB: nil, env: [:])
        }
        let name = String(spec[..<colon])
        var quant = "bf16", decoder = "stock"
        var cacheGB: Int? = nil
        var env: [String: String] = [:]
        for kv in spec[spec.index(after: colon)...].split(separator: ",") {
            let parts = kv.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { throw BenchError.badSpec("\(spec) → '\(kv)'") }
            switch parts[0] {
            case "quant": quant = parts[1]
            case "decoder": decoder = parts[1]
            case "cache": cacheGB = Int(parts[1])
            default:
                if parts[0].hasPrefix("env.") { env[String(parts[0].dropFirst(4))] = parts[1] }
                else { throw BenchError.badSpec("\(spec) → unknown key '\(parts[0])'") }
            }
        }
        return BenchArm(name: name, quant: quant, decoder: decoder, cacheGB: cacheGB, env: env)
    }
}

enum BenchError: Error, CustomStringConvertible {
    case badSpec(String)
    var description: String { switch self { case .badSpec(let s): return "bad --arm spec: \(s)" } }
}

// MARK: - Records

struct RunRecord: Codable {
    let arm: String
    let block: Int
    let run: Int               // 0 = warmup (excluded from stats)
    let wallSeconds: Double
    let peakPhysGB: Double
    let mlxActiveGB: Double
    let mlxCacheGB: Double
    let thermalBefore: String
    let thermalAfter: String
    let outputMean: Float
    let outputStd: Float
    let cosineVsArmFirst: Double?   // intra-arm reproducibility (vs this arm's first measured run)
}

struct BlockRecord: Codable {
    let arm: String
    let block: Int
    let prewarmSeconds: Double
    let loadSeconds: Double
    let postDropFloorGB: Double     // phys after pipeline drop + clearCache (ratchet detector)
}

// MARK: - Helpers

private func thermalString() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

private func median(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    guard !s.isEmpty else { return 0 }
    return s.count % 2 == 1 ? s[s.count / 2] : 0.5 * (s[s.count / 2 - 1] + s[s.count / 2])
}

private func shellOut(_ launchPath: String, _ args: [String], cwd: String? = nil) -> String {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: launchPath)
    p.arguments = args
    if let cwd { p.currentDirectoryURL = URL(fileURLWithPath: cwd) }
    let pipe = Pipe(); p.standardOutput = pipe; p.standardError = Pipe()
    guard (try? p.run()) != nil else { return "" }
    p.waitUntilExit()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
}

/// Flat cosine + maxAbs between two same-shape arrays, computed in fp32 on the CPU stream.
private func compareOutputs(_ a: MLXArray, _ b: MLXArray) -> (cosine: Double, maxAbs: Double) {
    let x = a.asType(.float32).reshaped(-1)
    let y = b.asType(.float32).reshaped(-1)
    let dot = sum(x * y)
    let nx = sqrt(sum(x * x)), ny = sqrt(sum(y * y))
    let cos = dot / (nx * ny + 1e-12)
    let mad = MLX.max(abs(x - y))
    eval(cos, mad)
    return (Double(cos.item(Float.self)), Double(mad.item(Float.self)))
}

// MARK: - Harness

func benchE2E() async throws {
    let args = CommandLine.arguments

    func flag(_ name: String) -> Bool { args.contains(name) }
    func value(_ name: String) -> String? {
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        return args[i + 1]
    }
    func values(_ name: String) -> [String] {
        var out: [String] = []
        for (i, a) in args.enumerated() where a == name && i + 1 < args.count { out.append(args[i + 1]) }
        return out
    }

    // ---- Config
    let armSpecs = values("--arm")
    let arms = try (armSpecs.isEmpty ? ["A:quant=bf16", "B:quant=bf16"] : armSpecs).map(BenchArm.parse)
    // Default arms A/B identical = a NULL RUN: measures the noise floor of the whole protocol.
    let blocks = value("--blocks").flatMap(Int.init) ?? 2
    let runs = value("--runs").flatMap(Int.init) ?? 2
    let cooldown = value("--cooldown").flatMap(Double.init) ?? 45
    let frames = value("--frames").flatMap(Int.init) ?? 24
    let sizeStr = value("--size") ?? "704x512"
    let dims = sizeStr.lowercased().split(separator: "x").compactMap { Int($0) }
    let (width, height) = dims.count == 2 ? (dims[0], dims[1]) : (704, 512)
    let fps = value("--fps").flatMap(Double.init) ?? 24
    let twoStage = flag("--two-stage")
    let prompt = value("--prompt") ?? "a cat playing piano"
    let seed = value("--seed").flatMap(UInt64.init) ?? 42
    let label = value("--label") ?? "run"
    let outDir = value("--out") ?? "probes"

    // ---- Doctrine guards
    if ProcessInfo.processInfo.environment["MLX_PROFILE"] != nil {
        print("[bench-e2e] ⚠️ MLX_PROFILE is set — DISABLING it for this bench (profiler spans force")
        print("[bench-e2e]    per-span evals that break fusion; profile for splits, bench for ratios).")
        unsetenv("MLX_PROFILE")
    }
    let baselineCacheLimit = Memory.cacheLimit   // restored for arms without cache=

    let repoDir = FileManager.default.currentDirectoryPath
    let gitSHA = shellOut("/usr/bin/git", ["rev-parse", "--short", "HEAD"], cwd: repoDir)
    let gitDirty = shellOut("/usr/bin/git", ["status", "--porcelain"], cwd: repoDir).isEmpty ? "" : "+dirty"
    let hwModel = shellOut("/usr/sbin/sysctl", ["-n", "hw.model"])
    let memGB = (Int(shellOut("/usr/sbin/sysctl", ["-n", "hw.memsize"])) ?? 0) / 1_000_000_000
    let osVer = ProcessInfo.processInfo.operatingSystemVersionString

    let geometry = "\(width)x\(height)x\(frames)f fps=\(Int(fps)) \(twoStage ? "two-stage" : "one-stage")"
    print("[bench-e2e] arms=\(arms.map(\.name).joined(separator: ",")) blocks=\(blocks) runs=\(runs)/block cooldown=\(Int(cooldown))s")
    print("[bench-e2e] geometry=\(geometry) seed=\(seed) prompt=\"\(prompt)\"")
    print("[bench-e2e] host=\(hwModel) \(memGB)GB | \(osVer) | \(gitSHA)\(gitDirty)")

    let ltxDir = URL(fileURLWithPath: "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx")
    let gemmaDir = URL(fileURLWithPath: defaultGemma)

    // Every env key any arm sets gets cleared between blocks so state can never leak across arms.
    let allEnvKeys = Set(arms.flatMap { $0.env.keys })

    var runRecords: [RunRecord] = []
    var blockRecords: [BlockRecord] = []
    // First measured output per arm (kept for intra-arm reproducibility + cross-arm comparison).
    var firstOutput: [String: MLXArray] = [:]
    let started = Date()

    // Incremental evidence file (the RunWanStream lesson: a mid-run death must keep its receipts).
    // Rewritten after every run; the final .md/.json supersede it on clean exit.
    let stampEarly = { let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"
                       f.timeZone = TimeZone(identifier: "UTC"); return f.string(from: started) }()
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let partialPath = "\(outDir)/bench_e2e_\(label)_\(stampEarly).partial.jsonl"
    let jsonlEnc = JSONEncoder()
    func persistPartial(_ rec: RunRecord) {
        guard let line = try? jsonlEnc.encode(rec), let s = String(data: line, encoding: .utf8)
        else { return }
        if let fh = FileHandle(forWritingAtPath: partialPath) {
            fh.seekToEndOfFile(); fh.write((s + "\n").data(using: .utf8)!); try? fh.close()
        } else {
            try? (s + "\n").write(toFile: partialPath, atomically: true, encoding: .utf8)
        }
    }

    // ---- Protocol: block sets alternate arm order (ABBA)
    for block in 0..<blocks {
        let order = block % 2 == 0 ? arms : arms.reversed()
        for arm in order {
            print("\n[bench-e2e] ═══ block \(block) · arm \(arm.name) (quant=\(arm.quant) decoder=\(arm.decoder)) ═══")

            // Arm env: apply this arm's, clear everything else the harness knows about.
            for k in allEnvKeys { unsetenv(k) }
            for (k, v) in arm.env { setenv(k, v, 1) }
            Memory.cacheLimit = arm.cacheGB.map { $0 * 1_000_000_000 } ?? baselineCacheLimit

            // Prewarm exactly the files this arm will fault.
            let p0 = Date()
            var warm = [arm.transformerPath ?? ltxDir.appendingPathComponent("transformer-distilled.safetensors"),
                        arm.vaeDecoderPath ?? ltxDir.appendingPathComponent("vae_decoder.safetensors"),
                        ltxDir.appendingPathComponent("connector.safetensors"),
                        ltxDir.appendingPathComponent("audio_vae.safetensors"),
                        ltxDir.appendingPathComponent("vocoder.safetensors")]
            if twoStage {
                warm.append(ltxDir.appendingPathComponent("vae_encoder.safetensors"))
                warm.append(ltxDir.appendingPathComponent("spatial_upscaler_x2_v1_1.safetensors"))
            }
            warm.append(contentsOf: ((try? FileManager.default.contentsOfDirectory(
                at: gemmaDir, includingPropertiesForKeys: nil)) ?? []).filter { $0.pathExtension == "safetensors" })
            prewarmFiles(warm)
            let prewarmS = Date().timeIntervalSince(p0)

            let l0 = Date()
            var loadS = 0.0
            let sampler = PhysSampler()
            // The pipeline lives inside this scope so the block-floor below measures the process
            // AFTER its release — floor taken with the pipeline still referenced reads as the
            // resident DiT and the ratchet detector becomes meaningless (v1 bug, fixed).
            do {
            let pipeline = try await LTX2Pipeline.load(
                ltxDir: ltxDir, gemmaDir: gemmaDir,
                transformerPath: arm.transformerPath, vaeDecoderPath: arm.vaeDecoderPath)
            loadS = Date().timeIntervalSince(l0)
            print(String(format: "[bench-e2e] prewarm %.1fs · load %.1fs", prewarmS, loadS))

            sampler.start()

            func generate() async throws -> LTX2Pipeline.Output {
                let out = twoStage
                    ? try await pipeline.t2vTwoStage(prompt: prompt, height: height, width: width,
                                                     numFrames: frames, fps: fps, seed: seed)
                    : try await pipeline.t2v(prompt: prompt, height: height, width: width,
                                             numFrames: frames, fps: fps, seed: seed)
                eval(out.video); if let a = out.audio { eval(a) }
                return out
            }

            // run 0 = warmup: shape-specific kernel compile lands here; recorded, excluded from stats.
            for run in 0...runs {
                let tBefore = thermalString()
                sampler.resetMax()
                let t0 = Date()
                let out = try await generate()
                let wall = Date().timeIntervalSince(t0)
                let peak = gbOf(sampler.maxBytes())
                let active = Double(Memory.activeMemory) / 1e9
                let cache = Double(Memory.cacheMemory) / 1e9
                let tAfter = thermalString()

                let v32 = out.video.asType(.float32)
                let mean = MLX.mean(v32), std = MLX.std(v32)
                eval(mean, std)

                var cosFirst: Double? = nil
                if run > 0 {
                    if let ref = firstOutput[arm.name] {
                        cosFirst = compareOutputs(ref, out.video).cosine
                    } else {
                        firstOutput[arm.name] = out.video
                    }
                }
                let rec = RunRecord(
                    arm: arm.name, block: block, run: run, wallSeconds: wall,
                    peakPhysGB: peak, mlxActiveGB: active, mlxCacheGB: cache,
                    thermalBefore: tBefore, thermalAfter: tAfter,
                    outputMean: mean.item(Float.self), outputStd: std.item(Float.self),
                    cosineVsArmFirst: cosFirst)
                runRecords.append(rec)
                persistPartial(rec)
                let tag = run == 0 ? "warmup (excluded)" : "run \(run)"
                print(String(format: "[bench-e2e] %@ · %@: %.1fs · peak %.2f GB · thermal %@→%@%@",
                             arm.name, tag, wall, peak, tBefore, tAfter,
                             cosFirst.map { String(format: " · cos(first)=%.6f", $0) } ?? ""))
                if run == 0 { Memory.clearCache() }   // memBench convention: clean slate after compile
            }
            sampler.stop()
            }   // ← pipeline released here (ARC), so the floor below is post-drop for real

            // Drop + measure the floor (the ratchet detector: this should return to ~framework
            // baseline every block; a climbing floor = leaked residency).
            Memory.clearCache()
            let floor = gbOf(physFootprintBytes())
            blockRecords.append(BlockRecord(arm: arm.name, block: block,
                                            prewarmSeconds: prewarmS, loadSeconds: loadS,
                                            postDropFloorGB: floor))
            print(String(format: "[bench-e2e] block floor after drop+clearCache: %.2f GB", floor))

            if cooldown > 0 {
                print("[bench-e2e] cooldown \(Int(cooldown))s…")
                try await Task.sleep(nanoseconds: UInt64(cooldown * 1e9))
            }
        }
    }
    for k in allEnvKeys { unsetenv(k) }
    Memory.cacheLimit = baselineCacheLimit

    // ---- Cross-arm output comparison (vs the FIRST arm's first measured output)
    var crossArm: [(arm: String, cosine: Double, maxAbs: Double, expectIdentical: Bool)] = []
    if let refArm = arms.first, let refOut = firstOutput[refArm.name] {
        for arm in arms.dropFirst() {
            guard let out = firstOutput[arm.name] else { continue }
            let (cos, mad) = compareOutputs(refOut, out)
            crossArm.append((arm.name, cos, mad, arm.expectsIdenticalOutput(to: refArm)))
        }
    }

    // ---- Stats + verdicts
    struct ArmStats { let name: String; let med: Double; let min: Double; let max: Double
                      let byBlock: [Int: Double]; let peakMed: Double }
    var stats: [ArmStats] = []
    for arm in arms {
        let measured = runRecords.filter { $0.arm == arm.name && $0.run > 0 }
        let walls = measured.map(\.wallSeconds)
        var byBlock: [Int: Double] = [:]
        for b in 0..<blocks { byBlock[b] = median(measured.filter { $0.block == b }.map(\.wallSeconds)) }
        stats.append(ArmStats(name: arm.name, med: median(walls),
                              min: walls.min() ?? 0, max: walls.max() ?? 0,
                              byBlock: byBlock, peakMed: median(measured.map(\.peakPhysGB))))
    }

    func verdict(_ ref: ArmStats, _ arm: ArmStats) -> String {
        let d = arm.med - ref.med
        let spread = max(ref.max - ref.min, arm.max - arm.min)
        // Sign consistency across block sets (the compounding rule: sign-flip = noise).
        let signs = (0..<blocks).compactMap { b -> Double? in
            guard let r = ref.byBlock[b], let a = arm.byBlock[b], r > 0, a > 0 else { return nil }
            return a - r
        }
        let signFlip = signs.count > 1 && Set(signs.map { $0 > 0 }).count > 1
        if signFlip { return String(format: "Δmed %+.1fs — NOISE (sign flips between block sets)", d) }
        if abs(d) <= spread { return String(format: "Δmed %+.1fs ≤ spread %.1fs — NOISE", d, spread) }
        return String(format: "Δmed %+.1fs > spread %.1fs — MEASURABLE", d, spread)
    }

    // ---- Receipts
    let stamp = stampEarly
    let baseName = "bench_e2e_\(label)_\(stamp)"

    var md = "# bench-e2e receipt — \(label)\n\n"
    md += "- started: \(started) (UTC stamp \(stamp))\n"
    md += "- host: \(hwModel) · \(memGB) GB · \(osVer)\n"
    md += "- repo: \(gitSHA)\(gitDirty) · geometry: \(geometry) · seed \(seed)\n"
    md += "- prompt: \"\(prompt)\"\n"
    md += "- protocol: blocks=\(blocks) (ABBA order) · runs/block=\(runs) (+1 excluded warmup) · cooldown \(Int(cooldown))s · MLX_PROFILE off\n"
    md += "- arms:\n"
    for a in arms {
        md += "  - `\(a.name)`: quant=\(a.quant) decoder=\(a.decoder)"
        if let c = a.cacheGB { md += " cache=\(c)GB" }
        if !a.env.isEmpty { md += " env=\(a.env.map { "\($0)=\($1)" }.sorted().joined(separator: " "))" }
        md += "\n"
    }
    md += "\n## Per-run\n\n| arm | block | run | wall s | peak phys GB | mlx act/cache GB | thermal | cos(arm-first) |\n|---|---|---|---|---|---|---|---|\n"
    for r in runRecords {
        md += String(format: "| %@ | %d | %@ | %.1f | %.2f | %.1f/%.1f | %@→%@ | %@ |\n",
                     r.arm, r.block, r.run == 0 ? "warm" : "\(r.run)", r.wallSeconds, r.peakPhysGB,
                     r.mlxActiveGB, r.mlxCacheGB, r.thermalBefore, r.thermalAfter,
                     r.cosineVsArmFirst.map { String(format: "%.6f", $0) } ?? "—")
    }
    md += "\n## Blocks (load + ratchet detector)\n\n| arm | block | prewarm s | load s | post-drop floor GB |\n|---|---|---|---|---|\n"
    for b in blockRecords {
        md += String(format: "| %@ | %d | %.1f | %.1f | %.2f |\n",
                     b.arm, b.block, b.prewarmSeconds, b.loadSeconds, b.postDropFloorGB)
    }
    md += "\n## Arm stats (measured runs only)\n\n| arm | median s | min–max s | median peak GB |\n|---|---|---|---|\n"
    for s in stats {
        md += String(format: "| %@ | %.1f | %.1f–%.1f | %.2f |\n", s.name, s.med, s.min, s.max, s.peakMed)
    }
    if stats.count > 1 {
        md += "\n## Verdicts (vs `\(stats[0].name)`)\n\n"
        for s in stats.dropFirst() { md += "- `\(s.name)`: \(verdict(stats[0], s))\n" }
    }
    if !crossArm.isEmpty {
        md += "\n## Cross-arm output (first measured run, vs `\(arms[0].name)`)\n\n| arm | cosine | maxAbs | class |\n|---|---|---|---|\n"
        for c in crossArm {
            md += String(format: "| %@ | %.6f | %.5f | %@ |\n", c.arm, c.cosine, c.maxAbs,
                         c.expectIdentical ? "same-weights bf16 (expect ≈1)"
                                           : "not gated (q8 nondeterminism / divergent-by-design)")
        }
    }
    md += "\n> Doctrine: totals here are UNPROFILED. For a stage split, run the same arm once under\n"
    md += "> `MLX_PROFILE=csv` separately and never compare its timings across variants (PROFILING.md).\n"

    let mdPath = "\(outDir)/\(baseName).md"
    try md.write(toFile: mdPath, atomically: true, encoding: .utf8)

    struct Receipt: Codable {
        let label: String, stamp: String, host: String, memGB: Int, os: String, git: String
        let geometry: String, seed: UInt64, prompt: String
        let blocks: Int, runsPerBlock: Int, cooldownS: Double
        let runs: [RunRecord], blockRecords: [BlockRecord]
    }
    let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    let json = try enc.encode(Receipt(label: label, stamp: stamp, host: hwModel, memGB: memGB,
                                      os: osVer, git: gitSHA + gitDirty, geometry: geometry,
                                      seed: seed, prompt: prompt, blocks: blocks, runsPerBlock: runs,
                                      cooldownS: cooldown, runs: runRecords, blockRecords: blockRecords))
    try json.write(to: URL(fileURLWithPath: "\(outDir)/\(baseName).json"))

    print("\n[bench-e2e] ───────── summary ─────────")
    for s in stats {
        print(String(format: "[bench-e2e] %@  median %.1fs  (%.1f–%.1f)  peak %.2f GB",
                     s.name, s.med, s.min, s.max, s.peakMed))
    }
    if stats.count > 1 {
        for s in stats.dropFirst() { print("[bench-e2e] \(s.name) vs \(stats[0].name): \(verdict(stats[0], s))") }
    }
    for c in crossArm {
        print(String(format: "[bench-e2e] output %@ vs %@: cos=%.6f maxAbs=%.5f (%@)",
                     c.arm, arms[0].name, c.cosine, c.maxAbs,
                     c.expectIdentical ? "same-weights" : "divergent-by-design"))
    }
    try? FileManager.default.removeItem(atPath: partialPath)   // superseded by the final receipts
    print("[bench-e2e] receipts: \(mdPath) + .json")
}
