// T2VSpot25.swift — `--t2v-spot25`, the LTX-2.5 tier-admissibility measurement.
//
// WHY THIS EXISTS. `--stream-parity-gate 25-*` and `--stream-auto-gate 25-*` proved that 2.5's DiT
// streams (memcmp-exact, 0.0% steady stall at the real stage-2 N, AB-R-0088/0089) — but they
// measure the **DiT forward alone**. A tier declaration is not that. Acceptance is
// `measured stage-max ≤ 0.7× tier` across the WHOLE run (`LOW-TIER-PLAN.md:148`), and the peak
// stage moves: post-T3b/T3c it is *denoise* on the low tiers and the *decode window* on
// standard64. Quoting a DiT-only number as an admissibility result is the exact shape of
// over-claim this arm exists to prevent.
//
// 🔑 **STREAMING IS OPTED IN VIA THE CONFIG, NOT `LTX_STREAM_GRANULES`.** The env override is read
// by `LTX2Pipeline.effectiveGranuleDirectory` and does stream — but it bypasses
// `LTX2Configuration.streamedBlocks`, and `streamedBlocks` is ALSO what suppresses prewarming the
// 19–35 GB transformer (`prewarmPaths`: `let transformer = streamedBlocks ? nil : r.transformerPath`).
// Measuring through the env alone would stream the blocks while still paging the entire checkpoint
// first — a slower run, and a measurement of a configuration that does not ship. This arm sets the
// real opt-in (`streamedBlocks` + `granuleRootDirectory`) so what is measured is what ships.
//
// ⚠️ **int8 on 2.5 means `-ditq8`.** The path is NOT re-derived here: the config's own
// `LTXFamily.transformerRepoSuffix` owns that mapping, and on 2.5 `-q8` is the int8 TEXT-ENCODER
// sibling whose transformer symlinks back to bf16. A spot gate that hardcoded `-q8` would declare
// int8's ~23 GB while loading the 40 GB bf16 DiT.
//
// PROTOCOL. Warm up at 9f (kernel compile) and DISCARD it; `clearCache()`; sample the resident
// floor; then run the measured geometry with a 25 ms `PhysSampler`. **The acceptance number is the
// sampler's high-water, not a profiler span** — per-span phys is point-in-time and misses the
// mid-stage transient that actually sets the peak (BENCH.md:48). The profiler stays on for stage
// ATTRIBUTION only.
//
// usage: RunLTX2 --t2v-spot25 [W] [H] [F]
//   env: LTX_TIER=compact24|balanced32|standard64|max128   (default: standard64)
//        LTX_QUANT=int8|bf16                                (default: the tier's recommendation)
//        LTX_STREAM_25=1|0                                  (default: 1 — stream the DiT blocks)
//        LTX_T2V_SAVE=<path>                                (optional MP4 write)

import Foundation
import MLX
import MLXLTX2
import MLXToolKit
import LTX2

func t2vSpot25Gate(width: Int, height: Int, frames: Int) async throws {
    let env = ProcessInfo.processInfo.environment
    let base = "/Volumes/Satechi/Models/xocialize"
    let profile = env["LTX_TIER"].flatMap { LTX2Profile(rawValue: $0) } ?? .standard64
    let quantName = env["LTX_QUANT"] ?? {
        switch profile.recommendedQuant {
        case .int8: return "int8"
        case .int4: return "int8"   // 2.5 has NO q4 DiT — fall UP to int8, never derive a q4 path
        default: return "bf16"
        }
    }()
    let quant: Quant = quantName == "bf16" ? .bf16 : .int8
    // 2.5's quantized transformer tree. Named explicitly rather than derived so a wrong suffix is
    // a missing FILE (loud) instead of a silently-bf16 run under an int8 label.
    let transformerPath: URL? = quant == .int8
        ? URL(fileURLWithPath: "\(base)/ltx-2.5-mlx-ditq8/transformer-distilled.safetensors")
        : nil
    let streamed = (env["LTX_STREAM_25"] ?? "1") != "0"
    // ROOT, not the tree itself: `LTX2Configuration.resolvedGranuleDirectory` appends a
    // quant-keyed subdir (int8 → "q8", bf16 → "bf16"), matching 2.3's `ltx-granules/{bf16,q8,q4}`.
    let granuleRoot = URL(fileURLWithPath: "/Volumes/Satechi/Models/ltx-granules-25")
    let granuleTree = granuleRoot.appendingPathComponent(quant == .int8 ? "q8" : "bf16")

    if streamed, !FileManager.default.fileExists(
        atPath: granuleTree.appendingPathComponent("manifest.json").path) {
        print("[t2v-spot25] FAIL ❌ no granule tree at \(granuleTree.path) — lay it out with "
            + "`ltx-granule-layout <transformer.safetensors> \(granuleTree.path)` first")
        exit(2)
    }

    var cfg = LTX2Configuration(
        quant: quant,
        ltxDirectory: URL(fileURLWithPath: "\(base)/ltx-2.5-mlx"),
        transformerPath: transformerPath,
        gemmaDirectory: nil,          // 2.5's Gemma-4 encoder lives INSIDE the components tree
        modelsRootDirectory: URL(fileURLWithPath: "/Volumes/Satechi/Models"),
        profile: profile)
    if streamed {
        cfg.streamedBlocks = true
        cfg.granuleRootDirectory = granuleRoot
    }

    let lane = streamed ? "STREAMED" : "RESIDENT"
    print("[t2v-spot25] request \(width)×\(height)×\(frames)f \(quantName) · tier=\(profile.rawValue)"
        + " · DiT lane: \(lane)")
    if streamed { print("[t2v-spot25] granules \(granuleTree.path)") }

    // Prewarm off the config's OWN prewarmPaths — which, when streamedBlocks is set, deliberately
    // omit the transformer. Reading that list rather than building one is the point: it proves the
    // shipping config knows not to page a checkpoint it will stream.
    let p0 = Date()
    var warm: [URL] = []
    for p in cfg.prewarmPaths {
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: p.path, isDirectory: &isDir), isDir.boolValue {
            warm.append(contentsOf: ((try? FileManager.default.contentsOfDirectory(
                at: p, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.pathExtension == "safetensors" })
        } else {
            warm.append(p)
        }
    }
    prewarmFiles(warm)
    print(String(format: "[t2v-spot25] prewarm %.1fs (%d files%@)", Date().timeIntervalSince(p0),
                 warm.count, streamed ? ", transformer excluded by streamedBlocks" : ""))

    let sampler = PhysSampler(); sampler.start()
    let pkg = MLXLTX25Package(configuration: cfg)
    try await pkg.load()
    Memory.clearCache()

    func request(_ nf: Int) -> T2VRequest {
        T2VRequest(prompt: env["LTX_T2V_PROMPT"]
                       ?? "a fox running down a beach at sunset, waves rolling in",
                   numFrames: nf, fps: 24, width: width, height: height, seed: 42)
    }

    // Warmup at 9f — kernel compile is per-process and would otherwise land inside the measured
    // window. Discarded, and the cache cleared before the floor is taken.
    _ = try await pkg.run(request(9))
    Memory.clearCache()
    let floor = physFootprintBytes()
    print(String(format: "[t2v-spot25] resident floor (post-warmup + clearCache): %.2f GB",
                 gbOf(floor)))

    sampler.resetMax()
    let r0 = Date()
    let resp = try await pkg.run(request(frames)) as! T2VResponse
    if let save = env["LTX_T2V_SAVE"], !save.isEmpty {
        try resp.video.data.write(to: URL(fileURLWithPath: (save as NSString).expandingTildeInPath))
    }
    let peak = sampler.maxBytes(); sampler.stop()
    let activation = peak > floor ? peak - floor : 0

    // The acceptance arithmetic, printed rather than left to the reader — a raw peak next to a
    // tier name invites the reader to eyeball a comparison the rule defines precisely.
    //
    // ⚠️ Deliberately NOT a property on `LTX2Profile`: the budget is 0.7× unified
    // (`LOW-TIER-PLAN.md:33-35`), but the sources DISAGREE at the top tier — the T3 acceptance
    // table uses **89.6** for max128 (0.7×128) while `CLAUDE.md` uses **108.8** (0.85×128, the
    // "eval apps" rate). Baking either into the shipping config would settle a live question by
    // side effect. max128 is reported below with both, and the three tiers that matter for a
    // streaming claim are unambiguous.
    let budgetGB: Double = {
        switch profile {
        case .compact24: return 16.8
        case .balanced32: return 22.4
        case .standard64: return 44.8
        case .max128: return 89.6      // 0.7×; see the caveat above
        }
    }()
    let verdict = gbOf(peak) <= budgetGB ? "✅ WITHIN" : "❌ OVER"
    print(String(format: "[t2v-spot25] run %.1fs  mp4 %.1f MB", Date().timeIntervalSince(r0),
                 Double(resp.video.data.count) / 1_000_000))
    print(String(format: "[t2v-spot25] SPLIT lane=%@ quant=%@ floor=%.2f GB peak=%.2f GB act=%.2f GB",
                 lane as NSString, quantName as NSString, gbOf(floor), gbOf(peak), gbOf(activation)))
    print(String(format: "[t2v-spot25] ACCEPTANCE tier=%@ budget=%.2f GB · measured stage-max "
                     + "%.2f GB → %@",
                 profile.rawValue as NSString, budgetGB, gbOf(peak), verdict as NSString))
    print("[t2v-spot25] ⚠️ one geometry, one run — a tier DECLARATION needs the profile's own "
        + "clamped geometry and a repeat, not this single point")
}
