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
//        LTX_ENC=q8                                         (int8 TEXT encoder — 13 GB vs 22)
//        LTX_ENC_TREE=<dirname>                             (CONTROL ARMS ONLY — names an encoder
//                                                            tree directly, e.g. ltx-2.5-mlx-q4 or
//                                                            -poison. Bypasses resolution; never an
//                                                            acceptance number.)
//        LTX_T2V_SAVE=<path>                                (optional MP4 write)
//
// 🔑 MEASURED 2026-08-16 — NEITHER LEVER WORKS ALONE, and that is the whole finding. At
// compact24 (512x288x121): baseline 31.92 GB ❌ · streaming only 24.79 ❌ · int8 encoder only
// 31.88 ❌ · BOTH 14.57 ✅ (budget 16.8). Resident is DiT-bound at ~31.9 so the encoder swap is
// invisible there; streaming removes the DiT floor and exposes the ENCODER floor at ~24.8, which
// is geometry-INDEPENDENT (256x256x9 and 512x288x121 agree to 0.01 GB — that constancy is what
// identifies it as the bf16 12B encoder rather than the decode window).

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
    // LTX_NO_CHECKPOINT=1 points at a path that does not exist, which is exactly how a HOSTED v2
    // deployment looks: granules downloaded, transformer never fetched. `bindStore` serves the 59
    // globals from `globals.granule` and integrity comes from the manifest hashes. This proves on
    // REAL weights what `--stream-tiny-gate` proves synthetically.
    let noCheckpoint = env["LTX_NO_CHECKPOINT"] == "1"
    let transformerPath: URL? = noCheckpoint
        ? URL(fileURLWithPath: "\(base)/ltx-2.5-mlx-ditq8/DOES-NOT-EXIST.safetensors")
        : (quant == .int8
            ? URL(fileURLWithPath: "\(base)/ltx-2.5-mlx-ditq8/transformer-distilled.safetensors")
            : nil)
    // Tri-state like the config: unset ⇒ follow the profile's advice. Defaulting this to "on"
    // meant the harness could not measure the SHIPPING default on a tier that does not stream
    // (max128 printed "STREAMED" while the profile says resident) — a harness that cannot express
    // the shipping configuration cannot validate it.
    let streamOverride: Bool? = env["LTX_STREAM_25"].map { $0 != "0" }
    // ROOT, not the tree itself: `LTX2Configuration.resolvedGranuleDirectory` appends a
    // quant-keyed subdir (int8 → "q8", bf16 → "bf16"), matching 2.3's `ltx-granules/{bf16,q8,q4}`.
    let granuleRoot = URL(fileURLWithPath: "/Volumes/Satechi/Models/ltx-granules-25")
    let granuleTree = granuleRoot.appendingPathComponent(quant == .int8 ? "q8" : "bf16")

    let streamed = streamOverride ?? profile.recommendedStreamedBlocks
    if streamed, !FileManager.default.fileExists(
        atPath: granuleTree.appendingPathComponent("manifest.json").path) {
        print("[t2v-spot25] FAIL ❌ no granule tree at \(granuleTree.path) — lay it out with "
            + "`ltx-granule-layout <transformer.safetensors> \(granuleTree.path)` first")
        exit(2)
    }

    // 🔑 LTX_ENC=q8 swaps the TEXT ENCODER, not the DiT. `ltx-2.5-mlx-q8` is the int8
    // text-encoder sibling — 13 GB of gemma4 vs the bf16 tree's 22 GB — and its transformer
    // symlinks back to bf16, which is why it is a trap for a DiT arm and the right tree for an
    // ENCODER arm. `transformerPath` below still points at `-ditq8`, so the DiT is unaffected.
    // Measured 2026-08-16: 2.5's whole-run peak is geometry-INDEPENDENT at ~24.8 GB (256×256×9
    // and 512×288×121 agree to 0.01 GB), i.e. it is the encoder, not the decode window — so this
    // is the only knob that can move 2.5 below balanced32.
    // LTX_ENC is an OVERRIDE now, not the selector: unset ⇒ nil ⇒ the profile's advice decides
    // (compact24/balanced32 → int8). The tree is then derived from the config's OWN resolution
    // rather than re-implemented here — re-deriving it would let this harness pass while the
    // shipping resolution was broken, which is the failure mode `--ltx25-package-gate` case 18
    // exists to prevent.
    let encOverride: Quant? = switch env["LTX_ENC"] {
        case "q8", "int8": .int8
        case "bf16": .bf16
        default: nil
    }
    var cfg = LTX2Configuration(
        quant: quant,
        ltxDirectory: nil,            // set below, from the RESOLVED encoder precision
        transformerPath: transformerPath,
        gemmaDirectory: nil,          // 2.5's Gemma-4 encoder lives INSIDE the components tree
        modelsRootDirectory: URL(fileURLWithPath: "/Volumes/Satechi/Models"),
        profile: profile)
    cfg.textEncoderQuant = encOverride
    var encTree = cfg.effectiveTextEncoderQuant == .int8 ? "ltx-2.5-mlx-q8" : "ltx-2.5-mlx"
    // ⚠️ CONTROL-ARM HATCH — NOT a shipping path. `LTX_ENC_TREE` names an encoder tree directly so
    // the REJECTED (`-q4`, connector 0.996728) and DELIBERATELY-BROKEN (`-poison`, 0.829329 on real
    // tokens) siblings can be run as poison controls. A quality A/B whose deciding metric cannot
    // fail on a known-bad encoder is not a test — the repo-wide rule from AB-L-0017.
    // It bypasses the config's own resolution ON PURPOSE, which is exactly why it must never be
    // used for an acceptance number: case 18 exists to stop this harness re-deriving what it should
    // be reading. It announces itself loudly so no receipt can quote it by accident.
    if let raw = env["LTX_ENC_TREE"], !raw.isEmpty {
        encTree = raw
        FileHandle.standardError.write(Data(
            "⚠️  LTX_ENC_TREE=\(raw) — CONTROL ARM, encoder resolution BYPASSED. Not an acceptance number.\n"
                .utf8))
    }
    cfg.ltxDirectory = URL(fileURLWithPath: "\(base)/\(encTree)")
    cfg.streamedBlocks = streamOverride          // nil ⇒ the profile decides
    if streamed {
        cfg.granuleRootDirectory = granuleRoot
        // ⚠️ `forceStreamGate`, NOT `streamingOptions.gatePolicy` — the latter does not survive
        // resolution (package-gate case 33). Unset ⇒ nil ⇒ the profile decides; LTX_STREAM_GATE is
        // the explicit escape hatch in both directions.
        switch env["LTX_STREAM_GATE"] {
        case "force": cfg.forceStreamGate = true
        case "auto":  cfg.forceStreamGate = false
        default:      break            // follow the profile
        }
    }

    // i2v conditions on a clean first frame and pulls in the ~4.9 GB i2v-adapter LoRA, so it is a
    // materially different footprint from t2v — it must be measured, not inferred from the t2v row.
    let i2v = env["LTX_I2V"] == "1"
    let lane = streamed ? "STREAMED" : "RESIDENT"
    print("[t2v-spot25] request \(width)×\(height)×\(frames)f \(quantName) · tier=\(profile.rawValue)"
        + " · DiT lane: \(lane) · encoder: \(encTree) · mode: \(i2v ? "i2v" : "t2v")"
        + " · gate: \(cfg.resolvedStreamingOptions.gatePolicy.rawValue)")
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
    // The adapter is a real resident cost on the i2v path; leaving it cold would both slow the run
    // and measure paging rather than footprint (the pruna trap).
    if i2v { warm.append(URL(fileURLWithPath:
        "/Volumes/Satechi/Models/ltx-lora-cache/i2v-adapter.safetensors")) }
    prewarmFiles(warm)
    print(String(format: "[t2v-spot25] prewarm %.1fs (%d files%@)", Date().timeIntervalSince(p0),
                 warm.count, streamed ? ", transformer excluded by streamedBlocks" : ""))

    let sampler = PhysSampler(); sampler.start()
    // LTX_PROGRESS=1 observes the ENGINE's RunProgress plane — deliberately NOT the core's
    // LTX2Progress sink, which `MLXLTX25Package.run` rebinds for its own forwarding. Watching the
    // engine plane proves the whole chain (core -> wrapper -> contract) and prints exactly what a
    // UI would receive. Diagnostic only; unbound by default so gates and measurements are unchanged.
    let wantProgress = ProcessInfo.processInfo.environment["LTX_PROGRESS"] == "1"
    let progressSink: RunProgress.Sink = { r in
        let n = r.phase.rawValue
        var line = "[progress] \(n)"
        if let s = r.step, let t = r.totalSteps { line += " \(s)/\(t)" } else { line += " (no sub-steps)" }
        if let st = r.stage, let ts = r.totalStages { line += "  stage \(st)/\(ts)" }
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    let pkg = MLXLTX25Package(configuration: cfg)
    try await pkg.load()
    Memory.clearCache()
    // 🚨 DECLARE FROM HERE, not from the post-run sample below. `--mem-bench25` shipped the same
    // bug (AB-R-0078): its "resident floor" was one post-run, post-clearCache reading, which under
    // eviction measures RETAINED MMAP PAGES and is unstable run-to-run — 11.39 vs 5.72 GB for the
    // identical geometry and arm, while phys-after-load moved 0.00. This arm inherited the same
    // instrument for its floor=/act= columns until 2026-08-20. PEAK was never affected, so every
    // ACCEPTANCE number quoted from this harness stands; only the split was wrong.
    let afterLoad = physFootprintBytes()

    // Synthetic init frame, matching `--i2v-spot`'s approach: this is a FOOTPRINT arm and shapes
    // drive cost, so image CONTENT is irrelevant. ⚠️ Do not reuse this arm for a quality claim.
    let initPNG = i2v ? try syntheticInitPNG(width: width, height: height) : Data()
    func request(_ nf: Int) -> T2VRequest {
        let prompt = env["LTX_T2V_PROMPT"]
            ?? "a fox running down a beach at sunset, waves rolling in"
        guard i2v else {
            return T2VRequest(prompt: prompt, numFrames: nf, fps: 24,
                              width: width, height: height, seed: 42)
        }
        return T2VRequest(prompt: prompt,
                          initImage: MLXToolKit.Image(format: .png, data: initPNG),
                          numFrames: nf, fps: 24, width: width, height: height, seed: 42,
                          metaData: [LoRAMetaKeys.id: .string("i2v-adapter")])
    }

    // Warmup at 9f — kernel compile is per-process and would otherwise land inside the measured
    // window. Discarded, and the cache cleared before the floor is taken.
    _ = try await pkg.run(request(9))
    Memory.clearCache()
    let floor = physFootprintBytes()
    print(String(format: "[t2v-spot25] phys-after-load=%.2f GB · post-run retain=%.2f GB "
                     + "(both reported for comparison; the DECLARE line below uses neither)",
                 gbOf(afterLoad), gbOf(floor)))

    sampler.resetMax()
    let r0 = Date()
    let resp = try await RunProgress.$sink.withValue(wantProgress ? progressSink : nil) {
        try await pkg.run(request(frames))
    } as! T2VResponse
    if let save = env["LTX_T2V_SAVE"], !save.isEmpty {
        try resp.video.data.write(to: URL(fileURLWithPath: (save as NSString).expandingTildeInPath))
    }
    let peak = sampler.maxBytes()
    let held = sampler.minBytes(); sampler.stop()
    // ⚠️ THREE candidate residencies, and under EVICTION only one is honest:
    //   afterLoad — everything loaded at once; a transient the run never returns to. Measured
    //               22.13 GB against a 14.59 GB whole-run PEAK, i.e. a "resident" bigger than the
    //               peak and an activation of 0.00. This is the shape-inversion CLAUDE.md warns of.
    //   retain    — one post-run, post-clearCache sample; unstable run-to-run (11.39 vs 5.72 GB on
    //               identical work) because the weights are genuinely gone by then (AB-R-0078).
    //   held      — the LOW-WATER during the measured run: what survives across stage boundaries.
    // `held` is the one the governor should reserve; `peak − held` is the transient on top of it.
    let resident = min(held, peak)
    let activation = peak > resident ? peak - resident : 0

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
    print(String(format: "[t2v-spot25] SPLIT lane=%@ mode=%@ quant=%@ enc=%@ · "
                     + "resident(held)=%.2f GB act=%.2f GB peak=%.2f GB "
                     + "· [afterLoad=%.2f retain=%.2f — neither is declarable under eviction]",
                 lane as NSString, (i2v ? "i2v" : "t2v") as NSString, quantName as NSString,
                 (encTree.hasSuffix("-q8") ? "int8" : "bf16") as NSString,
                 gbOf(resident), gbOf(activation), gbOf(peak), gbOf(afterLoad), gbOf(floor)))
    print(String(format: "[t2v-spot25] DECLARE → residentBytes=%.2f GB peakActivationBytes=%.2f GB",
                 gbOf(resident), gbOf(activation)))
    print(String(format: "[t2v-spot25] ACCEPTANCE tier=%@ budget=%.2f GB · measured stage-max "
                     + "%.2f GB → %@",
                 profile.rawValue as NSString, budgetGB, gbOf(peak), verdict as NSString))
    print("[t2v-spot25] ⚠️ one geometry, one run — a tier DECLARATION needs the profile's own "
        + "clamped geometry and a repeat, not this single point")
}
