// ICSpot25.swift — `--ic-spot25`, one live IC-adapter render on the LTX-2.5 base through the REAL
// package surface (AB-A-0048). The `--lora-gate25` arm proves a 2.3-authored IC LoRA lands and
// detaches on the 2.5 DiT; it says nothing about the IC PATH — reference tokens + LoRA — which is
// what a consumer (LTX Studio's ModelSheet → Ingredients) actually runs. This arm is that run,
// headless, against `IC-P3-FIXTURE.md`'s canonical params so the four perceptual reads have their
// established baseline (the 2026-07-03 2.3 result in `LTX_TESTING/fixtures/`).
//
// It drives `MLXLTX25Package` with the same `ICMetaKeys` intake the app uses (`ic.adapterId`,
// `ic.adapterStrength`, `ic.referencePath`, `ic.referenceStrength`) and the dual-part prompt
// `"Reference sheet: {elements}\n\nGenerated video: {action}"` — nothing here re-implements a
// resolution the wrapper owns (registry entry → `stage2: skip`, downscale 1, looped-still ingest).
//
// usage: RunLTX2 --ic-spot25 [W] [H] [F]            (default 768 448 121 — the fixture geometry)
//   env: LTX_IC_ADAPTER=ingredients                   registry id (ingredients | union-control | lipdub)
//        LTX_IC_REF=<path>                            reference sheet PNG (or clip for video-ref adapters)
//        LTX_IC_DESC=<text> | LTX_IC_DESC_FILE=<path> the "Reference sheet:" half (elements string)
//        LTX_IC_ACTION=<text>                          the "Generated video:" half
//        LTX_IC_STRENGTH=1.4 · LTX_IC_REF_STRENGTH=1.0
//        LTX_IC_QUANT=bf16|int8                        DiT precision (default bf16 — the fixture's faithful tier)
//        LTX_IC_BASE=1                                 SAME prompt/seed with NO adapter (the A/B control arm)
//        LTX_IC_SAVE=<path.mp4>                        write the artifact (never swallowed)
//        LTX_SEED=42
//
// Geometry is UNCONSTRAINED (`profile: nil`) on purpose — the fixture is 768 wide and max128 would
// clamp it to 704 (IC-P3-FIXTURE.md "Notes for the IC leg"). Numbers from this arm are a
// FIXTURE-PARITY measurement, not a tier-admissibility claim; `--t2v-spot25` owns those.

import Foundation
import MLX
import MLXLTX2
import MLXToolKit
import LTX2

func icSpot25(width: Int, height: Int, frames: Int) async throws {
    let env = ProcessInfo.processInfo.environment
    let base = "/Volumes/Satechi/Models/xocialize"
    let adapter = env["LTX_IC_ADAPTER"] ?? "ingredients"
    let baseArm = env["LTX_IC_BASE"] == "1"
    let quantName = (env["LTX_IC_QUANT"] ?? "bf16").lowercased()
    let quant: Quant = (quantName == "int8" || quantName == "q8") ? .int8 : .bf16
    let seed = env["LTX_SEED"].flatMap { UInt64($0) } ?? 42

    let fixtures = "/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/LTX_TESTING/fixtures"
    let refPath = env["LTX_IC_REF"]
        ?? "\(fixtures)/modelsheet-ingredients-bundle/Tactical.ingredients/reference_sheet.png"
    let desc: String = {
        if let f = env["LTX_IC_DESC_FILE"], let s = try? String(contentsOfFile: f, encoding: .utf8) {
            return s.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return env["LTX_IC_DESC"] ?? "a young woman with tousled shoulder-length silver-grey hair"
    }()
    let action = env["LTX_IC_ACTION"]
        ?? "cinematic anime action scene, a medium tracking shot on a rain-slicked city rooftop at night, "
        + "neon signs glowing faintly through the mist far below. the silver-haired woman in the dark-navy "
        + "tactical suit walks briskly toward the rooftop edge, her coat-tail panel and hair moving in the "
        + "wind, scanning the skyline with a focused expression. she crouches behind a ventilation unit, "
        + "draws the pistol from her thigh holster in one smooth practiced motion, and whispers in a low "
        + "steady voice: 'target's early... moving now.' she vaults over the unit and sprints along the "
        + "rooftop as the camera tracks alongside her. the audio is immersive: steady rain on concrete, her "
        + "boots splashing through shallow puddles, the soft creak of her harness, a distant siren, no music"
    // The community format verified against the Space (IC-P3-FIXTURE.md): blank-line separator, no labels.
    let prompt = "Reference sheet: \(desc)\n\nGenerated video: \(action)"
    let loraStrength = env["LTX_IC_STRENGTH"].flatMap { Float($0) } ?? 1.4
    let refStrength = env["LTX_IC_REF_STRENGTH"].flatMap { Float($0) } ?? 1.0

    guard FileManager.default.fileExists(atPath: refPath) else {
        print("[ic-spot25] FAIL ❌ reference not found: \(refPath)"); exit(2)
    }
    let transformerPath: URL? = quant == .int8
        ? URL(fileURLWithPath: "\(base)/ltx-2.5-mlx-ditq8/transformer-distilled.safetensors")
        : nil
    var cfg = LTX2Configuration(
        family: .ltx25,
        quant: quant,
        ltxDirectory: URL(fileURLWithPath: "\(base)/ltx-2.5-mlx"),
        transformerPath: transformerPath,
        gemmaDirectory: nil,           // 2.5's Gemma-4 encoder lives inside the components tree
        modelsRootDirectory: URL(fileURLWithPath: "/Volumes/Satechi/Models"),   // ⇒ ltx-lora-cache/
        profile: nil)                  // Unconstrained: the 768-wide fixture must not clamp to 704
    cfg.streamedBlocks = false         // resident DiT — the fixture regime, not a low-tier claim

    print("[ic-spot25] \(baseArm ? "BASE (no adapter)" : "IC adapter=\(adapter) lora=\(loraStrength) ref=\(refStrength)")"
        + " · \(width)×\(height)×\(frames)f seed \(seed) · DiT \(quantName) · 2.5 tree \(cfg.ltxDirectory?.lastPathComponent ?? "?")")
    print("[ic-spot25] reference: \(refPath)")

    let p0 = Date()
    var warm: [URL] = []
    for p in cfg.prewarmPaths {
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: p.path, isDirectory: &isDir), isDir.boolValue {
            warm.append(contentsOf: ((try? FileManager.default.contentsOfDirectory(
                at: p, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.pathExtension == "safetensors" })
        } else { warm.append(p) }
    }
    if !baseArm { warm.append(URL(fileURLWithPath: "/Volumes/Satechi/Models/ltx-lora-cache/\(adapter).safetensors")) }
    prewarmFiles(warm)
    print(String(format: "[ic-spot25] prewarm %.1fs (%d files)", Date().timeIntervalSince(p0), warm.count))

    let sampler = PhysSampler(); sampler.start()
    let pkg = MLXLTX25Package(configuration: cfg)
    let l0 = Date()
    try await pkg.load()
    Memory.clearCache()
    let loadS = Date().timeIntervalSince(l0)
    let afterLoad = physFootprintBytes()

    var meta: [String: MetaValue] = [:]
    if !baseArm {
        meta[ICMetaKeys.adapterId] = .string(adapter)
        meta[ICMetaKeys.adapterStrength] = .double(Double(loraStrength))
        meta[ICMetaKeys.referencePath] = .string(refPath)
        meta[ICMetaKeys.referenceStrength] = .double(Double(refStrength))
    }
    let request = T2VRequest(prompt: prompt, numFrames: frames, fps: 24,
                             width: width, height: height, seed: seed, metaData: meta)

    let progressSink: RunProgress.Sink = { r in
        var line = "[progress] \(r.phase.rawValue)"
        if let s = r.step, let t = r.totalSteps { line += " \(s)/\(t)" }
        if let st = r.stage, let ts = r.totalStages { line += "  stage \(st)/\(ts)" }
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }
    sampler.resetMax()
    let r0 = Date()
    let resp = try await RunProgress.$sink.withValue(
        env["LTX_PROGRESS"] == "1" ? progressSink : nil) { try await pkg.run(request) } as! T2VResponse
    let runS = Date().timeIntervalSince(r0)
    let peak = sampler.maxBytes(); let held = sampler.minBytes(); sampler.stop()

    let save = env["LTX_IC_SAVE"].map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        ?? URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
            "ic-spot25-\(baseArm ? "base" : adapter)-\(quantName)-\(width)x\(height)-\(frames)f-s\(seed).mp4")
    try resp.video.data.write(to: save)
    let resident = min(held, peak)
    print(String(format: "[ic-spot25] DONE · load %.1fs · run %.1fs · phys-after-load %.2f GB · peak %.2f GB · held %.2f GB · act %.2f GB",
                 loadS, runS, gbOf(afterLoad), gbOf(peak), gbOf(resident), gbOf(peak > resident ? peak - resident : 0)))
    print("[ic-spot25] → \(save.path) (\(resp.video.data.count / 1_048_576) MiB)")
    await pkg.unload()
}
