// TwoStageVariantGates.swift — `--two-stage-variants-gate [x2|x1_5|temporal|all]`
//
// Gates the VARIANT-AWARE TWO-STAGE WIRING (LTX2Pipeline.t2vTwoStage geometry derivation),
// not the upsampler modules themselves — those are oracle-gated at cosine 1.000000 by
// `--upsampler-variants-gate`. There is no oracle for x1.5/temporal *two-stage*: neither
// the oracle nor upstream ltx-pipelines ships a consumer (verified 2026-08-05), so this
// composition is ours and its acceptance is:
//   1. geometry pre-flight rejects mis-sized targets LOUDLY before any heavy phase,
//   2. each variant lands EXACTLY on the requested output shape,
//   3. the x2 path reproduces the pre-wiring pipeline bit-for-bit (bf16 e2e is
//      deterministic — receipt `probes/bench_e2e_pre2s_20260805-*`).
//
// Geometry legs (all tiny, ~13 s each on the desktop):
//   x2       448×320× 9f  (target lat 10×14 even       → stage 1 224×160)
//   x1_5     480×288× 9f  (target lat  9×15 ÷3         → stage 1 320×192)
//   temporal 448×320×17f  (target latF 3 odd           → stage 1 9f at fps/2)

import Foundation
import LTX2
import MLX

func twoStageVariantsGate(which: String) async throws {
    let base = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx"
    let ltxDir = URL(fileURLWithPath: base)
    let gemmaDir = URL(fileURLWithPath: defaultGemma)

    struct Leg {
        let key: String, file: String, width: Int, height: Int, frames: Int
        // Pre-wiring fingerprint for the regression leg (nil = record-only).
        let preMean: Float?, preStd: Float?
    }
    let legs = [
        Leg(key: "x2", file: "spatial_upscaler_x2_v1_1.safetensors",
            width: 448, height: 320, frames: 9, preMean: -0.25540963, preStd: 0.46904683),
        Leg(key: "x1_5", file: "spatial_upscaler_x1_5_v1_0.safetensors",
            width: 480, height: 288, frames: 9, preMean: nil, preStd: nil),
        Leg(key: "temporal", file: "temporal_upscaler_x2_v1_0.safetensors",
            width: 448, height: 320, frames: 17, preMean: nil, preStd: nil),
    ].filter { which == "all" || $0.key == which }
    guard !legs.isEmpty else {
        print("[two-stage-variants] unknown selector '\(which)' (x2|x1_5|temporal|all)"); exit(2)
    }

    var warm = [ltxDir.appendingPathComponent("transformer-distilled.safetensors")]
    for f in ["connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
              "audio_vae.safetensors", "vocoder.safetensors"] + legs.map(\.file) {
        warm.append(ltxDir.appendingPathComponent(f))
    }
    warm.append(contentsOf: ((try? FileManager.default.contentsOfDirectory(at: gemmaDir, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" })
    prewarmFiles(warm)

    let pipeline = try await LTX2Pipeline.load(ltxDir: ltxDir, gemmaDir: gemmaDir, transformerPath: nil)
    var allPass = true
    func check(_ name: String, _ ok: Bool, _ detail: String) {
        allPass = allPass && ok
        print("[two-stage-variants] \(ok ? "✅" : "❌") \(name) — \(detail)")
    }

    // --- Negative pre-flight legs: every rejection must be TwoStageError, thrown fast ---
    func expectGeometryError(_ name: String, file: String, w: Int, h: Int, f: Int) async {
        pipeline.upsamplerFile = file
        let t0 = Date()
        do {
            _ = try await pipeline.t2vTwoStage(prompt: "x", height: h, width: w, numFrames: f, fps: 24, seed: 42)
            check(name, false, "was ACCEPTED — pre-flight validation missing")
        } catch let e as TwoStageError {
            let fast = Date().timeIntervalSince(t0) < 5     // pre-flight, not a stage-1 denoise
            check(name, fast, "\(e)\(fast ? "" : " (but took \(Int(Date().timeIntervalSince(t0)))s — thrown too late)")")
        } catch {
            check(name, false, "wrong error type: \(error)")
        }
    }
    await expectGeometryError("reject x2 @480×288 (lat 15×9 odd)",
                              file: "spatial_upscaler_x2_v1_1.safetensors", w: 480, h: 288, f: 9)
    await expectGeometryError("reject x1_5 @448×320 (lat 14×10 not ÷3)",
                              file: "spatial_upscaler_x1_5_v1_0.safetensors", w: 448, h: 320, f: 9)
    await expectGeometryError("reject temporal @121f (latF 16 even)",
                              file: "temporal_upscaler_x2_v1_0.safetensors", w: 448, h: 320, f: 121)
    await expectGeometryError("reject missing checkpoint",
                              file: "no_such_upsampler.safetensors", w: 448, h: 320, f: 9)

    // --- Positive legs ---
    for leg in legs {
        pipeline.upsamplerFile = leg.file
        let t0 = Date()
        let out = try await pipeline.t2vTwoStage(prompt: "a cat playing piano",
                                                 height: leg.height, width: leg.width,
                                                 numFrames: leg.frames, fps: 24, seed: 42)
        let wall = Date().timeIntervalSince(t0)
        let v32 = out.video.asType(.float32)
        let mean = MLX.mean(v32).item(Float.self), std = MLX.std(v32).item(Float.self)
        let shapeOK = out.video.shape == [1, 3, leg.frames, leg.height, leg.width]
        let finite = mean.isFinite && std.isFinite && std > 0.05 && std < 2.0
        check("\(leg.key) shape", shapeOK, "\(out.video.shape) vs [1, 3, \(leg.frames), \(leg.height), \(leg.width)]")
        check("\(leg.key) stats", finite,
              String(format: "mean=%.9g std=%.9g wall=%.1fs", mean, std, wall))
        if let pm = leg.preMean, let ps = leg.preStd {
            // bf16 e2e is bit-deterministic — the wired x2 path must reproduce the
            // pre-wiring pipeline exactly (tolerance covers receipt print rounding only).
            let same = abs(mean - pm) <= 2e-6 && abs(std - ps) <= 2e-6
            check("\(leg.key) pre-wiring regression", same,
                  String(format: "Δmean=%.3g Δstd=%.3g vs bench_e2e_pre2s receipt", mean - pm, std - ps))
        }
    }

    print(allPass ? "[two-stage-variants] PASS ✅" : "[two-stage-variants] FAIL ❌")
    if !allPass { exit(1) }
}
