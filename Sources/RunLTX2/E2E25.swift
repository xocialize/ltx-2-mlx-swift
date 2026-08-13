// E2E25.swift — a real LTX-2.5 two-stage generation through the Swift pipeline.
//
// The gates cover every seam individually; this covers the ASSEMBLED path, which no unit
// gate can substitute for. It is the first thing that answers "does a 2.5 checkpoint
// actually produce a video through LTX2Pipeline", as opposed to "is each component right".

import Foundation
import LTX2
import MLX

func e2e25(width: Int, height: Int, frames: Int) async throws {
    let base = ProcessInfo.processInfo.environment["LTX25_DIR"]
        ?? "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx"
    let ltxDir = URL(fileURLWithPath: base)
    guard FileManager.default.fileExists(atPath: base) else {
        print("[e2e25] model dir not present: \(base)"); exit(2)
    }

    // The whole point: this must resolve as 2.5 from the checkpoint, or the run silently
    // exercises the 2.3 path and proves nothing about the delta.
    let is25 = LTX2Pipeline.isLTX25(ltxDir: ltxDir)
    print("[e2e25] dir=\(base)")
    print("[e2e25] isLTX25=\(is25) (gemma4-12b-ltx-v1 present)")
    guard is25 else {
        print("[e2e25] FAIL — 2.5 not detected; the run would silently take the 2.3 path")
        fflush(stdout); exit(1)
    }

    var warm = [ltxDir.appendingPathComponent("transformer-distilled.safetensors")]
    for f in ["connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
              "audio_vae.safetensors", "vocoder.safetensors",
              "spatial_upscaler_x2_v1_1.safetensors"] {
        warm.append(ltxDir.appendingPathComponent(f))
    }
    let g4 = LTX2Pipeline.gemma4Dir(ltxDir: ltxDir)
    warm.append(contentsOf: ((try? FileManager.default.contentsOfDirectory(at: g4, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" })
    prewarmFiles(warm)

    // gemmaDir is unused on 2.5 (the encoder is in-dir) but the loader still takes it.
    let pipeline = try await LTX2Pipeline.load(ltxDir: ltxDir, gemmaDir: g4, transformerPath: nil)

    let t0 = Date()
    MLX.GPU.resetPeakMemory()
    let out = try await pipeline.t2vTwoStage(
        prompt: "A lone lighthouse on a rocky headland during a storm, waves exploding "
              + "against the rocks, the beam sweeping through sheets of rain. Cinematic.",
        height: height, width: width, numFrames: frames, fps: 24, seed: 4242)
    let secs = Date().timeIntervalSince(t0)
    let peak = Double(MLX.GPU.peakMemory) / 1e9

    let px = out.video
    eval(px)
    let finite = MLX.all(MLX.isFinite(px)).item(Bool.self)
    let mean = px.asType(.float32).mean().item(Float.self)
    let f32 = px.asType(.float32)
    let std = MLX.sqrt(MLX.mean(MLX.square(f32 - MLX.mean(f32)))).item(Float.self)
    print(String(format: "[e2e25] %@ in %.1fs  peak %.2f GB", "\(px.shape)" as NSString, secs, peak))
    print(String(format: "[e2e25] pixels finite=%@ mean=%.4f std=%.4f", finite ? "yes" : "NO", mean, std))

    var fails: [String] = []
    if !finite { fails.append("output contains NaN/Inf") }
    // A degenerate frame (all one value) is the classic silent-failure signature.
    if std < 0.01 { fails.append(String(format: "output is nearly constant (std %.5f) — degenerate", std)) }

    if let a = out.audio {
        eval(a)
        print("[e2e25] audio \(a.shape) finite=\(MLX.all(MLX.isFinite(a)).item(Bool.self))")
    }
    if fails.isEmpty { print("[e2e25] PASS ✅  2.5 generates end to end through LTX2Pipeline"); fflush(stdout) }
    else { for f in fails { print("[e2e25] FAIL — \(f)") }; fflush(stdout); exit(1) }
}
