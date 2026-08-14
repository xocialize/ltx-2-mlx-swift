// E2E25.swift — a real LTX-2.5 two-stage generation through the Swift pipeline.
//
// The gates cover every seam individually; this covers the ASSEMBLED path, which no unit
// gate can substitute for. It is the first thing that answers "does a 2.5 checkpoint
// actually produce a video through LTX2Pipeline", as opposed to "is each component right".

import Foundation
import LTX2
import MLX

/// print + flush. Redirected stdout is block-buffered, so an abnormal exit would otherwise
/// discard every diagnostic printed before it — the failure then looks like a broken build.
private func say(_ m: String) { print(m); fflush(stdout) }

func e2e25(width: Int, height: Int, frames: Int) async throws {
    let base = ProcessInfo.processInfo.environment["LTX25_DIR"]
        ?? "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx"
    let ltxDir = URL(fileURLWithPath: base)
    guard FileManager.default.fileExists(atPath: base) else {
        say("[e2e25] model dir not present: \(base)"); exit(2)
    }

    // The whole point: this must resolve as 2.5 from the checkpoint, or the run silently
    // exercises the 2.3 path and proves nothing about the delta.
    let is25 = LTX2Pipeline.isLTX25(ltxDir: ltxDir)
    say("[e2e25] dir=\(base)")
    say("[e2e25] isLTX25=\(is25) (gemma4-12b-ltx-v1 present)")
    guard is25 else {
        say("[e2e25] FAIL — 2.5 not detected; the run would silently take the 2.3 path")
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
    // Resident floor BEFORE the run, read from activeMemory — note resetPeakMemory() ZEROES
    // the counter rather than seeding it with current usage, so peakMemory here reads 0.00.
    let floorGB = Double(MLX.GPU.activeMemory) / 1e9
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
    // ⚠️ NEVER use %s or %@ with a Swift String in String(format:) — it arrives as a bridged
    // object pointer and CFString runs strlen on it, segfaulting. That mistake cost this
    // project a phantom "2.5 eviction bug": the crash fired AFTER the video was generated,
    // and an attribution table then compared a scaffolded binary against a clean-HEAD
    // baseline, blaming the env var. Interpolate instead.
    say("[e2e25] \(px.shape) in \(String(format: "%.1f", secs))s  floor \(String(format: "%.2f", floorGB)) GB  peak \(String(format: "%.2f", peak)) GB")
    say("[e2e25] pixels finite=\(finite ? "yes" : "NO") mean=\(String(format: "%.4f", mean)) std=\(String(format: "%.4f", std))")

    var fails: [String] = []
    if !finite { fails.append("output contains NaN/Inf") }
    // A degenerate frame (all one value) is the classic silent-failure signature.
    if std < 0.01 { fails.append(String(format: "output is nearly constant (std %.5f) — degenerate", std)) }

    if let a = out.audio {
        eval(a)
        say("[e2e25] audio \(a.shape) finite=\(MLX.all(MLX.isFinite(a)).item(Bool.self))")
    }
    if fails.isEmpty { print("[e2e25] PASS ✅  2.5 generates end to end through LTX2Pipeline"); fflush(stdout) }
    else { for f in fails { print("[e2e25] FAIL — \(f)") }; fflush(stdout); exit(1) }
}

/// A real DFR run on the 2.5 checkpoint — the ASSEMBLED path `--dfr-gate` cannot substitute for.
///
/// ⚠️ `rounds` is a QUALITY/parity feature, not a speed lever: the Python measurement has them at
/// x2.767 (r=1) / x3.542 (r=2), and an operator judged NATIVE generation better. Timing is printed
/// as run context, NOT as an A/B — single-pair desktop timings drift ±15 s (see BENCH.md).
func dfr25(width: Int, height: Int, frames: Int, rounds: Int) async throws {
    let base = ProcessInfo.processInfo.environment["LTX25_DIR"]
        ?? "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx"
    let ltxDir = URL(fileURLWithPath: base)
    guard FileManager.default.fileExists(atPath: base) else {
        say("[dfr25] model dir not present: \(base)"); exit(2)
    }
    guard LTX2Pipeline.isLTX25(ltxDir: ltxDir) else {
        say("[dfr25] FAIL — 2.5 not detected; DFR needs a generated-keyframe checkpoint")
        fflush(stdout); exit(1)
    }

    // Resolve the plan BEFORE loading anything: a mis-sized request should cost milliseconds.
    let plan = try DFRPlan.resolve(requestedFrames: frames, height: height, width: width,
                                   fps: 24, rounds: rounds)
    say("[dfr25] dir=\(base)")
    say("[dfr25] request \(width)x\(height) \(frames)f rounds=\(rounds)"
        + " → \(plan.width)x\(plan.height), canvas \(plan.canvas.canvasFrames)f"
        + " (segment \(plan.canvas.segment), slots \(plan.canvas.slotPositions))")
    say("[dfr25] deliver \(plan.deliveredFrames)f, playback \(Int(plan.playbackFPS)) fps,"
        + " conditioning \(Int(plan.conditioningFPS)) fps, audio \(plan.audioKeepTokens)/\(plan.audioTokens) tokens")

    var warm = [ltxDir.appendingPathComponent("transformer-distilled.safetensors")]
    for f in ["connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
              "audio_vae.safetensors", "vocoder.safetensors",
              "spatial_upscaler_x2_v1_1.safetensors", "temporal_upscaler_x2_v1_0.safetensors"] {
        warm.append(ltxDir.appendingPathComponent(f))
    }
    let g4 = LTX2Pipeline.gemma4Dir(ltxDir: ltxDir)
    warm.append(contentsOf: ((try? FileManager.default.contentsOfDirectory(at: g4, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" })
    prewarmFiles(warm)

    let pipeline = try await LTX2Pipeline.load(ltxDir: ltxDir, gemmaDir: g4, transformerPath: nil)
    let t0 = Date()
    let floorGB = Double(MLX.GPU.activeMemory) / 1e9
    MLX.GPU.resetPeakMemory()
    let (out, latents) = try await pipeline.dfr(
        prompt: "A lone lighthouse on a rocky headland during a storm, waves exploding "
              + "against the rocks, the beam sweeping through sheets of rain. Cinematic.",
        height: height, width: width, numFrames: frames, fps: 24, seed: 4242,
        temporalUpsampleRounds: rounds)
    let secs = Date().timeIntervalSince(t0)
    let peak = Double(MLX.GPU.peakMemory) / 1e9

    let px = out.video
    eval(px)
    let finite = MLX.all(MLX.isFinite(px)).item(Bool.self)
    let f32 = px.asType(.float32)
    let mean = f32.mean().item(Float.self)
    let std = MLX.sqrt(MLX.mean(MLX.square(f32 - MLX.mean(f32)))).item(Float.self)
    say("[dfr25] \(px.shape) in \(String(format: "%.1f", secs))s  floor \(String(format: "%.2f", floorGB)) GB  peak \(String(format: "%.2f", peak)) GB")
    say("[dfr25] pixels finite=\(finite ? "yes" : "NO") mean=\(String(format: "%.4f", mean)) std=\(String(format: "%.4f", std))")
    say("[dfr25] generated keyframes at \(latents.generatedKeyframePositions)"
        + " (\(latents.generatedKeyframeLatents.count) latents)")

    var fails: [String] = []
    if !finite { fails.append("output contains NaN/Inf") }
    if std < 0.01 { fails.append(String(format: "output is nearly constant (std %.5f) — degenerate", std)) }
    // The frame contract is the whole point of the rounds: assert the DELIVERED clip, not the canvas.
    if px.dim(2) != plan.deliveredFrames {
        fails.append("delivered \(px.dim(2)) frames, expected \(plan.deliveredFrames)")
    }
    if latents.generatedKeyframePositions.contains(where: { $0 >= plan.deliveredFrames }) {
        fails.append("reported a keyframe outside the delivered clip: \(latents.generatedKeyframePositions)")
    }
    if let a = out.audio {
        eval(a)
        let aFinite = MLX.all(MLX.isFinite(a)).item(Bool.self)
        // 48 kHz stereo; the trim must keep the audio inside the picture's duration.
        let audSecs = Double(a.dim(2)) / 48_000.0
        let vidSecs = Double(px.dim(2)) / latents.playbackFPS
        say("[dfr25] audio \(a.shape) finite=\(aFinite ? "yes" : "NO")"
            + " \(String(format: "%.3f", audSecs))s vs video \(String(format: "%.3f", vidSecs))s"
            + " @ \(Int(latents.playbackFPS)) fps")
        if !aFinite { fails.append("audio contains NaN/Inf") }
        if audSecs > vidSecs + 0.1 {
            fails.append(String(format: "audio %.3fs outlasts the %.3fs picture — the canvas trim did not apply",
                                audSecs, vidSecs))
        }
    }
    if fails.isEmpty { print("[dfr25] PASS ✅  DFR generates end to end on the real 2.5 checkpoint"); fflush(stdout) }
    else { for f in fails { print("[dfr25] FAIL — \(f)") }; fflush(stdout); exit(1) }
}
