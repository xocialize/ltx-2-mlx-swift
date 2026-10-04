// Outpaint25.swift — `--outpaint25`, IC outpainting on the LTX-2.5 base (`LTX2Pipeline.icOutpaint`).
//
// Generation only. Canvas construction (padding the source, the fill colour, the pixel mask),
// pasting the original back over the source region, and every metric live in the Python harness
// (`LTX_TESTING/outpaint/`) — so this CLI takes tensors in and gives tensors out, and the
// reference never passes through a lossy codec on its way to the VAE.
//
// usage: RunLTX2 --outpaint25 <job.safetensors> <out-prefix> [<job2.safetensors> <out-prefix2> ...]
//   job keys:  reference (1,3,F,H,W) in [-1,1] — the canvas-sized source, fill region blanked
//              mask      (H,W) or (F,H,W) in [0,1], optional — 1 = real source, 0 = fill region
//   writes:    <prefix>.safetensors  video (F,H,W,3) uint8, decoded and UNCOMPOSITED
//              <prefix>.mp4          the same frames as a preview (+ the generated audio)
//   env: LTX_OP_LORA=<path>                adapter (default ltx-lora-cache/in-outpainting.safetensors)
//        LTX_OP_LORA_STRENGTH=1.0
//        LTX_OP_MASK=diffusers|ltxcore|none convention for the mask (default diffusers = the Space)
//        LTX_OP_TWO_STAGE=1|0              x2 upsample + bare refine (default 1 = the Space)
//        LTX_OP_PROMPT=<text>              default: the Space's generic margin prompt
//        LTX_OP_FPS=24 · LTX_SEED=42 · LTX_OP_QUANT=bf16|int8 · LTX25_DIR=<2.5 tree>
//
// One process loads the base + adapter once and runs every job in order — load is minutes and an
// episode is hundreds of shots, so the per-job cost is what matters. Numbers printed per job are
// run context (wall, phys peak); A/B timing claims belong to `--bench-e2e`.

import Foundation
import MLX
import MLXLTX2
import MLXToolKit
import LTX2

/// The ltx-community outpaint Space's default prompt, verbatim — "describe only the new region",
/// per the adapter card, with a generic description when the user supplies none.
let outpaintDefaultPrompt = "the scene continues naturally beyond the original frame, consistent style and lighting; "
    + "seamlessly extend the scene into the empty margins, matching the existing content."

func outpaint25(jobs: [(input: String, prefix: String)]) async throws {
    let env = ProcessInfo.processInfo.environment
    // LTX_OP_BASE=23 runs the 2.3 tree — the base every published outpaint adapter was TRAINED on,
    // i.e. the control arm for "does this 2.3 adapter transfer to 2.5". 2.5 stays the default.
    let on23 = env["LTX_OP_BASE"] == "23"
    let base = on23 ? "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx"
        : (env["LTX25_DIR"] ?? "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
    let ltxDir = URL(fileURLWithPath: base)
    guard LTX2Pipeline.isLTX25(ltxDir: ltxDir) != on23 else {
        print("[outpaint25] FAIL ❌ \(base) is not the LTX-\(on23 ? "2.3" : "2.5") tree it was selected as"); exit(2)
    }
    let loraPath = env["LTX_OP_LORA"] ?? "/Volumes/Satechi/Models/ltx-lora-cache/in-outpainting.safetensors"
    let baseArm = loraPath == "none"   // the no-adapter control: same reference, same seed, bare base
    guard baseArm || FileManager.default.fileExists(atPath: loraPath) else {
        print("[outpaint25] FAIL ❌ adapter not found: \(loraPath)"); exit(2)
    }
    let loraStrength = env["LTX_OP_LORA_STRENGTH"].flatMap { Float($0) } ?? 1.0
    let maskMode = (env["LTX_OP_MASK"] ?? "diffusers").lowercased()
    let convention: ICAttentionMask.Convention = maskMode == "ltxcore" ? .ltxCore : .diffusers
    let twoStage = env["LTX_OP_TWO_STAGE"] != "0"
    // LTX_OP_STAGE2=anchored pins the source-region tokens to the job's `anchor` (+ `anchor_keep`)
    // in stage 2; `bare` (default) is the Space's composition. See `icOutpaint`.
    let anchored = twoStage && env["LTX_OP_STAGE2"] == "anchored"
    let prompt = env["LTX_OP_PROMPT"] ?? outpaintDefaultPrompt
    let fps = env["LTX_OP_FPS"].flatMap { Double($0) } ?? 24
    let seed = env["LTX_SEED"].flatMap { UInt64($0) } ?? 42
    let quantName = (env["LTX_OP_QUANT"] ?? "bf16").lowercased()
    let transformerPath: URL? = (quantName == "int8" || quantName == "q8")
        ? URL(fileURLWithPath: "\(base)-ditq8/transformer-distilled.safetensors") : nil

    // Validate every job's shapes BEFORE the multi-minute load — a bad job file should cost ms.
    for job in jobs {
        let io = try MLX.loadArrays(url: URL(fileURLWithPath: job.input))
        guard let ref = io["reference"], ref.ndim == 5, ref.dim(0) == 1, ref.dim(1) == 3 else {
            print("[outpaint25] FAIL ❌ \(job.input): `reference` must be (1,3,F,H,W)"); exit(2)
        }
        let f = ref.dim(2), h = ref.dim(3), w = ref.dim(4)
        guard (f - 1) % 8 == 0, h % 32 == 0, w % 32 == 0 else {
            print("[outpaint25] FAIL ❌ \(job.input): reference \(w)x\(h)x\(f)f needs F=8k+1 and W,H multiples of 32"); exit(2)
        }
        if anchored {
            guard let a = io["anchor"], a.shape == [1, 3, f, h * 2, w * 2], io["anchor_keep"] != nil else {
                print("[outpaint25] FAIL ❌ \(job.input): LTX_OP_STAGE2=anchored needs `anchor` [1,3,\(f),\(h * 2),\(w * 2)] + `anchor_keep`"); exit(2)
            }
        }
        if let m = io["mask"] {
            guard (m.ndim == 2 && m.dim(0) == h && m.dim(1) == w)
                    || (m.ndim == 3 && m.dim(0) == f && m.dim(1) == h && m.dim(2) == w) else {
                print("[outpaint25] FAIL ❌ \(job.input): mask \(m.shape) must be (H,W) or (F,H,W) of the reference"); exit(2)
            }
        }
    }

    print("[outpaint25] base LTX-\(on23 ? "2.3" : "2.5") · adapter \(URL(fileURLWithPath: loraPath).lastPathComponent) @ \(loraStrength) · mask \(maskMode)"
        + " · two-stage \(twoStage ? (anchored ? "x2 anchored" : "x2 bare") : "off") · DiT \(quantName) · seed \(seed) · \(jobs.count) job(s)")
    var warm = [transformerPath ?? ltxDir.appendingPathComponent("transformer-distilled.safetensors")]
    if !baseArm { warm.append(URL(fileURLWithPath: loraPath)) }
    for f in ["connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
              "audio_vae.safetensors", "vocoder.safetensors", "spatial_upscaler_x2_v1_1.safetensors"] {
        warm.append(ltxDir.appendingPathComponent(f))
    }
    // 2.5 carries its Gemma-4 encoder in-dir; 2.3 uses the external 4-bit Gemma-3.
    let gemmaDir = on23 ? URL(fileURLWithPath: defaultGemma) : LTX2Pipeline.gemma4Dir(ltxDir: ltxDir)
    warm.append(contentsOf: ((try? FileManager.default.contentsOfDirectory(at: gemmaDir, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" })
    prewarmFiles(warm)

    let l0 = Date()
    let pipeline = try await LTX2Pipeline.load(ltxDir: ltxDir, gemmaDir: gemmaDir, transformerPath: transformerPath)
    // The DiT is built inside `load`, so the factors attach here and a mis-dialect adapter fails
    // before job 1 rather than after a stage-1 denoise.
    pipeline.reusePromptEmbeddings = true   // every job shares one prompt
    if !baseArm { try pipeline.setLoRAs([(URL(fileURLWithPath: loraPath), loraStrength)]) }
    print(String(format: "[outpaint25] loaded in %.1fs · adapter targets %d", Date().timeIntervalSince(l0),
                 pipeline.activeLoRATargets))
    guard baseArm || pipeline.activeLoRATargets > 0 else {
        print("[outpaint25] FAIL ❌ the adapter resolved ZERO targets — wrong dialect or not an LTX LoRA"); exit(1)
    }

    for (i, job) in jobs.enumerated() {
        let io = try MLX.loadArrays(url: URL(fileURLWithPath: job.input))
        let refPixels = io["reference"]!
        let f = refPixels.dim(2), h = refPixels.dim(3), w = refPixels.dim(4)
        let ref = try pipeline.encodeReference(pixels: refPixels.asType(.float32), fps: fps)
        let fLat = (f - 1) / 8 + 1
        var cross: MLXArray? = nil
        if maskMode != "none", var m = io["mask"] {
            if m.ndim == 2 { m = broadcast(m.expandedDimensions(axis: 0), to: [f, h, w]) }
            cross = ICAttentionMask.latentCrossMask(pixelMask: m, latentFrames: fLat,
                                                    latentHeight: h / 32, latentWidth: w / 32)
        }
        let sampler = PhysSampler(); sampler.start()
        let r0 = Date()
        // LTX_OP_DEBUG=t2v: plain one-stage t2v at the job's canvas in THIS process — separates a
        // load-path fault from an outpaint-path fault.
        let out = env["LTX_OP_DEBUG"] == "t2v"
            ? try await pipeline.t2v(prompt: prompt, height: h, width: w, numFrames: f, fps: fps, seed: seed)
            : try await pipeline.icOutpaint(
            prompt: prompt, reference: ref, referenceCrossMask: cross, convention: convention,
            height: h, width: w, numFrames: f, fps: fps, seed: seed, twoStage: twoStage,
            stage2Anchor: anchored ? io["anchor"].flatMap { a in io["anchor_keep"].map { (a, $0) } } : nil)
        let wall = Date().timeIntervalSince(r0)
        let peak = sampler.maxBytes(); sampler.stop()

        // (1,3,F,H,W) [-1,1] → (F,H,W,3) uint8
        let px = out.video
        let u8 = (MLX.clip((px[0].transposed(1, 2, 3, 0).asType(.float32) + 1) * 127.5 + 0.5, min: 0, max: 255))
            .asType(.uint8)
        eval(u8)
        let finite = MLX.all(MLX.isFinite(px)).item(Bool.self)
        let std = MLX.sqrt(MLX.mean(MLX.square(px.asType(.float32) - MLX.mean(px.asType(.float32))))).item(Float.self)
        try MLX.save(arrays: ["video": u8], url: URL(fileURLWithPath: "\(job.prefix).safetensors"))
        // Finished tensors nothing else touches; the encoder hops to @InferenceActor.
        nonisolated(unsafe) let previewFrames = px.transposed(0, 2, 3, 4, 1)
        nonisolated(unsafe) let previewAudio = out.audio
        let mp4 = try await encodeMP4(frames: previewFrames, fps: fps, audio: previewAudio)
        try mp4.write(to: URL(fileURLWithPath: "\(job.prefix).mp4"))
        let crossNote = cross.map { c -> String in
            let v = c.asArray(Float.self)
            return String(format: "cross: %d tokens, %d zero, %d fractional",
                          v.count, v.filter { $0 == 0 }.count, v.filter { $0 > 0 && $0 < 1 }.count)
        } ?? "no mask"
        print(String(format: "[outpaint25] job %d/%d %dx%dx%df → %@ · run %.1fs · peak %.2f GB · finite %@ std %.4f · %@",
                     i + 1, jobs.count, w, h, f, "\(u8.dim(2))x\(u8.dim(1))", wall, gbOf(peak),
                     finite ? "yes" : "NO", std, crossNote))
        print("[outpaint25]   → \(job.prefix).{safetensors,mp4}")
        fflush(stdout)
        if !finite || std < 0.01 { print("[outpaint25] FAIL ❌ degenerate output"); exit(1) }
    }
}
