// DiT25QuantProbe.swift — per-forward q8-vs-bf16 probe for the LTX-2.5 DiT.
//
// WHY THIS IS ONE-ARM-PER-PROCESS. 2.3's `--dit-q8-gate` spanned 0.996388–0.999915 over five
// runs (parent CLAUDE.md), and that spread appeared ACROSS PROCESS INVOCATIONS — in-process the
// int8 path self-repeats bit-exactly (the HV2 streaming acceptance memcmp depends on it). So a
// gate that loads both arms in one process and prints one cosine would report a spread of exactly
// zero and prove nothing about the thing that actually varies. This writes ONE arm's forward to
// disk; the driver runs it N times per arm and `--dit25-probe-compare` reports the SPREAD.
//
// It also keeps only ONE DiT resident (bf16 38 GB / q8 20.6 GB), so the probe never needs 58 GB.
//
// WHAT IS COMPARED. Not Swift-vs-oracle parity (there is no 2.5 DiT golden, and parity is not the
// open question) but **q8-vs-bf16 on identical inputs** — the number the quant-ladder doctrine
// quotes ("q8 ≈ bf16, single-forward cosine ~0.9999") and the number a lower-tier declaration
// rests on. bf16-vs-bf16 across processes is the must-fail control: on 2.3 that arm was exactly
// deterministic, so if it moves here the instrument, not the quant, is what is being measured.
//
// INPUT CONSTRUCTION (in-distribution, and the inputs are SAVED so the comparator can prove both
// arms saw the same ones bit-for-bit rather than assume it):
//   - text embeds: the real 2.5 Gemma-4 49-state golden → the real 2.5 connector. Not synthetic.
//   - positions:   `Positions.video/audio` at the shipping envelope's stage-2 grid.
//   - keyframes:   `firstLatentFrameKeyframesMask`, exactly as `t2vTwoStage` builds it for 2.5.
//   - latent:      unit-variance noise from a fixed key at the stage-2 entry sigma. Stage 2's real
//                  init is `noise·σ₀ + upscaled·(1−σ₀)`, which needs a full stage-1 run to
//                  reproduce; std ≈ 0.914 there vs 1.0 here, so this is the right MAGNITUDE and
//                  the right SHAPE at the right token count, not a byte-exact stage-2 init.

import Foundation
import LTX2
import MLX
import MLXRandom

private let probeBase = "/Volumes/Satechi/Models/xocialize"
private let probeBase23 = "/Volumes/Satechi/Models/dgrauet"

/// Shipping envelope, stage-2 grid: 704×512×121f → F=16, H=16, W=22 → 5632 video tokens
/// (the N=5632 already quoted in the attention receipts).
private struct ProbeGeometry {
    var width = 704, height = 512, frames = 121
    let fps: Double = 24
    var fLat: Int { (frames - 1) / 8 + 1 }      // 16
    var hLat: Int { height / 32 }               // 16
    var wLat: Int { width / 32 }                // 22
    var videoTokens: Int { fLat * hLat * wLat } // 5632
    var audioTokens: Int { Positions.audioTokenCount(numFrames: frames, fps: fps) }
    var sigma: Float { Positions.stage2Sigmas[0] }  // 0.909375
}

/// Run ONE arm's DiT forward and write inputs + outputs to `outPath`.
///
/// The `*-23` arms run the IDENTICAL measurement on LTX-2.3, whose q8 checkpoint carries the very
/// recipe the 2.5 one replicated (transformer_blocks-only int8 g64). Without them, "2.5 q8 vs bf16
/// = 0.9982" has no comparison point on this metric — the 0.9999/0.9991 figures in the quant-ladder
/// doctrine were not measured this way, so reading 0.9982 against them would be comparing
/// instruments. Each version gets its OWN in-distribution text conditioning (2.5: Gemma-4 golden →
/// 2.5 connector; 2.3: the `text_encode` golden's connector outputs), which is the point — the
/// quantity compared across versions is each one's quant delta, not a shared absolute.
func dit25QuantProbe(arm: String, outPath: String, width: Int? = nil, height: Int? = nil,
                     frames: Int? = nil) throws {
    let treeName: String, base: String, is25: Bool
    switch arm {
    case "bf16":        (treeName, base, is25) = ("ltx-2.5-mlx", probeBase, true)
    case "q8", "ditq8": (treeName, base, is25) = ("ltx-2.5-mlx-ditq8", probeBase, true)
    case "bf16-23":     (treeName, base, is25) = ("ltx-2.3-mlx", probeBase23, false)
    case "q8-23":       (treeName, base, is25) = ("ltx-2.3-mlx-q8", probeBase23, false)
    default:
        print("[dit25-probe] FAIL ❌ unknown arm '\(arm)' (expected bf16 | ditq8 | bf16-23 | q8-23)")
        exit(2)
    }
    let tree = URL(fileURLWithPath: "\(base)/\(treeName)")
    let ditPath = tree.appendingPathComponent("transformer-distilled.safetensors")
    guard FileManager.default.fileExists(atPath: ditPath.path) else {
        print("[dit25-probe] FAIL ❌ missing \(ditPath.path)"); exit(2)
    }
    // Resolve the version from the CHECKPOINT — a path-name parse would happily probe the wrong one,
    // and a 2.3 tree silently answering for a 2.5 arm is the failure this guard exists for.
    guard LTX2Pipeline.isLTX25(ltxDir: tree) == is25 else {
        print("[dit25-probe] FAIL ❌ arm '\(arm)' expects isLTX25=\(is25) but \(tree.path) reads the "
              + "opposite — the run would prove nothing")
        exit(2)
    }

    var g = ProbeGeometry()
    if let width { g.width = width }
    if let height { g.height = height }
    if let frames { g.frames = frames }
    print("[dit25-probe] arm=\(arm) tree=\(treeName) isLTX25=\(is25)")
    print("[dit25-probe] geom=\(g.width)x\(g.height)x\(g.frames)f → latent \(g.fLat)x\(g.hLat)x\(g.wLat)"
          + "  videoTokens=\(g.videoTokens) audioTokens=\(g.audioTokens) sigma=\(g.sigma)")

    // --- text embeds: each version's own in-distribution conditioning ---
    let videoText: MLXArray, audioText: MLXArray
    if is25 {
        // real Gemma-4 49 states → the real 2.5 connector
        let goldens = try MLX.loadArrays(url: URL(fileURLWithPath: "\(goldensBase)/gemma4/goldens.safetensors"))
        guard let mask = goldens["attention_mask"] else {
            print("[dit25-probe] FAIL ❌ gemma4 goldens missing attention_mask"); exit(2)
        }
        var hidden: [MLXArray] = []
        for i in 0 ..< 49 {
            guard let h = goldens[String(format: "gemma_hidden_%02d", i)] else {
                print("[dit25-probe] FAIL ❌ gemma4 goldens missing state \(i)"); exit(2)
            }
            hidden.append(h)
        }
        let connector = try Connector.load(connectorPath: tree.appendingPathComponent("connector.safetensors"))
        (videoText, audioText) = connector(hiddenStates: hidden, mask: mask)
    } else {
        // 2.3: the text_encode golden already stores the CONNECTOR outputs, so no connector run.
        let te = try MLX.loadArrays(url: URL(fileURLWithPath: "\(goldensBase)/text_encode/goldens.safetensors"))
        guard let v = te["video_embeds"], let a = te["audio_embeds"] else {
            print("[dit25-probe] FAIL ❌ text_encode goldens missing video_embeds / audio_embeds"); exit(2)
        }
        (videoText, audioText) = (v, a)
    }
    eval(videoText, audioText)
    print("[dit25-probe] text embeds video \(videoText.shape) audio \(audioText.shape)")

    // --- latent: fixed key, so every run and every arm gets the identical array ---
    MLXRandom.seed(0x2025_0814)
    let videoLatent = MLXRandom.normal([1, g.videoTokens, 128]).asType(.float32)
    let audioLatent = MLXRandom.normal([1, g.audioTokens, 128]).asType(.float32)
    eval(videoLatent, audioLatent)

    let videoPositions = Positions.video(F: g.fLat, H: g.hLat, W: g.wLat, fps: Float(g.fps))
    let audioPositions = Positions.audio(tokens: g.audioTokens)
    // 2.3 checkpoints carry no `keyframes_abs_pos_embedding`, so `DiT` ignores the mask there;
    // pass nil rather than relying on that, so the 2.3 arm is explicitly the 2.3 forward.
    let kfMask = is25
        ? LTX2Pipeline.firstLatentFrameKeyframesMask(
            totalTokens: g.videoTokens, tokensPerLatentFrame: g.hLat * g.wLat)
        : nil
    let sigma = MLXArray([g.sigma])

    print("[dit25-probe] loading DiT…")
    let t0 = Date()
    let dit = try DiT.load(weightsPath: ditPath, config: DiTConfig(), computeDtype: .bfloat16)
    print(String(format: "[dit25-probe] load %.1fs", Date().timeIntervalSince(t0)))

    let f0 = Date()
    let (video, audio) = dit(
        videoLatent: videoLatent, audioLatent: audioLatent, sigma: sigma,
        videoText: videoText, audioText: audioText,
        videoPositions: videoPositions, audioPositions: audioPositions,
        keyframesMask: kfMask)
    eval(video, audio!)
    print(String(format: "[dit25-probe] forward %.1fs", Date().timeIntervalSince(f0)))

    // Save the INPUTS alongside the outputs: the comparator asserts bit-identity across arms, so
    // "both arms saw the same input" is a checked control rather than an assumption about RNG.
    var out: [String: MLXArray] = [
        "video_v": video.asType(.float32),
        "audio_v": audio!.asType(.float32),
        "in_video_latent": videoLatent,
        "in_audio_latent": audioLatent,
        "in_video_text": videoText.asType(.float32),
        "in_audio_text": audioText.asType(.float32),
        "in_sigma": sigma,
    ]
    // Absent on the 2.3 arms; the comparator only requires the inputs that are present in BOTH
    // files to match, so a 2.5-vs-2.3 comparison is refused rather than silently half-checked.
    if let kfMask { out["in_kf_mask"] = kfMask.asType(.float32) }
    try MLX.save(arrays: out, url: URL(fileURLWithPath: outPath))
    print("[dit25-probe] wrote \(outPath)")
    print(String(format: "[dit25-probe] SUMMARY arm=%@ videoTokens=%d audioTokens=%d sigma=%.6f out=%@",
                 arm as NSString, g.videoTokens, g.audioTokens, g.sigma, outPath as NSString))
}

/// Compare two probe outputs: inputs must be bit-identical, then report the forward cosine.
func dit25ProbeCompare(_ aPath: String, _ bPath: String) throws {
    let a = try MLX.loadArrays(url: URL(fileURLWithPath: aPath))
    let b = try MLX.loadArrays(url: URL(fileURLWithPath: bPath))

    guard aPath != bPath else {
        // A file compared with itself reads bit-exact no matter what — a verdict that cannot fail
        // is not a verdict. (The first driver produced exactly this: BSD `seq 4 3` counts DOWN.)
        print("[dit25-cmp] FAIL ❌ refusing to compare \(aPath) with itself"); exit(2)
    }
    // Control first: if the inputs differ, the cosine below is measuring the inputs, not the arms.
    var inputsIdentical = true
    for k in ["in_video_latent", "in_audio_latent", "in_video_text", "in_audio_text",
              "in_sigma", "in_kf_mask"] {
        switch (a[k], b[k]) {
        case (nil, nil):
            continue                        // absent from both arms (in_kf_mask on the 2.3 pair)
        case (nil, _), (_, nil):
            // Present in one file only ⇒ the two arms did not run the same forward at all
            // (a 2.5-vs-2.3 pairing). Refuse rather than compare five of six inputs and call it.
            print("[dit25-cmp] FAIL ❌ input \(k) present in only one file — these arms are not comparable")
            exit(2)
        case let (x?, y?):
            let diff = MLX.max(MLX.abs(x.asType(.float32) - y.asType(.float32))).item(Float.self)
            if diff != 0 {
                inputsIdentical = false
                print(String(format: "[dit25-cmp] ⚠️ input %@ differs, maxAbs=%.3e", k as NSString, diff))
            }
        }
    }
    print(inputsIdentical
          ? "[dit25-cmp] inputs bit-identical ✅ (the cosine below is the arms, not the inputs)"
          : "[dit25-cmp] inputs DIFFER ❌ — the comparison below is void")

    let vCos = cosine(a["video_v"]!, b["video_v"]!), vMax = maxAbs(a["video_v"]!, b["video_v"]!)
    let aCos = cosine(a["audio_v"]!, b["audio_v"]!), aMax = maxAbs(a["audio_v"]!, b["audio_v"]!)
    let bitExact = vMax == 0 && aMax == 0
    print(String(format: "[dit25-cmp] VIDEO cosine=%.8f maxAbs=%.6f   AUDIO cosine=%.8f maxAbs=%.6f",
                 vCos, vMax, aCos, aMax))
    print(String(format: "[dit25-cmp] RESULT %@ vs %@ videoCos=%.8f audioCos=%.8f videoMaxAbs=%.6f audioMaxAbs=%.6f bitExact=%@",
                 (aPath as NSString).lastPathComponent as NSString,
                 (bPath as NSString).lastPathComponent as NSString,
                 vCos, aCos, vMax, aMax, (bitExact ? "yes" : "no") as NSString))
}
