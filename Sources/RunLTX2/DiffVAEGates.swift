// DiffVAEGates.swift — parity gates for the two LTX-2.5 components that bench-matrix
// arms C and D were blocked on: the DiffVAE decoder and the duration head.
//
//   --diffvae-gate    claim C3 / arm C (decode triangle)
//   --duration-gate   claim C6 / arm D (duration-head honesty check)
//
// Goldens: `parity/dump_diffvae_goldens.py` and `parity/dump_duration_goldens.py`, both
// dumped from the Python-MLX oracle these ports were written against. Each fixture also
// carries the upstream TORCH tensors where the oracle's reference bank had them; those are
// REPORTED, not gated — the oracle already characterized that residual (fp32 accumulation,
// not a math gap) and its verdicts on C3/C6 are settled. These gates exist to prove the
// Swift port reproduces the oracle, not to re-open either claim.
//
// Threshold: per-component cosine >= 0.999, the board-wide bar.

import Foundation
import MLX
import LTX2

private let diffvaeWeights = "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx/vae_diffusion_decoder.safetensors"
private let durationWeights = "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx/duration_head.safetensors"

// MARK: - DiffVAE

/// Six-stage DiffVAE parity gate, in dependency order so a failure localizes.
///
/// ⚠️ THE NOISE CANVAS IS INJECTED, NEVER DRAWN. DiffVAE is a stochastic 1-step-x0 decoder
/// that starts from pure noise, and MLX-Swift and MLX-Python RNG streams are not
/// bit-identical (determinism doctrine, ISSUES I1) — a seeded gate would measure the RNG,
/// not the port. The fixture's own canvas is fed to `decode`, the same way
/// `--ancestral-step-gate` handles the sampler's noise. LTX25-PORT-PLAN §V C3 is explicit
/// that this decoder must never be gated on pixel equality from a fresh draw.
///
/// Stage [G] closes the loophole that creates: it re-decodes the same latent from the
/// fixture's SECOND canvas and requires the pixels to MOVE. Without it, a port that ignored
/// the canvas entirely — or a gate whose injection silently failed — would sail through the
/// fixed-noise comparison.
func diffVAEGate() throws {
    let dir = "\(goldensBase)/diffvae"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let url = URL(fileURLWithPath: diffvaeWeights)
    prewarmFiles([url])
    let dec = try DiffusionVideoDecoder.load(path: url)
    print("[diffvae-gate] weights: \(diffvaeWeights)")
    print("[diffvae-gate] backend: \(NA3D.backend.rawValue)  (LTX2_NA_IMPL)"); fflush(stdout)

    var ok = true
    /// Report a stage against the oracle golden, and against torch when the fixture has it.
    func check(_ tag: String, _ got: MLXArray, _ refKey: String) {
        guard let ref = io[refKey] else {
            print("[diffvae-gate] \(tag) MISSING golden '\(refKey)'"); ok = false; return
        }
        eval(got)
        let c = cosine(got, ref), m = maxAbs(got, ref)
        let pass = c >= 0.999
        ok = ok && pass
        var line = String(format: "[diffvae-gate] %@ cosine=%.6f maxAbs=%.2e %@",
                          tag as NSString, c, m, (pass ? "PASS" : "FAIL") as NSString)
        if let t = io["torch_\(refKey)"] {
            line += String(format: "   | vs torch cosine=%.6f maxAbs=%.2e", cosine(got, t), maxAbs(got, t))
        }
        print(line); fflush(stdout)
    }

    let cfg = DiffusionVideoDecoder.Config()
    let tile = (1, 8, 96)   // matches the fixture's oracle-side query tiling

    // [A] one deterministic NABlock — det_stages.3.0, dim 512, kernel (3,5,5)
    let xa = io["nablock_in"]!
    check("[A] NABlock      ", dec.naBlock(xa, "det_stages.3.0", kernel: (3, 5, 5), tile: tile),
          "nablock_out")

    // [B] one upsample — upsamples.3, stride (2,2,2), 512 -> 256, leading-frame drop
    check("[B] upsample     ",
          dec.upsample(xa, "upsamples.3", stride: (2, 2, 2), reduction: 2, dropLeadingFrame: true),
          "upsample_out")

    // [C]/[D] stage 5. `conv_in_x_t` and the AdaLN split are shared by both.
    let ctx = io["diffblock_ctx"]!
    let xt = io["diffblock_xt"]!
    let x0 = dec.stage5Input(xt)
    let shared = dec.sharedAdaLN(MLXArray([Float(1.0)]))

    check("[C] diff block   ",
          dec.diffusionNABlock(x0, "diff_blocks.0", context: ctx, shared: shared,
                               kernel: cfg.stage5Kernel, tile: tile),
          "diffblock_out")

    check("[D] stage-5 stack", dec.diffStep(context: ctx, xT: xt, t: MLXArray([Float(1.0)]), tile: tile),
          "stage5_out")

    // [E] the deterministic stage-1-to-4 ladder, with its ghost pad and ghost crop.
    let latent = io["decode_latent"]!
    let context = dec.decodeContext(latent, tile: tile)
    check("[E] ladder 1-4   ", context, "decode_context")

    // [F] the full end-to-end decode, fed the fixture's canvas.
    let noise = io["decode_noise"]!
    let pixels = dec.decode(latent, noise: noise, tile: tile)
    check("[F] full decode  ", pixels, "decode_pixels")

    // [G] canvas sensitivity — the decoder must actually be noise-driven.
    if let alt = io["decode_noise_alt"], let altRef = io["decode_pixels_alt"] {
        let pixels2 = dec.decode(latent, noise: alt, tile: tile)
        eval(pixels2)
        let moved = maxAbs(pixels2, pixels)
        let oracleMoved = maxAbs(altRef, io["decode_pixels"]!)
        // Loose bar on purpose: the point is "the canvas is load-bearing", not a second
        // parity check (that is [F] again, which is reported alongside).
        let sensitive = moved > 0.1 * oracleMoved
        ok = ok && sensitive
        print(String(format: "[diffvae-gate] [G] canvas drive  ours maxAbs=%.4f  oracle=%.4f  %@",
                     moved, oracleMoved, (sensitive ? "PASS" : "FAIL — canvas ignored") as NSString))
        print(String(format: "[diffvae-gate] [G] alt-canvas parity cosine=%.6f", cosine(pixels2, altRef)))
    }

    print(ok ? "[diffvae-gate] PASS ✅" : "[diffvae-gate] FAIL ❌")
    if !ok { exit(1) }
}

// MARK: - Duration head

/// Duration-head parity gate: predicted seconds AND the frame count that actually ships.
///
/// Two thresholds, deliberately different:
///
///  * SECONDS get a 1e-3 relative tolerance. The pooler softmaxes one query over ~2048 keys
///    and the pooled vector sits ~20× below the token scale, so 1e-6 differences in the
///    input projections amplify to ~2e-4 on the output. That is the shape, not the port.
///  * FRAMES must be EXACTLY equal. The frame count is the deliverable the pipeline
///    consumes, and at 24 fps one grid step is 8/24 = 0.333 s — four orders of magnitude
///    above the residual, so it is insensitive to it by construction. If this ever
///    disagrees, the cause is structural, not numeric.
///
/// All three input modes are exercised because the concatenation order (video first) is a
/// silent-failure axis: a swapped order still produces a plausible number.
func durationGate() throws {
    let dir = "\(goldensBase)/duration"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let url = URL(fileURLWithPath: durationWeights)
    prewarmFiles([url])
    let head = try DurationHead.load(path: url)
    print("[duration-gate] weights: \(durationWeights)")

    let v = io["video_tokens"]!
    let a = io["audio_tokens"]!
    print("[duration-gate] video tokens \(v.shape)  audio tokens \(a.shape)"); fflush(stdout)

    let cases: [(String, MLXArray?, MLXArray?)] = [
        ("both", v, a), ("video_only", v, nil), ("audio_only", nil, a),
    ]
    let expectedFrames = io["frames_24_30"]!.asType(.float32)

    var ok = true
    for (i, (name, vt, at)) in cases.enumerated() {
        let y = head(videoTokens: vt, audioTokens: at)
        eval(y)
        let ours = Double(y[0].item(Float.self))
        let ref = Double(io["seconds_\(name)"]![0].item(Float.self))
        let rel = abs(ours - ref) / abs(ref)
        let secondsPass = rel < 1e-3

        var line = String(format: "[duration-gate] %-11@ ours %9.6fs  oracle %9.6fs  rel %.2e  %@",
                          name as NSString, ours, ref, rel,
                          (secondsPass ? "PASS" : "FAIL") as NSString)
        if let t = io["torch_seconds_\(name)"] {
            let tr = Double(t[0].item(Float.self))
            line += String(format: "   | torch %9.6fs rel %.2e", tr, abs(ours - tr) / abs(tr))
        }
        print(line)

        // The deliverable: identical frame count after clamp + causal-grid snap.
        var framesPass = true
        for (j, fps) in [24.0, 30.0].enumerated() {
            let oursF = DurationHead.secondsToNumFrames(ours, frameRate: fps)
            let refF = Int(expectedFrames[i, j].item(Float.self))
            if oursF != refF {
                framesPass = false
                print("[duration-gate]   FRAME MISMATCH \(name) @\(Int(fps))fps: ours \(oursF) oracle \(refF)")
            }
            if (oursF - 1) % 8 != 0 {
                framesPass = false
                print("[duration-gate]   OFF-GRID \(name) @\(Int(fps))fps: \(oursF) frames, (f-1)%8 != 0")
            }
        }
        if framesPass {
            let f24 = DurationHead.secondsToNumFrames(ours, frameRate: 24.0)
            let f30 = DurationHead.secondsToNumFrames(ours, frameRate: 30.0)
            print("[duration-gate]   snaps to \(f24) frames @24fps / \(f30) @30fps — MATCH")
        }
        ok = ok && secondsPass && framesPass
        fflush(stdout)
    }

    // Clamp bounds: the head's raw output is unbounded, so the clamp is what keeps a wild
    // prediction on the grid. Pinned at both ends against the oracle's own snap.
    let probeS = io["clamp_probe_seconds"]!.asType(.float32)
    let probeF = io["clamp_probe_frames"]!.asType(.float32)
    for i in 0 ..< probeS.dim(0) {
        let s = Double(probeS[i].item(Float.self))
        let ours = DurationHead.secondsToNumFrames(s, frameRate: 24.0)
        let ref = Int(probeF[i].item(Float.self))
        let pass = ours == ref && (ours - 1) % 8 == 0 && ours >= 9 && ours <= 481
        ok = ok && pass
        print("[duration-gate] clamp \(s)s -> \(ours) frames (oracle \(ref)) \(pass ? "PASS" : "FAIL")")
    }

    print(ok ? "[duration-gate] PASS ✅" : "[duration-gate] FAIL ❌")
    if !ok { exit(1) }
}
