// DecodeTriangle.swift — bench matrix arm C (`--decode-triangle`), claim C3.
//
// C3: the diffusion video decoder gives "sharper faces, textures, on-screen text; better motion;
// fewer artifacts". `LTX25-PORT-PLAN.md` §V calls this **the strongest objective test of the seven**
// because it is cleanly DECODER-ISOLATED, and the isolation is the whole design here:
//
//   real clip → shared VAE ENCODE (once) → the SAME latents → conv decode  ┐
//                                                            → DiffVAE decode ┘ → PSNR/SSIM vs the
//                                                                                 REAL SOURCE
//
// Both decoders see byte-identical latents, so every difference is the decoder. Measuring against
// the real source (not against each other) is what makes "faithful" meaningful — two decoders can
// disagree with each other while one is simply right.
//
// ⚠️ **Ground truth must be a REAL clip, never a generation.** PSNR against a generated video would
// measure the generator's agreement with itself, not decode fidelity. Corpus is read IN PLACE off
// /Volumes/Satechi — the old "copy clips LOCAL first, AVFoundation can't read off Satechi" claim was
// measured FALSE and withdrawn (AB-L-0041).
//
// ⚠️ **DiffVAE IS STOCHASTIC** (1-step x0 from a pure-noise canvas). §V is explicit: fix the noise
// seed and **never gate on pixel equality**. This harness passes an explicit fixed-key canvas so a
// re-run reproduces, and reports a second draw's metric spread so the stochasticity is VISIBLE
// rather than assumed small.
//
// ⚠️ **The oracle already measured C3** (AB-R-0013/0014): conv is MORE faithful (42.6 vs 36.8 dB
// PSNR, 0.994 vs 0.986 SSIM) and on-screen text is IDENTICAL under Vision OCR. That does not refute
// the claim — PSNR ≠ perceptual, and a generative decoder may legitimately trade fidelity for
// apparent sharpness. This harness reproduces the FIDELITY + COST half on Swift; the perceptual
// half is an operator A/B, not a metric.
//
// usage: RunLTX2 --decode-triangle [clip.mp4] [W H F]

import Foundation
import LTX2
import MLX
import MLXLTX2
import MLXRandom

private let triangleTree = "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx"
private let defaultClip = "/Volumes/Satechi/Models/_ForgeSmokeCorpus/video/hard-cut-14s.mp4"

/// Peak-signal-to-noise ratio over [-1,1] pixel tensors (dynamic range 2.0).
private func psnr(_ a: MLXArray, _ b: MLXArray) -> Double {
    let mse = MLX.mean(MLX.square(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    guard mse > 0 else { return .infinity }
    return 10.0 * log10(4.0 / Double(mse))   // MAX_I = 2.0 (range [-1,1]) ⇒ MAX_I² = 4
}

/// Global SSIM with the standard C1/C2 stabilisers, computed per frame on luma and averaged.
/// Deliberately GLOBAL (not windowed): the reference figures this reproduces (AB-R-0013, and the
/// pruna-face-ab pass) are whole-frame numbers, and a windowed SSIM would not be comparable to them.
private func ssimLuma(_ a: MLXArray, _ b: MLXArray) -> Double {
    // (1,3,F,H,W) in [-1,1] → luma in [0,1]
    func luma(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let r = f[0..., 0], g = f[0..., 1], bl = f[0..., 2]
        return ((0.299 * r + 0.587 * g + 0.114 * bl) + 1.0) / 2.0     // (1,F,H,W)
    }
    let la = luma(a), lb = luma(b)
    let n = la.dim(1)
    var total = 0.0
    let c1 = 0.01 * 0.01, c2 = 0.03 * 0.03
    for i in 0 ..< n {
        let x = la[0, i].asType(.float32), y = lb[0, i].asType(.float32)
        let mx = MLX.mean(x).item(Float.self), my = MLX.mean(y).item(Float.self)
        let vx = MLX.mean(MLX.square(x - mx)).item(Float.self)
        let vy = MLX.mean(MLX.square(y - my)).item(Float.self)
        let cov = MLX.mean((x - mx) * (y - my)).item(Float.self)
        let num = (2 * Double(mx) * Double(my) + c1) * (2 * Double(cov) + c2)
        let den = (Double(mx) * Double(mx) + Double(my) * Double(my) + c1) * (Double(vx) + Double(vy) + c2)
        total += num / den
    }
    return total / Double(n)
}

func decodeTriangle(clip: String?, width: Int?, height: Int?, frames: Int?) async throws {
    let tree = URL(fileURLWithPath: triangleTree)
    let clipURL = URL(fileURLWithPath: clip ?? defaultClip)
    let w = width ?? 704, h = height ?? 512, f = frames ?? 25
    guard FileManager.default.fileExists(atPath: clipURL.path) else {
        print("[decode-triangle] FAIL ❌ missing clip \(clipURL.path)"); exit(2)
    }
    let convPath = tree.appendingPathComponent("vae_decoder.safetensors")
    let diffPath = tree.appendingPathComponent("vae_diffusion_decoder.safetensors")
    let encPath  = tree.appendingPathComponent("vae_encoder.safetensors")
    for p in [convPath, diffPath, encPath] where !FileManager.default.fileExists(atPath: p.path) {
        print("[decode-triangle] FAIL ❌ missing \(p.path)"); exit(2)
    }
    // Echo RESOLVED parameters — a harness that silently uses defaults has burned this project.
    print("[decode-triangle] clip=\(clipURL.lastPathComponent)  geom=\(w)x\(h)x\(f)f  tree=\(tree.lastPathComponent)")
    print("[decode-triangle] ground truth = REAL clip frames (read IN PLACE off Satechi; AB-L-0041)")

    // --- Ground truth: real frames ---
    let t0 = Date()
    let source = try await VideoInput.referenceClipFrames(url: clipURL, width: w, height: h,
                                                         frames: f, fps: 24)
    eval(source)
    print(String(format: "[decode-triangle] source frames %@  (%.1fs)", "\(source.shape)" as NSString,
                 Date().timeIntervalSince(t0)))

    // --- Shared encode: ONE latent, both decoders. This is the isolation. ---
    let encoder = try VideoVAEEncoder.load(path: encPath)
    let e0 = Date()
    let latent = encoder.normalizeLatent(encoder.encode(source.asType(.float32)))
    eval(latent)
    print(String(format: "[decode-triangle] shared latent %@  (encode %.1fs)",
                 "\(latent.shape)" as NSString, Date().timeIntervalSince(e0)))

    struct Arm { let name: String; let pixels: MLXArray; let secs: Double; let peakGB: Double }
    var arms: [Arm] = []

    func measure(_ name: String, _ body: () -> MLXArray) -> Arm {
        let sampler = PhysSampler(); sampler.start(); sampler.resetMax()
        let s = Date()
        let px = body()
        eval(px)
        let secs = Date().timeIntervalSince(s)
        let peak = sampler.maxBytes(); sampler.stop()
        return Arm(name: name, pixels: px, secs: secs, peakGB: Double(peak) / 1e9)
    }

    // 🚨 **PEAK IS NOT PER-DECODER IN A SHARED PROCESS — MEASURED, NOT ASSUMED.**
    // `PhysSampler` reads PROCESS phys, so whichever decoder runs SECOND also carries the first
    // one's resident weights and the accumulated MLX pool. The reversal control below proved this
    // is the whole signal: conv-first reported "DiffVAE +26.43 GB", diffvae-first reported
    // "−2.56 GB", because the SECOND arm read ~41 GB either way. **Any per-decoder peak from a
    // combined run is void.** Use `--decode-triangle-arm <conv|diffvae>` (ONE decoder per process)
    // for memory; this combined mode is for TIME and FIDELITY, which are order-robust (fidelity is
    // bit-identical across orders; DiffVAE time varied 20.48→19.70 s).
    // Same family as AB-R-0041 (a post-run "resident floor" that measured mmap retention) and
    // AB-L-0041 (an A/B where argument order was the entire effect).
    let reversed = ProcessInfo.processInfo.environment["LTX_TRIANGLE_REVERSE"] == "1"
    if reversed { print("[decode-triangle] ⚠️ REVERSED decoder order (diffvae first)") }
    func runConv() throws {
        let conv = try VideoVAEDecoder.load(path: convPath)
        arms.append(measure("conv") { conv.decode(encoder.denormalizeLatent(latent)) })
    }
    if !reversed { try runConv() }
    // --- DiffVAE, fixed canvas (reproducible) + a SECOND draw to expose the stochasticity ---
    var diffSecond: MLXArray? = nil
    do {
        let diff = try DiffusionVideoDecoder.load(path: diffPath)
        let den = encoder.denormalizeLatent(latent)
        arms.append(measure("diffvae") { diff.decode(den, key: MLXRandom.key(0x0C3_5EED)) })
        // Second draw: DIFFERENT key on the SAME latent. §V says never gate pixel equality here —
        // so quantify how much the metric itself moves between draws instead of assuming it is small.
        let d2 = diff.decode(den, key: MLXRandom.key(0x0C3_5EE2))
        eval(d2); diffSecond = d2
    }
    if reversed { try runConv() }

    print("\n[decode-triangle] ───── cost: TIME (peak is void here — see below) ─────")
    for a in arms {
        print(String(format: "[decode-triangle] %-8@ decode %7.2fs   (process peak %6.2f GB — NOT per-decoder)",
                     a.name as NSString, a.secs, a.peakGB))
    }
    if let conv = arms.first(where: { $0.name == "conv" }),
       let dv = arms.first(where: { $0.name == "diffvae" }) {
        print(String(format: "[decode-triangle] DiffVAE costs ×%.2f TIME vs conv", dv.secs / conv.secs))
    }
    print("[decode-triangle] 🚨 the process-peak column above is ORDER-DEPENDENT and must not be")
    print("[decode-triangle]    quoted as a per-decoder cost — the second arm carries the first's")
    print("[decode-triangle]    weights. Run `--decode-triangle-arm conv|diffvae` for memory.")

    print("\n[decode-triangle] ───── fidelity vs the REAL source ─────")
    for a in arms {
        let p = psnr(source, a.pixels), s = ssimLuma(source, a.pixels)
        print(String(format: "[decode-triangle] %-8@ PSNR %6.2f dB   SSIM %.4f", a.name as NSString, p, s))
    }
    if let dv = arms.first(where: { $0.name == "diffvae" }), let d2 = diffSecond {
        let p1 = psnr(source, dv.pixels), p2 = psnr(source, d2)
        let s1 = ssimLuma(source, dv.pixels), s2 = ssimLuma(source, d2)
        print(String(format: "[decode-triangle] diffvae 2nd draw: PSNR %6.2f dB (Δ%+.2f)  SSIM %.4f (Δ%+.4f)",
                     p2, p2 - p1, s2, s2 - s1))
        print("[decode-triangle] ↑ the spread between draws is the floor on any DiffVAE fidelity claim")
    }

    print("\n[decode-triangle] ⚠️ C3's verdict is the ORACLE's (AB-R-0013/0014): conv is MORE faithful")
    print("[decode-triangle]    (42.6 vs 36.8 dB, 0.994 vs 0.986 SSIM), on-screen text IDENTICAL under")
    print("[decode-triangle]    Vision OCR. PSNR ≠ perceptual — a generative decoder may trade fidelity")
    print("[decode-triangle]    for apparent sharpness. This reproduces FIDELITY+COST on Swift only;")
    print("[decode-triangle]    the perceptual half is an operator A/B, not a metric.")
}


/// ONE decoder per process — the only way a per-decoder peak is attributable.
/// (The combined `--decode-triangle` mode reports order-dependent process peaks; see its header.)
func decodeTriangleArm(_ which: String, clip: String?, width: Int?, height: Int?,
                       frames: Int?) async throws {
    let tree = URL(fileURLWithPath: triangleTree)
    let clipURL = URL(fileURLWithPath: clip ?? defaultClip)
    let w = width ?? 704, h = height ?? 512, f = frames ?? 25
    guard ["conv", "diffvae"].contains(which) else {
        print("[triangle-arm] FAIL ❌ arm must be conv|diffvae, got '\(which)'"); exit(2)
    }
    print("[triangle-arm] arm=\(which) clip=\(clipURL.lastPathComponent) geom=\(w)x\(h)x\(f)f")
    let baseline = physFootprintBytes()
    let source = try await VideoInput.referenceClipFrames(url: clipURL, width: w, height: h,
                                                          frames: f, fps: 24)
    let encoder = try VideoVAEEncoder.load(path: tree.appendingPathComponent("vae_encoder.safetensors"))
    let latent = encoder.denormalizeLatent(
        encoder.normalizeLatent(encoder.encode(source.asType(.float32))))
    eval(latent)
    let preDecode = physFootprintBytes()
    print(String(format: "[triangle-arm] phys baseline %.2f GB → pre-decode (source+encoder+latent) %.2f GB",
                 Double(baseline) / 1e9, Double(preDecode) / 1e9))

    let sampler = PhysSampler(); sampler.start(); sampler.resetMax()
    let t = Date()
    let px: MLXArray
    if which == "conv" {
        let d = try VideoVAEDecoder.load(path: tree.appendingPathComponent("vae_decoder.safetensors"))
        px = d.decode(latent)
    } else {
        let d = try DiffusionVideoDecoder.load(
            path: tree.appendingPathComponent("vae_diffusion_decoder.safetensors"))
        px = d.decode(latent, key: MLXRandom.key(0x0C3_5EED))
    }
    eval(px)
    let secs = Date().timeIntervalSince(t)
    let peak = sampler.maxBytes(); sampler.stop()
    print(String(format: "[triangle-arm] SUMMARY arm=%@ decode=%.2fs peak=%.2fGB attributable=%.2fGB",
                 which as NSString, secs, Double(peak) / 1e9,
                 Double(peak &- preDecode) / 1e9))
    print("[triangle-arm] 'attributable' = peak − pre-decode phys, i.e. what THIS decoder added.")
}
