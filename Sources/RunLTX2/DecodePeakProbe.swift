// DecodePeakProbe.swift — the decode phase's memory and exactness, video and audio (AB-T-0339).
//
// `--vae-decode-peak [W] [H] [F]`: the video VAE decode and the audio decode (audio VAE → vocoder)
// through the pipeline's own pool discipline (`LTX2Pipeline.withDecodePoolCap`,
// `LTX2Pipeline.decodeAudio`), in a chosen order, printing each step's phys high-water, its MLX
// active peak and the pool it leaves. `--ic-spot25` sees the decode only as one window of a whole
// run; this isolates it, so a decoder change can be measured in seconds and its output compared
// across builds (`--vae-encode-compare` reads the saved pixels too). `--vae-decode-exact-gate`
// checks the conv route against the CPU and the decode's pool invariance.
//
// Before AB-T-0339 the decode's last stage came back unevaluated and ran after the cap was lifted,
// and the audio decode ran uncapped; at 768×448×121 icT2V's order (video, then audio) read Δ35.4 GB
// and t2vTwoStage's (audio first) Δ25.1 GB — the IC arms' +11–13 GB over the base arm (AB-R-0598).
//
// The inputs are a seeded normal video latent (1, 128, F', H', W') and seeded audio tokens
// (1, T, 128). Memory is shape-dependent, not content-dependent.
//
//   env: LTX_DEC_STEPS=audio,video        comma list, run in order (default: the pipeline's order)
//        LTX_DEC_MODE=whole|tiled|chunked   whole decode, decodePixels' 2×2 halo-5 tiling, or its
//                                           8-frame halo-5 temporal chunks
//        LTX_DEC_WEIGHTS=<vae_decoder.safetensors>   (default: the 2.5 tree's stock decoder)
//        LTX_DEC_SAVE=<out.safetensors>               write {"pixels", "audio"} for cross-build comparison
//        LTX_VAE_CACHE_GB                             the decode cap lever, as in the pipeline

import Foundation
import MLX
import LTX2

func vaeDecodePeakProbe(width: Int, height: Int, frames: Int) throws {
    let env = ProcessInfo.processInfo.environment
    let tree = "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx"
    let weights = env["LTX_DEC_WEIGHTS"] ?? "\(tree)/vae_decoder.safetensors"
    let steps = (env["LTX_DEC_STEPS"] ?? "audio,video").split(separator: ",").map(String.init)
    let mode = env["LTX_DEC_MODE"] ?? "whole"
    let fLat = (frames + 7) / 8, hLat = height / 32, wLat = width / 32
    let audioT = Positions.audioTokenCount(numFrames: frames, fps: 24)
    print(String(format: "[vae-decode-peak] %dx%dx%df · latent (1,128,%d,%d,%d) · audio T=%d · steps %@ · %@ · %@",
                 width, height, frames, fLat, hLat, wLat, audioT, steps.joined(separator: ",") as NSString,
                 mode as NSString, (weights as NSString).lastPathComponent as NSString))

    let dec = try VideoVAEDecoder.load(path: URL(fileURLWithPath: weights))
    let audioVAE = try AudioVAEDecoder.load(path: URL(fileURLWithPath: "\(tree)/audio_vae.safetensors"))
    let vocoder = try Vocoder.load(path: URL(fileURLWithPath: "\(tree)/vocoder.safetensors"))
    let latent = MLXRandom.normal([1, 128, fLat, hLat, wLat], key: MLXRandom.key(42))
    let tokens = MLXRandom.normal([1, audioT, 128], key: MLXRandom.key(43))
    eval(latent, tokens)
    Memory.clearCache()

    let sampler = PhysSampler(); sampler.start()
    defer { sampler.stop() }
    sampler.resetMax()
    let floor = physFootprintBytes()
    print(String(format: "[vae-decode-peak] floor %.2f GB (weights load lazily, so the first step pays for its own)",
                 gbOf(floor)))

    var pixels: MLXArray?
    var waveform: MLXArray?
    for step in steps {
        Memory.peakMemory = 0
        let activeBefore = Memory.activeMemory
        _ = sampler.takePhaseMax()
        let t0 = Date()
        switch step {
        case "video":
            pixels = try LTX2Pipeline.withDecodePoolCap { () throws -> MLXArray in
                switch mode {
                case "tiled": return dec.decodeSpatialTiled(latent, tilesH: 2, tilesW: 2, halo: 5)
                case "chunked": return try dec.decodeChunked(latent, chunkFrames: 8, halo: 5)
                default: return dec.decode(latent)
                }
            }
            eval(pixels!)
        case "audio":
            waveform = LTX2Pipeline.decodeAudio(tokens, audioVAE: audioVAE, vocoder: vocoder)
        default:
            print("[vae-decode-peak] unknown step '\(step)'"); exit(2)
        }
        let dt = Date().timeIntervalSince(t0)
        let stepMax = sampler.takePhaseMax()
        print(String(format: "[vae-decode-peak]   %-5@ %.2fs · phys max %.2f GB (Δ floor %.2f) · MLX active peak Δ %.2f GB"
                     + " · active after %.2f GB · pool after %.2f GB",
                     step as NSString, dt, gbOf(stepMax), gbOf(stepMax > floor ? stepMax - floor : 0),
                     Double(Memory.peakMemory > activeBefore ? Memory.peakMemory - activeBefore : 0) / 1e9,
                     Double(Memory.activeMemory) / 1e9, Double(Memory.cacheMemory) / 1e9))
    }
    let peak = sampler.maxBytes()
    print(String(format: "[vae-decode-peak] peak %.2f GB · Δ floor %.2f GB · pixels %@ · audio %@",
                 gbOf(peak), gbOf(peak > floor ? peak - floor : 0),
                 pixels.map { "\($0.shape)" } ?? "—", waveform.map { "\($0.shape)" } ?? "—"))
    if let out = env["LTX_DEC_SAVE"], pixels != nil || waveform != nil {
        var arrays: [String: MLXArray] = [:]
        if let pixels { arrays["pixels"] = pixels.asType(.float32) }
        if let waveform { arrays["audio"] = waveform.asType(.float32) }
        try MLX.save(arrays: arrays, url: URL(fileURLWithPath: out))
        print("[vae-decode-peak] → \(out)")
    }
}

/// `--vae-decode-exact-gate`: the decoder's GPU conv route is exact-class and pool-invariant
/// (AB-T-0339).
///
/// 1. ROUTE. The cosine ≥ 0.999 gates cannot see Winograd's loss (relL2 ~5e-3 reads cosine
///    0.99999), and the shipping route relies on mlx's dispatch: a conv3d with non-zero depth
///    padding takes implicit GEMM, not the per-frame Winograd split mlx#3785 added. A seeded latent
///    (1,128,3,8,8) → 17×256×256 decodes on the CPU stream (the reference) and on the GPU through
///    each route; its last stage (128 → 128 at 17×64×64) sits inside the Winograd window:
///      · implicitGEMM (shipping) and exactConv (79b5814) must match the CPU to relL2 < 1e-4;
///      · stock is the poison arm and must MISS (relL2 > 1e-3), else exit 2, INCONCLUSIVE.
/// 2. POOL. The pipeline decodes under a pool cap of 0. On 79b5814's exactConv route that returned
///    different values from the same decode uncapped (relL2 1.8e-2 – 3.7e-2 on full-resolution
///    graphs, reproducible) — an mlx buffer-lifetime fault, not arithmetic. A seeded 768×448×121
///    latent decodes on the shipping route capped at 0 and uncapped; the two must be BIT-IDENTICAL.
///    There is no poison arm here: the failing graph needed exactConv's whole-stage evaluation,
///    which the decoder no longer builds.
func vaeDecodeExactGate() throws {
    let weights = ProcessInfo.processInfo.environment["LTX_DEC_WEIGHTS"]
        ?? "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx/vae_decoder.safetensors"
    var dec = try VideoVAEDecoder.load(path: URL(fileURLWithPath: weights))
    let small = MLXRandom.normal([1, 128, 3, 8, 8], key: MLXRandom.key(7))
    eval(small)
    let cpu = Device.withDefaultDevice(.cpu) { dec.decode(small) }
    func arm(_ route: VideoVAEDecoder.ConvRoute) -> Float {
        dec.convRoute = route
        return relL2(dec.decode(small), cpu)
    }
    let shipping = arm(.implicitGEMM), exactConv = arm(.exactConv), poison = arm(.stock)
    dec.convRoute = .implicitGEMM
    print(String(format: "[vae-decode-exact-gate] route · GPU vs CPU relL2 · implicitGEMM %.3e · exactConv %.3e · stock (poison) %.3e",
                 shipping, exactConv, poison))

    let big = MLXRandom.normal([1, 128, 16, 14, 24], key: MLXRandom.key(42))
    eval(big)
    func decodeCapped(_ capBytes: Int) -> MLXArray {
        let saved = Memory.cacheLimit
        Memory.cacheLimit = capBytes
        defer { Memory.cacheLimit = saved }
        let px = dec.decode(big)
        Memory.clearCache()
        return px
    }
    let capped = decodeCapped(0)
    let uncapped = decodeCapped(Memory.memoryLimit)
    let poolMax = maxAbs(capped, uncapped)
    print(String(format: "[vae-decode-exact-gate] pool  · 768×448×121 capped 0 vs uncapped maxAbs %.3e (must be 0)", poolMax))

    let routesExact = shipping < 1e-4 && exactConv < 1e-4
    let poisonCaught = poison > 1e-3
    let poolInvariant = poolMax == 0
    if routesExact && poisonCaught && poolInvariant {
        print("[vae-decode-exact-gate] PASS ✅")
    } else if !routesExact || !poolInvariant {
        print("[vae-decode-exact-gate] FAIL ❌ — "
              + (!routesExact ? "a route the decoder relies on is not exact-class on this mlx"
                              : "the shipping decode's values depend on the pool cap"))
        exit(1)
    } else {
        print("[vae-decode-exact-gate] INCONCLUSIVE — stock conv3d is exact here too, so the poison arm cannot fail")
        exit(2)
    }
}
