// EncodePeakProbe.swift — the video VAE ENCODE's memory and exactness (AB-T-0338).
//
// `--vae-encode-peak [W] [H] [F]`: the encode alone, with its phys high-water, MLX active peak and
// pool size. The IC arms (`--ic-spot25`) read the reference encode only as part of a whole run; this
// isolates it, so a change to the encoder can be measured in seconds and its output compared across
// builds (`--vae-encode-compare`). `--vae-encode-exact-gate` checks the GPU conv route against the CPU.
//
// The input is seeded uniform noise in [-1, 1] at (1, 3, F, H, W). Memory is shape-dependent, not
// content-dependent, so noise measures the same cost as a real clip.
//
//   env: LTX_ENC_WEIGHTS=<vae_encoder.safetensors>   (default: the 2.5 tree's encoder)
//        LTX_ENC_SAVE=<out.safetensors>               write {"latent"} for cross-build comparison
//        LTX_ENC_CPU=1                                run on the CPU stream (exact reference)

import Foundation
import MLX
import LTX2

func vaeEncodePeakProbe(width: Int, height: Int, frames: Int) throws {
    let env = ProcessInfo.processInfo.environment
    let weights = env["LTX_ENC_WEIGHTS"]
        ?? "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx/vae_encoder.safetensors"
    let onCPU = env["LTX_ENC_CPU"] == "1"
    let info = GPU.deviceInfo()
    print(String(format: "[vae-encode-peak] %dx%dx%df · %@ · maxRecommendedWorkingSet %.2f GB (gc limit ≈ %.2f GB)",
                 width, height, frames, onCPU ? "CPU" : "GPU",
                 Double(info.maxRecommendedWorkingSetSize) / 1e9,
                 0.95 * Double(info.maxRecommendedWorkingSetSize) / 1e9))

    let enc = try VideoVAEEncoder.load(path: URL(fileURLWithPath: weights))
    let pixels = MLXRandom.uniform(Float(-1) ..< Float(1), [1, 3, frames, height, width],
                                   key: MLXRandom.key(42))
    eval(pixels)
    Memory.clearCache()
    Memory.peakMemory = 0

    let sampler = PhysSampler(); sampler.start()
    sampler.resetMax()
    let floor = physFootprintBytes()
    let activeBefore = Memory.activeMemory
    let t0 = Date()
    let latent: MLXArray
    if onCPU {
        latent = Device.withDefaultDevice(.cpu) { enc.encode(pixels) }
    } else {
        latent = enc.encode(pixels)
    }
    eval(latent)
    let dt = Date().timeIntervalSince(t0)
    let peak = sampler.maxBytes(); sampler.stop()
    let mlxPeak = Memory.peakMemory
    print(String(format: "[vae-encode-peak] encode %.2fs · phys floor %.2f GB · peak %.2f GB · Δ %.2f GB"
                 + " · MLX active peak Δ %.2f GB · pool after %.2f GB · latent %@",
                 dt, gbOf(floor), gbOf(peak), gbOf(peak > floor ? peak - floor : 0),
                 Double(mlxPeak - activeBefore) / 1e9, Double(Memory.cacheMemory) / 1e9,
                 "\(latent.shape)" as NSString))
    if let out = env["LTX_ENC_SAVE"] {
        try MLX.save(arrays: ["latent": latent.asType(.float32)], url: URL(fileURLWithPath: out))
        print("[vae-encode-peak] → \(out)")
    }
}

/// `--vae-encode-compare <a.safetensors> <b.safetensors>`: relL2 / cosine / maxAbs between two
/// saved `--vae-encode-peak` latents.
func vaeEncodeCompare(_ a: String, _ b: String) throws {
    let x = try MLX.loadArrays(url: URL(fileURLWithPath: a))["latent"]!
    let y = try MLX.loadArrays(url: URL(fileURLWithPath: b))["latent"]!
    print(String(format: "[vae-encode-compare] relL2 %.3e · cosine %.8f · maxAbs %.3e",
                 relL2(x, y), cosine(x, y), maxAbs(x, y)))
}

/// ‖a − b‖ / ‖b‖ in fp32.
func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32), y = b.asType(.float32)
    return (MLX.sqrt(MLX.sum((x - y) * (x - y))) / MLX.sqrt(MLX.sum(y * y))).item(Float.self)
}

/// `--vae-encode-exact-gate`: the encoder's GPU conv route is exact-class (AB-T-0338).
///
/// The cosine ≥ 0.999 gates cannot see Winograd's loss (relL2 ~5e-3 reads cosine 0.99999), and the
/// shipping route relies on mlx's dispatch: a conv3d with non-zero depth padding takes implicit GEMM,
/// not the per-frame Winograd split mlx#3785 added. This gate checks that on the installed mlx. It
/// encodes a seeded 256×256×17 clip — stage 0 sits inside the Winograd window (C + O = 256, 66² ≥
/// 4096 per frame) — on the CPU stream (the reference) and on the GPU through each route:
///   · implicitGEMM (shipping) and exactConv (79b5814) must match the CPU to relL2 < 1e-4;
///   · stock is the poison arm and must MISS (relL2 > 1e-3), or the gate could not tell the routes
///     apart on this mlx — that is exit 2, INCONCLUSIVE, not a pass.
func vaeEncodeExactGate() throws {
    let weights = ProcessInfo.processInfo.environment["LTX_ENC_WEIGHTS"]
        ?? "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx/vae_encoder.safetensors"
    var enc = try VideoVAEEncoder.load(path: URL(fileURLWithPath: weights))
    let pixels = MLXRandom.uniform(Float(-1) ..< Float(1), [1, 3, 17, 256, 256], key: MLXRandom.key(7))
    eval(pixels)
    let cpu = Device.withDefaultDevice(.cpu) { enc.encode(pixels) }
    eval(cpu)
    func arm(_ route: VideoVAEEncoder.ConvRoute) -> Float {
        enc.convRoute = route
        let gpu = enc.encode(pixels)
        eval(gpu)
        return relL2(gpu, cpu)
    }
    let shipping = arm(.implicitGEMM), exactConv = arm(.exactConv), poison = arm(.stock)
    print(String(format: "[vae-encode-exact-gate] GPU vs CPU relL2 · implicitGEMM %.3e · exactConv %.3e · stock (poison) %.3e",
                 shipping, exactConv, poison))
    let routesExact = shipping < 1e-4 && exactConv < 1e-4
    let poisonCaught = poison > 1e-3
    if routesExact && poisonCaught {
        print("[vae-encode-exact-gate] PASS ✅")
    } else if !routesExact {
        print("[vae-encode-exact-gate] FAIL ❌ — a route the encoder relies on is not exact-class on this mlx")
        exit(1)
    } else {
        print("[vae-encode-exact-gate] INCONCLUSIVE — stock conv3d is exact here too, so the poison arm cannot fail")
        exit(2)
    }
}
