// RunLTX2 — parity gate driver for the LTX-2.3 Swift port.
//
// `--connector-gate`: feed the oracle's 49 Gemma hidden-state goldens through the
// Swift connector and compare the resulting video/audio embeds against the oracle
// goldens (cosine + max-abs). This isolates the connector port numerically without
// needing Gemma / mlx-swift-lm.
//
//   xcrun swift run RunLTX2 --connector-gate \
//       [goldens.safetensors] [connector.safetensors]

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXRandom
import LTX2
import MLXLTX2
import MLXToolKit

let defaultGoldens = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/text_encode/goldens.safetensors"
let defaultConnector = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx/connector.safetensors"
let defaultGemma = "/Volumes/DEV_ARCHIVE/models/mlx-community/gemma-3-12b-it-4bit"

func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let af = a.asType(.float32).reshaped(-1)
    let bf = b.asType(.float32).reshaped(-1)
    let dot = (af * bf).sum().item(Float.self)
    let na = MLX.sqrt((af * af).sum()).item(Float.self)
    let nb = MLX.sqrt((bf * bf).sum()).item(Float.self)
    return dot / (na * nb)
}

func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

func connectorGate(goldensPath: String, connectorPath: String) throws {
    print("[connector-gate] goldens:   \(goldensPath)")
    print("[connector-gate] connector: \(connectorPath)")

    let goldens = try MLX.loadArrays(url: URL(fileURLWithPath: goldensPath))
    guard let mask = goldens["attention_mask"],
          let expV = goldens["video_embeds"],
          let expA = goldens["audio_embeds"] else {
        fatalError("goldens missing attention_mask / video_embeds / audio_embeds")
    }

    var hidden: [MLXArray] = []
    for i in 0 ..< 49 {
        let key = String(format: "gemma_hidden_%02d", i)
        guard let h = goldens[key] else { fatalError("missing \(key)") }
        hidden.append(h)
    }
    print("[connector-gate] loaded \(hidden.count) hidden states, shape \(hidden[0].shape), mask \(mask.shape)")

    let connector = try Connector.load(connectorPath: URL(fileURLWithPath: connectorPath))
    let (video, audio) = connector(hiddenStates: hidden, mask: mask)
    eval(video, audio)

    print("[connector-gate] video out \(video.shape)  expected \(expV.shape)")
    print("[connector-gate] audio out \(audio.shape)  expected \(expA.shape)")

    let vCos = cosine(video, expV), vMax = maxAbs(video, expV)
    let aCos = cosine(audio, expA), aMax = maxAbs(audio, expA)
    print(String(format: "[connector-gate] VIDEO cosine=%.6f  maxAbs=%.5f", vCos, vMax))
    print(String(format: "[connector-gate] AUDIO cosine=%.6f  maxAbs=%.5f", aCos, aMax))

    // bf16 8-block stack: expect cosine >= 0.999, maxAbs within bf16 noise.
    let pass = vCos >= 0.999 && aCos >= 0.999
    print(pass ? "[connector-gate] PASS ✅" : "[connector-gate] FAIL ❌")
    if !pass { exit(1) }
}

func gemmaGate(goldensPath: String, gemmaDir: String) async throws {
    print("[gemma-gate] goldens: \(goldensPath)")
    print("[gemma-gate] gemma:   \(gemmaDir)")
    let goldens = try MLX.loadArrays(url: URL(fileURLWithPath: goldensPath))
    guard let tokenIds = goldens["token_ids"], let mask = goldens["attention_mask"] else {
        fatalError("goldens missing token_ids / attention_mask")
    }
    let encoder = try await GemmaEncoder.load(directory: URL(fileURLWithPath: gemmaDir))
    let states = encoder.allHiddenStates(tokenIds: tokenIds, attentionMask: mask)
    eval(states)
    print("[gemma-gate] got \(states.count) states, shape \(states[0].shape)")

    var worstCos: Float = 1, sumCos: Float = 0, worstMax: Float = 0, worstIdx = 0
    for i in 0 ..< states.count {
        guard let exp = goldens[String(format: "gemma_hidden_%02d", i)] else { fatalError("missing golden \(i)") }
        let c = cosine(states[i], exp), m = maxAbs(states[i], exp)
        sumCos += c; worstMax = max(worstMax, m)
        if c < worstCos { worstCos = c; worstIdx = i }
        if i < 3 || i == states.count - 1 { print(String(format: "[gemma-gate] layer %02d cosine=%.6f maxAbs=%.4f", i, c, m)) }
    }
    print(String(format: "[gemma-gate] mean cosine=%.6f  worst=%.6f (layer %d)  maxAbs=%.4f",
                 sumCos / Float(states.count), worstCos, worstIdx, worstMax))
    let pass = worstCos >= 0.999
    print(pass ? "[gemma-gate] PASS ✅" : "[gemma-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Full text-encode front-end: token_ids → Gemma 49 states → connector → (video, audio) embeds.
func textEncodeGate(goldensPath: String, gemmaDir: String, connectorPath: String) async throws {
    print("[text-encode-gate] composing Gemma → connector end-to-end")
    let goldens = try MLX.loadArrays(url: URL(fileURLWithPath: goldensPath))
    guard let tokenIds = goldens["token_ids"], let mask = goldens["attention_mask"],
          let expV = goldens["video_embeds"], let expA = goldens["audio_embeds"] else {
        fatalError("goldens missing required arrays")
    }
    // Prewarm BOTH weight sets off the archive volume (the I5 watchdog recipe: Connector.load
    // int8-quantizes at init, and those evals fault the connector safetensors inside live Metal
    // command buffers — cold pages off DEV_ARCHIVE exceed the watchdog deterministically; the
    // standalone --connector-gate only ever passed on page-warm runs). Production is immune via
    // the engine's WeightPrewarmer; CLI gates must do their own.
    var warm = ((try? FileManager.default.contentsOfDirectory(
        at: URL(fileURLWithPath: gemmaDir), includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" }
    warm.append(URL(fileURLWithPath: connectorPath))
    prewarmFiles(warm)
    print("[text-encode-gate] prewarmed \(warm.count) weight files"); fflush(stdout)
    // SEQUENTIAL like production `encodePrompt` (Gemma releases before the connector loads),
    // mirroring the real pipeline shape instead of a co-residency that no longer exists anywhere.
    let states: [MLXArray]
    do {
        let encoder = try await GemmaEncoder.load(directory: URL(fileURLWithPath: gemmaDir))
        print("[text-encode-gate] gemma loaded"); fflush(stdout)
        states = encoder.allHiddenStates(tokenIds: tokenIds, attentionMask: mask)
        eval(states)
        print("[text-encode-gate] 49 states done"); fflush(stdout)
    }   // encoder (model + context) released here
    Memory.clearCache()
    let connector = try Connector.load(connectorPath: URL(fileURLWithPath: connectorPath))
    print("[text-encode-gate] connector loaded"); fflush(stdout)
    let (video, audio) = connector(hiddenStates: states, mask: mask)
    eval(video, audio)
    print("[text-encode-gate] connector forward done"); fflush(stdout)
    let vCos = cosine(video, expV), aCos = cosine(audio, expA)
    print(String(format: "[text-encode-gate] VIDEO cosine=%.6f  AUDIO cosine=%.6f", vCos, aCos))
    let pass = vCos >= 0.999 && aCos >= 0.999
    print(pass ? "[text-encode-gate] PASS ✅" : "[text-encode-gate] FAIL ❌")
    if !pass { exit(1) }
}

func tinyDiTConfig() -> DiTConfig {
    var c = DiTConfig()
    c.numLayers = 2
    c.videoDim = 64; c.videoNumHeads = 2; c.videoHeadDim = 32
    c.audioDim = 32; c.audioNumHeads = 2; c.audioHeadDim = 16
    c.avCrossNumHeads = 2; c.avCrossHeadDim = 16
    c.videoPatchChannels = 8; c.audioPatchChannels = 8
    c.ffMult = 2.0; c.timestepEmbeddingDim = 32
    c.timestepScaleMultiplier = 1000.0; c.avCaTimestepScaleMultiplier = 1000.0
    return c
}

/// Small-scale DiT parity: tiny seeded LTXModel forward vs oracle goldens.
func ditTinyGate() throws {
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/dit_tiny"
    let weights = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/weights.safetensors"))
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    print("[dit-tiny-gate] \(weights.count) weight tensors")
    let dit = DiT(weights: weights, config: tinyDiTConfig())
    let (video, audio) = dit(
        videoLatent: io["video_latent"]!, audioLatent: io["audio_latent"]!, sigma: io["sigma"]!,
        videoText: io["video_text"], audioText: io["audio_text"],
        videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!)
    eval(video, audio)
    let vCos = cosine(video, io["video_v"]!), vMax = maxAbs(video, io["video_v"]!)
    let aCos = cosine(audio, io["audio_v"]!), aMax = maxAbs(audio, io["audio_v"]!)
    print(String(format: "[dit-tiny-gate] VIDEO cosine=%.6f maxAbs=%.5f  shape %@ vs %@", vCos, vMax, "\(video.shape)" as NSString, "\(io["video_v"]!.shape)" as NSString))
    print(String(format: "[dit-tiny-gate] AUDIO cosine=%.6f maxAbs=%.5f", aCos, aMax))
    let pass = vCos >= 0.999 && aCos >= 0.999
    print(pass ? "[dit-tiny-gate] PASS ✅" : "[dit-tiny-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Full-scale DiT parity: real distilled transformer (bf16) vs oracle goldens.
func ditFullGate() throws {
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/dit_full"
    let weightsPath = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx/transformer-distilled.safetensors"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    print("[dit-full-gate] loading real distilled transformer (bf16)…")
    let dit = try DiT.load(weightsPath: URL(fileURLWithPath: weightsPath), config: DiTConfig(), computeDtype: .bfloat16)
    let (video, audio) = dit(
        videoLatent: io["video_latent"]!, audioLatent: io["audio_latent"]!, sigma: io["sigma"]!,
        videoText: io["video_text"], audioText: io["audio_text"],
        videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!)
    eval(video, audio)
    let vCos = cosine(video, io["video_v"]!), vMax = maxAbs(video, io["video_v"]!)
    let aCos = cosine(audio, io["audio_v"]!), aMax = maxAbs(audio, io["audio_v"]!)
    print(String(format: "[dit-full-gate] VIDEO cosine=%.6f maxAbs=%.4f  shape %@", vCos, vMax, "\(video.shape)" as NSString))
    print(String(format: "[dit-full-gate] AUDIO cosine=%.6f maxAbs=%.4f  shape %@", aCos, aMax, "\(audio.shape)" as NSString))
    let pass = vCos >= 0.999 && aCos >= 0.999
    print(pass ? "[dit-full-gate] PASS ✅" : "[dit-full-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Video VAE decode parity: latent → pixels vs oracle golden (fp32).
func vaeDecodeGate() throws {
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/vae_decode"
    let weightsPath = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx/vae_decoder.safetensors"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let dec = try VideoVAEDecoder.load(path: URL(fileURLWithPath: weightsPath))
    let pixels = dec.decode(io["latent"]!)
    eval(pixels)
    let cos = cosine(pixels, io["pixels"]!), m = maxAbs(pixels, io["pixels"]!)
    print(String(format: "[vae-decode-gate] cosine=%.6f maxAbs=%.5f  shape %@ vs %@", cos, m, "\(pixels.shape)" as NSString, "\(io["pixels"]!.shape)" as NSString))
    let pass = cos >= 0.999
    print(pass ? "[vae-decode-gate] PASS ✅" : "[vae-decode-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Video VAE encode parity: pixels → latent vs oracle golden (fp32).
func vaeEncodeGate() throws {
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/vae_encode"
    let weightsPath = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx/vae_encoder.safetensors"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let enc = try VideoVAEEncoder.load(path: URL(fileURLWithPath: weightsPath))
    let latent = enc.encode(io["pixels"]!)
    eval(latent)
    let cos = cosine(latent, io["latent"]!), m = maxAbs(latent, io["latent"]!)
    print(String(format: "[vae-encode-gate] cosine=%.6f maxAbs=%.5f  shape %@ vs %@", cos, m, "\(latent.shape)" as NSString, "\(io["latent"]!.shape)" as NSString))
    let pass = cos >= 0.999
    print(pass ? "[vae-encode-gate] PASS ✅" : "[vae-encode-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Distilled denoise-loop parity (tiny scale): reuse the dit_tiny weights, run the
/// Euler loop over a short sigma schedule, compare final latents to the oracle.
func denoiseGate() throws {
    let tinyW = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/dit_tiny/weights.safetensors"
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/dit_denoise"
    let weights = try MLX.loadArrays(url: URL(fileURLWithPath: tinyW))
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let dit = DiT(weights: weights, config: tinyDiTConfig())
    let sigmas = io["sigmas"]!.asArray(Float.self)
    let (video, audio) = DenoiseLoop.run(
        dit: dit, videoLatent0: io["video_latent"]!, audioLatent0: io["audio_latent"]!, sigmas: sigmas,
        videoText: io["video_text"], audioText: io["audio_text"],
        videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!)
    eval(video, audio)
    let vCos = cosine(video, io["video_final"]!), vMax = maxAbs(video, io["video_final"]!)
    let aCos = cosine(audio, io["audio_final"]!), aMax = maxAbs(audio, io["audio_final"]!)
    print(String(format: "[denoise-gate] sigmas=%@", "\(sigmas)" as NSString))
    print(String(format: "[denoise-gate] VIDEO cosine=%.6f maxAbs=%.5f  AUDIO cosine=%.6f maxAbs=%.5f", vCos, vMax, aCos, aMax))
    let pass = vCos >= 0.999 && aCos >= 0.999
    print(pass ? "[denoise-gate] PASS ✅" : "[denoise-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// End-to-end one-stage t2v parity (real weights): noise → denoise → unpatchify → VAE decode.
func e2eGate() throws {
    let base = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx"
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/e2e_t2v"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    print("[e2e-gate] loading real DiT (bf16) + VAE decoder…")
    let dit = try DiT.load(weightsPath: URL(fileURLWithPath: "\(base)/transformer-distilled.safetensors"), config: DiTConfig(), computeDtype: .bfloat16)
    let dec = try VideoVAEDecoder.load(path: URL(fileURLWithPath: "\(base)/vae_decoder.safetensors"))
    let sigmas = io["sigmas"]!.asArray(Float.self)
    print("[e2e-gate] denoising \(sigmas.count - 1) steps…")
    let (vfinal, _) = DenoiseLoop.run(
        dit: dit, videoLatent0: io["video_latent"]!, audioLatent0: io["audio_latent"]!, sigmas: sigmas,
        videoText: io["video_text"], audioText: io["audio_text"],
        videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!)
    // unpatchify (1, Nv=128, 128) → (1, 128, F=2, H=8, W=8)
    let vspatial = vfinal.reshaped(1, 2, 8, 8, 128).transposed(0, 4, 1, 2, 3)
    let pixels = dec.decode(vspatial)
    eval(pixels)
    let cos = cosine(pixels, io["pixels"]!), m = maxAbs(pixels, io["pixels"]!)
    print(String(format: "[e2e-gate] pixels %@  cosine=%.6f maxAbs=%.5f  range [%.3f, %.3f]",
                 "\(pixels.shape)" as NSString, cos, m,
                 MLX.min(pixels).item(Float.self), MLX.max(pixels).item(Float.self)))
    // NOTE: the per-step wiring is the parity proof (1-step golden → ~0.99997).
    // Multi-step (8) final-pixel cosine is LOWER (~0.95) by design: the ~3e-5/step
    // bf16 op-ordering diff between MLX-Swift and MLX-Python libmlx amplifies over
    // autoregressive diffusion steps (skill: gate per-pass cosine + image validity,
    // not final-pixel cosine). Strict only for the 1-step wiring fixture.
    let oneStep = sigmas.count <= 2
    let pass = oneStep ? (cos >= 0.999) : (cos >= 0.90)
    print(pass ? "[e2e-gate] PASS ✅ — first end-to-end t2v frame matches the oracle" : "[e2e-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Audio VAE decode parity: audio latent → mel vs oracle golden (fp32).
func audioVaeDecodeGate() throws {
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/audio_vae_decode"
    let weightsPath = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx/audio_vae.safetensors"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let dec = try AudioVAEDecoder.load(path: URL(fileURLWithPath: weightsPath))
    let mel = dec.decode(io["latent"]!)
    eval(mel)
    let cos = cosine(mel, io["mel"]!), m = maxAbs(mel, io["mel"]!)
    print(String(format: "[audio-vae-decode-gate] cosine=%.6f maxAbs=%.5f  shape %@ vs %@", cos, m, "\(mel.shape)" as NSString, "\(io["mel"]!.shape)" as NSString))
    let pass = cos >= 0.999
    print(pass ? "[audio-vae-decode-gate] PASS ✅" : "[audio-vae-decode-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Vocoder+BWE parity: mel → 48kHz waveform vs oracle golden (fp32).
func vocoderGate() throws {
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/vocoder"
    let weightsPath = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx/vocoder.safetensors"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let voc = try Vocoder.load(path: URL(fileURLWithPath: weightsPath))
    let wav = voc(io["mel"]!)
    eval(wav)
    let cos = cosine(wav, io["wav"]!), m = maxAbs(wav, io["wav"]!)
    print(String(format: "[vocoder-gate] cosine=%.6f maxAbs=%.5f  shape %@ vs %@", cos, m, "\(wav.shape)" as NSString, "\(io["wav"]!.shape)" as NSString))
    let pass = cos >= 0.999
    print(pass ? "[vocoder-gate] PASS ✅" : "[vocoder-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Composed audio-decode parity: audio tokens (1,T,128) → unpatchify → AudioVAE → Vocoder → wav.
/// Mirrors LTX2Pipeline.decodeAudio without loading the full pipeline.
func audioDecodeGate() throws {
    let base = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx"
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/audio_decode"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let audioVAE = try AudioVAEDecoder.load(path: URL(fileURLWithPath: "\(base)/audio_vae.safetensors"))
    let voc = try Vocoder.load(path: URL(fileURLWithPath: "\(base)/vocoder.safetensors"))
    let tokens = io["tokens"]!
    let B = tokens.dim(0), T = tokens.dim(1)
    let audioLatent = tokens.reshaped(B, T, 8, 16).transposed(0, 2, 1, 3)  // AudioPatchifier.unpatchify
    let wav = voc(audioVAE.decode(audioLatent))
    eval(wav)
    let cos = cosine(wav, io["wav"]!), m = maxAbs(wav, io["wav"]!)
    print(String(format: "[audio-decode-gate] cosine=%.6f maxAbs=%.5f  shape %@ vs %@", cos, m, "\(wav.shape)" as NSString, "\(io["wav"]!.shape)" as NSString))
    let pass = cos >= 0.999
    print(pass ? "[audio-decode-gate] PASS ✅" : "[audio-decode-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Spatial-x2 upsampler parity: latent → 2×-spatial latent vs oracle golden (fp32).
func upsamplerGate() throws {
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/upsampler"
    let weightsPath = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx/spatial_upscaler_x2_v1_1.safetensors"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let up = try Upsampler.load(path: URL(fileURLWithPath: weightsPath))
    let out = up(io["latent"]!)
    eval(out)
    let cos = cosine(out, io["out"]!), m = maxAbs(out, io["out"]!)
    print(String(format: "[upsampler-gate] cosine=%.6f maxAbs=%.5f  shape %@ vs %@", cos, m, "\(out.shape)" as NSString, "\(io["out"]!.shape)" as NSString))
    let pass = cos >= 0.999
    print(pass ? "[upsampler-gate] PASS ✅" : "[upsampler-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Two-stage upscale-step parity: half-res latent → denorm → upsample → renorm vs oracle.
func upscaleStepGate() throws {
    let base = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx"
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/upscale_step"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let enc = try VideoVAEEncoder.load(path: URL(fileURLWithPath: "\(base)/vae_encoder.safetensors"))
    let up = try Upsampler.load(path: URL(fileURLWithPath: "\(base)/spatial_upscaler_x2_v1_1.safetensors"))
    let renorm = enc.normalizeLatent(up(enc.denormalizeLatent(io["half"]!)))
    eval(renorm)
    let cos = cosine(renorm, io["renorm"]!), m = maxAbs(renorm, io["renorm"]!)
    print(String(format: "[upscale-step-gate] cosine=%.6f maxAbs=%.5f  shape %@ vs %@", cos, m, "\(renorm.shape)" as NSString, "\(io["renorm"]!.shape)" as NSString))
    let pass = cos >= 0.999
    print(pass ? "[upscale-step-gate] PASS ✅" : "[upscale-step-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// q8 DiT parity: int8-quantized transformer (bf16 activations) vs oracle q8 golden.
func ditQ8Gate() throws {
    let q8 = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx-q8/transformer-distilled.safetensors"
    let base = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(base)/dit_full/io.safetensors"))   // inputs (reused)
    let exp = try MLX.loadArrays(url: URL(fileURLWithPath: "\(base)/dit_q8/io.safetensors"))     // q8 outputs
    print("[dit-q8-gate] loading int8 transformer…")
    let dit = try DiT.load(weightsPath: URL(fileURLWithPath: q8), config: DiTConfig(), computeDtype: .bfloat16)
    let (video, audio) = dit(
        videoLatent: io["video_latent"]!, audioLatent: io["audio_latent"]!, sigma: io["sigma"]!,
        videoText: io["video_text"], audioText: io["audio_text"],
        videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!)
    eval(video, audio)
    let vCos = cosine(video, exp["video_v"]!), vMax = maxAbs(video, exp["video_v"]!)
    let aCos = cosine(audio, exp["audio_v"]!), aMax = maxAbs(audio, exp["audio_v"]!)
    print(String(format: "[dit-q8-gate] VIDEO cosine=%.6f maxAbs=%.4f  AUDIO cosine=%.6f maxAbs=%.4f", vCos, vMax, aCos, aMax))
    let pass = vCos >= 0.999 && aCos >= 0.999
    print(pass ? "[dit-q8-gate] PASS ✅" : "[dit-q8-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// q4 DiT parity: int4-quantized transformer (bf16 activations) vs oracle q4 golden. Same
/// quant-aware path as q8 — `DiT.load` auto-detects 4-bit from the scales shape; this gate just
/// checks Swift-q4-forward == oracle-q4-forward on identical q4 weights (so the bar stays 0.999).
func ditQ4Gate() throws {
    let q4 = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx-q4/transformer-distilled.safetensors"
    let base = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(base)/dit_full/io.safetensors"))   // inputs (reused)
    let exp = try MLX.loadArrays(url: URL(fileURLWithPath: "\(base)/dit_q4/io.safetensors"))     // q4 outputs
    print("[dit-q4-gate] loading int4 transformer…")
    let dit = try DiT.load(weightsPath: URL(fileURLWithPath: q4), config: DiTConfig(), computeDtype: .bfloat16)
    let (video, audio) = dit(
        videoLatent: io["video_latent"]!, audioLatent: io["audio_latent"]!, sigma: io["sigma"]!,
        videoText: io["video_text"], audioText: io["audio_text"],
        videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!)
    eval(video, audio)
    let vCos = cosine(video, exp["video_v"]!), vMax = maxAbs(video, exp["video_v"]!)
    let aCos = cosine(audio, exp["audio_v"]!), aMax = maxAbs(audio, exp["audio_v"]!)
    print(String(format: "[dit-q4-gate] VIDEO cosine=%.6f maxAbs=%.4f  AUDIO cosine=%.6f maxAbs=%.4f", vCos, vMax, aCos, aMax))
    let pass = vCos >= 0.999 && aCos >= 0.999
    print(pass ? "[dit-q4-gate] PASS ✅" : "[dit-q4-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Per-token-timestep DiT parity (i2v foundation): bf16 transformer with mixed per-token
/// timesteps (frame-0 tokens at 0, rest at sigma) vs oracle golden. Reuses dit_full inputs;
/// timesteps + expected outputs come from the dit_pertoken fixture.
func ditPerTokenGate() throws {
    let bf16 = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx/transformer-distilled.safetensors"
    let base = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(base)/dit_full/io.safetensors"))       // base inputs
    let pt = try MLX.loadArrays(url: URL(fileURLWithPath: "\(base)/dit_pertoken/io.safetensors"))   // timesteps + outputs
    print("[dit-pertoken-gate] loading bf16 transformer…")
    let dit = try DiT.load(weightsPath: URL(fileURLWithPath: bf16), config: DiTConfig(), computeDtype: .bfloat16)
    let (video, audio) = dit(
        videoLatent: io["video_latent"]!, audioLatent: io["audio_latent"]!, sigma: io["sigma"]!,
        videoText: io["video_text"], audioText: io["audio_text"],
        videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!,
        videoTimesteps: pt["video_timesteps"]!, audioTimesteps: pt["audio_timesteps"]!)
    eval(video, audio)
    let vCos = cosine(video, pt["video_v"]!), vMax = maxAbs(video, pt["video_v"]!)
    let aCos = cosine(audio, pt["audio_v"]!), aMax = maxAbs(audio, pt["audio_v"]!)
    print(String(format: "[dit-pertoken-gate] VIDEO cosine=%.6f maxAbs=%.4f  AUDIO cosine=%.6f maxAbs=%.4f", vCos, vMax, aCos, aMax))
    let pass = vCos >= 0.999 && aCos >= 0.999
    print(pass ? "[dit-pertoken-gate] PASS ✅" : "[dit-pertoken-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// L1 runtime-LoRA hook gate (real distilled DiT, dit_full goldens):
///   1. lora-OFF forward matches the golden  → my dense hook didn't regress the base
///   2. apply LoRA → output changes, stays finite (cosine < 1 vs base, no NaN)
///   3. detach → output restores to the base exactly
func loraGate(loraPath: String) throws {
    let dir = "/Users/dustinnielson/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift/parity/goldens/dit_full"
    let weightsPath = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx/transformer-distilled.safetensors"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    print("[lora-gate] lora: \(loraPath)")
    print("[lora-gate] loading real distilled transformer (bf16)…")
    let dit = try DiT.load(weightsPath: URL(fileURLWithPath: weightsPath), config: DiTConfig(), computeDtype: .bfloat16)
    func fwd() -> (MLXArray, MLXArray) {
        let (v, a) = dit(
            videoLatent: io["video_latent"]!, audioLatent: io["audio_latent"]!, sigma: io["sigma"]!,
            videoText: io["video_text"], audioText: io["audio_text"],
            videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!)
        eval(v, a); return (v, a)
    }
    // 1. lora-off vs golden
    let (vBase, _) = fwd()
    let offCos = cosine(vBase, io["video_v"]!)
    print(String(format: "[lora-gate] lora-OFF vs golden: video cosine=%.6f", offCos))
    // 2. apply + forward
    try LTX2LoRA.apply(URL(fileURLWithPath: loraPath), strength: 1.0, to: dit)
    print("[lora-gate] applied \(dit.loraTargetCount) LoRA targets")
    let (vOn, aOn) = fwd()
    let onVsBase = cosine(vOn, vBase)
    let vMax = vOn.asType(.float32).max().item(Float.self)
    let aMax = aOn.asType(.float32).max().item(Float.self)
    let finite = vMax.isFinite && aMax.isFinite
    print(String(format: "[lora-gate] lora-ON vs base: video cosine=%.6f  finite=%@ (vmax=%.3f)",
                 onVsBase, finite ? "yes" : "no", vMax))
    // 3. detach → restore
    LTX2LoRA.detach(dit)
    let (vOff2, _) = fwd()
    let restoreCos = cosine(vOff2, vBase)
    print(String(format: "[lora-gate] detach vs base: video cosine=%.6f", restoreCos))
    let pass = offCos >= 0.999 && finite && onVsBase < 0.9999 && restoreCos >= 0.99999
    print(pass ? "[lora-gate] PASS ✅" : "[lora-gate] FAIL ❌")
    if !pass { exit(1) }
}

// MARK: - Memory bench (efficiency-sweep harness, contract 1.14.0 split footprint)
//
// Runs the FULL staged pipeline at the declared 704×512×9f two-stage envelope and reports the
// OS `phys_footprint` resident floor (post-run + clearCache = DiT-only, the stages self-evict) and
// the peak high-water across the run. The split footprint is declared from THESE: residentBytes =
// floor, peakActivationBytes = peak − floor. We sample `phys_footprint` (NOT `Memory.peakMemory`,
// which counts cumulative allocations and misleads under the cache cap — the Wan profiler lesson).

func gbOf(_ b: UInt64) -> Double { Double(b) / 1_000_000_000.0 }

/// OS `phys_footprint` (bytes) via `task_info(TASK_VM_INFO)` — the figure the MemoryGovernor and
/// Activity Monitor are grounded on. Returns 0 on failure.
func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

/// Background phys_footprint high-water sampler (the peak is a transient inside the denoise loop,
/// not observable at a phase boundary — poll it).
final class PhysSampler: @unchecked Sendable {
    private let lock = NSLock()
    private var _max: UInt64 = 0
    private var _running = false
    func start() {
        lock.lock(); _running = true; lock.unlock()
        let t = Thread { [weak self] in
            while self?.running == true {
                let p = physFootprintBytes()
                self?.observe(p)
                Thread.sleep(forTimeInterval: 0.025)
            }
        }
        t.stackSize = 1 << 20
        t.start()
    }
    var running: Bool { lock.lock(); defer { lock.unlock() }; return _running }
    func observe(_ p: UInt64) { lock.lock(); if p > _max { _max = p }; lock.unlock() }
    func resetMax() { lock.lock(); _max = physFootprintBytes(); lock.unlock() }
    func maxBytes() -> UInt64 { lock.lock(); defer { lock.unlock() }; return _max }
    func stop() { lock.lock(); _running = false; lock.unlock() }
}

/// Page weight files into the OS cache before `load()`'s GPU evals so a cold fault off the archive
/// volume never stalls a live Metal command buffer (the I5 watchdog abort; the engine's
/// `WeightPrewarmer` does this in-app — the CLI bench replicates it). Streams + discards; no big alloc.
func prewarmFiles(_ paths: [URL]) {
    for p in paths {
        guard let fh = try? FileHandle(forReadingFrom: p) else { continue }
        defer { try? fh.close() }
        // Each read returns an AUTORELEASED chunk — without a per-iteration pool drain the whole
        // file accumulates live in phys_footprint (observed: ~60 GB of "freed" chunks still resident
        // after a 50 GB prewarm, which then drove the GPU into the I5 watchdog on the next eval).
        var done = false
        while !done {
            autoreleasepool {
                guard let chunk = try? fh.read(upToCount: 64 << 20), !chunk.isEmpty else {
                    done = true
                    return
                }
                _ = chunk.count
            }
        }
    }
}

func memBenchGate(quant: String) async throws {
    let base = "/Volumes/DEV_ARCHIVE/models/dgrauet"
    let ltxDir = URL(fileURLWithPath: "\(base)/ltx-2.3-mlx")
    let gemmaDir = URL(fileURLWithPath: defaultGemma)
    let transformerPath: URL?
    switch quant {
    case "int8", "q8": transformerPath = URL(fileURLWithPath: "\(base)/ltx-2.3-mlx-q8/transformer-distilled.safetensors")
    case "int4", "q4": transformerPath = URL(fileURLWithPath: "\(base)/ltx-2.3-mlx-q4/transformer-distilled.safetensors")
    default: transformerPath = nil  // bf16
    }
    let h = 512, w = 704, nf = 9
    print("[mem-bench] quant=\(quant)  envelope=\(w)×\(h)×\(nf)f  path=two-stage  (per-stage evict)")

    // Prewarm (mirror the engine's WeightPrewarmer): the DiT transformer + all LTX components + Gemma.
    let p0 = Date()
    var warm = [transformerPath ?? ltxDir.appendingPathComponent("transformer-distilled.safetensors")]
    for f in ["connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
              "audio_vae.safetensors", "vocoder.safetensors", "spatial_upscaler_x2_v1_1.safetensors"] {
        warm.append(ltxDir.appendingPathComponent(f))
    }
    warm.append(contentsOf: ((try? FileManager.default.contentsOfDirectory(at: gemmaDir, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" })
    prewarmFiles(warm)
    print(String(format: "[mem-bench] prewarm %.1fs (%d files)", Date().timeIntervalSince(p0), warm.count))

    let sampler = PhysSampler(); sampler.start()
    let t0 = Date()
    let pipeline = try await LTX2Pipeline.load(ltxDir: ltxDir, gemmaDir: gemmaDir, transformerPath: transformerPath)
    print(String(format: "[mem-bench] load %.1fs  phys-after-load(DiT only)=%.2f GB", Date().timeIntervalSince(t0), gbOf(physFootprintBytes())))

    // Warmup (compiles size-specific kernels; not measured for peak).
    _ = try await pipeline.t2vTwoStage(prompt: "a cat playing piano", height: h, width: w, numFrames: nf, fps: 24, seed: 42)
    Memory.clearCache()
    let floor = physFootprintBytes()  // stages self-evicted → DiT weights + framework
    print(String(format: "[mem-bench] resident floor (post-run + clearCache): %.2f GB", gbOf(floor)))

    // Measured run — the sampler tracks the phys high-water across every stage.
    sampler.resetMax()
    let r0 = Date()
    let out = try await pipeline.t2vTwoStage(prompt: "a cat playing piano", height: h, width: w, numFrames: nf, fps: 24, seed: 42)
    eval(out.video); if let a = out.audio { eval(a) }
    let peak = sampler.maxBytes()
    sampler.stop()
    let activation = peak > floor ? peak - floor : 0
    print(String(format: "[mem-bench] run %.1fs", Date().timeIntervalSince(r0)))
    print(String(format: "[mem-bench] PEAK phys_footprint: %.2f GB", gbOf(peak)))
    print(String(format: "[mem-bench] DECLARE → residentBytes ≈ %.2f GB (%llu)  peakActivationBytes ≈ %.2f GB (%llu)",
                 gbOf(floor), floor, gbOf(activation), activation))
    print(String(format: "[mem-bench] SUMMARY quant=%@ resident=%llu peak=%llu activation=%llu", quant as NSString, floor, peak, activation))
}

// MARK: - i2v spot measure (BRIDGE-LTX-005) — tighten the max128 activation hint
//
// max128's `peakActivationBytesHint` is held at the conservative pre-T3b ceiling (52 GB) because
// the i2v/per-token path was UNMEASURED at the new 481f cap. This gate produces that datum: it
// drives the WRAPPER (so the profile clamp, the runtime i2v-adapter LoRA (~4.9 GB resident), and
// the MP4 encode are all inside the sampled window) with a synthetic init frame at the max128
// envelope, and reports the SPLIT line (floor / peak / activation) the hint is recalibrated from.
@InferenceActor
func i2vSpotGate(width: Int, height: Int, frames: Int) async throws {
    let cfg = LTX2Configuration(
        quant: .bf16,
        ltxDirectory: URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx"),
        gemmaDirectory: URL(fileURLWithPath: defaultGemma),
        // The LoRA cache root (`ltx-lora-cache/i2v-adapter.safetensors`, already fetched by the app).
        modelsRootDirectory: URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/weights"),
        profile: .max128)
    print("[i2v-spot] request \(width)×\(height)×\(frames)f bf16 · profile=max128 · lora=i2v-adapter")

    // Prewarm off the config's own `prewarmPaths` (dirs → their safetensors) + the LoRA file,
    // mirroring the engine's WeightPrewarmer so the cold load can't trip the Metal watchdog.
    let p0 = Date()
    var warm: [URL] = []
    for p in cfg.prewarmPaths {
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: p.path, isDirectory: &isDir), isDir.boolValue {
            warm.append(contentsOf: ((try? FileManager.default.contentsOfDirectory(
                at: p, includingPropertiesForKeys: nil)) ?? [])
                .filter { $0.pathExtension == "safetensors" })
        } else {
            warm.append(p)
        }
    }
    warm.append(URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/weights/ltx-lora-cache/i2v-adapter.safetensors"))
    prewarmFiles(warm)
    print(String(format: "[i2v-spot] prewarm %.1fs (%d files)", Date().timeIntervalSince(p0), warm.count))

    let sampler = PhysSampler(); sampler.start()
    let pkg = MLXLTX2Package(configuration: cfg)
    let t0 = Date()
    try await pkg.load()
    print(String(format: "[i2v-spot] load %.1fs  phys-after-load=%.2f GB",
                 Date().timeIntervalSince(t0), gbOf(physFootprintBytes())))
    // The load-time DiT kernel warmup runs with an UNCAPPED pool and retains its compile/activation
    // buffers (~60 GB observed above the 38 GB DiT). Entering the first run that bloated makes the
    // 4.9 GB LoRA-apply eval fault pages inside a live command buffer → the I5 GPU watchdog. The
    // app never sees this state (its runs start post-prepare, pool settled) — drop it before run.
    Memory.clearCache()
    print(String(format: "[i2v-spot] post-load clearCache → phys=%.2f GB", gbOf(physFootprintBytes())))

    let png = try syntheticInitPNG(width: width, height: height)
    func request(_ nf: Int) -> T2VRequest {
        T2VRequest(prompt: "a fox running down a beach at sunset, waves rolling in",
                   initImage: MLXToolKit.Image(format: .png, data: png),
                   numFrames: nf, fps: 24, width: width, height: height, seed: 42,
                   metaData: [LoRAMetaKeys.id: .string("i2v-adapter")])
    }

    // Warmup at 9f: compiles kernels + applies the LoRA + loads the decode/encode stacks —
    // excluded from the peak so the floor below is the honest steady-state resident set.
    _ = try await pkg.run(request(9))
    Memory.clearCache()
    let floor = physFootprintBytes()
    print(String(format: "[i2v-spot] resident floor (post-warmup + clearCache): %.2f GB  (DiT + i2v LoRA)", gbOf(floor)))

    sampler.resetMax()
    let r0 = Date()
    let resp = try await pkg.run(request(frames)) as! T2VResponse
    let peak = sampler.maxBytes(); sampler.stop()
    let activation = peak > floor ? peak - floor : 0
    let ranFrames = Int(((resp.video.durationSeconds ?? 0) * (resp.video.frameRate ?? 24)).rounded())
    print(String(format: "[i2v-spot] run %.1fs  ran %df  mp4 %.1f MB",
                 Date().timeIntervalSince(r0), ranFrames, Double(resp.video.data.count) / 1_000_000))
    print(String(format: "[i2v-spot] SPLIT floor=%.2f GB  peak=%.2f GB  act=%.2f GB", gbOf(floor), gbOf(peak), gbOf(activation)))
    print(String(format: "[i2v-spot] SUMMARY floor=%llu peak=%llu activation=%llu", floor, peak, activation))
}

/// Beach-horizon gradient PNG — synthetic but structured enough for a realistic VAE encode.
func syntheticInitPNG(width: Int, height: Int) throws -> Data {
    struct PNGError: Error {}
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                              bytesPerRow: 0, space: cs,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw PNGError() }
    let colors = [CGColor(red: 0.95, green: 0.65, blue: 0.35, alpha: 1),   // sunset sky
                  CGColor(red: 0.25, green: 0.45, blue: 0.70, alpha: 1),   // sea
                  CGColor(red: 0.85, green: 0.75, blue: 0.55, alpha: 1)]   // sand
    let grad = CGGradient(colorsSpace: cs, colors: colors as CFArray, locations: [0, 0.55, 1])!
    ctx.drawLinearGradient(grad, start: CGPoint(x: 0, y: CGFloat(height)), end: .zero, options: [])
    guard let img = ctx.makeImage() else { throw PNGError() }
    let out = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(out, "public.png" as CFString, 1, nil) else { throw PNGError() }
    CGImageDestinationAddImage(dest, img, nil)
    guard CGImageDestinationFinalize(dest) else { throw PNGError() }
    return out as Data
}

// MARK: - Encode stress gate — isolate the H.264 encoder stall (contention vs frame-count)
//
// Reproduces the encodeMP4 hang without the ~4min generation. Synthesizes N frames and encodes them,
// optionally under ~38 GB of resident memory pressure (--hog, like the bf16 DiT) and/or with the
// SOFTWARE encoder (--software, bypasses the hardware VideoToolbox media engine). Four combinations
// pin the cause: frames-only vs +hog isolates CONTENTION; hardware vs software isolates the HARDWARE
// media engine.
@InferenceActor
func encodeStressGate(frames n: Int, software: Bool, hog: Bool, audio: Bool) async throws {
    print("[encode-stress] N=\(n) frames  software=\(software)  hog(38GB)=\(hog)  audio=\(audio)")
    var hogs: [MLXArray] = []
    if hog {
        for _ in 0 ..< 38 { let a = MLXArray.zeros([250_000_000]).asType(.float32); eval(a); hogs.append(a) }  // 38×1GB
        print(String(format: "[encode-stress] resident pressure ≈ %.0f GB held", 38.0))
    }
    let frames = MLXRandom.normal([1, n, 512, 704, 3]).asType(.float32)
    // --audio: a second (audio) track makes AVAssetWriter INTERLEAVE — the suspected real cause of the
    // stall (video input readiness blocks waiting for audio, which we append only after all frames).
    let waveform: MLXArray? = audio
        ? MLXRandom.normal([1, 2, Int(Double(n) / 24.0 * 48000.0)]).asType(.float32) * 0.05
        : nil
    eval(frames); if let waveform { eval(waveform) }
    let t0 = Date()
    do {
        let data = try await encodeMP4(frames: frames, fps: 24, audio: waveform, software: software)
        print(String(format: "[encode-stress] PASS ✅  %.1fs  %d bytes", Date().timeIntervalSince(t0), data.count))
    } catch {
        print("[encode-stress] FAIL ❌  \(Int(Date().timeIntervalSince(t0)))s  \(error)")
    }
    _ = hogs.count  // keep pressure alive through the encode
}

// MARK: - VAE chunked-decode gate (LOW-TIER-PLAN T0/T1)
//
// Whole-frame decode = the EXACT reference; chunked decode must match it. Reports cosine + maxAbs +
// the minimum per-frame PSNR around each chunk seam (the RIFE seam-eval pattern), plus the phys
// peak of whole vs chunked — the memory win T1 exists for. Random latent: peak/parity are functions
// of SHAPE, not content. Usage: --vae-chunk-gate [Flat] [chunk] [halo]  (defaults 15, 5, 4 —
// F_lat 15 = 113 output frames @704×512).
@InferenceActor
func vaeChunkGate(fLat: Int, chunk: Int, halo: Int) async throws {
    let weightsPath = "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx/vae_decoder.safetensors"
    let dec = try VideoVAEDecoder.load(path: URL(fileURLWithPath: weightsPath))
    let hLat = 512 / 32, wLat = 704 / 32
    MLXRandom.seed(7)
    let latent = MLXRandom.normal([1, 128, fLat, hLat, wLat]).asType(.float32)
    eval(latent)
    print("[vae-chunk-gate] F_lat=\(fLat) (→\(8 * fLat - 7) frames @704×512)  chunk=\(chunk) halo=\(halo)")

    // Whole-frame reference + its phys peak.
    Memory.clearCache()
    let base = physFootprintBytes()
    let sampler = PhysSampler(); sampler.start(); sampler.resetMax()
    let t0 = Date()
    let whole = dec.decode(latent); eval(whole)
    let wholePeak = sampler.maxBytes()
    print(String(format: "[vae-chunk-gate] whole:   %.1fs  peakΔ=%.2f GB  out=%@",
                 Date().timeIntervalSince(t0), gbOf(wholePeak > base ? wholePeak - base : 0),
                 "\(whole.shape)" as NSString))

    // Chunked + its phys peak.
    Memory.clearCache()
    sampler.resetMax()
    let t1 = Date()
    let chunked = dec.decodeChunked(latent, chunkFrames: chunk, halo: halo); eval(chunked)
    let chunkPeak = sampler.maxBytes(); sampler.stop()
    print(String(format: "[vae-chunk-gate] chunked: %.1fs  peakΔ=%.2f GB  out=%@",
                 Date().timeIntervalSince(t1), gbOf(chunkPeak > base ? chunkPeak - base : 0),
                 "\(chunked.shape)" as NSString))

    guard whole.shape == chunked.shape else {
        print("[vae-chunk-gate] FAIL ❌ shape mismatch \(whole.shape) vs \(chunked.shape) — trim math wrong")
        exit(1)
    }
    let cos = cosine(chunked, whole), m = maxAbs(chunked, whole)
    // Min per-frame PSNR across ±2 pixel frames around each chunk seam (seam = latent boundary ×8).
    var minPSNR = Float.infinity; var minAt = -1
    var boundary = chunk
    while boundary < fLat {
        let seam = 8 * boundary - 7
        for f in max(0, seam - 2) ... min(whole.dim(2) - 1, seam + 2) {
            let d = (chunked[0..., 0..., f] - whole[0..., 0..., f]).asType(.float32)
            let mse = (d * d).mean().item(Float.self)
            let psnr = mse <= 1e-12 ? Float(99) : 10 * log10(4.0 / mse)   // range [-1,1] → peak²=4
            if psnr < minPSNR { minPSNR = psnr; minAt = f }
        }
        boundary += chunk
    }
    print(String(format: "[vae-chunk-gate] cosine=%.6f  maxAbs=%.5f  minSeamPSNR=%.1f dB (frame %d)",
                 cos, m, minPSNR, minAt))
    let pass = cos >= 0.9999 && minPSNR >= 60
    print(pass ? "[vae-chunk-gate] PASS ✅" : "[vae-chunk-gate] FAIL ❌ (grow halo or fix trim math)")
    if !pass { exit(1) }
}

/// BRIDGE-LTX-004: one-shot (system, user) chat completion on the SAME Gemma-3 the encoder uses —
/// the seam the app's prompt enhancer consumes via `GemmaTextGenerator`. Live gate (no golden:
/// generation is sampled); PASS = non-empty completion + clean load→generate→release.
func gemmaTextGenGate(gemmaDir: String) async throws {
    print("[gemma-textgen-gate] gemma: \(gemmaDir)")
    let system = """
    You are a prompt enhancer for a text-to-video model. Rewrite the user's brief as ONE flowing \
    present-tense paragraph with explicit camera movement and a description of the audio. Output \
    ONLY the enhanced prompt.
    """
    let user = "a lighthouse keeper climbing the spiral stairs at dusk\n\nTarget duration: ~5 seconds of video."
    let generator = GemmaTextGenerator(gemmaDirectory: URL(fileURLWithPath: gemmaDir))
    let t0 = Date()
    let out = try await generator.generate(system: system, user: user, maxTokens: 320)
    let dt = Date().timeIntervalSince(t0)
    print("[gemma-textgen-gate] completion (\(out.count) chars, \(String(format: "%.1f", dt))s):\n\(out)\n")
    let pass = !out.isEmpty && out != user
    print(String(format: "[gemma-textgen-gate] cache after clearCache: %.2f GB", Double(Memory.cacheMemory) / 1e9))
    print(pass ? "[gemma-textgen-gate] PASS ✅" : "[gemma-textgen-gate] FAIL ❌ (empty completion)")
}

let args = CommandLine.arguments
let positional = args.dropFirst().filter { !$0.hasPrefix("--") }
if args.contains("--connector-gate") {
    let goldens = positional.first ?? defaultGoldens
    let connector = positional.dropFirst().first ?? defaultConnector
    try connectorGate(goldensPath: goldens, connectorPath: connector)
} else if args.contains("--gemma-gate") {
    let goldens = positional.first ?? defaultGoldens
    let gemmaDir = positional.dropFirst().first ?? defaultGemma
    try await gemmaGate(goldensPath: goldens, gemmaDir: gemmaDir)
} else if args.contains("--gemma-textgen-gate") {
    try await gemmaTextGenGate(gemmaDir: positional.first ?? defaultGemma)
} else if args.contains("--text-encode-gate") {
    try await textEncodeGate(goldensPath: defaultGoldens, gemmaDir: defaultGemma, connectorPath: defaultConnector)
} else if args.contains("--dit-tiny-gate") {
    try ditTinyGate()
} else if args.contains("--dit-q8-gate") {
    try ditQ8Gate()
} else if args.contains("--dit-q4-gate") {
    try ditQ4Gate()
} else if args.contains("--dit-pertoken-gate") {
    try ditPerTokenGate()
} else if args.contains("--dit-full-gate") {
    try ditFullGate()
} else if args.contains("--lora-gate") {
    try loraGate(loraPath: positional.first ?? "/tmp/ltx_transition_lora.safetensors")
} else if args.contains("--vae-decode-gate") {
    try vaeDecodeGate()
} else if args.contains("--vae-encode-gate") {
    try vaeEncodeGate()
} else if args.contains("--audio-vae-decode-gate") {
    try audioVaeDecodeGate()
} else if args.contains("--vocoder-gate") {
    try vocoderGate()
} else if args.contains("--audio-decode-gate") {
    try audioDecodeGate()
} else if args.contains("--upsampler-gate") {
    try upsamplerGate()
} else if args.contains("--upscale-step-gate") {
    try upscaleStepGate()
} else if args.contains("--denoise-gate") {
    try denoiseGate()
} else if args.contains("--e2e-gate") {
    try e2eGate()
} else if args.contains("--vae-chunk-gate") {
    let ints = positional.compactMap { Int($0) }
    try await vaeChunkGate(fLat: ints.count > 0 ? ints[0] : 15,
                           chunk: ints.count > 1 ? ints[1] : 5,
                           halo: ints.count > 2 ? ints[2] : 4)
} else if args.contains("--encode-stress") {
    let n = positional.first.flatMap { Int($0) } ?? 41
    try await encodeStressGate(frames: n, software: args.contains("--software"), hog: args.contains("--hog"),
                               audio: args.contains("--audio"))
} else if args.contains("--mem-bench") {
    try await memBenchGate(quant: positional.first ?? "bf16")
} else if args.contains("--i2v-spot") {
    let ints = positional.compactMap { Int($0) }
    try await i2vSpotGate(width: ints.count > 0 ? ints[0] : 704,
                          height: ints.count > 1 ? ints[1] : 512,
                          frames: ints.count > 2 ? ints[2] : 481)
} else {
    print("usage: RunLTX2 --connector-gate | --gemma-gate | --text-encode-gate | --dit-tiny-gate  [goldens.safetensors] [path]")
    print("       RunLTX2 --mem-bench [bf16|int8|int4]   (efficiency-sweep footprint at 704×512×9f)")
    print("       RunLTX2 --i2v-spot [w] [h] [frames]    (BRIDGE-LTX-005 max128 i2v SPLIT measure, default 704 512 481)")
}
