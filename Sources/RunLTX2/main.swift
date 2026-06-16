// RunLTX2 — parity gate driver for the LTX-2.3 Swift port.
//
// `--connector-gate`: feed the oracle's 49 Gemma hidden-state goldens through the
// Swift connector and compare the resulting video/audio embeds against the oracle
// goldens (cosine + max-abs). This isolates the connector port numerically without
// needing Gemma / mlx-swift-lm.
//
//   xcrun swift run RunLTX2 --connector-gate \
//       [goldens.safetensors] [connector.safetensors]

import Foundation
import MLX
import LTX2

let defaultGoldens = "/Users/dustinnielson/Development/ltx-2-mlx-swift/parity/goldens/text_encode/goldens.safetensors"
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
    let encoder = try await GemmaEncoder.load(directory: URL(fileURLWithPath: gemmaDir))
    let states = encoder.allHiddenStates(tokenIds: tokenIds, attentionMask: mask)
    eval(states)
    let connector = try Connector.load(connectorPath: URL(fileURLWithPath: connectorPath))
    let (video, audio) = connector(hiddenStates: states, mask: mask)
    eval(video, audio)
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
    let dir = "/Users/dustinnielson/Development/ltx-2-mlx-swift/parity/goldens/dit_tiny"
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
    let dir = "/Users/dustinnielson/Development/ltx-2-mlx-swift/parity/goldens/dit_full"
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
    let dir = "/Users/dustinnielson/Development/ltx-2-mlx-swift/parity/goldens/vae_decode"
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
    let dir = "/Users/dustinnielson/Development/ltx-2-mlx-swift/parity/goldens/vae_encode"
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
} else if args.contains("--text-encode-gate") {
    try await textEncodeGate(goldensPath: defaultGoldens, gemmaDir: defaultGemma, connectorPath: defaultConnector)
} else if args.contains("--dit-tiny-gate") {
    try ditTinyGate()
} else if args.contains("--dit-full-gate") {
    try ditFullGate()
} else if args.contains("--vae-decode-gate") {
    try vaeDecodeGate()
} else if args.contains("--vae-encode-gate") {
    try vaeEncodeGate()
} else {
    print("usage: RunLTX2 --connector-gate | --gemma-gate | --text-encode-gate | --dit-tiny-gate  [goldens.safetensors] [path]")
}
