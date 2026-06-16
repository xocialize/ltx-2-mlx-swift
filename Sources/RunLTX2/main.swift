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
} else {
    print("usage: RunLTX2 --connector-gate | --gemma-gate | --text-encode-gate  [goldens.safetensors] [path]")
}
