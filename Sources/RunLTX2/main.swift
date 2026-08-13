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
import MLXFast
import MLXRandom
import LTX2
import MLXLTX2
import MLXToolKit

// Derived from this source file's location (Sources/RunLTX2/main.swift → package root)
// so the gates keep working when the tree is relocated — the old absolute path went
// stale when LTX_DEV moved off ~/Development.
let packageRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()   // Sources/RunLTX2
    .deletingLastPathComponent()   // Sources
    .deletingLastPathComponent()   // <package root>
let defaultGoldens = packageRoot
    .appendingPathComponent("parity/goldens/text_encode/goldens.safetensors").path
let goldensBase = packageRoot.appendingPathComponent("parity/goldens").path
let defaultConnector = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/connector.safetensors"
let defaultGemma = "/Volumes/Satechi/Models/mlx-community/gemma-3-12b-it-4bit"

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
    let states = try encoder.allHiddenStates(tokenIds: tokenIds, attentionMask: mask)
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

// MARK: - Tokenizer gate
//
// `--gemma-gate` reads token_ids/attention_mask FROM the golden and feeds them straight to the
// 49-state forward, so `GemmaEncoder.tokenize` is exercised by NO other gate on the board — a
// text encoder can gate at cosine 1.0 while being fed wrongly-tokenized input in production
// (AB-L-0011). This gate closes that hole with INTEGER EQUALITY, not cosine: any drift in
// special-token handling, truncation side, or padding placement is a hard fail.
//
//   xcrun swift run RunLTX2 --gemma-tokenizer-gate [goldensDir] [gemmaDir]

private struct TokenizerGoldenCase: Decodable {
    let index: Int
    let name: String
    let prompt: String
    let rawTokenCount: Int
    let keptTokenCount: Int
    let truncated: Bool
    let nValid: Int
    let nPad: Int
    let firstKeptId: Int?
    let startsWithBos: Bool
}

private struct TokenizerGoldenMeta: Decodable {
    let maxLength: Int
    let padTokenId: Int
    let unkTokenId: Int?
    let bosTokenId: Int?
    let cases: [TokenizerGoldenCase]
}

func gemmaTokenizerGate(goldensDir: String, gemmaDir: String) async throws {
    let dir = URL(fileURLWithPath: goldensDir)
    print("[gemma-tokenizer-gate] goldens: \(dir.path)")
    print("[gemma-tokenizer-gate] gemma:   \(gemmaDir)")

    let goldens = try MLX.loadArrays(url: dir.appendingPathComponent("goldens.safetensors"))
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let meta = try decoder.decode(
        TokenizerGoldenMeta.self,
        from: Data(contentsOf: dir.appendingPathComponent("prompts.json")))

    // Tokenizer only — no 12B weights. Same object production gets from `#huggingFaceLoadModel`.
    let tk = try await GemmaEncoder.loadTokenizer(directory: URL(fileURLWithPath: gemmaDir))
    let swiftPadToken = tk.unknownTokenId ?? 0
    print("[gemma-tokenizer-gate] maxLength=\(meta.maxLength)  oracle pad=\(meta.padTokenId)  "
          + "swift pad=\(swiftPadToken)  bos=\(meta.bosTokenId.map(String.init) ?? "nil")")

    var failures: [String] = []
    var failedCases = 0

    for c in meta.cases {
        let key = String(format: "case%02d", c.index)
        guard let expIds = goldens["\(key)_token_ids"], let expMask = goldens["\(key)_attention_mask"] else {
            fatalError("goldens missing \(key)_token_ids / \(key)_attention_mask")
        }
        let (gotIdsArr, gotMaskArr) = GemmaEncoder.tokenize(c.prompt, tokenizer: tk, maxLength: meta.maxLength)
        let gotIds = gotIdsArr.asType(.int32).reshaped(-1).asArray(Int32.self)
        let gotMask = gotMaskArr.asType(.int32).reshaped(-1).asArray(Int32.self)
        let wantIds = expIds.asType(.int32).reshaped(-1).asArray(Int32.self)
        let wantMask = expMask.asType(.int32).reshaped(-1).asArray(Int32.self)

        var problems: [String] = []

        // 0. Lengths first. zip() truncates to the shorter sequence, so a pure LENGTH
        //    disagreement — exactly what a maxLength/truncation regression produces — finds
        //    no differing pair, leaves the `?? -1` fallback in place, and index-traps on the
        //    next line. A gate that fatals is indistinguishable from a broken build.
        if gotMask.count != wantMask.count || gotIds.count != wantIds.count {
            problems.append("length mismatch: mask \(gotMask.count) vs \(wantMask.count), "
                            + "ids \(gotIds.count) vs \(wantIds.count) — regenerate the goldens "
                            + "(prompts.json maxLength and goldens.safetensors disagree)")
        }

        // 1. Attention mask — exact, every slot. Catches a padding-side or truncation-side flip.
        if problems.isEmpty, gotMask != wantMask {
            let first = zip(gotMask, wantMask).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? -1
            problems.append("attention_mask differs (first at \(first): got \(gotMask[first]) want \(wantMask[first]))")
        }
        let gotValid = gotMask.reduce(0) { $0 + Int($1) }
        if gotValid != c.nValid { problems.append("valid-token count \(gotValid) != \(c.nValid)") }

        // 2. Token ids at VALID positions — exact. This is the real tokenization surface:
        //    a missing/extra BOS, a different truncation side, or any BPE drift lands here.
        // Guarded by the length check above; indexing wantMask over gotMask's range is only
        // safe once the two are known equal.
        let validIdx = problems.isEmpty ? (0 ..< gotMask.count).filter { wantMask[$0] == 1 } : []
        let gotValidIds = validIdx.map { gotIds[$0] }
        let wantValidIds = validIdx.map { wantIds[$0] }
        if gotValidIds != wantValidIds {
            let d = zip(gotValidIds, wantValidIds).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? -1
            problems.append("token ids differ at valid position \(d): "
                            + "got \(gotValidIds[d]) want \(wantValidIds[d])")
        }
        // Explicit BOS report — the 2.5 tokenizer's post_processor emits NO BOS, so this line is
        // the tripwire when the Gemma-4 encoder lands (its `single: [Sequence A]` means BOS must
        // be prepended by hand). Reported per case, gated by the id comparison above.
        let gotFirstValid = validIdx.first.map { gotIds[$0] }
        if gotFirstValid.map({ Int($0) }) != c.firstKeptId {
            problems.append("first valid token \(gotFirstValid.map(String.init) ?? "nil") "
                            + "!= oracle \(c.firstKeptId.map(String.init) ?? "nil")")
        }

        // 3. Token ids at PAD positions — must be uniformly the port's pad token. The DIVERGENCE
        //    from the oracle's `<pad>` is deliberate and measured benign (see GemmaEncoder.swift
        //    `tokenize` + parity/probe_pad_token_effect.py); what must not drift is that every
        //    pad slot carries one real vocab token and is masked off.
        let padIdx = (0 ..< gotMask.count).filter { wantMask[$0] == 0 }
        if let bad = padIdx.first(where: { gotIds[$0] != Int32(swiftPadToken) }) {
            problems.append("pad slot \(bad) is \(gotIds[bad]), expected the port's pad token \(swiftPadToken)")
        }

        let name = c.name.padding(toLength: max(14, c.name.count), withPad: " ", startingAt: 0)
        print("[gemma-tokenizer-gate] \(String(format: "%02d", c.index)) \(name) "
              + "raw=\(c.rawTokenCount) kept=\(c.keptTokenCount) valid=\(c.nValid) pad=\(c.nPad) "
              + "trunc=\(c.truncated ? "Y" : "N") bos=\(c.startsWithBos ? "Y" : "N") "
              + (problems.isEmpty ? "✅" : "❌"))
        if !problems.isEmpty { failedCases += 1 }
        for p in problems {
            print("[gemma-tokenizer-gate]      ↳ \(p)")
            failures.append("\(c.name): \(p)")
        }
    }

    let padNote = swiftPadToken == meta.padTokenId
        ? "pad token matches the oracle (\(meta.padTokenId))"
        : "pad token \(swiftPadToken) != oracle \(meta.padTokenId) — DELIBERATE, measured "
          + "bit-irrelevant to the DiT-facing embeds (parity/goldens/pad_token_probe/report.json)"
    print("[gemma-tokenizer-gate] \(padNote)")
    print("[gemma-tokenizer-gate] \(meta.cases.count - failedCases)/\(meta.cases.count) cases exact")
    let pass = failures.isEmpty
    print(pass ? "[gemma-tokenizer-gate] PASS ✅" : "[gemma-tokenizer-gate] FAIL ❌")
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
    // command buffers — cold pages off slow/external storage exceed the watchdog deterministically; the
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
        states = try encoder.allHiddenStates(tokenIds: tokenIds, attentionMask: mask)
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


/// AB-L-0003 audit: the sigma feeding the SINUSOIDAL timestep embedding must stay fp32.
///
/// The Python oracle shipped a real bug here for months — the sampler built sigma as
/// bf16 and the model cast it to fp32 again, which recovers nothing. Rounding sigma
/// before the ×1000 multiply shifts the embedding argument, and the high-frequency
/// dimensions of a sinusoidal embedding have period ~1, so those dims are scrambled
/// (measured: worst-step velocity cosine 0.855 → 0.990 once fixed).
///
/// The Swift stack is structurally immune, and this gate pins the REASONS so a future
/// edit cannot quietly remove them:
///   1. the scalar path — `MLXArray([sigma])` from a Swift `Float` — is fp32,
///   2. the per-token path — `denoiseMask * sigma` — is fp32, which depends on
///      `MLXArray.zeros/ones` defaulting to `Float.self`,
///   3. bf16 would MEASURABLY differ on the real distilled schedule.
///
/// ⚠️ (3) is swept over the ACTUAL sigma schedule rather than one hand-picked value.
/// A first draft used sigma = 0.7 and read a delta of exactly ZERO — not because bf16
/// is harmless there, but because bf16(0.7) = 0.69921875 and 699.21875 rounds BACK to
/// 700.0 (bf16 spacing near 700 is 4). The two roundings cancel at that value. A gate
/// probing a single point can therefore "prove" immunity it has not tested.
func timestepDtypeGate() throws {
    var fails: [String] = []

    let mask = MLX.concatenated([MLXArray.zeros([1, 4, 1]), MLXArray.ones([1, 8, 1])], axis: 1)
    if mask.dtype != .float32 { fails.append("denoise mask dtype is \(mask.dtype), expected .float32") }

    var maxDelta: Float = 0
    var worstSigma: Float = 0
    for sigma in Positions.distilledSigmas where sigma > 0 {
        let scalar = MLXArray([sigma])
        if scalar.dtype != .float32 {
            fails.append("scalar sigma dtype is \(scalar.dtype) at σ=\(sigma), expected .float32")
        }
        let perToken = (mask * sigma).squeezed(axis: -1)
        if perToken.dtype != .float32 {
            fails.append("per-token sigma dtype is \(perToken.dtype) at σ=\(sigma), expected .float32")
        }
        let fp32Arg = (scalar * 1000.0).asType(.float32).item(Float.self)
        let bf16Arg = (scalar.asType(.bfloat16) * 1000.0).asType(.float32).item(Float.self)
        let d = Swift.abs(fp32Arg - bf16Arg)
        if d > maxDelta { maxDelta = d; worstSigma = sigma }
    }

    print(String(format: "[timestep-dtype-gate] scalar/perToken/mask all fp32 | worst bf16 drift σ=%.6f → arg off by %.3f",
                 worstSigma, maxDelta))
    if maxDelta == 0 {
        fails.append("no sigma in the distilled schedule discriminates bf16 from fp32 — "
                   + "this gate cannot detect a regression and must be re-designed")
    }
    if fails.isEmpty {
        print("[timestep-dtype-gate] PASS ✅  sigma stays fp32 into the sinusoidal embedding")
        fflush(stdout)
    } else {
        for f in fails { print("[timestep-dtype-gate] FAIL — \(f)") }
        fflush(stdout)   // a thrown error kills the process before stdout drains
        exit(1)
    }
}


/// LTX-2.5 DiT delta parity: the learned KEYFRAMES absolute-position embedding.
///
/// Same tiny-model shape as `ditTinyGate`, but the fixture has
/// `use_keyframes_abs_pos_embedding=True` and `ff_bias=False` — the two config-driven
/// 2.5 deltas. `ff_bias=False` needs NO Swift change: `dense()` adds a bias only when
/// the key exists, so a 2.5 checkpoint simply has none. This gate proves that too, by
/// running weights that lack the video FF biases.
func ditTinyKF25Gate() throws {
    let dir = "\(goldensBase)/dit_tiny_kf25"
    let weights = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/weights.safetensors"))
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    guard weights["keyframes_abs_pos_embedding"] != nil else {
        print("[dit-tiny-kf25-gate] FAIL — fixture has no keyframes_abs_pos_embedding"); fflush(stdout); exit(1)
    }
    let mask = io["keyframes_mask"]!
    let marked = mask.sum().item(Float.self), total = Float(mask.dim(1))
    guard marked > 0, marked < total else {
        print("[dit-tiny-kf25-gate] FAIL — mask marks \(marked)/\(total) tokens; an all-on or "
            + "all-off mask would pass even if the embedding hit the wrong tokens")
        fflush(stdout); exit(1)
    }
    print("[dit-tiny-kf25-gate] \(weights.count) weight tensors, mask marks \(Int(marked))/\(Int(total)) tokens")

    let dit = DiT(weights: weights, config: tinyDiTConfig())
    let (video, audio) = dit(
        videoLatent: io["video_latent"]!, audioLatent: io["audio_latent"]!, sigma: io["sigma"]!,
        videoText: io["video_text"]!, audioText: io["audio_text"]!,
        videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!,
        videoTimesteps: nil, audioTimesteps: nil, keyframesMask: mask)
    let vCos = cosine(video, io["video_v"]!), vMax = maxAbs(video, io["video_v"]!)
    let aCos = cosine(audio, io["audio_v"]!), aMax = maxAbs(audio, io["audio_v"]!)
    print(String(format: "[dit-tiny-kf25-gate] VIDEO cosine=%.6f maxAbs=%.5f", vCos, vMax))
    print(String(format: "[dit-tiny-kf25-gate] AUDIO cosine=%.6f maxAbs=%.5f", aCos, aMax))

    // Discrimination: WITHOUT the mask the embedding must not be applied, so the video
    // output must MOVE. If it did not, this gate would pass on a port that ignores the
    // mask entirely — the exact failure it exists to catch.
    let (videoNoKF, _) = dit(
        videoLatent: io["video_latent"]!, audioLatent: io["audio_latent"]!, sigma: io["sigma"]!,
        videoText: io["video_text"]!, audioText: io["audio_text"]!,
        videoPositions: io["video_positions"]!, audioPositions: io["audio_positions"]!,
        videoTimesteps: nil, audioTimesteps: nil, keyframesMask: nil)
    let drift = maxAbs(videoNoKF, io["video_v"]!)
    print(String(format: "[dit-tiny-kf25-gate] mask-off drift vs golden=%.5f (must be >> 0)", drift))

    var ok = vCos >= 0.9999 && aCos >= 0.9999
    if drift <= vMax {
        print("[dit-tiny-kf25-gate] FAIL — dropping the mask changes the output by \(drift), no more "
            + "than the parity error \(vMax); this gate cannot detect a port that ignores the mask")
        ok = false
    }
    if ok {
        print("[dit-tiny-kf25-gate] PASS ✅"); fflush(stdout)
    } else {
        print("[dit-tiny-kf25-gate] FAIL ❌"); fflush(stdout); exit(1)
    }
}


/// LTX-2.5 ancestral-Euler step parity (pure sampler math, INJECTED noise).
///
/// Noise is injected from the fixture rather than drawn: MLX-Swift and MLX-Python RNG
/// streams are not bit-identical (determinism doctrine / ISSUES I1), so a seeded gate
/// would be measuring the RNG instead of the step being ported.
func ancestralStepGate() throws {
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(goldensBase)/ancestral_step/io.safetensors"))
    let x = io["x"]!, x0 = io["x0"]!, noise = io["noise"]!
    let caseSigmas = io["case_sigmas"]!
    let n = Int(io["n_cases"]!.item(Float.self))
    var worst: Float = 0, worstCase = ""
    var zeroBoundaryChecked = false

    for i in 0 ..< n {
        let sigma = caseSigmas[i, 0].item(Float.self)
        let sigmaNext = caseSigmas[i, 1].item(Float.self)
        let eta = caseSigmas[i, 2].item(Float.self)
        let got = DenoiseLoop.eulerAncestralStep(x, x0, sigma: sigma, sigmaNext: sigmaNext,
                                                 noise: noise, eta: eta, sNoise: 1.0)
        let d = maxAbs(got, io["eta1_\(i)"]!)
        if d > worst { worst = d; worstCase = String(format: "σ %.5f→%.5f", sigma, sigmaNext) }
        if sigmaNext == 0 {
            zeroBoundaryChecked = true
            // σ_next == 0 must return x0 EXACTLY, not approximately.
            if maxAbs(got, x0) != 0 {
                print("[ancestral-step-gate] FAIL — σ_next=0 did not return x0 exactly")
                fflush(stdout); exit(1)
            }
        }
    }
    print(String(format: "[ancestral-step-gate] %d eta=1 cases, worst maxAbs=%.3e (%@)",
                 n, worst, worstCase as NSString))

    // eta = 0 must reduce to the plain Euler step — the property, not just the golden.
    let s = io["meta_eta0_sigma"]![0].item(Float.self)
    let sn = io["meta_eta0_sigma"]![1].item(Float.self)
    let eta0 = DenoiseLoop.eulerAncestralStep(x, x0, sigma: s, sigmaNext: sn,
                                              noise: noise, eta: 0.0, sNoise: 1.0)
    let vsGolden = maxAbs(eta0, io["eta0"]!)
    let vsPlain = maxAbs(eta0, DenoiseLoop.eulerStep(x, x0, sigma: s, sigmaNext: sn))
    print(String(format: "[ancestral-step-gate] eta=0: vs golden %.3e, vs plain euler %.3e",
                 vsGolden, vsPlain))

    // Discrimination: eta=1 must actually DIFFER from eta=0, or the gate proves nothing
    // about the ancestral branch (which is the entire 2.5 delta).
    let eta1 = DenoiseLoop.eulerAncestralStep(x, x0, sigma: s, sigmaNext: sn,
                                              noise: noise, eta: 1.0, sNoise: 1.0)
    let branchDelta = maxAbs(eta1, eta0)
    print(String(format: "[ancestral-step-gate] eta=1 vs eta=0 delta=%.4f (must be >> tolerance)", branchDelta))

    let tol: Float = 1e-5
    var ok = worst < tol && vsGolden < tol && vsPlain < 1e-4 && zeroBoundaryChecked
    if branchDelta <= tol {
        print("[ancestral-step-gate] FAIL — the ancestral branch changes nothing at these sigmas; "
            + "this gate cannot detect a port that ignores eta")
        ok = false
    }
    if !zeroBoundaryChecked {
        print("[ancestral-step-gate] FAIL — no σ_next=0 case in the fixture; the boundary is untested")
    }
    if ok { print("[ancestral-step-gate] PASS ✅"); fflush(stdout) }
    else { print("[ancestral-step-gate] FAIL ❌"); fflush(stdout); exit(1) }
}


/// Wiring gate: the LTX-2.5 deltas must reach production through DenoiseLoop, not just
/// through DiT and the step function.
///
/// ⚠️ This gate exists because the component gates could not see the real defect. An earlier
/// revision implemented keyframesMask on DiT/TiledDiT and eulerAncestralStep on DenoiseLoop,
/// gated both (--dit-tiny-kf25-gate at cosine 0.999999, --ancestral-step-gate at 9.3e-06),
/// and shipped with NEITHER reachable from `run`/`runConditioned` — the only entry points the
/// pipeline uses. Both gates stayed green. A gate that calls a component directly proves the
/// component works and says nothing about whether anything calls it.
///
/// So this one drives the LOOP and asserts each delta CHANGES the result.
func denoiseWiringGate() throws {
    let dir = "\(goldensBase)/dit_tiny_kf25"
    let weights = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/weights.safetensors"))
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let dit = DiT(weights: weights, config: tinyDiTConfig())
    let v = io["video_latent"]!, a = io["audio_latent"]!
    let vp = io["video_positions"]!, ap = io["audio_positions"]!
    let vt = io["video_text"]!, at = io["audio_text"]!
    let mask = io["keyframes_mask"]!
    let sigmas: [Float] = [1.0, 0.5, 0.0]
    var fails: [String] = []

    func loop(mask m: MLXArray?, eta: Float) throws -> MLXArray {
        try DenoiseLoop.run(
            dit: dit, videoLatent0: v, audioLatent0: a, sigmas: sigmas,
            videoText: vt, audioText: at, videoPositions: vp, audioPositions: ap,
            keyframesMask: m, ancestralEta: eta, ancestralNoiseSeed: 7).video
    }

    let base = try loop(mask: nil, eta: 0)
    let withMask = try loop(mask: mask, eta: 0)
    let withEta = try loop(mask: nil, eta: 1.0)
    eval(base, withMask, withEta)

    let maskDelta = maxAbs(withMask, base)
    let etaDelta = maxAbs(withEta, base)
    print(String(format: "[denoise-wiring-gate] keyframesMask delta=%.5f   ancestralEta delta=%.5f",
                 maskDelta, etaDelta))
    if maskDelta == 0 {
        fails.append("keyframesMask does not reach the DiT through DenoiseLoop.run — the 2.5 "
                   + "keyframes embedding would be silently absent in production")
    }
    if etaDelta == 0 {
        fails.append("ancestralEta does not change the trajectory — DenoiseLoop.run is still "
                   + "taking the plain Euler branch")
    }

    // runConditioned must thread them too; it is the i2v entry point.
    let condMask = try DenoiseLoop.runConditioned(
        dit: dit, videoLatent0: v, audioLatent0: a, sigmas: sigmas,
        videoText: vt, audioText: at, videoPositions: vp, audioPositions: ap,
        keyframesMask: mask, ancestralEta: 0, ancestralNoiseSeed: 7).video
    let condBase = try DenoiseLoop.runConditioned(
        dit: dit, videoLatent0: v, audioLatent0: a, sigmas: sigmas,
        videoText: vt, audioText: at, videoPositions: vp, audioPositions: ap,
        keyframesMask: nil, ancestralEta: 0, ancestralNoiseSeed: 7).video
    eval(condMask, condBase)
    let condDelta = maxAbs(condMask, condBase)
    print(String(format: "[denoise-wiring-gate] runConditioned mask delta=%.5f", condDelta))
    if condDelta == 0 {
        fails.append("keyframesMask does not reach the DiT through DenoiseLoop.runConditioned")
    }

    if fails.isEmpty {
        print("[denoise-wiring-gate] PASS ✅  both 2.5 deltas reach the DiT through the loops")
        fflush(stdout)
    } else {
        for f in fails { print("[denoise-wiring-gate] FAIL — \(f)") }
        fflush(stdout); exit(1)
    }
}


/// LTX-2.5 Gemma-4 49-state parity, per state, against the oracle.
///
/// The per-state breakdown localises WHICH of the three Gemma-3→Gemma-4 deltas broke:
///   * a wrong embed-scale dtype moves state 00,
///   * a wrong final norm moves ONLY state 48,
///   * a wrong norm convention inside the stack moves everything from 01 on.
/// Loads the 23.8 GB bf16 encoder, so this is a machine-with-headroom gate.
func gemma4Gate() async throws {
    let dir = "\(goldensBase)/gemma4"
    let g = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/goldens.safetensors"))
    let meta = try JSONSerialization.jsonObject(
        with: Data(contentsOf: URL(fileURLWithPath: "\(dir)/meta.json"))) as! [String: Any]
    let gemmaDir = ProcessInfo.processInfo.environment["LTX_GEMMA4_DIR"]
        ?? "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx/gemma4-12b-ltx-v1"
    print("[gemma4-gate] encoder: \(gemmaDir)")

    let encoder = try await Gemma4Encoder.load(directory: URL(fileURLWithPath: gemmaDir))
    let tokenIds = g["token_ids"]!, mask = g["attention_mask"]!
    let states = try encoder.allHiddenStates(tokenIds: tokenIds, attentionMask: mask)
    eval(states)

    let nStates = meta["n_states"] as? Int ?? 49
    guard states.count == nStates else {
        print("[gemma4-gate] FAIL — got \(states.count) states, want \(nStates)")
        fflush(stdout); exit(1)
    }

    var worstCos: Float = 1, worstIdx = 0, sumCos: Float = 0, worstMax: Float = 0
    for i in 0 ..< states.count {
        let exp = g[String(format: "gemma_hidden_%02d", i)]!
        let c = cosine(states[i], exp), m = maxAbs(states[i], exp)
        sumCos += c; worstMax = Swift.max(worstMax, m)
        if c < worstCos { worstCos = c; worstIdx = i }
        if i < 2 || i == states.count - 1 {
            print(String(format: "[gemma4-gate] state %02d cosine=%.6f maxAbs=%.4f", i, c, m))
        }
    }
    print(String(format: "[gemma4-gate] mean=%.6f  worst=%.6f (state %d)  maxAbs=%.4f",
                 sumCos / Float(states.count), worstCos, worstIdx, worstMax))

    // Discrimination: the final norm must actually do something, or a tap that skipped it
    // would pass. The fixture reports |state48 - state47| ~3088, so this is a real check.
    let finalNormDelta = maxAbs(states[states.count - 1], states[states.count - 2])
    print(String(format: "[gemma4-gate] |state48 - state47| = %.2f (final norm applied)", finalNormDelta))

    var ok = worstCos >= 0.999
    if finalNormDelta < 1.0 {
        print("[gemma4-gate] FAIL — states 47 and 48 are nearly identical; the final norm is "
            + "not being applied, and this gate could not tell")
        ok = false
    }
    if ok { print("[gemma4-gate] PASS ✅"); fflush(stdout) }
    else { print("[gemma4-gate] FAIL ❌"); fflush(stdout); exit(1) }
}


/// LTX-2.5 keyframe slots: structure, RoPE spans, and the seeded re-append.
///
/// Compared against the ORACLE's own numbers where they are exact integers or closed-form
/// (token counts, mask marking, temporal spans, evenly-spaced positions), rather than a
/// dumped golden — slots are integer/geometry work, so an exact check beats a tolerance.
func keyframeSlotsGate() throws {
    let F = 4, H = 4, W = 6, C = 128
    let fps: Float = 24.0
    let N = F * H * W
    var fails: [String] = []

    let latent = MLXRandom.normal([1, N, C])
    let clean = MLXArray.zeros([1, N, C])
    let dmask = MLXArray.ones([1, N, 1])
    let pos = Positions.video(F: F, H: H, W: W, fps: fps)

    let st = KeyframeSlots.append(
        latent: latent, clean: clean, denoiseMask: dmask, positions: pos,
        keyframesMask: nil, pixelFrameIndices: [48, 96], H: H, W: W, fps: fps,
        sigma: 1.0, noiseSeed: 7)
    eval(st.latent, st.keyframesMask, st.positions)

    // 1. token counts
    let want = N + 2 * H * W
    if st.latent.dim(1) != want { fails.append("token count \(st.latent.dim(1)) != \(want)") }
    if st.keyframesMask.dim(1) != st.latent.dim(1) {
        fails.append("keyframesMask length desynced from the sequence")
    }
    // 2. slots are DENOISED and MARKED; base is not marked
    let slotDenoise = st.denoiseMask[0..., N..., 0...].min().item(Float.self)
    let slotMark = st.keyframesMask[0..., N..., 0...].min().item(Float.self)
    let baseMark = st.keyframesMask[0..., ..<N, 0...].max().item(Float.self)
    if slotDenoise != 1 { fails.append("slot denoiseMask \(slotDenoise) != 1 — slots must be denoised") }
    if slotMark != 1 { fails.append("slot keyframesMask \(slotMark) != 1 — slots must be marked") }
    if baseMark != 0 { fails.append("base tokens are marked (\(baseMark)); only slots should be") }
    print(String(format: "[keyframe-slots-gate] tokens %d -> %d, slot denoise=%.0f mark=%.0f base mark=%.0f",
                 N, st.latent.dim(1), slotDenoise, slotMark, baseMark))

    // 3. RoPE spans: slot temporal coord must be exactly (t + 0.5) / fps, NOT an 8-frame span.
    let slotT = st.positions[0, N..., 0].asArray(Float.self)
    for (i, t) in [48, 96].enumerated() {
        let expect = (Float(t) + 0.5) / fps
        let got = slotT[i * H * W]
        if abs(got - expect) > 1e-4 {
            fails.append(String(format: "slot %d temporal %.6f != %.6f", i, got, expect))
        }
    }
    let baseSpan = st.positions[0, ..<N, 0].asArray(Float.self)
    let spanWidth = (baseSpan.max() ?? 0) - (baseSpan.min() ?? 0)
    print(String(format: "[keyframe-slots-gate] slot t = %.5f / %.5f (want %.5f / %.5f); base spans %.4f",
                 slotT[0], slotT[H * W], (48 + Float(0.5)) / fps, (96 + Float(0.5)) / fps, spanWidth))

    // 4. seeded re-append: seed must SURVIVE at small sigma and VANISH at sigma 1.
    let seed = MLXRandom.normal([1, C, 2, H, W])
    func slotBlock(sigma: Float) -> MLXArray {
        let s2 = KeyframeSlots.append(
            latent: latent, clean: clean, denoiseMask: dmask, positions: pos,
            keyframesMask: nil, pixelFrameIndices: [48, 96], H: H, W: W, fps: fps,
            sigma: sigma, noiseSeed: 7, initialKeyframes: seed)
        return s2.latent[0..., N..., 0...]
    }
    let seedTokens = seed.transposed(0, 2, 3, 4, 1).reshaped(1, 2 * H * W, C)
    let cosLow = cosine(slotBlock(sigma: 0.05), seedTokens)
    let cosHigh = cosine(slotBlock(sigma: 1.0), seedTokens)
    print(String(format: "[keyframe-slots-gate] seed survival: cos@sigma0.05=%.4f  cos@sigma1.0=%.4f",
                 cosLow, cosHigh))
    if cosLow < 0.99 { fails.append("seed does not survive at sigma 0.05 (cos \(cosLow)) — stage-2 re-append would re-invent the slot") }
    if abs(cosHigh) > 0.2 { fails.append("seed still present at sigma 1.0 (cos \(cosHigh)) — stage 1 must be pure noise") }

    // 5. evenly-spaced positions must match the oracle's banker's rounding.
    // Oracle-verified pairs (from gate_ltx25_slot_positions.py's 4482-case sweep).
    let cases: [(Int, Int, [Int])] = [(1, 97, [48]), (3, 97, [24, 48, 72]), (2, 25, [8, 16])]
    for (k, n, expect) in cases {
        let got = KeyframeSlots.evenlySpacedPositions(k, numFrames: n)
        if got != expect { fails.append("evenlySpaced(\(k), \(n)) = \(got) != \(expect)") }
    }

    // 6. extract must return ONE latent per slot, never a stacked clip.
    let kfs = st.extract(st.latent, H: H, W: W)
    if kfs.count != 2 { fails.append("extract returned \(kfs.count) latents, want 2") }
    if let first = kfs.first, first.dim(2) != 1 {
        fails.append("extracted keyframe has T=\(first.dim(2)); each must be a standalone one-frame latent")
    }
    print("[keyframe-slots-gate] extract -> \(kfs.count) x \(kfs.first.map { $0.shape } ?? [])")

    if fails.isEmpty { print("[keyframe-slots-gate] PASS ✅"); fflush(stdout) }
    else { for f in fails { print("[keyframe-slots-gate] FAIL — \(f)") }; fflush(stdout); exit(1) }
}


/// LTX-2.5 pipeline wiring: version detection and the first-latent-frame mask.
///
/// Checks detection against the REAL model directories on disk, not a synthetic fixture —
/// the failure this guards is "the pipeline silently treats a 2.5 checkpoint as 2.3", which
/// a fixture with a hand-made directory would not exercise.
func pipeline25Gate() throws {
    var fails: [String] = []
    let dirs: [(String, String, Bool)] = [
        ("2.5", "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx", true),
        ("2.3", "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx", false),
    ]
    for (label, path, want) in dirs {
        guard FileManager.default.fileExists(atPath: path) else {
            print("[pipeline-25-gate] SKIP \(label): \(path) not present"); continue
        }
        let got = LTX2Pipeline.isLTX25(ltxDir: URL(fileURLWithPath: path))
        print("[pipeline-25-gate] \(label) dir -> isLTX25=\(got) (want \(want))")
        if got != want { fails.append("\(label) detected as isLTX25=\(got), want \(want)") }
    }

    // Mask shape + marking: frame 0 marked, everything else not.
    let F = 4, H = 4, W = 6
    let m = LTX2Pipeline.firstLatentFrameKeyframesMask(totalTokens: F * H * W,
                                                       tokensPerLatentFrame: H * W)
    eval(m)
    let head = m[0..., ..<(H * W), 0...].min().item(Float.self)
    let tail = m[0..., (H * W)..., 0...].max().item(Float.self)
    print(String(format: "[pipeline-25-gate] mask %@ head=%.0f tail=%.0f", "\(m.shape)" as NSString, head, tail))
    if m.dim(1) != F * H * W { fails.append("mask length \(m.dim(1)) != \(F * H * W)") }
    if head != 1 { fails.append("first latent frame not marked (min \(head))") }
    if tail != 0 { fails.append("tokens beyond frame 0 are marked (max \(tail))") }

    if fails.isEmpty { print("[pipeline-25-gate] PASS ✅"); fflush(stdout) }
    else { for f in fails { print("[pipeline-25-gate] FAIL — \(f)") }; fflush(stdout); exit(1) }
}

/// Small-scale DiT parity: tiny seeded LTXModel forward vs oracle goldens.
func ditTinyGate() throws {
    let dir = "\(goldensBase)/dit_tiny"
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
    let dir = "\(goldensBase)/dit_full"
    let weightsPath = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/transformer-distilled.safetensors"
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
    let dir = "\(goldensBase)/vae_decode"
    let weightsPath = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/vae_decoder.safetensors"
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

/// PrunaVAED (pruned decoder) parity: latent → pixels vs oracle golden (fp32).
/// Exercises the channel-adapter blocks, which the stock decoder does not have.
/// Also checks the diffusers reference dump when present — that is the independent
/// ground truth for the adapter, since oracle and port were written from one reading.
func vaeDecodePrunaGate() throws {
    let dir = "\(goldensBase)/vae_decode_pruna"
    let weightsPath = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/vae_decoder_pruna.safetensors"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let dec = try VideoVAEDecoder.load(path: URL(fileURLWithPath: weightsPath))
    let pixels = dec.decode(io["latent"]!)
    eval(pixels)
    let cos = cosine(pixels, io["pixels"]!), m = maxAbs(pixels, io["pixels"]!)
    print(String(format: "[vae-decode-pruna-gate] cosine=%.6f maxAbs=%.5f  shape %@ vs %@", cos, m, "\(pixels.shape)" as NSString, "\(io["pixels"]!.shape)" as NSString))

    let refURL = URL(fileURLWithPath: "\(dir)/reference.safetensors")
    if FileManager.default.fileExists(atPath: refURL.path),
       let ref = try? MLX.loadArrays(url: refURL)["pixels"] {
        print(String(format: "[vae-decode-pruna-gate] vs diffusers reference: cosine=%.6f maxAbs=%.5f", cosine(pixels, ref), maxAbs(pixels, ref)))
    }

    let pass = cos >= 0.999
    print(pass ? "[vae-decode-pruna-gate] PASS ✅" : "[vae-decode-pruna-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Stock-vs-PrunaVAED decode A/B: wall time + phys_footprint peak at shipping tiers.
///
/// Upstream measured 1.68–2.08× on an H100 with ~50% lower peak VRAM; this is the Metal
/// number, which is the only one that governs our tier ladder. Decode's share of e2e has
/// never been measured on target hardware (the M1 stage-split table is still blank), so
/// this reports absolute seconds per decode, not just the ratio.
func vaeDecodeBench() throws {
    // (label, latentFrames, latentH, latentW) → pixels (8F−7, H·32, W·32).
    let tiers: [(String, Int, Int, Int)] = [
        ("512×288×121f", 16, 9, 16),
        ("704×512×121f", 16, 16, 22),
    ]
    let variants: [(String, String)] = [
        ("stock", "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/vae_decoder.safetensors"),
        ("pruna", "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/vae_decoder_pruna.safetensors"),
    ]
    let reps = 3
    let sampler = PhysSampler()
    sampler.start()
    defer { sampler.stop() }

    for (tier, f, h, wl) in tiers {
        print("\n[vae-decode-bench] tier \(tier)  latent (1,128,\(f),\(h),\(wl))")
        var best: [String: Double] = [:]
        for (name, path) in variants {
            let url = URL(fileURLWithPath: path)
            prewarmFiles([url])
            let dec = try VideoVAEDecoder.load(path: url)
            MLXRandom.seed(7)
            let latent = MLXRandom.normal([1, 128, f, h, wl]).asType(.float32)
            eval(latent)

            _ = { let p = dec.decode(latent); eval(p) }()   // warm the kernels
            Memory.clearCache()
            let floor = physFootprintBytes()
            sampler.resetMax()

            var times: [Double] = []
            for _ in 0 ..< reps {
                let t0 = Date()
                let px = dec.decode(latent)
                eval(px)
                times.append(Date().timeIntervalSince(t0))
            }
            let peak = sampler.maxBytes()
            let median = times.sorted()[reps / 2]
            best[name] = median
            print(String(
                format: "[vae-decode-bench]   %-5@ median %.3fs  (min %.3f max %.3f)  floor %.2f GB  peak %.2f GB  Δ %.2f GB",
                name as NSString, median, times.min()!, times.max()!,
                gbOf(floor), gbOf(peak), gbOf(peak > floor ? peak - floor : 0)
            ))
            Memory.clearCache()
        }
        if let s = best["stock"], let p = best["pruna"], p > 0 {
            print(String(format: "[vae-decode-bench]   → speedup %.2f×", s / p))
        }
    }
}

/// Video VAE encode parity: pixels → latent vs oracle golden (fp32).
func vaeEncodeGate() throws {
    let dir = "\(goldensBase)/vae_encode"
    let weightsPath = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/vae_encoder.safetensors"
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
    let tinyW = "\(goldensBase)/dit_tiny/weights.safetensors"
    let dir = "\(goldensBase)/dit_denoise"
    let weights = try MLX.loadArrays(url: URL(fileURLWithPath: tinyW))
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let dit = DiT(weights: weights, config: tinyDiTConfig())
    let sigmas = io["sigmas"]!.asArray(Float.self)
    let (video, audio) = try DenoiseLoop.run(
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

/// IC-LoRA reference-append parity (IC-LORA-PLAN P1): tiny DiT + oracle goldens from
/// dump_ic_tiny_goldens.py — the oracle's real VideoConditionByReferenceLatent.apply +
/// denoise_loop vs our ICVideoState.build + DenoiseLoop.runConditioned + slice.
/// Case a: strength 1.0, downscale 2 (position scaling); case b: strength 0.7, downscale 1
/// (partial denoise mask → per-token σ at reference tokens).
func icTinyGate() throws {
    let tinyW = "\(goldensBase)/dit_tiny/weights.safetensors"
    let dir = "\(goldensBase)/ic_tiny"
    let weights = try MLX.loadArrays(url: URL(fileURLWithPath: tinyW))
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let dit = DiT(weights: weights, config: tinyDiTConfig())
    let sigmas = io["sigmas"]!.asArray(Float.self)
    var allPass = true
    for name in ["a", "b"] {
        let p = io["\(name)_params"]!.asArray(Float.self)   // [downscale, strength, Nv, Nr]
        let ref = ReferenceConditioning(tokens: io["\(name)_ref_tokens"]!,
                                        positions: io["\(name)_ref_positions"]!,
                                        downscaleFactor: p[0], strength: p[1])
        let state = ICVideoState.build(targetLatent: io["\(name)_video_latent"]!,
                                       targetPositions: io["\(name)_video_positions"]!,
                                       references: [ref])
        let posMax = maxAbs(state.positions, io["\(name)_ext_positions"]!)
        let (vFull, audio) = try DenoiseLoop.runConditioned(
            dit: dit, videoLatent0: state.latent, audioLatent0: io["\(name)_audio_latent"]!,
            sigmas: sigmas,
            videoText: io["\(name)_video_text"], audioText: io["\(name)_audio_text"],
            videoPositions: state.positions, audioPositions: io["\(name)_audio_positions"]!,
            videoCleanLatent: state.clean, videoDenoiseMask: state.denoiseMask)
        let vSliced = state.slice(vFull)
        eval(vFull, audio)
        let fCos = cosine(vFull, io["\(name)_video_final_full"]!)
        let sCos = cosine(vSliced, io["\(name)_video_final"]!)
        let aCos = cosine(audio, io["\(name)_audio_final"]!)
        print(String(format: "[ic-tiny-gate] case %@ (ds=%.0f s=%.1f Nv=%.0f Nr=%.0f): posMaxAbs=%.6f  VIDEO full=%.6f sliced=%.6f  AUDIO=%.6f",
                     name as NSString, p[0], p[1], p[2], p[3], posMax, fCos, sCos, aCos))
        allPass = allPass && posMax < 1e-4 && fCos >= 0.999 && sCos >= 0.999 && aCos >= 0.999
    }
    // Case c — AUDIO reference appended (LipDub semantics: negative-time positions, video
    // unconditioned). Same append machinery on the audio state.
    do {
        let ref = ReferenceConditioning(tokens: io["c_ref_tokens"]!,
                                        positions: io["c_ref_audio_positions"]!,
                                        downscaleFactor: 1, strength: 1.0)
        let audioState = ICVideoState.build(targetLatent: io["c_audio_latent"]!,
                                            targetPositions: io["c_tgt_audio_positions"]!,
                                            references: [ref])
        let (video, afull) = try DenoiseLoop.runConditioned(
            dit: dit, videoLatent0: io["c_video_latent"]!, audioLatent0: audioState.latent,
            sigmas: sigmas,
            videoText: io["c_video_text"], audioText: io["c_audio_text"],
            videoPositions: io["c_video_positions"]!, audioPositions: audioState.positions,
            audioCleanLatent: audioState.clean, audioDenoiseMask: audioState.denoiseMask)
        let aSliced = audioState.slice(afull)
        eval(video, afull)
        let aFullCos = cosine(afull, io["c_audio_final_full"]!)
        let aCos = cosine(aSliced, io["c_audio_final"]!)
        let vCos = cosine(video, io["c_video_final"]!)
        print(String(format: "[ic-tiny-gate] case c (audio ref, neg-time): AUDIO full=%.6f sliced=%.6f  VIDEO=%.6f",
                     aFullCos, aCos, vCos))
        allPass = allPass && aFullCos >= 0.999 && aCos >= 0.999 && vCos >= 0.999
    }
    print(allPass ? "[ic-tiny-gate] PASS ✅" : "[ic-tiny-gate] FAIL ❌")
    if !allPass { exit(1) }
}

/// IC reference-ingest parity (IC-LORA-PLAN P2): a seeded still tiled 8k+1 → real VAE encoder →
/// tokens + positions, vs the oracle's iclora_utils encode glue. Also asserts the frame snap.
func icIngestGate() throws {
    // snapFrames unit checks (oracle: k = max(1,(f-1)//8) → 1+8k).
    let snaps = [(120, 113), (121, 121), (9, 9), (1, 9), (49, 49), (50, 49)]
    for (input, want) in snaps {
        let got = ReferenceConditioning.snapFrames(input)
        guard got == want else {
            print("[ic-ingest-gate] snapFrames(\(input)) = \(got), want \(want) FAIL ❌"); exit(1)
        }
    }
    let base = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx"
    let dir = "\(goldensBase)/ic_ingest"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let dims = io["dims"]!.asArray(Float.self)
    let (F, H, W) = (Int(dims[0]), Int(dims[1]), Int(dims[2]))
    let enc = try VideoVAEEncoder.load(path: URL(fileURLWithPath: "\(base)/vae_encoder.safetensors"))
    let video = MLX.broadcast(io["still"]!, to: [1, 3, F, H, W])   // same looped-still tiling
    let latent = enc.encode(video)
    let tokens = LTX2Pipeline.patchify(latent)
    let positions = Positions.video(F: latent.dim(2), H: latent.dim(3), W: latent.dim(4), fps: 24)
    eval(tokens, positions)
    let tCos = cosine(tokens, io["ref_tokens"]!), tMax = maxAbs(tokens, io["ref_tokens"]!)
    let pMax = maxAbs(positions, io["ref_positions"]!)
    print(String(format: "[ic-ingest-gate] tokens %@ cosine=%.6f maxAbs=%.5f  posMaxAbs=%.6f",
                 "\(tokens.shape)" as NSString, tCos, tMax, pMax))
    let pass = tCos >= 0.999 && pMax < 1e-4
    print(pass ? "[ic-ingest-gate] PASS ✅ (snapFrames 6/6)" : "[ic-ingest-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// End-to-end one-stage t2v parity (real weights): noise → denoise → unpatchify → VAE decode.
func e2eGate() throws {
    let base = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx"
    let dir = "\(goldensBase)/e2e_t2v"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    print("[e2e-gate] loading real DiT (bf16) + VAE decoder…")
    let dit = try DiT.load(weightsPath: URL(fileURLWithPath: "\(base)/transformer-distilled.safetensors"), config: DiTConfig(), computeDtype: .bfloat16)
    let dec = try VideoVAEDecoder.load(path: URL(fileURLWithPath: "\(base)/vae_decoder.safetensors"))
    let sigmas = io["sigmas"]!.asArray(Float.self)
    print("[e2e-gate] denoising \(sigmas.count - 1) steps…")
    let (vfinal, _) = try DenoiseLoop.run(
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
    let dir = "\(goldensBase)/audio_vae_decode"
    let weightsPath = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/audio_vae.safetensors"
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

/// Audio VAE encode parity (IC-LORA-PLAN P3b, LipDub): waveform → mel → latent vs oracle
/// golden (fp32), + the LipDub patchify/negative-positions helper on the golden latent.
func audioVaeEncodeGate() throws {
    let dir = "\(goldensBase)/audio_vae_encode"
    let weightsPath = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/audio_vae.safetensors"
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let enc = try AudioVAEEncoder.load(path: URL(fileURLWithPath: weightsPath))
    let processor = AudioMelProcessor()

    // Diagnostics: Swift-computed Slaney filterbank + Hann window vs the oracle's.
    let basisMax = maxAbs(processor.melBasis, io["mel_basis"]!)
    let windowMax = maxAbs(processor.window, io["window"]!)

    let mel = processor.waveformToMel(io["waveform"]!)
    let latent = enc.encode(mel)
    eval(mel, latent)
    let mCos = cosine(mel, io["mel"]!), mMax = maxAbs(mel, io["mel"]!)
    let lCos = cosine(latent, io["latent"]!), lMax = maxAbs(latent, io["latent"]!)
    print(String(format: "[audio-vae-encode-gate] basisMaxAbs=%.2e windowMaxAbs=%.2e", basisMax, windowMax))
    print(String(format: "[audio-vae-encode-gate] MEL    cosine=%.6f maxAbs=%.5f  shape %@ vs %@",
                 mCos, mMax, "\(mel.shape)" as NSString, "\(io["mel"]!.shape)" as NSString))
    print(String(format: "[audio-vae-encode-gate] LATENT cosine=%.6f maxAbs=%.5f  shape %@ vs %@",
                 lCos, lMax, "\(latent.shape)" as NSString, "\(io["latent"]!.shape)" as NSString))

    // LipDub helper on the GOLDEN latent (isolates the patchify/positions math).
    let (tokens, positions) = Positions.patchifyLipdubAudioReference(io["latent"]!)
    eval(tokens, positions)
    let tCos = cosine(tokens, io["ref_tokens"]!), tMax = maxAbs(tokens, io["ref_tokens"]!)
    let pMax = maxAbs(positions, io["ref_positions"]!)
    print(String(format: "[audio-vae-encode-gate] LIPDUB tokens cosine=%.6f maxAbs=%.5f  posMaxAbs=%.6f",
                 tCos, tMax, pMax))

    let pass = mCos >= 0.999 && lCos >= 0.999 && tCos >= 0.999 && tMax < 1e-5 && pMax < 1e-4
    print(pass ? "[audio-vae-encode-gate] PASS ✅" : "[audio-vae-encode-gate] FAIL ❌")
    if !pass { exit(1) }
}

/// Vocoder+BWE parity: mel → 48kHz waveform vs oracle golden (fp32).
func vocoderGate() throws {
    let dir = "\(goldensBase)/vocoder"
    let weightsPath = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/vocoder.safetensors"
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
    let base = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx"
    let dir = "\(goldensBase)/audio_decode"
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
    let dir = "\(goldensBase)/upsampler"
    let weightsPath = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/spatial_upscaler_x2_v1_1.safetensors"
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

/// x1.5-spatial + x2-temporal upsampler variants vs oracle goldens (GAP-ANALYSIS #5).
/// Loads each checkpoint through `Upsampler.load(path:)`, so the sidecar-config variant
/// resolution — the actual production path — is what gets exercised, not a hand-picked enum.
func upsamplerVariantsGate() throws {
    let base = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx"
    let cases: [(name: String, file: String, goldens: String, expect: Upsampler.Variant)] = [
        ("x1.5 spatial (rational 3/2)", "spatial_upscaler_x1_5_v1_0.safetensors", "upsampler_x1_5", .spatialX1_5),
        ("x2 temporal (drop frame 0)", "temporal_upscaler_x2_v1_0.safetensors", "upsampler_temporal", .temporalX2),
    ]
    var allPass = true
    for c in cases {
        let up = try Upsampler.load(path: URL(fileURLWithPath: "\(base)/\(c.file)"))
        let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(goldensBase)/\(c.goldens)/io.safetensors"))
        guard up.variant == c.expect else {
            print("[upsampler-variants-gate] \(c.name): sidecar resolved \(up.variant) ≠ \(c.expect) ❌")
            allPass = false
            continue
        }
        let out = up(io["latent"]!)
        eval(out)
        let cos = cosine(out, io["out"]!), m = maxAbs(out, io["out"]!)
        let shapeOK = out.shape == io["out"]!.shape
        let pass = cos >= 0.999 && shapeOK
        allPass = allPass && pass
        print(String(format: "[upsampler-variants-gate] %@: cosine=%.6f maxAbs=%.5f  shape %@ vs %@ %@",
                     c.name as NSString, cos, m, "\(out.shape)" as NSString,
                     "\(io["out"]!.shape)" as NSString, pass ? "✅" : "❌"))
    }
    print(allPass ? "[upsampler-variants-gate] PASS ✅" : "[upsampler-variants-gate] FAIL ❌")
    if !allPass { exit(1) }
}

/// Two-stage upscale-step parity: half-res latent → denorm → upsample → renorm vs oracle.
func upscaleStepGate() throws {
    let base = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx"
    let dir = "\(goldensBase)/upscale_step"
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
    let q8 = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx-q8/transformer-distilled.safetensors"
    let base = "\(goldensBase)"
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
    let q4 = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx-q4/transformer-distilled.safetensors"
    let base = "\(goldensBase)"
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
    let bf16 = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/transformer-distilled.safetensors"
    let base = "\(goldensBase)"
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
    let dir = "\(goldensBase)/dit_full"
    let weightsPath = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/transformer-distilled.safetensors"
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
    let base = "/Volumes/Satechi/Models/dgrauet"
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

// MARK: - Gap-queue P2 — LoRA fetch gate (no GPU, real network)
//
// Proves the first-use adapter download is never silent: fetches a registry adapter into a FRESH
// directory with a bound WeightDownloadProgress sink and prints every report. PASS = the file
// lands with plausible size AND at least one progress report fired. Cleans up after itself when
// it created the temp dir.
func loraFetchGate(id: String, directory: String?) async throws {
    let registry = try LoRARegistry.bundled()
    guard let entry = registry.entry(id: id) else {
        print("[lora-fetch-gate] FAIL ❌ unknown adapter '\(id)'")
        return
    }
    let ownsDir = directory == nil
    let dir = directory.map { URL(fileURLWithPath: $0) }
        ?? FileManager.default.temporaryDirectory.appendingPathComponent("lora-fetch-gate-\(UUID().uuidString)")
    defer { if ownsDir { try? FileManager.default.removeItem(at: dir) } }
    let cache = LoRACache(directory: dir)
    print("[lora-fetch-gate] \(id) → \(dir.path)  cached=\(cache.isCached(entry))")

    final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var n = 0
        func bump() -> Int { lock.lock(); defer { lock.unlock() }; n += 1; return n }
        var count: Int { lock.lock(); defer { lock.unlock() }; return n }
    }
    let reports = Counter()
    let t0 = Date()
    let file = try await WeightDownloadProgress.$sink.withValue({ fraction, rate in
        _ = reports.bump()
        let mbps = rate.map { String(format: "  %.1f MB/s", $0 / 1e6) } ?? ""
        print(String(format: "[lora-fetch-gate] %3.0f%%%@", fraction * 100, mbps))
    }) {
        try await cache.ensure(entry)
    }
    let bytes = ((try? FileManager.default.attributesOfItem(atPath: file.path))?[.size] as? UInt64) ?? 0
    let pass = bytes > 50_000_000 && reports.count >= 1
    print(String(format: "[lora-fetch-gate] %.2f GB in %.0fs · %d progress report(s) · isCached now %@",
                 Double(bytes) / 1e9, Date().timeIntervalSince(t0),
                 reports.count, "\(cache.isCached(entry))" as NSString))
    print(pass ? "[lora-fetch-gate] PASS ✅ — download visible end-to-end"
               : "[lora-fetch-gate] FAIL ❌ (bytes=\(bytes), reports=\(reports.count))")
}

// MARK: - SPEED-PLAN S4 — SDPA fused-path probe
//
// The DiT has a single SDPA call site (`MLXFast.scaledDotProductAttention`, mask .none), but MLX
// dispatches INTERNALLY (fused Metal kernel vs composed matmul+softmax) by shape/dtype — a silent
// fallback at long seqLen is exactly the wall the Wan profiling program found. This probe times
// the fast entry point against a manual compose at the production video self-attn geometry
// (B=1 H=32 D=128 bf16). Read: ratio ≈ 1× ⇒ the entry point is falling back internally and the
// S4 SDPA lever is LIVE; ratio ≫ 1× ⇒ the fused path is hit at that shape, look elsewhere.
func sdpaProbeGate() {
    let H = 32, D = 128
    let scale = 1.0 / Float(D).squareRoot()
    print("[sdpa-probe] B=1 H=\(H) D=\(D) bf16 mask=.none — MLXFast entry vs manual compose")
    // nv=2112 (512×288×121 compact envelope) · 5632 (704×512×121) · 21120 (704×512×481 max128)
    for n in [2112, 5632, 21120] {
        let q = MLXRandom.normal([1, H, n, D]).asType(.bfloat16)
        let k = MLXRandom.normal([1, H, n, D]).asType(.bfloat16)
        let v = MLXRandom.normal([1, H, n, D]).asType(.bfloat16)
        eval(q, k, v)
        // warmup both paths (kernel compile is not the datum)
        eval(MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none))
        eval(softmax(q.matmul(k.transposed(0, 1, 3, 2)) * scale, axis: -1).matmul(v))
        let reps = n > 10000 ? 3 : 10
        var tF = 0.0, tM = 0.0
        for _ in 0 ..< reps {
            let t0 = Date()
            eval(MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none))
            tF += Date().timeIntervalSince(t0)
        }
        for _ in 0 ..< reps {
            let t0 = Date()
            eval(softmax(q.matmul(k.transposed(0, 1, 3, 2)) * scale, axis: -1).matmul(v))
            tM += Date().timeIntervalSince(t0)
        }
        Memory.clearCache()
        print(String(format: "[sdpa-probe] N=%6d  fused %8.1f ms  manual %8.1f ms  ratio %.2f×",
                     n, tF / Double(reps) * 1000, tM / Double(reps) * 1000, tM / tF))
    }
    print("[sdpa-probe] ratio ≈ 1× ⇒ silent fallback (S4 SDPA lever live); ratio ≫ 1× ⇒ fused path hit")
}

// MARK: - TILING-PLAN prerequisite — does an attention MASK keep SDPA on the fused kernel?
//
// Modality tiling (GAP-ANALYSIS #7) introduces per-tile attention masks. The whole point of
// tiling is to avoid materializing the O(N²) scores tensor — so if passing a mask silently drops
// `scaledDotProductAttention` off the fused (flash, non-materializing) kernel onto the
// `softmax(QKᵀ)@V` fallback, tiling would RECREATE the exact tensor it exists to avoid and the
// memory math evaporates. `mlx-porting` (framework constraints) warns the fused path is
// conditionally gated, with mask dtype handling named as one of the gates. This settles it by
// measurement rather than by reading kernel source.
//
// Discriminator = PEAK MEMORY attributable to the call. A materialized fallback must allocate
// [B, H, N, N]; the mask itself is only [N, N]. At H=32 those differ by 32×, which is unmissable.
// Timing is reported alongside as a cross-check (the S4 probe's "ratio ≈ 1× ⇒ fallback" heuristic).
func sdpaMaskProbeGate() {
    let H = 32, D = 128
    let scale = 1.0 / Float(D).squareRoot()
    print("[sdpa-mask-probe] B=1 H=\(H) D=\(D) bf16 — .none vs .array(additive bf16) vs .array(bool)")
    print("[sdpa-mask-probe] fused ⇒ call-attributed peak ≈ output only; materialized ⇒ ≈ [1,H,N,N]")

    for n in [2112, 5632] {
        let q = MLXRandom.normal([1, H, n, D]).asType(.bfloat16)
        let k = MLXRandom.normal([1, H, n, D]).asType(.bfloat16)
        let v = MLXRandom.normal([1, H, n, D]).asType(.bfloat16)
        // Additive all-zero mask = a semantic NO-OP, so outputs must match the .none arm. That
        // doubles as a correctness check that the mask is applied additively, not as a multiplier.
        let addMask = MLXArray.zeros([n, n]).asType(.bfloat16)
        let boolMask = MLXArray.ones([n, n]).asType(.bool)          // keep-all
        eval(q, k, v, addMask, boolMask)

        let scoresGB = Double(H * n * n * 2) / 1e9
        let maskGB = Double(n * n * 2) / 1e9

        func arm(_ mode: MLXFast.ScaledDotProductAttentionMaskMode) -> (Double, Double, MLXArray) {
            // Warm the kernel first — compile is not the datum.
            eval(MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mode))
            Memory.clearCache()
            // ⚠️ Reset the high-water AFTER the warmup and the cache drain, or the warmup's own
            // peak swallows the measured call and every arm reads 0 (v1 of this probe did exactly
            // that and produced a confident, meaningless "peak+0.000 GB" for `.none` too).
            GPU.resetPeakMemory()
            let base = Memory.peakMemory
            let t0 = Date()
            let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mode)
            eval(out)
            let dt = Date().timeIntervalSince(t0) * 1000
            let attributed = Double(Memory.peakMemory > base ? Memory.peakMemory - base : 0) / 1e9
            return (attributed, dt, out)
        }

        let (memNone, tNone, outNone) = arm(.none)
        let (memAdd, tAdd, outAdd) = arm(.array(addMask))
        let (memBool, tBool, outBool) = arm(.array(boolMask))
        let outGB = Double(H * n * D * 2) / 1e9

        print(String(format: "[sdpa-mask-probe] N=%5d  |  materialized [1,H,N,N] = %.2f GB · mask %.3f GB · output %.3f GB",
                     n, scoresGB, maskGB, outGB))
        print(String(format: "[sdpa-mask-probe]   .none      peak+%.3f GB  %7.1f ms", memNone, tNone))
        print(String(format: "[sdpa-mask-probe]   .array add peak+%.3f GB  %7.1f ms  (%.2f× time vs none)",
                     memAdd, tAdd, tAdd / max(tNone, 1e-9)))
        print(String(format: "[sdpa-mask-probe]   .array bool peak+%.3f GB  %7.1f ms  (%.2f× time vs none)",
                     memBool, tBool, tBool / max(tNone, 1e-9)))
        // No-op mask ⇒ identical result. If these diverge the mask semantics are not what we assume.
        print(String(format: "[sdpa-mask-probe]   cos(add, none)=%.6f   cos(bool, none)=%.6f",
                     cosine(outAdd, outNone), cosine(outBool, outNone)))
        let verdict = memAdd > scoresGB * 0.5 ? "MATERIALIZED ❌ (tiling math would break)"
                                              : "fused ✅ (mask does not force materialization)"
        print("[sdpa-mask-probe]   verdict(additive): \(verdict)")
        Memory.clearCache()
    }
}

// MARK: - SPEED-PLAN S6 — quant speed ladder at a fixed production shape
//
// int4 exists for MEMORY; on M-series, int8/int4 quantizedMatmul can also be FASTER than bf16 at
// these (bandwidth-bound) matrix shapes. This gate produces the datum: same one-stage t2v run
// twice per quant (run 1 compiles the size-specific kernels, run 2 is the measured leg) at one
// fixed shape. Compare `run2` across --speed-bench bf16 | int8 | int4. Per-step detail: add
// MLX_PROFILE=1. Per-step latent redundancy (S3 ceiling): add LTX_STEP_DELTA=1.
func speedBenchGate(quant: String, width: Int, height: Int, frames: Int) async throws {
    let base = "/Volumes/Satechi/Models/dgrauet"
    let ltxDir = URL(fileURLWithPath: "\(base)/ltx-2.3-mlx")
    let gemmaDir = URL(fileURLWithPath: defaultGemma)
    let transformerPath: URL?
    switch quant {
    case "int8", "q8": transformerPath = URL(fileURLWithPath: "\(base)/ltx-2.3-mlx-q8/transformer-distilled.safetensors")
    case "int4", "q4": transformerPath = URL(fileURLWithPath: "\(base)/ltx-2.3-mlx-q4/transformer-distilled.safetensors")
    default: transformerPath = nil  // bf16
    }
    print("[speed-bench] quant=\(quant)  shape=\(width)×\(height)×\(frames)f  path=one-stage")

    let p0 = Date()
    var warm = [transformerPath ?? ltxDir.appendingPathComponent("transformer-distilled.safetensors")]
    for f in ["connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
              "audio_vae.safetensors", "vocoder.safetensors"] {
        warm.append(ltxDir.appendingPathComponent(f))
    }
    // An LTX_VAE_DECODER override is NOT in the list above, so without this it cold-faults off the
    // archive volume mid-decode and the A/B measures paging, not the decoder.
    if ProcessInfo.processInfo.environment["LTX_VAE_DECODER"] != nil {
        warm.append(ltxDir.appendingPathComponent("vae_decoder_pruna.safetensors"))
    }
    warm.append(contentsOf: ((try? FileManager.default.contentsOfDirectory(at: gemmaDir, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" })
    prewarmFiles(warm)
    print(String(format: "[speed-bench] prewarm %.1fs (%d files)", Date().timeIntervalSince(p0), warm.count))

    let sampler = PhysSampler(); sampler.start()
    let t0 = Date()
    let pipeline = try await LTX2Pipeline.load(ltxDir: ltxDir, gemmaDir: gemmaDir, transformerPath: transformerPath)
    print(String(format: "[speed-bench] load %.1fs", Date().timeIntervalSince(t0)))

    let r1 = Date()
    _ = try await pipeline.t2v(prompt: "a cat playing piano", height: height, width: width, numFrames: frames, fps: 24, seed: 42)
    print(String(format: "[speed-bench] run1 (kernel compile + generate) %.1fs", Date().timeIntervalSince(r1)))
    Memory.clearCache()

    sampler.resetMax()
    let r2 = Date()
    let out = try await pipeline.t2v(prompt: "a cat playing piano", height: height, width: width, numFrames: frames, fps: 24, seed: 42)
    eval(out.video); if let a = out.audio { eval(a) }
    let run2 = Date().timeIntervalSince(r2)
    sampler.stop()
    let outputSeconds = Double(frames) / 24.0
    print(String(format: "[speed-bench] run2 (measured) %.1fs → %.1f s per output-second (%.2fs clip)",
                 run2, run2 / outputSeconds, outputSeconds))
    print(String(format: "[speed-bench] peak phys_footprint %.2f GB", gbOf(sampler.maxBytes())))
    print(String(format: "[speed-bench] SUMMARY quant=%@ shape=%dx%dx%d run2_s=%.1f spos=%.2f",
                 quant as NSString, width, height, frames, run2, run2 / outputSeconds))
}

// MARK: - i2v spot measure (BRIDGE-LTX-005) — tighten the max128 activation hint
//
// max128's `peakActivationBytesHint` is held at the conservative pre-T3b ceiling (52 GB) because
// the i2v/per-token path was UNMEASURED at the new 481f cap. This gate produces that datum: it
// drives the WRAPPER (so the profile clamp, the runtime i2v-adapter LoRA (~4.9 GB resident), and
// the MP4 encode are all inside the sampled window) with a synthetic init frame at the max128
// envelope, and reports the SPLIT line (floor / peak / activation) the hint is recalibrated from.
@InferenceActor
/// GAP-ANALYSIS #8 receipt vehicle: wrapper-level plain t2v (the STREAMED mux lane) with the
/// floor/peak/activation split. A/B vs the materialized lane via `LTX_STREAM_MUX=0` — identical
/// code path otherwise. Peaks are the datum; single-run wall-clock is not (BENCH.md).
func t2vSpotGate(width: Int, height: Int, frames: Int) async throws {
    let env = ProcessInfo.processInfo.environment
    let base = "/Volumes/Satechi/Models/dgrauet"
    let cfg = LTX2Configuration(
        quant: .bf16,
        ltxDirectory: URL(fileURLWithPath: "\(base)/ltx-2.3-mlx"),
        transformerPath: nil,
        gemmaDirectory: URL(fileURLWithPath: defaultGemma),
        modelsRootDirectory: URL(fileURLWithPath: "/Volumes/Satechi/Models"),
        profile: nil)
    let lane = env["LTX_STREAM_MUX"] == "0" ? "MATERIALIZED" : "STREAMED"
    print("[t2v-spot] request \(width)×\(height)×\(frames)f bf16 · mux lane: \(lane)")

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
    prewarmFiles(warm)
    print(String(format: "[t2v-spot] prewarm %.1fs (%d files)", Date().timeIntervalSince(p0), warm.count))

    let sampler = PhysSampler(); sampler.start()
    let pkg = MLXLTX2Package(configuration: cfg)
    try await pkg.load()
    Memory.clearCache()

    // LTX_T2V_PROMPT/LTX_T2V_SAVE: perceptual A/B hooks (e.g. the Pruna face-prompt caveat —
    // decoder swapped via LTX_VAE_DECODER, trajectory bit-identical, so saved MP4s differ by
    // decode only). No timing claims ride on this gate, so thermal state is irrelevant here.
    // LTX_IC_ADAPTER + LTX_IC_REF: IC-adapter fixture hooks (union-control canny fixture) —
    // rides the wrapper's full ic.* intake (registry lookup, LoRA apply at entry default
    // strength, reference ingest at the entry's declared downscale).
    func request(_ nf: Int) -> T2VRequest {
        var meta: [String: MetaValue] = [:]
        if let ic = env["LTX_IC_ADAPTER"], !ic.isEmpty {
            meta[ICMetaKeys.adapterId] = .string(ic)
            if let ref = env["LTX_IC_REF"], !ref.isEmpty {
                meta[ICMetaKeys.referencePath] = .string(ref)
            }
        }
        return T2VRequest(prompt: env["LTX_T2V_PROMPT"] ?? "a fox running down a beach at sunset, waves rolling in",
                          numFrames: nf, fps: 24, width: width, height: height, seed: 42,
                          metaData: meta)
    }
    // Warmup at 9f (kernel compile + decode/encode stacks) — excluded from the measured peak.
    _ = try await pkg.run(request(9))
    Memory.clearCache()
    let floor = physFootprintBytes()
    print(String(format: "[t2v-spot] resident floor (post-warmup + clearCache): %.2f GB", gbOf(floor)))

    sampler.resetMax()
    let r0 = Date()
    let resp = try await pkg.run(request(frames)) as! T2VResponse
    if let save = env["LTX_T2V_SAVE"], !save.isEmpty {
        let url = URL(fileURLWithPath: (save as NSString).expandingTildeInPath)
        try resp.video.data.write(to: url)
        print("[t2v-spot] saved \(url.path) (\(String(format: "%.1f", Double(resp.video.data.count) / 1_000_000)) MB)")
    }
    let peak = sampler.maxBytes(); sampler.stop()
    let activation = peak > floor ? peak - floor : 0
    print(String(format: "[t2v-spot] run %.1fs  mp4 %.1f MB", Date().timeIntervalSince(r0),
                 Double(resp.video.data.count) / 1_000_000))
    print(String(format: "[t2v-spot] SPLIT lane=%@ floor=%.2f GB peak=%.2f GB activation=%.2f GB",
                 lane as NSString, gbOf(floor), gbOf(peak), gbOf(activation)))
    print(String(format: "[t2v-spot] SUMMARY lane=%@ shape=%dx%dx%d peak=%.2f act=%.2f",
                 lane as NSString, width, height, frames, gbOf(peak), gbOf(activation)))
}

func i2vSpotGate(width: Int, height: Int, frames: Int) async throws {
    // Tier + quant via env (BRIDGE-LTX-012 low-tier measurement): `LTX_TIER=<rawValue>` selects
    // the profile (default max128, preserving the BRIDGE-LTX-005 shape); the quant follows the
    // tier's recommendation (int4 low tiers → q4 transformer) unless `LTX_QUANT` overrides.
    let env = ProcessInfo.processInfo.environment
    let profile = env["LTX_TIER"].flatMap { LTX2Profile(rawValue: $0) } ?? .max128
    let quantName = env["LTX_QUANT"] ?? {
        switch profile.recommendedQuant {
        case .int4: return "int4"
        case .int8: return "int8"
        default: return "bf16"
        }
    }()
    let base = "/Volumes/Satechi/Models/dgrauet"
    let transformerPath: URL?
    let quant: Quant
    switch quantName {
    case "int8", "q8":
        transformerPath = URL(fileURLWithPath: "\(base)/ltx-2.3-mlx-q8/transformer-distilled.safetensors")
        quant = .int8
    case "int4", "q4":
        transformerPath = URL(fileURLWithPath: "\(base)/ltx-2.3-mlx-q4/transformer-distilled.safetensors")
        quant = .int4
    default:
        transformerPath = nil
        quant = .bf16
    }
    let cfg = LTX2Configuration(
        quant: quant,
        ltxDirectory: URL(fileURLWithPath: "\(base)/ltx-2.3-mlx"),
        transformerPath: transformerPath,
        gemmaDirectory: URL(fileURLWithPath: defaultGemma),
        // The LoRA cache root (`ltx-lora-cache/i2v-adapter.safetensors`, already fetched by the app).
        modelsRootDirectory: URL(fileURLWithPath: "/Volumes/Satechi/Models"),
        profile: profile)
    print("[i2v-spot] request \(width)×\(height)×\(frames)f \(quantName) · profile=\(profile.rawValue) · lora=i2v-adapter")

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
    warm.append(URL(fileURLWithPath: "/Volumes/Satechi/Models/ltx-lora-cache/i2v-adapter.safetensors"))
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
func vaeChunkGate(fLat: Int, chunk: Int, halo: Int, pruna: Bool = false) async throws {
    // `--pruna` runs the same seam check on the lean decoder: the halo/trim math is
    // width-independent, but its channel-adapter blocks sit inside each chunk's decode, so the
    // seam is worth re-proving on the variant that actually ships it.
    let file = pruna ? "vae_decoder_pruna.safetensors" : "vae_decoder.safetensors"
    let weightsPath = "/Volumes/Satechi/Models/dgrauet/ltx-2.3-mlx/\(file)"
    print("[vae-chunk-gate] decoder=\(pruna ? "pruna" : "stock")")
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
    let chunked = try dec.decodeChunked(latent, chunkFrames: chunk, halo: halo); eval(chunked)
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
} else if args.contains("--gemma-tokenizer-gate") {
    let dir = positional.first ?? "\(goldensBase)/gemma_tokenizer"
    let gemmaDir = positional.dropFirst().first ?? defaultGemma
    try await gemmaTokenizerGate(goldensDir: dir, gemmaDir: gemmaDir)
} else if args.contains("--gemma-textgen-gate") {
    try await gemmaTextGenGate(gemmaDir: positional.first ?? defaultGemma)
} else if args.contains("--text-encode-gate") {
    try await textEncodeGate(goldensPath: defaultGoldens, gemmaDir: defaultGemma, connectorPath: defaultConnector)
} else if args.contains("--pipeline-25-gate") {
    try pipeline25Gate()
} else if args.contains("--keyframe-slots-gate") {
    try keyframeSlotsGate()
} else if args.contains("--gemma4-gate") {
    try await gemma4Gate()
} else if args.contains("--denoise-wiring-gate") {
    try denoiseWiringGate()
} else if args.contains("--ancestral-step-gate") {
    try ancestralStepGate()
} else if args.contains("--dit-tiny-kf25-gate") {
    try ditTinyKF25Gate()
} else if args.contains("--timestep-dtype-gate") {
    try timestepDtypeGate()
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
} else if args.contains("--vae-decode-pruna-gate") {
    try vaeDecodePrunaGate()
} else if args.contains("--vae-decode-bench") {
    try vaeDecodeBench()
} else if args.contains("--vae-encode-gate") {
    try vaeEncodeGate()
} else if args.contains("--audio-vae-decode-gate") {
    try audioVaeDecodeGate()
} else if args.contains("--audio-vae-encode-gate") {
    try audioVaeEncodeGate()
} else if args.contains("--vocoder-gate") {
    try vocoderGate()
} else if args.contains("--audio-decode-gate") {
    try audioDecodeGate()
} else if args.contains("--upsampler-gate") {
    try upsamplerGate()
} else if args.contains("--upsampler-variants-gate") {
    try upsamplerVariantsGate()
} else if let i = args.firstIndex(of: "--two-stage-variants-gate") {
    let which = (i + 1 < args.count && !args[i + 1].hasPrefix("--")) ? args[i + 1] : "all"
    try await twoStageVariantsGate(which: which)
} else if args.contains("--upscale-step-gate") {
    try upscaleStepGate()
} else if args.contains("--denoise-gate") {
    try denoiseGate()
} else if args.contains("--ic-tiny-gate") {
    try icTinyGate()
} else if args.contains("--ic-ingest-gate") {
    try icIngestGate()
} else if args.contains("--e2e-gate") {
    try e2eGate()
} else if args.contains("--vae-chunk-gate") {
    let ints = positional.compactMap { Int($0) }
    try await vaeChunkGate(fLat: ints.count > 0 ? ints[0] : 15,
                           chunk: ints.count > 1 ? ints[1] : 5,
                           halo: ints.count > 2 ? ints[2] : 4,
                           pruna: args.contains("--pruna"))
} else if args.contains("--encode-stress") {
    let n = positional.first.flatMap { Int($0) } ?? 41
    try await encodeStressGate(frames: n, software: args.contains("--software"), hog: args.contains("--hog"),
                               audio: args.contains("--audio"))
} else if args.contains("--mem-bench") {
    try await memBenchGate(quant: positional.first ?? "bf16")
} else if args.contains("--lora-fetch-gate") {
    try await loraFetchGate(id: positional.first ?? "fantasy-anime",
                            directory: positional.dropFirst().first)
} else if args.contains("--sdpa-probe") {
    sdpaProbeGate()
} else if args.contains("--modality-tile-gate") {
    try await modalityTileGatesMain(args: args, positional: Array(positional))  // ModalityTileGates.swift
} else if args.contains("--sdpa-mask-probe") {
    sdpaMaskProbeGate()   // TILING-PLAN prerequisite: does a mask force the materialized fallback?
} else if args.contains("--speed-bench") {
    let quant = positional.first(where: { Int($0) == nil }) ?? "bf16"
    let ints = positional.compactMap { Int($0) }
    try await speedBenchGate(quant: quant,
                             width: ints.count > 0 ? ints[0] : 704,
                             height: ints.count > 1 ? ints[1] : 512,
                             frames: ints.count > 2 ? ints[2] : 121)
} else if args.contains("--lora-quant-gate") {
    // BRIDGE-LTX-012 fidelity gate: packed-factor delta vs full precision on the real adapter.
    let bits = positional.compactMap { Int($0) }.first ?? 8
    let path = positional.first { Int($0) == nil }
        ?? "/Volumes/Satechi/Models/ltx-lora-cache/i2v-adapter.safetensors"
    let (worst, mean) = try LTX2LoRA.factorQuantFidelity(
        url: URL(fileURLWithPath: path), bits: bits, sample: 48)
    print(String(format: "[lora-quant-gate] bits=%d sample=48  worst cos=%.6f  mean cos=%.6f", bits, worst, mean))
    let pass = worst >= 0.999
    print(pass ? "[lora-quant-gate] PASS ✅" : "[lora-quant-gate] FAIL ❌")
    if !pass { exit(1) }
} else if args.contains("--stream-tiny-gate") {
    // HV2 streaming gates (StreamGates.swift; STREAMING-PLAN.md)
    try streamTinyGate(scratch: NSTemporaryDirectory())
} else if args.contains("--stream-parity-gate") {
    try streamParityGate(quant: positional.first ?? "bf16")
} else if args.contains("--stream-auto-gate") {
    let quant = positional.first(where: { Int($0) == nil }) ?? "bf16"
    try streamAutoGate(quant: quant, videoN: positional.compactMap { Int($0) }.first)
} else if args.contains("--stream-budget-gate") {
    let quant = positional.first(where: { Int($0) == nil }) ?? "q4"
    let ints = positional.compactMap { Int($0) }
    try streamBudgetGate(
        quant: quant, videoN: ints.count > 0 ? ints[0] : 5632,
        budgetGB: ints.count > 1 ? Double(ints[1]) : 12.0,
        steps: ints.count > 2 ? ints[2] : 4)
} else if args.contains("--frame-codec-gate") {
    // GAP-ANALYSIS #1 / SPEED-PLAN S2 step 2: the on-device BGRA repack must be a PURE refactor.
    let ints = positional.compactMap { Int($0) }
    let r = frameCodecRepackSelfTest(height: ints.count > 0 ? ints[0] : 512,
                                     width: ints.count > 1 ? ints[1] : 704)
    print("[frame-codec-gate] \(r.detail)")
    print(r.ok ? "[frame-codec-gate] PASS ✅" : "[frame-codec-gate] FAIL ❌")
    if !r.ok { exit(1) }
} else if args.contains("--bench-verdict-selftest") {
    benchVerdictSelfTest()   // BenchAnalysis.swift: v2 verdict logic vs the harness's past failures
} else if args.contains("--bench-e2e") {
    try await benchE2E()
} else if args.contains("--vae-rf-probe") || args.contains("--vae-tile-gate") || args.contains("--vae-tile-bench") || args.contains("--vae-mux-bench") {
    try await tileGatesMain(args: args, positional: Array(positional))  // TileGates.swift (spatial tiling)
} else if args.contains("--t2v-spot") {
    let ints = positional.compactMap { Int($0) }
    try await t2vSpotGate(width: ints.count > 0 ? ints[0] : 704,
                          height: ints.count > 1 ? ints[1] : 512,
                          frames: ints.count > 2 ? ints[2] : 121)
} else if args.contains("--i2v-spot") {
    let ints = positional.compactMap { Int($0) }
    try await i2vSpotGate(width: ints.count > 0 ? ints[0] : 704,
                          height: ints.count > 1 ? ints[1] : 512,
                          frames: ints.count > 2 ? ints[2] : 481)
} else {
    print("usage: RunLTX2 --connector-gate | --gemma-gate | --text-encode-gate | --dit-tiny-gate  [goldens.safetensors] [path]")
    print("       RunLTX2 --mem-bench [bf16|int8|int4]   (efficiency-sweep footprint at 704×512×9f)")
    print("       RunLTX2 --speed-bench [bf16|int8|int4] [w] [h] [frames]   (SPEED-PLAN S6 quant ladder, default 704 512 121)")
    print("       RunLTX2 --bench-e2e --arm name:quant=bf16[,decoder=pruna][,cache=GB][,env.K=V] [--arm …]")
    print("                 [--blocks 2] [--runs 2] [--cooldown 45] [--size 704x512] [--frames 24] [--two-stage]")
    print("                 (protocol A/B harness: ABBA order, excluded warmups, phys sampling, receipts → probes/)")
    print("       RunLTX2 --sdpa-probe                   (SPEED-PLAN S4: fused-SDPA vs manual compose at production shapes)")
    print("       RunLTX2 --lora-fetch-gate [id] [dir]   (P2: adapter download with visible progress; default fantasy-anime, temp dir)")
    print("       RunLTX2 --i2v-spot [w] [h] [frames]    (BRIDGE-LTX-005 max128 i2v SPLIT measure, default 704 512 481)")
}
