// GenGapProbe.swift — `--gen-gap-probe`, the DIAGNOSTIC arm for AB-R-0080 / AB-A-0010.
//
// ⚠️ **DIAGNOSIS ONLY. This file changes no shipping behaviour.** It is a measurement arm, like
// `--enhancer-bench`; nothing in `LTX2` is modified by it and no default is altered. Every knob it
// moves is passed explicitly into `GenerateParameters` for the duration of one measured arm.
//
// WHAT IT ANSWERS. AB-R-0080 measured mlx-swift-lm generating ~2.6× slower than Python `mlx_lm` on
// the same model and machine, and reported the factor as **UNIFORM across prefill and decode**
// (≈2.9× / ≈2.6×). Uniformity was read as evidence of one systemic cause. That reading is only
// safe if the two phases were actually measured against the same suspect list — so this arm
// decomposes them properly and varies the one parameter the two stacks do not share.
//
// 🔑 **The parameter: prefill chunk size.** `mlx_lm.generate` prefills in chunks of
// `prefill_step_size = 512` (generate.py:474). `Gemma3TextModel.prepare` in mlx-swift-lm prefills
// in chunks of `defaultPrefillChunkSize = 128`, clamped to `config.sliding_window`
// (Gemma3Text.swift:451/462) — and `GenerateParameters.prefillStepSize` defaults to `nil`, so
// every caller that does not set it (ours included) gets 128. A 948-token prompt is therefore
// **8 chunk forwards in Swift against 2 in Python**. That is a prefill-only difference: decode
// feeds one token per step on both sides and never touches this path.
//
// HOW IT MEASURES. `ChatSession.streamDetails` yields a `.info(GenerateCompletionInfo)` carrying
// `promptTime` and `generateTime` SEPARATELY, so one generation gives both phases — no
// `maxTokens=1` second run, and no subtraction of two wall-clocks measured minutes apart. The
// arms sweep `prefillStepSize` over {128 (the default), 256, 512, 1024} at a fixed maxTokens, and
// a final arm drives the low-level `MLXLMCommon.generate` loop instead of `ChatSession` so the
// session wrapper's own cost is priced rather than assumed (angle 2 of the ask).
//
// PROTOCOL. Model loads ONCE for the whole sweep (a row is the pass, not the pass plus a reload).
// One warmup generation is discarded — first forward pays shape-specific Metal kernel compile
// (PROFILING.md §1). Arms then run in **ABBA order across rounds**, per BENCH.md: round 0 walks
// the arm list forward, round 1 walks it backward, so a one-way thermal ramp cancels instead of
// landing on whichever arm ran last. Medians are reported, not single runs.
//
// usage: RunLTX2 --gen-gap-probe [gemmaGenerativeDir] [maxTokens] [rounds] [seed]

import Foundation
import LTX2
import MLX
import MLXLMCommon

/// One measured arm: a prefill chunk size, and which driver ran it.
private struct GapArm {
    let id: String
    let prefillStepSize: Int?      // nil ⇒ the model's own default (128 for Gemma-3 text)
    let useChatSession: Bool
    let note: String
}

/// A single observation. `wallSeconds` is the caller-visible cost of the whole pass — what
/// `--enhancer-bench` times with a `Date()` bracket. `promptSeconds`/`genSeconds` are what
/// mlx-swift-lm itself reports for the two model phases. **Their difference is the harness
/// overhead that AB-R-0080 attributed to the model.**
private struct GapObs {
    let promptTokens: Int
    let promptSeconds: Double
    let genTokens: Int
    let genSeconds: Double
    let wallSeconds: Double
}

private func median(_ xs: [Double]) -> Double {
    guard !xs.isEmpty else { return .nan }
    let s = xs.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

func genGapProbeRun(gemmaDir: String?, maxTokens: Int, rounds: Int, seed: UInt64) async throws {
    let genDir = URL(fileURLWithPath: gemmaDir ?? defaultGemma)
    guard FileManager.default.fileExists(atPath: genDir.path) else {
        print("[gen-gap] FAIL ❌ generative checkpoint not present: \(genDir.path)")
        exit(2)
    }

    // Same system prompt and user template as `--enhancer-bench` and the oracle, so the numbers
    // here are directly comparable to AB-R-0080's table rather than to a new regime.
    let oraclePrompt = packageRoot
        .deletingLastPathComponent()
        .appendingPathComponent("ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/"
                              + "text_encoders/gemma/encoders/prompts/gemma_t2v_system_prompt.txt")
    guard let system = try? String(contentsOf: oraclePrompt, encoding: .utf8), !system.isEmpty else {
        print("[gen-gap] FAIL ❌ oracle system prompt not found at \(oraclePrompt.path) — this arm "
            + "must prefill the SAME 928 tokens the oracle does or it is not comparable.")
        exit(2)
    }
    let userText = "user prompt: a lighthouse keeper climbing the spiral stairs"
    let temperature: Float = 0.7

    let arms: [GapArm] = [
        .init(id: "P128", prefillStepSize: nil,  useChatSession: true,
              note: "DEFAULT — Gemma3TextModel.defaultPrefillChunkSize, what every unset caller gets"),
        .init(id: "P256", prefillStepSize: 256,  useChatSession: true, note: ""),
        .init(id: "P512", prefillStepSize: 512,  useChatSession: true,
              note: "mlx_lm's prefill_step_size default (generate.py:474) — the oracle's chunk"),
        .init(id: "P1024", prefillStepSize: 1024, useChatSession: true,
              note: "clamped to config.sliding_window; ≥ prompt ⇒ single chunk"),
        .init(id: "RAW512", prefillStepSize: 512, useChatSession: false,
              note: "low-level MLXLMCommon.generate, NOT ChatSession — prices the session wrapper"),
        .init(id: "BENCH", prefillStepSize: nil, useChatSession: true,
              note: "EXACTLY --enhancer-bench: fresh ChatSession + respond(to:), Date() bracket"),
        .init(id: "PREP", prefillStepSize: nil, useChatSession: true,
              note: "chat-template render + tokenize ONLY — no model forward"),
        .init(id: "TOKZ", prefillStepSize: nil, useChatSession: true,
              note: "tokenize the 4175-char system prompt ONLY — no template, no model"),
    ]

    print("[gen-gap] ───── DIAGNOSTIC arm for AB-R-0080 / AB-A-0010 (no shipping behaviour changed) ─────")
    print("[gen-gap] model        = \(genDir.path)")
    print("[gen-gap] system       = oracle gemma_t2v_system_prompt.txt (\(system.count) chars)")
    print("[gen-gap] user         = \"\(userText)\"")
    print(String(format: "[gen-gap] sampling     = temp %.1f  seed %llu  maxTokens %d",
                 temperature, seed, maxTokens))
    print("[gen-gap] rounds       = \(rounds) (ABBA: even rounds forward, odd rounds reversed)")
    print("[gen-gap] oracle ref   = prefill 0.66–0.71s · decode 65 tok/s (AB-R-0080 + re-run today)")
    print("[gen-gap] 🔑 hypothesis: Swift prefills in 128-token chunks, mlx_lm in 512 — 8 chunk "
        + "forwards vs 2 for a 948-token prompt. Decode never touches this path.")

    let warm = ((try? FileManager.default.contentsOfDirectory(at: genDir, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    prewarmFiles(warm)
    Memory.clearCache()

    let t0 = Date()
    let encoder = try await GemmaEncoder.load(directory: genDir)
    print(String(format: "[gen-gap] load %.1fs", Date().timeIntervalSince(t0)))

    /// One generation through `ChatSession`, returning the phase split the session itself reports
    /// AND the wall-clock the caller actually pays. The bracket starts before `ChatSession.init`
    /// because that is where `--enhancer-bench` starts its own — a pass includes building the
    /// session, rendering the chat template and tokenizing, not just the two model phases.
    func chatSessionPass(prefill: Int?) async throws -> GapObs {
        var params = GenerateParameters(maxTokens: maxTokens, temperature: temperature, seed: seed)
        params.prefillStepSize = prefill
        let w = Date()
        let session = ChatSession(encoder.context, instructions: system, generateParameters: params)
        var info: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: userText) {
            if let i = item.info { info = i }
        }
        let wall = Date().timeIntervalSince(w)
        guard let i = info else { throw ProbeError.noInfo }
        return .init(promptTokens: i.promptTokenCount, promptSeconds: i.promptTime,
                     genTokens: i.generationTokenCount, genSeconds: i.generateTime,
                     wallSeconds: wall)
    }

    /// EXACTLY what `--enhancer-bench` does: fresh `ChatSession`, `respond(to:)`, `Date()` bracket.
    /// No `streamDetails`, so the model's own phase split is unavailable — which is precisely why
    /// AB-R-0080 had to infer prefill from a separate `maxTokens=1` run and read the remainder as
    /// decode. Reported as wall only; the phase columns are left at zero.
    func enhancerBenchPass() async throws -> GapObs {
        let params = GenerateParameters(maxTokens: maxTokens, temperature: temperature, seed: seed)
        let w = Date()
        let session = ChatSession(encoder.context, instructions: system, generateParameters: params)
        _ = try await session.respond(to: userText)
        return .init(promptTokens: 947, promptSeconds: 0, genTokens: maxTokens, genSeconds: 0,
                     wallSeconds: Date().timeIntervalSince(w))
    }

    /// One generation through the low-level loop — same prompt, no `ChatSession` in the path.
    /// Built through the context's own processor so the chat template is identical to the session's.
    func rawPass(prefill: Int?) async throws -> GapObs {
        var params = GenerateParameters(maxTokens: maxTokens, temperature: temperature, seed: seed)
        params.prefillStepSize = prefill
        let chat: [Chat.Message] = [.system(system), .user(userText)]
        let w = Date()
        let input = try await encoder.context.processor.prepare(input: .init(chat: chat))
        // Explicit closure type: `generate(input:parameters:context:didGenerate:)` is overloaded on
        // `([Int]) -> …` and `(Int) -> …`, so an inferred `_` is ambiguous.
        let result = try MLXLMCommon.generate(input: input, parameters: params,
                                              context: encoder.context) { (_: [Int]) in
            GenerateDisposition.more
        }
        return .init(promptTokens: result.promptTokenCount, promptSeconds: result.promptTime,
                     genTokens: result.generationTokenCount, genSeconds: result.generateTime,
                     wallSeconds: Date().timeIntervalSince(w))
    }

    /// Times ONLY the input-preparation step — chat-template render + tokenize — with no model
    /// forward at all. If the wall-vs-phase discrepancy lives here, this row names it outright.
    func preparePass() async throws -> GapObs {
        let chat: [Chat.Message] = [.system(system), .user(userText)]
        let w = Date()
        let input = try await encoder.context.processor.prepare(input: .init(chat: chat))
        let wall = Date().timeIntervalSince(w)
        return .init(promptTokens: input.text.tokens.size, promptSeconds: 0,
                     genTokens: 0, genSeconds: 0, wallSeconds: wall)
    }

    /// Tokenization alone, through the same tokenizer the encoder path uses. Splits the fixed
    /// `PREP` cost into "render the Jinja chat template" vs "run the tokenizer".
    func tokenizePass() -> GapObs {
        let w = Date()
        let ids = encoder.context.tokenizer.encode(text: system)
        let wall = Date().timeIntervalSince(w)
        return .init(promptTokens: ids.count, promptSeconds: 0, genTokens: 0, genSeconds: 0,
                     wallSeconds: wall)
    }

    func pass(_ arm: GapArm) async throws -> GapObs {
        switch arm.id {
        case "BENCH": return try await enhancerBenchPass()
        case "PREP": return try await preparePass()
        case "TOKZ": return tokenizePass()
        default:
            return arm.useChatSession ? try await chatSessionPass(prefill: arm.prefillStepSize)
                                      : try await rawPass(prefill: arm.prefillStepSize)
        }
    }

    // Warmup — shape-specific kernel compile lands here and is EXCLUDED from every row.
    let w0 = Date()
    _ = try? await chatSessionPass(prefill: nil)
    print(String(format: "[gen-gap] warmup %.1fs (excluded)\n", Date().timeIntervalSince(w0)))

    var obs: [String: [GapObs]] = [:]
    for round in 0 ..< rounds {
        let order = round % 2 == 0 ? arms : arms.reversed().map { $0 }
        for arm in order {
            let o = try await pass(arm)
            obs[arm.id, default: []].append(o)
            print(String(format: "[gen-gap] r%d %-6@ prefill %4d tok in %5.2fs = %6.1f tok/s   "
                                + "decode %3d tok in %5.2fs = %5.1f tok/s   WALL %5.2fs "
                                + "(unaccounted %5.2fs)",
                         round, arm.id as NSString, o.promptTokens, o.promptSeconds,
                         Double(o.promptTokens) / max(o.promptSeconds, 1e-9),
                         o.genTokens, o.genSeconds,
                         Double(o.genTokens) / max(o.genSeconds, 1e-9),
                         o.wallSeconds, o.wallSeconds - o.promptSeconds - o.genSeconds))
            fflush(stdout)
        }
    }

    // ───────── summary ─────────
    print("\n[gen-gap] ───── MEDIANS over \(rounds) rounds ─────")
    print("[gen-gap] arm     prefill_s  prefill_tok/s   decode_tok/s     wall_s  unacct_s   note")
    for arm in arms {
        guard let rs = obs[arm.id], !rs.isEmpty else { continue }
        let ps = median(rs.map(\.promptSeconds))
        let pTokS = median(rs.map { Double($0.promptTokens) / max($0.promptSeconds, 1e-9) })
        let dTokS = median(rs.map { Double($0.genTokens) / max($0.genSeconds, 1e-9) })
        let ws = median(rs.map(\.wallSeconds))
        let un = median(rs.map { $0.wallSeconds - $0.promptSeconds - $0.genSeconds })
        print(String(format: "[gen-gap] %-6@  %8.2f  %13.1f  %13.1f  %9.2f  %8.2f   %@",
                     arm.id as NSString, ps, pTokS, dTokS, ws, un, arm.note as NSString))
    }

    // The comparison the ask turns on: does the prefill chunk size move prefill, and does it leave
    // decode alone? Printed as a ratio against the DEFAULT arm so the reading needs no arithmetic.
    if let base = obs["P128"], let tuned = obs["P512"], !base.isEmpty, !tuned.isEmpty {
        let bp = median(base.map(\.promptSeconds)), tp = median(tuned.map(\.promptSeconds))
        let bd = median(base.map { Double($0.genTokens) / max($0.genSeconds, 1e-9) })
        let td = median(tuned.map { Double($0.genTokens) / max($0.genSeconds, 1e-9) })
        print(String(format: "\n[gen-gap] chunk 128 → 512: prefill %.2fs → %.2fs (%.2f×)   "
                            + "decode %.1f → %.1f tok/s (%.2f×)",
                     bp, tp, bp / max(tp, 1e-9), bd, td, td / max(bd, 1e-9)))
        print("[gen-gap] ⇒ if prefill moves and decode does not, AB-R-0080's 'uniform factor, one "
            + "systemic cause' reading does not hold: the two phases have different suspects.")
    }
    if let cs = obs["P512"], let raw = obs["RAW512"], !cs.isEmpty, !raw.isEmpty {
        let cp = median(cs.map(\.promptSeconds)), rp = median(raw.map(\.promptSeconds))
        let cd = median(cs.map { Double($0.genTokens) / max($0.genSeconds, 1e-9) })
        let rd = median(raw.map { Double($0.genTokens) / max($0.genSeconds, 1e-9) })
        print(String(format: "[gen-gap] ChatSession vs raw generate (both chunk 512): prefill "
                            + "%.2fs vs %.2fs   decode %.1f vs %.1f tok/s",
                     cp, rp, cd, rd))
        print("[gen-gap] ⇒ prices angle 2 (ChatSession/tokenizer overhead) directly.")
    }
    print("\n[gen-gap] DONE — diagnosis only; no default was changed by this run.")
}

private enum ProbeError: Error { case noInfo }
