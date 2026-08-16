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
// 🔑 **THE TOKENIZE ARM'S 1.14 s IS NOW ROOT-CAUSED, AND THE FIRST FRAMING WAS WRONG.** TOKZ
// priced `Tokenizer.encode` on the enhancer's system prompt at 1.14 s vs 0.543 ms for Python
// `tokenizers`, and AB-A-0011 reported that as "swift-transformers is ~2100× slower". It is not a
// general slowness. The five probes below (`tokenizeScaling` → `tokenizeInstanceVsProcess`) narrow
// it to: **swift-transformers builds the added-token splitter as ONE `NSRegularExpression` with a
// capture group PER added token** (`Tokenizer.swift:517-523`), and ICU pays O(total capture groups)
// per match attempt — so cost is **quadratic in the added-token count**, charged once per input
// character that can begin an added token. Gemma-3 ships 6,415 added tokens (6,322 starting `<`,
// 31 starting `\n`), hence ~29 ms per newline; the prompt's 39 newlines ARE the 1.14 s.
//
// ⚠️ **SCOPE: this does NOT touch the LTX-2.5 text encoder.** `gemma4-12b-ltx-v1` has 24 added
// tokens and prices `\n` at 0.003 ms. Only the Gemma-3 prompt enhancer pays it. Filed upstream as
// huggingface/swift-transformers#383 with a verified behaviour-preserving fix (emit `(?:…)` when
// neither `lstrip` nor `rstrip` is set: 1123 ms → 1.89 ms on a 40-line prompt, token IDs identical).
//
// 🔑 **THE METHOD TRAP, worth more than the bug.** The first standalone reproducer did NOT
// reproduce — because it loaded `gemma4-12b-ltx-v1` while the probe's slow path used the
// *generative* checkpoint. Same API, same library revision, different tokenizer, 10⁴× apart. A
// "cannot reproduce standalone" result nearly became "it's our process, not the library"; what
// saved it was `tokenizeInstanceVsProcess` showing a FRESH tokenizer was equally slow in-process,
// which forced the question of what else differed. **Check the reproducer consumes the same input
// as the thing it is reproducing before believing a negative.**
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

    /// Scaling sweep: is `encode` LINEAR in input length, or worse? A constant factor and a
    /// superlinear curve are very different bug reports — the second is actionable upstream in a
    /// way the first is not. Reports µs/token, which is flat iff the cost is linear.
    func tokenizeScaling() {
        let unit = "The quick brown fox jumps over the lazy dog. "
        _ = encoder.context.tokenizer.encode(text: "warm up")
        print("[gen-gap] ───── tokenizer scaling (swift-transformers encode, no model) ─────")
        var prev: (n: Int, s: Double)? = nil
        for reps in [8, 16, 32, 64, 128, 256] {
            let text = String(repeating: unit, count: reps)
            let n = encoder.context.tokenizer.encode(text: text).count
            var best = Double.infinity
            for _ in 0 ..< 3 {
                let t = Date(); _ = encoder.context.tokenizer.encode(text: text)
                best = min(best, Date().timeIntervalSince(t))
            }
            var note = ""
            if let p = prev {
                let tokRatio = Double(n) / Double(p.n), timeRatio = best / p.s
                note = String(format: "   ×%.2f tokens → ×%.2f time", tokRatio, timeRatio)
            }
            print(String(format: "[gen-gap] tokens=%5d  encode=%9.2f ms   µs/token=%8.1f%@",
                         n, best * 1000, best * 1e6 / Double(n), note as NSString))
            prev = (n, best)
        }
        print("[gen-gap] ⇒ flat µs/token = linear; rising = superlinear (the actionable case).")
    }

    /// Bisect: the scaling sweep says encode is LINEAR at ~3 µs/token, so 927 tokens should cost
    /// ~3 ms — yet TOKZ on the real system prompt costs ~1.14 s. Same call, same size, 400× apart.
    /// So the cost is CONTENT-dependent, not length-dependent. This narrows it to a substring.
    func tokenizeBisect() {
        let tk = encoder.context.tokenizer
        func ms(_ text: String) -> Double {
            var best = Double.infinity
            for _ in 0 ..< 3 {
                let t = Date(); _ = tk.encode(text: text)
                best = min(best, Date().timeIntervalSince(t))
            }
            return best * 1000
        }
        _ = tk.encode(text: "warm")
        print("[gen-gap] ───── content bisect (why is the real prompt 400× the synthetic?) ─────")
        let synth = String(repeating: "The quick brown fox jumps over the lazy dog. ",
                           count: system.count / 45)
        print(String(format: "[gen-gap] REAL   chars=%5d tokens=%4d  %9.2f ms",
                     system.count, tk.encode(text: system).count, ms(system)))
        print(String(format: "[gen-gap] SYNTH  chars=%5d tokens=%4d  %9.2f ms",
                     synth.count, tk.encode(text: synth).count, ms(synth)))

        // Halve repeatedly, always keeping the slower half — lands on the offending region.
        var lo = system.startIndex, hi = system.endIndex, depth = 0
        while system.distance(from: lo, to: hi) > 60, depth < 12 {
            let mid = system.index(lo, offsetBy: system.distance(from: lo, to: hi) / 2)
            let a = String(system[lo ..< mid]), b = String(system[mid ..< hi])
            let ta = ms(a), tb = ms(b)
            print(String(format: "[gen-gap]   depth %2d: first %4d chars %8.2f ms | second %4d chars %8.2f ms",
                         depth, a.count, ta, b.count, tb))
            if ta >= tb { hi = mid } else { lo = mid }
            depth += 1
        }
        let culprit = String(system[lo ..< hi])
        print("[gen-gap] ⇒ hot region (\(culprit.count) chars): \(culprit.debugDescription)")
    }

    /// The bisect landed on `": <style>, <rest of prompt>." De"` — 32 chars, 117 ms. Angle
    /// brackets. Gemma's vocab carries thousands of `<unusedN>`-style added tokens, so the
    /// suspicion is the added-token scan, not BPE. These probes separate the candidates.
    func tokenizeTrigger() {
        let tk = encoder.context.tokenizer
        func ms(_ t: String) -> Double {
            var best = Double.infinity
            for _ in 0 ..< 5 { let s = Date(); _ = tk.encode(text: t)
                               best = min(best, Date().timeIntervalSince(s)) }
            return best * 1000
        }
        _ = tk.encode(text: "warm")
        print("[gen-gap] ───── trigger isolation ─────")
        let cases: [(String, String)] = [
            ("plain word",            "style"),
            ("plain, 7 chars",        "abcdefg"),
            ("ONE angle pair",        "<style>"),
            ("open bracket only",     "<"),
            ("close bracket only",    ">"),
            ("two opens",             "<<"),
            ("four opens",            "<<<<"),
            ("eight opens",           "<<<<<<<<"),
            ("bracketed x1",          "<a>"),
            ("bracketed x2",          "<a><b>"),
            ("bracketed x4",          "<a><b><c><d>"),
            ("bracketed x8",          "<a><b><c><d><e><f><g><h>"),
            ("real special token",    "<start_of_turn>"),
            ("no-bracket same len",   "start_of_turn__"),
        ]
        for (name, text) in cases {
            print(String(format: "[gen-gap]   %-22@ chars=%3d tokens=%3d  %9.3f ms",
                         name as NSString, text.count, tk.encode(text: text).count, ms(text)))
        }
    }

    /// `<` costs ~30 ms — but the prompt has only TWO of them and costs 1159 ms, so `<` is a
    /// member of a slow class, not the trigger. Price every distinct character in the prompt
    /// individually, then check that count × unit-cost reconstructs the whole-prompt time. If it
    /// does, the cost is per-character and the slow set is fully identified.
    func tokenizeCharCost() {
        let tk = encoder.context.tokenizer
        func ms(_ t: String) -> Double {
            var best = Double.infinity
            for _ in 0 ..< 5 { let s = Date(); _ = tk.encode(text: t)
                               best = min(best, Date().timeIntervalSince(s)) }
            return best * 1000
        }
        _ = tk.encode(text: "warm")
        var counts: [Character: Int] = [:]
        for c in system { counts[c, default: 0] += 1 }
        var priced: [(Character, Int, Double)] = []
        for (c, n) in counts { priced.append((c, n, ms(String(c)))) }
        priced.sort { $0.2 * Double($0.1) > $1.2 * Double($1.1) }
        print("[gen-gap] ───── per-character cost (distinct chars in the real prompt) ─────")
        print("[gen-gap]   char        count   ms/char    total ms")
        var total = 0.0
        for (c, n, t) in priced {
            total += t * Double(n)
            if t * Double(n) > 5 {
                print(String(format: "[gen-gap]   %-10@ %5d %9.3f %11.1f",
                             String(c).debugDescription as NSString, n, t, t * Double(n)))
            }
        }
        let slow = priced.filter { $0.2 > 1.0 }
        print(String(format: "[gen-gap]   … %d distinct chars total; %d cost >1 ms each",
                     priced.count, slow.count))
        print("[gen-gap]   SLOW SET: " + slow.map { String($0.0).debugDescription }
                                             .joined(separator: " "))
        print(String(format: "[gen-gap] ⇒ Σ(count × ms/char) = %.0f ms vs whole-prompt %.0f ms",
                     total, ms(system)))
    }

    /// A standalone process, same swift-transformers 1.3.3, same model folder, same
    /// `AutoTokenizer.from(modelFolder:)`, prices "\n" at 0.003 ms — this process prices it at
    /// 29.6 ms. So the cost is NOT the library version. Load a SECOND tokenizer, freshly, right
    /// here: if the fresh one is fast, the slowness belongs to the model-loaded INSTANCE; if both
    /// are slow, it belongs to the process (MLX resident, memory pressure).
    func tokenizeInstanceVsProcess(dir: URL) async {
        func ms(_ tk: any MLXLMCommon.Tokenizer, _ t: String) -> Double {
            _ = tk.encode(text: "warm")
            var best = Double.infinity
            for _ in 0 ..< 5 { let s = Date(); _ = tk.encode(text: t)
                               best = min(best, Date().timeIntervalSince(s)) }
            return best * 1000
        }
        print("[gen-gap] ───── instance vs process ─────")
        let loaded = encoder.context.tokenizer
        print(String(format: "[gen-gap]   model-loaded instance   \\n = %8.3f ms   '<' = %8.3f ms",
                     ms(loaded, "\n"), ms(loaded, "<")))
        guard let fresh = try? await GemmaEncoder.loadTokenizer(directory: dir) else {
            print("[gen-gap]   fresh load FAILED"); return
        }
        print(String(format: "[gen-gap]   fresh AutoTokenizer     \\n = %8.3f ms   '<' = %8.3f ms",
                     ms(fresh, "\n"), ms(fresh, "<")))
        print("[gen-gap]   type(loaded)=\(type(of: loaded))  type(fresh)=\(type(of: fresh))")
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
    tokenizeScaling()
    tokenizeBisect()
    tokenizeTrigger()
    tokenizeCharCost()
    await tokenizeInstanceVsProcess(dir: genDir)
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
