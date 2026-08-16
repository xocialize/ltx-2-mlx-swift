// EnhancerBattery.swift — bench matrix arm E (`--enhancer-bench`), claim C5.
//
// WHAT IT MEASURES — two axes, both required:
//   1. **TIME**: per-prompt enhancement wall-clock over a fixed battery (short → detailed), with
//      the prefill token count, the generated token count and the resulting tok/s, so a slow case
//      can be attributed to length rather than left as a mystery. The model loads ONCE for the
//      battery, so a row is the PASS, not the pass plus a reload — that is the number the oracle
//      reported (3.1 s) and the only one comparable to it.
//   2. **MEMORY**: `phys_footprint` floor BEFORE the generator loads, floor AFTER it loads (their
//      difference is the enhancer's resident weight cost as its own number), PEAK across the whole
//      enhancement phase, and the floor again AFTER release.
//
// ⚠️ **Axis 2 is the point, and a time-only harness would miss the actual finding.** C5's verdict
// is the ORACLE's — bridge receipt **AB-R-0018** / `ltx-2-mlx/docs/claims/C5-prompt-enhancer.md`:
// claim **SUPPORTED**, a **3.1 s** pass (7 words → 113), and the pre-registered prediction that it
// would take 15–30 s was **wrong by ~5×**. This harness REPRODUCES that on the Swift seam; it does
// not re-litigate it. A Swift result that disagreed with the oracle indicates a PORT or seam
// difference, not a new finding about the model.
//
// 🔑 **The real cost is architectural, not temporal: a SECOND RESIDENT 12B.** LTX-2.5's own text
// encoder cannot enhance — `gemma4_unified` is encode-only (upstream requires a separate
// `--prompt-enhancer-gemma-root`; our Gemma-4 wrapper raises, and the oracle's does too). So
// enhancement adds a whole generative checkpoint (~7.5 GB at 4-bit) alongside a pipeline whose own
// 2.5 peak is already 42.27 GB against `standard64`'s 44.8 GB budget (AB-R-0035 / AB-D-0014). The
// vendor's "minimal extra compute" framing does not mention that, and for a local-first product
// with a memory governor it is the item that binds. Hence the phase-boundary phys sampling below:
// the harness has to be able to PRINT that number, not infer it.
//
// Structural check, no weights: the header reads `model_type` out of both configs — the 2.5 tree's
// `gemma4-12b-ltx-v1/config.json` (expected `gemma4_unified` → encode-only) and the generative
// checkpoint's (expected `gemma3`). That is the architectural claim reproduced in Swift for free;
// proving it the expensive way would mean loading a 24 GB encoder just to watch a cast fail.
//
// DETERMINISM (§V of LTX25-PORT-PLAN, C5 row): the official gemma4 enhancer recipe is GREEDY and
// therefore deterministic; the gemma3 recipe this seam runs SAMPLES at temp 0.7. The harness pins
// `GenerateParameters.seed` (default 10 — the oracle's `mx.random.seed(10)`) and then **measures**
// whether a re-run with the same seed reproduces the string, rather than asserting it: the 4-bit
// quantized path has documented small-N nondeterminism elsewhere in this project, and a single
// flipped sampled token changes the whole continuation. `--greedy` runs the temp-0 (ArgMax) arm,
// which is the official recipe's sampling mode. Whichever ran is echoed in the header.
//
// ⚠️ SEAM CAVEAT, and it is a real one: the SHIPPING seam `GemmaTextGenerator.generate(system:
// user:maxTokens:temperature:)` exposes **no seed** — it builds `GenerateParameters` without one,
// so the sampler is seeded from system entropy and the app's enhancement is NOT reproducible even
// though the field exists one line away. The battery arm therefore drives `GemmaEncoder` +
// `ChatSession` directly so it can pin the seed; the last phase then runs the real seam once,
// unseeded, to price what the app actually pays (load → generate → release) and to confirm the
// ~7.5 GB comes back. Both are labeled in the output — do not read the seam row as a seeded one.
//
// NOT COVERED HERE (deliberate): arm E's "same run ± enhancement pass" half. Comparing two full
// generations belongs to `--bench-e2e`'s ABBA protocol, and at ≥121f a single-block arm is
// uninterpretable (BENCH.md; AB-R-0066). This harness isolates the enhancer so that comparison has
// a per-pass number to be measured against. The QUALITY half of C5 (raw-vs-enhanced generations)
// is unmeasured on both sides.
//
// usage: RunLTX2 --enhancer-bench [gemmaGenerativeDir] [maxTokens] [seed] [--greedy] [--no-seam]

import Foundation
import LTX2
import MLX
import MLXLMCommon

/// One battery case. `kind` is reporting only — nothing gates on it.
private struct EnhancerCase {
    let id: String
    let kind: String        // short | medium | detailed | multishot
    let prompt: String
}

/// Fixed battery: two short prompts (the regime the oracle measured — a 7-word input), one
/// mid-length, one already-detailed (where the enhancer has less to add but more to prefill), and
/// one multishot (C1's shape — it is the longest realistic input a user types).
private let enhancerBattery: [EnhancerCase] = [
    .init(id: "E1", kind: "short",
          prompt: "a lighthouse keeper climbing the spiral stairs"),
    .init(id: "E2", kind: "short",
          prompt: "a red fox in snow"),
    .init(id: "E3", kind: "medium",
          prompt: "A woman walks across a sunlit room, sits down at a piano and begins to play "
                + "while rain runs down the window behind her."),
    .init(id: "E4", kind: "detailed",
          prompt: "A lone lighthouse on a rocky headland during a storm at dusk, waves exploding "
                + "against the rocks in sheets of white spray, the beam sweeping through heavy "
                + "rain. The camera starts low near the waterline and cranes slowly upward, "
                + "holding the tower against a bruised purple sky. Handheld, anamorphic, grain, "
                + "the roar of surf and wind under a low horn."),
    .init(id: "E5", kind: "multishot",
          prompt: "Shot 1: a chef's hands dicing onions on a worn wooden board, steam rising "
                + "behind. Shot 2: the same kitchen from the doorway as a server pushes through, "
                + "the chef looking up. Shot 3: close on the finished plate under a service lamp."),
]

/// Fallback system prompt — OURS (same instruction as `--gemma-textgen-gate`). Used only when the
/// oracle's `gemma_t2v_system_prompt.txt` is not on disk; the header always says which one ran,
/// because the two produce different output lengths and therefore different wall-clocks.
private let fallbackSystemPrompt = """
You are a prompt enhancer for a text-to-video model. Rewrite the user's brief as ONE flowing \
present-tense paragraph with explicit camera movement and a description of the audio. Output \
ONLY the enhanced prompt.
"""

/// `model_type` from a HuggingFace-style config.json, without loading a single weight.
private func modelType(ofDirectory dir: URL) -> String? {
    guard let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    return json["model_type"] as? String
}

/// Total bytes of `*.safetensors` in a directory — the on-disk weight size, for the resident-cost
/// line. Cheap (attributes only, no reads).
private func safetensorBytes(_ dir: URL) -> UInt64 {
    let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
    var total: UInt64 = 0
    for f in files where f.pathExtension == "safetensors" {
        let attrs = try? FileManager.default.attributesOfItem(atPath: f.path)
        total += (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
    }
    return total
}

private func wordCount(_ s: String) -> Int {
    s.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" }).count
}

func enhancerBatteryRun(gemmaDir: String?, maxTokens: Int, seed: UInt64,
                        greedy: Bool, runSeam: Bool) async throws {
    // ───────── resolved parameters (echo what ACTUALLY ran, never what was intended — a sweep
    // once collapsed silently in this project and nearly shipped a false finding) ─────────
    let genDir = URL(fileURLWithPath: gemmaDir ?? defaultGemma)
    guard FileManager.default.fileExists(atPath: genDir.path) else {
        print("[enhancer-bench] FAIL ❌ generative checkpoint not present: \(genDir.path)")
        print("[enhancer-bench] arm E needs a SECOND, generative Gemma — that IS the C5 finding.")
        exit(2)
    }
    let genType = modelType(ofDirectory: genDir) ?? "<no config.json>"
    let genBytes = safetensorBytes(genDir)

    // The 2.5 encoder, for the structural half of C5. Read only — no weights, no GPU.
    let ltxDir = URL(fileURLWithPath: "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
    let g4Dir = LTX2Pipeline.gemma4Dir(ltxDir: ltxDir)
    let g4Type = FileManager.default.fileExists(atPath: g4Dir.path)
        ? (modelType(ofDirectory: g4Dir) ?? "<no config.json>") : "<tree absent>"
    let g4Bytes = FileManager.default.fileExists(atPath: g4Dir.path) ? safetensorBytes(g4Dir) : 0

    // System prompt: prefer the ORACLE's, so the Swift pass is generating against the same
    // instruction the 3.1 s figure was measured with. Vendor text stays on disk in the oracle repo
    // (it is Lightricks', not ours) — read, never vendored into this Apache-2.0 tree.
    let oraclePrompt = packageRoot
        .deletingLastPathComponent()
        .appendingPathComponent("ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/"
                              + "text_encoders/gemma/encoders/prompts/gemma_t2v_system_prompt.txt")
    let system: String
    let systemSource: String
    if let text = try? String(contentsOf: oraclePrompt, encoding: .utf8), !text.isEmpty {
        system = text
        systemSource = "oracle gemma_t2v_system_prompt.txt"
    } else {
        system = fallbackSystemPrompt
        systemSource = "⚠️ FALLBACK (ours, short) — oracle file not found at \(oraclePrompt.path)"
    }
    // The oracle wraps the raw prompt exactly like this (`base_encoder.enhance_t2v`).
    func userMessage(_ p: String) -> String { "user prompt: \(p)" }

    let capEnv = ProcessInfo.processInfo.environment["LTX_CACHE_LIMIT_GB"]
    if let g = capEnv, let gb = Int(g) { Memory.cacheLimit = gb * 1_000_000_000 }
    let temperature: Float = greedy ? 0.0 : 0.7

    print("[enhancer-bench] ───── resolved parameters ─────")
    print("[enhancer-bench] generator dir  = \(genDir.path)")
    print(String(format: "[enhancer-bench] generator      = model_type=%@  weights-on-disk=%.2f GB",
                 genType as NSString, gbOf(genBytes)))
    print(String(format: "[enhancer-bench] 2.5 encoder    = %@  model_type=%@  weights-on-disk=%.2f GB",
                 g4Dir.lastPathComponent as NSString, g4Type as NSString, gbOf(g4Bytes)))
    if g4Type == "gemma4_unified" {
        print("[enhancer-bench] ⇒ ENCODE-ONLY: the 2.5 text encoder CANNOT enhance, so enhancement "
            + "adds a SECOND resident model. That is C5's real cost (AB-R-0018).")
    } else if g4Type != "<tree absent>" {
        print("[enhancer-bench] ⚠️ 2.5 encoder model_type is '\(g4Type)', expected 'gemma4_unified' "
            + "— re-check the tree before reading the architectural claim below.")
    }
    print("[enhancer-bench] path           = GemmaEncoder(gemma3) + ChatSession  [Swift generative seam]")
    print(String(format: "[enhancer-bench] sampling       = %@  maxTokens=%d  seed=%@",
                 (greedy ? "GREEDY (temperature 0 → ArgMaxSampler) — the official gemma4 recipe's mode"
                         : "SAMPLED (temperature 0.7 → CategoricalSampler) — the oracle's gemma3 recipe") as NSString,
                 maxTokens,
                 (greedy ? "n/a (inert at temp 0)" : String(seed)) as NSString))
    print("[enhancer-bench] determinism    = \(greedy ? "expected by construction" : "seed pinned") "
        + "— MEASURED below by re-running \(enhancerBattery[0].id) with the same seed, not asserted")
    print("[enhancer-bench] system prompt  = \(systemSource) (\(system.count) chars)")
    print("[enhancer-bench] user template  = \"user prompt: <prompt>\"  (oracle enhance_t2v)")
    print("[enhancer-bench] cases          = \(enhancerBattery.count)  seam arm = \(runSeam ? "yes" : "skipped")")
    print("[enhancer-bench] MLX cache cap  = \(capEnv.map { "\($0) GB (LTX_CACHE_LIMIT_GB)" } ?? "UNCAPPED (⚠️ overstates the app path)")")
    print("[enhancer-bench] C5's verdict is the ORACLE's (AB-R-0018: SUPPORTED, 3.1 s pass). This "
        + "reproduces it on Swift and prices the second resident model.")

    // ───────── tokenizer first: input token counts cost nothing and survive a failed load ─────────
    let tk = try await GemmaEncoder.loadTokenizer(directory: genDir)
    func tokenCount(_ s: String) -> Int {
        let (_, mask) = GemmaEncoder.tokenize(s, tokenizer: tk, maxLength: 8192)
        return Int(mask.sum().item(Int32.self))   // includes the tokenizer's leading <bos>
    }
    let systemTokens = tokenCount(system)
    print(String(format: "\n[enhancer-bench] system prompt = %d tokens (prefilled on EVERY pass)", systemTokens))

    // ───────── memory phase boundaries ─────────
    // Prewarm first so the load measures LOAD, not a cold read off the volume (I9/I5 discipline);
    // the shipping seam does the same thing internally.
    let warm = ((try? FileManager.default.contentsOfDirectory(at: genDir, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    let pw = Date()
    prewarmFiles(warm)
    print(String(format: "[enhancer-bench] prewarm %.1fs (%d files)", Date().timeIntervalSince(pw), warm.count))
    Memory.clearCache()

    struct Row {
        let c: EnhancerCase
        let seconds: Double
        let inTokens: Int
        let outTokens: Int
        let outWords: Int
        let text: String
    }
    var rows: [Row] = []
    var determinismNote = "not run"
    let floorBefore = physFootprintBytes()
    print(String(format: "[enhancer-bench] phys BEFORE generator load: %.2f GB", gbOf(floorBefore)))

    let sampler = PhysSampler(); sampler.start()
    var afterLoad: UInt64 = 0
    var loadSeconds = 0.0
    var peakEnhance: UInt64 = 0

    do {
        let t0 = Date()
        let encoder = try await GemmaEncoder.load(directory: genDir)
        loadSeconds = Date().timeIntervalSince(t0)
        afterLoad = physFootprintBytes()
        print(String(format: "[enhancer-bench] load %.1fs  phys AFTER generator load: %.2f GB  "
                            + "→ SECOND RESIDENT MODEL = %.2f GB",
                     loadSeconds, gbOf(afterLoad), gbOf(afterLoad &- floorBefore)))

        let params = GenerateParameters(maxTokens: maxTokens, temperature: temperature, seed: greedy ? nil : seed)
        func pass(_ prompt: String) async throws -> String {
            // A FRESH session per prompt: `ChatSession` carries KV/history across turns, and a
            // battery whose later cases inherit earlier ones is not measuring one-shot enhancement.
            let session = ChatSession(encoder.context, instructions: system, generateParameters: params)
            let out = try await session.respond(to: userMessage(prompt))
            return out.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Warmup compiles size-specific kernels; EXCLUDED from every reported number.
        let w0 = Date()
        _ = try? await ChatSession(encoder.context, instructions: system,
                                   generateParameters: GenerateParameters(maxTokens: 8,
                                                                          temperature: temperature,
                                                                          seed: greedy ? nil : seed))
            .respond(to: userMessage(enhancerBattery[0].prompt))
        print(String(format: "[enhancer-bench] warmup %.1fs (8 tokens, excluded)", Date().timeIntervalSince(w0)))

        sampler.resetMax()
        print("\n[enhancer-bench] id  kind        in→out tok    words      s     tok/s")
        for c in enhancerBattery {
            let inTok = tokenCount(c.prompt)
            let t = Date()
            let out = try await pass(c.prompt)
            let dt = Date().timeIntervalSince(t)
            let outTok = tokenCount(out)
            rows.append(Row(c: c, seconds: dt, inTokens: inTok, outTokens: outTok,
                            outWords: wordCount(out), text: out))
            print(String(format: "[enhancer-bench] %@  %-10@ %4d→%4d   %3d→%3d  %6.2f  %6.1f",
                         c.id as NSString, c.kind as NSString, inTok, outTok,
                         wordCount(c.prompt), wordCount(out), dt, Double(outTok) / dt))
            fflush(stdout)
        }
        peakEnhance = sampler.maxBytes()

        // ───────── determinism, MEASURED ─────────
        let repeatOut = try await pass(enhancerBattery[0].prompt)
        let identical = repeatOut == rows[0].text
        determinismNote = identical
            ? "REPRODUCIBLE — same \(greedy ? "greedy" : "seed=\(seed)") re-run produced a byte-identical string"
            : "NOT reproducible — same \(greedy ? "greedy" : "seed=\(seed)") re-run DIVERGED "
              + "(\(rows[0].text.count) vs \(repeatOut.count) chars)"
        print("\n[enhancer-bench] determinism: \(determinismNote)")
        if !identical {
            let prefix = zip(rows[0].text, repeatOut).prefix { $0 == $1 }.count
            print("[enhancer-bench] first divergence at char \(prefix) — one flipped sampled token "
                + "changes the whole continuation; do NOT read this as a port defect without repeating it")
        }
    }
    // Scope exit released the model; clearCache returns the MLX pool.
    Memory.clearCache()
    let floorAfter = physFootprintBytes()
    sampler.stop()
    print(String(format: "[enhancer-bench] phys AFTER release + clearCache: %.2f GB (was %.2f GB before load)",
                 gbOf(floorAfter), gbOf(floorBefore)))

    // ───────── the SHIPPING seam, once: load → generate → release, unseeded ─────────
    var seamSeconds = 0.0
    var seamPeak: UInt64 = 0
    if runSeam {
        let seamSampler = PhysSampler(); seamSampler.start()
        seamSampler.resetMax()
        let s0 = Date()
        let generator = GemmaTextGenerator(gemmaDirectory: genDir)
        let out = try await generator.generate(system: system, user: userMessage(enhancerBattery[0].prompt),
                                               maxTokens: maxTokens, temperature: temperature)
        seamSeconds = Date().timeIntervalSince(s0)
        seamPeak = seamSampler.maxBytes()
        seamSampler.stop()
        print(String(format: "\n[enhancer-bench] SEAM GemmaTextGenerator.generate (load→generate→release, "
                            + "UNSEEDED): %.2fs, %d words, peak %.2f GB",
                     seamSeconds, wordCount(out), gbOf(seamPeak)))
        Memory.clearCache()
        print(String(format: "[enhancer-bench] SEAM phys after: %.2f GB — the transient design returns the model",
                     gbOf(physFootprintBytes())))
    }

    // ───────── summary ─────────
    let times = rows.map(\.seconds).sorted()
    let median = times.isEmpty ? Double.nan
        : (times.count % 2 == 1 ? times[times.count / 2]
                                : (times[times.count / 2 - 1] + times[times.count / 2]) / 2)
    let mean = times.isEmpty ? Double.nan : times.reduce(0, +) / Double(times.count)
    let totalOut = rows.reduce(0) { $0 + $1.outTokens }
    let totalTime = rows.reduce(0.0) { $0 + $1.seconds }
    let resident = afterLoad > floorBefore ? afterLoad - floorBefore : 0
    let activation = peakEnhance > afterLoad ? peakEnhance - afterLoad : 0

    print("\n[enhancer-bench] ───── TIME (axis 1) ─────")
    print(String(format: "[enhancer-bench] per-pass  min %.2fs · median %.2fs · mean %.2fs · max %.2fs  "
                        + "(%d passes, %d tokens, %.1f tok/s aggregate)",
                 times.first ?? .nan, median, mean, times.last ?? .nan,
                 rows.count, totalOut, Double(totalOut) / max(totalTime, 1e-9)))
    print(String(format: "[enhancer-bench] model load (prewarmed) %.1fs — the oracle's warm load was 1.4s "
                        + "and its pass 3.1s (AB-R-0018)", loadSeconds))
    if let e1 = rows.first {
        print(String(format: "[enhancer-bench] oracle-comparable case %@: %d words → %d words in %.2fs "
                            + "(oracle: 7 → 113 in 3.1s)",
                     e1.c.id as NSString, wordCount(e1.c.prompt), e1.outWords, e1.seconds))
    }
    print("[enhancer-bench] ───── MEMORY (axis 2 — the real cost) ─────")
    print(String(format: "[enhancer-bench] floor before load %.2f GB → after load %.2f GB  "
                        + "⇒ SECOND RESIDENT MODEL %.2f GB", gbOf(floorBefore), gbOf(afterLoad), gbOf(resident)))
    print(String(format: "[enhancer-bench] PEAK phys across the enhancement phase: %.2f GB "
                        + "(generation activation over the resident model: %.2f GB)",
                 gbOf(peakEnhance), gbOf(activation)))
    print(String(format: "[enhancer-bench] released back to %.2f GB", gbOf(floorAfter)))
    // ARITHMETIC on two receipted numbers, NOT a measurement made here — flagged as such because
    // this project has been burned by projected figures read as measured ones (AB-D-0012).
    print(String(format: "[enhancer-bench] ARITHMETIC (not measured here): 2.5's own measured peak is "
                        + "42.27 GB (AB-R-0035, 448×320×9, evicted+capped) against standard64's 44.8 GB "
                        + "budget; +%.2f GB of enhancer would be %.2f GB. Enhancement and generation "
                        + "must be SEQUENTIAL on that tier, which is what the transient seam enforces.",
                 gbOf(resident), 42.27 + gbOf(resident)))
    print("[enhancer-bench] determinism: \(determinismNote)")
    print(String(format: "[enhancer-bench] SUMMARY arm=E claim=C5 path=gemma3-generative sampling=%@ "
                        + "seed=%@ maxTokens=%d cases=%d median_s=%.2f mean_s=%.2f load_s=%.1f "
                        + "resident=%llu peak=%llu seam_s=%.2f seam_peak=%llu",
                 (greedy ? "greedy" : "temp0.7") as NSString,
                 (greedy ? "n/a" : String(seed)) as NSString,
                 maxTokens, rows.count, median, mean, loadSeconds, resident, peakEnhance,
                 seamSeconds, seamPeak))

    print("\n[enhancer-bench] ───── enhanced prompts (operator read; the QUALITY half of C5 is unmeasured) ─────")
    for r in rows {
        print("[enhancer-bench] \(r.c.id) « \(r.c.prompt) »\n    → \(r.text)\n")
    }
    print("[enhancer-bench] DONE — C5's verdict is the ORACLE's (AB-R-0018, SUPPORTED); this arm "
        + "reproduces the Swift-side timing and prices the second resident 12B.")
}
