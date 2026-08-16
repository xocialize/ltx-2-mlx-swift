// Gemma4TokenizerGate.swift — `--gemma4-tokenizer-gate`, the 2.5 tokenization tripwire.
//
// 🚨 **WHY THIS EXISTS: the 2.5 tokenize path was covered by NOTHING.**
// `--gemma-tokenizer-gate` exercises `GemmaEncoder.tokenize` — the 2.3 / Gemma-3 function — while
// LTX-2.5 actually runs `Gemma4Encoder.tokenize` (`LTX2Pipeline.swift:318`). Our own docs called the
// 2.3 gate "the tripwire for the Gemma-4 encoder", which it is not: it tests a different function.
// Same class as AB-L-0015 ("gate the WIRING, not just the component").
//
// 🔑 **IT ASSERTS THE VENDOR'S BEHAVIOUR, NOT OUR ORACLE'S.** AB-L-0045 is the reason: a golden
// dumped from our oracle proves Swift == oracle, and where the oracle inherits a bug the gate goes
// green on both being wrong. That is exactly what happened here — over-length truncation kept the
// LAST `maxLength` tokens in oracle AND Swift, and the 2.3 gate could never have seen it.
//
// The vendor semantics encoded below are VERIFIED, not assumed —
// `Lightricks/LTX-2` v1.2.0 `ltx_core/text_encoders/gemma/tokenizer.py:31-63`:
//   • BOS is prepended by hand (the 2.5 tokenizer's post_processor emits none)
//   • truncation is `tokenizer(truncation=True, max_length=…)` then `[bos, *ids][: max_length]`
//     — BOTH keep the FRONT. The vendor sets `padding_side` explicitly and never
//     `truncation_side`; the Gemma-4 `tokenizer_config.json` does not set it either, so HF's
//     default `"right"` applies. Measured on the real checkpoint: `truncation_side == "right"`,
//     and a >1024-token probe with distinct head/tail markers kept the HEAD and dropped the TAIL.
//   • left-pad to `max_length` with the native pad token.
//
// usage: RunLTX2 --gemma4-tokenizer-gate [gemma4Dir]

import Foundation
import LTX2
import MLX

func gemma4TokenizerGate(gemma4Dir: String?) async throws {
    let dir = URL(fileURLWithPath: gemma4Dir
        ?? "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx/gemma4-12b-ltx-v1")
    guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path) else {
        print("[gemma4-tokenizer-gate] FAIL ❌ no Gemma-4 dir at \(dir.path)"); exit(2)
    }
    print("[gemma4-tokenizer-gate] gemma4=\(dir.lastPathComponent)")
    // Tokenizer only — no 24 GB weight load. (Same trick as --gemma-tokenizer-gate: a tokenization
    // gate that needed the model would never be run.)
    let tk = try await GemmaEncoder.loadTokenizer(directory: dir)
    let ids = try Gemma4Encoder.specialTokenIds(directory: dir)
    let maxLength = 1024
    print("[gemma4-tokenizer-gate] bos=\(ids.bos) pad=\(ids.pad) maxLength=\(maxLength)")

    var failures: [String] = []
    func check(_ name: String, _ ok: Bool, _ detail: String) {
        print("[gemma4-tokenizer-gate] \(ok ? "✅" : "❌") \(name) — \(detail)")
        if !ok { failures.append(name) }
    }
    func run(_ text: String) -> (ids: [Int], mask: [Int]) {
        let (i, m) = Gemma4Encoder.tokenize(text, tokenizer: tk, bosId: ids.bos, padId: ids.pad,
                                            maxLength: maxLength)
        return (i.asArray(Int32.self).map(Int.init), m.asArray(Int32.self).map(Int.init))
    }

    // 1. Short prompt: BOS leads the valid span, left-padded, mask aligns.
    do {
        let (i, m) = run("a red fox in snow")
        let firstValid = m.firstIndex(of: 1) ?? -1
        check("01 shape", i.count == maxLength && m.count == maxLength, "\(i.count)/\(m.count)")
        check("02 left-padded", firstValid > 0 && i[0..<firstValid].allSatisfy { $0 == ids.pad },
              "first valid at \(firstValid)")
        check("03 BOS leads the valid span", firstValid >= 0 && i[firstValid] == ids.bos,
              "i[\(firstValid)]=\(firstValid >= 0 ? i[firstValid] : -1) (bos=\(ids.bos))")
    }

    // 2. 🔑 OVER-LENGTH — the case that was wrong, and that the 2.3 gate is structurally blind to.
    //    Distinct head/tail markers make the direction unambiguous: a suffix-truncating
    //    implementation keeps OMEGA and loses ALPHA *and* the BOS.
    do {
        let head = String(repeating: "ALPHA ", count: 40)
        let tail = String(repeating: "OMEGA ", count: 40)
        let long = head + String(repeating: "filler ", count: 4000) + tail
        let (i, m) = run(long)
        let headIds = tk.encode(text: "ALPHA"), tailIds = tk.encode(text: "OMEGA")
        func contains(_ needle: [Int]) -> Bool {
            guard let n = needle.last else { return false }   // last id = the content token
            return i.contains(n)
        }
        check("04 over-length fills the window", m.allSatisfy { $0 == 1 } && i.count == maxLength,
              "valid=\(m.reduce(0,+))/\(maxLength) (no padding when over-length)")
        check("05 BOS SURVIVES truncation", i.first == ids.bos,
              "i[0]=\(i.first ?? -1) (bos=\(ids.bos)) — suffix truncation would drop it")
        check("06 keeps the HEAD (vendor: truncation_side=right)", contains(headIds),
              "ALPHA present=\(contains(headIds))")
        check("07 drops the TAIL", !contains(tailIds),
              "OMEGA present=\(contains(tailIds)) — present would mean we kept the wrong end")
    }

    // 3. Exactly-full: prepending BOS must evict the LAST token, never the BOS.
    do {
        var probe = "x"
        var n = tk.encode(text: probe).count
        while n < maxLength { probe += " word"; n = tk.encode(text: probe).count }
        let (i, _) = run(probe)
        check("08 exactly-full keeps BOS at the front", i.first == ids.bos && i.count == maxLength,
              "len=\(i.count) i[0]=\(i.first ?? -1)")
    }

    print(failures.isEmpty
          ? "[gemma4-tokenizer-gate] PASS ✅ — 2.5 tokenization matches the VENDOR (v1.2.0), incl. "
            + "front-truncation with BOS preserved"
          : "[gemma4-tokenizer-gate] FAIL ❌ \(failures.count): \(failures.joined(separator: ", "))")
    if !failures.isEmpty { exit(1) }
}
