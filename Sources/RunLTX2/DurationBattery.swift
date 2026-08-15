// DurationBattery.swift — bench matrix arm D (`--duration-battery`), claim C6.
//
// `--duration-gate` proves the PORT is faithful (Swift == oracle on banked encodings). That is a
// different question from the one C6 asks, which is whether the head is HONEST: does the predicted
// duration track what the prompt actually implies? A parity gate cannot answer that — it would pass
// just as green on a head that returns a constant.
//
// So this battery encodes FRESH prompts through the real chain (Gemma-4 → connector → duration
// head) and reports predicted seconds and the snapped frame count against the duration each prompt
// implies. The encoder loads once; per-prompt encode is cheap.
//
// ⚠️ **The oracle already measured C6 and the verdict is PARTIAL** (AB-R-0015): responds to pacing
// (~37% longer for slow/extended), deterministic, correct clamp + grid — but short ≈ medium and NO
// response to explicit stated durations, over a narrow 3.4–6.4 s range. This battery exists to see
// whether the SWIFT port reproduces that shape, NOT to re-lit­igate the verdict. A Swift result that
// disagreed with the oracle would indicate a PORT bug, not a new finding about the model.
//
// usage: RunLTX2 --duration-battery [modelDir]

import Foundation
import LTX2
import MLX

/// Prompt battery. `implied` is the duration the prompt's language implies, used only for reporting
/// — it is a human judgement, not ground truth, and nothing gates on it.
private struct DurationCase {
    let group: String       // pacing | explicit | neutral
    let implied: String
    let prompt: String
}

private let durationBattery: [DurationCase] = [
    // --- pacing: the axis the oracle found the head DOES respond to
    .init(group: "pacing", implied: "very short",
          prompt: "A quick glance at a coffee cup on a table."),
    .init(group: "pacing", implied: "very short",
          prompt: "A camera flash goes off, one instant."),
    .init(group: "pacing", implied: "medium",
          prompt: "A woman walks across a room and sits down in a chair."),
    .init(group: "pacing", implied: "medium",
          prompt: "A cat jumps onto a windowsill and looks outside."),
    .init(group: "pacing", implied: "long",
          prompt: "A slow, sweeping pan across a wide mountain valley at dawn, "
                + "the camera drifting gradually from left to right."),
    .init(group: "pacing", implied: "long",
          prompt: "An extended, lingering shot of waves rolling slowly onto an empty beach, "
                + "unhurried, the tide creeping up the sand over a long stretch of time."),
    // --- explicit: the axis the oracle found the head IGNORES. Kept so the Swift port is checked
    //     against the same negative result — a port that suddenly "worked" here would be suspect.
    .init(group: "explicit", implied: "3 s stated",
          prompt: "A 3 second clip of a dog running through grass."),
    .init(group: "explicit", implied: "15 s stated",
          prompt: "A 15 second clip of a dog running through grass."),
    .init(group: "explicit", implied: "20 s stated",
          prompt: "A twenty second continuous shot of a city street at night."),
    // --- neutral control: no duration cue at all
    .init(group: "neutral", implied: "none",
          prompt: "A lighthouse on a rocky headland."),
]

func durationBatteryRun(modelDir: String?) async throws {
    let tree = URL(fileURLWithPath: modelDir ?? "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
    guard LTX2Pipeline.isLTX25(ltxDir: tree) else {
        print("[duration-battery] FAIL ❌ not a 2.5 tree: \(tree.path) — the duration head is 2.5-only")
        exit(2)
    }
    let g4 = LTX2Pipeline.gemma4Dir(ltxDir: tree)
    let headPath = tree.appendingPathComponent("duration_head.safetensors")
    guard FileManager.default.fileExists(atPath: headPath.path) else {
        print("[duration-battery] FAIL ❌ missing \(headPath.path)"); exit(2)
    }
    print("[duration-battery] tree=\(tree.lastPathComponent) cases=\(durationBattery.count)")

    let head = try DurationHead.load(path: headPath)
    let connector = try Connector.load(connectorPath: tree.appendingPathComponent("connector.safetensors"))
    print("[duration-battery] loading Gemma-4 encoder (~24 GB, once)…"); fflush(stdout)
    let t0 = Date()
    let encoder = try await Gemma4Encoder.load(directory: g4)
    // `loadTokenizer` lives on GemmaEncoder and is generation-agnostic (it builds the same
    // tokenizer `#huggingFaceLoadModel` installs); the 2.5-specific part is the BOS/pad ids below,
    // which come from the 2.5 tree — the 2.5 tokenizer's post_processor emits NO BOS, so
    // `Gemma4Encoder.tokenize` prepends it by hand (see AB-L-0011 / --gemma-tokenizer-gate).
    let tk = try await GemmaEncoder.loadTokenizer(directory: g4)
    let ids = try Gemma4Encoder.specialTokenIds(directory: g4)
    print(String(format: "[duration-battery] encoder ready %.1fs", Date().timeIntervalSince(t0)))

    struct Row { let c: DurationCase; let seconds: Double; let f24: Int }
    var rows: [Row] = []

    for c in durationBattery {
        let (tokenIds, mask) = Gemma4Encoder.tokenize(c.prompt, tokenizer: tk,
                                                      bosId: ids.bos, padId: ids.pad)
        let states = try encoder.allHiddenStates(tokenIds: tokenIds, attentionMask: mask)
        let (v, a) = connector(hiddenStates: states, mask: mask)
        eval(v, a)
        let y = head(videoTokens: v, audioTokens: a)
        eval(y)
        let secs = Double(y[0].item(Float.self))
        let f24 = DurationHead.secondsToNumFrames(secs, frameRate: 24)
        rows.append(Row(c: c, seconds: secs, f24: f24))
        print(String(format: "[duration-battery] %-9@ %-12@ → %6.2fs  %4df@24  | %@",
                     c.group as NSString, c.implied as NSString, secs, f24,
                     String(c.prompt.prefix(56)) as NSString))
        fflush(stdout)
    }

    // --- Reporting. Nothing here GATES: C6's verdict is the oracle's (AB-R-0015). These are the
    // same shape checks, so a Swift/oracle disagreement surfaces as a port smell.
    func mean(_ xs: [Double]) -> Double { xs.isEmpty ? .nan : xs.reduce(0,+) / Double(xs.count) }
    let pacingShort = rows.filter { $0.c.group == "pacing" && $0.c.implied == "very short" }.map(\.seconds)
    let pacingMed   = rows.filter { $0.c.group == "pacing" && $0.c.implied == "medium" }.map(\.seconds)
    let pacingLong  = rows.filter { $0.c.group == "pacing" && $0.c.implied == "long" }.map(\.seconds)
    let explicit    = rows.filter { $0.c.group == "explicit" }
    let all         = rows.map(\.seconds)

    print("\n[duration-battery] ───── shape ─────")
    print(String(format: "[duration-battery] pacing  short %.2fs · medium %.2fs · long %.2fs",
                 mean(pacingShort), mean(pacingMed), mean(pacingLong)))
    let pacingResponds = mean(pacingLong) > mean(pacingShort) * 1.15
    print("[duration-battery] responds to PACING (long > short by >15%): \(pacingResponds ? "YES" : "NO")"
          + String(format: "  (long/short = %.3f×)", mean(pacingLong) / mean(pacingShort)))
    let shortVsMed = abs(mean(pacingMed) - mean(pacingShort)) / mean(pacingShort)
    print(String(format: "[duration-battery] short vs medium separation: %.1f%% %@",
                 shortVsMed * 100, (shortVsMed < 0.10 ? "(TIED — matches the oracle's finding)" : "(separated)") as NSString))

    if let lo = explicit.first(where: { $0.c.implied.hasPrefix("3 ") }),
       let hi = explicit.first(where: { $0.c.implied.hasPrefix("15 ") }) {
        let ratio = hi.seconds / lo.seconds
        print(String(format: "[duration-battery] EXPLICIT '3 second' %.2fs vs '15 second' %.2fs → %.3f× %@",
                     lo.seconds, hi.seconds, ratio,
                     (ratio < 1.15 ? "— IGNORES stated duration (matches the oracle)" : "— RESPONDS (⚠️ diverges from the oracle; suspect a PORT change)") as NSString))
    }
    print(String(format: "[duration-battery] range %.2f–%.2fs over %d prompts",
                 all.min() ?? .nan, all.max() ?? .nan, all.count))
    // The clamp is a real behaviour, not a curiosity: the oracle saw video_only predict 24.8s →
    // clamped to 20s → 481 frames, so the upper bound IS exercised in practice.
    let clamped = rows.filter { $0.seconds >= 19.99 || $0.seconds <= 1.01 }
    print("[duration-battery] cases hitting the [1,20]s clamp: \(clamped.count)")
    print("[duration-battery] DONE — C6's verdict is the oracle's (AB-R-0015, PARTIAL); "
          + "this reports whether the Swift port reproduces that SHAPE.")
}
