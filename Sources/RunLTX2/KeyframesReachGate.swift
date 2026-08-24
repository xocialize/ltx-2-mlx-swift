// KeyframesReachGate.swift — `--keyframes-reach-gate` (AB-T-0090).
//
// THE RULE. `ltx_core/tools.py:184` ends `create_initial_state` with:
//
//     return replace(state, keyframes_mask=self._first_frame_keyframes_mask(state))
//
//   "Because the video encoder is causal, the first temporal latent frame covers 1 pixel frame
//    while every later one covers temporal_scale_factor. That makes it the same token class as a
//    generated keyframe slot, and the reference implementation marks it UNCONDITIONALLY --
//    independently of whether any keyframe slots exist."
//
// Every pipeline reaches it: `_build_state` → `create_noised_state` → `create_initial_state`.
// There is no branch. So on 2.5 every video denoise entry must hand the loop a mask marking the
// target's first latent frame.
//
// WHY THIS IS A SOURCE AUDIT AND NOT A NUMERIC CHECK. `--denoise-wiring-gate` already proves the
// mask REACHES the DiT once a caller passes one, and `--dit-tiny-kf25-gate` proves it changes the
// output. Neither can see a caller that never passes one — the failure is an omission at the call
// site, and it is invisible to any assertion made below the call site. The pipelines build their
// DiT internally (`ensureDiT()`), so there is no seam to inject a recording denoiser through
// without putting test-only plumbing in shipping code. Auditing the call sites is the instrument
// that actually matches the defect.
//
// This is the gate that was missing when i2v, icT2V and retake all shipped without the mask, each
// citing the previous one as precedent.

import Foundation
import MLX
import LTX2

func keyframesReachGate() throws {
    // Locate the source tree from THIS file, so the gate audits the tree it was built from rather
    // than a hardcoded path that can drift or silently vanish.
    let here = URL(fileURLWithPath: #filePath)                      // Sources/RunLTX2/<this>
    let sourcesLTX2 = here.deletingLastPathComponent()              // Sources/RunLTX2
        .deletingLastPathComponent()                                // Sources
        .appending(path: "LTX2")
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: sourcesLTX2.path, isDirectory: &isDir),
          isDir.boolValue else {
        // A gate that cannot run must FAIL, not skip. A skip reads as a pass in a green sweep.
        print("[keyframes-reach-gate] FAIL — cannot find \(sourcesLTX2.path); this gate audits "
            + "source and cannot run without it")
        fflush(stdout); exit(1)
    }

    /// Does the call starting at `lines[i]` pass a `keyframesMask:` argument?
    /// Walks from the CALL's opening paren to its match, checking each line before its parens.
    func callPassesMask(_ lines: [String], _ i: Int, _ entry: String) -> Bool? {
        let line = lines[i]
        guard let callRange = line.range(of: entry == "runConditioned"
                                         ? "DenoiseLoop.runConditioned("
                                         : "DenoiseLoop.run(") else { return nil }
        var depth = 0, j = i, sawMask = false, started = false
        var scanFrom = String(line[callRange.lowerBound...])
        scan: while j < lines.count {
            // ⚠️ Check the line BEFORE walking its parens. The closing paren very often sits on
            // the SAME line as the last argument — `keyframesMask: kfMask)` — and breaking on the
            // paren first skipped exactly the argument being looked for.
            if scanFrom.contains("keyframesMask:") { sawMask = true }
            for ch in scanFrom {
                if ch == "(" { depth += 1; started = true }
                else if ch == ")" {
                    depth -= 1
                    if started && depth == 0 { break scan }
                }
            }
            j += 1
            if j >= lines.count || j > i + 40 { break }   // runaway guard
            scanFrom = lines[j]
        }
        return sawMask
    }

    // ── POISON SELF-CHECK, before auditing anything real ─────────────────────────────────────
    //
    // This scanner has been wrong TWICE — once counting parens from column 0 (so a destructuring
    // `let (v, _) = try DenoiseLoop.runConditioned(` closed depth before any argument), once
    // breaking on the closing paren before reading the line it sat on (so a trailing
    // `keyframesMask: kfMask)` was invisible). Both produced false alarms. A third bug in the
    // other direction would make this gate report a clean sweep over broken code, which is
    // exactly the failure a gate exists to prevent — so prove the detector discriminates on known
    // input first, including the two shapes that already fooled it.
    let poisonHas = [
        "        let (vfinal, afinalOpt) = try DenoiseLoop.runConditioned(",
        "            dit: dit, videoLatent0: v, audioLatent0: a, sigmas: s,",
        "            videoCleanLatent: clean, videoDenoiseMask: mask,",
        "            keyframesMask: kfMask)",
    ]
    let poisonLacks = [
        "        let (vfinal, afinalOpt) = try DenoiseLoop.runConditioned(",
        "            dit: dit, videoLatent0: v, audioLatent0: a, sigmas: s,",
        "            videoCleanLatent: clean, videoDenoiseMask: mask)",
        "        let kfMask = keyframesMask: notAnArgument",   // must NOT be counted: outside the call
    ]
    guard callPassesMask(poisonHas, 0, "runConditioned") == true else {
        print("[keyframes-reach-gate] FAIL — POISON CONTROL: the scanner cannot see a mask passed "
            + "as the trailing argument; it would report correct code as broken")
        fflush(stdout); exit(1)
    }
    guard callPassesMask(poisonLacks, 0, "runConditioned") == false else {
        print("[keyframes-reach-gate] FAIL — POISON CONTROL: the scanner reports a mask on a call "
            + "that has none; this gate would pass the very bug it exists for")
        fflush(stdout); exit(1)
    }
    print("[keyframes-reach-gate] poison control ✅ — detector distinguishes a call WITH a trailing "
        + "keyframesMask from one WITHOUT, and ignores text outside the call")

    let files = try FileManager.default.contentsOfDirectory(at: sourcesLTX2,
                                                            includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "swift" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    struct Site { let file: String; let line: Int; let entry: String; var passesMask: Bool }
    var sites: [Site] = []

    for url in files {
        let text = try String(contentsOf: url, encoding: .utf8)
        let lines = text.components(separatedBy: "\n")
        for (i, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.contains("DenoiseLoop.run(") || trimmed.contains("DenoiseLoop.runConditioned(")
            else { continue }
            let entry = trimmed.contains("runConditioned(") ? "runConditioned" : "run"
            guard let passes = callPassesMask(lines, i, entry) else { continue }
            sites.append(Site(file: url.lastPathComponent, line: i + 1, entry: entry,
                              passesMask: passes))
        }
    }

    guard !sites.isEmpty else {
        print("[keyframes-reach-gate] FAIL — found no DenoiseLoop call sites at all; the scan is "
            + "broken and would pass anything")
        fflush(stdout); exit(1)
    }

    print("[keyframes-reach-gate] audited \(sites.count) video-denoise call sites in Sources/LTX2:")
    for s in sites {
        print("    \(s.passesMask ? "✅" : "❌")  \(s.file):\(s.line)  \(s.entry)")
    }

    // ── SHAPE CASES. The source audit proves a mask is PASSED; it cannot see a WRONG one. The
    //    subtle error is the IC layout: `ICVideoState` appends reference tokens after the target,
    //    so the mask must span nv+refs while still marking only the target's first latent frame.
    //    Building it over `nv` alone, or marking proportionally, both "look right" at the call.
    var shapeFails: [String] = []
    let frame0 = 22 * 16, nv = 5 * frame0, refs = 2 * frame0     // 5 latent frames + 2 refs

    let plain = LTX2Pipeline.firstLatentFrameKeyframesMask(totalTokens: nv,
                                                           tokensPerLatentFrame: frame0)
    let ic = LTX2Pipeline.firstLatentFrameKeyframesMask(totalTokens: nv + refs,
                                                        tokensPerLatentFrame: frame0)
    eval(plain, ic)
    func marked(_ m: MLXArray) -> Int { Int(m.asType(.float32).sum().item(Float.self).rounded()) }

    if plain.dim(1) != nv { shapeFails.append("plain mask spans \(plain.dim(1)), expected \(nv)") }
    if marked(plain) != frame0 {
        shapeFails.append("plain mask marks \(marked(plain)) tokens, expected exactly one latent "
            + "frame (\(frame0))")
    }
    if ic.dim(1) != nv + refs {
        shapeFails.append("IC mask spans \(ic.dim(1)) but the IC latent is \(nv + refs) tokens — "
            + "built over the target only, so it cannot align with the appended references")
    }
    if marked(ic) != frame0 {
        shapeFails.append("IC mask marks \(marked(ic)) tokens, expected exactly \(frame0): the "
            + "appended REFERENCE tokens must stay 0 (oracle uses extend_keyframes_mask(marked: "
            + "false) for reference and keyframe-image conditionings; only keyframe SLOTS are marked)")
    }
    // The marked tokens must be the HEAD, not just the right count.
    let icHeadSum = marked(ic[0..., 0 ..< frame0, 0...])
    if icHeadSum != frame0 {
        shapeFails.append("IC mask's marked tokens are not the leading latent frame "
            + "(head sum \(icHeadSum) of \(frame0))")
    }
    print("[keyframes-reach-gate] shape: plain \(plain.dim(1))t/\(marked(plain)) marked · "
        + "IC \(ic.dim(1))t/\(marked(ic)) marked, all in the leading frame")

    let missing = sites.filter { !$0.passesMask }
    if missing.isEmpty && shapeFails.isEmpty {
        print("[keyframes-reach-gate] PASS ✅  every video-denoise entry passes a keyframesMask, so "
            + "the target's first latent frame is marked on every 2.5 forward")
        fflush(stdout)
    } else {
        for f in shapeFails { print("[keyframes-reach-gate] FAIL — \(f)") }
        for s in missing {
            print("[keyframes-reach-gate] FAIL — \(s.file):\(s.line) calls \(s.entry) with NO "
                + "keyframesMask; on 2.5 the target's first latent frame silently loses its "
                + "learned keyframes embedding on every forward (oracle marks it unconditionally, "
                + "ltx_core/tools.py:184)")
        }
        fflush(stdout); exit(1)
    }
}
