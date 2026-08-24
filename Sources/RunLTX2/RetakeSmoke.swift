// RetakeSmoke.swift — end-to-end retake on REAL weights (AB-R-0133 piece 2 acceptance).
//
// t2v a tiny clip through the package, feed the MP4 straight back as a VEditRequest, and assert
// the SPAN CONTRACT on pixels: outside the span the retake must reproduce the source (held clean
// through every step); inside it must diverge (a different seed regenerates the region).
// ⚠️ Smoke, not a perceptual verdict — quality remains the operator's. This proves the chain
// (read → tiled encode → span-masked denoise → decode → mux) holds together and the masks land on
// the right frames in PIXELS, not just in the token grid the pure gate covers.

import Foundation
import MLX
import MLXLTX2
import MLXToolKit

func retakeSmoke() async throws {
    let cfg = LTX2Configuration(family: .ltx25,
                                modelsRootDirectory: URL(fileURLWithPath: "/Volumes/Satechi/Models"),
                                profile: .standard64)
    var c = cfg; c.quant = .int8
    c.ltxDirectory = URL(fileURLWithPath: "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
    c.transformerPath = URL(fileURLWithPath:
        "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx-ditq8/transformer-distilled.safetensors")
    let pkg = MLXLTX25Package(configuration: c)
    try await pkg.load()

    print("[retake-smoke] 1/3 t2v source (704×512×33, seed 42)…")
    let t2v = T2VRequest(prompt: "a red ball rolling across a wooden table",
                         numFrames: 33, fps: 24, width: 704, height: 512, seed: 42)
    let src = try await pkg.run(t2v) as! T2VResponse
    print("[retake-smoke]     source mp4 \(src.video.data.count / 1024) KB")

    print("[retake-smoke] 2/3 retake span [0.375, 1.0) (replace_video, seed 4242)…")
    let vedit = VEditRequest(video: src.video,
                             prompt: "a blue cube sliding across a wooden table",
                             seed: 4242, mode: VEditModes.replaceVideo,
                             metaData: [RetakeMetaKeys.start: .double(0.375),
                                        RetakeMetaKeys.duration: .double(0.625)])
    // PROGRESS BRIDGE (AB-A-0021 small ask). `runVideoEdit` used to install no core→engine
    // forward, so a retake reported nothing on the RunProgress plane while the app watched an
    // idle bar for minutes. Bind the sink the way the engine does and require real traffic:
    // an un-bridged build reaches the assertion with an EMPTY tally, so this fails closed.
    let tally = ProgressTally()
    let out = try await RunProgress.$sink.withValue({ r in tally.add(r) }) {
        () async throws -> VEditResponse in
        try await pkg.run(vedit) as! VEditResponse
    }
    print("[retake-smoke]     retake mp4 \(out.video.data.count / 1024) KB")
    let phases = tally.phases()
    print("[retake-smoke]     progress phases: \(phases.sorted().joined(separator: ", ")) "
        + "(\(tally.count()) reports)")
    // Denoise is the long pole — a bridge that only carried a start/finish event would leave the
    // bar frozen exactly where it was before, so require the STEPPED phase specifically.
    guard phases.contains(RunPhase.denoise.rawValue) else {
        print("[retake-smoke] FAIL ❌ — no '\(RunPhase.denoise.rawValue)' reports reached "
            + "RunProgress; the videoEdit progress bridge is not installed")
        exit(1)
    }
    guard tally.maxSteps() > 1 else {
        print("[retake-smoke] FAIL ❌ — denoise reported \(tally.maxSteps()) total steps; the "
            + "stepper has nothing to count")
        exit(1)
    }

    print("[retake-smoke] 3/3 span contract on decoded frames…")
    let dir = FileManager.default.temporaryDirectory
    let a = dir.appendingPathComponent("rt-src.mp4"), b = dir.appendingPathComponent("rt-out.mp4")
    try src.video.data.write(to: a); try out.video.data.write(to: b)
    // Span [0.5,1.0)@24fps by the START-boundary rule: latent frame k starts at (8k−7)/24 s, so
    // ≥0.5 ⇒ k ≥ 3 and <1.0 ⇒ k = 3 ONLY — pixel frames 17…24 regenerate. ⚠️ The first draft
    // probed pixel 16 as "inside": that is the LAST pixel of held latent frame 2, and the smoke
    // failed on its own arithmetic while the port was right — worth keeping as a reminder that
    // the probe indices need the same rigor as the mask they check.
    let fa = try await VideoInput.referenceClipFrames(url: a, width: 704, height: 512, frames: 33, fps: 24)
    let fb = try await VideoInput.referenceClipFrames(url: b, width: 704, height: 512, frames: 33, fps: 24)
    func frameCos(_ i: Int) -> Float {
        let x = fa[0..., 0..., i ..< (i + 1)].flattened(), y = fb[0..., 0..., i ..< (i + 1)].flattened()
        eval(x, y)
        return (x * y).sum().item(Float.self)
            / (sqrt((x * x).sum().item(Float.self)) * sqrt((y * y).sum().item(Float.self)) + 1e-9)
    }
    // Span [0.375, 1.0) frees latent frames 2 AND 3 (an earlier end of 1.05 silently
    // included latent 4 — start 25/24 s — and the 'after' probe failed INSIDE the span: the
    // mask was right and the probe was wrong, twice, which is why these indices are derived
    // in comments rather than eyeballed) → pixel frames 9…24; probes at 12 and 20.
    let before = frameCos(4), after = frameCos(30)
    let inside = min(frameCos(12), frameCos(20))
    print(String(format: "[retake-smoke] frame cos — before=%.4f inside=%.4f after=%.4f", before, inside, after))
    // Principled bar, not a magic margin: held frames define the noise band (double-H.264 ≈0.99),
    // and the regenerated frame must sit below it by ≥3× the held-frame SPREAD — the divergence
    // signal has to dominate the hold variance, whatever that variance is on this box.
    let heldFloor = min(before, after), heldSpread = max(abs(before - after), 0.001)
    let pass = before > 0.98 && after > 0.98 && (heldFloor - inside) >= 3 * heldSpread
    print(pass ? "[retake-smoke] PASS ✅ — span held outside, regenerated inside"
               : "[retake-smoke] FAIL ❌")
    if !pass { exit(1) }
}

/// Thread-safe tally for the progress assertions above — `RunProgress.Sink` is `@Sendable` and
/// the core reports from whatever task the pipeline is running on.
final class ProgressTally: @unchecked Sendable {
    private let lock = NSLock()
    private var seen: Set<String> = []
    private var n = 0
    private var maxTotal = 0
    func add(_ r: RunPhaseReport) {
        lock.lock(); defer { lock.unlock() }
        seen.insert(r.phase.rawValue)
        n += 1
        if r.phase == .denoise, let t = r.totalSteps { maxTotal = max(maxTotal, t) }
    }
    func phases() -> Set<String> { lock.lock(); defer { lock.unlock() }; return seen }
    func count() -> Int { lock.lock(); defer { lock.unlock() }; return n }
    func maxSteps() -> Int { lock.lock(); defer { lock.unlock() }; return maxTotal }
}
