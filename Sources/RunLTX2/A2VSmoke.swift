// A2VSmoke.swift — end-to-end a2v on REAL weights (AB-D-0044 acceptance).
//
// t2v a short clip through the package, then feed that MP4 back as a `audio_to_video` VEditRequest
// with a DIFFERENT prompt. Two things must hold in the delivered MP4:
//
//   * the AUDIO is the source track, preserved — a2v's contract is that the supplied track is
//     ground truth and comes back unmodified (Desktop: "return original audio (not VAE-decoded)
//     for fidelity"), so the muxed track must still correlate with what went in;
//   * the VIDEO is genuinely regenerated — a different prompt over the same audio must not return
//     the source frames.
//
// The pure wiring claims (freeze holds bit-exact, the track conditions the video) are covered
// cheaply by `--a2v-contract-gate`; this proves the whole real chain — read → audio VAE encode →
// two-stage frozen-audio denoise → upsample → decode → mux — holds together on the checkpoint.
// ⚠️ Smoke, not a perceptual verdict: whether the video FITS the audio is the operator's call.

import Foundation
import MLX
import MLXLTX2
import MLXToolKit

func a2vSmoke() async throws {
    let cfg = LTX2Configuration(family: .ltx25,
                                modelsRootDirectory: URL(fileURLWithPath: "/Volumes/Satechi/Models"),
                                profile: .standard64)
    var c = cfg; c.quant = .int8
    c.ltxDirectory = URL(fileURLWithPath: "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
    c.transformerPath = URL(fileURLWithPath:
        "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx-ditq8/transformer-distilled.safetensors")
    let pkg = MLXLTX25Package(configuration: c)
    try await pkg.load()

    // 704x512: /64 on both axes, so a2v's snap-DOWN is a no-op and the spatial-x2 second stage
    // gets an even stage-2 latent grid (22x16). 33 frames ⇒ 5 latent frames.
    print("[a2v-smoke] 1/3 t2v source with audio (704×512×33, seed 42)…")
    let t2v = T2VRequest(prompt: "a drummer playing a fast rhythm on a snare drum",
                         numFrames: 33, fps: 24, width: 704, height: 512, seed: 42)
    let src = try await pkg.run(t2v) as! T2VResponse
    print("[a2v-smoke]     source mp4 \(src.video.data.count / 1024) KB")

    print("[a2v-smoke] 2/3 a2v against that track, DIFFERENT prompt (seed 4242)…")
    let a2v = VEditRequest(video: src.video,
                           prompt: "close-up of hands clapping in a bright room",
                           width: 704, height: 512, numFrames: 33, fps: 24,
                           seed: 4242, mode: VEditModes.audioToVideo)
    let tally = ProgressTally()
    let out = try await RunProgress.$sink.withValue({ r in tally.add(r) }) {
        () async throws -> VEditResponse in
        try await pkg.run(a2v) as! VEditResponse
    }
    print("[a2v-smoke]     a2v mp4 \(out.video.data.count / 1024) KB")
    let phases = tally.phases()
    print("[a2v-smoke]     progress phases: \(phases.sorted().joined(separator: ", ")) "
        + "(\(tally.count()) reports)")

    // Reported duration must be the real clip length. `out.video` is (B,C,F,H,W) and reading its
    // dim(1) yields the CHANNEL count — the bug this smoke would have caught (3/24 = 0.125 s).
    let expectedDuration = 33.0 / 24.0
    print(String(format: "[a2v-smoke]     durationSeconds=%.4f (expect %.4f)",
                 out.video.durationSeconds ?? -1, expectedDuration))

    print("[a2v-smoke] 3/3 audio preserved + video regenerated…")
    let dir = FileManager.default.temporaryDirectory
    let a = dir.appendingPathComponent("a2v-src.mp4"), b = dir.appendingPathComponent("a2v-out.mp4")
    try src.video.data.write(to: a); try out.video.data.write(to: b)

    // --- audio: must still be the source track ---
    //
    // Compared LAG-TOLERANTLY on purpose, and the reason is measured rather than assumed. A
    // strict lag-0 cosine scored 0.933 on a preserved track; searching lags found the true
    // alignment at ONE sample, worth 0.981. (An earlier draft of this comment blamed AAC encoder
    // priming delay — wrong: priming would be ~1000+ samples. A single sample is the group delay
    // of the x3 linear 16→48 kHz interpolation in the mux, and at 16 kHz one sample is half a
    // period at Nyquist, so on percussive content it really does cost ~5% correlation.)
    //
    // What a2v promises is that the CONTENT is the supplied track and not a VAE round-trip, so the
    // instrument is best correlation over a small lag window, bounded so a grossly shifted or
    // substituted track cannot buy a high score with a big offset, plus the reversed-track control
    // below to prove the metric can still fail.
    let wa = try await AudioInput.referenceWaveform(url: a)
    let wb = try await AudioInput.referenceWaveform(url: b)
    let n = min(wa.dim(2), wb.dim(2))
    let xa = wa[0..., 0..., 0 ..< n].flattened().asType(.float32)
    let xb = wb[0..., 0..., 0 ..< n].flattened().asType(.float32)
    eval(xa, xb)

    /// Best normalized correlation of `y` against `x` over lags in ±maxLag (coarse sweep, then a
    /// fine pass around the winner — the whole window at stride 1 is thousands of dot products).
    func bestLagCorrelation(_ x: MLXArray, _ y: MLXArray, maxLag: Int) -> (corr: Float, lag: Int) {
        let len = x.dim(0)
        func corrAt(_ lag: Int) -> Float {
            // lag > 0 ⇒ y is DELAYED: compare x[lag...] against y[..<len-lag].
            let (xs, ys) = lag >= 0
                ? (x[lag ..< len], y[0 ..< (len - lag)])
                : (x[0 ..< (len + lag)], y[(-lag) ..< len])
            let num = (xs * ys).sum().item(Float.self)
            let den = sqrt((xs * xs).sum().item(Float.self)) * sqrt((ys * ys).sum().item(Float.self))
            return den > 0 ? num / den : 0
        }
        var best = (corr: -Float.infinity, lag: 0)
        for lag in stride(from: -maxLag, through: maxLag, by: 16) {
            let c = corrAt(lag); if c > best.corr { best = (c, lag) }
        }
        for lag in max(-maxLag, best.lag - 16) ... min(maxLag, best.lag + 16) {
            let c = corrAt(lag); if c > best.corr { best = (c, lag) }
        }
        return best
    }

    let (audioCos, audioLag) = bestLagCorrelation(xa, xb, maxLag: 4000)   // ±0.25 s at 16 kHz
    // Bar is 0.95, not 0.98: the mux's linear resample is genuinely lossy on bright content, and
    // 0.981 measured leaves no headroom for a brighter track to pass honestly. Deliberately NOT
    // a weakened gate — the discrimination now comes from two added constraints (the lag bound
    // here and the reversed-track control below, which scores 0.12), and the measured value is
    // printed every run so real drift is still visible.
    print(String(format: "[a2v-smoke] audio best-lag correlation(src, a2v) = %.4f at lag %d samples "
                 + "(%.1f ms) — must be > 0.95 with |lag| <= 32 (measured 0.981 @ 1 sample)",
                 audioCos, audioLag, Double(audioLag) * 1000.0 / 16000.0))

    // CONTROL: the same lag-tolerant metric against the track REVERSED. Same spectrum, same
    // energy, different content — so a high score here would mean the metric cannot tell the
    // supplied track from something that merely sounds like it, and the check above proves nothing.
    let xbRev = xb[.stride(by: -1)]
    eval(xbRev)
    let (ctrlCos, _) = bestLagCorrelation(xa, xbRev, maxLag: 4000)
    print(String(format: "[a2v-smoke] control: same metric vs REVERSED track = %.4f "
                 + "(must be well below the real score)", ctrlCos))

    // --- video: must NOT be the source frames ---
    let fa = try await VideoInput.referenceClipFrames(url: a, width: 704, height: 512, frames: 33, fps: 24)
    let fb = try await VideoInput.referenceClipFrames(url: b, width: 704, height: 512, frames: 33, fps: 24)
    func frameCos(_ i: Int) -> Float {
        let x = fa[0..., 0..., i ..< (i + 1)].flattened(), y = fb[0..., 0..., i ..< (i + 1)].flattened()
        eval(x, y)
        return (x * y).sum().item(Float.self)
            / (sqrt((x * x).sum().item(Float.self)) * sqrt((y * y).sum().item(Float.self)) + 1e-9)
    }
    let vidCos = (0 ..< 33).map(frameCos).reduce(0, +) / 33
    print(String(format: "[a2v-smoke] mean frame cosine(src, a2v) = %.4f (must be < 0.98 — "
                 + "a2v regenerates, it does not pass the source through)", vidCos))

    var ok = true
    if !phases.contains(RunPhase.denoise.rawValue) {
        print("[a2v-smoke] FAIL ❌ — no denoise reports reached RunProgress"); ok = false
    }
    if abs((out.video.durationSeconds ?? -1) - expectedDuration) > 0.01 {
        print("[a2v-smoke] FAIL ❌ — durationSeconds is wrong"); ok = false
    }
    if !(audioCos > 0.95) {
        print("[a2v-smoke] FAIL ❌ — the supplied track was NOT preserved through a2v"); ok = false
    }
    if abs(audioLag) > 32 {
        print("[a2v-smoke] FAIL ❌ — the track correlates only after a \(audioLag)-sample shift; "
            + "a2v must return the supplied audio in place, not time-displaced"); ok = false
    }
    if !(ctrlCos < 0.5 && audioCos > ctrlCos + 0.3) {
        print("[a2v-smoke] FAIL ❌ — the reversed-track control scored \(ctrlCos) against a real "
            + "score of \(audioCos); this metric cannot distinguish the supplied track, so the "
            + "preservation check above is not evidence"); ok = false
    }
    if !(vidCos < 0.98) {
        print("[a2v-smoke] FAIL ❌ — output video matches the source; a2v is passing frames "
            + "through instead of generating against the audio"); ok = false
    }
    print(ok ? "[a2v-smoke] PASS ✅ — track preserved, video regenerated against it"
             : "[a2v-smoke] FAIL ❌")
    if !ok { exit(1) }
}
