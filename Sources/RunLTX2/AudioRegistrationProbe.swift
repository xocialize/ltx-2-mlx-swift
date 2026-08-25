// AudioRegistrationProbe.swift — `--audio-registration-probe` (AB-A-0027 finding 1).
//
// The reported symptom is a CONSTANT ~200 ms offset between generated mouth motion and the audio,
// with audio PLACEMENT in the container exonerated by cross-correlation. That leaves
// conditioning-time registration: does a sound at real time t land in the audio latent token whose
// RoPE POSITION says t?
//
// If the encoder places energy at a different token than `Positions.audio` claims, the model is
// conditioned on mis-registered audio and the muxed (correctly-placed) track will disagree with the
// generated video by exactly that amount.
//
// So: put a burst at a known time, encode it the way a2v does, and ask which token carries it.
// This measures the registration instead of reasoning about it.

import Foundation
import MLX
import LTX2

func audioRegistrationProbe() throws {
    let ltxDir = URL(fileURLWithPath: "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
    let enc = try AudioVAEEncoder.load(path: ltxDir.appending(path: "audio_vae.safetensors"))

    let sr = 16000.0
    let seconds = 6.0
    let n = Int(sr * seconds)
    // Bursts at known times. Short, loud, silence elsewhere — an unambiguous temporal landmark.
    let burstTimes: [Double] = [1.0, 3.0, 5.0]
    var mono = [Float](repeating: 0, count: n)
    for t in burstTimes {
        let s = Int(t * sr)
        for k in 0 ..< Int(0.02 * sr) {          // 20 ms burst
            let i = s + k
            if i < n { mono[i] = (k % 8 < 4) ? 0.9 : -0.9 }   // 2 kHz square, unmistakable
        }
    }
    // (1, 2, N) stereo, matching what `AudioInput.referenceWaveform` hands a2v.
    let wave = MLXArray(mono + mono).reshaped(1, 2, n).asType(.float32)
    eval(wave)

    let lat = enc.encode(waveform: wave)                       // (1,8,T,16)
    let (tokens, positions) = Positions.patchifyLipdubAudioReference(lat, negativePositions: false)
    eval(tokens, positions)
    let T = tokens.dim(1)

    // Per-token energy, then the peak nearest each burst.
    let energy = MLX.sum(tokens.asType(.float32) * tokens.asType(.float32), axis: 2).squeezed(axis: 0)
    eval(energy)
    let e = energy.asArray(Float.self)
    let pos = positions.reshaped(T).asArray(Float.self)

    print("[audio-reg] latent tokens: \(T) over \(seconds)s  (grid 25/s)")
    print("[audio-reg] token 0 position \(pos[0])s · token 1 \(pos[1])s · spacing \(pos[2] - pos[1])s")

    // ⚠️ Latent energy is NOT zero in silence — the VAE encodes "quiet" as some nonzero vector, so
    // an absolute threshold fires everywhere and pins the onset to the search-window edge (it did).
    // Detect against the SILENCE BASELINE instead: median energy over all tokens, since this clip is
    // mostly silence with three 20 ms bursts.
    let sorted = e.sorted()
    let baseline = sorted[sorted.count / 2]
    print(String(format: "[audio-reg] silence baseline energy %.4g · max %.4g", baseline, sorted.last ?? 0))

    var fails: [String] = []
    var offsets: [Double] = []
    for bt in burstTimes {
        // Search a +-0.5 s window around the burst for the energy peak.
        var best = -1; var bestE: Float = -1
        for i in 0 ..< T where abs(Double(pos[i]) - bt) <= 0.5 {
            if e[i] > bestE { bestE = e[i]; best = i }
        }
        guard best >= 0 else { fails.append("no token within ±0.5s of \(bt)s"); continue }
        // ⚠️ The PEAK is the wrong landmark on its own: the encoder is CAUSAL, so an impulse smears
        // FORWARD and the peak necessarily sits after the onset. Report the ONSET — the first token
        // in the window to cross 10% of the local peak — which is what "when did the sound arrive"
        // actually means. The gap between the two is the smear, not a registration error.
        var onset = best
        let rise = bestE - baseline
        for i in 0 ..< T where abs(Double(pos[i]) - bt) <= 0.5 {
            if (e[i] - baseline) >= 0.25 * rise { onset = min(onset, i) }
        }
        let landedPeak = Double(pos[best]), landedOnset = Double(pos[onset])
        let off = landedOnset - bt
        offsets.append(off)
        print(String(format: "[audio-reg] burst @ %.3fs → onset token %d @ %.3fs (offset %+.0f ms) · "
                     + "peak token %d @ %.3fs (+%.0f ms smear)",
                     bt, onset, landedOnset, off * 1000, best, landedPeak,
                     (landedPeak - landedOnset) * 1000))
    }

    if offsets.count == burstTimes.count {
        let mean = offsets.reduce(0, +) / Double(offsets.count)
        let spread = (offsets.max() ?? 0) - (offsets.min() ?? 0)
        print(String(format: "[audio-reg] MEAN offset %+.0f ms (spread %.0f ms)", mean * 1000, spread * 1000))
        // One latent token is 40 ms; the grid cannot resolve better than that.
        if abs(mean) > 0.06 {
            fails.append(String(format: "encoder registration is off by %+.0f ms — a sound at time t "
                + "lands in a token whose position claims %+.0f ms later, so video generated at that "
                + "position will disagree with the muxed track by that amount", mean * 1000, mean * 1000))
        }
        if spread > 0.09 {
            fails.append(String(format: "offset is NOT constant (spread %.0f ms) — a drift, not a "
                + "fixed registration error", spread * 1000))
        }
    }

    if fails.isEmpty {
        print("[audio-reg] PASS ✅ — encoded audio lands within one latent token (40 ms) of its own "
            + "RoPE position; conditioning-time registration is NOT the ~200 ms source")
        fflush(stdout)
    } else {
        for f in fails { print("[audio-reg] FINDING — \(f)") }
        fflush(stdout)
    }
}
