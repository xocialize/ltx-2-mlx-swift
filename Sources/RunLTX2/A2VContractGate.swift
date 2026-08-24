// A2VContractGate.swift — the a2v denoise contract (AB-D-0044 follow-on).
//
// a2v's whole premise is "generate video AGAINST this track": the audio is held frozen and
// returned untouched, and the video must actually be conditioned on it. Both halves have a
// plausible silent failure:
//
//   * the freeze leaks — audio drifts over the steps, so the muxed track no longer matches the
//     video that was generated against it;
//   * the freeze becomes a DROP — audio is held but contributes nothing, and a2v degenerates into
//     an expensive t2v that ignores the track entirely.
//
// The second is the dangerous one: it produces perfectly good-looking video, passes any
// "did it run" check, and is only visible as "the video doesn't match the music". So this gate
// spends most of its cases proving the audio REACHES the video, with a determinism control so
// that difference can't be mistaken for noise.
//
// Tiny random weights are deliberate. Learned audio-video correlation needs the real checkpoint
// (that is what the e2e smoke is for), but WIRING is fully observable here in milliseconds.

import Foundation
import MLX
import LTX2

func a2vContractGate() throws {
    let dir = "\(goldensBase)/dit_tiny_kf25"
    let weights = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/weights.safetensors"))
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let dit = DiT(weights: weights, config: tinyDiTConfig())

    let vP = io["video_positions"]!, aP = io["audio_positions"]!
    let vT = io["video_text"]!, aT = io["audio_text"]!
    let nv = io["video_latent"]!.dim(1), na = io["audio_latent"]!.dim(1)
    // Latent channel width comes from the FIXTURE, not the real model's 128 — the tiny config
    // uses videoPatchChannels/audioPatchChannels = 8.
    let vD = io["video_latent"]!.dim(2), aD = io["audio_latent"]!.dim(2)
    let sigmas: [Float] = [1.0, 0.6, 0.3, 0.0]
    let zeroMask = MLXArray.zeros([1, na, 1])          // all-zero ⇒ FROZEN

    // Fixed video init across arms so any difference is attributable to the audio alone.
    MLXRandom.seed(7)
    let vInit = MLXRandom.normal([1, nv, vD]).asType(.float32)
    // Two DISTINCT driving tracks.
    MLXRandom.seed(11); let audioA = MLXRandom.normal([1, na, aD]).asType(.float32)
    MLXRandom.seed(12); let audioB = MLXRandom.normal([1, na, aD]).asType(.float32)
    eval(vInit, audioA, audioB)

    /// One a2v-shaped run: video denoises from `vInit`, audio is pinned to `frozen` (nil ⇒ the
    /// audio-free forward, i.e. no audio modality at all).
    func run(frozen: MLXArray?) throws -> (video: MLXArray, audio: MLXArray?) {
        let r = try DenoiseLoop.runConditioned(
            dit: dit, videoLatent0: vInit, audioLatent0: frozen, sigmas: sigmas,
            videoText: vT, audioText: aT, videoPositions: vP,
            audioPositions: frozen == nil ? nil : aP,
            audioCleanLatent: frozen, audioDenoiseMask: frozen == nil ? nil : zeroMask)
        eval(r.video); if let a = r.audio { eval(a) }
        return r
    }
    func bitwise(_ x: MLXArray, _ y: MLXArray) -> Bool { MLX.all(MLX.equal(x, y)).item(Bool.self) }
    func maxDiff(_ x: MLXArray, _ y: MLXArray) -> Float {
        MLX.max(MLX.abs(x.asType(.float32) - y.asType(.float32))).item(Float.self)
    }

    var fails: [String] = []

    // ── 1. THE FREEZE HOLDS, EXACTLY. Over every step the returned audio must be the latent we
    //       handed in — bitwise, not "close". This is also what makes `noise_scale=0.0` real: the
    //       vendor's audio spec never noises the encoded latent, so any drift here means we
    //       noised a track we promised to preserve.
    let arunA = try run(frozen: audioA)
    guard let outAudioA = arunA.audio else {
        print("[a2v-contract-gate] FAIL — frozen run returned no audio at all"); fflush(stdout); exit(1)
    }
    if !bitwise(outAudioA, audioA) {
        fails.append("case 1: frozen audio drifted over \(sigmas.count - 1) steps by "
            + "\(maxDiff(outAudioA, audioA)) — the track no longer matches the video generated against it")
    }
    print("[a2v-contract-gate] case 1 frozen audio bit-identical after \(sigmas.count - 1) steps: "
        + "\(bitwise(outAudioA, audioA))")

    // ── 2. DETERMINISM CONTROL, before any "it differs" claim. Same track, same init ⇒ identical
    //       video. Without this, case 3's difference could be nondeterminism and the gate would
    //       "prove" conditioning that isn't there.
    let arunA2 = try run(frozen: audioA)
    let selfDiff = maxDiff(arunA2.video, arunA.video)
    if selfDiff != 0 {
        fails.append("case 2: same audio + same init gave different video (\(selfDiff)) — the run "
            + "is nondeterministic, so case 3 cannot attribute anything to the audio")
    }
    print(String(format: "[a2v-contract-gate] case 2 determinism control: repeat run differs by %.6g (must be 0)", selfDiff))

    // ── 3. THE AUDIO REACHES THE VIDEO. Swap ONLY the frozen track; the video must move. If this
    //       is 0, a2v is generating video that ignores the track — the failure that still looks
    //       perfect on screen.
    let arunB = try run(frozen: audioB)
    let crossDiff = maxDiff(arunB.video, arunA.video)
    if crossDiff == 0 {
        fails.append("case 3: swapping the driving audio did NOT change the video — a2v is "
            + "ignoring the track and is just an expensive t2v")
    }
    print(String(format: "[a2v-contract-gate] case 3 different track ⇒ video moves by %.6g (must be > 0)", crossDiff))

    // ── 4. FROZEN ≠ ABSENT. A frozen modality could be held correctly and still be dropped from
    //       the forward; then the video would match the AUDIO-FREE run. It must not — a frozen
    //       track still participates through AV cross-attention (that is the whole mechanism).
    let arunNone = try run(frozen: nil)
    if arunNone.audio != nil {
        fails.append("case 4: the audio-free arm returned an audio tensor — it is not audio-free")
    }
    let vsAbsent = maxDiff(arunA.video, arunNone.video)
    if vsAbsent == 0 {
        fails.append("case 4: video with frozen audio is IDENTICAL to video with no audio "
            + "modality — the frozen track is being dropped, not conditioned on")
    }
    print(String(format: "[a2v-contract-gate] case 4 frozen-audio video vs audio-FREE video: %.6g (must be > 0)", vsAbsent))

    // ── 5. The conditioning signal must dominate the freeze's own numerical footprint: swapping
    //       tracks (case 3) has to move the video by a real fraction of what removing audio
    //       entirely does (case 4). A port that fed the audio in but crushed it to near-nothing
    //       would pass 3 and 4 on rounding dust; this asks for ≥1%.
    let ratio = vsAbsent > 0 ? crossDiff / vsAbsent : 0
    if !(ratio >= 0.01) {
        fails.append(String(format: "case 5: track-swap effect is only %.3f%% of the "
            + "audio-present-vs-absent effect — the audio is wired but numerically negligible",
            ratio * 100))
    }
    print(String(format: "[a2v-contract-gate] case 5 track-swap / audio-presence effect ratio: %.3f (must be >= 0.01)", ratio))

    if fails.isEmpty {
        print("[a2v-contract-gate] PASS ✅  5/5 — audio held bit-exact, run deterministic, and the "
            + "track measurably conditions the video (not dropped, not negligible)")
        fflush(stdout)
    } else {
        for f in fails { print("[a2v-contract-gate] FAIL — \(f)") }
        fflush(stdout)
        exit(1)
    }
}
