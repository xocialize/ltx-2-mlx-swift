// FrozenSigmaGate.swift — the oracle's FROZEN-MODALITY scalar-timestep rule.
//
// `modality_from_latent_state` (first-party `ltx_pipelines/utils/helpers.py:478`):
//
//     if state.frozen:
//         sigma = torch.zeros_like(sigma)
//     ... timesteps=timesteps_from_mask(state.denoise_mask, sigma)
//
// with the docstring "so prompt AdaLN and cross-modality gates match frozen conditioning
// (per-token timesteps are already zeroed via ``denoise_mask``)".
//
// The per-token half we always had (an all-zero denoise mask zeroes that side's per-token
// timesteps). The SCALAR half we did NOT: both modalities shared one scalar sigma, so a frozen
// modality's `av_ca_*_gate_adaln_single` and `*prompt_adaln_single` — the two scalar consumers a
// denoise mask cannot reach — were conditioned at the live step sigma instead of 0.
//
// This gate is deliberately mostly DISCRIMINATION. The fix's whole risk is that it silently
// changes a shipping path, so case 1 demands BITWISE equality on the fallback, and the rest prove
// each new lane actually moves something (a no-op fix would pass a looser gate).

import Foundation
import MLX
import LTX2

func frozenSigmaGate() throws {
    let dir = "\(goldensBase)/dit_tiny_kf25"
    let weights = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/weights.safetensors"))
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let dit = DiT(weights: weights, config: tinyDiTConfig())

    let vL = io["video_latent"]!, aL = io["audio_latent"]!
    let vT = io["video_text"]!, aT = io["audio_text"]!
    let vP = io["video_positions"]!, aP = io["audio_positions"]!
    let sig = io["sigma"]!
    let zero = MLXArray([Float(0)])

    func fwd(vs: MLXArray?, as aS: MLXArray?,
             vts: MLXArray? = nil, ats: MLXArray? = nil,
             sigma: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let (v, a) = dit(
            videoLatent: vL, audioLatent: aL, sigma: sigma ?? sig,
            videoText: vT, audioText: aT, videoPositions: vP, audioPositions: aP,
            videoTimesteps: vts, audioTimesteps: ats, keyframesMask: nil,
            videoSigma: vs, audioSigma: aS)
        eval(v, a!)
        return (v, a!)
    }
    // Bitwise, not approximate: the fallback must be the SAME GRAPH, not merely close.
    func bitwise(_ x: MLXArray, _ y: MLXArray) -> Bool {
        MLX.all(MLX.equal(x, y)).item(Bool.self)
    }
    func maxDiff(_ x: MLXArray, _ y: MLXArray) -> Float {
        MLX.max(MLX.abs(x.asType(.float32) - y.asType(.float32))).item(Float.self)
    }

    var fails: [String] = []

    // ── 1. FALLBACK IS EXACT. nil/nil must equal passing the shared sigma explicitly, bitwise,
    //       on both modalities. This is the regression guard for every shipping path (t2v, i2v,
    //       icT2V, DFR) — all of which pass nil and must not move by even an ulp.
    let (vBase, aBase) = fwd(vs: nil, as: nil)
    let (vExpl, aExpl) = fwd(vs: sig, as: sig)
    if !bitwise(vBase, vExpl) {
        fails.append("case 1 video: nil fallback ≠ explicit shared sigma (maxDiff \(maxDiff(vBase, vExpl)))")
    }
    if !bitwise(aBase, aExpl) {
        fails.append("case 1 audio: nil fallback ≠ explicit shared sigma (maxDiff \(maxDiff(aBase, aExpl)))")
    }
    print("[frozen-sigma-gate] case 1 fallback bitwise-exact: video=\(bitwise(vBase, vExpl)) audio=\(bitwise(aBase, aExpl))")

    // ── 2. AUDIO lane is live: zeroing ONLY the audio scalar must move the AUDIO output. If it
    //       does not, the audio-lane routing (av_ca_v2a_gate / audio_prompt_adaln) never happened.
    let (vAZero, aAZero) = fwd(vs: nil, as: zero)
    let dAudio = maxDiff(aAZero, aBase)
    if dAudio == 0 { fails.append("case 2: audioSigma=0 left the audio output untouched — audio lane not routed") }
    print(String(format: "[frozen-sigma-gate] case 2 audioSigma=0 → audio moves by %.6f (must be > 0)", dAudio))

    // ── 3. CROSS-MODAL REACH — the claim the fix rests on. A frozen modality's own output is
    //       discarded (its denoise mask restores the clean latent), so the fix would be cosmetic
    //       UNLESS its conditioning also reaches the OTHER modality. It does, through AV
    //       cross-attention: the frozen side's hidden states are what a2v/v2a attend to. Assert
    //       that, rather than asserting it in a comment — if this is 0 the fix is inert for
    //       replace_video/a2v and the doc comment on `DiT.callAsFunction` is wrong.
    let dVideoFromAudio = maxDiff(vAZero, vBase)
    if dVideoFromAudio == 0 {
        fails.append("case 3: audioSigma=0 did NOT change the video output — frozen-audio "
            + "conditioning does not reach video, so the fix is inert where it is claimed to matter")
    }
    print(String(format: "[frozen-sigma-gate] case 3 audioSigma=0 → VIDEO moves by %.6f via AV cross-attn (must be > 0)", dVideoFromAudio))

    // ── 4. VIDEO lane is live and INDEPENDENT of the audio lane: zeroing only the video scalar
    //       must move video, and must do so differently than zeroing only audio did. Without the
    //       second half, a bug that wired both params to the same lane would pass case 2 and 4.
    let (vVZero, _) = fwd(vs: zero, as: nil)
    let dVideo = maxDiff(vVZero, vBase)
    if dVideo == 0 { fails.append("case 4: videoSigma=0 left the video output untouched — video lane not routed") }
    if maxDiff(vVZero, vAZero) == 0 {
        fails.append("case 4b: videoSigma=0 and audioSigma=0 produced the SAME video — the two "
            + "params are wired to one lane")
    }
    print(String(format: "[frozen-sigma-gate] case 4 videoSigma=0 → video moves by %.6f; distinct from case 3: %@",
                 dVideo, maxDiff(vVZero, vAZero) != 0 ? "yes" : "NO"))

    // ── 5. END-TO-END FROZEN DETECTION in `runConditioned`. An all-zero audio denoise mask IS the
    //       oracle's `frozen` flag (it sets denoise_mask = zeros_like exactly then), so the loop
    //       must infer it and pass scalar 0 — with no new parameter at the call site.
    //
    //       Reference: one step at sigmas [1, 0] with plain Euler lands exactly on x0, so the
    //       returned video must equal `applyDenoiseMask`-free x0 from a forward whose audio scalar
    //       is 0 (and must NOT equal the one whose audio scalar is live).
    let aMaskFrozen = MLXArray.zeros([1, aL.dim(1), 1])
    let (vRun, _) = try DenoiseLoop.runConditioned(
        dit: dit, videoLatent0: vL, audioLatent0: aL, sigmas: [1.0, 0.0],
        videoText: vT, audioText: aT, videoPositions: vP, audioPositions: aP,
        audioCleanLatent: aL, audioDenoiseMask: aMaskFrozen)
    eval(vRun)

    // Hand-rolled single step. videoTimesteps stays nil (no video mask ⇒ scalar path), audio
    // per-token timesteps are the all-zero mask × sigma = zeros, exactly as the loop builds them.
    let atsZeros = (aMaskFrozen * Float(1.0)).squeezed(axis: -1)
    let one = MLXArray([Float(1)])   // the loop's sigmas[0] — NOT the fixture's sigma
    let (vRefFrozen, _) = fwd(vs: nil, as: zero, vts: nil, ats: atsZeros, sigma: one)
    let (vRefLive, _) = fwd(vs: nil, as: nil, vts: nil, ats: atsZeros, sigma: one)
    let x0Frozen = vL.asType(.float32) - Float(1.0) * vRefFrozen.asType(.float32)
    let x0Live = vL.asType(.float32) - Float(1.0) * vRefLive.asType(.float32)
    eval(x0Frozen, x0Live)
    let dToFrozen = maxDiff(vRun, x0Frozen), dToLive = maxDiff(vRun, x0Live)
    // The frozen reference must RECONSTRUCT the loop output (≈0), not merely be the nearer of
    // two bad fits. An earlier revision of this gate forwarded at the fixture's sigma instead of
    // the loop's, leaving both references ~0.22 away and 6e-6 apart — it "passed" on noise.
    if !(dToFrozen < 1e-5) {
        fails.append(String(format: "case 5: frozen reference does not reconstruct the loop output "
            + "(%.6g) — the gate is not measuring what it claims", dToFrozen))
    }
    if !(dToLive > 100 * max(dToFrozen, 1e-9)) {
        fails.append(String(format: "case 5: live reference is not clearly separated (%.6g vs %.6g) "
            + "— no discrimination", dToLive, dToFrozen))
    }
    print(String(format: "[frozen-sigma-gate] case 5 all-zero audio mask ⇒ frozen: dist(frozen-ref)=%.6g < dist(live-ref)=%.6g: %@",
                 dToFrozen, dToLive, dToFrozen < dToLive ? "yes" : "NO"))

    // ── 6. A PARTIAL mask is NOT frozen. i2v/icT2V pass masks that are 0 on conditioned tokens
    //       and 1 elsewhere; treating those as frozen would silently zero the scalar on our
    //       SHIPPING i2v path. Half-zero mask must behave like the live scalar, not the frozen one.
    let half = aL.dim(1) / 2
    let aMaskPartial = MLX.concatenated(
        [MLXArray.zeros([1, half, 1]), MLXArray.ones([1, aL.dim(1) - half, 1])], axis: 1)
    let (vPartial, _) = try DenoiseLoop.runConditioned(
        dit: dit, videoLatent0: vL, audioLatent0: aL, sigmas: [1.0, 0.0],
        videoText: vT, audioText: aT, videoPositions: vP, audioPositions: aP,
        audioCleanLatent: aL, audioDenoiseMask: aMaskPartial)
    eval(vPartial)
    let atsPartial = (aMaskPartial * Float(1.0)).squeezed(axis: -1)
    let (vPartRefLive, _) = fwd(vs: nil, as: nil, vts: nil, ats: atsPartial, sigma: one)
    let (vPartRefFrozen, _) = fwd(vs: nil, as: zero, vts: nil, ats: atsPartial, sigma: one)
    let x0PLive = vL.asType(.float32) - Float(1.0) * vPartRefLive.asType(.float32)
    let x0PFrozen = vL.asType(.float32) - Float(1.0) * vPartRefFrozen.asType(.float32)
    eval(x0PLive, x0PFrozen)
    let dPLive = maxDiff(vPartial, x0PLive), dPFrozen = maxDiff(vPartial, x0PFrozen)
    if !(dPLive < 1e-5) {
        fails.append(String(format: "case 6: a PARTIAL audio mask was treated as frozen — the LIVE "
            + "reference should reconstruct it exactly but is off by %.6g; this moves i2v", dPLive))
    }
    if !(dPFrozen > 100 * max(dPLive, 1e-9)) {
        fails.append(String(format: "case 6: no separation (%.6g vs %.6g)", dPFrozen, dPLive))
    }
    print(String(format: "[frozen-sigma-gate] case 6 partial audio mask ⇒ NOT frozen: dist(live-ref)=%.6g < dist(frozen-ref)=%.6g: %@",
                 dPLive, dPFrozen, dPLive < dPFrozen ? "yes" : "NO"))

    if fails.isEmpty {
        print("[frozen-sigma-gate] PASS ✅  6/6 — frozen modality gets scalar σ=0 on its own gate + "
            + "prompt AdaLN, it reaches the other modality through AV cross-attn, and nil/partial "
            + "masks are byte-for-byte unchanged")
        fflush(stdout)
    } else {
        for f in fails { print("[frozen-sigma-gate] FAIL — \(f)") }
        fflush(stdout)
        exit(1)
    }
}
