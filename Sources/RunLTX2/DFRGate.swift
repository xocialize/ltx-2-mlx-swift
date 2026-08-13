// DFRGate.swift — `--dfr-gate`: the DFR ORCHESTRATION, not its parts.
//
// The parts are already gated elsewhere and every one of them was green while nothing called
// them: `--dfr-layout-gate` (canvas/tiles/stitch, bit-exact), `--keyframe-slots-gate` (slot
// append/extract/seeding), `--ancestral-step-gate` (the SDE step), `--denoise-wiring-gate` (the
// 2.5 deltas reach the loops). This gate covers what sits ON TOP of them: the frame contract, the
// two trims, the two fps, tile-local conditioning, the audio-free round forward, per-tile seeds,
// and the post-ancestral clean re-blend.
//
// Modelled on `--denoise-wiring-gate`: every arm asserts a feature CHANGES the result, and each
// arm carries a POISON control — a deliberately-broken variant that must FAIL the same assertion.
// A gate whose deciding check passes a broken arm is not a gate (AB-L-0017); this project has been
// bitten by that five times.
//
// It runs on the tiny kf25 fixture DiT and the arithmetic itself, so it costs seconds and needs no
// 38 GB checkpoint. The assembled path on real weights is `--dfr25` (and `RunLTX2 --e2e25`).

import Foundation
import LTX2
import MLX
import MLXRandom

private struct GateFailures {
    var items: [String] = []
    mutating func check(_ ok: Bool, _ message: @autoclosure () -> String) {
        if !ok { items.append(message()) }
    }
    /// Assert a POISON arm fails the same predicate a healthy arm passes.
    mutating func poison(_ brokenPasses: Bool, _ what: String) {
        if brokenPasses {
            items.append("POISON CONTROL: \(what) — the broken arm PASSES, so this check cannot "
                       + "detect the bug it exists for")
        }
    }
}

func dfrGate() throws {
    var f = GateFailures()

    // ── 1. Frame contract, trims, and the two fps ────────────────────────────────────────────
    //
    // `DFRPlan` is what `generateDFR` reads, so these are statements about the shipping path.
    // 25 requested frames pad to a 25-frame canvas on the 24-segment grid; 9 pad UP to 25, which
    // is the case that makes both trims observable.
    let fps = 24.0
    for rounds in 0 ... 2 {
        let p = try DFRPlan.resolve(requestedFrames: 25, height: 320, width: 448, fps: fps, rounds: rounds)
        let want = (25 - 1) * (1 << rounds) + 1
        f.check(p.deliveredFrames == want,
                "rounds=\(rounds): delivered \(p.deliveredFrames) != (requested-1)·2^rounds+1 = \(want)")
        f.check(p.finalCanvasFrames >= p.deliveredFrames,
                "rounds=\(rounds): canvas \(p.finalCanvasFrames) < delivered \(p.deliveredFrames)")
        f.check(p.playbackFPS == fps * Double(1 << rounds),
                "rounds=\(rounds): playbackFPS \(p.playbackFPS) != fps·2^rounds — the clip would mux "
                + "as \(1 << rounds)x slow motion with the right frames")
        f.check(p.roundPlans.count == rounds, "rounds=\(rounds): \(p.roundPlans.count) round plans")
        print(String(format: "[dfr-gate] rounds=%d  delivered=%-3d canvas=%-3d playbackFPS=%.0f condFPS=%.0f",
                     rounds, p.deliveredFrames, p.finalCanvasFrames, p.playbackFPS, p.conditioningFPS))
    }

    // Conditioning fps is capped INDEPENDENTLY of playback. At 24 fps × 2 rounds playback is 96,
    // so the cap must BIND here — otherwise this arm cannot see an uncapped port.
    let capped = try DFRPlan.resolve(requestedFrames: 25, height: 320, width: 448, fps: fps, rounds: 2)
    f.check(capped.conditioningFPS == 60,
            "conditioning fps \(capped.conditioningFPS) != 60 — an uncapped 96 fps time base halves "
            + "every token's temporal span vs training")
    f.poison(capped.conditioningFPS == capped.playbackFPS,
             "the cap does not bind at 24fps×2 rounds (playback \(capped.playbackFPS))")

    // Rounds outside 0…2 must be refused, not silently clamped.
    var rejected = false
    do { _ = try DFRPlan.resolve(requestedFrames: 25, height: 320, width: 448, fps: fps, rounds: 3) }
    catch { rejected = true }
    f.check(rejected, "rounds=3 was accepted; the layout only defines 2^round tiling for 0…2")

    // Audio trim: 9 requested frames pad to a 25-frame canvas, so audio generated for the canvas
    // outlasts the delivered picture (`FrameCodec` writes with no `-shortest`).
    let padded = try DFRPlan.resolve(requestedFrames: 9, height: 320, width: 448, fps: fps, rounds: 0)
    f.check(padded.canvas.canvasFrames > padded.requestedFrames,
            "9 frames did not pad — this arm needs a padding canvas to mean anything")
    f.check(padded.needsTrim, "padded canvas reports needsTrim=false")
    f.check(padded.audioKeepTokens < padded.audioTokens,
            "audio not trimmed (\(padded.audioKeepTokens) of \(padded.audioTokens)) — the muxed clip "
            + "would outlast the picture")
    let vidSecs = Double(padded.deliveredFrames) / fps
    let audSecs = Double(padded.audioKeepTokens) / 25.0     // 25 audio latents per second
    f.check(abs(vidSecs - audSecs) < 0.05,
            String(format: "trimmed audio %.3fs vs video %.3fs", audSecs, vidSecs))
    print(String(format: "[dfr-gate] trim  9f→canvas %d, keep %d latent frames, audio %d→%d (%.3fs vs video %.3fs)",
                 padded.canvas.canvasFrames, padded.keepLatentFrames,
                 padded.audioTokens, padded.audioKeepTokens, audSecs, vidSecs))

    // Generated-keyframe positions must be filtered to the delivered clip: unfiltered, a 9-frame
    // request reports a keyframe at pixel frame 24 of a 9-frame video.
    let allPositions = padded.canvas.slotPositions
    let keptIdx = padded.keptKeyframeIndices(allPositions)
    let kept = keptIdx.map { allPositions[$0] }
    f.check(kept.allSatisfy { $0 < padded.deliveredFrames },
            "reported keyframes \(kept) include positions outside the \(padded.deliveredFrames)-frame clip")
    f.poison(kept.count == allPositions.count,
            "the filter dropped nothing on a padded canvas (positions \(allPositions), "
            + "delivered \(padded.deliveredFrames))")
    print("[dfr-gate] keyframes  canvas \(allPositions) → delivered \(kept)")
    // …and a MIXED case, so a filter that simply drops everything cannot pass. 57 frames pad to a
    // 65-frame canvas with keyframes at 32 and 64: the first is inside the clip, the second is not.
    let mixed = try DFRPlan.resolve(requestedFrames: 57, height: 320, width: 448, fps: fps, rounds: 0)
    let mixedAll = mixed.canvas.slotPositions
    let mixedKept = mixed.keptKeyframeIndices(mixedAll).map { mixedAll[$0] }
    print("[dfr-gate] keyframes  canvas \(mixedAll) → delivered \(mixedKept) (\(mixed.deliveredFrames)f)")
    f.check(!mixedKept.isEmpty, "the filter dropped every keyframe on a canvas that contains one")
    f.check(mixedKept.count < mixedAll.count,
            "the filter kept a keyframe at/after frame \(mixed.deliveredFrames)")

    // ── 2. Stage-2 slot SEEDING vs re-rolling from noise ──────────────────────────────────────
    //
    // Stage 2 re-appends the slots seeded with their UPSAMPLED stage-1 content so the detailing
    // pass refines them. Re-rolling from noise would silently invent new keyframes — the video
    // would still generate, so only a numeric check catches it.
    let H = 4, W = 6, C = 128, F = 4
    let slotPositions = [24, 48]
    let n = F * H * W
    let stage2Sigma = Positions.stage2Sigmas[0]
    let seedContent = MLXRandom.normal([1, C, slotPositions.count, H, W], key: MLXRandom.key(99))
    let seedTokens = seedContent.transposed(0, 2, 3, 4, 1).reshaped(1, slotPositions.count * H * W, C)

    func stage2State(slotInitials: MLXArray?, sigma: Float = stage2Sigma) -> DFRState {
        LTX2Pipeline.dfrState(
            baseTokens: n, initialLatent: MLXRandom.normal([1, n, C], key: MLXRandom.key(5)),
            sigma: sigma, seed: 2,
            positions: Positions.video(F: F, H: H, W: W, fps: 24), H: H, W: W,
            slotPositions: slotPositions, slotFPS: 24, slotSigma: sigma,
            slotNoiseSeed: 30_000, slotInitials: slotInitials)
    }
    //
    // ⚠️ Do NOT read this as "the seeded block should look like the seed". Stage 2 starts at
    // σ=0.909375, so the seed's weight is only 1−σ ≈ 0.091 and a cosine against the seed is ~0.1
    // in BOTH arms — a threshold on that number would be measuring noise. Both arms draw the SAME
    // slot noise, so their difference isolates the seed exactly: seeded − rerolled ≡ (1−σ)·seed.
    let seeded = stage2State(slotInitials: seedContent)
    let rerolled = stage2State(slotInitials: nil)
    let seededSlots = seeded.latent[0..., n..., 0...]
    let rerolledSlots = rerolled.latent[0..., n..., 0...]
    let contribution = seededSlots - rerolledSlots
    let expected = seedTokens * (1.0 - stage2Sigma)
    eval(contribution, expected)
    let contribErr = maxAbs(contribution, expected)
    let contribMag = MLX.abs(contribution).max().item(Float.self)
    print(String(format: "[dfr-gate] slot seeding @σ=%.6f: seed weight %.4f, |contribution|max=%.4f, err vs (1−σ)·seed=%.2e",
                 stage2Sigma, 1 - stage2Sigma, contribMag, contribErr))
    f.check(contribMag > 1e-2,
            "the seed contributes nothing to the stage-2 slot block — stage 2 would re-invent every "
            + "keyframe from noise instead of refining the stage-1 one")
    f.check(contribErr < 1e-5,
            String(format: "seed contribution is not (1−σ)·seed (err %.3e) — the append is not a "
                         + "lerp(initial, noise, σ)", contribErr))
    // Discrimination: at σ=1.0 the seed MUST be fully discarded (that is stage 1), so the same
    // "contributes" check must FAIL there — proving it measures survival, not merely shape.
    let atOne = stage2State(slotInitials: seedContent, sigma: 1.0)
    let atOneContrib = atOne.latent[0..., n..., 0...] - stage2State(slotInitials: nil, sigma: 1.0).latent[0..., n..., 0...]
    let atOneMag = MLX.abs(atOneContrib).max().item(Float.self)
    f.poison(atOneMag > 1e-2,
             String(format: "the seed still contributes %.4f at σ=1.0; stage 1 must start slots from "
                          + "pure noise", atOneMag))

    // ── 3. State assembly: order, masks, and the keyframes-mask UNION ─────────────────────────
    //
    // Anchors are appended BEFORE slots so the slot layout's firstToken lands past them; an
    // anchor is clean and (1−strength)-masked, a slot is noised, mask 1, and MARKED.
    let anchorLatent = MLXArray.ones([1, H * W, C]) * 7.0
    let anchors = [KeyframeConditioning(frameIdx: 0, latent: anchorLatent, H: H, W: W, fps: 24,
                                        strength: 0.95)]
    let tile = LTX2Pipeline.dfrState(
        baseTokens: n, initialLatent: MLXRandom.normal([1, n, C], key: MLXRandom.key(5)),
        sigma: 0.975, seed: 7,
        positions: Positions.video(F: F, H: H, W: W, fps: 24), H: H, W: W,
        anchors: anchors,
        slotPositions: slotPositions, slotFPS: 24, slotSigma: 0.975, slotNoiseSeed: 40_000)
    let mask = LTX2Pipeline.dfrKeyframesMask(tile, tokensPerFrame: H * W)
    eval(tile.latent, tile.denoiseMask, mask)

    let anchorSpan = n ..< (n + H * W)
    let slotSpan = (n + H * W) ..< tile.latent.dim(1)
    f.check(tile.slotLayout?.firstToken == n + H * W,
            "slot layout starts at \(tile.slotLayout?.firstToken ?? -1), expected \(n + H * W) — "
            + "anchors must be appended BEFORE slots or extract() reads the wrong span")
    let anchorMask = tile.denoiseMask[0..., anchorSpan, 0...].max().item(Float.self)
    let slotMask = tile.denoiseMask[0..., slotSpan, 0...].min().item(Float.self)
    f.check(abs(anchorMask - 0.05) < 1e-5,
            "anchor denoise mask \(anchorMask) != 1−0.95; the seam keyframe would not be preserved")
    f.check(slotMask == 1, "slot denoise mask \(slotMask) != 1 — slots must be generated")
    f.check(!tile.isUniformMask,
            "tile mask reads as uniform, so the run would take the SCALAR-timestep loop and the "
            + "anchors would be denoised like ordinary tokens")
    // Anchors are clean context: latent == clean at those tokens (which is also why the loop's
    // initial re-blend is a no-op there).
    f.check(maxAbs(tile.latent[0..., anchorSpan, 0...], tile.clean[0..., anchorSpan, 0...]) == 0,
            "anchor tokens are not clean in the assembled state")
    // The union: frame 0 marked, slots marked, anchors NOT marked (ordinary keyframe conditioning
    // does not receive the learned embedding).
    let frame0 = mask[0..., ..<(H * W), 0...].min().item(Float.self)
    let midBase = mask[0..., (H * W) ..< n, 0...].max().item(Float.self)
    let anchorMark = mask[0..., anchorSpan, 0...].max().item(Float.self)
    let slotMark = mask[0..., slotSpan, 0...].min().item(Float.self)
    print(String(format: "[dfr-gate] assembly  firstToken=%d  masks anchor=%.2f slot=%.0f | marks f0=%.0f base=%.0f anchor=%.0f slot=%.0f",
                 tile.slotLayout?.firstToken ?? -1, anchorMask, slotMask,
                 frame0, midBase, anchorMark, slotMark))
    f.check(frame0 == 1, "first latent frame not marked (\(frame0))")
    f.check(midBase == 0, "base tokens past frame 0 are marked (\(midBase))")
    f.check(anchorMark == 0, "anchor tokens are marked (\(anchorMark)) — only generated slots are")
    f.check(slotMark == 1, "slot marks were WIPED by the frame-0 mask (\(slotMark)); the union must "
                         + "be an element-wise max, not an assignment")

    // ── 4. Conditioning is TILE-LOCAL ─────────────────────────────────────────────────────────
    //
    // frame_idx 0 means the tile's FIRST frame. Pinning a global seam index onto a mid-canvas tile
    // anchors the wrong frame, silently.
    let pixelStart = 48, globalSeams = [48, 96]
    let local = DFRLayout.remapPositionsToLocal(globalSeams, pixelStart: pixelStart)
    f.check(local == [0, 48], "remapPositionsToLocal(\(globalSeams), \(pixelStart)) = \(local)")
    let localPos = KeyframeConditioning.keyframePositions(frameIdx: local[1], H: H, W: W, fps: 24)
    let globalPos = KeyframeConditioning.keyframePositions(frameIdx: globalSeams[1], H: H, W: W, fps: 24)
    let localT = localPos[0, 0, 0].item(Float.self), globalT = globalPos[0, 0, 0].item(Float.self)
    f.check(abs(localT - (48.0 + 0.5) / 24.0) < 1e-5,
            String(format: "tile-local keyframe t=%.5f, expected %.5f", localT, (48.0 + 0.5) / 24.0))
    f.poison(abs(localT - globalT) < 1e-5,
             String(format: "global (t=%.5f) and local (t=%.5f) keyframe positions coincide", globalT, localT))
    print(String(format: "[dfr-gate] tile-local  global %@ @start %d → local %@  (t %.5f vs global %.5f)",
                 "\(globalSeams)" as NSString, pixelStart, "\(local)" as NSString, localT, globalT))

    // ── 5–7. Loop behaviour, on the tiny kf25 fixture DiT ────────────────────────────────────
    let dir = "\(goldensBase)/dit_tiny_kf25"
    let weights = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/weights.safetensors"))
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors"))
    let dit = DiT(weights: weights, config: tinyDiTConfig())
    let v = io["video_latent"]!, a = io["audio_latent"]!
    let vp = io["video_positions"]!, ap = io["audio_positions"]!
    let vt = io["video_text"]!, at = io["audio_text"]!
    let sigmas: [Float] = [1.0, 0.5, 0.0]

    // 5. Audio-free round forward. Rounds denoise VIDEO-ONLY; the video genuinely differs because
    //    it loses its A2V cross-attention residual (upstream's `run_a2v` gate).
    let withAudio = try DenoiseLoop.run(
        dit: dit, videoLatent0: v, audioLatent0: a, sigmas: sigmas,
        videoText: vt, audioText: at, videoPositions: vp, audioPositions: ap)
    let audioFree = try DenoiseLoop.run(
        dit: dit, videoLatent0: v, audioLatent0: nil, sigmas: sigmas,
        videoText: vt, audioText: at, videoPositions: vp, audioPositions: nil)
    eval(withAudio.video, audioFree.video)
    let a2vDelta = maxAbs(audioFree.video, withAudio.video)
    print(String(format: "[dfr-gate] audio-free  audio out=%@  video Δ vs audio-present=%.5f",
                 audioFree.audio == nil ? "nil" : "PRESENT", a2vDelta))
    f.check(audioFree.audio == nil, "the audio-free loop returned an audio latent")
    f.check(withAudio.audio != nil, "the audio-present loop returned nil audio")
    f.poison(a2vDelta == 0,
             "the audio-free video is byte-identical to the audio-present one, so the A2V path was "
             + "never contributing and this arm proves nothing")

    // 6. Per-tile ancestral seed. Tiles are positionally identical, so a SHARED seed injects
    //    byte-identical noise into every one of them.
    func ancestral(seed: UInt64) throws -> MLXArray {
        try DenoiseLoop.run(
            dit: dit, videoLatent0: v, audioLatent0: nil, sigmas: sigmas,
            videoText: vt, audioText: at, videoPositions: vp, audioPositions: nil,
            ancestralEta: 0.5, ancestralNoiseSeed: seed).video
    }
    let tile0 = try ancestral(seed: 1_000)
    let tile1 = try ancestral(seed: 1_001)
    let tile0again = try ancestral(seed: 1_000)
    eval(tile0, tile1, tile0again)
    let seedSpread = maxAbs(tile0, tile1)
    print(String(format: "[dfr-gate] tile seeds  seed→seed+1 Δ=%.5f   same-seed Δ=%.5f",
                 seedSpread, maxAbs(tile0, tile0again)))
    f.check(seedSpread > 1e-4,
            "per-tile ancestral seeds produce the same trajectory — every tile would receive "
            + "byte-identical noise")
    f.check(maxAbs(tile0, tile0again) == 0, "the same ancestral seed is not reproducible")

    // 7. Post-ancestral clean re-blend (oracle `post_process_latent`). The renoise is added to
    //    EVERY token, so without the re-blend a conditioned token drifts off its reference by
    //    exactly the injected noise.
    //    ⚠️ The schedule deliberately does NOT end at σ=0: at σ_next==0 the ancestral step returns
    //    the already-blended x0 and the bug would be invisible at the output.
    let nv = v.dim(1), half = nv / 2
    let clean = MLXArray.ones([1, nv, v.dim(2)]) * 3.0
    let condMask = MLX.concatenated([MLXArray.zeros([1, half, 1]),
                                     MLXArray.ones([1, nv - half, 1])], axis: 1)
    let pinned = try DenoiseLoop.runConditioned(
        dit: dit, videoLatent0: v, audioLatent0: nil, sigmas: [1.0, 0.5, 0.25],
        videoText: vt, audioText: at, videoPositions: vp, audioPositions: nil,
        videoCleanLatent: clean, videoDenoiseMask: condMask,
        ancestralEta: 0.5, ancestralNoiseSeed: 3).video
    eval(pinned)
    let pinDrift = maxAbs(pinned[0..., ..<half, 0...], clean[0..., ..<half, 0...])
    let freeDrift = maxAbs(pinned[0..., half..., 0...], clean[0..., half..., 0...])
    print(String(format: "[dfr-gate] clean re-blend  mask-0 drift=%.6f   mask-1 drift=%.5f", pinDrift, freeDrift))
    f.check(pinDrift == 0,
            "mask-0 tokens drifted \(pinDrift) under ancestral renoise — the post-ancestral "
            + "re-blend is missing, so anchor keyframes lose their reference every step")
    f.poison(freeDrift == 0,
             "the generated half also equals the clean reference, so this arm cannot tell a "
             + "re-blend from an all-clean output")

    if f.items.isEmpty {
        print("[dfr-gate] PASS ✅  frame contract, trims, fps split, seeding, tile-local "
            + "conditioning, audio-free rounds, per-tile seeds and the clean re-blend all hold")
        fflush(stdout)
    } else {
        for item in f.items { print("[dfr-gate] FAIL — \(item)") }
        fflush(stdout); exit(1)
    }
}
