// KeyframeCondGate.swift — `--keyframe-cond-gate` (AB-A-0025).
//
// Pins the MECHANISM, because the mechanism is the part that is easy to get plausibly wrong. The
// app's original ask proposed pinning the final latent frame in place and MARKING it in
// `keyframesMask`. The oracle does neither: `VideoConditionByKeyframeIndex` APPENDS a token block
// with its own RoPE positions, and passes `extend_keyframes_mask(..., marked=False)` because
// `marked=True` is reserved for GENERATED slots.
//
// Both wrong versions would still produce a video. Endpoint PSNR would catch a total failure but
// not a subtly mis-positioned keyframe, and nothing else in the suite constrains this. So the gate
// asserts the shape of the conditioning itself, not just that a run completes.

import Foundation
import MLX
import LTX2
import MLXLTX2
import MLXToolKit

func keyframeCondGate() throws {
    var fails: [String] = []
    let H = 4, W = 6, F = 3, C = 8, fps: Float = 24
    let nv = F * H * W

    MLXRandom.seed(3)
    let target = MLXRandom.normal([1, nv, C]).asType(.float32)
    let kfTokens = MLXRandom.normal([1, H * W, C]).asType(.float32)
    eval(target, kfTokens)

    let frameIdx = 16
    let kf = KeyframeConditioning(frameIdx: frameIdx, latent: kfTokens, H: H, W: W,
                                  fps: fps, strength: 1.0)
    let vPos = Positions.video(F: F, H: H, W: W, fps: fps)
    // Target's own leading latent frame marked; nothing else — the AB-T-0090 rule.
    let kfMask = LTX2Pipeline.firstLatentFrameKeyframesMask(totalTokens: nv, tokensPerLatentFrame: H * W)

    let st = KeyframeConditionedState.append(
        latent: target,
        clean: MLXArray.zeros(target.shape).asType(.float32),
        denoiseMask: MLXArray.ones([1, nv, 1]).asType(.float32),
        positions: vPos, keyframesMask: kfMask, items: [kf])
    eval(st.latent, st.denoiseMask, st.positions)

    // ── 1. APPENDED, not pinned in place. Token count must GROW by exactly one latent frame.
    if st.latent.dim(1) != nv + H * W {
        fails.append("case 1: token count \(st.latent.dim(1)), expected \(nv + H * W) — a keyframe "
            + "must be APPENDED, not written into an existing latent frame")
    }
    if st.targetTokens != nv || st.slice(st.latent).dim(1) != nv {
        fails.append("case 1b: slice() does not recover exactly the \(nv) target tokens")
    }
    print("[keyframe-cond-gate] case 1 appended: \(nv) → \(st.latent.dim(1)) tokens, slice→\(st.slice(st.latent).dim(1))")

    // ── 2. POSITIONS. Temporal coord of the appended block must be (frameIdx + 0.5)/fps: the
    //       oracle offsets by frameIdx, narrows a single-pixel-frame keyframe to [start, start+1),
    //       and applies NO causal_fix at frameIdx != 0.
    let appended = st.positions[0..., nv..., 0...]
    let tCoord = appended[0, 0, 0].item(Float.self)
    let want = (Float(frameIdx) + 0.5) / fps
    if abs(tCoord - want) > 1e-6 {
        fails.append(String(format: "case 2: appended temporal coord %.6f, expected %.6f "
            + "((frameIdx + 0.5)/fps)", tCoord, want))
    }
    // Every appended token shares the one temporal coord (it is a single frame).
    let tSpread = MLX.max(MLX.abs(appended[0..., 0..., 0 ..< 1] - MLXArray(tCoord))).item(Float.self)
    if tSpread > 1e-6 {
        fails.append("case 2b: appended tokens do not share one temporal coord (spread \(tSpread))")
    }
    print(String(format: "[keyframe-cond-gate] case 2 temporal coord %.6f == (%d + 0.5)/%.0f ✓", tCoord, frameIdx, fps))

    // ── 3. NOT MARKED. The appended tokens must be 0 in keyframesMask; the target's leading frame
    //       must still be 1. Marking a supplied image applies an embedding it never trained with.
    guard let m = st.keyframesMask else {
        print("[keyframe-cond-gate] FAIL — keyframesMask vanished on append"); fflush(stdout); exit(1)
    }
    eval(m)
    let markedAppended = MLX.max(m[0..., nv..., 0...]).item(Float.self)
    let markedHead = MLX.min(m[0..., 0 ..< (H * W), 0...]).item(Float.self)
    if markedAppended != 0 {
        fails.append("case 3: appended keyframe tokens are MARKED (\(markedAppended)) — marked=True "
            + "is reserved for GENERATED slots (oracle: extend_keyframes_mask(marked: false))")
    }
    if markedHead != 1 {
        fails.append("case 3b: the target's leading latent frame lost its mark (\(markedHead))")
    }
    print("[keyframe-cond-gate] case 3 keyframesMask: appended=\(markedAppended) (must be 0), target head=\(markedHead) (must be 1)")

    // ── 4. DENOISE MASK = 1 − strength on the appended block, so strength 1.0 holds it clean.
    for (strength, expect) in [(Float(1.0), Float(0.0)), (Float(0.25), Float(0.75))] {
        let k = KeyframeConditioning(frameIdx: 8, latent: kfTokens, H: H, W: W, fps: fps, strength: strength)
        let s2 = KeyframeConditionedState.append(
            latent: target, clean: MLXArray.zeros(target.shape).asType(.float32),
            denoiseMask: MLXArray.ones([1, nv, 1]).asType(.float32),
            positions: vPos, keyframesMask: kfMask, items: [k])
        eval(s2.denoiseMask)
        let got = MLX.max(MLX.abs(s2.denoiseMask[0..., nv..., 0...] - MLXArray(expect))).item(Float.self)
        if got > 1e-6 {
            fails.append("case 4: strength \(strength) gave denoise mask ≠ \(expect) (off by \(got))")
        }
    }
    print("[keyframe-cond-gate] case 4 denoiseMask = 1 − strength on the appended block ✓")

    // ── 5. DISCRIMINATION — frameIdx must actually move the keyframe. Without this, a build that
    //       ignored frameIdx entirely would pass every case above.
    let kA = KeyframeConditioning(frameIdx: 8,  latent: kfTokens, H: H, W: W, fps: fps)
    let kB = KeyframeConditioning(frameIdx: 24, latent: kfTokens, H: H, W: W, fps: fps)
    eval(kA.positions, kB.positions)
    let dt = MLX.max(MLX.abs(kA.positions[0..., 0..., 0 ..< 1] - kB.positions[0..., 0..., 0 ..< 1])).item(Float.self)
    if abs(dt - 16.0 / fps) > 1e-6 {
        fails.append(String(format: "case 5: frames 8 vs 24 differ by %.6f s, expected %.6f — "
            + "frameIdx is not moving the keyframe as pixel frames", dt, 16.0 / fps))
    }
    // Spatial coords must NOT move with frameIdx.
    let ds = MLX.max(MLX.abs(kA.positions[0..., 0..., 1...] - kB.positions[0..., 0..., 1...])).item(Float.self)
    if ds != 0 { fails.append("case 5b: frameIdx moved the SPATIAL coords (\(ds))") }
    print(String(format: "[keyframe-cond-gate] case 5 frameIdx 8→24 shifts time by %.6f s, spatial unchanged ✓", dt))

    // ── 6. HELD CLEAN end-to-end. Through a real conditioned loop the appended tokens must come
    //       back exactly as supplied — that is what "the endpoint is the anchor" rests on.
    let dir = "\(goldensBase)/dit_tiny_kf25"
    if let w = try? MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/weights.safetensors")),
       let io = try? MLX.loadArrays(url: URL(fileURLWithPath: "\(dir)/io.safetensors")) {
        let dit = DiT(weights: w, config: tinyDiTConfig())
        let vT = io["video_text"]!, aT = io["audio_text"]!, aP = io["audio_positions"]!
        let na = io["audio_latent"]!.dim(1), aD = io["audio_latent"]!.dim(2)
        let vD = io["video_latent"]!.dim(2)
        // Rebuild at the fixture's channel width.
        let tgt = MLXRandom.normal([1, nv, vD]).asType(.float32)
        let kTok = MLXRandom.normal([1, H * W, vD]).asType(.float32)
        eval(tgt, kTok)
        let item = KeyframeConditioning(frameIdx: 16, latent: kTok, H: H, W: W, fps: fps)
        let s3 = KeyframeConditionedState.append(
            latent: tgt, clean: MLXArray.zeros(tgt.shape).asType(.float32),
            denoiseMask: MLXArray.ones([1, nv, 1]).asType(.float32),
            positions: vPos, keyframesMask: kfMask, items: [item])
        let (vOut, _) = try DenoiseLoop.runConditioned(
            dit: dit, videoLatent0: s3.latent, audioLatent0: MLXRandom.normal([1, na, aD]).asType(.float32),
            sigmas: [1.0, 0.5, 0.0], videoText: vT, audioText: aT,
            videoPositions: s3.positions, audioPositions: aP,
            videoCleanLatent: s3.clean, videoDenoiseMask: s3.denoiseMask,
            keyframesMask: s3.keyframesMask)
        eval(vOut)
        let held = vOut[0..., nv..., 0...]
        let drift = MLX.max(MLX.abs(held - kTok.asType(.float32))).item(Float.self)
        if drift > 1e-5 {
            fails.append("case 6: the appended keyframe DRIFTED by \(drift) through the loop — it "
                + "must be held exactly, or the endpoint will not be the anchor")
        }
        print(String(format: "[keyframe-cond-gate] case 6 keyframe held through 2 steps, drift %.3g ✓", drift))
    } else {
        fails.append("case 6: tiny fixture missing — cannot prove the keyframe is held end-to-end")
    }

    // ── 7. THE SURFACE. The list must parse, and the frame-0 split must be ENFORCED — accepting
    //       frame 0 here would apply guide semantics where the oracle replaces the latent, which
    //       reads as a quality bug rather than a routing one.
    let ok: MetaData = [KeyframeMetaKeys.keyframes: .array([
        .object([KeyframeMetaKeys.path: .string("/tmp/a.png"), KeyframeMetaKeys.frame: .int(120)]),
        .object([KeyframeMetaKeys.path: .string("/tmp/b.png"), KeyframeMetaKeys.frame: .int(60),
                 KeyframeMetaKeys.strength: .double(0.8)]),
    ])]
    let parsed = (try? KeyframeMetaKeys.parse(ok)) ?? []
    if parsed.count != 2 || parsed[0].frameIdx != 120 || parsed[1].frameIdx != 60
        || abs(parsed[0].strength - 1.0) > 1e-6 || abs(parsed[1].strength - 0.8) > 1e-6 {
        fails.append("case 7: list parse wrong — got \(parsed.map { ($0.frameIdx, $0.strength) })")
    }
    if !((try? KeyframeMetaKeys.parse([:])) ?? [KeyframeRequest(frameIdx: 1) { _, _ in MLXArray([0]) }]).isEmpty {
        fails.append("case 7b: absent metaData should yield NO keyframes")
    }
    let zero: MetaData = [KeyframeMetaKeys.keyframes: .array([
        .object([KeyframeMetaKeys.path: .string("/tmp/a.png"), KeyframeMetaKeys.frame: .int(0)])])]
    var rejectedZero = false
    do { _ = try KeyframeMetaKeys.parse(zero) } catch { rejectedZero = true }
    if !rejectedZero {
        fails.append("case 7c: frame 0 was ACCEPTED — it must route to initImage (latent REPLACE), "
            + "not to the append path")
    }
    let tooMany: MetaData = [KeyframeMetaKeys.keyframes: .array((1...6).map { i in
        .object([KeyframeMetaKeys.path: .string("/tmp/\(i).png"), KeyframeMetaKeys.frame: .int(i * 8)])
    })]
    var rejectedMany = false
    do { _ = try KeyframeMetaKeys.parse(tooMany) } catch { rejectedMany = true }
    if !rejectedMany { fails.append("case 7d: 6 keyframes accepted, cap is \(KeyframeMetaKeys.maxKeyframes)") }
    print("[keyframe-cond-gate] case 7 surface: 2 parsed, empty→none, frame 0 rejected=\(rejectedZero), "
        + ">5 rejected=\(rejectedMany)")

    if fails.isEmpty {
        print("[keyframe-cond-gate] PASS ✅  7/7 — appended (not pinned), positioned at "
            + "(frameIdx+0.5)/fps, UNMARKED in keyframesMask, held clean end-to-end")
        fflush(stdout)
    } else {
        for f in fails { print("[keyframe-cond-gate] FAIL — \(f)") }
        fflush(stdout); exit(1)
    }
}
