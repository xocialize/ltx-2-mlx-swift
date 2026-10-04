// OutpaintGates.swift — `--ic-mask-gate`: the pixel-masked IC path (`ICAttentionMask` +
// `DiT.withVideoSelfAttentionBias`) that outpainting and inpainting run on.
//
// Two kinds of evidence, because the oracle can only supply one of them:
//   [1–4] PARITY vs `parity/dump_ic_mask_goldens.py` — mask downsampling, block structure, both
//         additive conventions, and the masked tiny-DiT denoise against the oracle DiT. ⚠️ The
//         oracle's own pipeline mis-converts this mask (it adds the [0,1] mask raw), so the
//         dumper converts upstream's way and hands the oracle a bias; the oracle DiT is a fair
//         reference, its mask plumbing is not.
//   [5–6] SEMANTIC IDENTITIES no oracle is needed for, and the ones a wrong port cannot pass:
//         an all-ones mask must reproduce the unmasked IC run, and a ZERO cross mask must make
//         the reference invisible — the target tokens must come out as if no reference had been
//         appended at all. [6] runs AUDIO-FREE on purpose: with audio present the reference
//         tokens still reach audio through the AV-cross attention (upstream behaviour — the mask
//         is self-attention only), so the identity only holds with audio off.
//   [7]   DISCRIMINATION: the two conventions must produce DIFFERENT outputs on this fixture,
//         or [4] could not tell which one Swift applies.

import Foundation
import MLX
import LTX2

func icMaskGate() throws {
    let weights = try MLX.loadArrays(url: URL(fileURLWithPath: "\(goldensBase)/dit_tiny/weights.safetensors"))
    let io = try MLX.loadArrays(url: URL(fileURLWithPath: "\(goldensBase)/ic_mask/io.safetensors"))
    let dit = DiT(weights: weights, config: tinyDiTConfig())
    let sigmas = io["sigmas"]!.asArray(Float.self)
    let nv = io["video_latent"]!.dim(1)
    var allPass = true
    func check(_ ok: Bool, _ line: String) {
        print("[ic-mask-gate] \(ok ? "✅" : "❌") \(line)")
        allPass = allPass && ok
    }

    // [1] pixel mask → latent cross weights (area + causal temporal)
    let cross = ICAttentionMask.latentCrossMask(pixelMask: io["pixel_mask"]!,
                                                latentFrames: 2, latentHeight: 4, latentWidth: 4)
    let crossErr = maxAbs(cross, io["cross"]!)
    check(crossErr < 1e-6, String(format: "[1] latent cross mask maxAbs %.2e (fractional cells 0.75/0.5625/0.094/0.125)", crossErr))

    // [2] block structure
    let mult = ICAttentionMask.multiplicativeMask(numNoisy: nv, crossMasks: [cross])
    let multErr = maxAbs(mult, io["mult"]!)
    check(multErr < 1e-6, String(format: "[2] multiplicative block mask %@ maxAbs %.2e", "\(mult.shape)", multErr))

    // [3] additive conventions (masked entries compared by class: both ≤ -1e30)
    for (conv, key) in [(ICAttentionMask.Convention.ltxCore, "bias_ltxcore"), (.diffusers, "bias_diffusers")] {
        let ours = ICAttentionMask.additiveBias(mult, convention: conv, dtype: .float32).reshaped(mult.shape)
        let gold = io[key]!
        let huge = MLXArray(Float(-1e30))
        let oursMasked = MLX.less(ours, huge), goldMasked = MLX.less(gold, huge)
        let classAgree = MLX.all(MLX.equal(oursMasked, goldMasked)).item(Bool.self)
        let finiteErr = MLX.max(MLX.abs(MLX.which(goldMasked, MLXArray(Float(0)), ours - gold))).item(Float.self)
        check(classAgree && finiteErr < 1e-4,
              String(format: "[3] %@ bias: masked-class agree=%@ finite maxAbs %.2e", conv.rawValue,
                     classAgree ? "yes" : "NO", finiteErr))
    }

    // [4] masked tiny-DiT IC denoise vs the oracle DiT, per convention
    let ref = ReferenceConditioning(tokens: io["ref_tokens"]!, positions: io["ref_positions"]!)
    let state = ICVideoState.build(targetLatent: io["video_latent"]!,
                                   targetPositions: io["video_positions"]!, references: [ref])
    func icRun(bias: MLXArray?, audio: Bool = true) throws -> (MLXArray, MLXArray?) {
        try dit.withVideoSelfAttentionBias(bias) {
            try DenoiseLoop.runConditioned(
                dit: dit, videoLatent0: state.latent, audioLatent0: audio ? io["audio_latent"]! : nil,
                sigmas: sigmas, videoText: io["video_text"], audioText: io["audio_text"],
                videoPositions: state.positions, audioPositions: audio ? io["audio_positions"]! : nil,
                videoCleanLatent: state.clean, videoDenoiseMask: state.denoiseMask)
        }
    }
    var convOut: [String: MLXArray] = [:]
    for (conv, name) in [(ICAttentionMask.Convention.ltxCore, "ltxcore"), (.diffusers, "diffusers")] {
        let bias = ICAttentionMask.selfAttentionBias(numNoisy: nv, referenceCrossMask: cross,
                                                     convention: conv, dtype: .float32)
        let (vFull, aOpt) = try icRun(bias: bias)
        let vS = state.slice(vFull); eval(vFull, aOpt!)
        convOut[name] = vS
        let f = cosine(vFull, io["\(name)_video_final_full"]!)
        let s = cosine(vS, io["\(name)_video_final"]!)
        let a = cosine(aOpt!, io["\(name)_audio_final"]!)
        check(f >= 0.999 && s >= 0.999 && a >= 0.999,
              String(format: "[4] %@ masked IC denoise vs oracle DiT: video full %.6f sliced %.6f audio %.6f",
                     name, f, s, a))
    }

    // [5] all-ones mask ≡ unmasked IC (a zero bias through the masked kernel)
    do {
        let ones = MLXArray.ones([io["ref_tokens"]!.dim(1)])
        let b = ICAttentionMask.selfAttentionBias(numNoisy: nv, referenceCrossMask: ones,
                                                  convention: .ltxCore, dtype: .float32)
        let (vm, _) = try icRun(bias: b)
        let (vu, _) = try icRun(bias: nil)
        eval(vm, vu)
        let c = cosine(vm, vu), e = maxAbs(vm, vu)
        check(c >= 0.99999 && e < 1e-3, String(format: "[5] all-ones mask ≡ unmasked: cos %.7f maxAbs %.2e", c, e))
    }

    // [6] zero cross mask ⇒ the reference is invisible (audio-free; see header)
    do {
        let zeros = MLXArray.zeros([io["ref_tokens"]!.dim(1)])
        let b = ICAttentionMask.selfAttentionBias(numNoisy: nv, referenceCrossMask: zeros,
                                                  convention: .ltxCore, dtype: .float32)
        let (vm, _) = try icRun(bias: b, audio: false)
        let masked = state.slice(vm)
        let (alone, _) = try DenoiseLoop.run(
            dit: dit, videoLatent0: io["video_latent"]!, audioLatent0: nil, sigmas: sigmas,
            videoText: io["video_text"], audioText: io["audio_text"],
            videoPositions: io["video_positions"]!, audioPositions: nil)
        let (vu, _) = try icRun(bias: nil, audio: false)
        eval(masked, alone, vu)
        // The tiny random DiT barely uses its reference (1−cos ~6e-6), so cosine cannot separate
        // "invisible" from "faintly visible"; maxAbs can. Control: the UNMASKED reference's effect
        // on the same tokens. The masked residual must sit an order of magnitude under it.
        let c = cosine(masked, alone), e = maxAbs(masked, alone)
        let effect = maxAbs(state.slice(vu), alone)
        check(c >= 0.9999 && effect > 10 * e,
              String(format: "[6] zero cross ⇒ reference invisible: residual maxAbs %.2e vs unmasked-reference effect %.2e (%.0f×), cos %.7f", e, effect, effect / max(e, 1e-12), c))
    }

    // [7] the conventions must disagree on this fixture, or [4] could not tell them apart
    let diff = maxAbs(convOut["ltxcore"]!, convOut["diffusers"]!)
    check(diff > 1e-3, String(format: "[7] ltxCore vs diffusers outputs differ by maxAbs %.4f (oracle: 0.0075)", diff))

    // [8] the stage-2 anchor's 2× bilinear (align_corners=False, edge-clamped): a W=3 row [0,4,8]
    //     must become [0,1,3,5,7,8], the same on H, and a constant must stay constant.
    do {
        let row = MLXArray([Float(0), 4, 8]).reshaped(1, 1, 1, 1, 3)
        let upW = LTX2Pipeline.upsample2x(broadcast(row, to: [1, 1, 1, 3, 3]))
        let wantRow: [Float] = [0, 1, 3, 5, 7, 8]
        let gotRow = upW[0, 0, 0, 0].asArray(Float.self)
        let upH = LTX2Pipeline.upsample2x(broadcast(row.swappedAxes(3, 4), to: [1, 1, 1, 3, 3]))
        let gotCol = upH[0, 0, 0, 0..., 0].asArray(Float.self)
        let flat = LTX2Pipeline.upsample2x(MLXArray.full([1, 3, 2, 4, 5], values: MLXArray(Float(0.3))))
        let ok = gotRow == wantRow && gotCol == wantRow && flat.shape == [1, 3, 2, 8, 10]
            && MLX.max(MLX.abs(flat - 0.3)).item(Float.self) < 1e-6
        check(ok, "[8] anchor upsample2x: row \(gotRow) col \(gotCol) (want \(wantRow)), constant preserved, shape \(flat.shape)")
    }

    print(allPass ? "[ic-mask-gate] PASS ✅" : "[ic-mask-gate] FAIL ❌")
    if !allPass { exit(1) }
}
