// Keyframed.swift — supplied keyframe images on the two-stage t2v path (AB-A-0025).
//
// The MACHINERY already existed and is gated: `KeyframeConditioning` is a 1:1 port of the oracle's
// `VideoConditionByKeyframeIndex`, and `KeyframeConditionedState.append` extends `keyframesMask`
// with ZEROS. Both were built for DFR. This file is the WIRING, not new geometry.
//
// Two oracle facts this rests on, either of which is easy to get backwards (AB-A-0025 thread):
//
//  1. A supplied keyframe is APPENDED as its own token block — placeholder zeros in the noisy
//     latent, real tokens in the clean latent, its own RoPE positions time-offset to `frameIdx`
//     and narrowed to ONE pixel frame. It is NOT an in-place pin of a latent frame.
//  2. It is NOT marked in `keyframesMask`. `marked=True` is reserved for GENERATED keyframe slots;
//     marking a supplied image would apply a learned embedding the model never saw on such tokens
//     during training. Same slots-vs-first-frame distinction that produced AB-T-0090.
//
// ⚠️ frameIdx 0 is NOT handled here, deliberately. The oracle splits image conditioning by index
// (`combined_image_conditionings`): index 0 REPLACES the latent (`VideoConditionByLatentIndex`,
// which is our i2v path), every other index GUIDES via the append above. Routing frame 0 through
// this file would silently apply guide semantics where the vendor replaces. Frame 0 rides
// `T2VRequest.initImage`.
//
// Re-encoded PER STAGE at that stage's geometry, matching the vendor, which calls
// `image_conditionings_by_replacing_latent(..., height=stage_1_h, width=stage_1_w)` and again at
// full size — hence the `pixels` closure rather than a pre-encoded tensor.

import Foundation
import MLX
import MLXProfiling
import MLXRandom

/// One requested keyframe. `pixels` yields (1, 3, 1, H, W) in [-1,1] at the asked-for pixel size,
/// so each stage can encode at its own geometry the way the oracle does.
public struct KeyframeRequest {
    public let frameIdx: Int
    public let strength: Float
    public let pixels: (_ width: Int, _ height: Int) throws -> MLXArray

    public init(frameIdx: Int, strength: Float = 1.0,
                pixels: @escaping (_ width: Int, _ height: Int) throws -> MLXArray) {
        self.frameIdx = frameIdx
        self.strength = strength
        self.pixels = pixels
    }
}

extension LTX2Pipeline {

    /// Encode the requested keyframes at one stage's latent grid.
    func keyframeItems(_ reqs: [KeyframeRequest], hLat: Int, wLat: Int,
                       fps: Float) throws -> [KeyframeConditioning] {
        guard !reqs.isEmpty else { return [] }
        try ensureVAEEncoder()
        var items: [KeyframeConditioning] = []
        for r in reqs {
            let span = MLXProfiler.shared.begin("kf-ingest", "encode",
                note: "frame \(r.frameIdx) @ \(wLat * 32)x\(hLat * 32)")
            let px = try r.pixels(wLat * 32, hLat * 32)          // (1,3,1,H,W)
            let lat = vaeEncoder!.encode(px)                      // (1,128,1,hLat,wLat)
            let toks = LTX2Pipeline.patchify(lat)                 // (1, hLat*wLat, 128)
            eval(toks)
            MLXProfiler.shared.end(span)
            items.append(KeyframeConditioning(frameIdx: r.frameIdx, latent: toks,
                                              H: hLat, W: wLat, fps: fps, strength: r.strength))
        }
        return items
    }

    /// Append keyframes to a stage's denoise state, run it conditioned, and slice the target back.
    ///
    /// The appended tokens carry `denoiseMask = 1 − strength` (0 at strength 1.0 ⇒ held clean every
    /// step) and `keyframesMask = 0`. The target's own tokens keep whatever mask they had.
    /// Frame-0 clean latent + denoise mask for a stage, when an init image is supplied.
    ///
    /// This is the oracle's OTHER image-conditioning arm: `combined_image_conditionings` sends
    /// index 0 to `VideoConditionByLatentIndex`, which REPLACES the latent in place rather than
    /// appending. So first-frame and later-frame conditioning are genuinely different mechanisms,
    /// and both have to exist for a first+last request.
    func frame0Conditioning(_ pixels: (_ w: Int, _ h: Int) throws -> MLXArray,
                            hLat: Int, wLat: Int, nv: Int, dtype: DType) throws -> (clean: MLXArray, mask: MLXArray) {
        try ensureVAEEncoder()
        let frame0 = hLat * wLat
        let px = try pixels(wLat * 32, hLat * 32)
        let lat = vaeEncoder!.encode(px)
        let toks = LTX2Pipeline.patchify(lat)                        // (1, frame0, 128)
        eval(toks)
        let clean = MLX.concatenated([toks.asType(dtype),
                                      MLXArray.zeros([1, nv - frame0, toks.dim(2)]).asType(dtype)], axis: 1)
        // Held fully clean (strength 1.0) — an endpoint that drifts is not an anchor.
        let mask = MLX.concatenated([MLXArray.zeros([1, frame0, 1]).asType(dtype),
                                     MLXArray.ones([1, nv - frame0, 1]).asType(dtype)], axis: 1)
        return (clean, mask)
    }

    func runKeyframedStage(
        videoLatent: MLXArray, audioLatent: MLXArray, sigmas: [Float],
        videoText: MLXArray, audioText: MLXArray,
        videoPositions: MLXArray, audioPositions: MLXArray,
        keyframesMask: MLXArray?, items: [KeyframeConditioning],
        baseClean: MLXArray? = nil, baseMask: MLXArray? = nil,
        // a2v freezes the AUDIO while conditioning the video; t2v leaves audio free. Both go
        // through this one call so the keyframe machinery cannot drift between the two paths.
        audioCleanLatent: MLXArray? = nil, audioDenoiseMask: MLXArray? = nil,
        ancestralEta: Float, ancestralNoiseSeed: UInt64,
        label: String, stage: Int, totalStages: Int
    ) throws -> (video: MLXArray, audio: MLXArray) {
        let nv = videoLatent.dim(1)
        let st = KeyframeConditionedState.append(
            latent: videoLatent,
            clean: baseClean ?? MLXArray.zeros(videoLatent.shape).asType(videoLatent.dtype),
            denoiseMask: baseMask ?? MLXArray.ones([videoLatent.dim(0), nv, 1]).asType(videoLatent.dtype),
            positions: videoPositions,
            keyframesMask: keyframesMask,
            items: items)
        let (vfull, afull) = try DenoiseLoop.runConditioned(
            dit: try ensureDiT(), videoLatent0: st.latent, audioLatent0: audioLatent,
            sigmas: sigmas, videoText: videoText, audioText: audioText,
            videoPositions: st.positions, audioPositions: audioPositions,
            videoCleanLatent: st.clean, videoDenoiseMask: st.denoiseMask,
            audioCleanLatent: audioCleanLatent, audioDenoiseMask: audioDenoiseMask,
            keyframesMask: st.keyframesMask,
            ancestralEta: ancestralEta, ancestralNoiseSeed: ancestralNoiseSeed,
            label: label, stage: stage, totalStages: totalStages)
        return (st.slice(vfull), afull!)
    }
}
