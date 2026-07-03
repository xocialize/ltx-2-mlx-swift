// ReferenceConditioning.swift — IC-LoRA reference-token append (IC-LORA-PLAN P1).
//
// 1:1 port of the oracle's VideoConditionByReferenceLatent.apply
// (ltx_core_mlx/conditioning/types/reference_video_cond.py): a pre-encoded reference latent is
// APPENDED to the token sequence as clean, (1−strength)-masked context; its RoPE positions are
// appended with the SPATIAL axes (height, width) multiplied by `downscaleFactor` so a
// lower-resolution reference lands in the target's coordinate space (temporal axis unscaled).
// The adapted (IC-LoRA'd) attention reads the appended context; after the denoise loop the
// caller slices the first `targetTokens` back out — reference tokens are never decoded.
//
// Attention-strength (< 1.0) / pixel-masked conditioning builds a (B,N,N) mask in the oracle —
// NOT ported here (basic IC at strength 1.0 keeps attention_mask == None); it is a deferred
// slice (IC-LORA-PLAN "Deferred").

import MLX

/// One reference conditioning item: pre-encoded latent tokens + their (unscaled) positions,
/// computed at the reference's OWN latent grid via `Positions.computeVideoPositions`.
public struct ReferenceConditioning {

    /// The video VAE's 8k+1 frame requirement, snapped the way the oracle snaps reference media
    /// (`iclora_utils`: `k = max(1, (frames-1)//8)` → `1+8k`) — clamps DOWN, never up, with a
    /// floor of 9 pixel frames.
    public static func snapFrames(_ frames: Int) -> Int {
        let k = max(1, (frames - 1) / 8)
        return 1 + 8 * k
    }
    /// Reference latent tokens (B, Nr, C) — VAE-encoded + patchified, normalized like any latent.
    public let tokens: MLXArray
    /// Positions (B, Nr, 3) [time, height, width] at the reference's own grid.
    public let positions: MLXArray
    /// Target/reference resolution ratio (safetensors-header `reference_downscale_factor`).
    public let downscaleFactor: Float
    /// Conditioning strength: 1.0 = fully preserved (denoise mask 0), 0.0 = fully denoised.
    public let strength: Float

    public init(tokens: MLXArray, positions: MLXArray,
                downscaleFactor: Float = 1, strength: Float = 1.0) {
        self.tokens = tokens
        self.positions = positions
        self.downscaleFactor = downscaleFactor
        self.strength = strength
    }

    /// Positions scaled into the target coordinate space: spatial axes × downscaleFactor.
    var scaledPositions: MLXArray {
        guard downscaleFactor != 1 else { return positions }
        let scale = MLXArray([Float(1), downscaleFactor, downscaleFactor]).reshaped(1, 1, 3)
        return positions.asType(.float32) * scale
    }
}

/// The extended video-side denoise state after appending reference tokens — feed its fields to
/// `DenoiseLoop.runConditioned` (which already handles per-token σ + clean re-blend), then
/// `slice(_:)` the loop's output back to the target tokens.
public struct ICVideoState {
    public let latent: MLXArray       // (B, Nv+ΣNr, C): target latent ++ clean refs
    public let clean: MLXArray        // (B, Nv+ΣNr, C): target clean (zeros for t2v) ++ refs
    public let denoiseMask: MLXArray  // (B, Nv+ΣNr, 1): 1 at target, 1−strength at refs
    public let positions: MLXArray    // (B, Nv+ΣNr, 3): target ++ spatially-scaled ref positions
    /// Nv — the slice bound after the denoise loop.
    public let targetTokens: Int

    /// Build the extended state. `targetLatent` is the (already noised) generation region
    /// (B, Nv, C); `targetClean` defaults to zeros (t2v; pass the stage-2 upscaled latent or an
    /// i2v-blended clean when combining). Mirrors the oracle order: refs appended AFTER the
    /// target noise blend (legacy_scalar_blend flow used by ic_lora).
    public static func build(targetLatent: MLXArray,
                             targetPositions: MLXArray,
                             targetClean: MLXArray? = nil,
                             targetDenoiseMask: MLXArray? = nil,
                             references: [ReferenceConditioning]) -> ICVideoState {
        let B = targetLatent.dim(0), Nv = targetLatent.dim(1)
        var latent = targetLatent
        var clean = targetClean ?? MLXArray.zeros(targetLatent.shape, dtype: targetLatent.dtype)
        var mask = targetDenoiseMask ?? MLXArray.ones([B, Nv, 1]).asType(targetLatent.dtype)
        var positions = targetPositions.asType(.float32)
        for ref in references {
            let refTokens = ref.tokens.asType(latent.dtype)
            latent = concatenated([latent, refTokens], axis: 1)
            clean = concatenated([clean, refTokens], axis: 1)
            let refMask = MLXArray.full([B, ref.tokens.dim(1), 1],
                                        values: MLXArray(1.0 - ref.strength)).asType(mask.dtype)
            mask = concatenated([mask, refMask], axis: 1)
            positions = concatenated([positions, ref.scaledPositions], axis: 1)
        }
        return ICVideoState(latent: latent, clean: clean, denoiseMask: mask,
                            positions: positions, targetTokens: Nv)
    }

    /// Drop the appended reference tokens from a denoised latent: (B, Nv+ΣNr, C) → (B, Nv, C).
    public func slice(_ denoised: MLXArray) -> MLXArray {
        denoised[0..., ..<targetTokens, 0...]
    }
}
