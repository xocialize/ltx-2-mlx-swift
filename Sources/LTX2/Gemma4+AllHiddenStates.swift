// Gemma4+AllHiddenStates.swift — the LTX-2.5 49-state encoder tap.
//
// Sibling of `Gemma3+AllHiddenStates.swift`, against mlx-swift-lm's Gemma-4
// `@_spi(GemmaEncoder)` surface. Written as a CLIENT of that surface (no @testable,
// no internal access), which is also what proves the SPI exposure is sufficient.
//
// ⚠️ THREE deltas from the Gemma-3 tap, each a silent-divergence risk if the 2.3 code
// is copied verbatim:
//   1. EMBED SCALE DTYPE. Gemma-3 pre-rounds the scalar to bfloat16; Gemma-4 uses a
//      plain Float (`Gemma4TextModelInner.embedScale`). Reusing the 2.3 line changes
//      the numbers.
//   2. FINAL NORM. Gemma-4 applies its final norm to the LAST state (the HF
//      `output_hidden_states` convention the 2.5 oracle matches). Gemma-3's tap applies
//      none.
//   3. NORM CONVENTION. Gemma-3's `Gemma.RMSNorm` is `rmsNorm(x, weight: 1 + weight)`;
//      Gemma-4's is a plain `RMSNorm` with NO shift. We call the model's own `norm`
//      rather than re-implementing it, so this cannot drift.
//
// Retained from the 2.3 tap because they are load-bearing: the caller-supplied uniform
// mask on EVERY layer, a per-layer `eval` (Metal watchdog), and a per-layer
// cancellation check.

import Foundation
import MLX
@_spi(GemmaEncoder) import MLXLLM
@_spi(GemmaEncoder) import MLXLMCommon
import MLXNN

extension Gemma4TextModel {

    /// Every hidden state: the scaled embedding plus one per decoder layer, with the last
    /// state normed. Returns `numHiddenLayers + 1` states of shape `(B, T, hiddenSize)`.
    ///
    /// - Parameters:
    ///   - tokenIds: `(B, T)` token ids.
    ///   - mask: a uniform additive mask applied to EVERY layer (LTX's encoder semantics —
    ///     deliberately not Gemma's per-layer sliding/global masks).
    public func allHiddenStates(
        _ tokenIds: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) throws -> [MLXArray] {
        let inner = self.model

        // Plain Float scale — NOT pre-rounded to bf16 the way the Gemma-3 tap does it.
        var h = inner.embedTokens(tokenIds) * inner.embedScale
        eval(h)
        var states: [MLXArray] = [h]
        states.reserveCapacity(inner.config.numHiddenLayers + 1)

        for layer in inner.layers {
            try Task.checkCancellation()
            // No per-layer inputs, no KV sharing, no position offset: the LTX checkpoint has
            // PLE, MoE and KV-sharing all disabled, so every layer is self-contained and the
            // returned shared-KV / offset values are unused.
            let (out, _, _) = layer(h, mask: mask, cache: nil,
                                    perLayerInput: nil, sharedKV: nil, positionOffset: nil)
            h = out
            eval(h)   // watchdog: one command buffer per layer, not one for all 48
            states.append(h)
        }

        // The final norm lands on the LAST state only (HF output_hidden_states convention).
        states[states.count - 1] = inner.norm(states[states.count - 1])
        eval(states[states.count - 1])
        return states
    }
}
