// Gemma3+AllHiddenStates.swift — the 49-state encoder tap, client-side.
//
// Formerly a local patch carried in our mlx-swift-lm fork
// (`Libraries/MLXLLM/Models/Gemma3Text+AllHiddenStates.swift`, branch
// `ltx/gemma-all-hidden-states`). Upstream ml-explore/mlx-swift-lm#387 (merged
// 2026-07-21, `6608a35`) deliberately did NOT take the implementation — a single
// uniform mask across every layer matches the LTX-2 reference encoder but diverges
// from generic Gemma 3 per-layer sliding/global semantics, so it is pipeline-specific
// and belongs here. What upstream DID take is the minimal surface needed to write it
// from outside the module, behind `@_spi(GemmaEncoder)`:
//
//     Gemma3TextConfiguration.hiddenSize, .hiddenLayers
//     Gemma3TransformerBlock + .callAsFunction(_:mask:cache:)
//     Gemma3Model.embedTokens, .layers, .config
//
// Known limitation (disclosed upstream): `Gemma3TransformerBlock.callAsFunction`
// takes a NON-optional mask, so this surface supports a uniform-mask tap only —
// per-layer sliding/global masking cannot be reproduced through it. That is exactly
// what LTX-2 wants; see the mask note on `allHiddenStates` below.
//
// Mirrors the oracle ltx_core_mlx .../gemma/encoders/base_encoder.py
// `GemmaLanguageModel.get_all_hidden_states`.

import Foundation
import MLX
import MLXFast
@_spi(GemmaEncoder) import MLXLLM

extension Gemma3Model {
    /// Returns the embedding output plus each transformer layer's output —
    /// `hiddenLayers + 1` states, each shaped `(B, T, hiddenSize)`.
    ///
    /// A SINGLE uniform `mask` is applied to every layer. This is the text-encoder
    /// use: the caller supplies a combined causal+padding mask (see
    /// ``GemmaEncoder/combinedMask(attentionMask:)``) and the per-layer
    /// sliding-window/global mask selection used by `callAsFunction(_:mask:cache:)`
    /// is intentionally bypassed — that selection is what would diverge from the
    /// LTX-2 reference encoder.
    ///
    /// Throws `CancellationError` between layers when the surrounding task is
    /// cancelled — the per-layer `eval` already bounds each step, so a cancel
    /// (user quit / engine preempt) lands within ~one layer's compute instead of
    /// riding out the whole 48-layer forward.
    func allHiddenStates(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) throws -> [MLXArray] {
        var h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)

        var states: [MLXArray] = [h]
        for layer in layers {
            try Task.checkCancellation()
            h = layer(h, mask: mask, cache: nil)
            // Per-layer materialization keeps each Metal command buffer below the
            // macOS GPU watchdog (~10s) — without it, all layers fuse into one
            // dispatch and time out. Matches the oracle's LTX2_GEMMA_EVAL_EVERY=1.
            // Measured load-bearing in production at T=1024 on the 12B checkpoint.
            eval(h)
            states.append(h)
        }
        return states
    }
}

extension Gemma3TextModel {
    /// See ``Gemma3Model/allHiddenStates(_:mask:)``. Convenience forwarding from
    /// the top-level text model to its inner `model`.
    func allHiddenStates(
        _ inputs: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) throws -> [MLXArray] {
        try model.allHiddenStates(inputs, mask: mask)
    }
}
