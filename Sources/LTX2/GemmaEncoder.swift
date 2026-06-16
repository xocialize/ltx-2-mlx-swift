// GemmaEncoder.swift — Gemma 3 12B as a frozen text encoder (Path A reuse).
//
// Reuses mlx-swift-lm's `Gemma3TextModel` (loader + dual-RoPE sliding/global
// attention already done & parity-tested) and our `allHiddenStates` fork
// extension. Mirrors the oracle ltx_core_mlx .../gemma/encoders/base_encoder.py
// `GemmaLanguageModel.get_all_hidden_states`: embed·√h + all 48 layers under a
// SINGLE uniform combined causal+padding mask → 49 hidden states.

import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

public struct GemmaEncoder {
    public let model: Gemma3TextModel
    /// Holds the loaded context (model + tokenizer); we avoid naming the
    /// `Tokenizer` protocol directly (it's ambiguous across swift-transformers
    /// and swift-huggingface in this import set).
    public let context: ModelContext

    /// Load Gemma 3 from a local directory (config.json + quantized safetensors).
    public static func load(directory: URL) async throws -> GemmaEncoder {
        let configuration = ModelConfiguration(directory: directory)
        let ctx = try await #huggingFaceLoadModel(configuration: configuration)
        guard let gemma = ctx.model as? Gemma3TextModel else {
            fatalError("expected Gemma3TextModel, got \(type(of: ctx.model))")
        }
        return GemmaEncoder(model: gemma, context: ctx)
    }

    /// Combined causal + left-padding additive mask, bf16, shape (B, 1, T, T).
    /// Mirrors base_encoder.get_all_hidden_states: causal triu(-1e9, k=1) plus a
    /// padding term `(1 - attentionMask) * -1e9` broadcast over query positions.
    public static func combinedMask(attentionMask: MLXArray) -> MLXArray {
        let B = attentionMask.dim(0), T = attentionMask.dim(1)
        let causal = MLX.triu(MLXArray.ones([T, T]) * Float(-1e9), k: 1).asType(.bfloat16)  // (T,T)
        let pad = (1.0 - attentionMask.asType(.bfloat16).reshaped(B, 1, 1, T)) * Float(-1e9)   // (B,1,1,T)
        return causal.reshaped(1, 1, T, T) + pad                                                // (B,1,T,T)
    }

    /// 49 hidden states (embed + 48 layers), each (B, T, 3840).
    public func allHiddenStates(tokenIds: MLXArray, attentionMask: MLXArray) -> [MLXArray] {
        let mask = GemmaEncoder.combinedMask(attentionMask: attentionMask)
        return model.allHiddenStates(tokenIds.asType(.int32), mask: .array(mask))
    }

    // NOTE: tokenize() (left-pad to maxLength, mirroring LTXVGemmaTokenizer) is
    // deferred to the full text-encode pipeline — the parity gate feeds the
    // oracle's golden token_ids directly to isolate the forward pass.
}
