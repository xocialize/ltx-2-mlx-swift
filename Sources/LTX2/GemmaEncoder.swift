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

    /// Thrown when `gemma3` resolves to something other than `Gemma3TextModel` — in practice this
    /// means the HOST process linked MLXVLM, whose factory is probed first in mlx-swift-lm's
    /// process-global registry and shadows the text architecture (BRIDGE-LTX-003). Diagnosable
    /// error instead of the former `fatalError`.
    public struct WrongModelTypeError: Error, CustomStringConvertible {
        public let actual: String
        public var description: String {
            "GemmaEncoder: expected Gemma3TextModel, got \(actual). If the host app links MLXVLM "
            + "(directly or via another model package), mlx-swift-lm's registry resolves 'gemma3' "
            + "to the multimodal Gemma3 — remove the MLXVLM-linking dependency (see BRIDGE-LTX-003)."
        }
    }

    /// Load Gemma 3 from a local directory (config.json + quantized safetensors).
    public static func load(directory: URL) async throws -> GemmaEncoder {
        let configuration = ModelConfiguration(directory: directory)
        let ctx = try await #huggingFaceLoadModel(configuration: configuration)
        guard let gemma = ctx.model as? Gemma3TextModel else {
            throw WrongModelTypeError(actual: String(describing: type(of: ctx.model)))
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

    /// Tokenize + left-pad to `maxLength` (mirrors LTXVGemmaTokenizer, padding_side="left").
    /// Returns (tokenIds (1,maxLength), attentionMask (1,maxLength)).
    public func tokenize(_ text: String, maxLength: Int = 1024) -> (tokenIds: MLXArray, mask: MLXArray) {
        let tk = context.tokenizer
        var tokens = tk.encode(text: text.trimmingCharacters(in: .whitespacesAndNewlines))
        if tokens.count > maxLength { tokens = Array(tokens.suffix(maxLength)) }
        let pad = maxLength - tokens.count
        let padToken = tk.unknownTokenId ?? 0
        let ids = Array(repeating: padToken, count: pad) + tokens
        let m = Array(repeating: 0, count: pad) + Array(repeating: 1, count: tokens.count)
        return (MLXArray(ids.map { Int32($0) }).reshaped(1, maxLength),
                MLXArray(m.map { Int32($0) }).reshaped(1, maxLength))
    }
}
