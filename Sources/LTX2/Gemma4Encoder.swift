// Gemma4Encoder.swift — loading the LTX-2.5 in-dir text encoder.
//
// Sibling of `GemmaEncoder` for the Lightricks-tuned `gemma4_unified` checkpoint that
// LTX-2.5 conversions ship inside the model directory (`gemma4-12b-ltx-v1/`).
//
// ⚠️ The checkpoint resolves through mlx-swift-lm's factory to `Gemma4Model` (the wrapper),
// NOT to `Gemma4TextModel`. The 49-state tap extends the latter, so the wrapper's
// `languageModel` must be reachable — it is, at `@_spi(GemmaEncoder)` scope
// (ml-explore/mlx-swift-lm#530).
//
// ⚠️ TOKENIZATION DIFFERS FROM 2.3 AND THE DIFFERENCE IS SILENT. The 2.5 tokenizer's
// post_processor is `single: [Sequence A]` with no special tokens, so it emits NO `<bos>`,
// where Gemma-3's `single: [SpecialToken <bos>, Sequence A]` does. `GemmaEncoder.tokenize`
// relies on the post_processor, so it is correct on 2.3 only by accident of that config.
// BOS is prepended explicitly here, matching the oracle.

import Foundation
import HuggingFace
import MLX
import MLXFast
import MLXHuggingFace
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon
import Tokenizers

public struct Gemma4Encoder {
    public let model: Gemma4TextModel
    public let context: ModelContext

    public init(model: Gemma4TextModel, context: ModelContext) {
        self.model = model
        self.context = context
    }

    public struct WrongModelTypeError: Error, CustomStringConvertible {
        public let actual: String
        public var description: String {
            "Gemma4Encoder: expected Gemma4Model, got \(actual). `gemma4_unified` is claimed by "
            + "BOTH MLXLLM and MLXVLM factories; if the host links MLXVLM it resolves to the "
            + "multimodal Gemma4Unified, whose internals are private (BRIDGE-LTX-003)."
        }
    }

    /// Load the tuned Gemma-4 encoder from a local directory.
    public static func load(directory: URL) async throws -> Gemma4Encoder {
        let configuration = ModelConfiguration(directory: directory)
        let ctx = try await #huggingFaceLoadModel(configuration: configuration)
        guard let wrapper = ctx.model as? Gemma4Model else {
            throw WrongModelTypeError(actual: String(describing: type(of: ctx.model)))
        }
        return Gemma4Encoder(model: wrapper.languageModel, context: ctx)
    }

    /// Tokenize + left-pad, with the manual BOS the 2.5 tokenizer does not supply.
    ///
    /// Mirrors the oracle: encode the stripped text, prepend `<bos>` if absent, keep the LAST
    /// `maxLength` on overflow (which can drop that BOS again — matched behaviour), left-pad
    /// with `pad_token_id`.
    public static func tokenize(
        _ text: String,
        tokenizer tk: any MLXLMCommon.Tokenizer,
        bosId: Int,
        padId: Int,
        maxLength: Int = 1024
    ) -> (tokenIds: MLXArray, mask: MLXArray) {
        var tokens = tk.encode(text: text.trimmingCharacters(in: .whitespacesAndNewlines))
        if tokens.first != bosId { tokens.insert(bosId, at: 0) }
        if tokens.count > maxLength { tokens = Array(tokens.suffix(maxLength)) }
        let pad = maxLength - tokens.count
        let ids = Array(repeating: padId, count: pad) + tokens
        let m = Array(repeating: 0, count: pad) + Array(repeating: 1, count: tokens.count)
        return (MLXArray(ids.map { Int32($0) }).reshaped(1, maxLength),
                MLXArray(m.map { Int32($0) }).reshaped(1, maxLength))
    }

    /// The 49 hidden states, with the caller-supplied uniform mask on every layer.
    public func allHiddenStates(tokenIds: MLXArray, attentionMask: MLXArray) throws -> [MLXArray] {
        let mask = GemmaEncoder.combinedMask(attentionMask: attentionMask)
        return try model.allHiddenStates(tokenIds, mask: .array(mask))
    }
}
