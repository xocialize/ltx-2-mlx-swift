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

    /// `bos_token_id` / `pad_token_id` read from the checkpoint's config.json.
    ///
    /// Read rather than hardcoded: the oracle pads with `pad_token_id` (0 = `<pad>`), and the
    /// 2.3 Swift path's use of `unknownTokenId` (3 = `<unk>`) was only ever benign by accident
    /// of the connector discarding padded positions.
    public static func specialTokenIds(directory: URL) throws -> (bos: Int, pad: Int) {
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let text = (root["text_config"] as? [String: Any]) ?? root
        let bos = (text["bos_token_id"] as? Int) ?? (root["bos_token_id"] as? Int) ?? 2
        let pad = (text["pad_token_id"] as? Int) ?? (root["pad_token_id"] as? Int) ?? 0
        return (bos, pad)
    }

    /// Tokenize + left-pad, with the manual BOS the 2.5 tokenizer does not supply.
    ///
    /// ⟲ **CORRECTED 2026-08-16 (AB-T-0059): over-length input now keeps the FIRST `maxLength`
    /// tokens, not the last.** The previous implementation kept the LAST — inherited from the
    /// oracle, and WRONG against the vendor. Verified against `Lightricks/LTX-2` v1.2.0
    /// `ltx_core/text_encoders/gemma/tokenizer.py:31-63`, which does
    /// `tokenizer(text, truncation=True, max_length=…)` and then `[bos, *ids][: max_length]` —
    /// both of which keep the FRONT, because the vendor sets `padding_side` explicitly but never
    /// `truncation_side`, and the Gemma-4 `tokenizer_config.json` does not set it either, so HF's
    /// default applies. **Measured, not assumed:** `tk.truncation_side == "right"` on the real
    /// checkpoint, and a >1024-token probe with distinct head/tail markers kept the head and
    /// dropped the tail.
    ///
    /// The old behaviour was doubly wrong on a long prompt: it conditioned on the prompt's TAIL
    /// where the vendor uses its HEAD, and it discarded the BOS it had just prepended.
    ///
    /// Order matters and mirrors the vendor exactly: encode → truncate to `maxLength` (front) →
    /// prepend `<bos>` if absent → re-clip to `maxLength` (front, so an exactly-full prompt loses
    /// its LAST token rather than the BOS) → left-pad with `pad_token_id`.
    public static func tokenize(
        _ text: String,
        tokenizer tk: any MLXLMCommon.Tokenizer,
        bosId: Int,
        padId: Int,
        maxLength: Int = 1024
    ) -> (tokenIds: MLXArray, mask: MLXArray) {
        var tokens = tk.encode(text: text.trimmingCharacters(in: .whitespacesAndNewlines))
        if tokens.count > maxLength { tokens = Array(tokens.prefix(maxLength)) }   // HF truncation_side=right
        if tokens.first != bosId { tokens = Array(([bosId] + tokens).prefix(maxLength)) }
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
