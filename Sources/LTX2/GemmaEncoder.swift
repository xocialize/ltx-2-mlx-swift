// GemmaEncoder.swift — Gemma 3 12B as a frozen text encoder (Path A reuse).
//
// Reuses stock mlx-swift-lm's `Gemma3TextModel` (loader + dual-RoPE sliding/global
// attention already done & parity-tested); the 49-state tap itself is ours, in
// `Gemma3+AllHiddenStates.swift`, via the upstream `@_spi(GemmaEncoder)` surface.
// Mirrors the oracle ltx_core_mlx .../gemma/encoders/base_encoder.py
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

    /// Load ONLY the tokenizer from a local Gemma directory — no model weights touched.
    ///
    /// This is exactly what `#huggingFaceLoadModel` installs in `context.tokenizer`
    /// (`Tokenizers.AutoTokenizer.from(modelFolder:)` wrapped by the MLXLMCommon adaptor —
    /// see mlx-swift-lm `TokenizerLoaderMacro`), so `--gemma-tokenizer-gate` exercises the
    /// SAME tokenizer production uses without paying for the 12B load.
    public static func loadTokenizer(directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return #adaptHuggingFaceTokenizer(upstream)
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

    /// 49 hidden states (embed + 48 layers), each (B, T, 3840). Throws `CancellationError`
    /// between layers — a quit/Cancel during the encode forward stops within ~one layer
    /// instead of riding out the whole 12B pass (MVP M2 finding). See
    /// ``Gemma3Model/allHiddenStates(_:mask:)``.
    public func allHiddenStates(tokenIds: MLXArray, attentionMask: MLXArray) throws -> [MLXArray] {
        let mask = GemmaEncoder.combinedMask(attentionMask: attentionMask)
        return try model.allHiddenStates(tokenIds.asType(.int32), mask: .array(mask))
    }

    /// Tokenize + left-pad to `maxLength` (mirrors LTXVGemmaTokenizer, padding_side="left").
    /// Returns (tokenIds (1,maxLength), attentionMask (1,maxLength)).
    public func tokenize(_ text: String, maxLength: Int = 1024) -> (tokenIds: MLXArray, mask: MLXArray) {
        GemmaEncoder.tokenize(text, tokenizer: context.tokenizer, maxLength: maxLength)
    }

    /// The tokenization itself, over a bare tokenizer — so `--gemma-tokenizer-gate` can pin it
    /// without loading the 12B. Mirrors the oracle `GemmaLanguageModel.tokenize`
    /// (base_encoder.py): encode the stripped text WITH special tokens (Gemma-3's
    /// post_processor prepends `<bos>`), keep the LAST `maxLength` on overflow (which drops
    /// that `<bos>` — matched behavior, pinned as gate case 05), then left-pad.
    public static func tokenize(
        _ text: String,
        tokenizer tk: any MLXLMCommon.Tokenizer,
        maxLength: Int = 1024
    ) -> (tokenIds: MLXArray, mask: MLXArray) {
        var tokens = tk.encode(text: text.trimmingCharacters(in: .whitespacesAndNewlines))
        if tokens.count > maxLength { tokens = Array(tokens.suffix(maxLength)) }
        let pad = maxLength - tokens.count
        // DELIBERATE DIVERGENCE, measured benign: we pad with `<unk>`(3) where the oracle pads
        // with `pad_token_id` = `<pad>`(0). Verified empirically, not argued — see
        // `parity/probe_pad_token_effect.py` (receipt: parity/goldens/pad_token_probe/report.json),
        // run in the regime that can actually discriminate: prompt "a red fox", 4 valid tokens,
        // 1020/1024 slots (99.6%) padding, so pad tokens dominate if they matter at all.
        //   · Gemma hidden states at VALID positions, pad0 vs pad3: 0 of 15360 elements differ,
        //     on all 49 layers — the combined causal+padding mask isolates pad keys exactly.
        //   · FINAL connector outputs (what the DiT consumes): BIT-IDENTICAL —
        //     0/4194304 video, 0/2097152 audio elements differ.
        //   · Controls confirm the check could have failed: pad-POSITION hidden states differ
        //     hugely (maxAbs 1.01e5 @ layer 42), and re-running the connector with the mask
        //     forced to all-ones — so nothing is discarded — moves 4003767/4194304 video
        //     elements (maxAbs 1.25).
        // Two independent mechanisms make it irrelevant: `GemmaFeaturesExtractorV2` zeroes
        // padded positions before the projection, and `_replace_padding_with_registers` then
        // overwrites them with learned registers. The id only has to be a real vocab token.
        let padToken = tk.unknownTokenId ?? 0
        let ids = Array(repeating: padToken, count: pad) + tokens
        let m = Array(repeating: 0, count: pad) + Array(repeating: 1, count: tokens.count)
        return (MLXArray(ids.map { Int32($0) }).reshaped(1, maxLength),
                MLXArray(m.map { Int32($0) }).reshaped(1, maxLength))
    }
}
