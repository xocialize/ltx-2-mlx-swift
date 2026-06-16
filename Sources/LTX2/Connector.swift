// Connector.swift — Gemma → video/audio text-embedding connector.
//
// 1:1 functional port of:
//   ltx_core_mlx/text_encoders/gemma/feature_extractor.py  (GemmaFeaturesExtractorV2,
//       TextEmbeddingProjection, TextEncoderConnector)
//   ltx_core_mlx/text_encoders/gemma/embeddings_connector.py (Embeddings1DConnector,
//       ConnectorAttention, ConnectorFeedForward, ConnectorTransformerBlock)
//
// Runs the forward functionally over the loaded `connector.safetensors` weight dict
// (keys keep their oracle paths, "connector." prefix stripped). The connector stays
// bf16 in production (never quantized), so a functional forward is the production
// shape too — Modules can wrap it later if needed.

import Foundation
import MLX
import MLXNN

public struct Connector {
    /// Weights keyed by oracle path with the leading "connector." stripped, e.g.
    /// "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight".
    let w: [String: MLXArray]

    // Config (Gemma 3 12B → LTX-2.3 dims)
    let embeddingDim = 3840
    let videoDim = 4096
    let audioDim = 2048
    let videoHeadDim = 128
    let audioHeadDim = 64
    let numLayers = 8
    let numRegisters = 128
    let maxPos = 4096
    let theta: Float = 10000.0

    static let debug = ProcessInfo.processInfo.environment["LTX_DEBUG"] != nil

    private func dbg(_ label: String, _ x: MLXArray) {
        guard Connector.debug else { return }
        let xf = x.asType(.float32)
        let nan = MLX.isNaN(xf).sum().item(Int.self)
        let amax = MLX.abs(xf).max().item(Float.self)
        print(String(format: "  [dbg] %-40@ shape=%@ nan=%d absmax=%.4f", label as NSString, "\(x.shape)" as NSString, nan, amax))
    }

    public init(weights: [String: MLXArray]) {
        // Strip a leading "connector." if present (native safetensors keys carry it).
        // Compute the connector in fp32: the 188160-wide text-embedding projection
        // overflows in bf16 matmul (mlx-swift's libmlx differs from Python mlx here).
        // Weights are bf16 on disk → lossless upcast; the connector stays bf16 *on disk*.
        var stripped: [String: MLXArray] = [:]
        for (k, v) in weights {
            let key = k.hasPrefix("connector.") ? String(k.dropFirst("connector.".count)) : k
            stripped[key] = v.asType(.float32)
        }
        self.w = stripped
    }

    /// Load connector weights from a safetensors file.
    public static func load(connectorPath: URL) throws -> Connector {
        let arrays = try MLX.loadArrays(url: connectorPath)
        return Connector(weights: arrays)
    }

    // MARK: - GemmaFeaturesExtractorV2.__call__

    /// hiddenStates: 49 tensors each (B, T, 3840). mask: (B, T) int (1=valid).
    /// Returns (videoEmbeds (B,T,4096), audioEmbeds (B,T,2048)).
    public func callAsFunction(hiddenStates: [MLXArray], mask: MLXArray) -> (video: MLXArray, audio: MLXArray) {
        // per_token_rms: stack on last axis -> (B,T,D,L), RMS over D (axis=2), reshape (B,T,D*L)
        let encoded = MLX.stacked(hiddenStates.map { $0.asType(.float32) }, axis: -1)  // (B,T,D,L)
        let variance = MLX.mean(encoded * encoded, axis: 2, keepDims: true)
        let normed = encoded * MLX.rsqrt(variance + 1e-6)
        let B = normed.dim(0), T = normed.dim(1), D = normed.dim(2), L = normed.dim(3)
        var stacked = normed.reshaped(B, T, D * L)  // D-interleaved

        // zero padding positions
        let mask3 = mask.expandedDimensions(axis: -1).asType(stacked.dtype)
        stacked = stacked * mask3
        eval(stacked)  // _materialize (oracle): keep the 49-layer cast+stack off the projection buffer

        return connectorForward(stacked, mask: mask)
    }

    // MARK: - TextEncoderConnector.__call__

    private func connectorForward(_ stacked: MLXArray, mask: MLXArray) -> (video: MLXArray, audio: MLXArray) {
        // TextEmbeddingProjection: rescale then project (separate video/audio)
        let vScale = (Float(videoDim) / Float(embeddingDim)).squareRoot()
        let video0 = dense(stacked * vScale,
                           w["text_embedding_projection.video_aggregate_embed.weight"]!,
                           w["text_embedding_projection.video_aggregate_embed.bias"])
        let aScale = (Float(audioDim) / Float(embeddingDim)).squareRoot()
        let audio0 = dense(stacked * aScale,
                           w["text_embedding_projection.audio_aggregate_embed.weight"]!,
                           w["text_embedding_projection.audio_aggregate_embed.bias"])

        eval(video0, audio0)  // _materialize (oracle): split the 188160-wide projection into its own buffer
        dbg("video0", video0); dbg("audio0", audio0)
        let video = embeddings1D(video0, mask: mask, prefix: "video_embeddings_connector",
                                 dim: videoDim, headDim: videoHeadDim)
        let audio = embeddings1D(audio0, mask: mask, prefix: "audio_embeddings_connector",
                                 dim: audioDim, headDim: audioHeadDim)
        return (video, audio)
    }

    // MARK: - Embeddings1DConnector.__call__

    private func embeddings1D(_ hidden: MLXArray, mask: MLXArray, prefix: String, dim: Int, headDim: Int) -> MLXArray {
        let numHeads = dim / headDim
        let B = hidden.dim(0)

        // Replace padding tokens with tiled learnable registers (left-padded input)
        let lr = w["\(prefix).learnable_registers"]!                  // (numReg, dim)
        let registers = MLX.broadcast(lr.expandedDimensions(axis: 0), to: [B, numRegisters, dim])
        var h = replacePaddingWithRegisters(hidden, mask: mask, registers: registers)
        dbg("\(prefix) afterRegisters", h)

        // split-RoPE on 1D positions
        let seq = h.dim(1)
        let positions = MLXArray(0 ..< seq).asType(.float32).reshaped(1, seq, 1)
        let (cosF, sinF) = RoPE.precomputeSplit(positions: positions, innerDim: dim,
                                                numHeads: numHeads, theta: theta, maxPos: [maxPos])

        for i in 0 ..< numLayers {
            h = block(h, prefix: "\(prefix).transformer_1d_blocks.\(i)", cos: cosF, sin: sinF,
                      numHeads: numHeads, headDim: headDim)
            eval(h)  // per-block materialize (watchdog discipline, mirrors oracle)
            dbg("\(prefix) block\(i)", h)
        }

        // affine-free output norm
        return rms0(h, eps: 1e-6)
    }

    // MARK: - ConnectorTransformerBlock (pre-norm, affine-free)

    private func block(_ x: MLXArray, prefix: String, cos: MLXArray, sin: MLXArray, numHeads: Int, headDim: Int) -> MLXArray {
        var h = x + attention(rms0(x, eps: 1e-6), prefix: "\(prefix).attn1", cos: cos, sin: sin,
                              numHeads: numHeads, headDim: headDim)
        h = h + feedForward(rms0(h, eps: 1e-6), prefix: "\(prefix).ff")
        return h
    }

    // MARK: - ConnectorAttention (QK-RMSNorm + split-RoPE + per-head gating)

    private func attention(_ x: MLXArray, prefix: String, cos: MLXArray, sin: MLXArray, numHeads: Int, headDim: Int) -> MLXArray {
        let B = x.dim(0), N = x.dim(1)
        let scale = 1.0 / Float(headDim).squareRoot()

        var q = dense(x, w["\(prefix).to_q.weight"]!, w["\(prefix).to_q.bias"])
        var k = dense(x, w["\(prefix).to_k.weight"]!, w["\(prefix).to_k.bias"])
        let v0 = dense(x, w["\(prefix).to_v.weight"]!, w["\(prefix).to_v.bias"])

        // QK RMSNorm over full inner_dim (nn.RMSNorm default eps 1e-5)
        q = rmsW(q, w["\(prefix).q_norm.weight"]!, eps: 1e-5)
        k = rmsW(k, w["\(prefix).k_norm.weight"]!, eps: 1e-5)

        q = q.reshaped(B, N, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(B, N, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = v0.reshaped(B, N, numHeads, headDim).transposed(0, 2, 1, 3)

        q = RoPE.applySplit(q, cos: cos, sin: sin)
        k = RoPE.applySplit(k, cos: cos, sin: sin)

        var attn = q.matmul(k.transposed(0, 1, 3, 2)) * scale
        attn = MLX.softmax(attn, axis: -1)
        var out = attn.matmul(v)  // (B,H,N,headDim)

        // per-head gate: 2 * sigmoid(logits)
        let gateLogits = dense(x, w["\(prefix).to_gate_logits.weight"]!, w["\(prefix).to_gate_logits.bias"])  // (B,N,H)
        let gate = 2.0 * MLX.sigmoid(gateLogits)
        out = out * gate.transposed(0, 2, 1).expandedDimensions(axis: -1)  // (B,H,N,1)

        out = out.transposed(0, 2, 1, 3).reshaped(B, N, numHeads * headDim)
        return dense(out, w["\(prefix).to_out.0.weight"]!, w["\(prefix).to_out.0.bias"])
    }

    // MARK: - ConnectorFeedForward (GELU-approx)

    private func feedForward(_ x: MLXArray, prefix: String) -> MLXArray {
        var h = dense(x, w["\(prefix).net.0.proj.weight"]!, w["\(prefix).net.0.proj.bias"])
        h = geluApproximate(h)
        return dense(h, w["\(prefix).net.2.weight"]!, w["\(prefix).net.2.bias"])
    }

    // MARK: - _replace_padded_with_learnable_registers

    private func replacePaddingWithRegisters(_ hidden: MLXArray, mask: MLXArray, registers: MLXArray) -> MLXArray {
        let B = hidden.dim(0), seq = hidden.dim(1), dim = hidden.dim(2)
        let numReg = registers.dim(1)
        let numTiles = seq / numReg
        let tiled = MLX.tiled(registers, repetitions: [1, numTiles, 1])  // (B, seq, dim)

        var results: [MLXArray] = []
        for b in 0 ..< B {
            let nValid = Int(mask[b].sum().item(Int.self))
            let hb = hidden[b]                       // (seq, dim)
            let valid = hb[(seq - nValid)...]        // (nValid, dim) — valid tokens at END (left-pad)
            var adjusted = valid
            if nValid < seq {
                let padding = MLXArray.zeros([seq - nValid, dim], dtype: valid.dtype)
                adjusted = MLX.concatenated([valid, padding], axis: 0)
            }
            let flipped = MLX.concatenated([MLXArray.ones([nValid, 1]), MLXArray.zeros([seq - nValid, 1])], axis: 0)
                .asType(adjusted.dtype)
            let blended = flipped * adjusted + (1.0 - flipped) * tiled[b]
            results.append(blended)
        }
        return MLX.stacked(results, axis: 0)
    }

    // MARK: - helpers

    /// y = x @ W.T (+ b). W stored (out, in) as in PyTorch/MLX Linear.
    private func dense(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray?) -> MLXArray {
        var y = x.matmul(weight.transposed())
        if let bias { y = y + bias }
        return y
    }

    /// Affine-free RMS norm (norm computed in fp32, like mx.fast.rms_norm).
    private func rms0(_ x: MLXArray, eps: Float) -> MLXArray {
        let xf = x.asType(.float32)
        let v = MLX.mean(xf * xf, axis: -1, keepDims: true)
        return (xf * MLX.rsqrt(v + eps)).asType(x.dtype)
    }

    private func rmsW(_ x: MLXArray, _ weight: MLXArray, eps: Float) -> MLXArray {
        return rms0(x, eps: eps) * weight
    }
}
