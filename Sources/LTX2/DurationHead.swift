// DurationHead.swift — LTX-2.5 duration head: predicts shot duration in seconds.
//
// 1:1 functional port of the oracle's `ltx_core_mlx/duration_head/duration_head.py`.
//
// Consumes the CONNECTOR outputs (the same context the DiT receives), so it runs right
// after prompt encoding and before any latent work. No attention mask: the connector
// substitutes learnable registers for padded positions and marks the result fully
// attendable, so every token here is already valid.
//
// The regression target is trained in log-seconds; `exp` is applied at the end so callers
// always get seconds.
//
// This is the component behind claim C6 and bench-matrix arm D (the duration-head honesty
// check). The oracle's verdict on C6 is already in and this port does not re-litigate it:
// PARTIAL — the head responds to pacing cues in the prompt but not to explicit durations.
// What this port buys is the Swift-side arm of that battery.
//
// ## Weight layout
//
// ⚠️ Upstream uses `nn.MultiheadAttention`, whose checkpoint carries a FUSED
// `in_proj_weight` (3D, D) + `in_proj_bias`. The CONVERTER already split it into
// `q_proj` / `k_proj` / `v_proj`, and the shipped `duration_head.safetensors` has the
// split form (19 tensors, verified against the file) — so this port keys off the split
// names and never sees the fused one.
//
// ## Numerics
//
// The pooler softmaxes ONE query over ~2048 keys and the pooled vector sits ~20× below the
// token scale, so ordinary fp32 accumulation differences in the input projections (1.7e-6)
// amplify to ~1.6e-3 on `pooled` and ~2e-4 on the predicted seconds. That is inherent to
// the shape, not a kernel choice. It is four orders of magnitude below the frame-grid
// quantization the prediction feeds (8/24 s), so the shipped frame count is unaffected —
// which is why the gate checks BOTH a relative tolerance on the seconds and exact equality
// on the snapped frame count.
//
// fp32 throughout: non-DiT components in this port run fp32 (wide bf16 matmuls NaN in
// mlx-swift). The stored weights are bf16 on disk and cast on load.

import Foundation
import MLX
import MLXFast
import MLXNN

public struct DurationHead {

    public struct Config {
        public var videoCrossAttentionDim = 4096
        public var audioCrossAttentionDim = 2048
        public var poolerHiddenDim = 256
        public var numQueries = 1
        public var numPoolerHeads = 4
        public var mlpHiddenDim = 256
        public init() {}
    }

    public let cfg: Config
    let w: [String: MLXArray]

    public init(weights: [String: MLXArray], config: Config = Config()) {
        var m: [String: MLXArray] = [:]
        for (k, v) in weights {
            let key = k.hasPrefix("duration_head.") ? String(k.dropFirst("duration_head.".count)) : k
            m[key] = v.asType(.float32)
        }
        self.w = m
        self.cfg = config
    }

    public static func load(path: URL, config: Config = Config()) throws -> DurationHead {
        DurationHead(weights: try MLX.loadArrays(url: path), config: config)
    }

    private func dense(_ x: MLXArray, _ prefix: String) -> MLXArray {
        var y = x.matmul(w["\(prefix).weight"]!.transposed())
        if let b = w["\(prefix).bias"] { y = y + b }
        return y
    }

    /// Torch `nn.MultiheadAttention` equivalent (batch_first, no mask), split q/k/v.
    private func crossAttention(q: MLXArray, kv: MLXArray) -> MLXArray {
        let B = q.dim(0), Tq = q.dim(1), D = q.dim(2)
        let Tk = kv.dim(1)
        let H = cfg.numPoolerHeads, hd = D / H

        let qh = dense(q, "attention_pooler.cross_attn.q_proj").reshaped(B, Tq, H, hd).transposed(0, 2, 1, 3)
        let kh = dense(kv, "attention_pooler.cross_attn.k_proj").reshaped(B, Tk, H, hd).transposed(0, 2, 1, 3)
        let vh = dense(kv, "attention_pooler.cross_attn.v_proj").reshaped(B, Tk, H, hd).transposed(0, 2, 1, 3)

        var out = MLXFast.scaledDotProductAttention(
            queries: qh, keys: kh, values: vh, scale: Float(pow(Double(hd), -0.5)), mask: .none)
        out = out.transposed(0, 2, 1, 3).reshaped(B, Tq, D)
        return dense(out, "attention_pooler.cross_attn.out_proj")
    }

    /// Predict duration in seconds, shape (B,), from one or both connector outputs.
    ///
    /// At least one of `videoTokens` / `audioTokens` must be supplied. When both are given
    /// they are projected to the shared pooler dim, offset by their modality embedding, and
    /// CONCATENATED along the token axis before pooling — video first.
    public func callAsFunction(videoTokens: MLXArray? = nil,
                               audioTokens: MLXArray? = nil) -> MLXArray {
        precondition(videoTokens != nil || audioTokens != nil,
                     "DurationHead requires at least one of videoTokens / audioTokens")

        var groups: [MLXArray] = []
        if let v = videoTokens {
            groups.append(dense(v, "video_input_proj") + w["video_modality_emb"]!)
        }
        if let a = audioTokens {
            groups.append(dense(a, "audio_input_proj") + w["audio_modality_emb"]!)
        }
        let tokens = groups.count == 1 ? groups[0] : MLX.concatenated(groups, axis: 1)

        let B = tokens.dim(0)
        let queries = MLX.broadcast(w["attention_pooler.query_tokens"]!.expandedDimensions(axis: 0),
                                    to: [B, cfg.numQueries, cfg.poolerHiddenDim])
        let pooled = crossAttention(q: queries, kv: tokens)

        let hidden = geluApproximate(dense(pooled.reshaped(B, -1), "mlp_hidden"))
        let logDuration = dense(hidden, "mlp_out").squeezed(axis: -1)
        return MLX.exp(logDuration)
    }

    /// Clamp a predicted duration and snap it to the causal frame grid (8k+1).
    ///
    /// Mirrors upstream `seconds_to_clamped_num_frames`: clamp to `[minS, maxS]` (pipeline
    /// defaults 1.0 / 20.0), then round to the nearest valid frame count with
    /// `(frames - 1) % 8 == 0`.
    ///
    /// This — not the raw seconds — is the deliverable the pipeline consumes, which is why
    /// the gate checks it for EXACT equality against the reference while the seconds only
    /// have to land inside a relative tolerance.
    public static func secondsToNumFrames(_ seconds: Double, frameRate: Double,
                                          minS: Double = 1.0, maxS: Double = 20.0) -> Int {
        let clamped = max(minS, min(maxS, seconds))
        let raw = clamped * frameRate
        let k = ((raw - 1) / 8).rounded()
        return max(9, 8 * Int(k) + 1)
    }
}
