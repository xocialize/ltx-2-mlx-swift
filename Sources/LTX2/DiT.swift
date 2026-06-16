// DiT.swift — LTX-2.3 joint audio+video Diffusion Transformer.
//
// 1:1 functional port of ltx_core_mlx/model/transformer/{model,transformer,
// attention,feed_forward,adaln,timestep_embedding}.py. Scalar-timestep path
// (t2v/distilled — per-token timesteps for i2v/retake conditioning are a later
// addition). Weights are loaded as a flattened param-tree dict (keys match the
// oracle's `tree_flatten(model.parameters())`). Computed in fp32 for the parity
// gate (matches the saved goldens; bf16/quant come with the full-scale tier).

import Foundation
import MLX
import MLXFast
import MLXNN

public struct DiTConfig {
    public var numLayers = 48
    public var videoDim = 4096, audioDim = 2048
    public var videoNumHeads = 32, audioNumHeads = 32
    public var videoHeadDim = 128, audioHeadDim = 64
    public var avCrossNumHeads = 32, avCrossHeadDim = 64
    public var videoPatchChannels = 128, audioPatchChannels = 128
    public var ffMult: Float = 4.0
    public var timestepEmbeddingDim = 256
    public var timestepScaleMultiplier: Float = 1000.0
    public var avCaTimestepScaleMultiplier: Float = 1000.0
    public var ropeTheta: Float = 10000.0
    public var positionalMaxPos = [20, 2048, 2048]
    public var audioPositionalMaxPos = [20]
    public var normEps: Float = 1e-6
    public init() {}
}

public struct DiT {
    let w: [String: MLXArray]
    let cfg: DiTConfig
    let dtype: DType

    /// - computeDtype: fp32 for the tiny gate (vs LTX2_DIT_FP32 golden); bf16 for
    ///   full-scale real weights (matches the production oracle; 35GB stays 35GB).
    public init(weights: [String: MLXArray], config: DiTConfig, computeDtype: DType = .float32) {
        var m: [String: MLXArray] = [:]
        for (k, v) in weights {
            let key = k.hasPrefix("transformer.") ? String(k.dropFirst("transformer.".count)) : k
            m[key] = v.asType(computeDtype)
        }
        self.w = m
        self.cfg = config
        self.dtype = computeDtype
    }

    public static func load(weightsPath: URL, config: DiTConfig, computeDtype: DType = .float32) throws -> DiT {
        DiT(weights: try MLX.loadArrays(url: weightsPath), config: config, computeDtype: computeDtype)
    }

    // Per-block conditioning, computed once in the prelude.
    struct Cond {
        var videoAdaln, audioAdaln, videoPrompt, audioPrompt: MLXArray
        var avCaVideo, avCaAudio, avA2vGate, avV2aGate: MLXArray
        var videoText, audioText: MLXArray?
        var videoRope, audioRope, videoCrossRope, audioCrossRope: (MLXArray, MLXArray)?
    }

    /// Forward: (video_latent, audio_latent, sigma, text embeds, positions) → (video_v, audio_v).
    public func callAsFunction(
        videoLatent: MLXArray, audioLatent: MLXArray, sigma: MLXArray,
        videoText: MLXArray?, audioText: MLXArray?,
        videoPositions: MLXArray, audioPositions: MLXArray
    ) -> (video: MLXArray, audio: MLXArray) {
        let vd = cfg.videoDim, ad = cfg.audioDim, tED = cfg.timestepEmbeddingDim

        var videoHidden = dense(videoLatent.asType(dtype), "patchify_proj")
        var audioHidden = dense(audioLatent.asType(dtype), "audio_patchify_proj")

        let t = sigma.asType(.float32)
        let tEmb = timestepEmbedding(t * cfg.timestepScaleMultiplier, tED)
        let avFactor = cfg.avCaTimestepScaleMultiplier / cfg.timestepScaleMultiplier
        let tEmbAvGate = timestepEmbedding(t * cfg.timestepScaleMultiplier * avFactor, tED)

        let (videoAdaln, videoEmbeddedTs) = adalnSingle(tEmb, "adaln_single", 9, vd)
        let (avCaVideo, _) = adalnSingle(tEmb, "av_ca_video_scale_shift_adaln_single", 4, vd)
        let (avA2vGate, _) = adalnSingle(tEmbAvGate, "av_ca_a2v_gate_adaln_single", 1, vd)
        let (videoPrompt, _) = adalnSingle(tEmb, "prompt_adaln_single", 2, vd)
        let (audioAdaln, audioEmbeddedTs) = adalnSingle(tEmb, "audio_adaln_single", 9, ad)
        let (avCaAudio, _) = adalnSingle(tEmb, "av_ca_audio_scale_shift_adaln_single", 4, ad)
        let (avV2aGate, _) = adalnSingle(tEmbAvGate, "av_ca_v2a_gate_adaln_single", 1, ad)
        let (audioPrompt, _) = adalnSingle(tEmb, "audio_prompt_adaln_single", 2, ad)

        let videoRope = ropeFreqs(videoPositions, cfg.videoNumHeads, cfg.videoHeadDim,
                                  maxPos: Array(cfg.positionalMaxPos.prefix(videoPositions.dim(-1))))
        let audioRope = ropeFreqs(audioPositions, cfg.audioNumHeads, cfg.audioHeadDim,
                                  maxPos: cfg.audioPositionalMaxPos)
        let crossMax = max(cfg.positionalMaxPos[0], cfg.audioPositionalMaxPos[0])
        let videoCrossRope = ropeFreqs(videoPositions[0..., 0..., 0 ..< 1], cfg.avCrossNumHeads, cfg.avCrossHeadDim, maxPos: [crossMax])
        let audioCrossRope = ropeFreqs(audioPositions[0..., 0..., 0 ..< 1], cfg.avCrossNumHeads, cfg.avCrossHeadDim, maxPos: [crossMax])

        let cond = Cond(
            videoAdaln: videoAdaln, audioAdaln: audioAdaln, videoPrompt: videoPrompt, audioPrompt: audioPrompt,
            avCaVideo: avCaVideo, avCaAudio: avCaAudio, avA2vGate: avA2vGate, avV2aGate: avV2aGate,
            videoText: videoText?.asType(.float32), audioText: audioText?.asType(.float32),
            videoRope: videoRope, audioRope: audioRope, videoCrossRope: videoCrossRope, audioCrossRope: audioCrossRope)

        for i in 0 ..< cfg.numLayers {
            (videoHidden, audioHidden) = block(i, videoHidden, audioHidden, cond)
            if (i + 1) % 8 == 0 { eval(videoHidden, audioHidden) }
        }

        let videoOut = outputBlock(videoHidden, videoEmbeddedTs, "scale_shift_table", "proj_out")
        let audioOut = outputBlock(audioHidden, audioEmbeddedTs, "audio_scale_shift_table", "audio_proj_out")
        return (videoOut, audioOut)
    }

    // MARK: - BasicAVTransformerBlock

    private func block(_ i: Int, _ videoHidden0: MLXArray, _ audioHidden0: MLXArray, _ c: Cond) -> (MLXArray, MLXArray) {
        let p = "transformer_blocks.\(i)"
        let vd = cfg.videoDim, ad = cfg.audioDim
        var videoHidden = videoHidden0, audioHidden = audioHidden0

        let v = unpackAdaln(c.videoAdaln, "\(p).scale_shift_table", 9, vd)   // sa(0,1,2) ff(3,4,5) ca(6,7,8)
        let a = unpackAdaln(c.audioAdaln, "\(p).audio_scale_shift_table", 9, ad)
        let avV = unpackAdaln(c.avCaVideo, "\(p).scale_shift_table_a2v_ca_video", 4, vd)  // scale_a2v,shift_a2v,scale_v2a,shift_v2a
        let avA = unpackAdaln(c.avCaAudio, "\(p).scale_shift_table_a2v_ca_audio", 4, ad)
        let avVGate = gateParam(c.avA2vGate, "\(p).scale_shift_table_a2v_ca_video", vd)
        let avAGate = gateParam(c.avV2aGate, "\(p).scale_shift_table_a2v_ca_audio", ad)

        // 1. video self-attn
        let vNormSA = rms0(videoHidden) * (1.0 + v[1]) + v[0]
        videoHidden = videoHidden + attention(vNormSA, prefix: "\(p).attn1", numHeads: cfg.videoNumHeads, headDim: cfg.videoHeadDim, rope: c.videoRope) * v[2]
        // 2. audio self-attn
        let aNormSA = rms0(audioHidden) * (1.0 + a[1]) + a[0]
        audioHidden = audioHidden + attention(aNormSA, prefix: "\(p).audio_attn1", numHeads: cfg.audioNumHeads, headDim: cfg.audioHeadDim, rope: c.audioRope) * a[2]
        // 3. video text cross-attn (indices 6,7,8) + prompt table
        if let vText = c.videoText {
            let vNormCA = rms0(videoHidden) * (1.0 + v[7]) + v[6]
            let vp = unpackAdaln(c.videoPrompt, "\(p).prompt_scale_shift_table", 2, vd)
            let textScaled = vText * (1.0 + vp[1]) + vp[0]
            videoHidden = videoHidden + attention(vNormCA, prefix: "\(p).attn2", numHeads: cfg.videoNumHeads, headDim: cfg.videoHeadDim, kv: textScaled, useRope: false) * v[8]
        }
        // 4. audio text cross-attn
        if let aText = c.audioText {
            let aNormCA = rms0(audioHidden) * (1.0 + a[7]) + a[6]
            let ap = unpackAdaln(c.audioPrompt, "\(p).audio_prompt_scale_shift_table", 2, ad)
            let textScaled = aText * (1.0 + ap[1]) + ap[0]
            audioHidden = audioHidden + attention(aNormCA, prefix: "\(p).audio_attn2", numHeads: cfg.audioNumHeads, headDim: cfg.audioHeadDim, kv: textScaled, useRope: false) * a[8]
        }
        // 5-6. AV cross-modal (norm both once; tables: 0=scale_a2v 1=shift_a2v 2=scale_v2a 3=shift_v2a)
        let vNorm3 = rms0(videoHidden), aNorm3 = rms0(audioHidden)
        let videoQa2v = vNorm3 * (1.0 + avV[0]) + avV[1]
        let audioKVa2v = aNorm3 * (1.0 + avA[0]) + avA[1]
        let a2v = attention(videoQa2v, prefix: "\(p).audio_to_video_attn", numHeads: cfg.avCrossNumHeads, headDim: cfg.avCrossHeadDim, kv: audioKVa2v, rope: c.videoCrossRope, ropeK: c.audioCrossRope) * avVGate
        videoHidden = videoHidden + a2v
        let audioQv2a = aNorm3 * (1.0 + avA[2]) + avA[3]
        let videoKVv2a = vNorm3 * (1.0 + avV[2]) + avV[3]
        let v2a = attention(audioQv2a, prefix: "\(p).video_to_audio_attn", numHeads: cfg.avCrossNumHeads, headDim: cfg.avCrossHeadDim, kv: videoKVv2a, rope: c.audioCrossRope, ropeK: c.videoCrossRope) * avAGate
        audioHidden = audioHidden + v2a
        // 7. video FF (indices 3,4,5)
        let vNormFF = rms0(videoHidden) * (1.0 + v[4]) + v[3]
        videoHidden = videoHidden + feedForward(vNormFF, "\(p).ff") * v[5]
        // 8. audio FF
        let aNormFF = rms0(audioHidden) * (1.0 + a[4]) + a[3]
        audioHidden = audioHidden + feedForward(aNormFF, "\(p).audio_ff") * a[5]

        return (videoHidden, audioHidden)
    }

    // MARK: - Attention (single to_out, QK-RMSNorm eps=normEps, split-RoPE, gated, fused SDPA)

    private func attention(_ x: MLXArray, prefix: String, numHeads: Int, headDim: Int,
                           kv: MLXArray? = nil, rope: (MLXArray, MLXArray)? = nil,
                           ropeK: (MLXArray, MLXArray)? = nil, useRope: Bool = true) -> MLXArray {
        let kvInput = kv ?? x
        let B = x.dim(0)
        var q = rmsW(dense(x, "\(prefix).to_q"), w["\(prefix).q_norm.weight"]!, eps: cfg.normEps)
        var k = rmsW(dense(kvInput, "\(prefix).to_k"), w["\(prefix).k_norm.weight"]!, eps: cfg.normEps)
        var v = dense(kvInput, "\(prefix).to_v")
        q = q.reshaped(B, -1, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(B, -1, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(B, -1, numHeads, headDim).transposed(0, 2, 1, 3)
        if useRope, let (cosF, sinF) = rope {
            q = RoPE.applySplit(q, cos: cosF, sin: sinF)
            let (ck, sk) = ropeK ?? (cosF, sinF)
            k = RoPE.applySplit(k, cos: ck, sin: sk)
        }
        let scale = 1.0 / Float(headDim).squareRoot()
        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        // per-head gate
        let gate = 2.0 * MLX.sigmoid(dense(x, "\(prefix).to_gate_logits"))  // (B,N,H)
        out = out * gate.transposed(0, 2, 1).expandedDimensions(axis: -1)    // (B,H,N,1)
        out = out.transposed(0, 2, 1, 3).reshaped(B, -1, numHeads * headDim)
        return dense(out, "\(prefix).to_out")
    }

    private func feedForward(_ x: MLXArray, _ prefix: String) -> MLXArray {
        dense(geluApproximate(dense(x, "\(prefix).proj_in")), "\(prefix).proj_out")
    }

    // MARK: - conditioning helpers

    /// Unpack AdaLN params (B, P*dim) + per-block table[:P] → P arrays each (B,1,dim).
    private func unpackAdaln(_ params: MLXArray, _ tableKey: String, _ numParams: Int, _ dim: Int) -> [MLXArray] {
        let table = w[tableKey]!                       // (P_full, dim)
        let B = params.dim(0)
        let p = params.reshaped(B, numParams, dim) + table[0 ..< numParams].expandedDimensions(axis: 0)
        return (0 ..< numParams).map { p[0..., $0, 0...].expandedDimensions(axis: 1) }  // (B,1,dim)
    }

    /// 1-param gate: (B, dim) + table row 4 → (B,1,dim).
    private func gateParam(_ params: MLXArray, _ tableKey: String, _ dim: Int) -> MLXArray {
        let row = w[tableKey]![4]                       // (dim,)
        return (params + row).expandedDimensions(axis: 1)
    }

    private func adalnSingle(_ tEmb: MLXArray, _ prefix: String, _ numParams: Int, _ dim: Int) -> (MLXArray, MLXArray) {
        let embedded = timestepEmbedderMLP(tEmb, "\(prefix).emb.timestep_embedder")  // (B, dim)
        let params = dense(silu(embedded), "\(prefix).linear")                        // (B, numParams*dim)
        return (params, embedded)
    }

    private func timestepEmbedderMLP(_ x: MLXArray, _ prefix: String) -> MLXArray {
        dense(silu(dense(x, "\(prefix).linear1")), "\(prefix).linear2")
    }

    private func timestepEmbedding(_ t: MLXArray, _ dim: Int) -> MLXArray {
        let half = dim / 2
        let exponent = (-Foundation.log(Float(10000.0)) * MLXArray(0 ..< half).asType(.float32)) / Float(half)
        let freqs = MLX.exp(exponent)                                   // (half,)
        let args = t.asType(.float32).expandedDimensions(axis: -1) * freqs  // (B, half)
        return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)    // flip_sin_to_cos
    }

    private func ropeFreqs(_ positions: MLXArray, _ numHeads: Int, _ headDim: Int, maxPos: [Int]) -> (MLXArray, MLXArray) {
        RoPE.precomputeSplit(positions: positions, innerDim: numHeads * headDim, numHeads: numHeads, theta: cfg.ropeTheta, maxPos: maxPos)
    }

    private func outputBlock(_ x: MLXArray, _ embeddedTs: MLXArray, _ tableKey: String, _ projPrefix: String) -> MLXArray {
        var et = embeddedTs
        if et.ndim == 2 { et = et.expandedDimensions(axis: 1) }  // (B,1,dim)
        let table = w[tableKey]!                                  // (2, dim)
        let ssv = table.expandedDimensions(axis: 0).expandedDimensions(axis: 0) + et.expandedDimensions(axis: 2)  // (B,N|1,2,dim)
        let shift = ssv[0..., 0..., 0, 0...]
        let scale = ssv[0..., 0..., 1, 0...]
        let y = layerNormAffineFree(x) * (1.0 + scale) + shift
        return dense(y, projPrefix)
    }

    // MARK: - primitives

    private func dense(_ x: MLXArray, _ prefix: String) -> MLXArray {
        var y = x.matmul(w["\(prefix).weight"]!.transposed())
        if let b = w["\(prefix).bias"] { y = y + b }
        return y
    }

    private func rms0(_ x: MLXArray, eps: Float? = nil) -> MLXArray {
        let e = eps ?? cfg.normEps
        let xf = x.asType(.float32)  // fp32-internal, like mx.fast.rms_norm
        let v = MLX.mean(xf * xf, axis: -1, keepDims: true)
        return (xf * MLX.rsqrt(v + e)).asType(x.dtype)
    }

    private func rmsW(_ x: MLXArray, _ weight: MLXArray, eps: Float) -> MLXArray { rms0(x, eps: eps) * weight }

    private func layerNormAffineFree(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)  // fp32-internal, like mx.fast.layer_norm
        let mean = MLX.mean(xf, axis: -1, keepDims: true)
        let xc = xf - mean
        let v = MLX.mean(xc * xc, axis: -1, keepDims: true)
        return (xc * MLX.rsqrt(v + cfg.normEps)).asType(x.dtype)
    }
}
