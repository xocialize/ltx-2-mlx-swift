// DiffusionVideoDecoder.swift — LTX-2.5 DiffVAE, the diffusion video decoder.
//
// 1:1 functional port of the oracle's
// `ltx_core_mlx/model/video_vae/diffusion_video_decoder.py`, which follows upstream's
// COMBINED (full-volume) semantics. The chunked/deferred path is deliberately NOT
// followed: its W-chunking edge-replicates halo columns instead of shifting the NA
// window, which differs from full-volume NATTEN at the outermost `kw/2` columns.
//
// This is the second decoder for the 2.5 latent, alongside the conv `VideoVAEDecoder`.
// It is the component behind claim C3 ("Diffusion video decoder") and bench-matrix arm C
// (the decode triangle). On the oracle the fidelity verdict is already in and this port
// does not re-litigate it: at fixed latents the CONV decoder is the more faithful of the
// two (42.6 dB vs 36.8 dB PSNR), and DiffVAE is far slower (15.5 s vs 0.4 s at
// 512×768×9, decoder-isolated). What this port buys is the Swift-side arm of that
// comparison.
//
// ## Shape/layout contract
//
// The latent and the final pixels are channels-FIRST `(B, C, T, H, W)`; everything
// between `conv_in` and `conv_out` is channels-LAST `(B, T, H, W, C)`.
//
// ## Traps encoded here (each cost a careful read of the reference)
//
//  * PER-BLOCK CONTEXT RE-INJECTION. Every one of the 8 stage-5 blocks adds its own
//    `context_proj(context)` — the reference achieves this by having each block read an
//    aliased `[context | x]` buffer. Dropping it after the first block is an easy and
//    silent bug.
//  * ADALN CHUNK ORDER IS SCALE-THEN-SHIFT: `scale_msa, shift_msa, gate_msa, scale_mlp,
//    shift_mlp, gate_mlp, gate_ctx`, and the effective value is
//    `shared_adaln_chunk_i + scale_shift_table[i]`. The three gates are computed upstream
//    but discarded — they are pre-folded into `attn.proj` / `mlp.w_down` /
//    `context_proj`. (Verified: neither shipped checkpoint carries gate tensors, so the
//    fold is a no-op on both sides.)
//  * MODULATION IS APPLIED TO THE NORMED TENSOR: `norm(x) * (1 + scale) + shift`.
//  * THE TWO PIXEL-SHUFFLES PACK CHANNELS DIFFERENTLY: the upsample packs `(c, t, h, w)`
//    while patchify/unpatchify pack `(c, t, w, h)` — W before H.
//  * `norm_out` HAS NO MODULATION (bare RMSNorm), unlike most DiT heads.
//  * `decode` RETURNS [-1, 1]; the `(x+1)*0.5` clamp lives only in `decodeToRGB`.
//
// ## fp32
//
// Non-DiT components in this port run fp32 — wide bf16 matmuls NaN in mlx-swift, and the
// DiT is the only bf16 path (LTX_DEV/CLAUDE.md, "Reusable quirks"). Weights are cast on
// load here, same as `VideoVAEDecoder`.

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXProfiling

public struct DiffusionVideoDecoder {

    // MARK: - Configuration

    /// Stage geometry. The defaults are the shipped LTX-2.5 `vae_diffusion_decoder`
    /// checkpoint's actual shapes, read off the file rather than assumed (AB-R-0037).
    public struct Config {
        public var inChannels = 128
        public var outChannels = 3
        public var patchSize = 4
        public var headDim = 64
        public var tEmbDim = 384
        public var stageChannels = [2048, 1024, 512, 512, 256]
        public var stageDepths = [4, 6, 4, 2, 8]
        public var stageKernels: [(Int, Int, Int)] = [(3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)]
        public var stage5Kernel: (Int, Int, Int) = (11, 11, 11)
        public var upsampleStrides: [(Int, Int, Int)] = [(1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)]
        public var upsampleReductions = [2, 2, 1, 2]
        public var timestepScaleMultiplier: Float = 1000.0
        /// Only the 1-step x0 path is ported (see ``decode(_:noise:tile:key:)``).
        public var modelOutputType = "x0"

        public init() {}

        var contextChannels: Int { stageChannels[4] }
        var c5: Int { stageChannels[4] }
        /// Product of the temporal upsample strides — 8×.
        var temporalScale: Int { upsampleStrides.reduce(1) { $0 * $1.0 } }
        /// NATTEN last-frame border workaround: replicate this many trailing latent frames.
        var nattenTrailingPad: Int { (stageKernels[0].0 / 2) * 2 }
        /// Minimum latent tile the stage ladder can process — (3, 7, 7).
        var stageMinTileSizes: (Int, Int, Int) {
            (3, stageKernels.map(\.1).max() ?? 1, stageKernels.map(\.2).max() ?? 1)
        }
    }

    public let cfg: Config
    let w: [String: MLXArray]

    /// RoPE ladders, built once per decoder (they depend only on `headDim`).
    private let dimSplit: (Int, Int, Int)
    private let invT: MLXArray
    private let invH: MLXArray
    private let invW: MLXArray

    // MARK: - Loading

    public init(weights: [String: MLXArray], config: Config = Config()) {
        var m: [String: MLXArray] = [:]
        for (k, v) in weights {
            let key = k.hasPrefix("vae_diffusion_decoder.")
                ? String(k.dropFirst("vae_diffusion_decoder.".count)) : k
            m[key] = v.asType(.float32)
        }
        self.w = m
        self.cfg = config
        let split = NA3D.defaultRopeDimSplit(config.headDim)
        self.dimSplit = split
        self.invT = NA3D.ropeInvFreqs(split.0)
        self.invH = NA3D.ropeInvFreqs(split.1)
        self.invW = NA3D.ropeInvFreqs(split.2)
    }

    public static func load(path: URL, config: Config = Config()) throws -> DiffusionVideoDecoder {
        DiffusionVideoDecoder(weights: try MLX.loadArrays(url: path), config: config)
    }

    // MARK: - Primitives

    /// Weight lookup that names the missing key instead of trapping on a bare nil unwrap.
    ///
    /// A mistyped prefix here is otherwise a `Fatal error: Unexpectedly found nil` with no
    /// indication of WHICH of the 408 tensors was wanted — the expensive half of AB-R-0037.
    private func weight(_ key: String) -> MLXArray {
        guard let v = w[key] else {
            fatalError("DiffusionVideoDecoder: missing weight '\(key)' (checkpoint has \(w.count) tensors)")
        }
        return v
    }

    /// Both helpers take the module PREFIX and append the leaf themselves, so a caller never
    /// has to remember which of `.weight` / `.bias` a given layer carries.
    private func dense(_ x: MLXArray, _ prefix: String) -> MLXArray {
        var y = x.matmul(weight("\(prefix).weight").transposed())
        if let b = w["\(prefix).bias"] { y = y + b }
        return y
    }

    private func rms(_ x: MLXArray, _ prefix: String, eps: Float = 1e-6) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight("\(prefix).weight"), eps: eps)
    }

    /// `w_down( silu(x @ w_gate.T) * (x @ w_up.T) )` — all bias-free.
    private func swiGLU(_ x: MLXArray, _ prefix: String) -> MLXArray {
        dense(silu(dense(x, "\(prefix).w_gate")) * dense(x, "\(prefix).w_up"), "\(prefix).w_down")
    }

    /// Neighborhood attention with per-axis RoPE, returning the UN-projected output.
    ///
    /// The reference applies `attn.proj` outside, as part of the residual, so this stops
    /// short of it and the callers project.
    private func attention(_ x: MLXArray, _ prefix: String, kernel: (Int, Int, Int),
                           tile: (Int, Int, Int)) -> MLXArray {
        let B = x.dim(0), T = x.dim(1), H = x.dim(2), W = x.dim(3), C = x.dim(4)
        let hd = cfg.headDim
        let nh = C / hd

        func heads(_ t: MLXArray) -> MLXArray { t.reshaped(B, T, H, W, nh, hd) }

        var q = rms(heads(dense(x, "\(prefix).to_q")), "\(prefix).q_norm")
        var k = rms(heads(dense(x, "\(prefix).to_k")), "\(prefix).k_norm")
        let v = heads(dense(x, "\(prefix).to_v"))

        q = q * Float(pow(Double(hd), -0.5))  // folded before RoPE, as upstream
        q = NA3D.applyRope3D(q, invT: invT, invH: invH, invW: invW, dimSplit: dimSplit)
        k = NA3D.applyRope3D(k, invT: invT, invH: invH, invW: invW, dimSplit: dimSplit)

        return NA3D.attend(q, k, v, kernel: kernel, scale: 1.0, tile: tile).reshaped(B, T, H, W, C)
    }

    /// Deterministic block (stages 1-4): pre-norm attn + pre-norm SwiGLU.
    public func naBlock(_ x0: MLXArray, _ prefix: String, kernel: (Int, Int, Int),
                        tile: (Int, Int, Int)) -> MLXArray {
        var x = x0
        x = x + dense(attention(rms(x, "\(prefix).norm1"), "\(prefix).attn",
                                kernel: kernel, tile: tile), "\(prefix).attn.proj")
        return x + swiGLU(rms(x, "\(prefix).norm2"), "\(prefix).mlp")
    }

    /// Stage-5 block: context injection + modulated attn + modulated SwiGLU.
    ///
    /// `shared` is the 7-way AdaLN split; only chunks 0/1 (msa scale/shift) and 3/4 (mlp
    /// scale/shift) are consumed — the gates are pre-folded into the projections.
    public func diffusionNABlock(_ x0: MLXArray, _ prefix: String, context: MLXArray,
                                 shared: [MLXArray], kernel: (Int, Int, Int),
                                 tile: (Int, Int, Int)) -> MLXArray {
        let table = weight("\(prefix).scale_shift_table")
        func eff(_ i: Int) -> MLXArray { shared[i] + table[i].reshaped(1, 1, 1, 1, -1) }
        let sMSA = eff(0), shMSA = eff(1), sMLP = eff(3), shMLP = eff(4)

        var x = x0 + dense(context, "\(prefix).context_proj")   // every block re-injects
        var h = rms(x, "\(prefix).norm1") * (1.0 + sMSA) + shMSA
        x = x + dense(attention(h, "\(prefix).attn", kernel: kernel, tile: tile), "\(prefix).attn.proj")
        h = rms(x, "\(prefix).norm2") * (1.0 + sMLP) + shMLP
        return x + swiGLU(h, "\(prefix).mlp")
    }

    /// Linear then depth-to-space over (t, h, w). Channel packing: `(c, t, h, w)`.
    ///
    /// ⚠️ Note the packing differs from ``patchifyHW(_:patch:)`` — see the file header.
    public func upsample(_ x: MLXArray, _ prefix: String, stride: (Int, Int, Int), reduction: Int,
                         dropLeadingFrame: Bool) -> MLXArray {
        let B = x.dim(0), T = x.dim(1), H = x.dim(2), W = x.dim(3)
        let (p1, p2, p3) = stride
        let c = x.dim(4) / reduction
        var y = dense(x, "\(prefix).proj").reshaped(B, T, H, W, c, p1, p2, p3)
        // "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c"
        y = y.transposed(0, 1, 5, 2, 6, 3, 7, 4).reshaped(B, T * p1, H * p2, W * p3, c)
        if p1 == 2 && dropLeadingFrame { y = y[0..., 1...] }
        return y
    }

    // MARK: - Patchify

    /// (B,C,T,H,W) -> (B, C·patch², T, H/patch, W/patch).
    ///
    /// Reference packing is `(c p r q)` with q = h_sub, r = w_sub — i.e. channel index
    /// `c·patch² + w_sub·patch + h_sub`: **W before H**.
    public static func patchifyHW(_ x: MLXArray, patch: Int) -> MLXArray {
        let B = x.dim(0), C = x.dim(1), T = x.dim(2), H = x.dim(3), W = x.dim(4)
        var y = x.reshaped(B, C, T, H / patch, patch, W / patch, patch)  // c,t,hq,q,wr,r
        y = y.transposed(0, 1, 6, 4, 2, 3, 5)                            // b,c,r,q,t,h,w
        return y.reshaped(B, C * patch * patch, T, H / patch, W / patch)
    }

    /// Inverse of ``patchifyHW(_:patch:)``.
    public static func unpatchifyHW(_ x: MLXArray, patch: Int) -> MLXArray {
        let B = x.dim(0), CP = x.dim(1), T = x.dim(2), H = x.dim(3), W = x.dim(4)
        let C = CP / (patch * patch)
        var y = x.reshaped(B, C, patch, patch, T, H, W)  // c, r(w_sub), q(h_sub), t, h, w
        y = y.transposed(0, 1, 4, 5, 3, 6, 2)            // b, c, t, h, q, w, r
        return y.reshaped(B, C, T, H * patch, W * patch)
    }

    /// PixArt/diffusers sinusoid with `flip_sin_to_cos=True`, shift 0 → [cos, sin].
    public static func timestepEmbedding(_ t: MLXArray, dim: Int = 256) -> MLXArray {
        let half = dim / 2
        let exponent = MLXArray(0 ..< Int32(half)).asType(.float32) * Float(-log(10000.0) / Double(half))
        let freqs = MLX.exp(exponent)
        let emb = t.asType(.float32).expandedDimensions(axis: -1) * freqs.reshaped(1, half)
        let both = MLX.concatenated([MLX.sin(emb), MLX.cos(emb)], axis: -1)
        return MLX.concatenated([both[0..., half...], both[0..., ..<half]], axis: -1)
    }

    // MARK: - Stages

    /// (B,C,T,H,W) latent de-normalization (channels-first).
    func unNormalize(_ z: MLXArray) -> MLXArray {
        let m = weight("per_channel_statistics.mean").reshaped(1, -1, 1, 1, 1)
        let s = weight("per_channel_statistics.std").reshaped(1, -1, 1, 1, 1)
        return z * s + m
    }

    /// Latent → stage-4 context volume (channels-last), pre-crop.
    public func contextVolume(_ latent: MLXArray, tile: (Int, Int, Int) = (2, 32, 96)) -> MLXArray {
        let prof = MLXProfiler.shared
        var x = unNormalize(latent).transposed(0, 2, 3, 4, 1)
        x = dense(x, "conv_in")
        for i in 0 ..< 4 {
            let span = prof.begin("diffvae-decode", "det-stage\(i)", note: "in=\(x.shape)")
            for d in 0 ..< cfg.stageDepths[i] {
                x = naBlock(x, "det_stages.\(i).\(d)", kernel: cfg.stageKernels[i], tile: tile)
                eval(x)
            }
            x = upsample(x, "upsamples.\(i)", stride: cfg.upsampleStrides[i],
                         reduction: cfg.upsampleReductions[i], dropLeadingFrame: true)
            eval(x)
            prof.end(span)
        }
        return x
    }

    /// The noised pixel canvas patchified and projected into stage-5's channel space.
    ///
    /// Split out of ``diffStep(context:xT:t:tile:)`` so a gate can drive a single block with
    /// exactly the tensor the stack would have handed it.
    public func stage5Input(_ xT: MLXArray) -> MLXArray {
        dense(DiffusionVideoDecoder.patchifyHW(xT, patch: cfg.patchSize).transposed(0, 2, 3, 4, 1),
              "conv_in_x_t")
    }

    /// Stage 5: noised pixels + context → pixels (channels-first, [-1, 1]).
    public func diffStep(context: MLXArray, xT: MLXArray, t: MLXArray,
                         tile: (Int, Int, Int) = (2, 32, 96)) -> MLXArray {
        let prof = MLXProfiler.shared
        var x = stage5Input(xT)

        let shared = sharedAdaLN(t)

        for i in 0 ..< cfg.stageDepths[4] {
            let span = prof.begin("diffvae-decode", "diff-block\(i)", note: "in=\(x.shape)")
            x = diffusionNABlock(x, "diff_blocks.\(i)", context: context, shared: shared,
                                 kernel: cfg.stage5Kernel, tile: tile)
            eval(x)
            prof.end(span)
        }

        x = dense(rms(x, "norm_out"), "conv_out").transposed(0, 4, 1, 2, 3)
        return DiffusionVideoDecoder.unpatchifyHW(x, patch: cfg.patchSize)
    }

    /// The 7-way AdaLN split for timestep `t`, each chunk broadcast-shaped (B,1,1,1,C).
    public func sharedAdaLN(_ t: MLXArray) -> [MLXArray] {
        let embedded = DiffusionVideoDecoder.timestepEmbedding(t * cfg.timestepScaleMultiplier)
        let tEmb = dense(silu(dense(embedded, "t_embedder.timestep_embedder.linear1")),
                         "t_embedder.timestep_embedder.linear2")
        let h = dense(silu(tEmb), "shared_adaln.proj")
        let d = h.dim(-1) / 7
        return (0 ..< 7).map { h[0..., ($0 * d) ..< (($0 + 1) * d)].reshaped(-1, 1, 1, 1, d) }
    }

    // MARK: - Padding / cropping helpers

    /// Pad or crop `axis` to `size`. Port of upstream `diffusion_tiling.resize_axis`.
    ///
    /// `repeat_last` appends copies of the last slice (and crops from the end); `symmetric`
    /// edge-replicates both ends with `before = need / 2`, remainder to the end.
    static func resizeAxis(_ x: MLXArray, axis: Int, size: Int, mode: String) -> MLXArray {
        let length = x.dim(axis)
        if length == size { return x }
        var idx = Array(0 ..< length)
        if length < size {
            let need = size - length
            if mode == "repeat_last" {
                idx += Array(repeating: length - 1, count: need)
            } else {
                let before = need / 2, after = need - before
                idx = Array(repeating: 0, count: before) + idx + Array(repeating: length - 1, count: after)
            }
        } else {
            if mode == "repeat_last" {
                idx = Array(idx[0 ..< size])
            } else {
                let before = (length - size) / 2
                idx = Array(idx[before ..< (before + size)])
            }
        }
        return MLX.take(x, MLXArray(idx.map { Int32($0) }), axis: axis)
    }

    /// Pad (B,C,T,H,W) up to `stageMinTileSizes` = (3, 7, 7).
    func ensureMinLatentShape(_ latent: MLXArray) -> MLXArray {
        let (minT, minH, minW) = cfg.stageMinTileSizes
        var x = latent
        if x.dim(2) < minT { x = DiffusionVideoDecoder.resizeAxis(x, axis: 2, size: minT, mode: "repeat_last") }
        if x.dim(3) < minH { x = DiffusionVideoDecoder.resizeAxis(x, axis: 3, size: minH, mode: "symmetric") }
        if x.dim(4) < minW { x = DiffusionVideoDecoder.resizeAxis(x, axis: 4, size: minW, mode: "symmetric") }
        return x
    }

    // MARK: - Decode

    /// The stage-1-to-4 ladder with its ghost pad and crop — the deterministic half of a decode.
    ///
    /// Split out so the gate can check it independently of the stochastic stage 5.
    public func decodeContext(_ latent: MLXArray, tile: (Int, Int, Int) = (2, 32, 96)) -> MLXArray {
        var x = ensureMinLatentShape(latent)
        // NATTEN last-frame border workaround: replicate the final latent frame.
        x = DiffusionVideoDecoder.resizeAxis(x, axis: 2, size: x.dim(2) + cfg.nattenTrailingPad,
                                             mode: "repeat_last")
        var context = contextVolume(x, tile: tile)
        // crop the ghost appendix, keeping at least `stage5Kernel.0` frames
        let ghost = cfg.nattenTrailingPad * cfg.temporalScale
        let contentT = max(context.dim(1) - ghost, 1)
        let keep = min(context.dim(1), max(contentT, cfg.stage5Kernel.0))
        context = DiffusionVideoDecoder.resizeAxis(context, axis: 1, size: keep, mode: "repeat_last")
        eval(context)
        return context
    }

    /// Pixel shape stage 5 will produce for a given context volume.
    public func pixelShape(forContext context: MLXArray) -> [Int] {
        [context.dim(0), cfg.outChannels, context.dim(1),
         context.dim(2) * cfg.patchSize, context.dim(3) * cfg.patchSize]
    }

    /// Full 1-step x0 decode: latent (B,C,T,H,W) → pixels (B,3,8·(T−1)+1,H·32,W·32) in **[-1, 1]**.
    ///
    /// With `num_inference_steps = 1` and `model_output_type = "x0"` the reference
    /// short-circuits: no Euler step, no velocity transform — the model output IS the pixels,
    /// evaluated at t = 1.0 from a pure-noise canvas.
    ///
    /// ⚠️ THIS DECODER IS STOCHASTIC. The noise canvas is a real input, not an implementation
    /// detail: two decodes of the same latent with different draws give different pixels.
    /// Callers that need reproducibility must pass `noise` explicitly (or a fixed `key`), and
    /// no gate may assert pixel equality against a differently-drawn reference — MLX-Swift and
    /// MLX-Python RNG streams are not bit-identical (determinism doctrine, ISSUES I1). The
    /// parity gate injects the golden's own canvas for exactly this reason.
    public func decode(_ latent: MLXArray,
                       noise: MLXArray? = nil,
                       tile: (Int, Int, Int) = (2, 32, 96),
                       key: MLXArray? = nil) -> MLXArray {
        precondition(cfg.modelOutputType == "x0",
                     "only the 1-step x0 path is ported; got modelOutputType=\(cfg.modelOutputType)")
        // Remember the true content extent: `ensureMinLatentShape` may pad, and the reference
        // narrows back to content before emitting.
        let tIn = latent.dim(2), hIn = latent.dim(3), wIn = latent.dim(4)
        let padH = max(0, cfg.stageMinTileSizes.1 - hIn) / 2
        let padW = max(0, cfg.stageMinTileSizes.2 - wIn) / 2

        let context = decodeContext(latent, tile: tile)
        let pix = pixelShape(forContext: context)

        let canvas: MLXArray
        if let noise {
            precondition(noise.shape == pix, "noise \(noise.shape) != expected \(pix)")
            canvas = noise
        } else {
            canvas = MLXRandom.normal(pix, key: key)
        }

        let t = MLXArray.ones([context.dim(0)]).asType(.float32)
        let pixels = diffStep(context: context, xT: canvas, t: t, tile: tile)

        // Narrow to the requested content: T padding is repeat_last (content at the front),
        // H/W padding is symmetric (content offset by need/2).
        let wantF = min(pixels.dim(2), 8 * (tIn - 1) + 1)
        let ph = padH * 32, pw = padW * 32
        return pixels[0..., 0..., ..<wantF, ph ..< (ph + hIn * 32), pw ..< (pw + wIn * 32)]
    }

    /// ``decode(_:noise:tile:key:)`` mapped to [0, 1] — the reference's `decode_video` tail.
    public func decodeToRGB(_ latent: MLXArray,
                            noise: MLXArray? = nil,
                            tile: (Int, Int, Int) = (2, 32, 96),
                            key: MLXArray? = nil) -> MLXArray {
        MLX.clip((decode(latent, noise: noise, tile: tile, key: key) + 1.0) * 0.5, min: 0.0, max: 1.0)
    }
}
