// RoPE.swift — Rotary Position Embeddings, SPLIT type.
//
// 1:1 port of ltx_core_mlx/model/transformer/rope.py (SPLIT path only — the
// production checkpoints all use SPLIT; INTERLEAVED is legacy). The reference
// uses a LOG-SPACED frequency grid with fractional positions, NOT standard
// 1/theta^k RoPE. Shared by the text connector (1D positions) and the main
// DiT (3D video positions) — kept isomorphic with the oracle.

import Foundation
import MLX

public enum RoPE {

    /// generate_freq_grid: log-spaced frequency indices.
    /// Returns shape (innerDim / (2 * numPosDims),).
    static func freqGrid(theta: Float, numPosDims: Int, innerDim: Int) -> MLXArray {
        let nElem = 2 * numPosDims
        let numFreqs = innerDim / nElem
        // linspace(log(1)/log(theta), log(theta)/log(theta), numFreqs) == linspace(0, 1, numFreqs)
        let denom = Float(max(numFreqs - 1, 1))
        let lin = MLXArray(0 ..< numFreqs).asType(.float32) / denom  // [0, 1]
        let indices = MLX.pow(MLXArray(theta), lin)                  // theta ** lin
        return indices * (Float.pi / 2.0)
    }

    /// compute_freqs: angles from positions + freq grid.
    /// positions: (B, N, numPosDims) ; returns (B, N, numFreqs * numPosDims).
    static func computeFreqs(freqIndices: MLXArray, positions: MLXArray, maxPos: [Int]) -> MLXArray {
        let numPosDims = positions.dim(-1)
        let B = positions.dim(0)
        let N = positions.dim(1)

        // fractional positions: pos[...,i] / maxPos[i]  -> stack on last axis
        var fracCols: [MLXArray] = []
        for i in 0 ..< numPosDims {
            let col = positions[0..., 0..., i].asType(.float32) / Float(maxPos[i])
            fracCols.append(col)
        }
        let frac = MLX.stacked(fracCols, axis: -1)  // (B, N, numPosDims)

        // scaled = freqIndices * (frac[...,None]*2 - 1)  -> (B, N, numPosDims, numFreqs)
        let scaled = freqIndices * (frac.expandedDimensions(axis: -1) * 2.0 - 1.0)

        // transpose last two dims, flatten: (B, N, numFreqs, numPosDims) -> (B, N, -1)
        let freqs = scaled.transposed(0, 1, 3, 2).reshaped(B, N, -1)
        return freqs
    }

    /// precompute_rope_freqs (SPLIT). Returns (cos, sin), each (B, numHeads, N, headDim/2).
    static func precomputeSplit(
        positions: MLXArray,
        innerDim: Int,
        numHeads: Int,
        theta: Float = 10000.0,
        maxPos: [Int]
    ) -> (cos: MLXArray, sin: MLXArray) {
        let numPosDims = positions.dim(-1)
        let freqIndices = freqGrid(theta: theta, numPosDims: numPosDims, innerDim: innerDim)
        var freqs = computeFreqs(freqIndices: freqIndices, positions: positions, maxPos: maxPos)
        let B = freqs.dim(0)
        let N = freqs.dim(1)
        let numFreqs = freqs.dim(2)

        let expected = innerDim / 2
        let padSize = expected - numFreqs
        if padSize > 0 {
            let padding = MLXArray.zeros([B, N, padSize])
            freqs = MLX.concatenated([padding, freqs], axis: -1)  // pad at FRONT
        }

        let cosF = MLX.cos(freqs)
        let sinF = MLX.sin(freqs)
        let headDimHalf = innerDim / (2 * numHeads)
        let cosR = cosF.reshaped(B, N, numHeads, headDimHalf).transposed(0, 2, 1, 3)
        let sinR = sinF.reshaped(B, N, numHeads, headDimHalf).transposed(0, 2, 1, 3)
        return (cosR, sinR)
    }

    /// apply_split_rotary_emb. x: (B, H, N, headDim) ; cos/sin: (B, H, N, headDim/2).
    static func applySplit(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let cosF = cos.asType(x.dtype)
        let sinF = sin.asType(x.dtype)
        let half = x.dim(-1) / 2
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half ..< (2 * half)]
        return MLX.concatenated([x1 * cosF - x2 * sinF, x1 * sinF + x2 * cosF], axis: -1)
    }
}
