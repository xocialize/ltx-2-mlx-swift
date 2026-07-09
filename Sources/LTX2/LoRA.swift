// Runtime LoRA for the LTX-2.3 joint-AV DiT — the "extend, not swap" path.
//
// LTX's `DiT` is a `struct` with a flat weight dict and a functional `dense(x, key)`; there are no
// `Linear` modules to replace (unlike the Qwen-Image DiT). So a LoRA is applied by registering its
// low-rank factors in a sidecar `LoRAStore` keyed on the `dense()` prefix, and `dense` adds
// `(x · A) · B` in the activation path. Hot-swap = repopulate the store; nothing in the 22B base
// is duplicated or reloaded, and the add survives bf16/q8/q4 (the factors stay full precision).
//
// Dialect (verified against joyfox/LTX-2.3-Transition-LORA via the official Lightricks/LTX-2
// trainer): diffusers-PEFT — `diffusion_model.transformer_blocks.<i>.<module>.lora_{A,B}.weight`,
// rank 32, usually alpha-less (scale = strength). The kohya/comfy suffixes are handled too for
// community files. `lora_A` is [rank, in]; `lora_B` is [out, rank].

import Foundation
import MLX
import MLXRandom

/// One target's runtime factors — full-precision, or int8/int4-packed (BRIDGE-LTX-012: the
/// rank-256 i2v adapter is 4.93 GB bf16, the single biggest term in the low-tier i2v peak;
/// group-64 packing halves/quarters its residency, dispatched via `quantizedMatmul` in `dense`).
public enum LoRAFactors {
    /// `a` = [in, rank]; `b` = [rank, out] with the adapter's scale already baked in.
    case plain(a: MLXArray, b: MLXArray)
    /// Packed transposed factors for `quantizedMatmul(transpose: true)`:
    /// `aw` packs Aᵀ = [rank, in]; `bw` packs Bᵀ = [out, rank]. Group size 64.
    case quantized(aw: MLXArray, aScales: MLXArray, aBiases: MLXArray?,
                   bw: MLXArray, bScales: MLXArray, bBiases: MLXArray?, bits: Int)
}

/// Sidecar LoRA factors for a resident `DiT`, keyed on the `dense()` prefix
/// (e.g. `transformer_blocks.0.attn1.to_q`). Reference type so apply/detach mutate it without
/// mutating the `DiT` struct.
public final class LoRAStore {
    public var adapters: [String: LoRAFactors] = [:]
    public init() {}
    /// Restore the pristine base (drop all adapters).
    public func clear() { adapters.removeAll() }
    public var isEmpty: Bool { adapters.isEmpty }
}

public enum LTX2LoRA {

    public enum LoRAError: Error, LocalizedError {
        case incompletePair(String)
        case noTargets(String)
        public var errorDescription: String? {
            switch self {
            case .incompletePair(let p): return "LoRA layer \(p) is missing its lora_A or lora_B tensor."
            case .noTargets(let url): return "No recognizable LTX-2.3 LoRA tensors found in \(url)."
            }
        }
    }

    // A/B factor suffixes across LoRA dialects (down == A = [rank,in]; up == B = [out,rank]).
    private static let aSuffixes = [
        ".lora_A.default.weight", ".lora_A.weight", ".lora.down.weight", ".lora_down.weight",
    ]
    private static let bSuffixes = [
        ".lora_B.default.weight", ".lora_B.weight", ".lora.up.weight", ".lora_up.weight",
    ]
    private static let alphaSuffix = ".alpha"

    /// diffusers/comfy LoRA child path → this port's `dense()` prefix. Strips the `diffusion_model.` /
    /// `transformer.` top-level prefixes, then normalizes the LTX module names to the port's keys.
    /// (Mirrors the trainer's diffusers naming → DiT.swift `dense` prefixes.)
    static func remap(_ path: String) -> String {
        var s = path
        if s.hasPrefix("diffusion_model.") { s.removeFirst("diffusion_model.".count) }
        if s.hasPrefix("transformer.") { s.removeFirst("transformer.".count) }
        return s
            // attention output: trainer `to_out.0` → port single `to_out`
            .replacingOccurrences(of: ".to_out.0", with: ".to_out")
            // feed-forward: trainer `ff.net.0.proj`/`ff.net.2` → port `ff.proj_in`/`ff.proj_out`
            // (the same replacement covers `audio_ff.net.*` since it matches the `.net.*` tail).
            .replacingOccurrences(of: ".net.0.proj", with: ".proj_in")
            .replacingOccurrences(of: ".net.2", with: ".proj_out")
    }

    /// One LoRA's per-target low-rank factors, keyed by `dense()` prefix. `a` = [in, rank];
    /// `b` = [rank, out] with the effective scale baked in.
    struct Factors { var a: MLXArray; var b: MLXArray }

    static func factors(from url: URL, dtype: DType, strength: Float) throws -> [String: Factors] {
        let raw = try MLX.loadArrays(url: url)
        func match(_ key: String, _ suffixes: [String]) -> String? {
            for s in suffixes where key.hasSuffix(s) { return remap(String(key.dropLast(s.count))) }
            return nil
        }
        var aMats: [String: MLXArray] = [:]
        var bMats: [String: MLXArray] = [:]
        var alphas: [String: MLXArray] = [:]
        for (key, value) in raw {
            if let base = match(key, aSuffixes) { aMats[base] = value }
            else if let base = match(key, bSuffixes) { bMats[base] = value }
            else if key.hasSuffix(alphaSuffix) {
                alphas[remap(String(key.dropLast(alphaSuffix.count)))] = value
            }
        }
        guard !aMats.isEmpty else { throw LoRAError.noTargets(url.path) }

        var out: [String: Factors] = [:]
        for (base, aMat) in aMats {
            guard let bMat = bMats[base] else { throw LoRAError.incompletePair(base) }
            let rank = aMat.dim(0)  // diffusers lora_A is [rank, in]
            // alpha/rank when present; alpha-less adapters apply at scale = strength.
            let scale = strength * (alphas[base].map { $0.item(Float.self) / Float(rank) } ?? 1.0)
            out[base] = Factors(a: aMat.T.asType(dtype), b: (scale * bMat.T).asType(dtype))
        }
        return out
    }

    /// Combine one or more LoRAs into per-prefix `(a, b)` by rank-stacking (concat `a` on rank axis 1,
    /// `b` on rank axis 0), so the added term is the exact SUM of each adapter's low-rank contribution.
    static func combined(_ loras: [(url: URL, strength: Float)], dtype: DType)
        throws -> [String: (a: MLXArray, b: MLXArray)]
    {
        let perLoRA = try loras.map { try factors(from: $0.url, dtype: dtype, strength: $0.strength) }
        var bases = Set<String>()
        perLoRA.forEach { bases.formUnion($0.keys) }
        var out: [String: (a: MLXArray, b: MLXArray)] = [:]
        for base in bases {
            let present = perLoRA.compactMap { $0[base] }
            let aCat = present.count == 1 ? present[0].a : concatenated(present.map(\.a), axis: 1)
            let bCat = present.count == 1 ? present[0].b : concatenated(present.map(\.b), axis: 0)
            out[base] = (aCat, bCat)
        }
        return out
    }

    /// Make `loras` the active adapter set on `dit` (replaces any current set). Empty → detach.
    /// Only targets whose `dense()` weight actually exists are kept (keys-present; a remap that
    /// doesn't resolve to a real Linear is skipped rather than silently wrong).
    /// Make `loras` the active adapter set on `dit`. `factorQuantBits` (8 or 4) packs the factors
    /// group-64 at apply time — the low-tier residency lever (BRIDGE-LTX-012): the i2v adapter is
    /// 4.93 GB bf16 → ~2.5 GB int8 / ~1.3 GB int4, riding the DiT through the denoise peak.
    /// Quantization is per-target with an `eval` per pair, so the bf16 source pages fault in and
    /// free tensor-by-tensor (mmap) — the apply transient stays factor-sized, not adapter-sized.
    public static func apply(_ loras: [(url: URL, strength: Float)], to dit: DiT,
                             factorQuantBits: Int? = nil) throws {
        guard !loras.isEmpty else { dit.lora.clear(); return }
        let combo = try combined(loras, dtype: dit.dtype)
        var kept: [String: LoRAFactors] = [:]
        var skipped = 0
        for (prefix, ab) in combo {
            guard dit.hasDenseWeight(prefix) else { skipped += 1; continue }
            if let bits = factorQuantBits, bits == 8 || bits == 4 {
                // Pack the TRANSPOSED factors so dense() can use quantizedMatmul(transpose: true):
                // x·A = x @ (Aᵀ)ᵀ with Aᵀ = [rank, in]; h·B = h @ (Bᵀ)ᵀ with Bᵀ = [out, rank].
                let (aw, aS, aB) = MLX.quantized(ab.a.T, groupSize: 64, bits: bits)
                let (bw, bS, bB) = MLX.quantized(ab.b.T, groupSize: 64, bits: bits)
                var toEval = [aw, aS, bw, bS]
                if let aB { toEval.append(aB) }
                if let bB { toEval.append(bB) }
                eval(toEval)   // materialize + release this pair's bf16 source
                kept[prefix] = .quantized(aw: aw, aScales: aS, aBiases: aB,
                                          bw: bw, bScales: bS, bBiases: bB, bits: bits)
            } else {
                kept[prefix] = .plain(a: ab.a, b: ab.b)
            }
        }
        guard !kept.isEmpty else { throw LoRAError.noTargets("\(loras.map(\.url.lastPathComponent))") }
        if skipped > 0 {
            FileHandle.standardError.write(Data(
                "[LTX2LoRA] applied \(kept.count) targets, skipped \(skipped) unmatched\n".utf8))
        }
        if factorQuantBits != nil { Memory.clearCache() }   // drop the bf16 staging pool
        dit.lora.adapters = kept
    }

    public static func apply(_ url: URL, strength: Float = 1.0, to dit: DiT) throws {
        try apply([(url, strength)], to: dit)
    }

    /// Fidelity probe (BRIDGE-LTX-012, drives `RunLTX2 --lora-quant-gate`): cosine between the
    /// full-precision LoRA delta `(x·A)·B` and its group-64 packed dispatch, across the first
    /// `sample` targets of `url` on random bf16 activations. Returns (worst, mean) cosine.
    public static func factorQuantFidelity(url: URL, bits: Int, sample: Int = 32,
                                           strength: Float = 1.0) throws -> (worst: Float, mean: Float) {
        let all = try factors(from: url, dtype: .bfloat16, strength: strength)
        let picks = all.keys.sorted().prefix(max(1, sample))
        var worst: Float = 1, sum: Float = 0
        for key in picks {
            let f = all[key]!
            let x = MLXRandom.normal([1, 64, f.a.dim(0)]).asType(.bfloat16)
            let plain = x.matmul(f.a).matmul(f.b).asType(.float32).reshaped(-1)
            let (aw, aS, aB) = MLX.quantized(f.a.T, groupSize: 64, bits: bits)
            let (bw, bS, bB) = MLX.quantized(f.b.T, groupSize: 64, bits: bits)
            let h = MLX.quantizedMatmul(x, aw, scales: aS, biases: aB,
                                        transpose: true, groupSize: 64, bits: bits)
            let packed = MLX.quantizedMatmul(h, bw, scales: bS, biases: bB,
                                             transpose: true, groupSize: 64, bits: bits)
                .asType(.float32).reshaped(-1)
            let dot = (plain * packed).sum().item(Float.self)
            let n = (MLX.sqrt((plain * plain).sum()) * MLX.sqrt((packed * packed).sum())).item(Float.self)
            let c = n > 0 ? dot / n : 1
            worst = min(worst, c); sum += c
        }
        return (worst, sum / Float(picks.count))
    }

    /// Restore the pristine base.
    public static func detach(_ dit: DiT) { dit.lora.clear() }
}
