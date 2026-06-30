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

/// Sidecar LoRA factors for a resident `DiT`, keyed on the `dense()` prefix
/// (e.g. `transformer_blocks.0.attn1.to_q`). `a` = [in, rank]; `b` = [rank, out] with the adapter's
/// scale already baked in. Reference type so apply/detach mutate it without mutating the `DiT` struct.
public final class LoRAStore {
    public var adapters: [String: (a: MLXArray, b: MLXArray)] = [:]
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
    public static func apply(_ loras: [(url: URL, strength: Float)], to dit: DiT) throws {
        guard !loras.isEmpty else { dit.lora.clear(); return }
        let combo = try combined(loras, dtype: dit.dtype)
        var kept: [String: (a: MLXArray, b: MLXArray)] = [:]
        var skipped = 0
        for (prefix, ab) in combo {
            if dit.hasDenseWeight(prefix) { kept[prefix] = ab } else { skipped += 1 }
        }
        guard !kept.isEmpty else { throw LoRAError.noTargets("\(loras.map(\.url.lastPathComponent))") }
        if skipped > 0 {
            FileHandle.standardError.write(Data(
                "[LTX2LoRA] applied \(kept.count) targets, skipped \(skipped) unmatched\n".utf8))
        }
        dit.lora.adapters = kept
    }

    public static func apply(_ url: URL, strength: Float = 1.0, to dit: DiT) throws {
        try apply([(url, strength)], to: dit)
    }

    /// Restore the pristine base.
    public static func detach(_ dit: DiT) { dit.lora.clear() }
}
