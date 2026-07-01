import Foundation
import MLXToolKit

/// Memory-tier profile (LOW-TIER-PLAN T3): an envelope clamp + path policy + VAE decode window that
/// makes LTX-2.3 honestly declarable per tier. Activation is seqLen-scaled, so a profile that CLAMPS
/// the envelope can declare a far smaller `peakActivationBytesHint` than the 704×512 default —
/// requests beyond the envelope are clamped (not rejected). The recommended quant per tier is
/// advisory (`recommendedQuant`); the registered `quant` still decides the checkpoint.
/// 16 GB is deliberately ABSENT: the int4 DiT alone (11.3 GB) ≈ a 16 GB governor budget, and no
/// smaller LTX-2 checkpoint exists.
public enum LTX2Profile: String, Codable, Sendable, CaseIterable {
    /// 24 GB Macs (M5 MacBook Pro default): int4, one-stage, small envelope, tight decode window.
    case compact24
    /// 32 GB Macs: int4/int8, one-stage, mid envelope.
    case balanced32
    /// 64 GB Macs: int8, full two-stage at the standard envelope.
    case standard64
    /// 96–128 GB Macs: bf16 two-stage, long clips (240f proven at ~92 GB with chunked decode).
    case max128

    public var maxWidth: Int  { switch self { case .compact24: 512; case .balanced32: 576; default: 704 } }
    public var maxHeight: Int { switch self { case .compact24: 288; case .balanced32: 320; default: 512 } }
    public var maxFrames: Int {
        switch self { case .compact24: 121; case .balanced32: 161; case .standard64: 161; case .max128: 241 }
    }
    /// One-stage skips the spatial upsampler + full-res stage-2 refine — the low-tier denoise path.
    public var oneStage: Bool { self == .compact24 || self == .balanced32 }
    /// VAE temporal-decode window knob (see `LTX2Pipeline.decodePixels`; halo is fixed at 5).
    public var vaeChunkFrames: Int {
        switch self { case .compact24: 4; case .balanced32: 6; default: 8 }
    }
    public var recommendedQuant: Quant {
        switch self { case .compact24, .balanced32: .int4; case .standard64: .int8; case .max128: .bf16 }
    }
}

/// Init-time configuration for `MLXLTX2Package` (C9): where the LTX-2.3 component
/// weights and the Gemma-3 text encoder live. Per-request prompt/size/steps ride
/// the canonical `T2VRequest`, not here.
///
/// `ltxDirectory` holds the LTX safetensors (connector / transformer-distilled /
/// vae_decoder). `gemmaDirectory` is the Gemma-3 MLX weights dir (mlx-community/
/// gemma-3-12b-it-4bit). Both are environment-specific → excluded from Codable.
public struct LTX2Configuration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// Provenance repo id (the LTX-2.3 MLX collection).
    public var repo: String
    public var revision: String?
    /// Backbone quant of the distilled transformer. `QuantConfigured` surfaces it to the engine's
    /// `MemoryGovernor` so it charges the *registered* variant's `QuantFootprint` (bf16/int8/int4)
    /// instead of the bf16 max (engine ≥0.9.1; closes the q8/q4 over-reservation, LTX ENHANCEMENTS E14).
    public var quant: Quant
    /// Resolved LTX component directory (connector/vae_decoder/vae_encoder/audio_vae/
    /// vocoder/upsampler — these stay bf16 across quant variants).
    public var ltxDirectory: URL?
    /// Optional override for the DiT transformer file. Defaults to
    /// `ltxDirectory/transformer-distilled.safetensors` (bf16). Point at a quantized
    /// checkpoint (e.g. `.../ltx-2.3-mlx-q8/transformer-distilled.safetensors`) to run
    /// int8/int4 — the loader auto-detects quantization from the weights (scales/biases).
    /// Only the transformer is quantized; everything else loads from `ltxDirectory`.
    public var transformerPath: URL?
    /// Resolved Gemma-3 text-encoder directory.
    public var gemmaDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// Memory-tier profile (nil = unconstrained legacy behavior — no clamp, two-stage preferred,
    /// footprint falls back to the per-quant `QuantFootprint` measured at 704×512).
    public var profile: LTX2Profile?

    public init(
        repo: String = "dgrauet/ltx-2.3-mlx",
        revision: String? = nil,
        quant: Quant = .bf16,
        ltxDirectory: URL? = nil,
        transformerPath: URL? = nil,
        gemmaDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil,
        profile: LTX2Profile? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.ltxDirectory = ltxDirectory
        self.transformerPath = transformerPath
        self.gemmaDirectory = gemmaDirectory
        self.modelsRootDirectory = modelsRootDirectory
        self.profile = profile
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant, profile
    }
}

/// Per-profile transient hint (contract 1.14 `FootprintConfigured`): the activation peak is
/// seqLen-scaled, so a clamped profile declares its OWN transient instead of the 704×512 default.
/// `residentBytesHint` stays nil — the per-quant `QuantFootprint.residentBytes` (the DiT floor) is
/// envelope-independent. Values are max-over-phase at the profile's envelope; measured entries are
/// marked, others are estimates pending the T3 autorun re-baseline (LOW-TIER-PLAN).
extension LTX2Configuration: FootprintConfigured {
    public var residentBytesHint: UInt64? { nil }

    public var peakActivationBytesHint: UInt64? {
        guard let profile else { return nil }   // fall back to the per-quant QuantFootprint
        // MEASURED 2026-07-01 (T3 autorun, clamp exercised; see LOW-TIER-PLAN T3 RESULTS). These are
        // the HONEST current numbers — compact24/balanced32 do NOT yet fit their nominal tier
        // budgets (encode stage is the peak: Gemma+connector co-resident + hoisted fp32 projection
        // views; standard64's is the decode window's MLX-pool growth). T3b levers (sequential
        // Gemma→connector, connector int8 quantizedMatmul, scoped cacheLimit during decode) tighten
        // these toward the 0.7× budgets; re-baseline on each landing.
        switch profile {
        case .compact24:  return 25_000_000_000   // act 23.9 measured (peak 35.9 = encode-bound)
        case .balanced32: return 25_000_000_000   // act 23.5 measured (encode-bound, same stage)
        case .standard64: return 41_000_000_000   // act 39.1 measured (decode-window/cache-bound)
        case .max128:     return 50_000_000_000   // act 47.9 measured (240f i2v bf16, chunked)
        }
    }
}

/// Cold-start weight prewarm (engine ≥0.7.0): page the LTX + Gemma weight files into the OS
/// file cache before `load()` runs its GPU evals, so the cold load-time `eval` never faults
/// weights off slow/external storage inside a live Metal command buffer (the I5 cold-load
/// `kIOGPUCommandBufferCallbackErrorTimeout`). Only the config knows these resolved `/Volumes`
/// paths — execution is the engine's (`WeightPrewarmer`, best-effort).
///
/// **Quant-aware exclusion.** When `transformerPath` (a quant override, e.g. the q8 transformer)
/// is set, the default bf16 `transformer-distilled.safetensors` inside `ltxDirectory` is NEVER
/// loaded — so paging the whole dir would read ~35 GB of cold cost for nothing. In that case we
/// page `ltxDirectory`'s weight files *individually, minus that bf16 transformer*, plus the
/// override. (bf16 path keeps the simple whole-dir prewarm.)
extension LTX2Configuration: WeightPrewarming {
    /// Basename of the bf16 transformer that a `transformerPath` override replaces.
    static let defaultTransformerFile = "transformer-distilled.safetensors"

    public var prewarmPaths: [URL] {
        guard let ltxDir = ltxDirectory else {
            return [gemmaDirectory, transformerPath].compactMap { $0 }
        }
        // bf16 (no override): whole LTX dir + Gemma — the bf16 transformer in ltxDir IS used.
        guard let txPath = transformerPath else {
            return [ltxDir, gemmaDirectory].compactMap { $0 }
        }
        // Quant override active: page ltxDir's weight files except the unused bf16 transformer.
        let bf16 = ltxDir.appendingPathComponent(Self.defaultTransformerFile).standardizedFileURL
        let ltxWeights = ((try? FileManager.default.contentsOfDirectory(
            at: ltxDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" && $0.standardizedFileURL != bf16 }
        // Correctness-first fallback: if enumeration turned up nothing, page the whole dir.
        var paths = ltxWeights.isEmpty ? [ltxDir] : ltxWeights
        paths.append(txPath)
        if let gemma = gemmaDirectory { paths.append(gemma) }
        return paths
    }
}
