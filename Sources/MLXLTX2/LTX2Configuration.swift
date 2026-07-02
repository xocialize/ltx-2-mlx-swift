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
    /// 96–128 GB Macs: bf16 two-stage, long clips. 480f bf16 t2v MEASURED 67.61 GB post-T3b
    /// (BRIDGE-LTX-005; floor ~40.5 GB + ~40 MB/frame activation), so the envelope admits 481f.
    /// i2v at the cap MEASURED too (--i2v-spot 2026-07-01): 481f + 4.9 GB adapter peaks 72.73 GB.
    case max128

    public var maxWidth: Int  { switch self { case .compact24: 512; case .balanced32: 576; default: 704 } }
    public var maxHeight: Int { switch self { case .compact24: 288; case .balanced32: 320; default: 512 } }
    public var maxFrames: Int {
        // max128 241→481 (BRIDGE-LTX-005): 704×512×480f bf16 t2v measured 67.61 GB — the old cap
        // left ~60 GB of a 128 GB budget unused. 481 sits ON a measured point (8n+1 frame grid).
        switch self { case .compact24: 121; case .balanced32: 161; case .standard64: 161; case .max128: 481 }
    }
    /// One-stage skips the spatial upsampler + full-res stage-2 refine — the low-tier denoise path.
    public var oneStage: Bool { self == .compact24 || self == .balanced32 }
    /// VAE temporal-decode window knob (see `LTX2Pipeline.decodePixels`; halo is fixed at 5).
    public var vaeChunkFrames: Int {
        switch self { case .compact24: 4; case .balanced32: 6; default: 8 }
    }
    /// Low tiers evict the DiT after the last denoise step so the decode stage never carries it
    /// (T3c — decode-with-DiT-resident was the residual low-tier peak). Costs a DiT reload on the
    /// next request (mmap re-fault; kernels stay process-cached), accepted on these tiers.
    public var evictDiTBeforeDecode: Bool { self == .compact24 || self == .balanced32 }
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
    /// Text-encoder repo (Gemma-3), materialized when `gemmaDirectory` is nil (BRIDGE M4).
    public var gemmaRepo: String
    /// Override for the quantized-transformer repo; nil derives `<repo>-q8`/`-q4` from `quant`
    /// (bf16 rides the components repo — no separate source).
    public var transformerRepo: String?
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
        gemmaRepo: String = "mlx-community/gemma-3-12b-it-4bit",
        transformerRepo: String? = nil,
        quant: Quant = .bf16,
        ltxDirectory: URL? = nil,
        transformerPath: URL? = nil,
        gemmaDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil,
        profile: LTX2Profile? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.gemmaRepo = gemmaRepo
        self.transformerRepo = transformerRepo
        self.quant = quant
        self.ltxDirectory = ltxDirectory
        self.transformerPath = transformerPath
        self.gemmaDirectory = gemmaDirectory
        self.modelsRootDirectory = modelsRootDirectory
        self.profile = profile
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, gemmaRepo, transformerRepo, quant, profile
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        gemmaRepo = try c.decodeIfPresent(String.self, forKey: .gemmaRepo)
            ?? "mlx-community/gemma-3-12b-it-4bit"
        transformerRepo = try c.decodeIfPresent(String.self, forKey: .transformerRepo)
        quant = try c.decode(Quant.self, forKey: .quant)
        profile = try c.decodeIfPresent(LTX2Profile.self, forKey: .profile)
    }
}

// MARK: - Weight sources (auto-materialization, BRIDGE M4 / engine MAT gate)

extension LTX2Configuration: WeightSourcing {
    /// The component files LTX-2.3 needs beyond the transformer (optional features included —
    /// a first run materializes the full experience; the pipeline degrades per missing file
    /// only for explicit-dir setups).
    static let componentFiles = [
        "connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
        "audio_vae.safetensors", "vocoder.safetensors", "spatial_upscaler_x2_v1_1.safetensors",
    ]

    /// The repo serving the quantized transformer; bf16 rides the components repo.
    public var effectiveTransformerRepo: String? {
        if let transformerRepo { return transformerRepo }
        switch quant {
        case .int8: return repo + "-q8"
        case .int4: return repo + "-q4"
        default: return nil
        }
    }

    public var weightSources: [WeightSource] {
        var componentGlobs = Self.componentFiles
        if effectiveTransformerRepo == nil { componentGlobs.append(Self.defaultTransformerFile) }
        var sources = [
            WeightSource(role: "components", repo: repo, revision: revision, matching: componentGlobs),
            WeightSource(role: "text-encoder", repo: gemmaRepo),
        ]
        if let tRepo = effectiveTransformerRepo {
            sources.append(WeightSource(role: "transformer-\(quant.rawValue)", repo: tRepo,
                                        matching: [Self.defaultTransformerFile]))
        }
        return sources
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        func storeHas(_ repo: String, files: [String]) -> Bool {
            guard let dir = ModelStore(root: storeRoot).directory(for: repo) else { return false }
            return files.allSatisfy { fm.fileExists(atPath: dir.appending(path: $0).path) }
        }
        return weightSources.filter { source in
            switch source.role {
            case "components":
                if let dir = ltxDirectory,
                   fm.fileExists(atPath: dir.appending(path: Self.componentFiles[0]).path) { return false }
                return !storeHas(source.repo, files: source.matching ?? [])
            case "text-encoder":
                if let dir = gemmaDirectory,
                   fm.fileExists(atPath: dir.appending(path: "config.json").path) { return false }
                return !storeHas(source.repo, files: ["config.json"])
            default:   // transformer-<quant>
                if let path = transformerPath, fm.fileExists(atPath: path.path) { return false }
                return !storeHas(source.repo, files: [Self.defaultTransformerFile])
            }
        }
    }

    /// The configuration with nil directories resolved to the store layout — what `load()` uses
    /// AFTER materialization. Explicit directories always win.
    public func resolved(storeRoot: URL?) -> LTX2Configuration {
        let store = ModelStore(root: storeRoot)
        var cfg = self
        if cfg.ltxDirectory == nil { cfg.ltxDirectory = store.directory(for: repo) }
        if cfg.gemmaDirectory == nil { cfg.gemmaDirectory = store.directory(for: gemmaRepo) }
        if cfg.transformerPath == nil, let tRepo = effectiveTransformerRepo {
            cfg.transformerPath = store.directory(for: tRepo)?.appending(path: Self.defaultTransformerFile)
        }
        return cfg
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
        // MEASURED 2026-07-01 after T3b+T3c (sequential encode, connector int8, decode-scoped
        // cacheLimit, fully-sequential DiT on low tiers) — see LOW-TIER-PLAN FINAL RESULTS. The
        // hint is peak − the declared per-quant residentBytes, so the engine's charge
        // (resident + reserve) equals the true measured stage-max peak:
        //   compact24  peak 15.36 GB (budget 16.8 ✓)   balanced32 peak 16.07 (22.4 ✓)
        //   standard64 peak 37.51   (44.8 ✓)           max128     peak 92.2  (0.85×128 ✓)
        switch profile {
        case .compact24:  return 3_000_000_000    // 15.36 − 13 (int4 resident) + headroom
        case .balanced32: return 4_000_000_000    // 16.07 − 13 + headroom
        case .standard64: return 16_000_000_000   // 37.51 − 22 (int8 resident) + headroom
        case .max128:     return 36_000_000_000   // TIGHTENED off the 481f i2v spot measure
        // (RunLTX2 --i2v-spot, 2026-07-01): 704×512×481f bf16 i2v + the 4.9 GB i2v-adapter LoRA
        // peaks 72.73 GB (floor 43.40 incl. LoRA · act 29.33). The hint covers peak − declared
        // bf16 resident (72.73 − 40 = 32.73, LoRA residency rides in the transient) + headroom.
        // t2v is lighter (480f peak 67.61 — BRIDGE-LTX-005), so i2v is the binding path.
        // Charge: 40 + 36 = 76 GB ≤ 0.85×128 (was 92 pre-measure — 16 GB returned to the governor).
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
        // Operate on the store-resolved view so auto-materialize (nil-dir) configs prewarm the
        // downloaded layout on later cold launches; not-yet-downloaded paths simply don't exist
        // and the prewarmer skips them (best-effort). Explicit dirs resolve to themselves.
        let r = resolved(storeRoot: modelsRootDirectory)
        guard let ltxDir = r.ltxDirectory else {
            return [r.gemmaDirectory, r.transformerPath].compactMap { $0 }
        }
        return Self.prewarmPaths(ltxDir: ltxDir, transformerPath: r.transformerPath,
                                 gemmaDirectory: r.gemmaDirectory)
    }

    private static func prewarmPaths(ltxDir: URL, transformerPath: URL?, gemmaDirectory: URL?) -> [URL] {
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
