import Foundation
import MLXToolKit

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

    public init(
        repo: String = "dgrauet/ltx-2.3-mlx",
        revision: String? = nil,
        quant: Quant = .bf16,
        ltxDirectory: URL? = nil,
        transformerPath: URL? = nil,
        gemmaDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.ltxDirectory = ltxDirectory
        self.transformerPath = transformerPath
        self.gemmaDirectory = gemmaDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant
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
