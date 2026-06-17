import Foundation
import MLXToolKit

/// Init-time configuration for `MLXLTX2Package` (C9): where the LTX-2.3 component
/// weights and the Gemma-3 text encoder live. Per-request prompt/size/steps ride
/// the canonical `T2VRequest`, not here.
///
/// `ltxDirectory` holds the LTX safetensors (connector / transformer-distilled /
/// vae_decoder). `gemmaDirectory` is the Gemma-3 MLX weights dir (mlx-community/
/// gemma-3-12b-it-4bit). Both are environment-specific → excluded from Codable.
public struct LTX2Configuration: PackageConfiguration, ModelStorable {
    /// Provenance repo id (the LTX-2.3 MLX collection).
    public var repo: String
    public var revision: String?
    /// Backbone quant of the distilled transformer (selection metadata).
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
