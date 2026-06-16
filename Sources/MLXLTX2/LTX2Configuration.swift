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
    /// Resolved LTX component directory (connector/transformer-distilled/vae_decoder).
    public var ltxDirectory: URL?
    /// Resolved Gemma-3 text-encoder directory.
    public var gemmaDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "dgrauet/ltx-2.3-mlx",
        revision: String? = nil,
        quant: Quant = .bf16,
        ltxDirectory: URL? = nil,
        gemmaDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.ltxDirectory = ltxDirectory
        self.gemmaDirectory = gemmaDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant
    }
}
