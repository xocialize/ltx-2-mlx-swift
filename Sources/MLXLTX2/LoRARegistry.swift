// Curated registry of LTX-2.3 LoRAs for per-request effect selection.
//
// The manifest (Resources/ltx-lora-registry.json) references the original HuggingFace repos
// directly; each adapter's .safetensors is lazy-downloaded and cached on first use (no
// re-hosting). Dialect/remap across the diffusers / diffusion_model / kohya forms is handled by
// `LTX2LoRA` in the LTX2 core. Ported from qwen-image-edit-swift's LoRARegistry.
//
// SCOPE: plain style/motion/likeness LoRAs (weight delta). IC-LoRAs (Detailer / Water-Sim /
// Motion-Track / Ingredients) need a reference-video conditioning input path and are NOT served
// here — see LORA-PLAN.md L4.

import Foundation
import MLXToolKit

/// The conditioning input an effect expects — drives which input picker the UI reveals.
/// `none` = pure text-to-video; `image` = image-to-video (rides `T2VRequest.initImage`);
/// `video` = video-to-video (reserved; needs a package v2v path before it's selectable).
public enum LoRAInputKind: String, Codable, Sendable { case none, image, video }

/// One selectable effect. `repo`/`weightFile` resolve to a HF `resolve/main` download URL.
public struct LoRAEntry: Codable, Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let repo: String
    public let weightFile: String
    public let defaultStrength: Float
    public let trigger: String
    /// Conditioning input this effect needs (optional in JSON; absent → `.none`).
    public let input: LoRAInputKind?

    /// Effective input kind (absent → `.none`).
    public var inputKind: LoRAInputKind { input ?? .none }
}

/// The decoded registry plus id lookup + lazy file resolution.
public struct LoRARegistry: Codable, Sendable {
    public let schemaVersion: Int
    public let base: String
    public let adapters: [LoRAEntry]

    public func entry(id: String) -> LoRAEntry? { adapters.first { $0.id == id } }

    /// Load the manifest bundled with the package.
    public static func bundled() throws -> LoRARegistry {
        guard let url = Bundle.module.url(forResource: "ltx-lora-registry", withExtension: "json")
        else { throw LoRARegistryError.manifestMissing }
        return try JSONDecoder().decode(LoRARegistry.self, from: Data(contentsOf: url))
    }
}

public enum LoRARegistryError: Error, LocalizedError {
    case manifestMissing
    case unknownAdapter(String)
    case download(String, underlying: String)

    public var errorDescription: String? {
        switch self {
        case .manifestMissing: return "Bundled LTX LoRA registry manifest not found."
        case .unknownAdapter(let id): return "No LoRA with id '\(id)' in the registry."
        case .download(let id, let underlying):
            return "Failed to download LoRA '\(id)': \(underlying)"
        }
    }
}

/// Lazy-downloads + caches registry LoRAs from HuggingFace `resolve/main`.
public struct LoRACache: Sendable {
    public let directory: URL

    public init(directory: URL) { self.directory = directory }

    /// Local path for an entry's cached file (id-named so display order / weightFile renames
    /// don't fork the cache).
    public func localURL(for entry: LoRAEntry) -> URL {
        directory.appendingPathComponent("\(entry.id).safetensors")
    }

    /// Return the cached file, downloading it on first use. Atomic (download to a temp file,
    /// then move) so a partial download never poisons the cache.
    public func ensure(_ entry: LoRAEntry) async throws -> URL {
        let dest = localURL(for: entry)
        if FileManager.default.fileExists(atPath: dest.path) { return dest }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = "huggingface.co"
        // Percent-encode the path so non-ASCII / spaced weight names resolve.
        comps.path = "/\(entry.repo)/resolve/main/\(entry.weightFile)"
        guard let url = comps.url else {
            throw LoRARegistryError.download(entry.id, underlying: "bad URL for \(entry.weightFile)")
        }
        do {
            let (tmp, response) = try await URLSession.shared.download(from: url)
            if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                throw LoRARegistryError.download(entry.id, underlying: "HTTP \(http.statusCode)")
            }
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: tmp, to: dest)
            return dest
        } catch let e as LoRARegistryError {
            throw e
        } catch {
            throw LoRARegistryError.download(entry.id, underlying: error.localizedDescription)
        }
    }
}

/// `metaData` keys the package reads for per-request effect selection.
public enum LoRAMetaKeys {
    /// Registry id of the effect to apply (absent / empty = pristine base, no LoRA).
    public static let id = "loraId"
    /// Optional strength override; defaults to the entry's `defaultStrength`.
    public static let strength = "loraStrength"
}

extension MetaValue {
    /// String payload.
    public var asString: String? { if case .string(let s) = self { return s }; return nil }
    /// Numeric payload as Float (accepts int or double).
    public var asFloat: Float? {
        switch self {
        case .double(let d): return Float(d)
        case .int(let i): return Float(i)
        default: return nil
        }
    }
}
