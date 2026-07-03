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
    // Schema-v2 fields the WRAPPER needs at run time (the full v2 schema lives in
    // ltx-features-swift; unknown JSON keys decode-ignore in both directions):
    /// "plain" (weight delta, the default) | "ic" (needs reference conditioning — must ride the
    /// IC metaData intake, never the plain `loraId` path).
    public let kind: String?
    /// Spatial position scale for reference tokens (safetensors-header
    /// `reference_downscale_factor`; absent → 1).
    public let referenceDownscale: Int?
    /// IC stage policy: "skip" (one stage at target res) | "clean" | "keep".
    public let stage2: String?
    /// Non-permissive license — hosts must gate behind an explicit acknowledgment.
    public let licenseGated: Bool?

    /// Effective input kind (absent → `.none`).
    public var inputKind: LoRAInputKind { input ?? .none }
    public var isIC: Bool { kind == "ic" }

    /// Public memberwise init — consumers with richer registry schemas (ltx-features-swift's
    /// schema v2) map their entries down to this runtime subset (e.g. to reuse `LoRACache`).
    public init(id: String, displayName: String, repo: String, weightFile: String,
                defaultStrength: Float, trigger: String, input: LoRAInputKind? = nil,
                kind: String? = nil, referenceDownscale: Int? = nil, stage2: String? = nil,
                licenseGated: Bool? = nil) {
        self.id = id
        self.displayName = displayName
        self.repo = repo
        self.weightFile = weightFile
        self.defaultStrength = defaultStrength
        self.trigger = trigger
        self.input = input
        self.kind = kind
        self.referenceDownscale = referenceDownscale
        self.stage2 = stage2
        self.licenseGated = licenseGated
    }
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
            // Gated HF repos (e.g. the Lightricks IC-LoRAs) need an authenticated request from an
            // account that accepted the terms: honor HF_TOKEN env, else the HF CLI token file.
            var request = URLRequest(url: url)
            if let token = Self.hfToken() {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            let (tmp, response) = try await URLSession.shared.download(for: request)
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

    /// HF auth token for gated repos: `HF_TOKEN` env, else the HF CLI token file.
    static func hfToken() -> String? {
        if let env = ProcessInfo.processInfo.environment["HF_TOKEN"], !env.isEmpty { return env }
        let file = FileManager.default.homeDirectoryForCurrentUser
            .appending(path: ".cache/huggingface/token")
        guard let raw = try? String(contentsOf: file, encoding: .utf8) else { return nil }
        let token = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return token.isEmpty ? nil : token
    }
}

/// `metaData` keys the package reads for per-request effect selection.
public enum LoRAMetaKeys {
    /// Registry id of the effect to apply (absent / empty = pristine base, no LoRA).
    public static let id = "loraId"
    /// Optional strength override; defaults to the entry's `defaultStrength`.
    public static let strength = "loraStrength"
}

/// IC-adapter request keys (IC-LORA-PLAN P3/P4 — interim metaData transport until the engine's
/// `ConditioningInput` contract lands, P5). The reference image rides as a FILE PATH: the host
/// writes the picked/composed sheet to a temp file and passes it here.
public enum ICMetaKeys {
    /// Registry id of the IC adapter (kind == "ic"). Selecting an IC adapter via plain `loraId`
    /// is rejected — it would apply the weights WITHOUT their reference conditioning.
    public static let adapterId = "ic.adapterId"
    /// Optional LoRA strength override; defaults to the entry's `defaultStrength` (Ingredients: 1.4).
    public static let adapterStrength = "ic.adapterStrength"
    /// File path of the reference image (looped-still ingest; sheet for Ingredients).
    public static let referencePath = "ic.referencePath"
    /// Conditioning strength for the reference tokens (default 1.0 = fully preserved).
    public static let referenceStrength = "ic.referenceStrength"
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
