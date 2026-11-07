// HarnessAdapters.swift — typed adapters for harnesses that drive the package directly.
//
// Through the engine, a request's `adapters` are resolved, fetched at the pinned commit and
// hash-checked by the engine, which stamps `resolvedAdapters` before `run` (contract 1.53.0,
// AB-A-0197). These harnesses construct the package themselves, so they do the stamping here:
// the same pre-flight (`AdapterSupport.plan`, via `LTXAdapters.resolve`) against the bundled
// catalog, pointed at already-local files. They never download; fetch a missing adapter through
// the engine (`prepareAdapters`) or copy it into the directory below.

import Foundation
import MLXLTX2
import MLXToolKit

enum HarnessAdapters {
    /// Local adapter files, `<registry id>.safetensors` — the layout the package's own cache used
    /// before 1.53.0: `$LTX_LORA_CACHE`, else `<LTX_MODELS_ROOT or /Volumes/Satechi/Models>/ltx-lora-cache`.
    static var directory: URL {
        let env = ProcessInfo.processInfo.environment
        if let dir = env["LTX_LORA_CACHE"], !dir.isEmpty { return URL(fileURLWithPath: dir) }
        let root = env["LTX_MODELS_ROOT"].flatMap { $0.isEmpty ? nil : $0 } ?? "/Volumes/Satechi/Models"
        return URL(fileURLWithPath: root).appendingPathComponent("ltx-lora-cache")
    }

    /// The local file for a catalog descriptor (qualified or bare id).
    static func file(for descriptor: AdapterDescriptor) -> URL {
        let prefix = LTXAdapters.namespace + ":"
        let bare = descriptor.id.hasPrefix(prefix) ? String(descriptor.id.dropFirst(prefix.count)) : descriptor.id
        return directory.appendingPathComponent("\(bare).safetensors")
    }

    /// The qualified catalog id for a registry id (`ingredients` → `ltx:ingredients`).
    static func catalogID(_ id: String) -> String {
        id.contains(":") ? id : "\(LTXAdapters.namespace):\(id)"
    }

    /// `request` with its `adapters` resolved the way the engine would, against local files.
    static func resolved(_ request: T2VRequest, family: LTXFamily) throws -> T2VRequest {
        guard !request.adapters.isEmpty else { return request }
        let catalog = try LTXAdapters.catalog(for: family)
        let resolved = try LTXAdapters.resolve(request.adapters, family: family, catalog: catalog,
                                               file: file(for:))
        for adapter in resolved where !FileManager.default.fileExists(atPath: adapter.fileURL.path) {
            throw PackageError.configurationMismatch(
                expected: "adapter \(adapter.id) at \(adapter.fileURL.path)",
                got: "no local file (fetch it through the engine, or set LTX_LORA_CACHE)")
        }
        return request.resolving(resolved)
    }

    /// A file as conditioning media, by extension: .mp4/.mov video, .wav audio, else an image.
    static func media(path: String) throws -> ConditioningMedia {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        switch url.pathExtension.lowercased() {
        case "mp4", "m4v": return .video(Video(format: .mp4, data: data))
        case "mov": return .video(Video(format: .mov, data: data))
        case "wav": return .audio(Audio(format: .wav, data: data))
        case "jpg", "jpeg": return .image(Image(format: .jpeg, data: data))
        default: return .image(Image(format: .png, data: data))
        }
    }

    /// An IC adapter's reference bound to the first slot whose media matches the file — what the
    /// pre-1.53 `ic.referencePath` meant for every LTX IC adapter.
    static func reference(adapter id: String, path: String, strength: Double? = nil,
                          family: LTXFamily) throws -> ConditioningInput {
        let media = try media(path: path)
        guard let descriptor = try LTXAdapters.catalog(for: family).first(where: { $0.id == catalogID(id) }),
              let slot = descriptor.conditioning.first(where: { $0.media == media.kind }) else {
            throw PackageError.configurationMismatch(
                expected: "an IC adapter \(catalogID(id)) with a \(media.kind.rawValue) slot",
                got: path)
        }
        return ConditioningInput(role: slot.role, media: media, strength: strength)
    }
}
