// LTXAdapters.swift — LTX on the engine's typed adapter contract (contract 1.53.0 + 1.55.0's
// conditioning groups; AB-A-0197, L2 of AB-D-0155).
//
// Before this, the package read adapters from `metaData` (`loraId`/`loraStrength` and the `ic.*`
// keys, the reference riding as a FILE PATH) and fetched the weights itself from unpinned
// `resolve/main` URLs into its own cache. Now:
//   • each t2v surface declares `AdapterSupport` (`support(for:)`), so the engine refuses an adapter
//     the surface can't run before admission;
//   • the engine owns the catalog (`catalog(for:)` builds it from the bundled format-3 registry,
//     ids qualified `<namespace>:<id>`), fetches each file at its pinned commit, checks its sha256,
//     and hands the package `resolvedAdapters`, with the IC conditioning already ordered into slots
//     and the `exactlyOne` groups checked;
//   • the package only applies what it was handed: the low-rank ADD for every adapter, and for an
//     IC adapter the reference ingest its bundled registry entry describes (`ingest`, downscale,
//     audio reference).

import CoreGraphics
import Foundation
import ImageIO
import MLXToolKit
import UniformTypeIdentifiers

// MARK: - The bundled registry (adapter registry format 3)

/// One entry of the bundled registry: the engine's `AdapterDescriptor` plus the registry-only
/// fields this package needs at run time. Format 3 is a superset of `AdapterDescriptor`, so the
/// same JSON object decodes as both (`ltx-lora-registry/FORMAT.md`).
public struct LTXAdapterEntry: Sendable, Decodable {
    /// The engine-facing subset, under the registry's own (unqualified) id.
    public let descriptor: AdapterDescriptor
    public let status: String?
    public let trigger: String?
    public let promptTemplate: String?
    public let runtime: Runtime
    /// Per slot: how the package prepares the input (`videoClip`, `loopedStillVideo`,
    /// `sheetBuilder`, `audioTrack`) and the reference's default conditioning strength.
    public let slots: [Slot]

    public struct Runtime: Sendable, Decodable, Equatable {
        /// Spatial scale of the reference tokens (the safetensors header's
        /// `reference_downscale_factor`); nil = 1.
        public let referenceDownscale: Int?
        /// IC stage policy. ⚠️ The package runs IC one stage at target resolution whatever this says.
        public let stage2: String?
        /// The adapter also conditions on reference AUDIO (LipDub): a bound audio slot, else the
        /// reference video's own track.
        public let audioReference: Bool?

        static let none = Runtime(referenceDownscale: nil, stage2: nil, audioReference: nil)
    }

    public struct Slot: Sendable, Decodable, Equatable {
        public let role: String
        public let ingest: String?
        public let defaultStrength: Double?
    }

    private enum CodingKeys: String, CodingKey {
        case status, trigger, promptTemplate, runtime, conditioning
    }

    public init(from decoder: any Decoder) throws {
        descriptor = try AdapterDescriptor(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        status = try container.decodeIfPresent(String.self, forKey: .status)
        trigger = try container.decodeIfPresent(String.self, forKey: .trigger)
        promptTemplate = try container.decodeIfPresent(String.self, forKey: .promptTemplate)
        runtime = try container.decodeIfPresent(Runtime.self, forKey: .runtime) ?? .none
        slots = try container.decodeIfPresent([Slot].self, forKey: .conditioning) ?? []
    }

    public func slot(_ role: String) -> Slot? { slots.first { $0.role == role } }
}

/// The bundled registry (`Resources/ltx-lora-registry.json`, synced from `ltx-lora-registry`'s
/// `registry.json` — format 3, not the v2 export).
public struct LTXAdapterRegistry: Sendable, Decodable {
    public let schemaVersion: Int
    public let namespace: String
    public let adapters: [LTXAdapterEntry]

    /// The registry bundled with the package.
    public static func bundled() throws -> LTXAdapterRegistry {
        guard let url = Bundle.module.url(forResource: "ltx-lora-registry", withExtension: "json") else {
            throw PackageError.configurationMismatch(
                expected: "the bundled LTX adapter registry", got: "no ltx-lora-registry.json in the bundle")
        }
        return try JSONDecoder().decode(LTXAdapterRegistry.self, from: Data(contentsOf: url))
    }

    /// The entry an adapter id names: the qualified catalog id (`ltx:ingredients`) or the bare
    /// registry id.
    public func entry(id: String) -> LTXAdapterEntry? {
        let bare = id.hasPrefix(namespace + ":") ? String(id.dropFirst(namespace.count + 1)) : id
        return adapters.first { $0.descriptor.id == bare }
    }

    /// The entry for a resolved adapter: by id, else by its pinned content (a host may register a
    /// descriptor under its own id; the bytes still identify it).
    public func entry(for adapter: ResolvedAdapter) -> LTXAdapterEntry? {
        entry(id: adapter.id)
            ?? adapters.first { $0.descriptor.source.sha256 == adapter.source.sha256 }
    }
}

// MARK: - Declarations and the catalog

public enum LTXAdapters {
    /// The registry's namespace, which qualifies catalog ids (`ltx:ingredients`).
    public static let namespace = "ltx"

    /// The dialects the DiT's low-rank ADD (`LTX2LoRA`) is verified on: every registry entry is
    /// diffusers-PEFT.
    public static let formats: [AdapterFormat] = [.diffusersPEFT]

    /// The base-model name adapters must declare for a family — `AdapterSupport.baseModel`, matched
    /// against the registry's `baseModels`.
    public static func baseModel(for family: LTXFamily) -> String {
        family == .ltx25 ? "ltx-2.5" : "ltx-2.3"
    }

    /// The t2v surface's adapter support (contract 1.53.0). Per request: the low-rank ADD is
    /// applied and detached on the resident DiT, and a DiT reloaded after a low-tier eviction is
    /// re-applied before the next denoise. One adapter per request, plain or IC: that is the
    /// combination the gates and live fixtures cover. Every quant: the ADD rides q8/q4 unchanged.
    public static func support(for family: LTXFamily) -> AdapterSupport {
        AdapterSupport(baseModel: baseModel(for: family), formats: formats,
                       kinds: [.plain, .referenceConditioned], maxCount: 1,
                       strengthRange: AdapterStrengthRange(min: 0, max: 2),
                       quants: nil, binding: .perRequest)
    }

    /// The engine catalog for a family: the bundled registry's adapters that fit it, under their
    /// qualified ids (`ltx:<id>`), ready for `engine.registerAdapters(_:)`. nil family = every
    /// adapter in the registry.
    public static func catalog(for family: LTXFamily? = nil) throws -> [AdapterDescriptor] {
        let registry = try LTXAdapterRegistry.bundled()
        return registry.adapters.map(\.descriptor)
            .filter { family == nil || $0.baseModels.contains(baseModel(for: family!)) }
            .map { qualified($0, namespace: registry.namespace) }
    }

    /// `descriptor` under the id `<namespace>:<id>` (unchanged when already qualified).
    public static func qualified(_ descriptor: AdapterDescriptor, namespace: String) -> AdapterDescriptor {
        let prefix = namespace + ":"
        guard !descriptor.id.hasPrefix(prefix) else { return descriptor }
        return AdapterDescriptor(
            id: prefix + descriptor.id, title: descriptor.title, kind: descriptor.kind,
            format: descriptor.format, baseModels: descriptor.baseModels,
            capabilities: descriptor.capabilities, source: descriptor.source,
            license: descriptor.license, defaultStrength: descriptor.defaultStrength,
            strengthRange: descriptor.strengthRange, conditioning: descriptor.conditioning,
            conditioningGroups: descriptor.conditioningGroups, footprint: descriptor.footprint)
    }

    /// Resolve `refs` the way the engine does, against a local file per adapter — for tests and
    /// harnesses that drive the package directly (the engine stamps `resolvedAdapters` itself).
    public static func resolve(_ refs: [AdapterRef], family: LTXFamily, catalog: [AdapterDescriptor],
                               file: (AdapterDescriptor) -> URL) throws -> [ResolvedAdapter] {
        try support(for: family).plan(refs, capability: .textToVideo, quant: nil) { id in
            catalog.first { $0.id == id }
        }.map { plan in
            ResolvedAdapter(id: plan.descriptor.id, kind: plan.descriptor.kind,
                            format: plan.descriptor.format, fileURL: file(plan.descriptor),
                            source: plan.descriptor.source, strength: plan.strength,
                            conditioning: plan.conditioning,
                            footprint: plan.descriptor.resolvedFootprint)
        }
    }
}

// MARK: - Run-time split

/// What `run` does with a request's `resolvedAdapters`.
struct LTXAdapterRun {
    /// Every adapter's low-rank ADD, in request order.
    let loras: [(url: URL, strength: Float)]
    /// Identity of the applied set: the same bytes at the same strengths are not re-applied.
    let key: [String]
    /// The reference-conditioned adapter, if any, with its registry entry (nil when the host
    /// registered an adapter this registry does not carry).
    let ic: (adapter: ResolvedAdapter, entry: LTXAdapterEntry?)?

    static let empty = LTXAdapterRun(loras: [], key: [], ic: nil)

    init(loras: [(url: URL, strength: Float)], key: [String],
         ic: (adapter: ResolvedAdapter, entry: LTXAdapterEntry?)?) {
        self.loras = loras
        self.key = key
        self.ic = ic
    }

    /// Check what only the package knows (the dialect, one IC adapter), then split. The engine
    /// already checked base model, kind, strength range and conditioning.
    init(_ adapters: [ResolvedAdapter], registry: LTXAdapterRegistry?) throws {
        for adapter in adapters where !LTXAdapters.formats.contains(adapter.format) {
            throw PackageError.unsupportedRequestFeature(
                "adapter \"\(adapter.id)\" is in the \(adapter.format) dialect; LTX reads "
                    + LTXAdapters.formats.map(\.rawValue).joined(separator: ", "))
        }
        let ics = adapters.filter { $0.kind == .referenceConditioned }
        if ics.count > 1 {
            throw PackageError.unsupportedRequestFeature(
                "adapters — LTX runs one reference-conditioned adapter per request, got \(ics.count)")
        }
        loras = adapters.map { ($0.fileURL, Float($0.strength)) }
        key = adapters.map { "\($0.source.sha256)@\($0.strength)" }
        ic = ics.first.map { ($0, registry?.entry(for: $0)) }
    }
}

// MARK: - Ingredients sheet

/// Compose an Ingredients reference sheet from subject images and an optional location shot —
/// the registry's `sheetBuilder` ingest, so a request can bind `subject_images` (+
/// `location_image`) instead of a finished `reference_sheet`. Same layout as ltx-features-swift's
/// `SheetComposer.compose` (up to 6 subject panels at native aspect in a top row, a full-width
/// location band at 40% height, black gutters, no text), built oversized at 1456×825 so identity
/// detail survives the downscale at ingest.
enum IngredientsSheet {
    static let locationRole = "location_image"
    static let width = 1456, height = 825, gutter: CGFloat = 12, locationFraction: CGFloat = 0.4

    static func compose(subjects: [Image], location: Image?) throws -> Image {
        func decode(_ image: Image, _ what: String) throws -> CGImage {
            guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                throw PackageError.configurationMismatch(
                    expected: "a decodable \(what) image (PNG/JPEG/…)", got: "undecodable image data")
            }
            return cg
        }
        let subjectImages = try subjects.map { try decode($0, "subject") }
        let locationImage = try location.map { try decode($0, "location") }
        guard !subjectImages.isEmpty || locationImage != nil,
              let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            throw PackageError.configurationMismatch(
                expected: "subject or location images for the Ingredients sheet", got: "none")
        }
        let W = CGFloat(width), H = CGFloat(height), g = gutter
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))
        ctx.interpolationQuality = .high
        let bandH: CGFloat = locationImage == nil ? 0
            : (subjectImages.isEmpty ? H - 2 * g : H * locationFraction)
        if let locationImage {
            draw(locationImage, fitIn: CGRect(x: g, y: g, width: W - 2 * g,
                                              height: bandH - (subjectImages.isEmpty ? 0 : g)), in: ctx)
        }
        if !subjectImages.isEmpty {
            // CG origin is bottom-left: the location band sits at the bottom, subjects above it.
            let rowY = locationImage == nil ? g : bandH + g
            let rowH = H - rowY - g
            let n = CGFloat(subjectImages.count)
            let panelW = (W - (n + 1) * g) / n
            for (i, img) in subjectImages.enumerated() {
                draw(img, fitIn: CGRect(x: g + CGFloat(i) * (panelW + g), y: rowY,
                                        width: panelW, height: rowH), in: ctx)
            }
        }
        let out = NSMutableData()
        guard let sheet = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)
        else {
            throw PackageError.configurationMismatch(
                expected: "an encodable Ingredients sheet", got: "PNG encode failed")
        }
        CGImageDestinationAddImage(dest, sheet, nil)
        CGImageDestinationFinalize(dest)
        return Image(format: .png, data: out as Data, width: width, height: height)
    }

    /// Aspect-fit `image` centered in `rect`; black shows around it.
    private static func draw(_ image: CGImage, fitIn rect: CGRect, in ctx: CGContext) {
        let iw = CGFloat(image.width), ih = CGFloat(image.height)
        guard iw > 0, ih > 0, rect.width > 0, rect.height > 0 else { return }
        let scale = min(rect.width / iw, rect.height / ih)
        let w = iw * scale, h = ih * scale
        ctx.draw(image, in: CGRect(x: rect.midX - w / 2, y: rect.midY - h / 2, width: w, height: h))
    }
}

// MARK: - Request options

/// The one `metaData` key the IC path still reads. It is an OUTPUT choice, not adapter selection:
/// adapters, their strengths and their references ride `T2VRequest.adapters` (contract 1.53.0).
public enum ICMetaKeys {
    /// `true`: mux the ACTUAL dub audio into the output MP4 instead of the model's regenerated
    /// track, for an adapter that conditions on reference audio (LipDub). The correct LipDub
    /// deliverable (I7: distilled audio generation is prosodic babble, a model property — the
    /// regenerated track is conditioning-grade only).
    public static let muxDubAudio = "ic.muxDubAudio"
}

/// Scratch files for media that arrive as bytes but are read through AVFoundation by URL. Removed
/// when the run ends.
final class RunScratch: @unchecked Sendable {
    private let lock = NSLock()
    private var urls: [URL] = []

    func write(_ data: Data, ext: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-ic-\(UUID().uuidString).\(ext)")
        try data.write(to: url)
        lock.lock(); urls.append(url); lock.unlock()
        return url
    }

    func removeAll() {
        lock.lock(); let all = urls; urls = []; lock.unlock()
        for url in all { try? FileManager.default.removeItem(at: url) }
    }
}
