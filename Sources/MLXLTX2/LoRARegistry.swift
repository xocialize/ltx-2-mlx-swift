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
import MLXProfiling
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
    /// IC adapter also appends a reference-AUDIO stream (LipDub): the wrapper builds it from
    /// `ic.dubAudioPath` (falling back to the reference video's own track).
    public let audioReference: Bool?

    /// Effective input kind (absent → `.none`).
    public var inputKind: LoRAInputKind { input ?? .none }
    public var isIC: Bool { kind == "ic" }

    /// Public memberwise init — consumers with richer registry schemas (ltx-features-swift's
    /// schema v2) map their entries down to this runtime subset (e.g. to reuse `LoRACache`).
    public init(id: String, displayName: String, repo: String, weightFile: String,
                defaultStrength: Float, trigger: String, input: LoRAInputKind? = nil,
                kind: String? = nil, referenceDownscale: Int? = nil, stage2: String? = nil,
                licenseGated: Bool? = nil, audioReference: Bool? = nil) {
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
        self.audioReference = audioReference
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

    /// True when the entry's weights are already materialized locally (ensure() needs no network).
    /// Hosts use this to route a first-use fetch through an explicit download phase (gap-queue P2)
    /// instead of paying it silently inside a timed generation.
    public func isCached(_ entry: LoRAEntry) -> Bool {
        FileManager.default.fileExists(atPath: localURL(for: entry).path)
    }

    /// Return the cached file, downloading it on first use. Atomic (download to a temp file,
    /// then move) so a partial download never poisons the cache.
    ///
    /// A first-use fetch is never silent (gap-queue P2): the kickoff logs the resolved URL, byte
    /// progress reaches whatever `WeightDownloadProgress` sink the CALLER bound (captured at
    /// start — the TaskLocal does not flow onto URLSession's delegate queue), and completion
    /// logs size + elapsed.
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
            MLXProfiler.shared.note("LoRA '\(entry.id)': fetching \(url.absoluteString)")
            let t0 = Date()
            // Classic session-level delegate + explicit downloadTask: the async
            // `download(for:delegate:)` convenience does NOT deliver didWriteData progress to a
            // task-level delegate (verified via --lora-fetch-gate), so the fetch would be silent.
            let delegate = DownloadProgressDelegate(label: "LoRA '\(entry.id)'",
                                                    sink: WeightDownloadProgress.sink,
                                                    holdDirectory: directory)
            let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
            defer { session.finishTasksAndInvalidate() }
            let tmp = try await delegate.download(request, using: session)
            WeightDownloadProgress.report(fraction: 1.0)
            let bytes = ((try? FileManager.default.attributesOfItem(atPath: tmp.path))?[.size] as? UInt64) ?? 0
            MLXProfiler.shared.note(String(format: "LoRA '%@': downloaded %.2f GB in %.0fs",
                                           entry.id as NSString, Double(bytes) / 1e9,
                                           Date().timeIntervalSince(t0)))
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: tmp, to: dest)
            return dest
        } catch let e as LoRARegistryError {
            throw e
        } catch is CancellationError {
            throw CancellationError()   // keep cancellation typed — hosts report "cancelled", not "failed"
        } catch let u as URLError where u.code == .cancelled {
            throw CancellationError()   // URLSession surfaces task cancellation as URLError(.cancelled)
        } catch {
            throw LoRARegistryError.download(entry.id, underlying: error.localizedDescription)
        }
    }

    /// Session-level URLSession delegate: forwards byte progress to a CAPTURED sink (the
    /// `WeightDownloadProgress` TaskLocal does not flow onto URLSession's delegate queue) and
    /// bridges the classic downloadTask callbacks to async/await. Progress is throttled to
    /// ≥1% / ≥2 s steps so a multi-GB fetch produces ~100 updates, not thousands. Mutable state
    /// is confined to the session's serial delegate queue + an NSLock for the one-shot
    /// continuation (@unchecked Sendable).
    private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        private let sink: WeightDownloadProgress.Sink?
        private let label: String
        private let holdDirectory: URL
        private let started = Date()
        private var lastFraction = 0.0
        private var lastAt = Date.distantPast
        private var announced = false

        private let lock = NSLock()
        private var continuation: CheckedContinuation<URL, Error>?
        private var task: URLSessionDownloadTask?

        init(label: String, sink: WeightDownloadProgress.Sink?, holdDirectory: URL) {
            self.label = label
            self.sink = sink
            self.holdDirectory = holdDirectory
        }

        /// Run the download; returns the relocated temp file (inside `holdDirectory`, so the
        /// caller's atomic move stays on one volume). Cancelling the surrounding Swift Task
        /// cancels the URLSession task (→ URLError.cancelled → typed CancellationError upstream).
        func download(_ request: URLRequest, using session: URLSession) async throws -> URL {
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { cont in
                    lock.lock()
                    continuation = cont
                    let t = session.downloadTask(with: request)
                    task = t
                    lock.unlock()
                    t.resume()
                }
            } onCancel: {
                lock.lock()
                let t = task
                lock.unlock()
                t?.cancel()
            }
        }

        private func finish(_ result: Result<URL, Error>) {
            lock.lock()
            let cont = continuation
            continuation = nil
            lock.unlock()
            switch result {
            case .success(let url): cont?.resume(returning: url)
            case .failure(let error): cont?.resume(throwing: error)
            }
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                        totalBytesExpectedToWrite: Int64) {
            if !announced {
                announced = true
                let size = totalBytesExpectedToWrite > 0
                    ? String(format: "%.2f GB", Double(totalBytesExpectedToWrite) / 1e9)
                    : "unknown size"
                MLXProfiler.shared.note("\(label): downloading (\(size))")
            }
            guard totalBytesExpectedToWrite > 0 else { return }
            let fraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
            let now = Date()
            guard fraction - lastFraction >= 0.01 || now.timeIntervalSince(lastAt) >= 2 else { return }
            lastFraction = fraction
            lastAt = now
            sink?(fraction, Double(totalBytesWritten) / max(now.timeIntervalSince(started), 0.001))
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                        didFinishDownloadingTo location: URL) {
            // `location` dies when this callback returns — relocate synchronously, into the
            // cache directory so the caller's final move never crosses volumes.
            if let http = downloadTask.response as? HTTPURLResponse,
               !(200..<300).contains(http.statusCode) {
                try? FileManager.default.removeItem(at: location)
                finish(.failure(NSError(domain: "LoRACache", code: http.statusCode,
                                        userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode)"])))
                return
            }
            do {
                try FileManager.default.createDirectory(at: holdDirectory, withIntermediateDirectories: true)
                let hold = holdDirectory.appendingPathComponent(".partial-\(UUID().uuidString)")
                try FileManager.default.moveItem(at: location, to: hold)
                finish(.success(hold))
            } catch {
                finish(.failure(error))
            }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            // Success already resumed in didFinishDownloadingTo; this catches transport errors
            // (including cancellation → URLError.cancelled).
            if let error { finish(.failure(error)) }
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
    /// File path of the dub audio (LipDub-class adapters, `audioReference: true`); absent →
    /// the reference video's own audio track.
    public static let dubAudioPath = "ic.dubAudioPath"
    /// "true"/bool: mux the ACTUAL dub audio into the output MP4 instead of the model's
    /// regenerated track. The correct LipDub deliverable (I7: distilled audio generation is
    /// prosodic babble, a model property — the regenerated track is conditioning-grade only).
    public static let muxDubAudio = "ic.muxDubAudio"
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
