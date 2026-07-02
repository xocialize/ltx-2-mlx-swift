// GemmaTextGenerator.swift — text generation on LTX's own Gemma-3 (BRIDGE-LTX-004).
//
// The host app's prompt enhancer needs a chat completion, and LTX already ships an
// instruction-tuned Gemma-3-12B-it as its text encoder. Exposing generation HERE means the app
// links nothing beyond this package (no second LLM stack, no build-plugin trust, no app-side
// prewarm hack), and the load resolves through the SAME text-only path as the encoder — this
// module never links MLXVLM, so `gemma3` cannot shadow to the multimodal factory (BRIDGE-LTX-003).
//
// **DISPOSITION (BRIDGE-LTX-006, 2026-07-02): this is the UNGOVERNED convenience.** Engine-hosted
// apps should use `mlx-gemma-llm-swift` (`GemmaLLMPackage`, think/PROD, published v0.1.0) instead
// — the same text-only Gemma behind the full engine lifecycle: admission, footprint charge
// (measured int4 10 GB + ~1 GB), WeightPrewarming, license gate, evict. This type stays for
// engine-less consumers and this package's own CLI gate (`RunLTX2 --gemma-textgen-gate`); it does
// deterministic load→generate→release with a local prewarm, and the engine never knows it ran.

import Foundation
import MLX
import MLXLMCommon

public struct GemmaTextGenerator: Sendable {
    public let gemmaDirectory: URL

    public init(gemmaDirectory: URL) {
        self.gemmaDirectory = gemmaDirectory
    }

    /// One-shot (system, user) → completion on the Gemma-3 weights LTX uses as its text encoder.
    ///
    /// Deterministic transient: load → generate → release. The model lives only inside this call
    /// and `Memory.clearCache()` runs on exit (success or throw), so the ~7 GB Gemma residency is
    /// returned before a subsequent video run's activation peak — enhancement + generation must
    /// not stack past a tier envelope. Weight files are paged into the OS file cache first
    /// (mirroring the engine's `WeightPrewarmer`) because this load happens OUTSIDE an engine
    /// `prepare`, where a cold read off a slow/external volume would otherwise trip the Metal
    /// command-buffer watchdog (`kIOGPUCommandBufferCallbackErrorTimeout`).
    public func generate(system: String, user: String,
                         maxTokens: Int = 512,
                         temperature: Float = 0.7) async throws -> String {
        defer { Memory.clearCache() }   // runs after the do-scope releases the model
        let text: String
        do {
            Self.prewarm(directory: gemmaDirectory)
            let encoder = try await GemmaEncoder.load(directory: gemmaDirectory)
            let session = ChatSession(
                encoder.context,
                instructions: system,
                generateParameters: GenerateParameters(maxTokens: maxTokens,
                                                       temperature: temperature))
            text = try await session.respond(to: user)
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Page every weight file under `directory` into the unified buffer cache (stream + discard,
    /// no large allocation) — the package-side twin of the engine's `WeightPrewarmer`.
    static func prewarm(directory: URL) {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        else { return }
        let chunk = 64 * 1024 * 1024
        for url in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
        where url.pathExtension == "safetensors" {
            guard let handle = try? FileHandle(forReadingFrom: url) else { continue }
            while let data = try? handle.read(upToCount: chunk), !data.isEmpty {}
            try? handle.close()
        }
    }
}
