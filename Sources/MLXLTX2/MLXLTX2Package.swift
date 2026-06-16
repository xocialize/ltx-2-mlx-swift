import Foundation
import LTX2
import MLX
import MLXToolKit

/// MLXEngine package: Lightricks **LTX-2.3** distilled, exposing the canonical
/// `textToVideo` surface. One loaded pipeline = Gemma-3 text encoder + connector +
/// joint-AV DiT + 128-ch video VAE decoder (all parity-validated vs the ltx-2-mlx
/// oracle). Single-stage distilled t2v; video-only output for now (audio VAE/vocoder +
/// the two-stage upsampler are follow-ups).
///
/// ⚠️ EVAL-ONLY: the weights ship under the **LTX-2 Community License** (non-permissive:
/// §2 revenue gate, §3 derivative-copyleft, §A.20 non-compete). The manifest declares it
/// honestly as `LicenseRef-LTX-2-Community` on BOTH layers (the port code is itself a §3
/// Derivative). The default `.permissiveOnly` gate REJECTS it by design — admission for
/// capability-evaluation is an explicit host-side policy decision, never shippable.
///
/// Engine-owned lifecycle (C13): construct from `LTX2Configuration`, page in with `load()`,
/// drive `run(_:)`, reclaim with `unload()`. Lifecycle isolated to `InferenceActor`; the
/// non-`Sendable` pipeline never crosses the boundary. Cancellation honored before/after
/// the (heavy) generation.
@InferenceActor
public final class MLXLTX2Package: ModelPackage {
    public typealias Configuration = LTX2Configuration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7 weight + C8 port-code BOTH carry the LTX-2 Community License: the converted
            // weights AND this port are "Derivatives" under §3. Non-permissive by design.
            license: LicenseDeclaration(
                weightLicense: "LicenseRef-LTX-2-Community",
                portCodeLicense: "LicenseRef-LTX-2-Community"),
            provenance: Provenance(
                sourceRepo: "dgrauet/ltx-2.3-mlx",  // LTX-2.3 MLX collection (eval mirror)
                revision: "main",
                tier: 3),  // multi-component pipeline (Gemma + connector + DiT + VAE)
            requirements: RequirementsManifest(
                // DERIVED estimate (not yet memory-harnessed): bf16 resident ≈ Gemma-3-12B
                // 4bit 7.5 + connector 5.9 + distilled DiT 35 + vae_decoder 0.8 ≈ 50 GB, plus
                // activation/scratch headroom. Re-ground with a measured one-stage run
                // (MemoryProbe) before claiming a tier below .max.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 52_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max),
            specialties: [
                SpecialtyWeight(.general, strength: 0.6),
            ],
            surfaces: [
                T2VContract.descriptor(
                    name: "ltx-2.3-t2v",
                    summary: "Lightricks LTX-2.3 distilled text-to-video (MLX, eval-only). "
                        + "Joint audio-video DiT with a Gemma-3 text encoder and a 128-ch video "
                        + "VAE; single-stage distilled (8 steps). Video-only output currently "
                        + "(synchronized audio decode is a follow-up).")
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: LTX2Pipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        guard let ltxDir = configuration.ltxDirectory, let gemmaDir = configuration.gemmaDirectory else {
            throw PackageError.configurationMismatch(
                expected: "LTX2Configuration with ltxDirectory + gemmaDirectory set",
                got: "missing weight directories")
        }
        pipeline = try await LTX2Pipeline.load(ltxDir: ltxDir, gemmaDir: gemmaDir)
    }

    public func unload() async { pipeline = nil }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .textToVideo, let t2v = request as? T2VRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()
        let fps = t2v.fps ?? 24
        let pixels = pipeline.t2v(
            prompt: t2v.prompt,
            height: t2v.height ?? 256,
            width: t2v.width ?? 256,
            numFrames: t2v.numFrames ?? 9,
            fps: fps,
            seed: t2v.seed)
        try Task.checkCancellation()
        // LTX decoder is channels-first (1,3,F,H,W); the codec wants channels-last (1,F,H,W,3).
        let framesCL = pixels.transposed(0, 2, 3, 4, 1)
        let mp4 = try await encodeMP4(frames: framesCL, fps: fps)
        return T2VResponse(video: Video(
            format: .mp4, data: mp4,
            durationSeconds: Double(framesCL.dim(1)) / fps, frameRate: fps))
    }
}

extension MLXLTX2Package {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(MLXLTX2Package.self)
    }
}
