import Foundation
import LTX2
import MLX
import MLXToolKit

/// MLXEngine package: Lightricks **LTX-2.3** distilled, exposing the canonical
/// `textToVideo` surface. One loaded pipeline = Gemma-3 text encoder + connector +
/// joint-AV DiT + 128-ch video VAE + audio VAE + BigVGAN/BWE vocoder (all parity-
/// validated vs the ltx-2-mlx oracle). Produces **synchronized audio-video** (the
/// joint DiT denoises both; audio is muxed into the MP4 when the audio_vae/vocoder
/// weights are present). Single-stage distilled; the two-stage upsampler is a follow-up.
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
                // MEASURED 2026-06-16 (Xcode agent, M5 Max / 128 GB, two-stage 704×512 × 9f,
                // seed 42): peak OS phys_footprint = 82.81 GB (real working-set high-water — DiT
                // denoise + VAE-decode activations on top of the ~52 GB weight residency).
                // Declared as residentBytes ≈ 84 GB (measured peak + headroom) = the governor's
                // true max-simultaneous basis (TI2V precedent), NOT the 52 GB weight estimate.
                // PEAK SCALES WITH SEQ-LEN (resolution × frames); 84 GB is grounded at 704×512×9f
                // — re-measure before claiming higher res/frames fit a tier. (Static manifest =
                // one figure per quant.)
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 84_000_000_000),
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
                        + "Joint audio-video DiT with a Gemma-3 text encoder, a 128-ch video VAE, "
                        + "and a BigVGAN/BWE audio vocoder; single-stage distilled (8 steps). "
                        + "Output is an MP4 with synchronized 48kHz stereo audio.")
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
        let h = t2v.height ?? 512, wd = t2v.width ?? 704, nf = t2v.numFrames ?? 9
        // Prefer the two-stage distilled path (half-res → upsample → refine) when the
        // encoder + upsampler are present; else single-stage at target resolution.
        let out = pipeline.supportsTwoStage
            ? pipeline.t2vTwoStage(prompt: t2v.prompt, height: h, width: wd, numFrames: nf, fps: fps, seed: t2v.seed)
            : pipeline.t2v(prompt: t2v.prompt, height: h, width: wd, numFrames: nf, fps: fps, seed: t2v.seed)
        try Task.checkCancellation()
        // LTX decoder is channels-first (1,3,F,H,W); the codec wants channels-last (1,F,H,W,3).
        // out.audio is 48kHz stereo (1,2,T) when the audio components are present → muxed.
        let framesCL = out.video.transposed(0, 2, 3, 4, 1)
        let mp4 = try await encodeMP4(frames: framesCL, fps: fps, audio: out.audio, audioSampleRate: 48000)
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
