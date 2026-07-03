import Foundation
import LTX2
import MLX
import MLXProfiling
import MLXToolKit

/// MLXEngine package: Lightricks **LTX-2.3** distilled, exposing the canonical
/// `textToVideo` surface. One loaded pipeline = Gemma-3 text encoder + connector +
/// joint-AV DiT + 128-ch video VAE + audio VAE + BigVGAN/BWE vocoder (all parity-
/// validated vs the ltx-2-mlx oracle). Produces **synchronized audio-video** (the
/// joint DiT denoises both; audio is muxed into the MP4 when the audio_vae/vocoder
/// weights are present). Single-stage distilled; the two-stage upsampler is a follow-up.
///
/// LICENSE (two layers, declared honestly):
///   • **weights** → **LTX-2 Community License** (`LicenseRef-LTX-2-Community`): source-available
///     with a §2 revenue gate, §3 terms, and a §A.20 non-compete.
///   • **port code** → **Apache-2.0**: this is our own implementation. Lightricks licenses their
///     own inference code (`ltx-core`/`ltx-pipelines`, see the open-source LTX-Desktop) as
///     Apache-2.0 — i.e. the upstream authors do NOT treat inference code as a §3 derivative of
///     the weights. Our port mirrors that posture.
/// The engine (≥0.6.0) places LTX-2-Community on the `permissiveAllowlist`, so BOTH layers are
/// admitted by the default `.permissiveOnly` policy — no eval-acknowledged relaxation needed.
/// (The weights' own §2/§A.20 terms still bind any downstream use; that's a usage obligation,
/// not an engine-admission gate.)
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
            // C7 weights = LTX-2 Community License; C8 port code = Apache-2.0 (our own
            // implementation, mirroring Lightricks' Apache-2.0 inference code). Both layers
            // clear the default `.permissiveOnly` gate (LTX-2-Community is allowlisted as of
            // engine 0.6.0).
            license: LicenseDeclaration(
                weightLicense: .ltx2Community,
                portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "dgrauet/ltx-2.3-mlx",  // LTX-2.3 MLX collection (eval mirror)
                revision: "main",
                tier: 3),  // multi-component pipeline (Gemma + connector + DiT + VAE)
            requirements: RequirementsManifest(
                // SPLIT FOOTPRINT (contract 1.14.0) — RE-MEASURED 2026-06-30 in LTXVideoTesting
                // (M5 Max / 128 GB, two-stage 704×512 × 9f, seed 42) AFTER the per-stage
                // load→use→evict refactor of `LTX2Pipeline`. Only the DiT backbone stays resident
                // through the denoise peak; Gemma+Connector evict after text-encode and the VAE
                // decode stack loads after denoise. Because weights are lazy-mmap, the DiT isn't
                // even materialized during encode → the text encoder and the DiT are NEVER
                // co-resident, so the peak is the DiT denoise ALONE. That dropped the bf16 peak
                // from 82.81 GB (old all-resident, declared 84) to 51.95 GB measured.
                //
                //   residentBytes      = the DiT weight floor (post-run phys_footprint after the
                //                        pipeline's stage-evict clearCache → DiT-only resident).
                //   peakActivationBytes = the denoise transient (measured peak − floor). It is
                //                        ~dtype-INDEPENDENT (same bf16 compute): 13.1 GB bf16 /
                //                        12.1 GB int8 / 15.1 GB int4 — so the engine reserves ONE
                //                        such transient across ALL residents (serialized inference),
                //                        which is the co-residency win the split exists to capture.
                //
                // PEAK SCALES WITH SEQ-LEN (resolution × frames) — re-measure before claiming a
                // higher res/frame envelope fits a tier. (Each value = measured + ~20% headroom.)
                footprints: [
                    // bf16 DiT = 37.99 GB on disk. MEASURED: floor 38.85 GB · peak 51.95 GB · act 13.10 GB.
                    QuantFootprint(quant: .bf16, residentBytes: 40_000_000_000, peakActivationBytes: 16_000_000_000),
                    // int8: ONLY the transformer-block Linears are int8 (everything else stays bf16);
                    // q8 transformer = 20.6 GB on disk. MEASURED: floor 21.49 GB · peak 33.59 GB · act 12.10 GB.
                    QuantFootprint(quant: .int8, residentBytes: 22_000_000_000, peakActivationBytes: 15_000_000_000),
                    // int4: same quant-aware path (bits auto-detected from the scales shape); q4
                    // transformer = 11.3 GB on disk. MEASURED: floor 12.19 GB · peak 27.25 GB · act 15.06 GB
                    // (q4 dequant scratch nudges the transient up). NB q4 is artifact-free but DIVERGES
                    // the distilled sample vs bf16/q8 (larger per-step quant error shifts the trajectory).
                    QuantFootprint(quant: .int4, residentBytes: 13_000_000_000, peakActivationBytes: 18_000_000_000),
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
                    summary: "Lightricks LTX-2.3 distilled text/image-to-video (MLX, research port). "
                        + "Joint audio-video DiT with a Gemma-3 text encoder, a 128-ch video VAE, "
                        + "and a BigVGAN/BWE audio vocoder; distilled (8 steps). Supports i2v via "
                        + "`initImage` (first-frame conditioning). Output is an MP4 with synced 48kHz audio. "
                        + "Optional runtime LoRA via metaData: `loraId` (registry id of a style/motion "
                        + "effect; absent = base) + `loraStrength` (override; default per entry), "
                        + "hot-swapped on the resident DiT.")
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: LTX2Pipeline?
    // Per-request runtime-LoRA state (the "extend" capability): curated registry + lazy HF cache,
    // plus the currently-applied selection so an unchanged request doesn't re-apply.
    private var registry: LoRARegistry?
    private var cache: LoRACache?
    private var appliedLoRA: (id: String, strength: Float)?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        // Auto-materialization (BRIDGE M4): nil directories resolve to the engine ModelStore
        // layout, downloading missing declared sources first (progress reaches the engine's
        // PreparationMonitor via WeightDownloadProgress). Explicit directories always win and
        // never touch the network — the DEV_ARCHIVE flows are unchanged.
        var cfg = configuration
        if cfg.ltxDirectory == nil || cfg.gemmaDirectory == nil
            || (cfg.transformerPath == nil && cfg.effectiveTransformerRepo != nil) {
            guard let root = cfg.modelsRootDirectory else {
                throw PackageError.configurationMismatch(
                    expected: "explicit weight directories OR an engine model store (useModelStore)",
                    got: "no ltxDirectory/gemmaDirectory and no modelsRootDirectory")
            }
            let missing = cfg.missingWeightSources(storeRoot: root)
            if !missing.isEmpty {
                try await WeightMaterializer.materialize(missing, into: root)
            }
            cfg = cfg.resolved(storeRoot: root)
        }
        guard let ltxDir = cfg.ltxDirectory, let gemmaDir = cfg.gemmaDirectory else {
            throw PackageError.configurationMismatch(
                expected: "resolvable weight directories",
                got: "materialization did not yield ltxDirectory/gemmaDirectory")
        }
        pipeline = try await LTX2Pipeline.load(ltxDir: ltxDir, gemmaDir: gemmaDir,
                                               transformerPath: cfg.transformerPath)

        // Runtime-LoRA registry (HF-referencing manifest) + lazy download cache. Optional: if the
        // bundled manifest is missing, LoRA selection is simply unavailable (base still runs).
        registry = try? LoRARegistry.bundled()
        let cacheRoot = configuration.modelsRootDirectory
            ?? (try? FileManager.default.url(for: .cachesDirectory, in: .userDomainMask,
                                             appropriateFor: nil, create: true))
            ?? FileManager.default.temporaryDirectory
        cache = LoRACache(directory: cacheRoot.appendingPathComponent("ltx-lora-cache"))
        appliedLoRA = nil
    }

    public func unload() async {
        pipeline = nil; appliedLoRA = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .textToVideo, let t2v = request as? T2VRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // Per-request runtime LoRA (the "extend" capability), carried opaquely in metaData:
        //   metaData["loraId"]       registry id of the effect (absent/empty → pristine base)
        //   metaData["loraStrength"] optional strength override (else the entry default)
        // Hot-swapped on the resident DiT (no reload); unchanged selection across calls is a no-op.
        // IC adapters (kind == "ic") must ride the IC intake below — as a plain LoRA they'd apply
        // WITHOUT their reference conditioning and silently generate garbage-adjacent output.
        let icId = t2v.metaData[ICMetaKeys.adapterId]?.asString.flatMap { $0.isEmpty ? nil : $0 }
        let selectedId = icId
            ?? t2v.metaData[LoRAMetaKeys.id]?.asString.flatMap { $0.isEmpty ? nil : $0 }
        if let id = selectedId {
            guard let registry, let cache, let entry = registry.entry(id: id) else {
                throw LoRARegistryError.unknownAdapter(id)
            }
            if entry.isIC && icId == nil {
                throw PackageError.configurationMismatch(
                    expected: "IC adapter '\(id)' via the ic.* metaData intake (needs a reference)",
                    got: "plain loraId selection")
            }
            let strength = t2v.metaData[LoRAMetaKeys.strength]?.asFloat
                ?? t2v.metaData[ICMetaKeys.adapterStrength]?.asFloat
                ?? entry.defaultStrength
            // Re-apply when the selection changed OR the DiT was evicted/reloaded since (low-tier
            // decode evicts the DiT; a reloaded DiT is a pristine base → activeLoRATargets == 0).
            if appliedLoRA?.id != id || appliedLoRA?.strength != strength || pipeline.activeLoRATargets == 0 {
                let file = try await cache.ensure(entry)
                let span = MLXProfiler.shared.begin("lora-apply", id, note: file.lastPathComponent)
                try pipeline.setLoRAs([(file, strength)])
                MLXProfiler.shared.end(span)
                appliedLoRA = (id, strength)
            }
        } else if appliedLoRA != nil {
            pipeline.clearLoRAs()
            appliedLoRA = nil
        }
        try Task.checkCancellation()

        let fps = t2v.fps ?? 24
        var h = t2v.height ?? 512, wd = t2v.width ?? 704, nf = t2v.numFrames ?? 9
        // Tier profile (LOW-TIER-PLAN T3): CLAMP the request to the profile's envelope (what makes
        // the declared `peakActivationBytesHint` honest), select the one-stage path on low tiers,
        // and set the VAE decode window. `LTX_ONE_STAGE=1` forces one-stage for headless measurement.
        if let p = configuration.profile {
            wd = min(wd, p.maxWidth); h = min(h, p.maxHeight); nf = min(nf, p.maxFrames)
            wd = max(64, (wd / 32) * 32); h = max(64, (h / 32) * 32)   // latent grid is /32
            pipeline.vaeChunkFrames = p.vaeChunkFrames
            pipeline.evictDiTBeforeDecode = p.evictDiTBeforeDecode
        }
        let oneStage = configuration.profile?.oneStage
            ?? (ProcessInfo.processInfo.environment["LTX_ONE_STAGE"] == "1")

        // IC reference ingest (IC-LORA-PLAN P2/P3): the reference image rides metaData as a file
        // path (interim until the ConditioningInput contract lands, P5). Built at the CLAMPED
        // generation geometry / the adapter's declared downscale; frames snapped 8k+1.
        var references: [ReferenceConditioning] = []
        var audioReferences: [ReferenceConditioning] = []
        if let icId {
            guard let registry, let entry = registry.entry(id: icId) else {
                throw LoRARegistryError.unknownAdapter(icId)
            }
            guard let refPath = t2v.metaData[ICMetaKeys.referencePath]?.asString, !refPath.isEmpty else {
                throw PackageError.configurationMismatch(
                    expected: "ic.referencePath (file path of the reference image) for '\(icId)'",
                    got: "no reference")
            }
            let ds = max(1, entry.referenceDownscale ?? 1)
            let refFrames = ReferenceConditioning.snapFrames(nf)
            // Route by media: a video clip samples frames by time (Cameraman-class refs);
            // anything else decodes as a still and tiles (Ingredients sheets).
            let pixels: MLXArray
            if VideoInput.looksLikeVideo(refPath) {
                pixels = try await VideoInput.referenceClipFrames(
                    url: URL(fileURLWithPath: refPath),
                    width: wd / ds, height: h / ds, frames: refFrames, fps: fps)
            } else {
                let refData = try Data(contentsOf: URL(fileURLWithPath: refPath))
                pixels = try ImageInput.referenceStillFrames(
                    Image(format: .png, data: refData), width: wd / ds, height: h / ds, frames: refFrames)
            }
            let refStrength = t2v.metaData[ICMetaKeys.referenceStrength]?.asFloat ?? 1.0
            references.append(try pipeline.encodeReference(
                pixels: pixels, fps: fps, downscaleFactor: Float(ds), strength: refStrength))

            // LipDub-class adapters (`audioReference: true`): the dub audio appends to the AUDIO
            // stream with negative-time positions. Source = ic.dubAudioPath (a wav/audio file),
            // falling back to the reference video's own track (the oracle's default).
            if entry.audioReference == true {
                let audioPath = t2v.metaData[ICMetaKeys.dubAudioPath]?.asString ?? refPath
                let span = Double(nf) / fps
                let waveform = try await AudioInput.referenceWaveform(
                    url: URL(fileURLWithPath: audioPath), maxSeconds: span)
                audioReferences.append(try pipeline.encodeAudioReference(waveform: waveform))
            }
        }

        let out: LTX2Pipeline.Output
        if !references.isEmpty || !audioReferences.isEmpty {
            // IC path: ONE stage at target resolution (`stage2: skip` — the community-blessed
            // Ingredients config; `clean`/`keep` two-stage policies are a follow-up slice).
            out = try await pipeline.icT2V(prompt: t2v.prompt, references: references,
                                           audioReferences: audioReferences,
                                           height: h, width: wd, numFrames: nf, fps: fps, seed: t2v.seed)
        } else if let initImage = t2v.initImage {
            // i2v: condition on the first frame. Decode + preprocess the image to (1,3,1,H,W),
            // then run the one-stage conditioned path (holds frame 0 clean, denoises the rest).
            let initFrame = try ImageInput.initFrameTensor(initImage, width: wd, height: h)
            out = try await pipeline.i2v(prompt: t2v.prompt, initFrame: initFrame,
                                         height: h, width: wd, numFrames: nf, fps: fps, seed: t2v.seed)
        } else {
            // t2v: two-stage (half-res → upsample → refine) when available AND the tier allows it;
            // one-stage at target resolution otherwise (the low-tier denoise path).
            out = (pipeline.supportsTwoStage && !oneStage)
                ? try await pipeline.t2vTwoStage(prompt: t2v.prompt, height: h, width: wd, numFrames: nf, fps: fps, seed: t2v.seed)
                : try await pipeline.t2v(prompt: t2v.prompt, height: h, width: wd, numFrames: nf, fps: fps, seed: t2v.seed)
        }
        try Task.checkCancellation()
        // LTX decoder is channels-first (1,3,F,H,W); the codec wants channels-last (1,F,H,W,3).
        // out.audio is 48kHz stereo (1,2,T) when the audio components are present → muxed.
        let framesCL = out.video.transposed(0, 2, 3, 4, 1)
        eval(framesCL)                 // materialize pixels before we free the compute cache
        // Memory hygiene: free the MLX buffer pool (grows to ~20 GB during decode) before returning.
        // (The H.264 post-generation stall is fixed by defaulting `encodeMP4` to the SOFTWARE encoder,
        // NOT by this — freeing the cache alone did not un-stall the hardware media engine.)
        Memory.clearCache()
        let mp4Span = MLXProfiler.shared.begin("encode-mp4", "h264+aac", note: "\(framesCL.dim(1)) frames")
        let mp4 = try await encodeMP4(frames: framesCL, fps: fps, audio: out.audio, audioSampleRate: 48000)
        MLXProfiler.shared.end(mp4Span)
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
