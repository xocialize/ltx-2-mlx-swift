// LTX2Pipeline.swift — distilled t2v/i2v: prompt → frames(+audio).
//
// Assembles the parity-validated pieces: GemmaEncoder (49-state extract) →
// Connector (video/audio embeds) → noised latents → DenoiseLoop (distilled
// Euler) → unpatchify → VideoVAEDecoder → pixels, with a jointly-denoised audio
// latent → Audio VAE + vocoder → 48kHz stereo. Single-stage and two-stage
// (half-res → upsample → refine) distilled paths.
//
// MEMORY (per-stage load → use → evict, engine 1.14.0 efficiency sweep):
// only the DiT backbone stays resident for the pipeline lifetime — it is the
// runtime-LoRA hot-swap target AND the denoise peak. The text encoder (Gemma +
// Connector ≈ 19 GB: gemma-3-12b-4bit + the fp32 connector), the VAE decoder
// stack, and the two-stage encoder/upsampler are loaded around the phase that
// uses them and evicted (ref→nil + `Memory.clearCache()`) before the next, so
// they are NEVER co-resident with the DiT denoise activation peak. This is the
// Wan T5 pattern generalized across all transient stages: the declared
// `residentBytes` is the DiT floor, `peakActivationBytes` the denoise transient.

import Foundation
import MLX
import MLXProfiling
import MLXRandom

public final class LTX2Pipeline {
    /// The DiT backbone — resident by default (the runtime-LoRA hot-swap target and the denoise
    /// peak). On low tiers (`evictDiTBeforeDecode`) it is DROPPED after the last denoise step so
    /// the decode stage never carries it (T3c: decode-with-DiT was the residual low-tier peak);
    /// `ensureDiT()` reloads on the next request (kernels stay process-cached — no recompile,
    /// weights re-fault from the mmap page cache). LoRA state lives on the DiT instance, so a
    /// reload yields a PRISTINE base — `activeLoRATargets == 0` is the wrapper's re-apply signal.
    private var ditStorage: DiT?
    private let ditPath: URL
    private var didWarmup = false

    /// Evict the DiT after denoise, before the decode stage (set by the wrapper from the tier
    /// profile; `LTX_EVICT_DIT=1/0` overrides for measurement).
    public var evictDiTBeforeDecode = false

    /// The requested runtime-LoRA set — REMEMBERED across DiT evict/reload cycles (LoRA factors
    /// live on the DiT instance, so `ensureDiT` must re-apply after every reload; without this a
    /// low-tier request would silently generate BASE after the pre-encode DiT drop).
    private var activeLoRASpec: [(url: URL, strength: Float)] = []

    @discardableResult
    func ensureDiT() throws -> DiT {
        if let d = ditStorage { return d }
        let span = MLXProfiler.shared.begin("load", "dit-reload")
        let d = try DiT.load(weightsPath: ditPath, config: DiTConfig(), computeDtype: .bfloat16)
        if !didWarmup { d.warmup(); didWarmup = true }   // kernels are per-process; once is enough
        if !activeLoRASpec.isEmpty { try LTX2LoRA.apply(activeLoRASpec, to: d) }
        MLXProfiler.shared.end(span)
        ditStorage = d
        return d
    }

    /// The low-tier sequential-DiT policy: profile-set, `LTX_EVICT_DIT=1/0` overrides.
    private var sequentialDiT: Bool {
        let env = ProcessInfo.processInfo.environment["LTX_EVICT_DIT"]
        return env == "1" || (env != "0" && evictDiTBeforeDecode)
    }

    /// Drop the DiT on low tiers so a stage that doesn't need it never carries it. Used BOTH
    /// before the connector loads (encode is the T3b-measured peak: connector int8 quantize
    /// scratch + a warmed-resident DiT ≈ 26 GB co-resident) AND after the last denoise step
    /// (decode). `ensureDiT()` reloads in seconds (mmap re-fault; kernels process-cached).
    private func dropDiTIfSequential() {
        guard sequentialDiT, !keepStagesResident, ditStorage != nil else { return }
        ditStorage = nil
        Memory.clearCache()
    }

    // Evictable stages — held only around the phase that needs them (load → use → evict).
    // Stored loaders (`ltxDir`/`gemmaDir`) let a dropped stage re-load on the next request.
    private let ltxDir: URL
    private let gemmaDir: URL
    private var gemma: GemmaEncoder?
    private var connector: Connector?
    private var vae: VideoVAEDecoder?
    private var audioVAE: AudioVAEDecoder?
    private var vocoder: Vocoder?
    private var vaeEncoder: VideoVAEEncoder?   // for two-stage upscale denorm/renorm stats + i2v encode
    private var upsampler: Upsampler?          // spatial_x2

    // File availability, probed once at load — drives `supportsTwoStage` / audio presence
    // WITHOUT holding the components resident (they were `!= nil` checks before).
    private let hasAudio: Bool
    private let hasEncoder: Bool
    private let hasUpsampler: Bool

    /// Keep evictable stages resident after use instead of dropping them (the big-RAM-tier
    /// refinement — trades residency for skipping the per-request reload). Default: evict, so the
    /// denoise peak never carries idle encoder/decoder weights. The wrapper can flip this on a tier
    /// with ample headroom; the footprint math (declared peak = DiT + denoise activation) assumes
    /// the default.
    public var keepStagesResident = false

    init(dit: DiT, ditPath: URL, ltxDir: URL, gemmaDir: URL, hasAudio: Bool, hasEncoder: Bool, hasUpsampler: Bool) {
        self.ditStorage = dit
        self.didWarmup = true      // LTX2Pipeline.load warmed it
        self.ditPath = ditPath
        self.ltxDir = ltxDir
        self.gemmaDir = gemmaDir
        self.hasAudio = hasAudio
        self.hasEncoder = hasEncoder
        self.hasUpsampler = hasUpsampler
    }

    /// True when the two-stage (half-res → upsample → refine) path is available. File-based now
    /// (the encoder/upsampler are loaded on demand, not held resident), so this no longer pins them.
    public var supportsTwoStage: Bool { hasEncoder && hasUpsampler }

    // MARK: - Runtime LoRA (extend, not swap) — public seam for the engine wrapper

    /// Make `loras` the active runtime-LoRA set (replaces any current set; empty array detaches).
    /// Hot-swap on a resident DiT: no base reload, only the small low-rank factors change. If the
    /// DiT is currently evicted (low-tier sequencing), the spec is stored and applied on the next
    /// `ensureDiT()` — no eager reload just to attach factors.
    public func setLoRAs(_ loras: [(url: URL, strength: Float)]) throws {
        activeLoRASpec = loras
        if let d = ditStorage { try LTX2LoRA.apply(loras, to: d) }
    }

    /// Restore the pristine base (drop all runtime LoRAs).
    public func clearLoRAs() {
        activeLoRASpec = []
        if let d = ditStorage { LTX2LoRA.detach(d) }
    }

    /// Number of currently-adapted DiT targets. 0 = pristine base — INCLUDING after a low-tier
    /// DiT evict/reload (LoRA factors live on the DiT instance), so the wrapper re-applies on 0.
    public var activeLoRATargets: Int { ditStorage?.loraTargetCount ?? 0 }

    // MARK: - Per-stage residency (load → use → evict)

    /// Text-encode stage, SEQUENTIAL (LOW-TIER-PLAN T3b lever 1): Gemma and the connector never
    /// co-reside. Gemma loads → tokenize + 49 hidden states → **eval + drop Gemma** → connector
    /// loads → embeds → eval + drop connector. The hidden states are small materialized
    /// (49 × (1,1024,3840) ≈ 0.4 GB bf16), so sequencing cuts ~6.5 GB (Gemma 4-bit) off the
    /// encode-stage peak — which T3 measurement showed is THE peak on low tiers. `isolation`
    /// inherits the caller's actor (the wrapper's `@InferenceActor`).
    private func encodePrompt(
        _ prompt: String, isolation: isolated (any Actor)? = #isolation
    ) async throws -> (video: MLXArray, audio: MLXArray) {
        let prof = MLXProfiler.shared
        // Low tiers: the encode stage never needs the DiT — drop it (warmed at load) BEFORE Gemma,
        // not just before the connector: T3c iteration measured encode/gemma at 21.0 GB with the
        // warmed DiT (11.9) still resident, and the connector's int8 quantize-at-load scratch was
        // the peak before that (~26 GB co-resident). Fully sequential low-tier stages:
        // [Gemma] → [connector] → [DiT denoise] → [VAE decode]. Reloads at denoise (mmap re-fault).
        dropDiTIfSequential()

        // Cancellation checkpoints (MVP-READINESS M2/M3 extension): quit/Cancel during the encode
        // phase stops at the next sub-stage boundary (load → forward → connector) instead of
        // riding the whole encode. The Gemma 49-layer forward itself is one fork call (not
        // checkpointable without a fork API change) — worst case ≈ that forward's few seconds.
        try Task.checkCancellation()
        let gSpan = prof.begin("encode", "gemma", note: "gemma-3-12b 4-bit")
        if gemma == nil { gemma = try await GemmaEncoder.load(directory: gemmaDir) }
        try Task.checkCancellation()
        let (ids, mask) = gemma!.tokenize(prompt)
        let states = try gemma!.allHiddenStates(tokenIds: ids, attentionMask: mask)
        eval(states); eval(mask)   // materialize BEFORE dropping Gemma (lazy graph would pin it)
        prof.end(gSpan)
        if !keepStagesResident { gemma = nil; Memory.clearCache() }

        try Task.checkCancellation()
        let cSpan = prof.begin("encode", "connector")
        if connector == nil {
            connector = try Connector.load(connectorPath: ltxDir.appending(path: "connector.safetensors"))
        }
        let (video, audio) = connector!(hiddenStates: states, mask: mask)
        eval(video, audio)
        prof.end(cSpan)
        if !keepStagesResident { connector = nil; Memory.clearCache() }
        return (video, audio)
    }

    /// Page in the VAE decoder stack (video + optional audio) — deferred until AFTER denoise so it
    /// is never co-resident with the denoise activation peak.
    private func ensureDecoder() throws {
        if vae == nil { vae = try VideoVAEDecoder.load(path: ltxDir.appending(path: "vae_decoder.safetensors")) }
        if hasAudio {
            if audioVAE == nil { audioVAE = try AudioVAEDecoder.load(path: ltxDir.appending(path: "audio_vae.safetensors")) }
            if vocoder == nil { vocoder = try Vocoder.load(path: ltxDir.appending(path: "vocoder.safetensors")) }
        }
    }

    /// LOW-TIER-PLAN T1: decode long clips in temporal chunks so the decode-stage peak is
    /// window-bound, not clip-length-bound. Gate-validated (`--vae-chunk-gate`, 233f @704×512):
    /// whole-frame 67.8 GB vs chunk8/halo5 window-bound; halo 5 = seam PSNR ≥66 dB / cosine 1.000000
    /// (halo 4 → 59 dB, fails the 60 dB bar; exact receptive field is 13.5 latent frames — decayed
    /// influence makes 5 perceptually exact). Engages only when the clip exceeds the chunk window
    /// (below that, whole-frame is strictly cheaper). Env overrides: LTX_VAE_CHUNK / LTX_VAE_HALO
    /// (0 disables chunking).
    /// Decode window in latent frames — set from the tier profile by the wrapper (LOW-TIER-PLAN T3);
    /// `LTX_VAE_CHUNK` env still overrides for experiments.
    public var vaeChunkFrames = 8

    private func decodePixels(_ spatial: MLXArray) throws -> MLXArray {
        let env = ProcessInfo.processInfo.environment
        let chunk = env["LTX_VAE_CHUNK"].flatMap { Int($0) } ?? vaeChunkFrames
        let halo = env["LTX_VAE_HALO"].flatMap { Int($0) } ?? 5
        // T3b lever 3 (the Wan `cacheLimit` lever): the MLX pool retains the decode's conv
        // intermediates — measured +36.7 GB inside ONE 18-frame window @704×512 on top of ~24 GB
        // active. A decode-scoped cap forces buffer reuse; restored after. `LTX_VAE_CACHE_GB`
        // overrides (-1 = leave uncapped).
        let capGB = env["LTX_VAE_CACHE_GB"].flatMap { Int($0) } ?? 0
        let saved = Memory.cacheLimit
        if capGB >= 0 { Memory.cacheLimit = capGB * 1_000_000_000 }
        defer { if capGB >= 0 { Memory.cacheLimit = saved } }
        let fLat = spatial.dim(2)
        guard chunk > 0, fLat > chunk + 2 * halo else { return vae!.decode(spatial) }
        return try vae!.decodeChunked(spatial, chunkFrames: chunk, halo: halo)
    }

    private func dropDecoder() {
        guard !keepStagesResident else { return }
        vae = nil; audioVAE = nil; vocoder = nil
        Memory.clearCache()
    }

    /// Page in the VAE encoder (two-stage denorm/renorm stats; i2v init-frame encode).
    private func ensureVAEEncoder() throws {
        if vaeEncoder == nil { vaeEncoder = try VideoVAEEncoder.load(path: ltxDir.appending(path: "vae_encoder.safetensors")) }
    }

    /// Page in the spatial 2× upsampler (two-stage only).
    private func ensureUpsampler() throws {
        if upsampler == nil { upsampler = try Upsampler.load(path: ltxDir.appending(path: "spatial_upscaler_x2_v1_1.safetensors")) }
    }

    /// Evict the two-stage encoder + upsampler after the upscale step (before the stage-2 denoise
    /// peak). Caller must `eval` the upscaled latent first. Also covers i2v (encoder only).
    private func dropUpscaler() {
        guard !keepStagesResident else { return }
        vaeEncoder = nil; upsampler = nil
        Memory.clearCache()
    }

    /// Output of a t2v run: video pixels (1,3,F,H,W) and optional 48kHz stereo audio (1,2,T).
    public struct Output {
        public let video: MLXArray
        public let audio: MLXArray?
    }

    /// Load the pipeline. Eagerly loads ONLY the persistent DiT backbone (the resident floor + LoRA
    /// target); every transient stage (Gemma/Connector text encoder, VAE decode stack, two-stage
    /// encoder/upsampler) is loaded on demand around its phase and evicted after. `ltxDir` holds
    /// connector/transformer-distilled/vae_decoder (+ optional audio_vae/vocoder/vae_encoder/
    /// upsampler); `gemmaDir` is the Gemma-3 weights dir. Audio decode is enabled when both
    /// audio_vae.safetensors and vocoder.safetensors exist.
    public static func load(ltxDir: URL, gemmaDir: URL, transformerPath: URL? = nil) async throws -> LTX2Pipeline {
        // DIAGNOSTIC LEVER: `LTX_CACHE_LIMIT_GB=N` caps the MLX buffer pool (uncapped by default).
        // An unbounded cache inflates phys_footprint until the OS pages — the suspected cause of the
        // 48f "<10% GPU, 1000s" stall. Set this to test whether capping restores throughput.
        if let g = ProcessInfo.processInfo.environment["LTX_CACHE_LIMIT_GB"], let gb = Int(g) {
            Memory.cacheLimit = gb * 1_000_000_000
            MLXProfiler.shared.note("Memory.cacheLimit set to \(gb) GB (LTX_CACHE_LIMIT_GB)")
        }
        // transformerPath override → quantized checkpoint (q8/q4); DiT auto-detects quant.
        let ditPath = transformerPath ?? ltxDir.appending(path: "transformer-distilled.safetensors")
        // Cold-load cancellation checkpoints (M2/M3 extension): a Cancel/quit during "Loading"
        // stops before/after the two heavy phases (weight load, kernel warmup) instead of
        // waiting the whole load out.
        try Task.checkCancellation()
        let dit = try DiT.load(weightsPath: ditPath, config: DiTConfig(), computeDtype: .bfloat16)
        MLXProfiler.shared.note(String(format: "DiT.load done (lazy) · phys=%.2f GB", Double(physFootprintBytes()) / 1e9))
        try Task.checkCancellation()
        // Pay the one-time Metal kernel-compile cost here (in "Loading"), not on the first denoise
        // step where it idles the GPU and looks like a hang. See DiT.warmup / PROFILING.md.
        let warm = Date(); dit.warmup()
        MLXProfiler.shared.note(String(format: "DiT kernel warmup: %.1fs · phys=%.2f GB · pool=%.2f GB",
                                       Date().timeIntervalSince(warm),
                                       Double(physFootprintBytes()) / 1e9,
                                       Double(Memory.cacheMemory) / 1e9))
        let fm = FileManager.default
        let hasAudio = fm.fileExists(atPath: ltxDir.appending(path: "audio_vae.safetensors").path)
                    && fm.fileExists(atPath: ltxDir.appending(path: "vocoder.safetensors").path)
        let hasEncoder = fm.fileExists(atPath: ltxDir.appending(path: "vae_encoder.safetensors").path)
        let hasUpsampler = fm.fileExists(atPath: ltxDir.appending(path: "spatial_upscaler_x2_v1_1.safetensors").path)
        return LTX2Pipeline(dit: dit, ditPath: ditPath, ltxDir: ltxDir, gemmaDir: gemmaDir,
                            hasAudio: hasAudio, hasEncoder: hasEncoder, hasUpsampler: hasUpsampler)
    }

    /// Flow-matching noised init: noise·σ + clean·(1-σ). Stage-1 (clean=0,σ=1)=noise;
    /// stage-2 starts from the upscaled latent at σ₀.
    static func noiseInit(clean: MLXArray, sigma: Float, shape: [Int], seed: UInt64?) -> MLXArray {
        if let seed { MLXRandom.seed(seed) }
        let noise = MLXRandom.normal(shape)
        return noise * sigma + clean * (1.0 - sigma)
    }

    /// Re-patchify a (1,128,F,H,W) latent → tokens (1, F·H·W, 128).
    public static func patchify(_ latent: MLXArray) -> MLXArray {
        let B = latent.dim(0), C = latent.dim(1), F = latent.dim(2), H = latent.dim(3), W = latent.dim(4)
        return latent.transposed(0, 2, 3, 4, 1).reshaped(B, F * H * W, C)
    }

    /// Decode a denoised audio latent (1, T, 128) → 48kHz stereo waveform (1, 2, T_48k).
    /// AudioPatchifier.unpatchify: (B,T,128) → (B,8,T,16), then Audio VAE → mel → vocoder+BWE.
    /// Requires the decoder stage to be resident (`ensureDecoder()` ran).
    func decodeAudio(_ audioTokens: MLXArray) -> MLXArray? {
        guard let audioVAE, let vocoder else { return nil }
        let B = audioTokens.dim(0), T = audioTokens.dim(1)
        let audioLatent = audioTokens.reshaped(B, T, 8, 16).transposed(0, 2, 1, 3)  // (B,8,T,16)
        let mel = audioVAE.decode(audioLatent)                                       // (B,2,T',64)
        return vocoder(mel)                                                          // (B,2,T_48k)
    }

    /// Text-to-video(+audio). Returns video pixels (1,3,F,H,W) in [-1,1] (channels-first)
    /// and optional 48kHz stereo audio.
    public func t2v(
        prompt: String, height: Int = 256, width: Int = 256, numFrames: Int = 9,
        fps: Double = 24, seed: UInt64? = nil, isolation: isolated (any Actor)? = #isolation
    ) async throws -> Output {
        // 2. Latent geometry
        let fLat = (numFrames + 7) / 8, hLat = height / 32, wLat = width / 32
        let nv = fLat * hLat * wLat
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)
        MLXProfiler.shared.beginRun(String(format:
            "t2v(one-stage) %dx%d %df fps=%.0f | fLat=%d nv=%d audioT=%d | steps=%d",
            width, height, numFrames, fps, fLat, nv, audioT, Positions.distilledSigmas.count - 1))

        // 1. Text encode — sequential Gemma → connector (never co-resident), self-evicting.
        let (videoEmbeds, audioEmbeds) = try await encodePrompt(prompt)

        // 3. Noised init (t2v starts from pure noise at σ_max)
        if let seed { MLXRandom.seed(seed) }
        let videoLatent = MLXRandom.normal([1, nv, 128])
        let audioLatent = MLXRandom.normal([1, audioT, 128])
        let videoPositions = Positions.video(F: fLat, H: hLat, W: wLat, fps: Float(fps))
        let audioPositions = Positions.audio(tokens: audioT)

        // 4. Distilled Euler denoise (joint video + audio) — the memory peak (DiT only resident).
        let (vfinal, afinal) = try DenoiseLoop.run(
            dit: try ensureDiT(), videoLatent0: videoLatent, audioLatent0: audioLatent, sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: videoPositions, audioPositions: audioPositions, label: "")
        eval(vfinal, afinal)
        dropDiTIfSequential()   // low tiers: decode never carries the DiT (T3c)

        // 5. Video: unpatchify → (1, 128, F, H, W) → VAE decode → pixels (decoder loaded now).
        let vspatial = vfinal.reshaped(1, fLat, hLat, wLat, 128).transposed(0, 4, 1, 2, 3)
        try ensureDecoder()
        let decSpan = MLXProfiler.shared.begin("vae-decode", "video", note: "\(numFrames)f")
        let pixels = try decodePixels(vspatial)
        // 6. Audio: decode the jointly-denoised audio latent (if audio components loaded)
        let waveform = decodeAudio(afinal)
        eval(pixels); if let waveform { eval(waveform) }
        MLXProfiler.shared.end(decSpan)
        dropDecoder()
        MLXProfiler.shared.endRun()
        return Output(video: pixels, audio: waveform)
    }

    /// Image-to-video (first-frame conditioning, frame_idx=0). `initFrame` is (1,3,1,H,W) in
    /// [-1,1] at the target resolution. VAE-encodes it to the frame-0 latent and holds those
    /// tokens clean — per-token timestep 0 in the DiT + a per-step clean re-blend — while the
    /// rest denoise from noise. `strength` 1.0 = fully condition on the frame; <1.0 lets the
    /// frame be partially re-noised (denoise_mask = 1-strength). One-stage distilled; requires
    /// the VAE encoder (falls back to t2v if absent). Keyframe (frame_idx>0) + two-stage i2v
    /// are follow-ups.
    public func i2v(
        prompt: String, initFrame: MLXArray, height: Int = 512, width: Int = 704, numFrames: Int = 9,
        fps: Double = 24, seed: UInt64? = nil, strength: Float = 1.0,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> Output {
        guard hasEncoder else {
            return try await t2v(prompt: prompt, height: height, width: width, numFrames: numFrames, fps: fps, seed: seed)
        }
        // 1. Text encode — sequential Gemma → connector (never co-resident), self-evicting.
        let (videoEmbeds, audioEmbeds) = try await encodePrompt(prompt)

        // 2. Latent geometry
        let fLat = (numFrames + 7) / 8, hLat = height / 32, wLat = width / 32
        let frame0 = hLat * wLat, nv = fLat * frame0
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)

        // 3. Encode the init frame → frame-0 clean latent tokens (1, frame0, 128), normalized.
        try ensureVAEEncoder()
        let refLatent = vaeEncoder!.encode(initFrame)           // (1,128,1,hLat,wLat)
        let refTokens = LTX2Pipeline.patchify(refLatent)        // (1, frame0, 128)
        eval(refTokens)
        dropUpscaler()                                          // drop the VAE encoder before denoise

        // 4. Full-size clean latent (frame 0 = ref, rest 0) + denoise mask (1-strength at frame 0).
        let cleanVideo = MLX.concatenated([refTokens, MLXArray.zeros([1, nv - frame0, 128])], axis: 1)
        let maskHead = MLXArray.zeros([1, frame0, 1]) + (1.0 - strength)   // conditioned tokens
        let videoMask = MLX.concatenated([maskHead, MLXArray.ones([1, nv - frame0, 1])], axis: 1)

        // 5. Noised init + conditioned denoise (audio is unconditioned → scalar path).
        if let seed { MLXRandom.seed(seed) }
        let videoLatent = MLXRandom.normal([1, nv, 128])
        let audioLatent = MLXRandom.normal([1, audioT, 128])
        let (vfinal, afinal) = try DenoiseLoop.runConditioned(
            dit: try ensureDiT(), videoLatent0: videoLatent, audioLatent0: audioLatent, sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: Positions.video(F: fLat, H: hLat, W: wLat, fps: Float(fps)),
            audioPositions: Positions.audio(tokens: audioT),
            videoCleanLatent: cleanVideo, videoDenoiseMask: videoMask)
        eval(vfinal, afinal)
        dropDiTIfSequential()   // low tiers: decode never carries the DiT (T3c)

        // 6. Decode
        let vspatial = vfinal.reshaped(1, fLat, hLat, wLat, 128).transposed(0, 4, 1, 2, 3)
        try ensureDecoder()
        let pixels = try decodePixels(vspatial)
        let waveform = decodeAudio(afinal)
        eval(pixels); if let waveform { eval(waveform) }
        dropDecoder()
        return Output(video: pixels, audio: waveform)
    }

    /// VAE-encode a reference video (1,3,F,H,W) in [-1,1], F = 8k+1, at the reference's OWN
    /// resolution → `ReferenceConditioning` (tokens + positions at the ref grid). The IC-LoRA
    /// ingest path (IC-LORA-PLAN P2): the encoder loads around this call and drops immediately —
    /// never co-resident with Gemma or the DiT.
    public func encodeReference(
        pixels: MLXArray, fps: Double, downscaleFactor: Float = 1, strength: Float = 1.0
    ) throws -> ReferenceConditioning {
        try ensureVAEEncoder()
        let span = MLXProfiler.shared.begin("ic-ingest", "vae-encode-ref",
            note: "\(pixels.dim(2))f \(pixels.dim(4))x\(pixels.dim(3))")
        let latent = vaeEncoder!.encode(pixels)          // (1,128,fLat,hLat,wLat), normalized
        let tokens = LTX2Pipeline.patchify(latent)       // (1, Nr, 128)
        eval(tokens)
        MLXProfiler.shared.end(span)
        let positions = Positions.video(F: latent.dim(2), H: latent.dim(3), W: latent.dim(4),
                                        fps: Float(fps))
        dropUpscaler()                                   // drop the encoder before denoise
        return ReferenceConditioning(tokens: tokens, positions: positions,
                                     downscaleFactor: downscaleFactor, strength: strength)
    }

    /// IC-LoRA conditioned t2v — ONE stage at TARGET resolution (the `stage2: skip` policy the
    /// community reference usage blesses for Ingredients; the adapter's LoRA stays applied for
    /// the whole generation). References are appended per the parity-gated P1 path
    /// (`ICVideoState`): clean (1−strength)-masked tokens + scaled positions, per-token σ in the
    /// denoise, sliced off before decode. Empty `references` falls through to plain `t2v`.
    public func icT2V(
        prompt: String, references: [ReferenceConditioning],
        height: Int = 448, width: Int = 704, numFrames: Int = 121,
        fps: Double = 24, seed: UInt64? = nil, isolation: isolated (any Actor)? = #isolation
    ) async throws -> Output {
        guard !references.isEmpty else {
            return try await t2v(prompt: prompt, height: height, width: width,
                                 numFrames: numFrames, fps: fps, seed: seed)
        }
        let fLat = (numFrames + 7) / 8, hLat = height / 32, wLat = width / 32
        let nv = fLat * hLat * wLat
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)
        let nr = references.reduce(0) { $0 + $1.tokens.dim(1) }
        MLXProfiler.shared.beginRun(String(format:
            "t2v(ic one-stage) %dx%d %df fps=%.0f | nv=%d +ref=%d audioT=%d | steps=%d",
            width, height, numFrames, fps, nv, nr, audioT, Positions.distilledSigmas.count - 1))

        let (videoEmbeds, audioEmbeds) = try await encodePrompt(prompt)

        if let seed { MLXRandom.seed(seed) }
        let videoLatent = MLXRandom.normal([1, nv, 128])
        let audioLatent = MLXRandom.normal([1, audioT, 128])
        let state = ICVideoState.build(
            targetLatent: videoLatent,
            targetPositions: Positions.video(F: fLat, H: hLat, W: wLat, fps: Float(fps)),
            references: references)

        let (vfull, afinal) = try DenoiseLoop.runConditioned(
            dit: try ensureDiT(), videoLatent0: state.latent, audioLatent0: audioLatent,
            sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: state.positions, audioPositions: Positions.audio(tokens: audioT),
            videoCleanLatent: state.clean, videoDenoiseMask: state.denoiseMask, label: "ic-")
        let vfinal = state.slice(vfull)
        eval(vfinal, afinal)
        dropDiTIfSequential()

        let vspatial = vfinal.reshaped(1, fLat, hLat, wLat, 128).transposed(0, 4, 1, 2, 3)
        try ensureDecoder()
        let decSpan = MLXProfiler.shared.begin("vae-decode", "video", note: "\(numFrames)f")
        let pixels = try decodePixels(vspatial)
        let waveform = decodeAudio(afinal)
        eval(pixels); if let waveform { eval(waveform) }
        MLXProfiler.shared.end(decSpan)
        dropDecoder()
        MLXProfiler.shared.endRun()
        return Output(video: pixels, audio: waveform)
    }

    /// Two-stage distilled t2v (the `generate --distilled` flow): stage-1 denoise at
    /// HALF resolution → unpatchify → encoder-stats denorm → upsampler 2× → renorm →
    /// re-patchify → stage-2 refine at full resolution. Requires `supportsTwoStage`.
    public func t2vTwoStage(
        prompt: String, height: Int = 512, width: Int = 704, numFrames: Int = 9,
        fps: Double = 24, seed: UInt64? = nil, isolation: isolated (any Actor)? = #isolation
    ) async throws -> Output {
        guard hasEncoder, hasUpsampler else {
            return try await t2v(prompt: prompt, height: height, width: width, numFrames: numFrames, fps: fps, seed: seed)
        }

        let fLat = (numFrames + 7) / 8
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)
        let s2 = Positions.stage2Sigmas
        let sigma0 = s2[0]
        let nv2 = fLat * (height / 32) * (width / 32)          // stage-2 (full-res) video token count
        let nv1p = fLat * (height / 2 / 32) * (width / 2 / 32) // stage-1 (half-res) token count
        MLXProfiler.shared.beginRun(String(format:
            "t2vTwoStage %dx%d %df fps=%.0f | fLat=%d nv1=%d nv2=%d audioT=%d | steps s1=%d s2=%d",
            width, height, numFrames, fps, fLat, nv1p, nv2, audioT,
            Positions.distilledSigmas.count - 1, s2.count - 1))

        // 1. Text encode — sequential Gemma → connector (never co-resident), self-evicting.
        let (videoEmbeds, audioEmbeds) = try await encodePrompt(prompt)

        // --- Stage 1: half resolution (denoise peak #1, DiT only resident) ---
        let hHalf = height / 2, wHalf = width / 2
        let hLat1 = hHalf / 32, wLat1 = wHalf / 32, nv1 = fLat * hLat1 * wLat1
        let v1 = LTX2Pipeline.noiseInit(clean: MLXArray.zeros([1, nv1, 128]), sigma: 1.0, shape: [1, nv1, 128], seed: seed)
        let a1 = LTX2Pipeline.noiseInit(clean: MLXArray.zeros([1, audioT, 128]), sigma: 1.0, shape: [1, audioT, 128], seed: seed.map { $0 &+ 1 })
        let (v1f, a1f) = try DenoiseLoop.run(
            dit: try ensureDiT(), videoLatent0: v1, audioLatent0: a1, sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: Positions.video(F: fLat, H: hLat1, W: wLat1, fps: Float(fps)),
            audioPositions: Positions.audio(tokens: audioT), label: "s1-")
        eval(v1f, a1f)

        // --- Upscale 2× in un-normalized latent space (encoder+upsampler loaded only here) ---
        let upSpan = MLXProfiler.shared.begin("upscale", "vae-enc+upsampler", note: "half→full latent")
        try ensureVAEEncoder(); try ensureUpsampler()
        let v1spatial = v1f.reshaped(1, fLat, hLat1, wLat1, 128).transposed(0, 4, 1, 2, 3)  // (1,128,F,h1,w1)
        let upscaled = vaeEncoder!.normalizeLatent(upsampler!(vaeEncoder!.denormalizeLatent(v1spatial)))  // (1,128,F,2h1,2w1)
        eval(upscaled)
        MLXProfiler.shared.end(upSpan)
        dropUpscaler()                                          // evict before the stage-2 denoise peak
        let hLat2 = hLat1 * 2, wLat2 = wLat1 * 2
        let v2tokens = LTX2Pipeline.patchify(upscaled)  // (1, F*2h1*2w1, 128)

        // --- Stage 2: full resolution refine (init = noise·σ₀ + upscaled·(1-σ₀)) ---
        let v2init = LTX2Pipeline.noiseInit(clean: v2tokens, sigma: sigma0, shape: v2tokens.shape, seed: seed.map { $0 &+ 2 })
        let a2init = LTX2Pipeline.noiseInit(clean: a1f, sigma: sigma0, shape: a1f.shape, seed: seed.map { $0 &+ 2 })
        let (v2f, a2f) = try DenoiseLoop.run(
            dit: try ensureDiT(), videoLatent0: v2init, audioLatent0: a2init, sigmas: s2,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: Positions.video(F: fLat, H: hLat2, W: wLat2, fps: Float(fps)),
            audioPositions: Positions.audio(tokens: audioT), label: "s2-")
        eval(v2f, a2f)
        dropDiTIfSequential()   // low tiers: decode never carries the DiT (T3c)

        let vspatial = v2f.reshaped(1, fLat, hLat2, wLat2, 128).transposed(0, 4, 1, 2, 3)
        try ensureDecoder()
        let decSpan = MLXProfiler.shared.begin("vae-decode", "video", note: "\(fLat*8-7)f full-res")
        let pixels = try decodePixels(vspatial)
        eval(pixels)
        MLXProfiler.shared.end(decSpan)
        let audSpan = MLXProfiler.shared.begin("audio-decode", "audioVAE+vocoder")
        let waveform = decodeAudio(a2f)
        if let waveform { eval(waveform) }
        MLXProfiler.shared.end(audSpan)
        dropDecoder()
        MLXProfiler.shared.endRun()
        return Output(video: pixels, audio: waveform)
    }
}
