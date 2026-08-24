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

    /// HV2 weight streaming (STREAMING-PLAN.md): when set, `ensureDiT` builds the DiT with
    /// slot-backed blocks refilled from this granule tree instead of loading them resident
    /// (set by the wrapper from `LTX2Configuration.resolvedGranuleDirectory`, or
    /// `LTX_STREAM_GRANULES=<dir>` overrides for measurement). The self-calibrating gate
    /// falls back resident automatically when the arithmetic doesn't clear; each entry point
    /// arms `gateEvaluationThresholdTokens` with its LARGEST stage's token count so a
    /// half-res stage-1 forward never condemns a run whose stage 2 would stream.
    public var streamingGranuleDirectory: URL?
    /// Streaming knobs (group size, gate policy/margin) — wrapper- or harness-set.
    public var streamingOptions = BlockStreamingOptions()

    private var effectiveGranuleDirectory: URL? {
        if let env = ProcessInfo.processInfo.environment["LTX_STREAM_GRANULES"] {
            return env.isEmpty ? nil : URL(fileURLWithPath: env)
        }
        return streamingGranuleDirectory
    }

    /// `LTX_STREAM_GATE=force` pins the streamer to `.forceStream` — a MEASUREMENT affordance in
    /// the same style as `LTX_STREAM_GRANULES` above, not a shipping default.
    ///
    /// 🔑 **Why it exists (2026-08-19, AB-R-0105).** The `.auto` gate compares measured IO (S)
    /// against measured compute (C(N)) and streams only when IO can outrun compute. At SMALL N,
    /// C(N) is noisy — at `compact24`'s N=2430 it read 3.34 / 7.43 / 3.52 GiB/s across three runs,
    /// flipping the verdict and the PEAK between 14.6 GB (streamed) and 33.6 GB (fell back
    /// resident). A tier cannot be declared on a footprint that depends on a coin flip, so the
    /// question "what does compact24 cost if streaming is GUARANTEED" needs asking directly.
    /// ⚠️ Forcing it where C(N) > S means the refill genuinely cannot keep up: expect real stall.
    /// That is the trade being measured, not a bug.
    private var effectiveStreamingOptions: BlockStreamingOptions {
        var o = streamingOptions
        if ProcessInfo.processInfo.environment["LTX_STREAM_GATE"] == "force" {
            o.gatePolicy = .forceStream
        }
        return o
    }

    /// Arm the streamer's gate threshold for the request being run (no-op when resident).
    ///
    /// 🚨 Under modality tiling this MUST be the largest **per-forward** count, not the untiled
    /// total. The kit measures tokens inside `beginForward` and only evaluates the gate when
    /// `forwardTokens >= gateEvaluationThresholdTokens`; with tiling every forward carries
    /// ~N/k tokens, so arming with the untiled N means the threshold is never reached, the verdict
    /// stays `.undecided` forever, and the run streams unconditionally however bad the arithmetic.
    /// Callers pass the tiler's `largestTileTokenCount` when a tiler is installed.
    /// ⚠️ Note the real coupling this exposes: the gate's own `N_min ≈ N·C/S` is derived from the
    /// same N, so **tiling genuinely makes the streaming gate harder to clear** — the two memory
    /// levers partly cancel rather than compose freely (TILING-PLAN hazard 1).
    func armStreamingGate(largestStageTokens: Int) {
        ditStorage?.blockStreamer?.gateEvaluationThresholdTokens = largestStageTokens
    }

    /// Token-grid tiling for the DiT forward. `nil`/identity ⇒ the plain resident path, byte for
    /// byte. Env overrides mirror the oracle's CLI: `LTX_TILE_FRAMES` (`--tile-frames`),
    /// `LTX_TILE_SPATIAL` (`--tile-spatial`), `LTX_TILE_OVERLAP` (`--tile-overlap`, default 2).
    public var modalityTiling: TileCountConfig?

    private var effectiveModalityTiling: TileCountConfig? {
        let env = ProcessInfo.processInfo.environment
        let f = env["LTX_TILE_FRAMES"].flatMap { Int($0) }
        let s = env["LTX_TILE_SPATIAL"].flatMap { Int($0) }
        if f != nil || s != nil {
            let ov = env["LTX_TILE_OVERLAP"].flatMap { Int($0) } ?? 2
            let cfg = TileCountConfig(tileFrames: f ?? 1, tileSpatial: s ?? 1, overlap: ov)
            return cfg.isIdentity ? nil : cfg
        }
        guard let cfg = modalityTiling, !cfg.isIdentity else { return nil }
        return cfg
    }

    /// Wrap `dit` in a tiler when tiling is configured, and re-arm the streaming gate with the
    /// per-forward token count. Returns the plain DiT (and arms with `untiledTokens`) otherwise.
    private func denoiser(_ dit: DiT, F: Int, H: Int, W: Int,
                          positions: MLXArray, untiledTokens: Int, audioTokens: Int) -> any LTXDenoiser {
        guard let cfg = effectiveModalityTiling else {
            armStreamingGate(largestStageTokens: untiledTokens)
            return dit
        }
        do {
            let tiler = try VideoModalityTiler(tiling: cfg, latentShape: (F: F, H: H, W: W),
                                               positions: positions)
            armStreamingGate(largestStageTokens: tiler.largestTileTokenCount + audioTokens)
            MLXProfiler.shared.note(
                "modality tiling \(cfg.frames.numTiles)×\(cfg.height.numTiles)×\(cfg.width.numTiles)"
                + " overlap=\(cfg.frames.overlap) → \(tiler.tiles.count) tiles,"
                + " largest \(tiler.largestTileTokenCount) video tokens (untiled \(untiledTokens - audioTokens))")
            return TiledDiT(inner: dit, tiler: tiler)
        } catch {
            // A tiling config the grid cannot honour must not silently change the output —
            // fall back to the untiled path loudly, exactly as the streaming gate does.
            MLXProfiler.shared.note("modality tiling DISABLED (\(error)) — running untiled")
            armStreamingGate(largestStageTokens: untiledTokens)
            return dit
        }
    }

    /// Stop the refill thread between the denoise and decode phases (slots and bindings
    /// stay; the next request's first forward re-activates IO).
    func quiesceStreaming() {
        ditStorage?.blockStreamer?.finish()
    }

    /// Pack runtime-LoRA factors to int8/int4 at apply (set by the wrapper from the tier profile;
    /// `LTX_LORA_QUANT=8|4|0` overrides for measurement). nil = full-precision factors.
    public var loraFactorQuantBits: Int?

    /// The effective factor-packing choice: env override wins, else the profile-set knob.
    private var effectiveLoRAQuantBits: Int? {
        if let env = ProcessInfo.processInfo.environment["LTX_LORA_QUANT"], let v = Int(env) {
            return v == 8 || v == 4 ? v : nil
        }
        return loraFactorQuantBits
    }

    /// The requested runtime-LoRA set — REMEMBERED across DiT evict/reload cycles (LoRA factors
    /// live on the DiT instance, so `ensureDiT` must re-apply after every reload; without this a
    /// low-tier request would silently generate BASE after the pre-encode DiT drop).
    private var activeLoRASpec: [(url: URL, strength: Float)] = []

    /// Build the DiT in whichever regime applies. ONE construction site, shared by `load()` and
    /// `ensureDiT()`, so a streamed deployment cannot accidentally take the resident path in one
    /// of them — which is exactly what used to happen: `load()` always built a RESIDENT DiT (and
    /// warmed it), eviction then dropped it, and only the reload came back streamed. That made the
    /// checkpoint mandatory even for a fully streamed run, and it is the reload the resident path
    /// is measured paying.
    static func makeDiT(ditPath: URL, granuleDir: URL?,
                        options: BlockStreamingOptions) throws -> DiT {
        guard let granuleDir else {
            return try DiT.load(weightsPath: ditPath, config: DiTConfig(), computeDtype: .bfloat16)
        }
        // ⚠️ `checkpoint:` may point at a file that does not exist — that is the HOSTED v2 case,
        // not an error. `bindStore` serves the globals from `globals.granule` and takes integrity
        // from the manifest hashes; it throws only if the checkpoint is absent AND the tree is v1.
        let streamer = try LTXBlockStreamer(granuleDir: granuleDir, options: options)
        return try DiT(streaming: streamer, checkpoint: ditPath, config: DiTConfig(),
                       computeDtype: .bfloat16)
    }

    @discardableResult
    func ensureDiT() throws -> DiT {
        if let d = ditStorage { return d }
        let span = MLXProfiler.shared.begin("load", "dit-reload")
        let d: DiT
        if let granuleDir = effectiveGranuleDirectory {
            // Streamed blocks: slots + globals only. A reload after an evict
            // re-binds fresh slots (the granule tree is the weight source; the
            // checkpoint provenance-gates it and provides the 58 globals).
            let streamer = try LTXBlockStreamer(granuleDir: granuleDir, options: effectiveStreamingOptions)
            d = try DiT(
                streaming: streamer, checkpoint: ditPath, config: DiTConfig(),
                computeDtype: .bfloat16)
        } else {
            d = try DiT.load(weightsPath: ditPath, config: DiTConfig(), computeDtype: .bfloat16)
        }
        if !didWarmup { d.warmup(); didWarmup = true }   // kernels are per-process; once is enough
        // (streamed: the warmup sweep doubles as S calibration, gate suspended inside warmup)
        if !activeLoRASpec.isEmpty { try LTX2LoRA.apply(activeLoRASpec, to: d, factorQuantBits: effectiveLoRAQuantBits) }
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
    func dropDiTIfSequential() {
        guard sequentialDiT, !keepStagesResident, ditStorage != nil else { return }
        ditStorage = nil
        Memory.clearCache()
    }

    // Evictable stages — held only around the phase that needs them (load → use → evict).
    // Stored loaders (`ltxDir`/`gemmaDir`) let a dropped stage re-load on the next request.
    let ltxDir: URL
    private let gemmaDir: URL
    /// Override for the video VAE decoder file (nil = stock `vae_decoder.safetensors`).
    /// A pruned sibling such as PrunaVAED loads through the same `VideoVAEDecoder`, which
    /// derives every width from the weights and enables its channel adapters on sight.
    private let vaeDecoderPath: URL?
    private var gemma: GemmaEncoder?
    private var gemma4: Gemma4Encoder?
    private var connector: Connector?
    private var vae: VideoVAEDecoder?
    private var audioVAE: AudioVAEDecoder?
    private var vocoder: Vocoder?
    var vaeEncoder: VideoVAEEncoder?   // for two-stage upscale denorm/renorm stats + i2v encode
    var upsampler: Upsampler?          // any variant; selected via `upsamplerFile`
    /// Temporal x2 latent upsampler — DFR rounds only, in its own slot because a round needs it
    /// while `upsampler` still holds the SPATIAL checkpoint stage 2 used.
    var temporalUpsampler: Upsampler?

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

    init(dit: DiT, ditPath: URL, ltxDir: URL, gemmaDir: URL, hasAudio: Bool, hasEncoder: Bool,
         hasUpsampler: Bool, vaeDecoderPath: URL? = nil) {
        self.ditStorage = dit
        self.didWarmup = true      // LTX2Pipeline.load warmed it
        self.ditPath = ditPath
        self.ltxDir = ltxDir
        self.gemmaDir = gemmaDir
        self.vaeDecoderPath = vaeDecoderPath
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
        if let d = ditStorage { try LTX2LoRA.apply(loras, to: d, factorQuantBits: effectiveLoRAQuantBits) }
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
    /// Which LTX generation this model directory is.
    ///
    /// Detected from the CHECKPOINT, mirroring the oracle's `resolve_text_encoder`: LTX-2.5
    /// conversions ship the tuned Gemma-4 encoder in-dir as `gemma4-12b-ltx-v1/`, and 2.3 uses
    /// an external Gemma-3. Deliberately not a path-name parse — a renamed or local copy must
    /// still be recognised (the oracle had exactly that bug: slots were created on the
    /// checkpoint capability but marked only when the directory name parsed as >= 2.5).
    var isLTX25: Bool { Self.isLTX25(ltxDir: ltxDir) }

    var gemma4Dir: URL { Self.gemma4Dir(ltxDir: ltxDir) }

    /// Static form so the wiring can be gated without loading a 38 GB checkpoint.
    public static func gemma4Dir(ltxDir: URL) -> URL { ltxDir.appending(path: "gemma4-12b-ltx-v1") }
    public static func isLTX25(ltxDir: URL) -> Bool {
        var isDir: ObjCBool = false
        let ok = FileManager.default.fileExists(atPath: gemma4Dir(ltxDir: ltxDir).path, isDirectory: &isDir)
        return ok && isDir.boolValue
    }

    /// (B, N, 1) mask marking the target's FIRST latent frame.
    ///
    /// On 2.5 the first latent frame is always marked: the causal video encoder makes it cover
    /// a single pixel frame while the rest cover 8, so it IS a keyframe and receives the
    /// trained `keyframes_abs_pos_embedding`. Slots (when present) add their own marks on top.
    public static func firstLatentFrameKeyframesMask(
        totalTokens: Int, tokensPerLatentFrame: Int, batch: Int = 1
    ) -> MLXArray {
        let head = MLXArray.ones([batch, Swift.min(tokensPerLatentFrame, totalTokens), 1])
        if tokensPerLatentFrame >= totalTokens { return head.asType(.bfloat16) }
        let tail = MLXArray.zeros([batch, totalTokens - tokensPerLatentFrame, 1])
        return concatenated([head, tail], axis: 1).asType(.bfloat16)
    }

    func encodePrompt(
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
        LTX2Progress.report(.encode)
        let gSpan = prof.begin("encode", "gemma",
                               note: isLTX25 ? "gemma4-12b-ltx-v1 bf16" : "gemma-3-12b 4-bit")
        let ids: MLXArray, mask: MLXArray, states: [MLXArray]
        if isLTX25 {
            // ⚠️ The 2.5 tokenizer emits NO <bos> (post_processor `single: [Sequence A]`),
            // where Gemma-3's does — Gemma4Encoder.tokenize prepends it and pads with
            // pad_token_id. Using GemmaEncoder here would silently feed BOS-less input.
            if gemma4 == nil { gemma4 = try await Gemma4Encoder.load(directory: gemma4Dir) }
            try Task.checkCancellation()
            let cfg = try Gemma4Encoder.specialTokenIds(directory: gemma4Dir)
            (ids, mask) = Gemma4Encoder.tokenize(prompt, tokenizer: gemma4!.context.tokenizer,
                                                 bosId: cfg.bos, padId: cfg.pad)
            states = try gemma4!.allHiddenStates(tokenIds: ids, attentionMask: mask)
        } else {
            if gemma == nil { gemma = try await GemmaEncoder.load(directory: gemmaDir) }
            try Task.checkCancellation()
            (ids, mask) = gemma!.tokenize(prompt)
            states = try gemma!.allHiddenStates(tokenIds: ids, attentionMask: mask)
        }
        eval(states); eval(mask)   // materialize BEFORE dropping Gemma (lazy graph would pin it)
        prof.end(gSpan)
        if !keepStagesResident { gemma = nil; gemma4 = nil; Memory.clearCache() }

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
    func ensureDecoder() throws {
        if vae == nil {
            vae = try VideoVAEDecoder.load(
                path: vaeDecoderPath ?? ltxDir.appending(path: "vae_decoder.safetensors"))
        }
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

    /// GAP-ANALYSIS #8 — the streamed decode→mux handoff. When passed to `t2v`/`t2vTwoStage`:
    /// audio is decoded FIRST and delivered via `onAudioReady` (so an MP4 writer can finish its
    /// audio track before any frame — the interleave-deadlock ordering), then video pixels stream
    /// to `onVideoChunk` per decode chunk as channels-first `(1, 3, f, H, W)`, and the WHOLE pixel
    /// volume never materializes. `Output.video` is then an EMPTY placeholder (`dim(2) == 0`) —
    /// callers that stream must not read it.
    public struct StreamingSinks {
        public let onAudioReady: (MLXArray?) throws -> Void
        public let onVideoChunk: (MLXArray) throws -> Void
        public init(onAudioReady: @escaping (MLXArray?) throws -> Void,
                    onVideoChunk: @escaping (MLXArray) throws -> Void) {
            self.onAudioReady = onAudioReady
            self.onVideoChunk = onVideoChunk
        }
    }

    /// Whether this pipeline decodes an audio track (audio_vae + vocoder present) — the wrapper
    /// needs it BEFORE running, to declare the writer's audio input up front.
    public var audioEnabled: Bool { hasAudio }

    func decodePixels(_ spatial: MLXArray) throws -> MLXArray {
        var parts: [MLXArray] = []
        try decodePixels(spatial) { parts.append($0) }
        return parts.count == 1 ? parts[0] : MLX.concatenated(parts, axis: 2)
    }

    /// Sink form — one chunk-policy code path for both lanes (the VideoVAE pattern).
    func decodePixels(_ spatial: MLXArray, sink: (MLXArray) throws -> Void) throws {
        // Whole-clip decode reports once here; the chunked path refines with per-chunk
        // step/totalSteps from inside `decodeChunked`.
        LTX2Progress.report(.decode)
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
        // Spatial halo+crop tiling (BLOCKSTREAM-EXPANSION-EVAL §3, `--vae-tile-gate`): 4K-only by
        // arithmetic — at ≤720p the tile windows span the whole grid, so default OFF; opt in via
        // LTX_VAE_TILES ("2" → 2×2, "2x3" → tilesH×tilesW) + LTX_VAE_SHALO (spatial halo, default 5
        // ≈ 74 dB seam; 16 = bit-exact). Composes inside each temporal chunk (outer temporal,
        // inner spatial).
        let sHalo = env["LTX_VAE_SHALO"].flatMap { Int($0) } ?? 5

        // 🔑 AUTO-TILE FROM THE LATENT GRID (2026-08-23, AB-D-0041/AB-R-0130). Tiling is no longer
        // caller-opt-in: at geometries where it pays it is GUARANTEED, which is what makes the
        // tiled footprint DECLARABLE. Exactly the rule streaming already follows — compact24 may
        // declare streamed numbers only because `.forceStream` is pinned (AB-R-0107). An opt-in
        // lever cannot back a declaration: it fails OPEN the moment a caller does not use it.
        //
        // The threshold is the halo arithmetic, not a tuned constant. A 2x2 split replaces one
        // full-grid decode with four windows of (axis/2 + 2*halo); that is only smaller than the
        // grid when `axis > 4*halo`. Below it the window MEETS OR EXCEEDS the grid and tiling does
        // strictly more work for nothing — at 704x512 (grid 22x16) the window would be 21x18.
        //
        //   704x512   grid 22x16 -> no-op (16 <= 20)
        //   1280x704  grid 40x22 -> 2x2   (measured 42.03 -> 33.55 GB)
        //   1920x1088 grid 60x34 -> 2x2   (measured 93.68 -> 49.30 GB)
        //
        // ⚠️ 2x2 ONLY, deliberately. 3x3/4x4 measured lower (33.84 / 27.56) but 2x2 already puts
        // every geometry comfortably inside budget, and higher counts add seams — 2 internal seam
        // lines vs 4 vs 6 — where **only 2x2 has been perceptually cleared** (AB-D-0041). Halo
        // re-decode also grows: total decoded area is 2.1x the grid at 2x2, 3.5x at 4x4.
        // ⚠️ `LTX_VAE_TILES` remains an override for measurement, in BOTH directions ("1" disables).
        let gridH = spatial.dim(3), gridW = spatial.dim(4)
        let autoTile = (gridW > 4 * sHalo && gridH > 4 * sHalo) ? 2 : 1
        let tileSpec = (env["LTX_VAE_TILES"] ?? String(autoTile))
            .split(separator: "x").compactMap { Int($0) }
        let tilesH = tileSpec.first ?? autoTile
        let tilesW = tileSpec.count > 1 ? tileSpec[1] : tilesH
        let fLat = spatial.dim(2)
        guard chunk > 0, fLat > chunk + 2 * halo else {
            let px = vae!.decodeSpatialTiled(spatial, tilesH: tilesH, tilesW: tilesW, halo: sHalo)
            try sink(px)
            return
        }
        try vae!.decodeChunked(spatial, chunkFrames: chunk, halo: halo,
                               spatialTilesH: tilesH, spatialTilesW: tilesW, spatialHalo: sHalo,
                               sink: sink)
    }

    func dropDecoder() {
        guard !keepStagesResident else { return }
        vae = nil; audioVAE = nil; vocoder = nil
        Memory.clearCache()
    }

    /// Page in the VAE encoder (two-stage denorm/renorm stats; i2v init-frame encode).
    func ensureVAEEncoder() throws {
        if vaeEncoder == nil { vaeEncoder = try VideoVAEEncoder.load(path: ltxDir.appending(path: "vae_encoder.safetensors")) }
    }

    /// Two-stage upsampler checkpoint: a file name inside `ltxDir` or an absolute path.
    /// Default = the x2 checkpoint every upstream two-stage pipeline hardcodes; the x1.5 and
    /// temporal variants are OUR two-stage composition (neither the oracle nor upstream
    /// `ltx-pipelines` ships a consumer for them — verified 2026-08-05 against both repos),
    /// built on the three individually-gated checkpoints (`--upsampler-variants-gate`,
    /// cosine 1.000000). `LTX_UPSAMPLER=<name|abs path>` overrides for CLI A/Bs, mirroring
    /// the `LTX_VAE_DECODER` idiom.
    public var upsamplerFile: String = LTX2Pipeline.defaultUpsamplerFile()
    private var loadedUpsamplerFile: String?

    static func defaultUpsamplerFile() -> String {
        let raw = (ProcessInfo.processInfo.environment["LTX_UPSAMPLER"] ?? "")
            .trimmingCharacters(in: .whitespaces)
        return raw.isEmpty ? "spatial_upscaler_x2_v1_1.safetensors" : raw
    }

    /// The selected upsampler checkpoint as a concrete URL.
    var upsamplerURL: URL {
        upsamplerFile.contains("/")
            ? URL(fileURLWithPath: (upsamplerFile as NSString).expandingTildeInPath)
            : ltxDir.appending(path: upsamplerFile)
    }

    /// Page in the selected latent upsampler (two-stage only). Reloads when the selection
    /// changed since the last load (keepStagesResident would otherwise pin the old variant).
    func ensureUpsampler() throws {
        if upsampler == nil || loadedUpsamplerFile != upsamplerFile {
            guard FileManager.default.fileExists(atPath: upsamplerURL.path) else {
                throw TwoStageError.upsamplerMissing(upsamplerURL.path)
            }
            upsampler = try Upsampler.load(path: upsamplerURL)
            loadedUpsamplerFile = upsamplerFile
        }
    }

    /// Evict the two-stage encoder + upsampler after the upscale step (before the stage-2 denoise
    /// peak). Caller must `eval` the upscaled latent first. Also covers i2v (encoder only).
    func dropUpscaler() {
        guard !keepStagesResident else { return }
        vaeEncoder = nil; upsampler = nil
        Memory.clearCache()
    }

    /// Page in the temporal x2 upsampler (DFR rounds only).
    ///
    /// ⚠️ A missing checkpoint is a hard error, never a silent skip: without the trained module the
    /// densified latent decodes into a periodic temporal artefact and nothing else reports it
    /// (oracle `_load_temporal_upsampler`). The variant is asserted for the same reason — pointing
    /// this at a spatial checkpoint would change the geometry, not the frame count.
    func ensureTemporalUpsampler() throws {
        guard temporalUpsampler == nil else { return }
        let url = ltxDir.appending(path: "temporal_upscaler_x2_v1_0.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw TwoStageError.upsamplerMissing(url.path)
        }
        let variant = try Upsampler.peekVariant(path: url)
        guard variant == .temporalX2 else {
            throw TwoStageError.badGeometry(
                "temporal_upscaler_x2_v1_0.safetensors resolved to variant \(variant.rawValue), "
                + "expected temporal_x2 — DFR rounds double FRAMES, not resolution")
        }
        temporalUpsampler = try Upsampler.load(path: url)
    }

    func dropTemporalUpsampler() {
        guard !keepStagesResident else { return }
        temporalUpsampler = nil
        Memory.clearCache()
    }

    /// Output of a t2v run: video pixels (1,3,F,H,W) and optional 48kHz stereo audio (1,2,T).
    public struct Output {
        public let video: MLXArray
        public let audio: MLXArray?
        public init(video: MLXArray, audio: MLXArray?) {
            self.video = video
            self.audio = audio
        }
    }

    /// Load the pipeline. Eagerly loads ONLY the persistent DiT backbone (the resident floor + LoRA
    /// target); every transient stage (Gemma/Connector text encoder, VAE decode stack, two-stage
    /// encoder/upsampler) is loaded on demand around its phase and evicted after. `ltxDir` holds
    /// connector/transformer-distilled/vae_decoder (+ optional audio_vae/vocoder/vae_encoder/
    /// upsampler); `gemmaDir` is the Gemma-3 weights dir. Audio decode is enabled when both
    /// audio_vae.safetensors and vocoder.safetensors exist.
    public static func load(ltxDir: URL, gemmaDir: URL, transformerPath: URL? = nil,
                            vaeDecoderPath: URL? = nil, granuleDirectory: URL? = nil,
                            streamingOptions: BlockStreamingOptions = .init()) async throws
        -> LTX2Pipeline {
        // DIAGNOSTIC LEVER: `LTX_CACHE_LIMIT_GB=N` caps the MLX buffer pool (uncapped by default).
        // An unbounded cache inflates phys_footprint until the OS pages — the suspected cause of the
        // 48f "<10% GPU, 1000s" stall. Set this to test whether capping restores throughput.
        if let g = ProcessInfo.processInfo.environment["LTX_CACHE_LIMIT_GB"], let gb = Int(g) {
            Memory.cacheLimit = gb * 1_000_000_000
            MLXProfiler.shared.note("Memory.cacheLimit set to \(gb) GB (LTX_CACHE_LIMIT_GB)")
        }
        // transformerPath override → quantized checkpoint (q8/q4); DiT auto-detects quant.
        let ditPath = transformerPath ?? ltxDir.appending(path: "transformer-distilled.safetensors")
        // DIAGNOSTIC LEVER: `LTX_VAE_DECODER=pruna|stock|<abs-path>` picks the video decoder without
        // an app change (BRIDGE-LTX-014). An explicit `vaeDecoderPath` from the config always wins.
        // A missing file falls back to stock with a note rather than failing a test run.
        let vaePath = vaeDecoderPath ?? Self.envVAEDecoder(ltxDir: ltxDir)
        // Cold-load cancellation checkpoints (M2/M3 extension): a Cancel/quit during "Loading"
        // stops before/after the two heavy phases (weight load, kernel warmup) instead of
        // waiting the whole load out.
        try Task.checkCancellation()
        // Streamed from the START when granules are supplied — no resident load, and therefore no
        // requirement that the checkpoint exist at all.
        let granuleDir = granuleDirectory
            ?? ProcessInfo.processInfo.environment["LTX_STREAM_GRANULES"]
                .flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0) }
        let dit = try makeDiT(ditPath: ditPath, granuleDir: granuleDir, options: streamingOptions)
        MLXProfiler.shared.note(String(
            format: "DiT %@ done (lazy) · phys=%.2f GB",
            granuleDir == nil ? "resident load" : "STREAMED bind",
            Double(physFootprintBytes()) / 1e9))
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
        let upsamplerSel = defaultUpsamplerFile()
        let upsamplerProbe = upsamplerSel.contains("/")
            ? URL(fileURLWithPath: (upsamplerSel as NSString).expandingTildeInPath)
            : ltxDir.appending(path: upsamplerSel)
        let hasUpsampler = fm.fileExists(atPath: upsamplerProbe.path)
        let p = LTX2Pipeline(dit: dit, ditPath: ditPath, ltxDir: ltxDir, gemmaDir: gemmaDir,
                             hasAudio: hasAudio, hasEncoder: hasEncoder, hasUpsampler: hasUpsampler,
                             vaeDecoderPath: vaePath)
        p.streamingGranuleDirectory = granuleDirectory
        p.streamingOptions = streamingOptions
        return p
    }

    /// Resolve `LTX_VAE_DECODER` to a decoder file, or nil for the stock default.
    private static func envVAEDecoder(ltxDir: URL) -> URL? {
        let raw = (ProcessInfo.processInfo.environment["LTX_VAE_DECODER"] ?? "")
            .trimmingCharacters(in: .whitespaces)
        guard !raw.isEmpty else { return nil }
        switch raw.lowercased() {
        case "stock", "default", "off":
            return nil
        default:
            let url = ["pruna", "lean"].contains(raw.lowercased())
                ? ltxDir.appending(path: "vae_decoder_pruna.safetensors")
                : URL(fileURLWithPath: (raw as NSString).expandingTildeInPath)
            guard FileManager.default.fileExists(atPath: url.path) else {
                MLXProfiler.shared.note("LTX_VAE_DECODER=\(raw): \(url.lastPathComponent) not found — using stock decoder")
                return nil
            }
            MLXProfiler.shared.note("LTX_VAE_DECODER=\(raw) → \(url.lastPathComponent)")
            return url
        }
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
        fps: Double = 24, seed: UInt64? = nil, streaming: StreamingSinks? = nil,
        isolation: isolated (any Actor)? = #isolation
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
        let dit = try ensureDiT()
        // one-stage: this IS the largest stage. `denoiser` arms the gate (tiled or not).
        let engine = denoiser(dit, F: fLat, H: hLat, W: wLat, positions: videoPositions,
                              untiledTokens: nv + audioT, audioTokens: audioT)
        // FIRST-LATENT-FRAME KEYFRAMES MASK (AB-T-0090). The oracle populates this in
        // `create_initial_state` (`ltx_core/tools.py:184`) for EVERY state, on EVERY pipeline —
        // "the reference implementation marks it unconditionally -- independently of whether any
        // keyframe slots exist". The causal video encoder makes the target's first latent frame
        // cover ONE pixel frame while later frames cover 8, which puts it in the same token class
        // as a generated keyframe slot. Inert on 2.3 (no such weight in the checkpoint).
        let kfMask = isLTX25
            ? Self.firstLatentFrameKeyframesMask(totalTokens: nv, tokensPerLatentFrame: hLat * wLat)
            : nil
        let (vfinal, afinalOpt) = try DenoiseLoop.run(
            dit: engine, videoLatent0: videoLatent, audioLatent0: audioLatent, sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: videoPositions, audioPositions: audioPositions,
            keyframesMask: kfMask, label: "")
        // The loop returns nil audio only on the audio-free (DFR round) path; this one always
        // supplies an audio latent, so the unwrap cannot fail.
        let afinal = afinalOpt!
        eval(vfinal, afinal)
        quiesceStreaming()      // stop granule IO before the decode phase
        dropDiTIfSequential()   // low tiers: decode never carries the DiT (T3c)

        // 5. Video: unpatchify → (1, 128, F, H, W) → VAE decode → pixels (decoder loaded now).
        let vspatial = vfinal.reshaped(1, fLat, hLat, wLat, 128).transposed(0, 4, 1, 2, 3)
        try ensureDecoder()
        // 6→5 REORDER (behavior-preserving: the two decodes are independent): audio first, so the
        // streamed lane can finish the writer's audio track before the first video frame.
        let waveform = decodeAudio(afinal)
        if let waveform { eval(waveform) }
        let decSpan = MLXProfiler.shared.begin("vae-decode", "video", note: "\(numFrames)f")
        let pixels: MLXArray
        if let streaming {
            try streaming.onAudioReady(waveform)
            try decodePixels(vspatial, sink: streaming.onVideoChunk)
            pixels = MLXArray.zeros([1, 3, 0, height, width])   // placeholder — documented on StreamingSinks
        } else {
            pixels = try decodePixels(vspatial)
            eval(pixels)
        }
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
        _ = try ensureDiT()
        armStreamingGate(largestStageTokens: nv + audioT)  // one-stage i2v
        // FIRST-LATENT-FRAME KEYFRAMES MASK (AB-T-0090). The oracle populates this in
        // `create_initial_state` (`ltx_core/tools.py:184`) for EVERY state, on EVERY pipeline —
        // "the reference implementation marks it unconditionally -- independently of whether any
        // keyframe slots exist". The causal video encoder makes the target's first latent frame
        // cover ONE pixel frame while later frames cover 8, which puts it in the same token class
        // as a generated keyframe slot. Inert on 2.3 (no such weight in the checkpoint).
        // ⚠️ Independent of `videoMask`: the denoise mask says which tokens are CONDITIONED,
        // this says which are the first-latent-frame token class. i2v conditions frame 0 AND
        // marks it; the two masks coincide here by coincidence, not by rule.
        let kfMask = isLTX25
            ? Self.firstLatentFrameKeyframesMask(totalTokens: nv, tokensPerLatentFrame: frame0)
            : nil
        let (vfinal, afinalOpt) = try DenoiseLoop.runConditioned(
            dit: try ensureDiT(), videoLatent0: videoLatent, audioLatent0: audioLatent, sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: Positions.video(F: fLat, H: hLat, W: wLat, fps: Float(fps)),
            audioPositions: Positions.audio(tokens: audioT),
            videoCleanLatent: cleanVideo, videoDenoiseMask: videoMask,
            keyframesMask: kfMask)
        let afinal = afinalOpt!   // audio always supplied here (see t2v)
        eval(vfinal, afinal)
        quiesceStreaming()      // stop granule IO before the decode phase
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
        LTX2Progress.report(.encode)
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

    /// VAE-encode a reference AUDIO waveform (1, 2, T) 16 kHz stereo in [-1,1] →
    /// `ReferenceConditioning` with NEGATIVE-time positions (the LipDub convention: the ref
    /// audio sits before the target span). Encoder loads around the call, drops immediately.
    public func encodeAudioReference(waveform: MLXArray, strength: Float = 1.0) throws -> ReferenceConditioning {
        LTX2Progress.report(.encode)
        let enc = try AudioVAEEncoder.load(path: ltxDir.appending(path: "audio_vae.safetensors"))
        let span = MLXProfiler.shared.begin("ic-ingest", "audio-vae-encode-ref",
            note: "\(waveform.dim(2)) samples")
        let latent = enc.encode(waveform: waveform)                          // (1,8,T,16)
        let (tokens, positions) = Positions.patchifyLipdubAudioReference(latent)
        eval(tokens); eval(positions)
        MLXProfiler.shared.end(span)
        Memory.clearCache()                                                   // encoder is transient
        return ReferenceConditioning(tokens: tokens, positions: positions,
                                     downscaleFactor: 1, strength: strength)
    }

    /// IC-LoRA conditioned t2v — ONE stage at TARGET resolution (the `stage2: skip` policy the
    /// community reference usage blesses for Ingredients; the adapter's LoRA stays applied for
    /// the whole generation). Video references append per the parity-gated P1 path
    /// (`ICVideoState`); `audioReferences` (LipDub) append to the AUDIO stream the same way
    /// (negative-time positions, clean mask-0 context, sliced after denoise). Both empty falls
    /// through to plain `t2v`.
    public func icT2V(
        prompt: String, references: [ReferenceConditioning],
        audioReferences: [ReferenceConditioning] = [],
        height: Int = 448, width: Int = 704, numFrames: Int = 121,
        fps: Double = 24, seed: UInt64? = nil, isolation: isolated (any Actor)? = #isolation
    ) async throws -> Output {
        guard !references.isEmpty || !audioReferences.isEmpty else {
            return try await t2v(prompt: prompt, height: height, width: width,
                                 numFrames: numFrames, fps: fps, seed: seed)
        }
        let fLat = (numFrames + 7) / 8, hLat = height / 32, wLat = width / 32
        let nv = fLat * hLat * wLat
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)
        let nr = references.reduce(0) { $0 + $1.tokens.dim(1) }
        let nra = audioReferences.reduce(0) { $0 + $1.tokens.dim(1) }
        MLXProfiler.shared.beginRun(String(format:
            "t2v(ic one-stage) %dx%d %df fps=%.0f | nv=%d +ref=%d audioT=%d +aref=%d | steps=%d",
            width, height, numFrames, fps, nv, nr, audioT, nra, Positions.distilledSigmas.count - 1))

        let (videoEmbeds, audioEmbeds) = try await encodePrompt(prompt)

        if let seed { MLXRandom.seed(seed) }
        let videoLatent = MLXRandom.normal([1, nv, 128])
        let audioLatent = MLXRandom.normal([1, audioT, 128])
        let state = ICVideoState.build(
            targetLatent: videoLatent,
            targetPositions: Positions.video(F: fLat, H: hLat, W: wLat, fps: Float(fps)),
            references: references)
        // Audio-side append: ICVideoState's concat semantics are modality-agnostic (the oracle's
        // AudioConditionByReferenceLatent mirrors the video class 1:1).
        let audioState = ICVideoState.build(
            targetLatent: audioLatent,
            targetPositions: Positions.audio(tokens: audioT),
            references: audioReferences)

        // FIRST-LATENT-FRAME KEYFRAMES MASK (AB-T-0090). The oracle populates this in
        // `create_initial_state` (`ltx_core/tools.py:184`) for EVERY state, on EVERY pipeline —
        // "the reference implementation marks it unconditionally -- independently of whether any
        // keyframe slots exist". The causal video encoder makes the target's first latent frame
        // cover ONE pixel frame while later frames cover 8, which puts it in the same token class
        // as a generated keyframe slot. Inert on 2.3 (no such weight in the checkpoint).
        // IC layout is TARGET-FIRST with references appended (`ICVideoState.build`), so the
        // target's first latent frame is still tokens [0, hLat*wLat) and the appended reference
        // tokens get 0 — matching the oracle, whose reference/keyframe-image conditionings all
        // call `extend_keyframes_mask(..., marked=False)` while only keyframe SLOTS use
        // `marked=True` (`ltx_core/conditioning/types/`).
        let kfMask = isLTX25
            ? Self.firstLatentFrameKeyframesMask(totalTokens: state.latent.dim(1),
                                                 tokensPerLatentFrame: hLat * wLat)
            : nil
        let (vfull, afull) = try DenoiseLoop.runConditioned(
            dit: try ensureDiT(), videoLatent0: state.latent, audioLatent0: audioState.latent,
            sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: state.positions, audioPositions: audioState.positions,
            videoCleanLatent: references.isEmpty ? nil : state.clean,
            videoDenoiseMask: references.isEmpty ? nil : state.denoiseMask,
            audioCleanLatent: audioReferences.isEmpty ? nil : audioState.clean,
            audioDenoiseMask: audioReferences.isEmpty ? nil : audioState.denoiseMask,
            keyframesMask: kfMask, label: "ic-")
        let vfinal = state.slice(vfull)
        let afinal = audioState.slice(afull!)   // audio always supplied here (see t2v)
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
    /// Resolved stage-1/stage-2 latent grids for a two-stage run. Extracted verbatim out of
    /// `t2vTwoStage` so `audioToVideo` reuses the SAME validation instead of a second copy that
    /// can drift — the variant math is the part that silently denoises at the wrong resolution
    /// when it is wrong, so it gets exactly one implementation.
    struct TwoStageGeometry {
        let variant: Upsampler.Variant
        let fLat1: Int, hLat1: Int, wLat1: Int, fps1: Double
        let fLat2: Int, hLat2: Int, wLat2: Int
        var nv1: Int { fLat1 * hLat1 * wLat1 }
        var nv2: Int { fLat2 * hLat2 * wLat2 }
    }

    /// The upsampler variant fixes the stage-1 ↔ stage-2 latent relation:
    ///   x2:       (F, H/2, W/2) → (F, H, W)      target pixels ÷64
    ///   x1.5:     (F, 2H/3, 2W/3) → (F, H, W)    target pixels ÷96 (stage-1 latents
    ///             land even automatically, which the blur-downsample needs exact)
    ///   temporal: ((F+1)/2, H, W) → (F, H, W)    target latent frames ODD (the module
    ///             drops doubled frame 0), i.e. pixel frames ≡ 1 (mod 16); stage 1 runs
    ///             at fps/2 so both stages span the same seconds.
    /// x2 is upstream's composition; x1.5/temporal two-stage is OURS (no oracle or upstream
    /// consumer exists — see `upsamplerFile`). Validated BEFORE any heavy phase.
    func resolveTwoStageGeometry(height: Int, width: Int, numFrames: Int,
                                 fps: Double) throws -> TwoStageGeometry {
        guard FileManager.default.fileExists(atPath: upsamplerURL.path) else {
            throw TwoStageError.upsamplerMissing(upsamplerURL.path)
        }
        let variant = try Upsampler.peekVariant(path: upsamplerURL)
        let fLat2 = (numFrames + 7) / 8
        let latH2 = height / 32, latW2 = width / 32
        guard height % 32 == 0, width % 32 == 0 else {
            throw TwoStageError.badGeometry("target \(width)x\(height) must be divisible by 32")
        }
        let fLat1: Int, hLat1: Int, wLat1: Int, fps1: Double
        switch variant {
        case .spatialX2:
            guard latH2 % 2 == 0, latW2 % 2 == 0 else {
                throw TwoStageError.badGeometry(
                    "x2 two-stage needs target pixels divisible by 64, got \(width)x\(height)")
            }
            (fLat1, hLat1, wLat1, fps1) = (fLat2, latH2 / 2, latW2 / 2, fps)
        case .spatialX1_5:
            guard latH2 % 3 == 0, latW2 % 3 == 0 else {
                throw TwoStageError.badGeometry(
                    "x1.5 two-stage needs target pixels divisible by 96, got \(width)x\(height)")
            }
            (fLat1, hLat1, wLat1, fps1) = (fLat2, latH2 * 2 / 3, latW2 * 2 / 3, fps)
        case .temporalX2:
            guard fLat2 % 2 == 1 else {
                throw TwoStageError.badGeometry(
                    "temporal two-stage needs pixel frames ≡ 1 (mod 16) so target latent frames "
                    + "are odd (the upsampler drops doubled frame 0), got \(numFrames)f "
                    + "(latent \(fLat2))")
            }
            (fLat1, hLat1, wLat1, fps1) = ((fLat2 + 1) / 2, latH2, latW2, fps / 2)
        }
        return TwoStageGeometry(variant: variant, fLat1: fLat1, hLat1: hLat1, wLat1: wLat1,
                                fps1: fps1, fLat2: fLat2, hLat2: latH2, wLat2: latW2)
    }

    public func t2vTwoStage(
        prompt: String, height: Int = 512, width: Int = 704, numFrames: Int = 9,
        fps: Double = 24, seed: UInt64? = nil, streaming: StreamingSinks? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> Output {
        guard hasEncoder, hasUpsampler else {
            return try await t2v(prompt: prompt, height: height, width: width, numFrames: numFrames, fps: fps, seed: seed,
                                 streaming: streaming)
        }

        // --- Variant-derived geometry (see `resolveTwoStageGeometry`) ---
        let geo = try resolveTwoStageGeometry(height: height, width: width,
                                              numFrames: numFrames, fps: fps)
        let variant = geo.variant
        let (fLat1, hLat1, wLat1, fps1) = (geo.fLat1, geo.hLat1, geo.wLat1, geo.fps1)
        let (fLat2, latH2, latW2) = (geo.fLat2, geo.hLat2, geo.wLat2)
        let audioT = Positions.audioTokenCount(numFrames: numFrames, fps: fps)
        let s2 = Positions.stage2Sigmas
        let sigma0 = s2[0]
        let nv2 = geo.nv2                      // stage-2 (target-res) video token count
        let nv1 = geo.nv1                      // stage-1 token count
        MLXProfiler.shared.beginRun(String(format:
            "t2vTwoStage %dx%d %df fps=%.0f | %@ | fLat=%d→%d nv1=%d nv2=%d audioT=%d | steps s1=%d s2=%d",
            width, height, numFrames, fps, variant.rawValue, fLat1, fLat2, nv1, nv2, audioT,
            Positions.distilledSigmas.count - 1, s2.count - 1))

        // 1. Text encode — sequential Gemma → connector (never co-resident), self-evicting.
        let (videoEmbeds, audioEmbeds) = try await encodePrompt(prompt)

        // --- Stage 1 (denoise peak #1, DiT only resident) ---
        let v1 = LTX2Pipeline.noiseInit(clean: MLXArray.zeros([1, nv1, 128]), sigma: 1.0, shape: [1, nv1, 128], seed: seed)
        let a1 = LTX2Pipeline.noiseInit(clean: MLXArray.zeros([1, audioT, 128]), sigma: 1.0, shape: [1, audioT, 128], seed: seed.map { $0 &+ 1 })
        _ = try ensureDiT()
        // Two-stage: gate on STAGE 2's tokens — stage 1 streams at the IO floor
        // rather than condemning a run whose full-res stage would stream
        // (STREAMING-PLAN §4).
        armStreamingGate(largestStageTokens: nv2 + audioT)
        // LTX-2.5 stage 1: ancestral Euler (eta 1.0, noise seed = seed + 10000) and the
        // first-latent-frame keyframes mask on every forward. Both are inert on 2.3, where
        // `isLTX25` is false and the checkpoint carries no keyframes embedding.
        let kfMask1 = isLTX25
            ? Self.firstLatentFrameKeyframesMask(totalTokens: v1.dim(1),
                                                 tokensPerLatentFrame: hLat1 * wLat1)
            : nil
        let (v1f, a1fOpt) = try DenoiseLoop.run(
            dit: try ensureDiT(), videoLatent0: v1, audioLatent0: a1, sigmas: Positions.distilledSigmas,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: Positions.video(F: fLat1, H: hLat1, W: wLat1, fps: Float(fps1)),
            audioPositions: Positions.audio(tokens: audioT),
            keyframesMask: kfMask1,
            ancestralEta: isLTX25 ? 1.0 : 0,
            ancestralNoiseSeed: (seed ?? 0) &+ 10_000,
            label: "s1-", stage: 1, totalStages: 2)
        let a1f = a1fOpt!   // audio always supplied here (see t2v)
        eval(v1f, a1f)

        // --- Upscale in un-normalized latent space (encoder+upsampler loaded only here) ---
        LTX2Progress.report(.upsample)
        let upSpan = MLXProfiler.shared.begin("upscale", "vae-enc+upsampler",
                                              note: "\(variant.rawValue) stage1→stage2 latent")
        try ensureVAEEncoder(); try ensureUpsampler()
        let v1spatial = v1f.reshaped(1, fLat1, hLat1, wLat1, 128).transposed(0, 4, 1, 2, 3)  // (1,128,F1,h1,w1)
        let upscaled = vaeEncoder!.normalizeLatent(upsampler!(vaeEncoder!.denormalizeLatent(v1spatial)))
        eval(upscaled)
        MLXProfiler.shared.end(upSpan)
        dropUpscaler()                                          // evict before the stage-2 denoise peak
        // Geometry contract: the module's actual output must land exactly on the target
        // latent grid — a mismatch here means the variant math above is wrong, and stage 2
        // would silently denoise at the wrong resolution.
        guard upscaled.dim(2) == fLat2, upscaled.dim(3) == latH2, upscaled.dim(4) == latW2 else {
            throw TwoStageError.badGeometry(
                "upsampler(\(variant.rawValue)) produced latent \(upscaled.shape), expected "
                + "(1, 128, \(fLat2), \(latH2), \(latW2)) for target \(width)x\(height)x\(numFrames)f")
        }
        let hLat2 = latH2, wLat2 = latW2
        let v2tokens = LTX2Pipeline.patchify(upscaled)  // (1, F2*h2*w2, 128)

        // --- Stage 2: full resolution refine (init = noise·σ₀ + upscaled·(1-σ₀)) ---
        let v2init = LTX2Pipeline.noiseInit(clean: v2tokens, sigma: sigma0, shape: v2tokens.shape, seed: seed.map { $0 &+ 2 })
        let a2init = LTX2Pipeline.noiseInit(clean: a1f, sigma: sigma0, shape: a1f.shape, seed: seed.map { $0 &+ 2 })
        // Stage 2 keeps plain Euler (the oracle only switches stage 1 to ancestral) but still
        // needs the keyframes mask — the trained embedding applies on every 2.5 forward.
        let kfMask2 = isLTX25
            ? Self.firstLatentFrameKeyframesMask(totalTokens: v2init.dim(1),
                                                 tokensPerLatentFrame: hLat2 * wLat2)
            : nil
        let (v2f, a2fOpt) = try DenoiseLoop.run(
            dit: try ensureDiT(), videoLatent0: v2init, audioLatent0: a2init, sigmas: s2,
            videoText: videoEmbeds, audioText: audioEmbeds,
            videoPositions: Positions.video(F: fLat2, H: hLat2, W: wLat2, fps: Float(fps)),
            audioPositions: Positions.audio(tokens: audioT),
            keyframesMask: kfMask2,
            label: "s2-", stage: 2, totalStages: 2)
        let a2f = a2fOpt!   // audio always supplied here (see t2v)
        eval(v2f, a2f)
        quiesceStreaming()      // stop granule IO before the decode phase
        dropDiTIfSequential()   // low tiers: decode never carries the DiT (T3c)

        let vspatial = v2f.reshaped(1, fLat2, hLat2, wLat2, 128).transposed(0, 4, 1, 2, 3)
        try ensureDecoder()
        // Audio FIRST (see t2v — the streamed lane needs the writer's audio track closed
        // before the first video frame; the two decodes are independent).
        let audSpan = MLXProfiler.shared.begin("audio-decode", "audioVAE+vocoder")
        let waveform = decodeAudio(a2f)
        if let waveform { eval(waveform) }
        MLXProfiler.shared.end(audSpan)
        let decSpan = MLXProfiler.shared.begin("vae-decode", "video", note: "\(fLat2*8-7)f full-res")
        let pixels: MLXArray
        if let streaming {
            try streaming.onAudioReady(waveform)
            try decodePixels(vspatial, sink: streaming.onVideoChunk)
            pixels = MLXArray.zeros([1, 3, 0, height, width])
        } else {
            pixels = try decodePixels(vspatial)
            eval(pixels)
        }
        MLXProfiler.shared.end(decSpan)
        dropDecoder()
        MLXProfiler.shared.endRun()
        return Output(video: pixels, audio: waveform)
    }
}

/// Two-stage wiring failures — all thrown BEFORE any heavy phase (encode/denoise), so a
/// mis-sized request costs milliseconds, not a stage-1 denoise.
public enum TwoStageError: Error, CustomStringConvertible {
    case upsamplerMissing(String)
    case badGeometry(String)
    public var description: String {
        switch self {
        case .upsamplerMissing(let p): return "two-stage upsampler checkpoint not found: \(p)"
        case .badGeometry(let m): return "two-stage geometry: \(m)"
        }
    }
}
