import Foundation
import LTX2
import MLXToolKit

/// Memory-tier profile (LOW-TIER-PLAN T3): an envelope clamp + path policy + VAE decode window that
/// makes LTX-2.3 honestly declarable per tier. Activation is seqLen-scaled, so a profile that CLAMPS
/// the envelope can declare a far smaller `peakActivationBytesHint` than the 704×512 default —
/// requests beyond the envelope are clamped (not rejected). The recommended quant per tier is
/// advisory (`recommendedQuant`); the registered `quant` still decides the checkpoint.
/// 16 GB is deliberately ABSENT: the int4 DiT alone (11.3 GB) ≈ a 16 GB governor budget, and no
/// smaller LTX-2 checkpoint exists.
public enum LTX2Profile: String, Codable, Sendable, CaseIterable {
    /// 24 GB Macs (M5 MacBook Pro default): int4, one-stage, small envelope, tight decode window.
    case compact24
    /// 32 GB Macs: int4/int8, one-stage, mid envelope.
    case balanced32
    /// 64 GB Macs: int8, full two-stage at the standard envelope.
    case standard64
    /// 96–128 GB Macs: bf16 two-stage, long clips. 480f bf16 t2v MEASURED 67.61 GB post-T3b
    /// (BRIDGE-LTX-005; floor ~40.5 GB + ~40 MB/frame activation), so the envelope admits 481f.
    /// i2v at the cap MEASURED too (--i2v-spot 2026-07-01): 481f + 4.9 GB adapter peaks 72.73 GB.
    case max128

    public var maxWidth: Int  { switch self { case .compact24: 512; case .balanced32: 576; default: 704 } }
    public var maxHeight: Int { switch self { case .compact24: 288; case .balanced32: 320; default: 512 } }
    public var maxFrames: Int {
        // max128 241→481 (BRIDGE-LTX-005): 704×512×480f bf16 t2v measured 67.61 GB — the old cap
        // left ~60 GB of a 128 GB budget unused. 481 sits ON a measured point (8n+1 frame grid).
        switch self { case .compact24: 121; case .balanced32: 161; case .standard64: 161; case .max128: 481 }
    }
    /// One-stage skips the spatial upsampler + full-res stage-2 refine — the low-tier denoise path.
    public var oneStage: Bool { self == .compact24 || self == .balanced32 }
    /// VAE temporal-decode window knob (see `LTX2Pipeline.decodePixels`; halo is fixed at 5).
    public var vaeChunkFrames: Int {
        switch self { case .compact24: 4; case .balanced32: 6; default: 8 }
    }
    /// Evict the DiT around the stages that do not need it — ON FOR EVERY TIER as of the
    /// LTX-2.5 measurement below.
    ///
    /// Originally low-tier only (T3c: decode-with-DiT-resident was the residual low-tier peak).
    /// LTX-2.5 widened the case decisively, because its text encoder is a bf16 12B rather than
    /// 2.3's 4-bit one, so the PRE-ENCODE drop now dominates: measured on one binary,
    /// ABAB-alternated at 448×320×9 (AB-R-0030),
    ///
    ///     off: peak 62.40 GB   (DiT floor 37.98 + 24.42 encoder co-resident)
    ///     on:  peak 40.66 GB   (DiT floor 37.98 + 2.68 denoise activation)
    ///     → −21.74 GB, −34.8%, output BIT-IDENTICAL (mean −0.1274, std 0.4222 both arms)
    ///
    /// Wall-clock was NOISE between arms (13.8–15.8 s, overlapping), so the reload cost this
    /// pays for — mmap re-fault, kernels stay process-cached — does not show above run-to-run
    /// variance at this size. A third of peak memory for no measurable time is worth taking on
    /// every tier, and headroom is what absorbs pressure from whatever else the machine is
    /// doing. `LTX_EVICT_DIT=0` still forces it off for an A/B.
    public var evictDiTBeforeDecode: Bool { true }
    /// Runtime-LoRA factor packing (BRIDGE-LTX-012): on the 24/32 GB tiers the rank-256 i2v
    /// adapter (4.93 GB bf16) rides the DiT through the denoise peak — measured 18.86 GB on the
    /// 24 GB target vs the 16.8 budget. int8 group-64 factors halve that residency (≈16.4 GB,
    /// under budget); fidelity gated by `RunLTX2 --lora-quant-gate`. nil = full-precision factors.
    public var loraFactorQuantBits: Int? {
        switch self { case .compact24, .balanced32: return 8; default: return nil }
    }
    public var recommendedQuant: Quant {
        switch self { case .compact24, .balanced32: .int4; case .standard64: .int8; case .max128: .bf16 }
    }

    /// Advisory, like `recommendedQuant`: which TEXT-ENCODER precision this tier wants.
    ///
    /// 🔑 **Measured, not guessed (AB-R-0090/0104/0105).** With the DiT streamed, LTX-2.5's peak is
    /// set by a **geometry-INDEPENDENT ~24.8 GB encoder floor** (256×256×9 and 512×288×121 agree to
    /// 0.01 GB). That floor is over both low-tier budgets, so on `compact24`/`balanced32` the int8
    /// encoder is not a nicety — it is load-bearing: 24.8 → **14.6 / 15.4 GB**. Quality is settled
    /// by a BLIND 4-pair perceptual A/B (AB-R-0104). Above those tiers bf16 fits comfortably and
    /// stays the default, since it is the reproducible arm.
    public var recommendedTextEncoderQuant: Quant {
        switch self { case .compact24, .balanced32: .int8; default: .bf16 }
    }

    /// Advisory: whether this tier should PIN the streamer on rather than let the runtime gate decide.
    ///
    /// 🚨 **`.auto` is not safe to DECLARE a low tier on.** The gate streams only when measured IO
    /// (S) outruns measured compute (C(N)), and **C(N) at small N is noisy**: at `compact24`'s
    /// N=2430 it read 3.34 / 7.43 / 3.52 GiB/s over three runs, flipping the verdict and swinging
    /// PEAK between **14.6 GB (streamed) and 33.6 GB (fell back resident)** — 1 run in 3. A
    /// footprint that depends on a coin flip cannot be declared; the honest `.auto` number is the
    /// FALLBACK peak, which busts the budget.
    /// ✅ Forcing it costs nothing measurable: 3 forced runs read **14.56 GB every time**, at
    /// 47.3–68.1 s against `.auto`'s 46.5–69.0 s — overlapping spreads, no stall penalty. The
    /// expected memory-for-time trade did not materialise, because a spurious fallback ALSO costs
    /// time (it loads 18.38 GiB).
    /// 🚨 **THE RULE IS NOT "small N" — it is "does the FALLBACK still fit".** An earlier version of
    /// this advised `compact24` only, reasoning from N headroom. That was wrong and it was
    /// load-bearing: `.auto` may decline to stream on ANY run, and a tier whose budget is met only
    /// while streaming busts outright when it does. Measured resident (fallback) peaks against
    /// budget:
    ///
    ///     compact24    31.92  vs 16.8   🚨 busts       → PIN
    ///     balanced32   33.13  vs 22.4   🚨 busts       → PIN
    ///     standard64   27.31  vs 44.8   ✅ still fits  → `.auto` is safe
    ///
    /// So `balanced32` must be pinned too, even though it streamed 6/6 in measurement — 6/6 is a
    /// probability, and the declaration has to hold on the run that goes the other way.
    /// `standard64` genuinely does not need it: its fallback is admissible, so the gate is free to
    /// choose.
    public var recommendedForcedStreamGate: Bool {
        switch self { case .compact24, .balanced32: true; default: false }
    }

    /// Whether this tier should STREAM the DiT blocks by default.
    ///
    /// 🔑 **On/off is not a quality question — streamed output is bit-identical to resident**
    /// (`--stream-parity-gate` memcmp + poison control). It is a cost question, and the costs run
    /// opposite ways per tier:
    ///
    ///   compact24 / balanced32 — **REQUIRED.** Resident busts the budget outright (31.92/16.8 and
    ///     33.13/22.4). Without streaming 2.5 does not run here at all.
    ///   standard64 — **BENEFICIAL.** Resident fits (27.31/44.8), but streaming measured 24% FASTER
    ///     (105.6 s vs 139.1 s) because eviction reloads the DiT every stage, and it frees ~8 GB
    ///     for co-tenancy. Default on.
    ///   max128 — **OFF.** bf16 already fits with room, so streaming buys headroom nobody needs at
    ///     the price of ~6× read amplification — and max128's streamed behaviour is UNMEASURED.
    ///     Do not default to an unmeasured configuration just because it is available.
    public var recommendedStreamedBlocks: Bool {
        switch self { case .max128: false; default: true }
    }
}

/// Init-time configuration for `MLXLTX2Package` (C9): where the LTX-2.3 component
/// weights and the Gemma-3 text encoder live. Per-request prompt/size/steps ride
/// the canonical `T2VRequest`, not here.
///
/// `ltxDirectory` holds the LTX safetensors (connector / transformer-distilled /
/// vae_decoder). `gemmaDirectory` is the Gemma-3 MLX weights dir (mlx-community/
/// gemma-3-12b-it-4bit). Both are environment-specific → excluded from Codable.
/// Which LTX generation a configuration serves. The PIPELINE detects this from the checkpoint
/// (`LTX2Pipeline.isLTX25` keys off the in-dir `gemma4-12b-ltx-v1/`, never a path-name parse) and
/// that stays the authority at runtime. This axis exists for what has to be decided BEFORE any
/// weights are on disk: default repos, which sibling repo carries a quantized transformer, and
/// where the text encoder lives.
///
/// Defaults to `.ltx23`, and it is deliberately absent from `CodingKeys` fallbacks in a way that
/// makes a config persisted before this existed decode as 2.3 — the generation it was written for.
public enum LTXFamily: String, Codable, Sendable, CaseIterable {
    case ltx23, ltx25

    public var defaultRepo: String { self == .ltx25 ? "mlx-community/ltx-2.5-mlx" : "xocialize/ltx-2.3-mlx" }

    /// 2.5's text encoder ships INSIDE the components tree (`gemma4-12b-ltx-v1/`), so there is no
    /// separate encoder repo to materialize — `LTX2Pipeline.gemma4Dir` derives it from `ltxDirectory`.
    /// 2.3's is a normal external repo.
    var defaultGemmaRepo: String? { self == .ltx25 ? nil : "mlx-community/gemma-3-12b-it-4bit" }

    /// 🚨 **The suffix differs, and getting it wrong is SILENT.** On 2.3 the quantized-DiT siblings
    /// are `<repo>-q8` / `-q4`. On 2.5 `<repo>-q8` is the int8 **TEXT-ENCODER** sibling (AB-D-0013):
    /// it carries only `gemma4-12b-ltx-v1/` shards and SYMLINKS `transformer-distilled.safetensors`
    /// back to the bf16 file. Deriving `-q8` there would register `.int8` — charging the engine
    /// int8's ~22 GB resident footprint — while actually loading the 40 GB bf16 DiT, i.e. a ~18 GB
    /// under-declaration that admits a run the governor should have refused. The 2.5 quantized DiT
    /// is `-ditq8`. int4 has no 2.5 sibling at all (contraindicated, AB-D-0013/0014) → nil.
    func transformerRepoSuffix(for quant: Quant) -> String? {
        switch (self, quant) {
        case (.ltx23, .int8): return "-q8"
        case (.ltx23, .int4): return "-q4"
        case (.ltx25, .int8): return "-ditq8"
        case (.ltx25, .int4): return nil        // no q4 2.5 DiT exists; do NOT derive one
        default: return nil                     // bf16 rides the components repo
        }
    }

    /// Sibling repo carrying a quantized TEXT ENCODER, or nil to use the components repo as-is.
    ///
    /// 🔑 **This is what `<repo>-q8` on 2.5 actually IS**, and naming it here is the point: the
    /// suffix has been documented for months only as a *trap* for the DiT axis above, which is
    /// true and which obscured that it is the correct and only source of the int8 Gemma-4 encoder.
    /// The tree carries `gemma4-12b-ltx-v1/` int8 shards (13 GB vs the bf16 tree's 22 GB) and
    /// symlinks every other component back, so pointing the COMPONENTS repo at it swaps the
    /// encoder and nothing else — which is exactly how AB-R-0031 built it ("zero wiring changes").
    ///
    /// 2.3 returns nil at every quant: its encoder is an EXTERNAL repo (`gemmaRepo`), not in-tree,
    /// so a suffix here would name a components sibling that does not carry an encoder at all.
    ///
    /// ⚠️ Only int8 exists. int4 g64 was REJECTED by the gate (connector output 0.996728 on valid
    /// positions vs a 0.999879 bf16 floor, AB-D-0010) — do not derive a `-q4` encoder.
    func textEncoderRepoSuffix(for quant: Quant) -> String? {
        switch (self, quant) {
        case (.ltx25, .int8): return "-q8"
        default: return nil
        }
    }

    /// Published granule trees (`xocialize/ltx-2.{3,5}-granules`), laid out with one subdirectory
    /// per quant — exactly the shape `resolvedGranuleDirectory` appends to.
    ///
    /// 🔑 **Granules are a DETERMINISTIC re-layout of the same bytes, so they are distributed, not
    /// regenerated per install.** Each tree carries per-file SHA-256 in its manifest plus
    /// `source_repo`/`source_revision` provenance stamps, so a downloader can verify what it got
    /// rather than trusting the filename (the whole point of manifest v2).
    var granuleRepo: String { self == .ltx25 ? "xocialize/ltx-2.5-granules"
                                             : "xocialize/ltx-2.3-granules" }
}

public struct LTX2Configuration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// Which LTX generation this configuration serves — see `LTXFamily`. Affects only pre-download
    /// resolution; the pipeline still detects the generation from the checkpoint at load.
    public var family: LTXFamily = .ltx23
    /// The repo serving the LTX-2.3 MLX components.
    ///
    /// 🚨 **Weight-durability rule** (`mlx-porting` skill, Step 8): a shipped source never points
    /// at a namespace we do not control. The default is `xocialize/ltx-2.3-mlx`, a byte-verified
    /// ⚠️ **PARTIAL mirror — 7 of upstream's 13 safetensors** (verified 2026-08-21 by comparing
    /// published blob sizes: every shared file matches byte-for-byte, so the "verbatim" claim
    /// holds — but `transformer-dev`, `transformer-distilled-1.1`, `spatial_upscaler_x1_5_v1_0`,
    /// `temporal_upscaler_x2_v1_0` and both distilled LoRAs were never mirrored).
    /// 🚨 **That has a FUNCTIONAL consequence**: `Upsampler` supports and gates all three variants
    /// (x2 / x1.5-rational / temporal-x2), but our published tree ships only x2 — so a consumer
    /// fetching from `repo` cannot use two of the three we built. Tracked as the full from-origin
    /// 2.3 port. verbatim mirror of `dgrauet/ltx-2.3-mlx` (every tensor SHA-256-matched at mirror time, and
    /// the LTX-2 Community License propagated per its §3). The upstream conversion lives on a
    /// PERSONAL account, which can disappear via account deletion or a username change — routes
    /// an org cannot take. `Provenance.sourceRepo` still names dgrauet, because that is where
    /// these bytes originate; this is only where they are fetched from.
    ///
    /// ⚠️ `effectiveTransformerRepo` derives the quant repos as `repo + "-q8"` / `"-q4"`, so
    /// overriding this to another namespace requires that namespace to carry all three.
    public var repo: String
    public var revision: String?
    /// Text-encoder repo (Gemma-3), materialized when `gemmaDirectory` is nil (BRIDGE M4).
    public var gemmaRepo: String
    /// Override for the quantized-transformer repo; nil derives `<repo>-q8`/`-q4` from `quant`
    /// (bf16 rides the components repo — no separate source).
    public var transformerRepo: String?
    /// Backbone quant of the distilled transformer. `QuantConfigured` surfaces it to the engine's
    /// `MemoryGovernor` so it charges the *registered* variant's `QuantFootprint` (bf16/int8/int4)
    /// instead of the bf16 max (engine ≥0.9.1; closes the q8/q4 over-reservation, LTX ENHANCEMENTS E14).
    public var quant: Quant
    /// Resolved LTX component directory (connector/vae_decoder/vae_encoder/audio_vae/
    /// vocoder/upsampler — these stay bf16 across quant variants).
    public var ltxDirectory: URL?
    /// Optional override for the DiT transformer file. Defaults to
    /// `ltxDirectory/transformer-distilled.safetensors` (bf16). Point at a quantized
    /// checkpoint (e.g. `.../ltx-2.3-mlx-q8/transformer-distilled.safetensors`) to run
    /// int8/int4 — the loader auto-detects quantization from the weights (scales/biases).
    /// Only the transformer is quantized; everything else loads from `ltxDirectory`.
    public var transformerPath: URL?
    /// Optional override for the video VAE decoder file. Defaults to
    /// `ltxDirectory/vae_decoder.safetensors` (stock). Point at a pruned sibling — e.g. the
    /// converted `vae_decoder_pruna.safetensors` (PrunaVAED) — for the lean decode tier:
    /// ~2× faster decode and a materially smaller decode activation peak, at a decode that is
    /// near-identical but not bit-exact. The decoder derives every channel width from the
    /// weights and enables PrunaVAED's channel-adapter blocks on sight, so this is the only
    /// switch; encoder, latents and every other component are untouched.
    ///
    /// Not auto-materializable yet: PrunaAI publishes diffusers-format weights that must go
    /// through `scripts/convert_pruna_vae_decoder.py` first, so this takes an explicit path
    /// until the converted file is hosted alongside the other components.
    public var vaeDecoderPath: URL?
    /// Resolved Gemma-3 text-encoder directory.
    public var gemmaDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// Memory-tier profile (nil = unconstrained legacy behavior — no clamp, two-stage preferred,
    /// footprint falls back to the per-quant `QuantFootprint` measured at 704×512).
    public var profile: LTX2Profile?
    /// HV2 weight streaming (STREAMING-PLAN.md): stream the DiT's 48 transformer blocks from a
    /// per-block granule tree (two ~2×`groupSize`-block slots resident, background pread refill,
    /// self-calibrating gate with automatic output-invisible resident fallback) instead of
    /// loading them resident. Requires `granuleRootDirectory` pointing at a tree laid out by
    /// `ltx-granule-layout` with `bf16/`, `q8/`, `q4/` subdirectories (the configured `quant`
    /// picks the subdirectory; provenance vs the transformer checkpoint is verified at bind).
    /// Both fields are EXCLUDED from Codable — the root is an environment path like
    /// `ltxDirectory`, and encoding the flag alone would buy nothing while making the key
    /// mandatory for configs persisted before it existed (the bernini d02cfa1 pattern).
    /// 🔑 **TRI-STATE: `nil` (the default) means FOLLOW THE PROFILE'S ADVICE.** Read
    /// `effectiveStreamedBlocks`, never this field.
    ///
    /// Streaming is the better default wherever it applies, not merely an escape valve: output is
    /// **bit-identical** (`--stream-parity-gate` memcmp, with a poisoned-slot control proving the
    /// compare has teeth), stall at real generation sizes is **0.0%**, and it is measurably FASTER
    /// than resident because `evictDiTAroundStages` makes the resident path RELOAD the DiT every
    /// stage — 105.6 s vs 139.1 s at standard64 (AB-R-0090).
    ///
    /// ⚠️ Its real cost is **read amplification**: ~6× the bytes read per generation (a full sweep
    /// per denoise step rather than one load). Overlapped, so no wall-clock — but real SSD/power
    /// load, and UNMEASURED on battery. That is why `max128` does NOT default to it: there the
    /// resident footprint already fits with room, so streaming would buy headroom nobody needs at
    /// a cost we have not characterised.
    public var streamedBlocks: Bool?

    /// Resolved: explicit override → profile advice → `false`.
    public var effectiveStreamedBlocks: Bool {
        streamedBlocks ?? profile?.recommendedStreamedBlocks ?? false
    }
    /// Root of the granule store (e.g. `/Volumes/Satechi/Models/ltx-granules`).
    public var granuleRootDirectory: URL?

    /// Streaming knobs forwarded to `BlockStreamer` — group size, gate policy, margin.
    ///
    /// Default `.auto` preserves today's behaviour exactly; a caller opts into pinning with
    /// `streamingOptions.gatePolicy = .forceStream`, and `LTX2Profile.recommendedForcedStreamGate`
    /// says which tiers want that (advisory — this field decides, mirroring `recommendedQuant`).
    /// EXCLUDED from Codable: `BlockStreamingOptions` is `Sendable` but not `Codable`, and the
    /// policy is a deployment decision rather than persisted model identity.
    public var streamingOptions = BlockStreamingOptions()

    /// Precision of the TEXT ENCODER, independent of `quant` (which is the DiT's).
    ///
    /// 🔑 **Two separate axes, and conflating them is the documented 2.5 footgun.** `quant` selects
    /// the transformer sibling (`-ditq8` on 2.5); this selects the components repo carrying the
    /// encoder (`-q8` on 2.5). They are different repos with different suffixes and different
    /// footprints — see `LTXFamily.transformerRepoSuffix` vs `textEncoderRepoSuffix`.
    ///
    /// 🔑 **TRI-STATE: `nil` (the default) means FOLLOW THE PROFILE'S ADVICE.** A non-nil value is
    /// an explicit override and always wins. Read the resolved value from
    /// `effectiveTextEncoderQuant`, never from this field.
    ///
    /// Auto-following is deliberate (operator decision 2026-08-21): on `compact24`/`balanced32` the
    /// int8 encoder is not a preference, it is **load-bearing** — with the DiT streamed, 2.5's peak
    /// is set by a geometry-independent ~24.8 GB encoder floor that is over both budgets, and int8
    /// takes it to 14.6 / 15.4 GB (AB-R-0090/0106). A caller who picks a low tier and leaves this
    /// alone should get a configuration that FITS rather than one that silently busts the governor.
    ///
    /// ⚠️ **This changes nothing for configs persisted before it existed.** The advice is `.bf16`
    /// on `standard64`/`max128` — the old default — and 2.5 was never admissible on the low tiers,
    /// so no persisted 2.5 config can be sitting on a profile whose advice differs.
    ///
    /// ⚠️ **Only affects REPO resolution** (what `weightSources` materializes). When `ltxDirectory`
    /// is set you are naming a tree directly, so point it at the sibling (`…-q8/`) instead —
    /// `LTX2Pipeline.gemma4Dir` derives the encoder from `ltxDirectory` and neither this axis nor
    /// the profile can override an explicit path.
    public var textEncoderQuant: Quant?

    /// Resolved text-encoder precision: explicit override → profile advice → `.bf16`.
    public var effectiveTextEncoderQuant: Quant {
        textEncoderQuant ?? profile?.recommendedTextEncoderQuant ?? .bf16
    }

    /// 🔑 **TRI-STATE: `nil` (the default) means FOLLOW THE PROFILE'S ADVICE.** `true`/`false` is an
    /// explicit override and always wins — the escape hatch for measurement and for callers who
    /// know what they are doing.
    ///
    /// Auto-following matters because `.auto` is not safe to DECLARE a low tier on: the runtime
    /// gate compares measured IO against measured compute, and at `compact24`'s N the compute
    /// estimate is noisy enough to flip the verdict — PEAK swung 14.6 ↔ 33.6 GB across three runs,
    /// 1 in 3 falling back over budget (AB-R-0105). Pinning costs nothing measurable.
    ///
    /// ⚠️ Setting `streamingOptions.gatePolicy` directly does NOT survive resolution — use this.
    /// `resolvedStreamingOptions` is what the wrapper forwards.
    public var forceStreamGate: Bool?

    /// Streaming options as actually forwarded: `streamingOptions` with `gatePolicy` resolved from
    /// the override, else the profile's advice, else left as-is.
    public var resolvedStreamingOptions: BlockStreamingOptions {
        var o = streamingOptions
        if let forced = forceStreamGate {
            o.gatePolicy = forced ? .forceStream : .auto
        } else if profile?.recommendedForcedStreamGate == true {
            o.gatePolicy = .forceStream
        }
        return o
    }

    /// The granule tree for the configured quant, when streaming is enabled.
    public var resolvedGranuleDirectory: URL? {
        guard effectiveStreamedBlocks, let root = granuleRootDirectory else { return nil }
        let sub: String
        switch quant {
        case .int8: sub = "q8"
        case .int4: sub = "q4"
        default: sub = "bf16"
        }
        return root.appendingPathComponent(sub)
    }

    public init(
        family: LTXFamily = .ltx23,
        repo: String? = nil,
        revision: String? = nil,
        gemmaRepo: String? = nil,
        transformerRepo: String? = nil,
        quant: Quant = .bf16,
        ltxDirectory: URL? = nil,
        transformerPath: URL? = nil,
        vaeDecoderPath: URL? = nil,
        gemmaDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil,
        profile: LTX2Profile? = nil
    ) {
        self.family = family
        self.repo = repo ?? family.defaultRepo
        self.revision = revision
        // On 2.5 the encoder is in-tree, so there is no repo to name; the field keeps 2.3's value
        // as an inert default rather than becoming Optional across every existing call site.
        self.gemmaRepo = gemmaRepo ?? family.defaultGemmaRepo ?? "mlx-community/gemma-3-12b-it-4bit"
        self.transformerRepo = transformerRepo
        self.quant = quant
        self.ltxDirectory = ltxDirectory
        self.transformerPath = transformerPath
        self.vaeDecoderPath = vaeDecoderPath
        self.gemmaDirectory = gemmaDirectory
        self.modelsRootDirectory = modelsRootDirectory
        self.profile = profile
    }

    private enum CodingKeys: String, CodingKey {
        case family, repo, revision, gemmaRepo, transformerRepo, quant, profile
        case textEncoderQuant
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Absent ⇒ `.ltx23`: a config persisted before this key existed was written for 2.3, and
        // that is what it must keep decoding as (the bernini d02cfa1 pattern — a new key must never
        // become mandatory for configs that predate it).
        family = try c.decodeIfPresent(LTXFamily.self, forKey: .family) ?? .ltx23
        // Absent ⇒ nil ⇒ FOLLOW THE PROFILE. Safe for configs persisted before this key existed:
        // the advice is `.bf16` on standard64/max128 (the old behaviour) and 2.5 was never
        // admissible on the low tiers, so no persisted 2.5 config sits on a profile that advises
        // otherwise. (The bernini d02cfa1 rule still holds — a new key never becomes mandatory.)
        textEncoderQuant = try c.decodeIfPresent(Quant.self, forKey: .textEncoderQuant)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        gemmaRepo = try c.decodeIfPresent(String.self, forKey: .gemmaRepo)
            ?? family.defaultGemmaRepo ?? "mlx-community/gemma-3-12b-it-4bit"
        transformerRepo = try c.decodeIfPresent(String.self, forKey: .transformerRepo)
        quant = try c.decode(Quant.self, forKey: .quant)
        profile = try c.decodeIfPresent(LTX2Profile.self, forKey: .profile)
    }
}

// MARK: - Weight sources (auto-materialization, BRIDGE M4 / engine MAT gate)

extension LTX2Configuration: WeightSourcing {
    /// The component files LTX-2.3 needs beyond the transformer (optional features included —
    /// a first run materializes the full experience; the pipeline degrades per missing file
    /// only for explicit-dir setups).
    /// ⚠️ **The `_config.json` SIDECARS are not optional.** `Upsampler.peekVariant` resolves which
    /// variant a checkpoint is (x2 / x1.5-rational / temporal-x2) by reading `<stem>_config.json`
    /// WITHOUT touching the weights — so a fetch that brings the safetensors and drops the sidecar
    /// delivers a file the pipeline cannot classify.
    ///
    /// ⟲ **CORRECTED 2026-08-21.** This list carried ONE upscaler and NO sidecars, so 2.3 could
    /// fetch only x2 and could not identify even that. `componentFiles25` had them all along —
    /// 2.3 was simply left behind when the variant support landed, and the omission was invisible
    /// because every local tree has the files on disk already. **The port supports and GATES all
    /// three variants (`--upsampler-variants-gate`, `--two-stage-variants-gate`); two of them were
    /// unreachable for anyone fetching from the declared repo.**
    static let componentFiles = [
        "connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
        "audio_vae.safetensors", "vocoder.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors", "spatial_upscaler_x2_v1_1_config.json",
        "temporal_upscaler_x2_v1_0.safetensors", "temporal_upscaler_x2_v1_0_config.json",
    ]

    /// 2.3-only. ⚠️ **The x1.5-rational upscaler exists for 2.3 and NOT for 2.5** — putting it in
    /// the shared list makes every 2.5 arm demand a file its tree has never contained. Caught by
    /// the reach gate's file-completeness pass within a minute of making exactly that mistake.
    static let componentFiles23 = [
        "spatial_upscaler_x1_5_v1_0.safetensors", "spatial_upscaler_x1_5_v1_0_config.json",
    ]

    /// 2.5-only components, on top of `componentFiles`. The Gemma-4 encoder shards are matched by
    /// directory glob because they live in-tree (`gemma4-12b-ltx-v1/`) rather than in their own
    /// repo — which is also what `LTX2Pipeline.isLTX25` keys off, so a components fetch that
    /// dropped them would make the tree resolve as 2.3 and silently run the wrong pipeline.
    /// 2.5-only, ON TOP of `componentFiles` — which now carries every upscaler variant and its
    /// sidecar for both families, so the entries that used to be duplicated here are gone.
    static let componentFiles25 = [
        "gemma4-12b-ltx-v1/*",
        "config.json", "embedded_config.json",
        "vae_diffusion_decoder.safetensors", "duration_head.safetensors",
    ]

    /// The repo serving the quantized transformer; bf16 rides the components repo.
    ///
    /// ⚠️ The suffix is FAMILY-dependent — see `LTXFamily.transformerRepoSuffix`. On 2.5, `-q8` is
    /// the text-encoder sibling and would silently serve a bf16 DiT under an int8 declaration.
    public var effectiveTransformerRepo: String? {
        if let transformerRepo { return transformerRepo }
        guard let suffix = family.transformerRepoSuffix(for: quant) else { return nil }
        return repo + suffix
    }

    /// The components repo actually materialized — `repo`, or its quantized-encoder sibling when
    /// `textEncoderQuant` names one. Everything but the encoder is identical in that tree (the
    /// sibling symlinks the rest back), so this swaps the encoder and nothing else.
    public var effectiveComponentsRepo: String {
        guard let suffix = family.textEncoderRepoSuffix(for: effectiveTextEncoderQuant) else {
            return repo
        }
        return repo + suffix
    }

    public var weightSources: [WeightSource] {
        var componentGlobs = Self.componentFiles
        if family == .ltx25 { componentGlobs.append(contentsOf: Self.componentFiles25) }
        if family == .ltx23 { componentGlobs.append(contentsOf: Self.componentFiles23) }
        var sources = [
            WeightSource(role: "components", repo: effectiveComponentsRepo, revision: revision,
                         matching: componentGlobs),
        ]
        // 🔑 STREAMING REPLACES the transformer download, it does not add to it. The granule tree
        // is a re-layout of the same bytes plus a globals sidecar, and `bindStore` serves globals
        // from that sidecar when the checkpoint is absent (manifest v2's whole purpose). Fetching
        // both would double the DiT's disk for no benefit — 70 GB instead of 35 on bf16.
        if effectiveStreamedBlocks {
            sources.append(WeightSource(role: "granules", repo: family.granuleRepo))
        }
        // 2.5's Gemma-4 encoder lives INSIDE the components tree, so it is not a separate source —
        // declaring one would materialize an unrelated Gemma-3 repo the 2.5 pipeline never opens.
        if family.defaultGemmaRepo != nil || family == .ltx23 {
            sources.append(WeightSource(role: "text-encoder", repo: gemmaRepo))
        }
        // 🔑 The transformer is its OWN source, from the quant sibling when one exists and
        // otherwise from the BASE repo — never from `effectiveComponentsRepo`. It used to ride the
        // components glob when there was no sibling (bf16), which meant an encoder-swapped config
        // demanded the 35 GB bf16 transformer from the `-q8` tree: 35 GB of duplication to satisfy
        // a combination nobody ships. Streaming replaces this source entirely.
        if !effectiveStreamedBlocks {
            sources.append(WeightSource(role: "transformer-\(quant.rawValue)",
                                        repo: effectiveTransformerRepo ?? repo,
                                        matching: [Self.defaultTransformerFile]))
        }
        return sources
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        func storeHas(_ repo: String, files: [String]) -> Bool {
            guard let dir = ModelStore(root: storeRoot).directory(for: repo) else { return false }
            return files.allSatisfy { fm.fileExists(atPath: dir.appending(path: $0).path) }
        }
        return weightSources.filter { source in
            switch source.role {
            case "components":
                if let dir = ltxDirectory,
                   fm.fileExists(atPath: dir.appending(path: Self.componentFiles[0]).path) {
                    // On 2.5 the encoder rides this source, and its absence is not cosmetic:
                    // `isLTX25` keys off that directory, so a components tree missing it resolves
                    // as 2.3 and silently runs the wrong pipeline. Treat it as still-missing.
                    if family == .ltx25, !LTX2Pipeline.isLTX25(ltxDir: dir) { return true }
                    return false
                }
                // `matching` carries globs on 2.5 (`gemma4-12b-ltx-v1/*`); a literal existence check
                // on a glob never succeeds, so it would report the tree perpetually missing.
                let literals = (source.matching ?? []).filter { !$0.contains("*") }
                return !storeHas(source.repo, files: literals)
            case "text-encoder":
                if let dir = gemmaDirectory,
                   fm.fileExists(atPath: dir.appending(path: "config.json").path) { return false }
                return !storeHas(source.repo, files: ["config.json"])
            default:   // transformer-<quant>
                if let path = transformerPath, fm.fileExists(atPath: path.path) { return false }
                return !storeHas(source.repo, files: [Self.defaultTransformerFile])
            }
        }
    }

    /// The configuration with nil directories resolved to the store layout — what `load()` uses
    /// AFTER materialization. Explicit directories always win.
    public func resolved(storeRoot: URL?) -> LTX2Configuration {
        let store = ModelStore(root: storeRoot)
        var cfg = self
        if cfg.ltxDirectory == nil { cfg.ltxDirectory = store.directory(for: repo) }
        if cfg.gemmaDirectory == nil {
            // 2.5's encoder is in-tree. `LTX2Pipeline` derives `gemma4Dir` from `ltxDir` and ignores
            // whatever is passed, but `MLXLTX2Package.load` REQUIRES a non-nil gemmaDirectory — so
            // resolving this to a Gemma-3 store path that 2.5 never materializes would leave it nil
            // and throw `configurationMismatch` on a perfectly complete 2.5 tree.
            cfg.gemmaDirectory = family == .ltx25
                ? cfg.ltxDirectory.map { LTX2Pipeline.gemma4Dir(ltxDir: $0) }
                : store.directory(for: gemmaRepo)
        }
        if cfg.transformerPath == nil, let tRepo = effectiveTransformerRepo {
            cfg.transformerPath = store.directory(for: tRepo)?.appending(path: Self.defaultTransformerFile)
        }
        // The granule ROOT resolves like any other repo; `resolvedGranuleDirectory` appends the
        // quant subdir the published trees are laid out with (`bf16/`, `q8/`, `q4/`).
        if cfg.granuleRootDirectory == nil, cfg.effectiveStreamedBlocks {
            cfg.granuleRootDirectory = store.directory(for: family.granuleRepo)
        }
        return cfg
    }
}

/// Config-aware footprint (contract 1.14 `FootprintConfigured`): what the governor charges for THIS
/// registration, overriding the per-quant `QuantFootprint` when the config's resolved shape differs
/// from the quant-keyed declaration.
///
/// 🔑 **2.5 low tiers STREAM, and streaming is not a property of the quant** (AB-T-0069): the same
/// int8 checkpoint is 23 GB resident on the resident path and ~0.5 GB resident + ~15 GB activation
/// streamed. The per-quant declaration (23+5 = 28) therefore REFUSED `compact24`/`balanced32` —
/// tiers that measure 15.49 / 17.14 against budgets of 16.8 / 22.4 (AB-R-0106, worst of 18 runs).
/// These hints carry the measured split for the RESOLVED lane; `nil` falls through to the quant
/// numbers, which stay the honest envelope for every resident lane.
///
/// ⚠️ **Streamed hints are declared ONLY where streaming is GUARANTEED**: family `.ltx25`, streaming
/// resolved on, and the gate PINNED (`recommendedForcedStreamGate` advisory or explicit `true`).
/// Never on `.auto` — the runtime gate may fall back resident output-invisibly (AB-R-0105: 1 run in
/// 3 at compact24's N), and a streamed declaration over a resident run under-declares — fail-OPEN,
/// worse than the fail-closed refusal this extension exists to fix (AB-A-0012). The corollary: an
/// explicit `forceStreamGate = false` / `streamedBlocks = false` on a low tier gets RESIDENT
/// numbers and is refused. That refusal is correct — the run it describes would bust the budget.
/// `MLXLTX2Package.load()` enforces the same invariant from the other side: a pinned-tier config
/// whose granule tree cannot be resolved throws instead of silently building a resident DiT.
///
/// ⚠️ **Family-keyed, deliberately.** This one config type serves 2.3 (resident low tiers, int4)
/// and 2.5 (streamed low tiers, int8) — and the engine reads hints from the config AS HANDED, not
/// from `MLXLTX25Package.coerced()`'s copy (`MLXServeEngine.swift:436`). Register 2.5 with
/// `family: .ltx25`; a family-defaulted config gets 2.3's numbers and fails CLOSED on the low
/// tiers. Gate cases 39–43 pin all of this.
extension LTX2Configuration: FootprintConfigured {
    /// True when this registration is guaranteed to stream the DiT — the ONLY condition under
    /// which the streamed footprint may be declared (see the fail-open note above).
    private var declaresStreamedFootprint: Bool {
        family == .ltx25
            && effectiveStreamedBlocks
            && resolvedStreamingOptions.gatePolicy == .forceStream
    }

    /// Expected weight bytes READ per run for the resolved lane (contract 1.35.0, AB-A-0013).
    ///
    /// 🔑 PERFORMANCE ONLY — never refuses. Deliberately separate from the crash floor
    /// (`minSustainedReadBytesPerSecond`, declared on bf16 only): a slow volume makes a STREAMED
    /// lane slow, not unsafe. Declaring the volume lets the ENGINE project I/O time from its own
    /// measured B/s instead of every app re-deriving per-tier sweep arithmetic (AB-D-0038).
    ///
    /// **Streamed lanes re-read the swept set EVERY step** — only ~1.53 GiB of slots stay resident
    /// against an 18.37 GiB sweep, so the working set cannot be carried between steps. **Resident
    /// lanes read once, at load.** Measured: 33 s of I/O at 4.4 GiB/s inside a 47–68 s compact24
    /// clip; the same clip is ~334 s of I/O on a ~475 MB/s volume, with correct output throughout.
    ///
    /// ⚠️ Steps are the SIGMA SCHEDULE, not a guess: one-stage tiers run `distilledSigmas`
    /// (9 entries → 8 transitions); two-stage adds `stage2Sigmas` (4 → 3), so 11.
    /// ⚠️ An `.auto` lane that falls back resident reads far LESS than this. Over-projecting I/O on
    /// a fallback is the safe direction for an advisory (it never refuses), but do not read this
    /// number as a measurement of a fallback run.
    public var expectedWeightReadBytesPerRunHint: UInt64? {
        guard family == .ltx25, let profile else { return nil }
        // int8 sweep MEASURED from the live streamer: "sweep 18.37 GiB" at both compact24 and
        // standard64. bf16 scaled by the on-disk DiT ratio (37.99 / 20.6 = 1.844) — 2.5's int8
        // quantizes only the transformer-block Linears, which is exactly what gets swept.
        let sweepInt8: UInt64 = 19_724_000_000
        let sweepBf16: UInt64 = 36_380_000_000
        if effectiveStreamedBlocks {
            let steps: UInt64 = profile.oneStage ? 8 : 11
            return (quant == .bf16 ? sweepBf16 : sweepInt8) * steps
        }
        // Resident: the checkpoint is read once at load. On-disk sizes.
        return quant == .bf16 ? 37_990_000_000 : 20_600_000_000
    }

    public var residentBytesHint: UInt64? {
        guard declaresStreamedFootprint, let profile else { return nil }
        switch profile {
        case .compact24, .balanced32:
            // Measured resident while streaming: 0.41–0.63 GB (slots + globals + LoRA factors)
            // across all 18 runs (AB-R-0106). Declared 0.8 GB.
            return 800_000_000
        default:
            return nil   // standard64/max128: the quant-keyed resident stays the honest envelope
        }
    }

    public var peakActivationBytesHint: UInt64? {
        guard let profile else { return nil }   // fall back to the per-quant QuantFootprint
        if declaresStreamedFootprint {
            // MEASURED 2026-08-21 (AB-R-0106: 3 tiers × 2 modes × 3 reps, worst-case per tier,
            // streamed DiT + int8 encoder at each profile's own clamped envelope). i2v is the
            // binding mode on both tiers. Declared = worst measured + headroom, inside the
            // corridor [worst measured, tier budget]:
            //   compact24  worst 15.49 (i2v, 92% of 16.8) → charge 0.8 + 15.4 = 16.2 (96% of budget)
            //   balanced32 worst 17.14 (i2v, 77% of 22.4) → charge 0.8 + 17.2 = 18.0 (80% of budget)
            // compact24's corridor is inherently narrow — the operator accepted the 92% measurement
            // explicitly (AB-D-0035), and headroom-over-measured beats distance-from-budget here:
            // the budget side is exact arithmetic, the measurement side is content-sensitive.
            switch profile {
            case .compact24:  return 15_400_000_000
            case .balanced32: return 17_200_000_000
            default: break    // standard64 streams by default too, but declares below
            }
        }
        switch family {
        case .ltx25:
            switch profile {
            case .compact24, .balanced32:
                // Resident/unpinned 2.5 low tier: quant numbers (23+5) → refused. Correct —
                // that configuration measures 31.92/33.13 GB resident and busts the budget.
                return nil
            case .standard64:
                // int8 23+5 = 28 covers BOTH lanes: streamed 19.43 (AB-R-0106) and the `.auto`
                // resident fallback 27.31 (AB-R-0105) — which is why standard64 may keep `.auto`.
                // (Replaces the 2.3-era 16 GB hint, which charged 39 for a lane measuring ≤27.31.)
                return nil
            case .max128:
                return 36_000_000_000  // resident bf16 lane, unchanged: 481f i2v spot, charge 40+36=76
            }
        case .ltx23:
            // 2.3-era hints, UNCHANGED (T3b+T3c LOW-TIER-PLAN measurements, 2026-07-01): the hint
            // is peak − the declared per-quant residentBytes, so charge equals measured stage-max:
            //   compact24  peak 15.36 (budget 16.8 ✓)   balanced32 peak 16.07 (22.4 ✓)
            //   standard64 peak 37.51 (44.8 ✓)          max128     peak 92.2  (0.85×128 ✓)
            switch profile {
            case .compact24:  return 3_000_000_000    // 15.36 − 13 (int4 resident) + headroom
            case .balanced32: return 4_000_000_000    // 16.07 − 13 + headroom
            case .standard64: return 16_000_000_000   // 37.51 − 22 (int8 resident) + headroom
            case .max128:     return 36_000_000_000   // 481f i2v spot: peak 72.73, charge 40+36=76
            }
        }
    }
}

/// Cold-start weight prewarm (engine ≥0.7.0): page the LTX + Gemma weight files into the OS
/// file cache before `load()` runs its GPU evals, so the cold load-time `eval` never faults
/// weights off slow/external storage inside a live Metal command buffer (the I5 cold-load
/// `kIOGPUCommandBufferCallbackErrorTimeout`). Only the config knows these resolved `/Volumes`
/// paths — execution is the engine's (`WeightPrewarmer`, best-effort).
///
/// **Override-aware exclusion.** An override means the default file it replaces is NEVER loaded,
/// so paging the whole dir reads that file's cold cost for nothing — ~35 GB for the bf16
/// transformer under a q8/q4 override, ~0.8 GB for the stock decoder under a lean-decoder
/// override. With any override set we page `ltxDirectory`'s weight files *individually, minus the
/// superseded defaults*, plus the overrides. With no override the simple whole-dir prewarm stands.
extension LTX2Configuration: WeightPrewarming {
    /// Basename of the bf16 transformer that a `transformerPath` override replaces.
    static let defaultTransformerFile = "transformer-distilled.safetensors"
    /// Basename of the stock video decoder that a `vaeDecoderPath` override replaces.
    static let defaultVAEDecoderFile = "vae_decoder.safetensors"

    public var prewarmPaths: [URL] {
        // Operate on the store-resolved view so auto-materialize (nil-dir) configs prewarm the
        // downloaded layout on later cold launches; not-yet-downloaded paths simply don't exist
        // and the prewarmer skips them (best-effort). Explicit dirs resolve to themselves.
        let r = resolved(storeRoot: modelsRootDirectory)
        // Streamed blocks: NEVER prewarm the transformer checkpoint — the blocks refill from
        // granules (F_NOCACHE by design) and only the ~58 small globals load from it; paging
        // the full 20–38 GB file would defeat the point (the quant-aware prewarm lesson, one
        // level up).
        let transformer = effectiveStreamedBlocks ? nil : r.transformerPath
        guard let ltxDir = r.ltxDirectory else {
            return [r.gemmaDirectory, transformer, r.vaeDecoderPath].compactMap { $0 }
        }
        return Self.prewarmPaths(ltxDir: ltxDir, transformerPath: transformer,
                                 vaeDecoderPath: r.vaeDecoderPath, gemmaDirectory: r.gemmaDirectory,
                                 excludeDefaultTransformer: effectiveStreamedBlocks)
    }

    private static func prewarmPaths(ltxDir: URL, transformerPath: URL?, vaeDecoderPath: URL?,
                                     gemmaDirectory: URL?, excludeDefaultTransformer: Bool = false)
        -> [URL]
    {
        let overrides = [transformerPath, vaeDecoderPath].compactMap { $0 }
        // No override: whole LTX dir + Gemma — every weight file in ltxDir IS used…
        if overrides.isEmpty && !excludeDefaultTransformer {
            return [ltxDir, gemmaDirectory].compactMap { $0 }
        }
        var superseded = Set<URL>()
        if transformerPath != nil || excludeDefaultTransformer {
            superseded.insert(ltxDir.appendingPathComponent(defaultTransformerFile).standardizedFileURL)
        }
        if vaeDecoderPath != nil {
            superseded.insert(ltxDir.appendingPathComponent(defaultVAEDecoderFile).standardizedFileURL)
        }
        let ltxWeights = ((try? FileManager.default.contentsOfDirectory(
            at: ltxDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" && !superseded.contains($0.standardizedFileURL) }
        // Correctness-first fallback: if enumeration turned up nothing, page the whole dir.
        var paths = ltxWeights.isEmpty ? [ltxDir] : ltxWeights
        paths.append(contentsOf: overrides)
        if let gemma = gemmaDirectory { paths.append(gemma) }
        // An override living inside ltxDir is already in the enumeration — don't page it twice.
        var seen = Set<URL>()
        return paths.filter { seen.insert($0.standardizedFileURL).inserted }
    }
}
