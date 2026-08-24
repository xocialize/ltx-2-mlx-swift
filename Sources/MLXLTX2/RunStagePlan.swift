// RunStagePlan.swift — the PRE-RUN stage list a stepper UI needs on first paint.
//
// 🔑 WHY THIS EXISTS. `RunProgress` tells a host which phase is happening NOW. A stepper cannot be
// drawn from that: the connecting line needs every node before the first event arrives, and the
// node list is CONFIGURATION-DEPENDENT. Measured 2026-08-22:
//
//   compact24  (one-stage): encode -> denoise -> decode -> postprocess              = 4 nodes
//   standard64 (two-stage): encode -> denoise(1/2) -> upsample -> denoise(2/2)
//                           -> decode -> postprocess                                = 6 nodes
//
// The events DO distinguish them (one-stage denoise carries no `stage`), but only after they
// arrive — too late to draw the line. Hence a plan resolved up front from the same config the
// pipeline will use.
//
// ⚠️ SCOPE: this covers OUR run. Prompt enhancement is a DIFFERENT package (engine-governed
// `GemmaLLMPackage`), so a host that enhances composes its own node ahead of these. Documented
// rather than faked — inventing a node for work this package does not perform would make the plan
// lie about who reports what.

import Foundation
import LTX2
import MLXToolKit

/// One node in a run's stage plan — a stepper node, plus what its card can expect to show.
public struct RunStage: Sendable, Equatable, Codable {
    /// Stable unique id, e.g. `"denoise.2"`. Distinct per occurrence.
    public let id: String
    /// The `RunPhase` raw value whose events drive this node, or `nil` when the work emits no
    /// phase today (see `emitsProgress`).
    public let phase: String?
    /// 1-based occurrence of `phase` within the run — `denoise` appears twice in two-stage.
    /// Correlate with the event's `stage` field when `totalStages` is present.
    public let occurrence: Int
    /// Short label for the stepper node.
    public let title: String
    /// One line for the card.
    public let detail: String
    /// False when this node does NOT report through `RunProgress` at all — the UI should show it
    /// as work-in-progress without expecting events. ⚠️ Honest by design: see `weights` / `adapter`.
    public let emitsProgress: Bool
    /// Whether the phase emits `step`/`totalSteps` a card can display.
    public let emitsSubSteps: Bool
    /// Expected sub-step total when it is knowable BEFORE the run, else `nil`.
    /// ⚠️ `nil` means "genuinely unknown up front", never "zero".
    public let expectedSubSteps: Int?

    public init(id: String, phase: String?, occurrence: Int = 1, title: String, detail: String,
                emitsProgress: Bool, emitsSubSteps: Bool, expectedSubSteps: Int?) {
        self.id = id; self.phase = phase; self.occurrence = occurrence
        self.title = title; self.detail = detail
        self.emitsProgress = emitsProgress; self.emitsSubSteps = emitsSubSteps
        self.expectedSubSteps = expectedSubSteps
    }
}

public extension LTX2Configuration {
    /// The ordered stage plan for a run under THIS configuration.
    ///
    /// - Parameters:
    ///   - initImage: the request carries an init image (i2v) — adds the adapter-fetch node.
    ///   - storeRoot: engine model store; when supplied and sources are missing, a materialize
    ///     node is prepended. Pass `nil` to omit that check.
    ///   - adapterCached: whether the i2v adapter is already local (`LoRACache.isCached`). When
    ///     `false` the fetch node is included — it is a **4.93 GB** download that happens INSIDE
    ///     `run()` (AB-R-0114), so a UI that omits it shows a stall with no explanation.
    func plannedStages(initImage: Bool = false,
                       storeRoot: URL? = nil,
                       adapterCached: Bool = true) -> [RunStage] {
        var stages: [RunStage] = []

        // ── Pre-generation work that does NOT report through RunProgress ─────────────────────
        if let storeRoot, !missingWeightSources(storeRoot: storeRoot).isEmpty {
            stages.append(RunStage(
                id: "weights", phase: nil, title: "Download model",
                detail: "Fetching model weights — first run only.",
                emitsProgress: false,      // engine-side WeightDownloadProgress, not RunProgress
                emitsSubSteps: false, expectedSubSteps: nil))
        }
        if initImage && !adapterCached {
            stages.append(RunStage(
                id: "adapter", phase: nil, title: "Prepare image adapter",
                detail: "One-time 4.93 GB download for image-to-video.",
                emitsProgress: false,      // fetched inside run(); no phase is emitted
                emitsSubSteps: false, expectedSubSteps: nil))
        }

        // ── The generation phases ───────────────────────────────────────────────────────────
        // ⚠️ encode's sub-step total is the ENCODER's layer count. 48 for the shipping Gemma-4
        // and Gemma-3 checkpoints; declared here because the plan is built before weights load,
        // so it is the known-checkpoint value rather than a read one.
        stages.append(RunStage(
            id: "encode", phase: "encode", title: "Read prompt",
            detail: "Encoding the prompt through the text encoder.",
            emitsProgress: true, emitsSubSteps: true, expectedSubSteps: 48))

        let twoStage = !(profile?.oneStage ?? false)
        let s1 = Positions.distilledSigmas.count - 1
        if twoStage {
            let s2 = Positions.stage2Sigmas.count - 1
            stages.append(RunStage(
                id: "denoise.1", phase: "denoise", occurrence: 1, title: "Generate",
                detail: "First pass — composing the video and audio.",
                emitsProgress: true, emitsSubSteps: true, expectedSubSteps: s1))
            // upsample stays a BARE MARKER, deliberately (AB-T-0079 explicitly scoped it as
            // "do it only if a natural loop exists; do NOT manufacture one"). It is one encoder
            // call plus one upsampler call. The upsampler does contain `for i in 0..<4` res-block
            // loops, but those are an implementation detail, not user-visible units of work —
            // emitting 8 ticks for a step that finishes in seconds is precisely the fake heartbeat
            // the ticket forbids, and a UI would render it as progress. Decode was fixed because
            // its units (decode windows) are real; this one has none, so it stays honest and
            // indeterminate.
            stages.append(RunStage(
                id: "upsample", phase: "upsample", title: "Upscale",
                detail: "Raising the latent to the target resolution.",
                emitsProgress: true, emitsSubSteps: false, expectedSubSteps: nil))
            stages.append(RunStage(
                id: "denoise.2", phase: "denoise", occurrence: 2, title: "Refine",
                detail: "Second pass — adding detail at full resolution.",
                emitsProgress: true, emitsSubSteps: true, expectedSubSteps: s2))
        } else {
            stages.append(RunStage(
                id: "denoise.1", phase: "denoise", occurrence: 1, title: "Generate",
                detail: "Composing the video and audio.",
                emitsProgress: true, emitsSubSteps: true, expectedSubSteps: s1))
        }

        // decode now ALWAYS reports countable sub-steps (AB-T-0079). It used to report per-CHUNK
        // only, and chunking requires `latentFrames > vaeChunkFrames + 2*halo` — false at
        // 1080p x 121f (16 vs 18), i.e. silent exactly where decode is slowest (AB-R-0118). The
        // fix was not a new counter but a better unit: the phase counts DECODE WINDOWS
        // (chunks x spatial tiles), and HD geometries are tiled 2x2, so the units were already
        // there. A short 704x512 clip is one window and honestly reports 1/1.
        //
        // `expectedSubSteps` stays nil ON PURPOSE. The count is chunks x tiles, which depends on
        // resolution, frame count, and the LTX_VAE_CHUNK / LTX_VAE_HALO / LTX_VAE_TILES overrides —
        // none of which this plan is given. Promising a number here that the run then contradicts
        // would be worse than promising none: render the node indeterminate until its first report
        // carries the real total.
        stages.append(RunStage(
            id: "decode", phase: "decode", title: "Render frames",
            detail: "Decoding latents to pixels and audio.",
            emitsProgress: true, emitsSubSteps: true, expectedSubSteps: nil))
        stages.append(RunStage(
            id: "postprocess", phase: "postprocess", title: "Finish",
            detail: "Muxing video and audio into the final file.",
            emitsProgress: true, emitsSubSteps: false, expectedSubSteps: nil))
        return stages
    }
}

/// What a run will ACTUALLY produce, resolved from a request before the run starts (AB-T-0080).
///
/// The pipeline silently transforms geometry in three ways, and a caller previously discovered all
/// of them in the exported file:
///   * the tier envelope CLAMPS width/height/frames,
///   * spatial dims snap DOWN to the latent grid,
///   * frame counts snap DOWN to the 8k+1 grid — asking for 24 frames delivers **17**.
public struct ResolvedGeometry: Sendable, Equatable {
    /// Geometry the pipeline will run at.
    public let width: Int, height: Int, numFrames: Int
    /// Pixel frames the FILE will contain. Differs from `numFrames` whenever the request was not
    /// already on the 8k+1 grid: the pipeline rounds to whole latent frames and emits `8·F−7`.
    public let deliveredFrames: Int
    public let fps: Double
    /// Exactly what was asked for, so a host can explain the difference rather than just show it.
    public let requestedWidth: Int, requestedHeight: Int, requestedNumFrames: Int

    public var widthChanged: Bool { width != requestedWidth }
    public var heightChanged: Bool { height != requestedHeight }
    public var framesChanged: Bool { deliveredFrames != requestedNumFrames }
    public var changed: Bool { widthChanged || heightChanged || framesChanged }

    /// Human-readable reasons, in the order they were applied. Empty when nothing changed.
    public let notes: [String]

    public var summary: String {
        changed
            ? "\(requestedWidth)×\(requestedHeight)×\(requestedNumFrames)f → "
              + "\(width)×\(height)×\(deliveredFrames)f"
            : "\(width)×\(height)×\(deliveredFrames)f"
    }
}

public extension LTX2Configuration {
    /// Resolve a request to the geometry a run will actually use and deliver.
    ///
    /// 🔑 THIS IS THE SINGLE SOURCE OF TRUTH — `MLXLTX2Package.run` calls it rather than repeating
    /// the rules. A resolver that merely *describes* the pipeline drifts from it, and a geometry
    /// preview that disagrees with the run is worse than none.
    ///
    /// - Parameter envelopeOverride: mirrors `LTX_ENVELOPE_OVERRIDE=1`, the measurement-only hatch
    ///   that lifts the RESOLUTION clamp (never the frame clamp).
    func resolvedGeometry(width: Int? = nil, height: Int? = nil, numFrames: Int? = nil,
                          fps: Double? = nil, envelopeOverride: Bool = false) -> ResolvedGeometry {
        let reqW = width ?? 704, reqH = height ?? 512, reqF = numFrames ?? 9
        var w = reqW, h = reqH, nf = reqF
        var notes: [String] = []

        if let p = profile {
            if envelopeOverride {
                if nf > p.maxFrames {
                    notes.append("frames clamped to the \(p.rawValue) cap (\(p.maxFrames)); the "
                        + "override lifts resolution only")
                }
                nf = min(nf, p.maxFrames)
            } else {
                if w > p.maxWidth || h > p.maxHeight {
                    notes.append("clamped to the \(p.rawValue) envelope "
                        + "(\(p.maxWidth)×\(p.maxHeight))")
                }
                if nf > p.maxFrames {
                    notes.append("frames clamped to the \(p.rawValue) cap (\(p.maxFrames))")
                }
                w = min(w, p.maxWidth); h = min(h, p.maxHeight); nf = min(nf, p.maxFrames)
            }
            let preSnapW = w, preSnapH = h
            w = max(64, (w / 32) * 32); h = max(64, (h / 32) * 32)   // latent grid is /32
            if w != preSnapW || h != preSnapH {
                notes.append("snapped down to the /32 latent grid")
            }

            // ⚠️ TWO-STAGE NEEDS /64, NOT /32. The shipping spatial-x2 upsampler requires an EVEN
            // stage-2 latent grid, so a /32-but-not-/64 target makes `resolveTwoStageGeometry`
            // THROW — after the text encode has already run. Before this, a 1200-wide request on
            // standard64 snapped to 1184 (latent 37, odd) and the run FAILED rather than being
            // reduced. Resolving to a runnable geometry is the whole point of this API.
            //
            // Applied only when two-stage will actually run: the one-stage tiers have no upsampler
            // in the loop, and compact24's 288 ceiling is an odd latent (9) that is perfectly legal
            // there.
            if !(profile?.oneStage ?? false) {
                let pre = (w, h)
                w = max(64, (w / 64) * 64); h = max(64, (h / 64) * 64)
                if (w, h) != pre {
                    notes.append("snapped down to /64 for the two-stage spatial upsampler")
                }
            }
        }

        // Frames: the pipeline rounds UP to whole latent frames and emits 8·F−7 pixels, so any
        // request off the 8k+1 grid comes back SHORTER (24 → 17).
        let fLat = (nf + 7) / 8
        let delivered = 8 * fLat - 7
        if delivered != reqF {
            notes.append("frames land on the 8k+1 grid (\(reqF) → \(delivered))")
        }

        return ResolvedGeometry(
            width: w, height: h, numFrames: nf, deliveredFrames: delivered, fps: fps ?? 24,
            requestedWidth: reqW, requestedHeight: reqH, requestedNumFrames: reqF, notes: notes)
    }
}
