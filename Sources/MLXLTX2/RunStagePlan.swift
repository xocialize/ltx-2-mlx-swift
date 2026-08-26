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

/// Which edit path's geometry rules to apply. They genuinely differ, which is exactly why a host
/// cannot reproduce them and why this enum exists rather than a `Bool`.
public enum EditGeometryMode: String, Sendable, CaseIterable {
    /// retake / extend / replace-audio-and-video: dims come from the SOURCE and are never upscaled
    /// above it, then snap DOWN to /32.
    case retake
    /// a2v: an explicit request WINS over the source (there is no source video to match — the video
    /// is generated from noise against the track), then snaps DOWN to /64.
    case audioToVideo
}

public extension LTX2Configuration {
    /// Resolve a SOURCE-derived edit request to the geometry the run will use and deliver.
    ///
    /// 🔑 Companion to the request-derived `resolvedGeometry(width:height:...)`, and the same rule
    /// applies: `runVideoEdit` and `runAudioToVideo` CALL this rather than repeating the arithmetic.
    /// The edit sheets resolve from the source clip, so without this they display source dims that
    /// the run then snaps down — promising something the pipeline does not deliver (AB-T-0094).
    ///
    /// `requested*` carries the SOURCE dims, not a request: that is what the sheet already shows and
    /// what the user is comparing the result against.
    ///
    /// 🚨 `sourceDurationSeconds` MUST BE THE **CONTAINER** DURATION — `AVAsset.duration`, which is
    /// what `runVideoEdit` feeds this. It is NOT the track duration, and the two differ: AVFoundation
    /// reports an AAC track ~23 ms shorter than its container (10.018 vs 10.041 measured on the app's
    /// fixture, ltx-studio 2026-08-25), because of encoder delay/padding.
    ///
    /// That tiny difference AMPLIFIES through the 8k+1 floor, and worst exactly where it is most
    /// likely to matter — on a clip whose frame count lands ON the grid:
    ///
    ///     container 10.041s × 24 = 240.98 → 241 → floors to 241
    ///     track     10.018s × 24 = 240.43 → 240 → floors to **233**
    ///
    /// 8 frames — a third of a second — from a 23 ms disagreement, silently. Prefer the
    /// `sourceAsset:` overload below, which cannot be got wrong.
    ///
    /// ⚠️ THREE WAYS THIS DIFFERS FROM THE REQUEST-DERIVED RESOLVER — none are incidental:
    ///  1. **Snap grid is per-mode, not per-tier.** Generation picks /32 or /64 by whether two-stage
    ///     runs; the edit paths pin it by MODE (retake /32, a2v /64) regardless of tier.
    ///  2. **retake never upscales.** An explicit width above the source is pulled back down to it
    ///     (the vendor's `video_resolution.py` rule). a2v has no source video to match, so there an
    ///     explicit request wins outright.
    ///  3. **Frames are floored to the grid with a floor of 9**, where generation rounds UP to whole
    ///     latent frames. Below 9 they diverge: generation delivers 1, the edit paths deliver 9.
    ///     `numFrames == deliveredFrames` here because the edit paths run at the snapped count.
    func resolvedGeometry(sourceWidth: Int, sourceHeight: Int, sourceDurationSeconds: Double,
                          mode: EditGeometryMode, width: Int? = nil, height: Int? = nil,
                          numFrames: Int? = nil, fps: Double? = nil) -> ResolvedGeometry {
        let f = fps ?? 24
        let reqW = sourceWidth, reqH = sourceHeight
        let reqF = numFrames ?? Int((sourceDurationSeconds * f).rounded())
        var notes: [String] = []

        var w: Int, h: Int
        switch mode {
        case .retake:
            w = min(width ?? sourceWidth, sourceWidth)
            h = min(height ?? sourceHeight, sourceHeight)
            if (width ?? 0) > sourceWidth || (height ?? 0) > sourceHeight {
                notes.append("held at the source size — a retake never upscales")
            }
        case .audioToVideo:
            w = width ?? sourceWidth
            h = height ?? sourceHeight
        }

        if let p = profile {
            if w > p.maxWidth || h > p.maxHeight {
                notes.append("clamped to the \(p.rawValue) envelope (\(p.maxWidth)×\(p.maxHeight))")
            }
            w = min(w, p.maxWidth); h = min(h, p.maxHeight)
        }

        let grid = mode == .audioToVideo ? 64 : 32
        let preW = w, preH = h
        w = max(64, (w / grid) * grid); h = max(64, (h / grid) * grid)
        if w != preW || h != preH {
            notes.append(mode == .audioToVideo
                ? "snapped down to /64 for the two-stage spatial upsampler"
                : "snapped down to the /32 latent grid")
        }

        var frames = reqF
        if let p = profile {
            if frames > p.maxFrames {
                notes.append("frames clamped to the \(p.rawValue) cap (\(p.maxFrames))")
            }
            frames = min(frames, p.maxFrames)
        }
        frames = max(9, ((frames - 1) / 8) * 8 + 1)
        if frames != reqF {
            notes.append("frames land on the 8k+1 grid (\(reqF) → \(frames))")
        }

        return ResolvedGeometry(
            width: w, height: h, numFrames: frames, deliveredFrames: frames, fps: f,
            requestedWidth: reqW, requestedHeight: reqH, requestedNumFrames: reqF, notes: notes)
    }
}

#if canImport(AVFoundation)
import AVFoundation

public extension LTX2Configuration {
    /// Resolve edit geometry directly from an asset, deriving duration, natural size and frame rate
    /// EXACTLY as `runVideoEdit` does.
    ///
    /// 🔑 PREFER THIS over the explicit-dimensions overload. It exists because the caller's only real
    /// freedom in that API — which duration to pass — is a trap: `AVAsset.duration` (container) and
    /// the audio track's duration differ by ~23 ms on AAC, and that difference can move the delivered
    /// frame count by 8 through the 8k+1 floor. Here the host cannot pick the wrong one, because it
    /// does not pick at all.
    ///
    /// Mirrors `runVideoEdit`'s fallbacks: 704×512 and 24 fps when there is no video track, which is
    /// the normal case for a2v against an audio-only container.
    func resolvedGeometry(sourceAsset asset: AVAsset, mode: EditGeometryMode,
                          width: Int? = nil, height: Int? = nil,
                          numFrames: Int? = nil, fps: Double? = nil) async throws -> ResolvedGeometry {
        let duration = try await asset.load(.duration).seconds
        let track = try await asset.loadTracks(withMediaType: .video).first
        var natural = CGSize(width: 704, height: 512)
        var nominalFPS: Double = 24
        if let track {
            natural = try await track.load(.naturalSize)
            nominalFPS = Double(try await track.load(.nominalFrameRate))
        }
        return resolvedGeometry(
            sourceWidth: Int(natural.width), sourceHeight: Int(natural.height),
            sourceDurationSeconds: duration, mode: mode,
            width: width, height: height, numFrames: numFrames, fps: fps ?? nominalFPS)
    }
}
#endif
