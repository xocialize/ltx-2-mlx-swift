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

        // ⚠️ decode reports per-chunk ONLY on long clips: chunking requires
        // `latentFrames > vaeChunkFrames + 2*halo`, which is FALSE at e.g. 1080p x 121f (16 vs 18).
        // So `emitsSubSteps` is deliberately false here — the plan must not promise counts the run
        // will not deliver, and this is exactly the case where decode is slowest (AB-R-0118).
        stages.append(RunStage(
            id: "decode", phase: "decode", title: "Render frames",
            detail: "Decoding latents to pixels and audio.",
            emitsProgress: true, emitsSubSteps: false, expectedSubSteps: nil))
        stages.append(RunStage(
            id: "postprocess", phase: "postprocess", title: "Finish",
            detail: "Muxing video and audio into the final file.",
            emitsProgress: true, emitsSubSteps: false, expectedSubSteps: nil))
        return stages
    }
}
