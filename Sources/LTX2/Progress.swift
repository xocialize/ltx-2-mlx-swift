// Progress.swift — core-owned ambient run-phase seam.
//
// The LTX2 core stays engine-free (MLXToolKit is a WRAPPER dependency by design — see
// Package.swift), so the core reports generation phases into its OWN task-local sink and the
// MLXLTX2 wrapper forwards events into the engine contract's `RunProgress` (contract 1.18.0,
// ENGINE-NEEDS V2). Same shape as the contract plane: coarse phase + optional 1-based
// step/totalSteps within the phase + stage/totalStages when a phase runs twice (two-stage
// denoise). No-op when unbound, so the CLI / gates run unchanged; RunLTX2 can bind a print
// sink for terminal progress if wanted.
//
// Task-local (not a pipeline property) so the binding scopes to exactly one generation —
// mirrors the engine's own WeightDownloadProgress/RunProgress pattern.

public enum LTX2Progress {
    /// Canonical phase names — match the engine contract's `RunPhase` constants verbatim so
    /// the wrapper forwards raw values with no mapping table.
    public enum Phase: String, Sendable {
        case encode      // Gemma/connector text encode + VAE reference/init-frame encode
        case denoise     // the distilled Euler loop (per-step)
        case upsample    // two-stage latent upsample (vae-enc → spatial ×2 → renorm)
        case decode      // VAE decode to pixels + audio decode (per-chunk when chunked)
    }

    public struct Event: Sendable, Equatable {
        public var phase: Phase
        public var step: Int?
        public var totalSteps: Int?
        public var stage: Int?
        public var totalStages: Int?

        public init(phase: Phase, step: Int? = nil, totalSteps: Int? = nil,
                    stage: Int? = nil, totalStages: Int? = nil) {
            self.phase = phase
            self.step = step
            self.totalSteps = totalSteps
            self.stage = stage
            self.totalStages = totalStages
        }
    }

    public typealias Sink = @Sendable (Event) -> Void

    @TaskLocal public static var sink: Sink?

    /// Report a phase observation to the ambient sink (no-op if none is bound).
    static func report(_ phase: Phase, step: Int? = nil, totalSteps: Int? = nil,
                       stage: Int? = nil, totalStages: Int? = nil) {
        sink?(Event(phase: phase, step: step, totalSteps: totalSteps,
                    stage: stage, totalStages: totalStages))
    }
}
