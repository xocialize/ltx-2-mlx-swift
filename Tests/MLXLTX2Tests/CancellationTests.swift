// CancellationTests.swift — MLXLTX2 through the engine's CAN gate (offline, no MLX kernels).
// LTX is the package that PROVED the cancellation program (run-lifecycle V1–V3): the cooperative
// checkpoints below are the LTX-proven placements the gate's own vocabulary was drawn from, and
// the per-step/per-chunk RunProgress plane (contract 1.18, ENGINE-NEEDS V2) makes the cadence
// RunProgress-EVIDENCED rather than merely declared. CAN-1/2 drive the real run() pre-cancelled
// (the entry checkpoint fires before notLoaded validation or weights); CAN-3 is the document of
// record for the checkpoint cadence.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXLTX2

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = MLXLTX2Package(configuration: LTX2Configuration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2VRequest(prompt: "probe", numFrames: 9, width: 704, height: 512))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // textToVideo is a long-run capability (and every footprint's peak activation is
        // multi-GB) — the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: MLXLTX2Package.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: MLXLTX2Package.manifest,
            posture: .cadence([
                // Denoise: `try Task.checkCancellation()` once per denoise step in BOTH loop
                // variants (LTX2/DenoiseLoop.swift:75 unconditioned, :119 conditioned), each
                // immediately followed by LTX2Progress.report(.denoise, step: i+1, totalSteps:)
                // which the wrapper forwards verbatim into RunProgress — evidenced.
                .init(phase: .denoise, unit: .step, reportsRunProgress: true),
                // VAE decode: `try Task.checkCancellation()` once per temporal decode chunk
                // (LTX2/VideoVAE.swift:93) + LTX2Progress.report(.decode, step: chunkIndex,
                // totalSteps: totalChunks) at :95 — evidenced.
                .init(phase: .decode, unit: .chunk, reportsRunProgress: true),
                // Encode: checkpoints at the encoder sub-stage seams — Gemma load → tokenize/
                // forward → connector (LTX2/LTX2Pipeline.swift:168/172/179). The 49-layer Gemma
                // forward itself is one fork call (worst case ≈ that forward). RunProgress here
                // is phase-level only (no per-seam step), so declared, not evidenced.
                .init(phase: .encode, unit: .step, reportsRunProgress: false),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
