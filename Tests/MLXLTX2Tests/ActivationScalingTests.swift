import Foundation
import MLXServeConformance
import MLXToolKit
import Testing
@testable import MLXLTX2

/// Contract 1.41.0 (AB-A-0074): the per-profile geometry envelope declared as a line in
/// pixel-frames, and the request → pixel-frames mapping the engine enforces it with. Offline —
/// no MLX kernels, no weights.
struct ActivationScalingTests {
    /// The registration each 2.5 tier ships with (the same shape `LTX25PackageGate` builds):
    /// family 2.5, the profile's own streaming advisory, the tier's quant.
    static func tier25(_ profile: LTX2Profile) -> LTX2Configuration {
        var c = LTX2Configuration(family: .ltx25, repo: "mlx-community/ltx-2.5-mlx", profile: profile)
        c.quant = profile == .max128 ? .bf16 : .int8
        return c
    }

    /// Measured streamed(+tiled) activation per lane, pixel-frames → bytes. Receipts: AB-R-0106
    /// (the low tiers and standard64 at 704×512×161), AB-R-0130 (720p×241 tiled), AB-T-0098
    /// (1080p at 121/241/481). Add a row for every new corner; a row the line no longer covers
    /// is a re-declaration, not a test to loosen.
    static let measured: [LTX2Profile: [(pixelFrames: Double, activationBytes: UInt64)]] = [
        .compact24: [(17_842_176, 15_020_000_000)],
        .balanced32: [(29_675_520, 16_650_000_000)],
        .standard64: [(58_032_128, 18_800_000_000), (217_169_920, 32_960_000_000)],
        .max128: [(252_764_160, 48_500_000_000), (503_439_360, 59_510_000_000),
                  (1_004_789_760, 73_620_000_000)],
    ]

    // MARK: - The contract's own gate, per profile

    @Test(arguments: LTX2Profile.allCases)
    func fitGatePassesForEveryTier(profile: LTX2Profile) {
        let report = FootprintConformance.check(manifest: MLXLTX25Package.manifest,
                                                configuration: Self.tier25(profile))
        #expect(report.passed, Comment(rawValue: "\(profile): \(report.summary)"))
        // Not the scalar-only pass: the lane pair was checked and the mapping is enforceable.
        #expect(!report.checks.contains { $0.name.hasPrefix("FIT-0") }, Comment(rawValue: "\(profile): \(report.summary)"))
        #expect(report.checks.contains { $0.name == "FIT-3 enforceable" && $0.passed },
                Comment(rawValue: "\(profile): \(report.summary)"))
    }

    // MARK: - The line against the receipts

    @Test(arguments: LTX2Profile.allCases)
    func theLineCoversEveryMeasuredCorner(profile: LTX2Profile) throws {
        let cfg = Self.tier25(profile)
        let line = try #require(cfg.activationScalingHint)
        #expect(line.axis == .pixelFrames)
        let points = Self.measured[profile]!
        for p in points {
            #expect(line.projectedBytes(at: p.pixelFrames) >= p.activationBytes,
                    Comment(rawValue: "\(profile) under-projects the measured activation at \(p.pixelFrames) px-f"))
        }
        // Rule 2: the ceiling is the largest geometry measured, no further.
        #expect(line.measuredCeiling == points.map(\.pixelFrames).max()!)
        // Rule 3 by hand (FIT-2 above checks it through the contract).
        let reserve = try #require(cfg.peakActivationBytesHint)
        #expect(line.isCovered(by: reserve))
    }

    @Test func theLowTiersStayGuardedLikeTheirReserve() {
        // A low tier that will NOT stream gets no streamed number — neither the reserve nor
        // the line (fail-closed, the AB-A-0012 condition; gate cases 42/43).
        var off = Self.tier25(.compact24); off.forceStreamGate = false
        #expect(off.peakActivationBytesHint == nil)
        #expect(off.activationScalingHint == nil)
        // 2.3 is prior work measured at one geometry per tier: scalar-only, FIT-0.
        let v23 = LTX2Configuration(repo: "xocialize/ltx-2.3-mlx", profile: .compact24)
        #expect(v23.activationScalingHint == nil)
        let report = FootprintConformance.check(manifest: MLXLTX2Package.manifest, configuration: v23)
        #expect(report.passed, Comment(rawValue: report.summary))
    }

    // MARK: - The mapping

    @Test func workloadIsTheResolvedGeometryInPixelFrames() {
        let cfg = Self.tier25(.standard64)
        // 704×512 is on the /64 grid and 161 is the cap: unchanged → 58 032 128.
        #expect(cfg.workloadUnits(for: T2VRequest(prompt: "x", numFrames: 161, width: 704, height: 512))
                == 58_032_128)
        // The default geometry (704×512×9) maps too — nil only means "unknowable".
        #expect(cfg.workloadUnits(for: T2VRequest(prompt: "x")) == Double(704 * 512 * 9))
    }

    @Test(arguments: LTX2Profile.allCases)
    func theClampKeepsEveryT2VInsideTheCeiling(profile: LTX2Profile) throws {
        // The profile clamps rather than refuses, so a request past the envelope resolves INSIDE
        // the ceiling and the engine's refusal can never fire for a t2v — it is a backstop.
        let cfg = Self.tier25(profile)
        let line = try #require(cfg.activationScalingHint)
        let huge = T2VRequest(prompt: "x", numFrames: 2001, width: 4096, height: 4096)
        let units = try #require(cfg.workloadUnits(for: huge))
        #expect(line.covers(units), Comment(rawValue: "\(profile): \(units) px-f exceeds \(line.measuredCeiling)"))
        // And the envelope corner itself is the ceiling for every tier whose cap was measured
        // exactly (compact24, balanced32, max128); standard64's corner sits past its frame cap.
        let corner = Double(profile.maxWidth * profile.maxHeight * profile.maxFrames)
        if profile != .standard64 { #expect(units == corner && corner == line.measuredCeiling) }
    }

    @Test func anA2VRequestFollowsTheTrackWhenFramesAreNotPinned() throws {
        let cfg = Self.tier25(.standard64)
        // 2.5 s of 16 kHz silence → 60 frames at 24 fps → floored to the 8k+1 grid (57), at the
        // 704×512 natural size — exactly what `runAudioToVideo` would resolve to.
        let track = Audio(format: .wav, data: TestWAV.silence(seconds: 2.5, sampleRate: 16000))
        let units = try #require(cfg.workloadUnits(for: T2VRequest(prompt: "x", initAudio: track)))
        let expected = cfg.resolvedGeometry(sourceWidth: 704, sourceHeight: 512,
                                            sourceDurationSeconds: 2.5, mode: .audioToVideo)
        #expect(units == Double(expected.width * expected.height * expected.numFrames))
        #expect(expected.numFrames == 57)
        // Pinned frames win over the track, as on the run path.
        let pinned = cfg.workloadUnits(for: T2VRequest(prompt: "x", initAudio: track, numFrames: 121))
        #expect(pinned == Double(704 * 512 * 121))
    }

    @Test func unknowableIsNilNeverANumber() {
        let cfg = Self.tier25(.standard64)
        // A videoEdit's geometry derives from the source clip: not sizable pre-admission.
        let edit = VEditRequest(video: Video(format: .mp4, data: Data()), prompt: "x")
        #expect(cfg.workloadUnits(for: edit) == nil)
        // Audio the header cannot size.
        let garbage = Audio(format: .wav, data: Data("not a wav".utf8))
        #expect(cfg.workloadUnits(for: T2VRequest(prompt: "x", initAudio: garbage)) == nil)
        // The measurement-only hatch lifts the mapping with the clamp.
        setenv("LTX_ENVELOPE_OVERRIDE", "1", 1)
        defer { unsetenv("LTX_ENVELOPE_OVERRIDE") }
        #expect(cfg.workloadUnits(for: T2VRequest(prompt: "x")) == nil)
    }
}

/// Minimal RIFF/WAVE PCM16 writer — enough container for the header read.
enum TestWAV {
    static func silence(seconds: Double, sampleRate: Int = 24000, channels: Int = 1) -> Data {
        let frames = Int((seconds * Double(sampleRate)).rounded())
        let blockAlign = channels * 2
        let dataBytes = frames * blockAlign
        var d = Data()
        func u32(_ v: Int) { d.append(contentsOf: withUnsafeBytes(of: UInt32(v).littleEndian) { Array($0) }) }
        func u16(_ v: Int) { d.append(contentsOf: withUnsafeBytes(of: UInt16(v).littleEndian) { Array($0) }) }
        d.append(contentsOf: Array("RIFF".utf8)); u32(36 + dataBytes)
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16)
        u16(1); u16(channels); u32(sampleRate); u32(sampleRate * blockAlign); u16(blockAlign); u16(16)
        d.append(contentsOf: Array("data".utf8)); u32(dataBytes)
        d.append(Data(count: dataBytes))
        return d
    }
}
