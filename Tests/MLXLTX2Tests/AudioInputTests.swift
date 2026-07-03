// AudioInputTests.swift — offline round-trip for the dub-audio ingest (IC-LORA-PLAN P3b):
// synthesize an MP4 with an AAC track via the package's own encoder, decode via AudioInput,
// and assert the 16 kHz stereo contract + trimming. AAC is lossy → energy/shape checks only.

import Foundation
import LTX2
import MLX
import XCTest
@testable import MLXLTX2

final class AudioInputTests: XCTestCase {

    func testReferenceWaveformResampleTrimAndShape() async throws {
        // 2 s of 440 Hz sine at 48 kHz stereo + 9 tiny video frames (the muxer needs both).
        let sr = 48000.0, seconds = 2.0
        let n = Int(sr * seconds)
        var wave = [Float](repeating: 0, count: 2 * n)
        for i in 0 ..< n {
            let v = sinf(2 * .pi * 440 * Float(i) / Float(sr)) * 0.5
            wave[i] = v; wave[n + i] = v
        }
        let audio = MLXArray(wave, [1, 2, n])
        let frames = MLXArray.zeros([1, 9, 32, 32, 3])
        let mp4 = try await encodeMP4(frames: frames, fps: 24, audio: audio, audioSampleRate: sr)
        let url = FileManager.default.temporaryDirectory.appending(path: "ai-test-\(UUID().uuidString).mp4")
        try mp4.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let out = try await AudioInput.referenceWaveform(url: url)
        XCTAssertEqual(out.dim(0), 1)
        XCTAssertEqual(out.dim(1), 2)                       // stereo contract
        // ~2 s at 16 kHz (AAC priming/padding gives slack).
        XCTAssertGreaterThan(out.dim(2), Int(16000 * 1.8))
        XCTAssertLessThan(out.dim(2), Int(16000 * 2.3))
        // Signal survived (RMS of a 0.5-amp sine ≈ 0.35; AAC keeps it near).
        let f = out.asType(.float32)
        let rms = MLX.sqrt(MLX.mean(f * f)).item(Float.self)
        XCTAssertGreaterThan(rms, 0.15, "sine energy must survive the round trip")
        XCTAssertLessThan(MLX.max(MLX.abs(f)).item(Float.self), 1.5)

        // Trimming: maxSeconds caps the span (LipDub aligns the dub to the clip length).
        let trimmed = try await AudioInput.referenceWaveform(url: url, maxSeconds: 1.0)
        XCTAssertLessThanOrEqual(trimmed.dim(2), 16000)
    }
}
