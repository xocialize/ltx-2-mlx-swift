// VideoInputTests.swift — offline round-trip for the videoClip ingest (IC-LORA-PLAN P3b):
// synthesize an MP4 with the package's own encoder, decode via referenceClipFrames, and assert
// time-sampling, ORIENTATION (the NO-flip doctrine, probe-style), and value mapping. H.264 is
// lossy → coarse color targets with wide tolerances.

import Foundation
import LTX2
import MLX
import XCTest
@testable import MLXLTX2

final class VideoInputTests: XCTestCase {

    func testLooksLikeVideo() {
        XCTAssertTrue(VideoInput.looksLikeVideo("/tmp/ref.mp4"))
        XCTAssertTrue(VideoInput.looksLikeVideo("/tmp/REF.MOV"))
        XCTAssertFalse(VideoInput.looksLikeVideo("/tmp/sheet.png"))
        XCTAssertFalse(VideoInput.looksLikeVideo("/tmp/sheet.jpeg"))
    }

    func testReferenceClipSamplingAndOrientation() async throws {
        // 17-frame 24 fps source, 64×64: frame 0 = top GREEN / bottom BLUE (orientation probe);
        // frames 1…16 solid RED (time-sampling probe).
        let F = 17, H = 64, W = 64
        var vals = [Float](repeating: -1, count: F * H * W * 3)
        func set(_ f: Int, _ y: Int, _ x: Int, _ c: Int, _ v: Float) {
            vals[((f * H + y) * W + x) * 3 + c] = v
        }
        for y in 0 ..< H { for x in 0 ..< W {
            set(0, y, x, y < H / 2 ? 1 : 2, 1.0)          // frame 0: green top, blue bottom
            for f in 1 ..< F { set(f, y, x, 0, 1.0) }      // frames 1+: red
        } }
        let frames = MLXArray(vals, [1, F, H, W, 3])
        let mp4 = try await encodeMP4(frames: frames, fps: 24)
        let url = FileManager.default.temporaryDirectory.appending(path: "vi-test-\(UUID().uuidString).mp4")
        try mp4.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let t = try await VideoInput.referenceClipFrames(url: url, width: W, height: H, frames: 9, fps: 24)
        XCTAssertEqual(t.shape, [1, 3, 9, H, W])
        let f = t.asType(.float32)
        // Frame 0 orientation: tensor row 0 = TOP = GREEN; bottom row = BLUE.
        XCTAssertGreaterThan(f[0, 1, 0, 4, 32].item(Float.self), 0.3, "frame-0 top must be GREEN")
        XCTAssertLessThan(f[0, 2, 0, 4, 32].item(Float.self), 0.0)
        XCTAssertGreaterThan(f[0, 2, 0, 59, 32].item(Float.self), 0.3, "frame-0 bottom must be BLUE")
        // Frame 8 (t = 8/24 s → source frame 8): RED.
        XCTAssertGreaterThan(f[0, 0, 8, 32, 32].item(Float.self), 0.3, "frame 8 must be RED (time sampling)")
        XCTAssertLessThan(f[0, 1, 8, 32, 32].item(Float.self), 0.0)
        // Short-clip clamp: requesting 9 frames at 24 fps from a 17-frame clip never throws
        // (already proven above); a longer request clamps to the final frame.
        let long = try await VideoInput.referenceClipFrames(url: url, width: W, height: H, frames: 25, fps: 24)
        XCTAssertGreaterThan(long.asType(.float32)[0, 0, 24, 32, 32].item(Float.self), 0.3,
                             "past-the-end frames clamp to the last (RED) frame")
    }
}
