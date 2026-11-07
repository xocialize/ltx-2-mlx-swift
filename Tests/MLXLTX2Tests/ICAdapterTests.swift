// ICAdapterTests.swift — offline checks for the IC reference ingest (IC-LORA-PLAN P3/P4): frame
// snapping and the still-reference orientation. The adapter contract itself (registry, catalog,
// slots, groups, ADP gate) is in AdapterTests.

import CoreGraphics
import Foundation
import ImageIO
import MLXToolKit
import UniformTypeIdentifiers
import LTX2
import XCTest
@testable import MLXLTX2

final class ICAdapterTests: XCTestCase {

    func testMuxDubAudioKeyIsStable() {
        // The one IC metaData key left after typed adapters (an output choice, not selection).
        XCTAssertEqual(ICMetaKeys.muxDubAudio, "ic.muxDubAudio")
    }

    func testSnapFramesMatchesOracle() {
        XCTAssertEqual(ReferenceConditioning.snapFrames(121), 121)
        XCTAssertEqual(ReferenceConditioning.snapFrames(120), 113)
        XCTAssertEqual(ReferenceConditioning.snapFrames(1), 9)
    }

    /// The red-top/blue-bottom ORIENTATION PROBE from the i2v inversion fix (ImageInput NO-flip
    /// doctrine), applied to the IC reference ingest. An inverted reference wouldn't self-right
    /// the way i2v's pinned frame 0 did — it would silently degrade identity transfer — so the
    /// probe is a standing test, not a one-off check.
    func testReferenceStillFramesOrientationAndTiling() throws {
        // Build a 32×32 PNG: top half RED, bottom half BLUE.
        let ctx = CGContext(data: nil, width: 32, height: 32, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        // CG origin is bottom-left: bottom half BLUE at y 0..<16, top half RED at y 16..<32.
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: 32, height: 16))
        ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 16, width: 32, height: 16))
        let cg = ctx.makeImage()!
        let out = NSMutableData()
        let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, cg, nil)
        CGImageDestinationFinalize(dest)

        let tensor = try ImageInput.referenceStillFrames(
            Image(format: .png, data: out as Data), width: 32, height: 32, frames: 9)
        XCTAssertEqual(tensor.shape, [1, 3, 9, 32, 32])
        let t = tensor.asType(.float32)
        // Tensor row 0 = TOP row (no flip): red channel ≈ +1 at top, blue ≈ +1 at bottom.
        XCTAssertGreaterThan(t[0, 0, 0, 0, 16].item(Float.self), 0.9, "top row must be RED — inversion regression")
        XCTAssertLessThan(t[0, 2, 0, 0, 16].item(Float.self), -0.9)
        XCTAssertGreaterThan(t[0, 2, 0, 31, 16].item(Float.self), 0.9, "bottom row must be BLUE")
        XCTAssertLessThan(t[0, 0, 0, 31, 16].item(Float.self), -0.9)
        // Looped-still tiling: a later frame is identical to frame 0.
        XCTAssertGreaterThan(t[0, 0, 7, 0, 16].item(Float.self), 0.9)
    }
}
