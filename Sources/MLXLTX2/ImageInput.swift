// ImageInput.swift — decode an engine `Image` into the i2v init-frame tensor.
//
// The VAE encoder wants pixels (1, 3, 1, H, W) in [-1,1] at the target video resolution.
// We decode the image bytes (CoreGraphics), aspect-fill center-crop to (W×H), flip so the
// pixel buffer is top-row-first, and normalize. NOTE: the oracle also runs a lossy H.264
// CRF round-trip (default crf=33) before encoding as a quality-matching step — DEFERRED here
// (a perceptual detail; i2v is not bit-matched to the oracle anyway, see ISSUES bf16 doctrine).

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXToolKit

enum ImageInput {
    /// Decode + preprocess `image` to (1, 3, 1, height, width), bf16, channels-first, [-1,1].
    static func initFrameTensor(_ image: Image, width: Int, height: Int) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            throw PackageError.configurationMismatch(
                expected: "a decodable initImage (PNG/JPEG/…)", got: "undecodable image data")
        }
        let bytesPerRow = width * 4
        var buf = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard let ctx = CGContext(
            data: &buf, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            throw PackageError.configurationMismatch(
                expected: "an RGB bitmap context for initImage", got: "context allocation failed")
        }
        // Aspect-fill center-crop: scale so the image covers (width×height), center it.
        let iw = CGFloat(cg.width), ih = CGFloat(cg.height)
        let scale = max(CGFloat(width) / iw, CGFloat(height) / ih)
        let dw = iw * scale, dh = ih * scale
        let ox = (CGFloat(width) - dw) / 2, oy = (CGFloat(height) - dh) / 2
        // Flip vertically so buffer row 0 = top of the image (matches frame decode orientation).
        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1, y: -1)
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: ox, y: oy, width: dw, height: dh))

        // RGBA8 buffer → [-1,1] RGB floats (drop alpha) → (H,W,3).
        var floats = [Float](repeating: 0, count: height * width * 3)
        var o = 0
        for p in stride(from: 0, to: buf.count, by: 4) {
            floats[o] = Float(buf[p]) / 255.0 * 2.0 - 1.0
            floats[o + 1] = Float(buf[p + 1]) / 255.0 * 2.0 - 1.0
            floats[o + 2] = Float(buf[p + 2]) / 255.0 * 2.0 - 1.0
            o += 3
        }
        let hwc = MLXArray(floats, [height, width, 3])
        // (H,W,3) → (3,H,W) → (1,3,1,H,W)
        return hwc.transposed(2, 0, 1).expandedDimensions(axis: 0).expandedDimensions(axis: 2).asType(.bfloat16)
    }
}
