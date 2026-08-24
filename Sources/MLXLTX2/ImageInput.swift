// ImageInput.swift — decode an engine `Image` into the i2v init-frame tensor.
//
// The VAE encoder wants pixels (1, 3, 1, H, W) in [-1,1] at the target video resolution.
// We decode the image bytes (CoreGraphics), aspect-fill center-crop to (W×H) — the bitmap
// context is already top-row-first, no flip — and normalize.
//
// ⚠️ DEFERRED PARITY GAP — H.264 CRF ROUND-TRIP (AB-T-0091). The oracle re-compresses every
// conditioning image through a single-frame H.264 encode/decode BEFORE encoding it to a latent,
// "to match the compression the model was trained on"
// (`ltx_pipelines/utils/media_io/decode.py:413`). It is not a fallback or a nicety: `preprocess`
// RAISES on an unresolved CRF rather than picking one, and every pipeline calls
// `ImageConditioner.resolve_crf(images)` near the top of `__call__`.
//
// The CRF is a property of the MODEL GENERATION, and this note previously recorded the wrong one.
// `utils/constants.py`: `DEFAULT_IMAGE_CRF = 33` (through LTX-2.3) but `LTX_2_4_IMAGE_CRF = 18`
// from 2.4 — so **LTX-2.5, our primary target, wants CRF 18, not 33**. Anyone implementing this
// from the old note would have conditioned at 2.3's compression on a 2.5 checkpoint.
//
// Still deferred, and the obstacle is real rather than cosmetic: AVFoundation/VideoToolbox exposes
// quality/bitrate, not x264's CRF, so "CRF 18" has no exact Apple-side equivalent — an approximate
// mapping could land further from the training distribution than pristine pixels do. Quantify the
// effect before picking a mapping (see AB-T-0091).

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
        // NO flip: `CGContext.draw(CGImage:)` into a bitmap context already yields a top-row-first
        // buffer with correct orientation (verified empirically — a red-top/blue-bottom probe reads
        // row0=RED unflipped, row0=BLUE with the old translate/scale flip). The previous flip here
        // INVERTED the init frame → i2v videos opened with an upside-down first frame that "fell"
        // upright as the conditioning released (only frame 0 is pinned).
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

    /// Reference-still ingest (IC-LORA-PLAN P2): decode `image`, **STRETCH**-resize to
    /// (width×height) — the community reference usage's plain `resize`, deliberately NOT the
    /// i2v aspect-fill — and tile to `frames` identical frames: (1, 3, F, H, W) bf16 in [-1,1].
    /// `frames` must already be 8k+1 (`ReferenceConditioning.snapFrames`).
    static func referenceStillFrames(_ image: Image, width: Int, height: Int, frames: Int) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            throw PackageError.configurationMismatch(
                expected: "a decodable reference image (PNG/JPEG/…)", got: "undecodable image data")
        }
        let bytesPerRow = width * 4
        var buf = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard let ctx = CGContext(
            data: &buf, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            throw PackageError.configurationMismatch(
                expected: "an RGB bitmap context for the reference image", got: "context allocation failed")
        }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))   // stretch, no crop

        var floats = [Float](repeating: 0, count: height * width * 3)
        var o = 0
        for p in stride(from: 0, to: buf.count, by: 4) {
            floats[o] = Float(buf[p]) / 255.0 * 2.0 - 1.0
            floats[o + 1] = Float(buf[p + 1]) / 255.0 * 2.0 - 1.0
            floats[o + 2] = Float(buf[p + 2]) / 255.0 * 2.0 - 1.0
            o += 3
        }
        let hwc = MLXArray(floats, [height, width, 3])
        let one = hwc.transposed(2, 0, 1)                       // (3,H,W)
            .expandedDimensions(axis: 0).expandedDimensions(axis: 2)   // (1,3,1,H,W)
        return MLX.broadcast(one, to: [1, 3, frames, height, width]).asType(.bfloat16)
    }
}
