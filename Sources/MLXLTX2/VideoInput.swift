// VideoInput.swift — decode a reference VIDEO clip into the IC ingest tensor (IC-LORA-PLAN P3b).
//
// Mirrors the community reference usage's `_prep_reference` (the ingredients Space, shared by the
// Cameraman-class adapters): sample frames BY TIME at the target fps (source fps resampled),
// **aspect-FILL center-crop** to (W×H) — video refs crop, unlike the sheet's stretch — and clamp
// the last frame for clips shorter than the requested span. Frames must already be 8k+1
// (`ReferenceConditioning.snapFrames`). Orientation follows the ImageInput NO-flip doctrine
// (CG bitmap contexts are top-row-first).

import AVFoundation
import CoreGraphics
import Foundation
import MLX
import MLXToolKit

enum VideoInput {

    /// Decode `url` to (1, 3, frames, height, width) bf16 in [-1,1]: frame i sampled at time
    /// i/fps (clamped to the clip's duration), aspect-fill center-cropped.
    static func referenceClipFrames(url: URL, width: Int, height: Int,
                                    frames: Int, fps: Double) async throws -> MLXArray {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration > 0 else {
            throw PackageError.configurationMismatch(
                expected: "a decodable reference video", got: "zero-duration asset at \(url.lastPathComponent)")
        }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = CMTime(value: 1, timescale: 60)   // ≤ one 60fps frame late

        var floats = [Float](repeating: 0, count: frames * height * width * 3)
        let bytesPerRow = width * 4
        var buf = [UInt8](repeating: 0, count: height * bytesPerRow)
        for i in 0 ..< frames {
            let t = min(Double(i) / fps, duration - 0.001)
            let time = CMTime(seconds: max(0, t), preferredTimescale: 600)
            let cg = try await generator.image(at: time).image
            guard let ctx = CGContext(
                data: &buf, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else {
                throw PackageError.configurationMismatch(
                    expected: "an RGB bitmap context for the reference clip", got: "context allocation failed")
            }
            // Aspect-FILL center-crop (ImageOps.fit semantics) — NO flip.
            let iw = CGFloat(cg.width), ih = CGFloat(cg.height)
            let scale = max(CGFloat(width) / iw, CGFloat(height) / ih)
            let dw = iw * scale, dh = ih * scale
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: (CGFloat(width) - dw) / 2, y: (CGFloat(height) - dh) / 2,
                                    width: dw, height: dh))
            var o = i * height * width * 3
            for p in stride(from: 0, to: buf.count, by: 4) {
                floats[o] = Float(buf[p]) / 255.0 * 2.0 - 1.0
                floats[o + 1] = Float(buf[p + 1]) / 255.0 * 2.0 - 1.0
                floats[o + 2] = Float(buf[p + 2]) / 255.0 * 2.0 - 1.0
                o += 3
            }
        }
        // (F,H,W,3) → (3,F,H,W) → (1,3,F,H,W)
        let fhwc = MLXArray(floats, [frames, height, width, 3])
        return fhwc.transposed(3, 0, 1, 2).expandedDimensions(axis: 0).asType(.bfloat16)
    }

    /// Is `path` a video by extension (the `ic.referencePath` router's cheap check — image decode
    /// is attempted for everything else).
    static func looksLikeVideo(_ path: String) -> Bool {
        ["mp4", "mov", "m4v", "avi", "webm", "mkv"].contains(
            URL(fileURLWithPath: path).pathExtension.lowercased())
    }
}
