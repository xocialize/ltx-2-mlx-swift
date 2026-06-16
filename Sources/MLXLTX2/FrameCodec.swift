// FrameCodec.swift — channels-last frame tensor → H.264 MP4 (pure AVFoundation).
// Adapted from ti2v-5b-mlx-swift's FrameCodec. The wrapper transposes the LTX
// decoder output (channels-first [1,3,F,H,W]) to channels-last [1,F,H,W,3] in
// [-1,1] before calling.

import AVFoundation
import CoreGraphics
import Foundation
import MLX
import MLXToolKit

enum FrameCodecError: Error {
    case pixelBufferAllocation
    case writerSetup(String)
    case badFrames(String)
    case appendFailed(String)
    case writeIncomplete(String)
}

/// One channels-last frame [H, W, 3] in [-1, 1] → interleaved RGB bytes.
private func rgbBytes(_ frame: MLXArray) -> (bytes: [UInt8], width: Int, height: Int) {
    let h = frame.dim(0), w = frame.dim(1)
    let scaled = (frame.asType(.float32) + 1) * Float(127.5)
    let rgb = clip(scaled, min: 0, max: 255).asType(.uint8)
    eval(rgb)
    return (rgb.asArray(UInt8.self), w, h)
}

private func pixelBuffer(rgb: [UInt8], width: Int, height: Int, pool: CVPixelBufferPool) throws -> CVPixelBuffer {
    var out: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out)
    guard let buffer = out else { throw FrameCodecError.pixelBufferAllocation }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0 ..< height {
        for x in 0 ..< width {
            let src = (y * width + x) * 3, dst = y * stride + x * 4
            base[dst + 0] = rgb[src + 2]  // B
            base[dst + 1] = rgb[src + 1]  // G
            base[dst + 2] = rgb[src + 0]  // R
            base[dst + 3] = 255
        }
    }
    return buffer
}

/// Encode channels-last frames [1, T, H, W, 3] in [-1, 1] as an H.264 MP4 at `fps`.
@InferenceActor
func encodeMP4(frames: MLXArray, fps: Double) async throws -> Data {
    guard frames.ndim == 5, frames.dim(1) > 0 else {
        throw FrameCodecError.badFrames("expected [1,T,H,W,3], got \(frames.shape)")
    }
    let t = frames.dim(1), h = frames.dim(2), w = frames.dim(3)
    let url = FileManager.default.temporaryDirectory.appending(path: "ltx2-\(UUID().uuidString).mp4")
    defer { try? FileManager.default.removeItem(at: url) }

    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h,
    ])
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
    ])
    guard writer.canAdd(input) else { throw FrameCodecError.writerSetup("cannot add input") }
    writer.add(input)
    guard writer.startWriting() else {
        throw FrameCodecError.writerSetup(writer.error?.localizedDescription ?? "startWriting")
    }
    writer.startSession(atSourceTime: .zero)

    let frameDuration = CMTime(value: CMTimeValue((600.0 / fps).rounded()), timescale: 600)
    for i in 0 ..< t {
        let (bytes, fw, fh) = rgbBytes(frames[0, i, 0..., 0..., 0...])
        guard let pool = adaptor.pixelBufferPool else { throw FrameCodecError.writerSetup("no pool") }
        let buffer = try pixelBuffer(rgb: bytes, width: fw, height: fh, pool: pool)
        while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(5)) }
        guard adaptor.append(buffer, withPresentationTime: CMTimeMultiply(frameDuration, multiplier: Int32(i))) else {
            throw FrameCodecError.appendFailed("frame \(i)/\(t) err=\(String(describing: writer.error))")
        }
    }
    input.markAsFinished()
    await writer.finishWriting()
    guard writer.status == .completed, FileManager.default.fileExists(atPath: url.path) else {
        throw FrameCodecError.writeIncomplete("status=\(writer.status.rawValue) err=\(String(describing: writer.error))")
    }
    return try Data(contentsOf: url)
}
