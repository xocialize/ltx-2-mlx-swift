// FrameCodec.swift — channels-last frame tensor → H.264 MP4 (pure AVFoundation).
// Adapted from ti2v-5b-mlx-swift's FrameCodec. The wrapper transposes the LTX
// decoder output (channels-first [1,3,F,H,W]) to channels-last [1,F,H,W,3] in
// [-1,1] before calling.

import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import LTX2
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

/// Encode channels-last frames [1, T, H, W, 3] in [-1, 1] as an H.264 MP4 at `fps`,
/// optionally muxing a stereo audio track [1, 2, T_audio] in [-1, 1] at `audioSampleRate`.
/// Encode frames [1,T,H,W,3] in [-1,1] → MP4 Data.
///
/// **Defaults to a SOFTWARE-only H.264 encoder.** The hardware VideoToolbox media engine STALLS when
/// it contends with MLX's active post-generation GPU context: measured cold, the AVAssetWriter input
/// stops draining (`isReadyForMoreMediaData` stuck) and hangs at frame 32/41 for a 48-frame clip,
/// while the software encoder does the same 41 frames in ~3.7 s. Isolated with `RunLTX2
/// --encode-stress`: hardware is fine for the same frames with idle memory OR a 38 GB idle
/// allocation — it only stalls *after real MLX compute*, which is always the case for LTX output.
/// `software: false` or `LTX_ENCODE=hardware` opts back into the faster (~1.2 s) hardware path for
/// callers that don't encode right after heavy Metal work.
@InferenceActor
public func encodeMP4(frames: MLXArray, fps: Double, audio: MLXArray? = nil, audioSampleRate: Double = 48000,
                      software: Bool = true) async throws -> Data {
    guard frames.ndim == 5, frames.dim(1) > 0 else {
        throw FrameCodecError.badFrames("expected [1,T,H,W,3], got \(frames.shape)")
    }
    let t = frames.dim(1), h = frames.dim(2), w = frames.dim(3)
    let url = FileManager.default.temporaryDirectory.appending(path: "ltx2-\(UUID().uuidString).mp4")
    defer { try? FileManager.default.removeItem(at: url) }

    // env overrides the param: LTX_ENCODE = "hardware" | "software" | (unset → the `software` arg).
    let env = ProcessInfo.processInfo.environment["LTX_ENCODE"]
    let forceSoftware = env == "software" || (env != "hardware" && software)
    var videoSettings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h,
    ]
    if forceSoftware {
        // VideoToolbox encoder-specification keys (string form → no VideoToolbox import needed):
        // require a software-only encoder so the hardware media engine is not used.
        videoSettings[AVVideoEncoderSpecificationKey] = [
            "EnableHardwareAcceleratedVideoEncoder": false,
            "RequireSoftwareOnlyVideoEncoder": true,
        ] as [String: Any]
        LTX2Profiler.shared.note("encode-mp4 using SOFTWARE H.264 encoder (hardware VideoToolbox bypassed)")
    } else {
        LTX2Profiler.shared.note("encode-mp4 using HARDWARE H.264 encoder (LTX_ENCODE=hardware) — may stall after heavy MLX compute")
    }
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
    ])
    guard writer.canAdd(input) else { throw FrameCodecError.writerSetup("cannot add input") }
    writer.add(input)

    var audioInput: AVAssetWriterInput?
    if audio != nil {
        let ai = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: audioSampleRate,
            AVNumberOfChannelsKey: 2, AVEncoderBitRateKey: 192_000,
        ])
        ai.expectsMediaDataInRealTime = false
        if writer.canAdd(ai) { writer.add(ai); audioInput = ai }
    }

    guard writer.startWriting() else {
        throw FrameCodecError.writerSetup(writer.error?.localizedDescription ?? "startWriting")
    }
    writer.startSession(atSourceTime: .zero)

    let frameDuration = CMTime(value: CMTimeValue((600.0 / fps).rounded()), timescale: 600)
    for i in 0 ..< t {
        let (bytes, fw, fh) = rgbBytes(frames[0, i, 0..., 0..., 0...])
        guard let pool = adaptor.pixelBufferPool else { throw FrameCodecError.writerSetup("no pool") }
        let buffer = try pixelBuffer(rgb: bytes, width: fw, height: fh, pool: pool)
        // The H.264/VideoToolbox encoder can stop draining at higher frame counts (GPU/memory
        // contention with the resident model) — `isReadyForMoreMediaData` then stays false forever
        // and this loop spins at ~0% CPU (the "looks like a loop / GPU <10%" hang). Bound the wait so
        // a stall becomes a loud, localized error instead of an invisible forever-hang.
        var waited = 0.0
        while !input.isReadyForMoreMediaData {
            try await Task.sleep(for: .milliseconds(5)); waited += 0.005
            if waited.rounded() != (waited - 0.005).rounded(), Int(waited) % 5 == 0 {
                LTX2Profiler.shared.note("encode-mp4 ⚠ waiting on H.264 encoder at frame \(i)/\(t) — \(Int(waited))s (isReadyForMoreMediaData=false)")
            }
            if waited > 90 {
                throw FrameCodecError.appendFailed("H.264 encoder stalled at frame \(i)/\(t): isReadyForMoreMediaData=false for 90s — VideoToolbox is not draining (GPU/memory contention with the resident model). This is the post-generation hang, not the denoise.")
            }
        }
        guard adaptor.append(buffer, withPresentationTime: CMTimeMultiply(frameDuration, multiplier: Int32(i))) else {
            throw FrameCodecError.appendFailed("frame \(i)/\(t) err=\(String(describing: writer.error))")
        }
        if i % 8 == 0 { LTX2Profiler.shared.note("encode-mp4 frame \(i)/\(t)") }
    }
    input.markAsFinished()

    if let audioInput, let audio {
        let buffer = try audioSampleBuffer(audio, sampleRate: audioSampleRate)
        while !audioInput.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(5)) }
        guard audioInput.append(buffer) else {
            throw FrameCodecError.appendFailed("audio err=\(String(describing: writer.error))")
        }
        audioInput.markAsFinished()
    }

    await writer.finishWriting()
    guard writer.status == .completed, FileManager.default.fileExists(atPath: url.path) else {
        throw FrameCodecError.writeIncomplete("status=\(writer.status.rawValue) err=\(String(describing: writer.error))")
    }
    return try Data(contentsOf: url)
}

/// Build an LPCM CMSampleBuffer from a stereo waveform [1, 2, T] in [-1, 1] (interleaved 16-bit).
private func audioSampleBuffer(_ audio: MLXArray, sampleRate: Double) throws -> CMSampleBuffer {
    let frames = audio.dim(2)
    // (1,2,T) → (T,2) interleaved → int16 [2T]
    let stereo = MLX.stacked([audio[0, 0, 0...], audio[0, 1, 0...]], axis: -1)  // (T,2)
    let i16 = (clip(stereo, min: -1, max: 1) * Float(32767)).asType(.int16).reshaped(-1)
    eval(i16)
    let samples = i16.asArray(Int16.self)
    let dataSize = samples.count * MemoryLayout<Int16>.size

    var asbd = AudioStreamBasicDescription(
        mSampleRate: sampleRate, mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
        mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
        mChannelsPerFrame: 2, mBitsPerChannel: 16, mReserved: 0)
    var formatDesc: CMAudioFormatDescription?
    CMAudioFormatDescriptionCreate(allocator: kCFAllocatorDefault, asbd: &asbd,
                                   layoutSize: 0, layout: nil, magicCookieSize: 0, magicCookie: nil,
                                   extensions: nil, formatDescriptionOut: &formatDesc)

    var blockBuffer: CMBlockBuffer?
    CMBlockBufferCreateWithMemoryBlock(allocator: kCFAllocatorDefault, memoryBlock: nil,
                                       blockLength: dataSize, blockAllocator: kCFAllocatorDefault,
                                       customBlockSource: nil, offsetToData: 0, dataLength: dataSize,
                                       flags: 0, blockBufferOut: &blockBuffer)
    try samples.withUnsafeBytes { raw in
        let status = CMBlockBufferReplaceDataBytes(with: raw.baseAddress!, blockBuffer: blockBuffer!,
                                                   offsetIntoDestination: 0, dataLength: dataSize)
        guard status == kCMBlockBufferNoErr else { throw FrameCodecError.appendFailed("CMBlockBuffer \(status)") }
    }

    var sampleBuffer: CMSampleBuffer?
    var timing = CMSampleTimingInfo(
        duration: CMTime(value: 1, timescale: CMTimeScale(sampleRate)),
        presentationTimeStamp: .zero, decodeTimeStamp: .invalid)
    var sampleSize = 4
    CMSampleBufferCreate(allocator: kCFAllocatorDefault, dataBuffer: blockBuffer, dataReady: true,
                         makeDataReadyCallback: nil, refcon: nil, formatDescription: formatDesc,
                         sampleCount: CMItemCount(frames), sampleTimingEntryCount: 1, sampleTimingArray: &timing,
                         sampleSizeEntryCount: 1, sampleSizeArray: &sampleSize, sampleBufferOut: &sampleBuffer)
    guard let buf = sampleBuffer else { throw FrameCodecError.appendFailed("CMSampleBufferCreate") }
    return buf
}
