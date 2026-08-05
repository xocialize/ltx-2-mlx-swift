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
import MLXProfiling
import MLXRandom
import MLXToolKit

enum FrameCodecError: Error {
    case pixelBufferAllocation
    case writerSetup(String)
    case badFrames(String)
    case appendFailed(String)
    case writeIncomplete(String)
}

/// One channels-last frame [H, W, 3] in [-1, 1] → interleaved **BGRA** bytes.
///
/// The channel reverse (RGB→BGR) and the opaque alpha plane are done **on-device**, so the host
/// receives bytes already in the pixel buffer's layout and only has to bulk-copy them. The former
/// version returned RGB and repacked to BGRA with a per-pixel scalar loop on the CPU — ~43.6 M
/// iterations at 704×512×121f, for work the GPU does in one kernel. See `GAP-ANALYSIS.md` §3c
/// (this loop was a Swift-only regression: the Python oracle does a single bulk `memoryview`
/// copy) and `SPEED-PLAN.md` S2 step 2.
private func bgraBytes(_ frame: MLXArray) -> (bytes: [UInt8], width: Int, height: Int) {
    let h = frame.dim(0), w = frame.dim(1)
    let scaled = clip((frame.asType(.float32) + 1) * Float(127.5), min: 0, max: 255)
    let bgr = MLX.take(scaled, MLXArray([2, 1, 0] as [Int32]), axis: -1)   // RGB → BGR
    let alpha = MLXArray.ones([h, w, 1]) * Float(255)
    let bgra = MLX.concatenated([bgr, alpha], axis: -1).asType(.uint8)     // [H, W, 4]
    eval(bgra)
    return (bgra.asArray(UInt8.self), w, h)
}

/// Copy already-BGRA bytes into a pooled pixel buffer — one `memcpy` per row (or one for the
/// whole plane when the pool hands back a tightly-packed buffer).
private func pixelBuffer(bgra: [UInt8], width: Int, height: Int, pool: CVPixelBufferPool) throws -> CVPixelBuffer {
    var out: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out)
    guard let buffer = out else { throw FrameCodecError.pixelBufferAllocation }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    let rowBytes = width * 4
    bgra.withUnsafeBufferPointer { src in
        guard let s = src.baseAddress else { return }
        if stride == rowBytes {
            memcpy(base, s, height * rowBytes)
        } else {
            for y in 0 ..< height {
                memcpy(base + y * stride, s + y * rowBytes, rowBytes)
            }
        }
    }
    return buffer
}

/// Validation entry point for the on-device BGRA repack (`RunLTX2 --frame-codec-gate`).
///
/// Recomputes the OLD host-side per-pixel repack for the same input and asserts the on-device
/// path is **byte-identical**. That is the right gate for this change: the optimization must be a
/// pure refactor — the encoder has to see exactly the bytes it saw before — and byte-identity
/// proves it without needing to decode an H.264 round trip (which is lossy and would prove less).
/// Random input is deliberate: values well outside [-1, 1] exercise the clip path at both ends.
public func frameCodecRepackSelfTest(height: Int = 512, width: Int = 704, seed: UInt64 = 42)
    -> (ok: Bool, detail: String)
{
    MLXRandom.seed(seed)
    let frame = MLXRandom.normal([height, width, 3]).asType(.float32)
    eval(frame)

    let (newBytes, w, h) = bgraBytes(frame)

    // The pre-2026-08-04 path, reconstructed verbatim.
    let scaled = (frame.asType(.float32) + 1) * Float(127.5)
    let rgb = clip(scaled, min: 0, max: 255).asType(.uint8)
    eval(rgb)
    let src = rgb.asArray(UInt8.self)
    var old = [UInt8](repeating: 0, count: h * w * 4)
    for y in 0 ..< h {
        for x in 0 ..< w {
            let s = (y * w + x) * 3, d = (y * w + x) * 4
            old[d + 0] = src[s + 2]  // B
            old[d + 1] = src[s + 1]  // G
            old[d + 2] = src[s + 0]  // R
            old[d + 3] = 255
        }
    }

    guard newBytes.count == old.count else {
        return (false, "byte count \(newBytes.count) != expected \(old.count)")
    }
    for i in 0 ..< old.count where newBytes[i] != old[i] {
        let px = i / 4, chan = i % 4
        return (false, "first mismatch at byte \(i) (pixel \(px), channel \(chan)): "
            + "on-device \(newBytes[i]) vs host \(old[i])")
    }
    return (true, "\(old.count) bytes byte-identical (\(h)×\(w) BGRA)")
}

/// Incremental MP4 writer — the GAP-ANALYSIS #8 seam on the encoder side.
///
/// Lifecycle: `init` (declares whether an audio track exists — AVAssetWriter inputs must all be
/// added before `startWriting`) → `attachAudio` (appends + finishes the WHOLE audio track before
/// any video frame — the two-track interleave-deadlock fix, PROFILING.md §2: with
/// `expectsMediaDataInRealTime = false` the writer parks the video input once it runs ~43 frames
/// ahead of an empty audio track) → any number of `appendSync(framesChunk:)` calls with
/// channels-last chunks `[1, t, H, W, 3]` in [-1, 1] → `finish()` returns the MP4 bytes.
///
/// `appendSync` is deliberately SYNCHRONOUS: the pipeline's decode loop is synchronous GPU work on
/// one actor (per-chunk `eval` + `clearCache`), and threading an async sink through it fights
/// Swift 6 region isolation for no benefit. Encoder back-pressure at chunk cadence is effectively
/// never hit (hardware H.264 drains 121f in ~0.2 s); when it is, a bounded 2 ms usleep poll with
/// the same 90 s loud-error guard applies.
///
/// `encodeMP4(frames:)` below is exactly init → attachAudio → one appendSync → finish, so the
/// materialized and streamed paths share one code path and one validation surface
/// (`--encode-stress`, `--frame-codec-gate`).
///
/// Concurrency: a plain class, NOT actor-bound — the pipeline drives it from ONE isolation context
/// through synchronous sink closures (`LTX2Pipeline.StreamingSinks`), and binding it to an actor
/// would make those closures uncallable. `@unchecked Sendable` per the fleet idiom for
/// single-context GPU/AV-state classes; do not share an instance across concurrent contexts.
public final class MP4StreamWriter: @unchecked Sendable {
    private let url: URL
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let audioInput: AVAssetWriterInput?
    private let audioSampleRate: Double
    private let frameDuration: CMTime
    private var frameIndex = 0
    private var audioAttached = false
    private var finished = false

    /// **Defaults to the HARDWARE VideoToolbox H.264 encoder** (SPEED-PLAN S2 step 1); the former
    /// software default was a misdiagnosis artifact of the interleave deadlock. `software: true`
    /// or `LTX_ENCODE=software` opts back into the software-only encoder.
    public init(width: Int, height: Int, fps: Double,
                expectAudio: Bool, audioSampleRate: Double = 48000,
                software: Bool = false) throws {
        url = FileManager.default.temporaryDirectory.appending(path: "ltx2-\(UUID().uuidString).mp4")
        self.audioSampleRate = audioSampleRate

        // env overrides the param: LTX_ENCODE = "hardware" | "software" | (unset → the `software` arg).
        let env = ProcessInfo.processInfo.environment["LTX_ENCODE"]
        let forceSoftware = env == "software" || (env != "hardware" && software)
        var videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
        ]
        if forceSoftware {
            videoSettings[AVVideoEncoderSpecificationKey] = [
                "EnableHardwareAcceleratedVideoEncoder": false,
                "RequireSoftwareOnlyVideoEncoder": true,
            ] as [String: Any]
            MLXProfiler.shared.note("encode-mp4 using SOFTWARE H.264 encoder (LTX_ENCODE=software or caller opt-out)")
        } else {
            MLXProfiler.shared.note("encode-mp4 using HARDWARE H.264 encoder (default; audio-first interleave fix in place)")
        }
        writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        input.expectsMediaDataInRealTime = false
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        guard writer.canAdd(input) else { throw FrameCodecError.writerSetup("cannot add input") }
        writer.add(input)

        if expectAudio {
            let ai = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: audioSampleRate,
                AVNumberOfChannelsKey: 2, AVEncoderBitRateKey: 192_000,
            ])
            ai.expectsMediaDataInRealTime = false
            if writer.canAdd(ai) { writer.add(ai); audioInput = ai } else { audioInput = nil }
        } else {
            audioInput = nil
        }

        guard writer.startWriting() else {
            throw FrameCodecError.writerSetup(writer.error?.localizedDescription ?? "startWriting")
        }
        writer.startSession(atSourceTime: .zero)
        frameDuration = CMTime(value: CMTimeValue((600.0 / fps).rounded()), timescale: 600)
    }

    /// AUDIO FIRST — the load-bearing ordering (see the class doc). Must be called before the
    /// first `appendSync` whenever the writer was created with `expectAudio: true`; passing `nil`
    /// (e.g. the pipeline produced no waveform) closes the track empty so the interleaver never
    /// waits on it.
    public func attachAudio(_ waveform: MLXArray?) throws {
        guard !audioAttached else { return }
        audioAttached = true
        guard let audioInput else { return }
        if let waveform {
            let buffer = try audioSampleBuffer(waveform, sampleRate: audioSampleRate)
            var waited = 0.0
            while !audioInput.isReadyForMoreMediaData {
                usleep(2000); waited += 0.002
                if waited > 90 { throw FrameCodecError.appendFailed("audio input not ready for 90s") }
            }
            guard audioInput.append(buffer) else {
                throw FrameCodecError.appendFailed("audio err=\(String(describing: writer.error))")
            }
        }
        audioInput.markAsFinished()
    }

    /// Append a channels-last chunk `[1, t, H, W, 3]` in [-1, 1]. Frame timestamps continue from
    /// the running total, so chunk boundaries are invisible in the output timeline.
    public func appendSync(framesChunk frames: MLXArray, totalFrames: Int? = nil) throws {
        guard !finished else { throw FrameCodecError.appendFailed("append after finish") }
        guard audioInput == nil || audioAttached else {
            throw FrameCodecError.appendFailed("attachAudio must precede the first video frame (interleave deadlock)")
        }
        guard frames.ndim == 5, frames.dim(1) > 0 else {
            throw FrameCodecError.badFrames("expected [1,t,H,W,3], got \(frames.shape)")
        }
        let t = frames.dim(1)
        let total = totalFrames ?? (frameIndex + t)
        for i in 0 ..< t {
            let (bytes, fw, fh) = bgraBytes(frames[0, i, 0..., 0..., 0...])
            guard let pool = adaptor.pixelBufferPool else { throw FrameCodecError.writerSetup("no pool") }
            let buffer = try pixelBuffer(bgra: bytes, width: fw, height: fh, pool: pool)
            // Bounded wait: a stall here must be a loud, localized error, never a silent forever-spin.
            var waited = 0.0
            while !input.isReadyForMoreMediaData {
                if writer.status == .failed {
                    throw FrameCodecError.appendFailed("writer FAILED at frame \(frameIndex)/\(total): \(String(describing: writer.error))")
                }
                usleep(2000); waited += 0.002
                if waited > 90 {
                    throw FrameCodecError.appendFailed("encoder stalled at frame \(frameIndex)/\(total): isReadyForMoreMediaData=false for 90s (writer.status=\(writer.status.rawValue)). If audio is present this smells like the AVAssetWriter interleave deadlock — audio must be appended BEFORE the video frames.")
                }
            }
            guard adaptor.append(buffer, withPresentationTime: CMTimeMultiply(frameDuration, multiplier: Int32(frameIndex))) else {
                throw FrameCodecError.appendFailed("frame \(frameIndex)/\(total) err=\(String(describing: writer.error))")
            }
            if frameIndex % 8 == 0 { MLXProfiler.shared.note("encode-mp4 frame \(frameIndex)/\(total)") }
            frameIndex += 1
        }
    }

    /// Close the video track, finalize the container, return the bytes, delete the temp file.
    public func finish() async throws -> Data {
        guard !finished else { throw FrameCodecError.appendFailed("finish called twice") }
        finished = true
        defer { try? FileManager.default.removeItem(at: url) }
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed, FileManager.default.fileExists(atPath: url.path) else {
            throw FrameCodecError.writeIncomplete("status=\(writer.status.rawValue) err=\(String(describing: writer.error))")
        }
        return try Data(contentsOf: url)
    }
}

/// Encode channels-last frames [1, T, H, W, 3] in [-1, 1] as an H.264 MP4 at `fps`, optionally
/// muxing a stereo audio track [1, 2, T_audio] in [-1, 1] at `audioSampleRate`. Composed over
/// `MP4StreamWriter` — one code path for the materialized and streamed lanes.
@InferenceActor
public func encodeMP4(frames: MLXArray, fps: Double, audio: MLXArray? = nil, audioSampleRate: Double = 48000,
                      software: Bool = false) async throws -> Data {
    guard frames.ndim == 5, frames.dim(1) > 0 else {
        throw FrameCodecError.badFrames("expected [1,T,H,W,3], got \(frames.shape)")
    }
    let writer = try MP4StreamWriter(width: frames.dim(3), height: frames.dim(2), fps: fps,
                                     expectAudio: audio != nil, audioSampleRate: audioSampleRate,
                                     software: software)
    try writer.attachAudio(audio)
    try writer.appendSync(framesChunk: frames, totalFrames: frames.dim(1))
    return try await writer.finish()
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
