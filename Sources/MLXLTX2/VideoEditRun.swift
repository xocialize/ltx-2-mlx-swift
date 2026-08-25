// VideoEditRun.swift — the `.videoEdit` surface: local retake/extend (AB-R-0133 piece 2c).
//
// Desktop-parity wiring over `LTX2Pipeline.retake/extend`. The span rides metaData because
// `VEditRequest` has no canonical time-span field yet (flagged on AB-A-0021 — if a span ever
// becomes canonical, these keys become the compatibility alias).

import AVFoundation
import Foundation
import MLX
import LTX2
import MLXToolKit

public enum RetakeMetaKeys {
    public static let start = "retake.start"          // seconds, Double
    public static let duration = "retake.duration"    // seconds, Double
    public static let extendSeconds = "extend.seconds" // seconds, Double (extend mode)
}

public enum VEditModes {
    public static let replaceVideo: Mode = "replace_video"
    public static let replaceAudio: Mode = "replace_audio"
    public static let replaceAudioAndVideo: Mode = "replace_audio_and_video"
    public static let extend: Mode = "extend"
    /// a2v — generate video AGAINST a supplied audio track (the track is held frozen and comes
    /// back unmodified). The audio rides `VEditRequest.video`: a clip whose audio drives the
    /// generation, or an audio-only container (no video track ⇒ geometry falls back to the
    /// request's width/height/numFrames, then to defaults). Filed as AB-A-0023 — the contract has
    /// no audio-input field yet, the same reason retake's span rides metaData.
    public static let audioToVideo: Mode = "audio_to_video"
}

/// Lift the 16 kHz reader output to 48 kHz by linear interpolation. FrameCodec's AAC writer
/// REJECTS 16 kHz outright ("Cannot Encode Media", found by the e2e smoke), and the source is
/// already band-limited to 8 kHz by the reader, so this loses nothing further — it is still a V1
/// caveat versus demuxing the original 48 kHz track untouched.
func upsample16to48(_ src: MLXArray) -> MLXArray {
    let t = src.dim(2)
    let a = src.asType(.float32)
    let nxt = MLX.concatenated([a[0..., 0..., 1 ..< t], a[0..., 0..., (t - 1) ..< t]], axis: 2)
    let up = MLX.stacked([a, a * (2.0 / 3.0) + nxt * (1.0 / 3.0),
                          a * (1.0 / 3.0) + nxt * (2.0 / 3.0)], axis: 3)
        .reshaped(1, src.dim(1), 3 * t)
    eval(up)
    return up
}

extension MLXLTX2Package {
    func runVideoEdit(_ vedit: VEditRequest, pipeline: LTX2Pipeline) async throws -> VEditResponse {
        // 1. Source to a temp file for the AVFoundation readers.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-vedit-\(UUID().uuidString).\(vedit.video.format.rawValue)")
        try vedit.video.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let asset = AVURLAsset(url: tmp)
        let srcDuration = try await asset.load(.duration).seconds
        let track = try await asset.loadTracks(withMediaType: .video).first
        var natural = CGSize(width: 704, height: 512)
        var nominalFPS: Double = 24
        if let track {
            natural = try await track.load(.naturalSize)
            nominalFPS = Double(try await track.load(.nominalFrameRate))
        }
        let fps = vedit.fps ?? nominalFPS

        // a2v takes only the AUDIO from the container, so it branches before the frame read: an
        // audio-only container has no video track to decode, and even for a real clip the source
        // frames are never used (the video is generated from noise against the track).
        if (vedit.mode ?? VEditModes.replaceAudioAndVideo) == VEditModes.audioToVideo {
            return try await runAudioToVideo(vedit, pipeline: pipeline, source: tmp,
                                            sourceDuration: srcDuration, natural: natural, fps: fps)
        }

        // 2. Geometry: SOURCE-derived dims snap DOWN to /32 and never upscale (the vendor's
        //    video_resolution.py rule — opposite to generation's snap-UP /64), then clamp to the
        //    tier envelope like every other run.
        var w = min(vedit.width ?? Int(natural.width), Int(natural.width))
        var h = min(vedit.height ?? Int(natural.height), Int(natural.height))
        if let p = configuration.profile { w = min(w, p.maxWidth); h = min(h, p.maxHeight) }
        w = max(64, (w / 32) * 32); h = max(64, (h / 32) * 32)
        var frames = vedit.numFrames ?? Int((srcDuration * fps).rounded())
        if let p = configuration.profile { frames = min(frames, p.maxFrames) }
        frames = max(9, ((frames - 1) / 8) * 8 + 1)                       // 8k+1 grid

        // 3. Read the source (frames at target geometry; audio 16 kHz stereo when present).
        let pixels = try await VideoInput.referenceClipFrames(
            url: tmp, width: w, height: h, frames: frames, fps: fps)
        let sourceAudio = try? await AudioInput.referenceWaveform(url: tmp, maxSeconds: Double(frames) / fps)

        // 4. Mode + span from metaData.
        let mode = vedit.mode ?? VEditModes.replaceAudioAndVideo
        func metaDouble(_ key: String) -> Double? { vedit.metaData[key]?.asFloat.map(Double.init) }
        // Run-phase progress: the same core→engine bridge the T2V path installs
        // (`MLXLTX2Package.swift`, "Run-phase progress"). Without it a retake ran completely
        // dark — the core emitted its encode/denoise/decode phases into an unbound task-local
        // and the engine's RunProgress plane saw nothing (the app's AB-A-0021 small ask).
        let forward: LTX2Progress.Sink = { e in
            RunProgress.report(RunPhase(rawValue: e.phase.rawValue),
                               step: e.step, totalSteps: e.totalSteps,
                               stage: e.stage, totalStages: e.totalStages)
        }
        let out: LTX2Pipeline.Output = try await LTX2Progress.$sink.withValue(forward) {
            () async throws -> LTX2Pipeline.Output in
            if mode == VEditModes.extend {
                guard let add = metaDouble(RetakeMetaKeys.extendSeconds), add > 0 else {
                    throw PackageError.configurationMismatch(
                        expected: "metaData[\(RetakeMetaKeys.extendSeconds)] > 0 for extend",
                        got: "absent or non-positive")
                }
                return try await pipeline.extend(
                    prompt: vedit.prompt, sourcePixels: pixels, sourceAudioWaveform: sourceAudio,
                    fps: fps, addSeconds: add, seed: vedit.seed)
            }
            guard let start = metaDouble(RetakeMetaKeys.start),
                  let duration = metaDouble(RetakeMetaKeys.duration), duration > 0 else {
                throw PackageError.configurationMismatch(
                    expected: "metaData[\(RetakeMetaKeys.start)] + [\(RetakeMetaKeys.duration)]",
                    got: "absent or non-positive")
            }
            let rmode: RetakeMode = switch mode {
            case VEditModes.replaceVideo: .replaceVideo
            case VEditModes.replaceAudio: .replaceAudio
            default: .replaceAudioAndVideo
            }
            return try await pipeline.retake(
                prompt: vedit.prompt, sourcePixels: pixels, sourceAudioWaveform: sourceAudio,
                fps: fps, start: start, duration: duration, mode: rmode, seed: vedit.seed)
        }

        // 5. Mux. ⚠️ replace_video keeps the SOURCE track (frozen audio round-tripped through the
        //    VAE is near- but not bit-identical) — 16 kHz reader output, a V1 caveat vs the
        //    original 48 kHz track, noted rather than hidden.
        let muxAudio: MLXArray?
        let muxRate: Double
        if mode == VEditModes.replaceVideo, let src = sourceAudio {
            muxAudio = upsample16to48(src); muxRate = 48000
        } else {
            muxAudio = out.audio; muxRate = 48000
        }
        // (B,C,F,H,W) → channels-last (B,F,H,W,C), exactly as the T2V mux path does — passing
        // the raw pipeline layout hands the writer width=dim(3)=H and made AVFoundation refuse
        // the session outright ("Cannot Open"; found by the e2e smoke).
        let framesCL = out.video.transposed(0, 2, 3, 4, 1)
        let mp4 = try await encodeMP4(frames: framesCL, fps: fps, audio: muxAudio,
                                audioSampleRate: muxRate)
        return VEditResponse(video: Video(format: .mp4, data: mp4,
                                          // framesCL is (B,F,H,W,C) — dim(1) is FRAMES here.
                                          // `out.video` is (B,C,F,H,W), so its dim(1) is the
                                          // channel count and reported 3/fps for every edit.
                                          durationSeconds: Double(framesCL.dim(1)) / fps,
                                          frameRate: fps))
    }

    /// a2v — generate video against a supplied audio track (`VEditModes.audioToVideo`).
    ///
    /// Mirrors LTX Desktop's `DistilledA2VPipeline` contract: two-stage distilled, audio frozen in
    /// both stages, and the ORIGINAL waveform muxed back rather than the VAE round-trip.
    func runAudioToVideo(_ vedit: VEditRequest, pipeline: LTX2Pipeline, source: URL,
                         sourceDuration: Double, natural: CGSize, fps: Double) async throws -> VEditResponse {
        // Geometry. Unlike retake there is no source video to match, so an explicit request wins,
        // then the container's natural size, then the t2v default. Snap DOWN to /64, not /32: the
        // spatial-x2 upsampler we ship needs the stage-2 latent grid even, and a /32-but-not-/64
        // target would fail `resolveTwoStageGeometry` after the text encode had already run.
        var w = vedit.width ?? Int(natural.width)
        var h = vedit.height ?? Int(natural.height)
        if let p = configuration.profile { w = min(w, p.maxWidth); h = min(h, p.maxHeight) }
        w = max(64, (w / 64) * 64); h = max(64, (h / 64) * 64)

        // Duration follows the AUDIO when the caller doesn't pin frames — a2v's premise is that
        // the track is the ground truth, so the default is "cover the track", clamped to the tier.
        var frames = vedit.numFrames ?? Int((sourceDuration * fps).rounded())
        if let p = configuration.profile { frames = min(frames, p.maxFrames) }
        frames = max(9, ((frames - 1) / 8) * 8 + 1)                       // 8k+1 grid

        // Read the track at exactly the video's span. A longer track is truncated here, and a
        // shorter one is zero-padded downstream in `encodeFrozenAudio` (Desktop pads too).
        let waveform = try await AudioInput.referenceWaveform(
            url: source, maxSeconds: Double(frames) / fps)

        let forward: LTX2Progress.Sink = { e in
            RunProgress.report(RunPhase(rawValue: e.phase.rawValue),
                               step: e.step, totalSteps: e.totalSteps,
                               stage: e.stage, totalStages: e.totalStages)
        }
        // Image conditioning on a2v (AB-T-0096). Previously these were parsed nowhere on this
        // path, so anything a caller attached was SILENTLY dropped — which is what made a missing
        // capability look like a quality problem on AB-A-0027.
        let kfReqs = try KeyframeMetaKeys.parse(vedit.metaData)
        let initClosure: ((Int, Int) throws -> MLXArray)? =
            vedit.metaData[KeyframeMetaKeys.initPath]?.asString.map { p in
                { w, hh in try ImageInput.initFrameTensor(path: p, width: w, height: hh) }
            }
        let out = try await LTX2Progress.$sink.withValue(forward) {
            () async throws -> LTX2Pipeline.Output in
            try await pipeline.audioToVideo(
                prompt: vedit.prompt, audioWaveform: waveform,
                height: h, width: w, numFrames: frames, fps: fps, seed: vedit.seed,
                keyframes: kfReqs, initImage: initClosure)
        }

        // `Output.audio` is the ORIGINAL 16 kHz waveform (a2v returns the track untouched), so it
        // still needs the 48 kHz lift the AAC writer demands.
        let muxAudio = out.audio.map { upsample16to48($0) }
        let framesCL = out.video.transposed(0, 2, 3, 4, 1)                // (B,C,F,H,W) → (B,F,H,W,C)
        let mp4 = try await encodeMP4(frames: framesCL, fps: fps, audio: muxAudio,
                                      audioSampleRate: 48000)
        return VEditResponse(video: Video(format: .mp4, data: mp4,
                                          durationSeconds: Double(framesCL.dim(1)) / fps,
                                          frameRate: fps))
    }

}
