// AudioInput.swift — decode a dub-audio source into the LipDub ingest tensor (IC-LORA-PLAN P3b).
//
// Accepts a standalone audio file (wav/m4a/…) OR a video container (uses its audio track) —
// AVAssetReader converts to 16 kHz STEREO float PCM (the oracle's `load_audio(target_sample_rate
// =16000, mono=False)`; mono sources are channel-duplicated by the reader). Output (1, 2, T)
// float32 in [-1,1] — exactly what `AudioVAEEncoder.encode(waveform:)` consumes.

import AVFoundation
import Foundation
import MLX
import MLXToolKit

enum AudioInput {

    /// Decode `url`'s (first) audio track → (1, 2, T) float32 at `sampleRate`. `maxSeconds`
    /// trims long sources to the clip span (LipDub aligns the dub to the video duration).
    static func referenceWaveform(url: URL, sampleRate: Double = 16000,
                                  maxSeconds: Double? = nil) async throws -> MLXArray {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw PackageError.configurationMismatch(
                expected: "an audio track in the dub source", got: "none in \(url.lastPathComponent)")
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 2,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
        ])
        reader.add(output)
        reader.startReading()

        var interleaved = [Float]()
        while let sample = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(sample) else { continue }
            var total = 0
            var pointer: UnsafeMutablePointer<Int8>?
            CMBlockBufferGetDataPointer(block, atOffset: 0, lengthAtOffsetOut: nil,
                                        totalLengthOut: &total, dataPointerOut: &pointer)
            if let pointer {
                pointer.withMemoryRebound(to: Float.self, capacity: total / 4) { fp in
                    interleaved.append(contentsOf: UnsafeBufferPointer(start: fp, count: total / 4))
                }
            }
        }
        guard reader.status == .completed, interleaved.count >= 2 else {
            throw PackageError.configurationMismatch(
                expected: "decodable dub audio", got: "reader status \(reader.status.rawValue)")
        }
        var frames = interleaved.count / 2
        if let maxSeconds { frames = min(frames, Int(maxSeconds * sampleRate)) }
        // Deinterleave LRLR… → (1, 2, T).
        var channels = [Float](repeating: 0, count: 2 * frames)
        for i in 0 ..< frames {
            channels[i] = interleaved[2 * i]
            channels[frames + i] = interleaved[2 * i + 1]
        }
        return MLXArray(channels, [1, 2, frames])
    }
}
