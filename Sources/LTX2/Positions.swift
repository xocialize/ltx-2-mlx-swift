// Positions.swift — pixel-space RoPE positions + the distilled sigma schedule.
//
// 1:1 port of ltx_core_mlx/utils/positions.py (compute_video_positions /
// compute_audio_positions / compute_audio_token_count) + the DISTILLED_SIGMAS
// table from ltx_pipelines_mlx/scheduler.py.

import Foundation
import MLX

public enum Positions {
    static let videoTemporalScale: Float = 8
    static let videoSpatialScale: Float = 32
    static let audioDownsampleFactor: Float = 4
    static let audioHopLength: Float = 160
    static let audioSampleRate: Float = 16000
    static let audioLatentsPerSecond: Float = 16000.0 / 160.0 / 4.0  // 25

    /// Distilled Euler sigma schedule (stage 1; includes terminal 0.0).
    public static let distilledSigmas: [Float] = [
        1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
    ]
    /// Stage-2 refine sigma schedule (two-stage distilled; 3 steps).
    public static let stage2Sigmas: [Float] = [0.909375, 0.725, 0.421875, 0.0]

    public static func audioTokenCount(numFrames: Int, fps: Double) -> Int {
        Int((Double(numFrames) / fps * Double(audioLatentsPerSecond)).rounded())
    }

    /// 3D pixel-space video positions, (1, F·H·W, 3) = [time/fps, h·32+16, w·32+16].
    public static func video(F: Int, H: Int, W: Int, fps: Float) -> MLXArray {
        let idx = MLXArray(0 ..< F).asType(.float32)
        let fStarts = MLX.maximum(idx * videoTemporalScale + 1 - videoTemporalScale, 0.0)
        let fEnds = MLX.maximum((idx + 1) * videoTemporalScale + 1 - videoTemporalScale, 0.0)
        let fMids = (fStarts + fEnds) / 2.0 / fps                               // (F,)
        let hMids = MLXArray(0 ..< H).asType(.float32) * videoSpatialScale + videoSpatialScale / 2.0
        let wMids = MLXArray(0 ..< W).asType(.float32) * videoSpatialScale + videoSpatialScale / 2.0

        let fGrid = MLX.broadcast(fMids.reshaped(F, 1, 1), to: [F, H, W])
        let hGrid = MLX.broadcast(hMids.reshaped(1, H, 1), to: [F, H, W])
        let wGrid = MLX.broadcast(wMids.reshaped(1, 1, W), to: [F, H, W])
        let pos = MLX.stacked([fGrid, hGrid, wGrid], axis: -1).reshaped(F * H * W, 3)
        return pos.expandedDimensions(axis: 0).asType(.float32)                 // (1, N, 3)
    }

    /// 1D audio positions in seconds, (1, T, 1).
    public static func audio(tokens T: Int) -> MLXArray {
        let idx = MLXArray(0 ..< T).asType(.float32)
        let starts = MLX.maximum(idx * audioDownsampleFactor + 1 - audioDownsampleFactor, 0.0) * audioHopLength / audioSampleRate
        let ends = MLX.maximum((idx + 1) * audioDownsampleFactor + 1 - audioDownsampleFactor, 0.0) * audioHopLength / audioSampleRate
        let mids = (starts + ends) / 2.0
        return mids.reshaped(1, T, 1).asType(.float32)
    }
}
