// TiledDiT.swift — drop-in DiT wrapper that runs the video forward tile-by-tile.
//
// Sits at the ONE production seam (`DenoiseLoop.x0`), receives the full token sequence, and for
// each tile: slices the video state, runs the inner denoiser on the smaller sequence, and blends
// the result back with trapezoidal weights. Audio is passed WHOLE to every tile and the outputs
// averaged — per-tile audio differs only through the joint audio↔video cross-attention, so the
// average is meaningful rather than redundant (the oracle's rationale; its cost scales with tile
// count, which `TILING-PLAN.md` flags for measurement).
//
// Composition with block streaming is forced by construction: streaming lives INSIDE
// `DiT.callAsFunction` (a branch on `store.streamer`), so a tiler can only sit outside —
// `TiledDiT(inner: DiT(streaming:))`. Each tile is therefore one full granule sweep.
// 🚨 That is why `LTX2Pipeline.armStreamingGate` must be armed with `largestTileTokenCount`, not
// the untiled count: the kit measures per-forward tokens, and arming with the untiled total means
// no forward ever reaches the threshold, the verdict stays `.undecided` forever, and the run
// streams unconditionally regardless of the arithmetic (TILING-PLAN hazard 1).

import Foundation
import MLX
import MLXProfiling

/// The forward surface `DenoiseLoop` drives. `DiT` and `TiledDiT` both satisfy it, so the loop is
/// agnostic to whether tiling is installed.
public protocol LTXDenoiser {
    func callAsFunction(
        videoLatent: MLXArray, audioLatent: MLXArray, sigma: MLXArray,
        videoText: MLXArray?, audioText: MLXArray?,
        videoPositions: MLXArray, audioPositions: MLXArray,
        videoTimesteps: MLXArray?, audioTimesteps: MLXArray?,
        keyframesMask: MLXArray?
    ) -> (video: MLXArray, audio: MLXArray)
}

extension DiT: LTXDenoiser {}

public struct TiledDiT: LTXDenoiser {
    public let inner: any LTXDenoiser
    public let tiler: VideoModalityTiler

    public init(inner: any LTXDenoiser, tiler: VideoModalityTiler) {
        self.inner = inner
        self.tiler = tiler
    }

    public func callAsFunction(
        videoLatent: MLXArray, audioLatent: MLXArray, sigma: MLXArray,
        videoText: MLXArray?, audioText: MLXArray?,
        videoPositions: MLXArray, audioPositions: MLXArray,
        videoTimesteps: MLXArray? = nil, audioTimesteps: MLXArray? = nil,
        keyframesMask: MLXArray? = nil
    ) -> (video: MLXArray, audio: MLXArray) {
        let B = videoLatent.dim(0), D = videoLatent.dim(2)
        var videoOut = MLXArray.zeros([B, videoLatent.dim(1), D]).asType(videoLatent.dtype)
        var audioParts: [MLXArray] = []
        audioParts.reserveCapacity(tiler.tiles.count)

        for (i, tile) in tiler.tiles.enumerated() {
            let span = MLXProfiler.shared.begin("denoise", "tile\(i)",
                note: "tok=\(tile.tokenCount) gen=\(tile.numGen) cond=\(tile.numCond)")

            let tLatent = tiler.sliceTokens(videoLatent, tile: tile)
            let tPositions = tiler.sliceTokens(videoPositions, tile: tile)
            // ⚠️ nil MUST stay nil. The oracle broadcasts scalar sigma into per-token timesteps
            // only inside its own Modality (so the tiler can slice them), then re-checks the
            // caller's kwarg before forwarding. Passing a broadcast through would move t2v — our
            // SHIPPING path — from the scalar AdaLN branch to the per-token one, a real numerical
            // divergence, and a consistent one a tiled-vs-untiled gate could mistake for expected
            // tiling drift. See TILING-PLAN hazard 3.
            let tTimesteps = videoTimesteps.map { ts in
                tiler.sliceTokens(ts.expandedDimensions(axis: -1), tile: tile).squeezed(axis: -1)
            }
            // The keyframes mask is per-VIDEO-TOKEN, so it must be sliced with the tile like
            // the latent and positions are. Forwarding the full-length mask would mark the
            // wrong tokens in every tile but the first — silently, since the shapes still
            // broadcast against a smaller tile only when the mask happens to be length 1.
            let tKeyframes = keyframesMask.map { tiler.sliceTokens($0, tile: tile) }

            let (vOut, aOut) = inner.callAsFunction(
                videoLatent: tLatent, audioLatent: audioLatent, sigma: sigma,
                videoText: videoText, audioText: audioText,
                videoPositions: tPositions, audioPositions: audioPositions,
                videoTimesteps: tTimesteps, audioTimesteps: audioTimesteps,
                keyframesMask: tKeyframes)

            videoOut = tiler.blend(vOut, tile: tile, into: videoOut)
            audioParts.append(aOut)
            // Keep the pool tile-bound — the same cadence the VAE tilers use. Without it the lazy
            // graph holds every tile's activations and tiling buys nothing.
            eval(videoOut)
            Memory.clearCache()
            MLXProfiler.shared.end(span)
        }

        let audioOut = audioParts.count == 1
            ? audioParts[0]
            : MLX.mean(MLX.stacked(audioParts, axis: 0), axis: 0)
        return (videoOut, audioOut)
    }
}
