// QuantAB.swift — bf16 vs int8 text-encoder perceptual A/B.
//
// The numbers say int8 g64 is faithful (valid-token cosine 0.999820 vs a 0.999879 bf16
// floor). But the q8 SAMPLE MOVES — mean −0.1421 vs −0.1274 — so it is a different video,
// not a degraded copy of the same one. That is the C7/C2 situation: the arms do not share a
// composition, so no no-reference metric can settle it and a single pair proves nothing.
// Only operator judgement, repeated across seeds, can.
//
// Design follows the lessons this program paid for:
//   * MULTI-SEED. C2's spatial verdict only became a capability result at 4/4; one pair is
//     equally consistent with a luckier roll.
//   * BLINDED. Every A/B in this program so far shipped files named `_bf16`/`_q8`, so the
//     arm was visible while judging. Here each clip gets an opaque name and the key is
//     written to a separate file the judge should not open first.
//   * ENCODER-ISOLATED. The two model trees differ ONLY in gemma4-12b-ltx-v1/; everything
//     else is symlinked to the same bytes, so any difference is the encoder.

import Foundation
import LTX2
import MLXLTX2
import MLX

private func sayAB(_ m: String) { print(m); fflush(stdout) }

func quantAB(width: Int, height: Int, frames: Int, seeds: [UInt64]) async throws {
    let bf16Dir = URL(fileURLWithPath: "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
    // B arm is selectable so the same blinded protocol covers the DiT experiment:
    //   LTX_AB_TREE=ltx-2.5-mlx-ditq8  → int8 DiT (bf16 encoder)
    //   unset                          → int8 ENCODER (bf16 DiT), the original arm
    // Both trees differ from bf16Dir in exactly ONE component, which is what keeps the
    // comparison attributable.
    let bTree = ProcessInfo.processInfo.environment["LTX_AB_TREE"] ?? "ltx-2.5-mlx-q8"
    let q8Dir = URL(fileURLWithPath: "/Volumes/Satechi/Models/xocialize/\(bTree)")
    for d in [bf16Dir, q8Dir] where !FileManager.default.fileExists(atPath: d.path) {
        sayAB("[quant-ab] missing model tree: \(d.path)"); exit(2)
    }
    sayAB("[quant-ab] arm A tree = ltx-2.5-mlx (bf16)   arm B tree = \(bTree)")
    let outName = bTree == "ltx-2.5-mlx-q8" ? "quant-ab" : "quant-ab-\(bTree)"
    let out = URL(fileURLWithPath: "/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/LTX_TESTING/\(outName)")
    try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

    // Prompts chosen to cover what this program has seen break first: a face (the pruna
    // face-smearing risk), fine text-like structure, and high-motion detail.
    let prompts = [
        "A close-up portrait of an older fisherman in a wool cap, weathered face, soft window light, shallow depth of field. Cinematic 35mm.",
        "A lone lighthouse on a rocky headland during a storm, waves exploding against the rocks, the beam sweeping through sheets of rain.",
        "A hummingbird hovering at a red trumpet flower in a sunlit garden, wings blurred, petals trembling. Macro, high shutter speed.",
    ]

    var key: [String] = ["# quant-ab key — DO NOT OPEN BEFORE JUDGING", ""]
    var pairIndex = 0

    for (pi, prompt) in prompts.enumerated() {
        for seed in seeds {
            // Randomise which arm gets which letter, per pair.
            let bf16First = (seed &+ UInt64(pi)) % 2 == 0
            let arms: [(String, URL)] = bf16First
                ? [("A", bf16Dir), ("B", q8Dir)]
                : [("A", q8Dir), ("B", bf16Dir)]

            for (letter, dir) in arms {
                let pipeline = try await LTX2Pipeline.load(ltxDir: dir, gemmaDir: LTX2Pipeline.gemma4Dir(ltxDir: dir),
                                                          transformerPath: nil)
                let o = try await pipeline.t2vTwoStage(prompt: prompt, height: height, width: width,
                                                       numFrames: frames, fps: 24, seed: seed)
                let px = o.video
                eval(px)
                let writer = try MP4StreamWriter(width: width, height: height, fps: 24, expectAudio: false)
                try writer.appendSync(framesChunk: px.transposed(0, 2, 3, 4, 1), totalFrames: frames)
                let data = try await writer.finish()
                let name = String(format: "pair%02d_%@.mp4", pairIndex, letter)
                try data.write(to: out.appending(path: name))
                let arm = dir == bf16Dir ? "bf16" : "int8-g64"
                key.append("pair\(String(format: "%02d", pairIndex))_\(letter) = \(arm)   seed=\(seed)  prompt=\(pi)")
                sayAB("[quant-ab] wrote \(name)  (\(arm), seed \(seed))")
            }
            key.append("")
            pairIndex += 1
        }
    }
    try key.joined(separator: "\n").write(to: out.appending(path: "KEY.txt"), atomically: true, encoding: .utf8)
    sayAB("[quant-ab] \(pairIndex) blinded pairs in \(out.path)")
    sayAB("[quant-ab] judge A vs B per pair, THEN read KEY.txt")
}
