// KeyframeMetaKeys.swift — supplied keyframes ride metaData (AB-A-0025).
//
// A genuine LIST, at the app's request: last-frame and multi-keyframe (≤5) are the SAME code path
// in the vendor CLI (`--image PATH FRAME_IDX STRENGTH`, repeatable), so a single-purpose
// "lastImage" field would have needed widening immediately. `MetaValue` carries `.array`/`.object`,
// so the list is expressible directly rather than as flat indexed keys:
//
//     metaData["kf.keyframes"] = .array([
//         .object(["path": .string("/…/kf-last.png"), "frame": .int(120)]),          // strength defaults to 1.0
//         .object(["path": .string("/…/mid.png"),     "frame": .int(60), "strength": .double(0.8)]),
//     ])
//
// ⚠️ PATHS, not bytes: `MetaValue` has no data case, so an image arrives as a file path — the same
// interim shape the IC reference image uses, pending a canonical conditioning-input contract
// (AB-A-0023 asks for the audio equivalent).
//
// ⚠️ frame 0 is REJECTED, not quietly accepted. The oracle's `combined_image_conditionings` splits
// by index: 0 REPLACES the latent (our i2v path — `T2VRequest.initImage`), every other index
// GUIDES via an appended token block. Accepting 0 here would apply guide semantics where the
// vendor replaces, which surfaces as a quality complaint rather than a routing error.

import Foundation
import MLX
import LTX2
import MLXToolKit

public enum KeyframeMetaKeys {
    /// metaData key holding the keyframe list.
    public static let keyframes = "kf.keyframes"
    /// Per-entry object fields.
    public static let path = "path", frame = "frame", strength = "strength"

    /// Upstream's local 2.5 cap. Each keyframe costs a full latent frame of tokens, so this is a
    /// token-budget guard, not a style preference.
    public static let maxKeyframes = 5

    /// Build pipeline requests from metaData. Returns [] when no keyframes were asked for.
    public static func parse(_ meta: MetaData) throws -> [KeyframeRequest] {
        guard let raw = meta[keyframes] else { return [] }
        guard let list = raw.asArray else {
            throw PackageError.configurationMismatch(
                expected: "metaData[\(keyframes)] to be an ARRAY of objects", got: "a non-array value")
        }
        guard !list.isEmpty else { return [] }
        guard list.count <= maxKeyframes else {
            throw PackageError.configurationMismatch(
                expected: "at most \(maxKeyframes) keyframes (each costs a full latent frame of tokens)",
                got: "\(list.count)")
        }
        var out: [KeyframeRequest] = []
        for (i, entry) in list.enumerated() {
            guard let o = entry.asObject else {
                throw PackageError.configurationMismatch(
                    expected: "\(keyframes)[\(i)] to be an object with \(path)/\(frame)", got: "a non-object entry")
            }
            guard let p = o[path]?.asString, !p.isEmpty else {
                throw PackageError.configurationMismatch(
                    expected: "\(keyframes)[\(i)].\(path) — an image file path", got: "absent or empty")
            }
            guard let f = o[frame]?.asInt else {
                throw PackageError.configurationMismatch(
                    expected: "\(keyframes)[\(i)].\(frame) — an integer pixel frame index", got: "absent or non-integral")
            }
            guard f > 0 else {
                throw PackageError.configurationMismatch(
                    expected: "\(keyframes)[\(i)].\(frame) > 0 — frame 0 REPLACES the latent and "
                        + "rides initImage (the oracle's combined_image_conditionings split)",
                    got: "\(f)")
            }
            let s = o[strength]?.asFloat ?? 1.0
            // Re-read per stage: each stage encodes at its OWN geometry, as the vendor does.
            out.append(KeyframeRequest(frameIdx: f, strength: s) { w, h in
                try ImageInput.initFrameTensor(path: p, width: w, height: h)
            })
        }
        return out
    }
}
