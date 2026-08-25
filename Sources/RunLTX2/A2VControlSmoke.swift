// A2VControlSmoke.swift — `--a2v-control` (AB-A-0027). OUR port under EXACTLY the inputs both
// vendor arms received: same audio file, same init frame, same geometry, same seed, same prompt.
// The operator's original clip came from the app with parameters we do not know, so it was never a
// fair control — this is.
import Foundation
import MLX
import MLXLTX2
import MLXToolKit

func a2vControlSmoke() async throws {
    let env = ProcessInfo.processInfo.environment
    let audio = env["A2V_AUDIO"]!, portrait = env["A2V_IMAGE"]!, out = env["A2V_OUT"]!
    let w = 704, h = 512, f = 241

    // ⚠️ TIER MATTERS FOR THE CONTROL, not for quality: standard64 caps maxFrames at 161, so a
    // 241-frame request is CLAMPED and the audio truncated with it — the first attempt produced
    // 161 frames against the vendor's 241 and was not a like-for-like control at all.
    let profile: LTX2Profile = (env["A2V_TIER"].flatMap { LTX2Profile(rawValue: $0) }) ?? .max128
    var c = LTX2Configuration(family: .ltx25,
                              modelsRootDirectory: URL(fileURLWithPath: "/Volumes/Satechi/Models"),
                              profile: profile)
    c.quant = .int8
    c.ltxDirectory = URL(fileURLWithPath: "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
    c.transformerPath = URL(fileURLWithPath:
        "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx-ditq8/transformer-distilled.safetensors")
    let pkg = MLXLTX25Package(configuration: c)
    try await pkg.load()

    // The audio rides VEditRequest.video as the carrier (AB-A-0023); only its audio track is read.
    let carrier = try Data(contentsOf: URL(fileURLWithPath: audio))
    let req = VEditRequest(
        video: Video(format: .mp4, data: carrier),
        prompt: "close-up portrait of a red-haired woman talking to the camera against a plain tan "
              + "background, mouth moving as she speaks",
        width: w, height: h, numFrames: f, fps: 24, seed: 42,
        mode: VEditModes.audioToVideo,
        metaData: [KeyframeMetaKeys.initPath: .string(portrait)])
    let t0 = Date()
    let resp = try await pkg.run(req) as! VEditResponse
    // Fail loudly if the tier clamped us off the vendor's geometry — a shorter clip is not a control.
    let geo = c.resolvedGeometry(width: w, height: h, numFrames: f, fps: 24)
    if geo.deliveredFrames != f {
        print("[a2v-control] ⚠️ CLAMPED: asked \(f) frames, tier \(profile.rawValue) delivers "
            + "\(geo.deliveredFrames) — NOT comparable to the vendor's \(f)")
    }
    try resp.video.data.write(to: URL(fileURLWithPath: out))
    print(String(format: "[a2v-control] wrote %@ (%.1f MB) in %.0fs",
                 out, Double(resp.video.data.count) / 1e6, Date().timeIntervalSince(t0)))
}
