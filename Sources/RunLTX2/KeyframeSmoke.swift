// KeyframeSmoke.swift — `--keyframe-smoke`: first+last on REAL weights, graded by the APP'S harness
// (AB-A-0025). Deliberately drives the PACKAGE surface (metaData list + initImage), not the
// pipeline directly, so what is graded is what the app will call.
import Foundation
import MLX
import MLXLTX2
import MLXToolKit

func keyframeSmoke() async throws {
    let env = ProcessInfo.processInfo.environment
    let smoke = env["LTX_KF_DIR"] ?? "/Volumes/Satechi/Models/_smoke"
    let first = "\(smoke)/kf-first.png", last = "\(smoke)/kf-last.png"
    let w = Int(env["LTX_KF_W"] ?? "1280")!, h = Int(env["LTX_KF_H"] ?? "704")!
    let f = Int(env["LTX_KF_F"] ?? "121")!
    let out = env["LTX_KF_OUT"] ?? "\(smoke)/kf-local.mp4"

    var c = LTX2Configuration(family: .ltx25,
                              modelsRootDirectory: URL(fileURLWithPath: "/Volumes/Satechi/Models"),
                              profile: .standard64)
    c.quant = .int8
    c.ltxDirectory = URL(fileURLWithPath: "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
    c.transformerPath = URL(fileURLWithPath:
        "/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx-ditq8/transformer-distilled.safetensors")
    let pkg = MLXLTX25Package(configuration: c)
    try await pkg.load()

    let firstData = try Data(contentsOf: URL(fileURLWithPath: first))
    print("[kf-smoke] first+last \(w)×\(h)×\(f)  anchors: \(first) / \(last)")
    let req = T2VRequest(
        prompt: env["LTX_KF_PROMPT"] ?? "a cinematic shot, smooth natural motion",
        initImage: Image(format: .png, data: firstData),        // frame 0 REPLACES the latent
        numFrames: f, fps: 24, width: w, height: h, seed: 42,
        metaData: [KeyframeMetaKeys.keyframes: .array([
            .object([KeyframeMetaKeys.path: .string(last),
                     KeyframeMetaKeys.frame: .int(f - 1)])       // last frame is APPENDED
        ])])
    let t0 = Date()
    let resp = try await pkg.run(req) as! T2VResponse
    try resp.video.data.write(to: URL(fileURLWithPath: out))
    print(String(format: "[kf-smoke] wrote %@ (%.1f MB) in %.0fs",
                 out, Double(resp.video.data.count) / 1e6, Date().timeIntervalSince(t0)))
    print("[kf-smoke] grade with the APP'S harness:  \(smoke)/kf-endpoint-psnr.sh \(out)")
}
