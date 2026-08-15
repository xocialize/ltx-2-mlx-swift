import Foundation
import AVFoundation
import CoreVideo

// Honest test of "AVFoundation can't read straight off /Volumes/Satechi" — an unsourced
// parenthetical in LTX25-PORT-PLAN.md §V. Tests BOTH halves of what "can't read" could mean:
// (1) asset/track metadata load, (2) real frame DECODE via AVAssetReader.
func probe(_ p: String) async {
    let url = URL(fileURLWithPath: p)
    let exists = FileManager.default.fileExists(atPath: p)
    print("── \(p)")
    print("   exists=\(exists)")
    guard exists else { print("   SKIP"); return }
    let t0 = Date()
    let asset = AVURLAsset(url: url)
    do {
        let dur = try await asset.load(.duration)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { print("   ❌ no video track"); return }
        let size = try await track.load(.naturalSize)
        let fps = try await track.load(.nominalFrameRate)
        print(String(format: "   ✅ metadata: %.2fs  %.0fx%.0f  %.2f fps  (%.0f ms)",
                     CMTimeGetSeconds(dur), size.width, size.height, fps,
                     Date().timeIntervalSince(t0) * 1000))
        let d0 = Date()
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(out)
        guard reader.startReading() else {
            print("   ❌ startReading FAILED: \(String(describing: reader.error))"); return
        }
        var n = 0
        var checksum: UInt64 = 0
        while let sb = out.copyNextSampleBuffer() {
            n += 1
            if n == 1, let pb = CMSampleBufferGetImageBuffer(sb) {
                CVPixelBufferLockBaseAddress(pb, .readOnly)
                if let base = CVPixelBufferGetBaseAddress(pb) {
                    let bytes = base.assumingMemoryBound(to: UInt8.self)
                    for i in stride(from: 0, to: 4096, by: 7) { checksum &+= UInt64(bytes[i]) }
                }
                CVPixelBufferUnlockBaseAddress(pb, .readOnly)
            }
        }
        let st = reader.status
        print(String(format: "   %@ decode: %d frames, status=%d, first-frame checksum=%llu  (%.0f ms)",
                     (st == .completed && n > 0 ? "✅" : "❌"), n, st.rawValue, checksum,
                     Date().timeIntervalSince(d0) * 1000))
        if let e = reader.error { print("   reader.error=\(e)") }
    } catch {
        print("   ❌ THREW: \(error)")
    }
}

let sem = DispatchSemaphore(value: 0)
Task {
    for p in CommandLine.arguments.dropFirst() { await probe(p) }
    sem.signal()
}
sem.wait()
