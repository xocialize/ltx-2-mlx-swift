// ltx-granule-layout — lay an LTX transformer checkpoint out as per-block granule files for
// LTXBlockStreamer (STREAMING-PLAN.md; sibling of wan-core's wan-granule-layout).
//
//   ltx-granule-layout <transformer.safetensors> <output-dir> [block-prefix]
//                      [--source-repo <id>] [--source-revision <sha|main>]
//
// The default block prefix is the on-disk dialect `transformer.transformer_blocks.`, shared by
// LTX-2.3 and LTX-2.5 (2.5 differs only in 84 keys/block vs 86 and 59 globals vs 58, both of which
// the layout derives rather than assumes).
//
// 🔑 **STAMP ANY TREE YOU MIGHT HOST.** `source_file`/`source_size` anchor a tree to a checkpoint
// on THIS disk; they cannot say which PUBLISHED artifact it came from. For a hosted tree the chain
// back upstream is then inferential — match the byte size by hand and hope. `--source-repo` and
// `--source-revision` record it machine-checkably, which is the whole point of manifest v2.
//
// ⚠️ The CLI accepted neither until 2026-08-21, although `LTXGranuleLayout.write` has always taken
// them — so every tree laid out through this tool was UNSTAMPED, and the 2.5 trees had to be
// re-laid before they could be published. A parameter that exists in the library but not in the
// only tool that calls it is, in practice, a parameter that does not exist.
//
// ⚠️ **A stamp is a claim, so this VERIFIES it before writing** (unless --no-verify): the declared
// revision must resolve on the Hub and the declared file's size there must equal the local file's.
// An unverified provenance stamp is worse than none — it looks authoritative while asserting
// nothing.

import Foundation
import LTX2

let raw = Array(CommandLine.arguments.dropFirst())
func flag(_ name: String) -> String? {
    guard let i = raw.firstIndex(of: name), i + 1 < raw.count else { return nil }
    return raw[i + 1]
}
let positional = { () -> [String] in
    var out: [String] = []; var i = 0
    while i < raw.count {
        if raw[i].hasPrefix("--") { i += raw[i] == "--no-verify" ? 1 : 2; continue }
        out.append(raw[i]); i += 1
    }
    return out
}()

guard positional.count >= 2 else {
    print("""
    usage: ltx-granule-layout <transformer.safetensors> <output-dir> [block-prefix]
                              [--source-repo <id>] [--source-revision <sha|main>] [--no-verify]
    """)
    exit(2)
}
let source = URL(fileURLWithPath: positional[0])
let outDir = URL(fileURLWithPath: positional[1])
let prefix = positional.count >= 3 ? positional[2] : LTXGranuleLayout.ltxBlockPrefix
let sourceRepo = flag("--source-repo")
let sourceRevision = flag("--source-revision")
let verify = !raw.contains("--no-verify")

if sourceRepo == nil {
    print("[ltx-granule-layout] ⚠️ NO --source-repo — this tree will be UNSTAMPED and must not be "
        + "hosted (a downloader could not verify what it came from).")
}

// Verify the stamp against the live Hub before writing a claim we cannot back.
if let repo = sourceRepo, verify {
    let rev = sourceRevision ?? "main"
    let name = source.lastPathComponent
    let localSize = (try? FileManager.default
        .attributesOfItem(atPath: source.path)[.size] as? Int) ?? nil
    let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/\(rev)?recursive=true")!
    let sem = DispatchSemaphore(value: 0)
    var remoteSize: Int?
    var reachable = false
    URLSession.shared.dataTask(with: url) { data, resp, _ in
        defer { sem.signal() }
        reachable = (resp as? HTTPURLResponse)?.statusCode == 200
        guard let data, reachable,
              let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return }
        for e in arr where (e["path"] as? String) == name {
            remoteSize = e["size"] as? Int
        }
    }.resume()
    sem.wait()

    guard reachable else {
        print("[ltx-granule-layout] FAIL ❌ --source-repo \(repo)@\(rev) is not reachable. "
            + "Publish it first, or pass --no-verify to stamp anyway (NOT for a hosted tree).")
        exit(1)
    }
    guard let remoteSize, let localSize, remoteSize == localSize else {
        print("[ltx-granule-layout] FAIL ❌ \(name) is \(localSize.map(String.init) ?? "?") bytes "
            + "locally but \(remoteSize.map(String.init) ?? "absent") at \(repo)@\(rev) — the "
            + "stamp would assert a provenance these bytes do not have.")
        exit(1)
    }
    print("[ltx-granule-layout] provenance VERIFIED: \(repo)@\(rev)/\(name) = \(remoteSize) bytes")
}

do {
    let result = try LTXGranuleLayout.write(
        safetensors: source, outputDir: outDir, blockPrefix: prefix,
        sourceRepo: sourceRepo, sourceRevision: sourceRevision
    ) { done, total in
        print(String(format: "[ltx-granule-layout] block %3d/%d", done, total)); fflush(stdout)
    }
    let m = result.manifest
    print(String(
        format: "[ltx-granule-layout] DONE %d blocks · %d tensors/block · streamed %.2f GiB · "
            + "written %.2f GiB · %.1fs → %@",
        m.blockCount, m.blocks[0].tensors.count,
        Double(m.streamedBytes) / 1_073_741_824,
        Double(result.writtenBytes) / 1_073_741_824,
        result.wallSeconds, result.outputDir.path))
    print("[ltx-granule-layout] globals: \(m.globalKeys.count) keys · stamp: "
        + "\(sourceRepo.map { "\($0)@\(sourceRevision ?? "main")" } ?? "NONE")")
} catch {
    print("[ltx-granule-layout] FAILED: \(error.localizedDescription)")
    exit(1)
}
