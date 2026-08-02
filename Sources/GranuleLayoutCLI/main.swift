// ltx-granule-layout — lay an LTX-2.3 transformer checkpoint out as per-block
// granule files for LTXBlockStreamer (STREAMING-PLAN.md; sibling of wan-core's
// wan-granule-layout).
//
//   ltx-granule-layout <transformer.safetensors> <output-dir> [block-prefix]
//
// The default block prefix is the LTX-2.3 on-disk dialect
// (`transformer.transformer_blocks.`). The copy is quantization-agnostic bytes,
// F_NOCACHE both ends; the manifest records provenance (source name + size) so
// a stale tree is refused at bind.

import Foundation
import LTX2

let args = CommandLine.arguments.dropFirst()
guard args.count >= 2 else {
    print("usage: ltx-granule-layout <transformer.safetensors> <output-dir> [block-prefix]")
    exit(2)
}
let source = URL(fileURLWithPath: Array(args)[0])
let outDir = URL(fileURLWithPath: Array(args)[1])
let prefix = args.count >= 3 ? Array(args)[2] : LTXGranuleLayout.ltxBlockPrefix

do {
    let result = try LTXGranuleLayout.write(
        safetensors: source, outputDir: outDir, blockPrefix: prefix
    ) { done, total in
        print(String(format: "[ltx-granule-layout] block %3d/%d", done, total))
        fflush(stdout)
    }
    let m = result.manifest
    print(String(
        format: "[ltx-granule-layout] DONE %d blocks · %d tensors/block · "
            + "streamed %.2f GiB · written %.2f GiB · %.1fs → %@",
        m.blockCount, m.blocks[0].tensors.count,
        Double(m.streamedBytes) / 1_073_741_824,
        Double(result.writtenBytes) / 1_073_741_824,
        result.wallSeconds, result.outputDir.path))
    print("[ltx-granule-layout] globals: \(m.globalKeys.count) keys stay resident")
} catch {
    print("[ltx-granule-layout] FAILED: \(error.localizedDescription)")
    exit(1)
}
