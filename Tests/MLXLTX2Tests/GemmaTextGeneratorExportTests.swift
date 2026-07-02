import Foundation
import Testing
import MLXLTX2
// Deliberately NOT `import LTX2`: this file is the compile-time proof that the BRIDGE-LTX-004
// re-export gives app consumers full member/initializer access through `import MLXLTX2` alone
// (the original typealias exposed only the type NAME across the module boundary — the Xcode
// agent had to add `import LTX2` to reach the init; the scoped `@_exported import` fixes that).

@Suite struct GemmaTextGeneratorExportTests {
    /// Init + `generate` signature must resolve with only MLXLTX2 imported. No model load —
    /// the closure is never run; type-checking this file IS the test.
    @Test func membersVisibleThroughSingleImport() {
        let gen = GemmaTextGenerator(gemmaDirectory: URL(fileURLWithPath: "/nonexistent"))
        let _: (String, String, Int, Float) async throws -> String = gen.generate
    }
}
