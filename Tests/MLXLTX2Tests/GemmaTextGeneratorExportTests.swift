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
    ///
    /// The trailing `UInt64?` is the enhancement seed (AB-R-0075). It is pinned HERE because an
    /// unapplied method reference ignores default arguments, so this line is a real compile-time
    /// assertion that the seed crosses the `@_exported` boundary — an app consumer importing only
    /// MLXLTX2 can pin its enhancement.
    @Test func membersVisibleThroughSingleImport() {
        let gen = GemmaTextGenerator(gemmaDirectory: URL(fileURLWithPath: "/nonexistent"))
        let _: (String, String, Int, Float, UInt64?) async throws -> String = gen.generate
    }

    /// The default must stay `nil` — the documented decision is that unseeded callers keep the
    /// historical entropy-seeded behaviour, so the fix cannot silently change what anyone
    /// generates. Checked by calling through the seam's defaults: only a call with NO seed
    /// argument type-checks against a 4-argument application.
    @Test func seedIsOptionalAndDefaulted() {
        let gen = GemmaTextGenerator(gemmaDirectory: URL(fileURLWithPath: "/nonexistent"))
        let unseeded: () async throws -> String = {
            try await gen.generate(system: "s", user: "u", maxTokens: 8, temperature: 0.7)
        }
        _ = unseeded   // never invoked — no weights, no load; the type-check is the assertion
    }
}
