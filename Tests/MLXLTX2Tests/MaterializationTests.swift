// MaterializationTests.swift — MLXLTX2 through the engine's MAT gate (BRIDGE M4): the first
// package validated against the generalized auto-materialization contract. Offline — no network.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXLTX2

final class MaterializationTests: XCTestCase {

    /// A temp dir holding the probe files that make an explicit-dir config read as satisfied.
    private func satisfiedDirs() throws -> (ltx: URL, gemma: URL, cleanup: () -> Void) {
        let base = FileManager.default.temporaryDirectory.appending(path: "ltx-mat-\(UUID().uuidString)")
        let ltx = base.appending(path: "ltx"), gemma = base.appending(path: "gemma")
        try FileManager.default.createDirectory(at: ltx, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: gemma, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: ltx.appending(path: "connector.safetensors").path, contents: Data([0]))
        // Since 12b2671 the transformer is its own weight source; a satisfied bf16 tree carries it
        // in the components directory (no quant sibling exists for bf16).
        FileManager.default.createFile(atPath: ltx.appending(path: "transformer-distilled.safetensors").path, contents: Data([0]))
        FileManager.default.createFile(atPath: gemma.appending(path: "config.json").path, contents: Data([0]))
        return (ltx, gemma, { try? FileManager.default.removeItem(at: base) })
    }

    // MARK: - Engine MAT gate

    func testMATGatePassesBF16() throws {
        let (ltx, gemma, cleanup) = try satisfiedDirs()
        defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: LTX2Configuration(),
            satisfiedConfiguration: LTX2Configuration(ltxDirectory: ltx, gemmaDirectory: gemma))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testMATGatePassesQuant() throws {
        let (ltx, gemma, cleanup) = try satisfiedDirs()
        defer { cleanup() }
        // Satisfied quant config additionally needs an existing transformerPath.
        let tx = ltx.appending(path: "tx-q8.safetensors")
        FileManager.default.createFile(atPath: tx.path, contents: Data([0]))
        let report = MaterializationConformance.check(
            freshConfiguration: LTX2Configuration(quant: .int8),
            satisfiedConfiguration: LTX2Configuration(quant: .int8, ltxDirectory: ltx,
                                                      transformerPath: tx, gemmaDirectory: gemma))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Source declaration shape

    /// Since 12b2671 (2026-08-21) the transformer is its OWN source on every quant — bf16 from the
    /// base repo (no sibling), never from the components glob, so an encoder-swapped config cannot
    /// demand the 35 GB bf16 DiT from a `-q8` tree.
    func testBF16DeclaresThreeSourcesWithTheTransformerItsOwn() {
        let sources = LTX2Configuration().weightSources
        XCTAssertEqual(sources.map(\.role), ["components", "text-encoder", "transformer-bf16"])
        XCTAssertFalse(sources[0].matching!.contains("transformer-distilled.safetensors"))
        XCTAssertEqual(sources[1].repo, "mlx-community/gemma-3-12b-it-4bit")
        XCTAssertEqual(sources[2].repo, "xocialize/ltx-2.3-mlx")   // base repo: bf16 has no sibling
        XCTAssertEqual(sources[2].matching, ["transformer-distilled.safetensors"])
    }

    /// The bf16 transformer lives in the components tree; an explicit `ltxDirectory` carrying it
    /// satisfies the transformer source (the store layout resolves the same way with
    /// `transformerPath` nil). Without the file, the source is honestly missing.
    func testExplicitBF16DirectorySatisfiesTheTransformerSource() throws {
        let (ltx, gemma, cleanup) = try satisfiedDirs()
        defer { cleanup() }
        let cfg = LTX2Configuration(ltxDirectory: ltx, gemmaDirectory: gemma)
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: nil).isEmpty)
        try FileManager.default.removeItem(at: ltx.appending(path: "transformer-distilled.safetensors"))
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: nil).map(\.role), ["transformer-bf16"])
    }

    func testQuantDeclaresDerivedTransformerRepo() {
        let sources = LTX2Configuration(quant: .int4).weightSources
        XCTAssertEqual(sources.map(\.role), ["components", "text-encoder", "transformer-int4"])
        XCTAssertEqual(sources[2].repo, "xocialize/ltx-2.3-mlx-q4")
        // The quant config's components glob must EXCLUDE the 35 GB bf16 transformer.
        XCTAssertFalse(sources[0].matching!.contains("transformer-distilled.safetensors"))
        // Explicit override wins.
        let custom = LTX2Configuration(transformerRepo: "org/custom-q4", quant: .int4).weightSources
        XCTAssertEqual(custom[2].repo, "org/custom-q4")
    }

    // MARK: - Store-layout probe + resolution

    func testStoreLayoutSatisfiesAndResolves() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "ltx-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = LTX2Configuration()   // bf16: components + gemma + the transformer (its own source)
        // Empty store: everything missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 3)
        // Populate the expected layout — paths from ModelStore so the fixture tracks the
        // engine's canonical models--org--name layout (contract 1.22.0), not a stale literal.
        let store = ModelStore(root: root)
        let ltxDir = store.directory(for: "xocialize/ltx-2.3-mlx")!
        let gemmaDir = store.directory(for: "mlx-community/gemma-3-12b-it-4bit")!
        try FileManager.default.createDirectory(at: ltxDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: gemmaDir, withIntermediateDirectories: true)
        for f in LTX2Configuration.componentFiles + LTX2Configuration.componentFiles23
                + ["transformer-distilled.safetensors"] {
            FileManager.default.createFile(atPath: ltxDir.appending(path: f).path, contents: Data([0]))
        }
        FileManager.default.createFile(atPath: gemmaDir.appending(path: "config.json").path, contents: Data([0]))
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        // Resolution lands on the store layout.
        let resolved = cfg.resolved(storeRoot: root)
        XCTAssertEqual(resolved.ltxDirectory?.path, ltxDir.path)
        XCTAssertEqual(resolved.gemmaDirectory?.path, gemmaDir.path)
        XCTAssertNil(resolved.transformerPath)   // bf16 rides ltxDirectory
        // Quant resolution derives the transformer path from the quant repo.
        let q8 = LTX2Configuration(quant: .int8).resolved(storeRoot: root)
        XCTAssertEqual(q8.transformerPath?.path,
                       store.directory(for: "xocialize/ltx-2.3-mlx-q8")!
                           .appending(path: "transformer-distilled.safetensors").path)
    }

    func testPrewarmPathsUseResolvedStoreLayout() {
        let root = URL(fileURLWithPath: "/tmp/some-store")
        let cfg = LTX2Configuration(modelsRootDirectory: root)
        // Nil dirs + store root ⇒ prewarm targets the resolved store layout (files may not exist
        // yet on a true first run — the prewarmer is best-effort).
        let store = ModelStore(root: root)
        let paths = cfg.prewarmPaths.map(\.path)
        XCTAssertTrue(paths.contains(store.directory(for: "xocialize/ltx-2.3-mlx")!.path))
        XCTAssertTrue(paths.contains(store.directory(for: "mlx-community/gemma-3-12b-it-4bit")!.path))
    }

    func testCodableRoundTripCarriesRepos() throws {
        let cfg = LTX2Configuration(transformerRepo: "org/custom-q8", quant: .int8)
        let decoded = try JSONDecoder().decode(LTX2Configuration.self,
                                               from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.gemmaRepo, cfg.gemmaRepo)
        XCTAssertEqual(decoded.transformerRepo, "org/custom-q8")
        XCTAssertEqual(decoded.quant, .int8)
    }
}
