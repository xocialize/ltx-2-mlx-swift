import Foundation
import MLXToolKit
import Testing
@testable import MLXLTX2

/// Offline conformance checks for the MLXLTX2 ModelPackage manifest — no MLX
/// kernels run, so these execute under `swift test`.
struct ManifestTests {
    @Test func declaresTextToVideo() {
        let m = MLXLTX2Package.manifest
        #expect(m.capabilities.contains(.textToVideo))
        #expect(m.surfaces.count == 1)
        #expect(m.surfaces[0].capability == .textToVideo)
    }

    @Test func contractVersionMatches() {
        #expect(MLXLTX2Package.manifest.contractVersion == ContractVersion.current)
    }

    /// Contract 1.40.0 — a2v is DECLARED on the 2.5 t2v surface (so the engine admits
    /// `T2VRequest.initAudio`) and NOT on 2.3, which never shipped a2v and must be refused rather
    /// than silently run a plain t2v (AB-A-0023 / AB-A-0066: gated, our own verdict).
    @Test func initAudioIsDeclaredOn25Only() {
        let s25 = MLXLTX25Package.manifest.surfaces.first { $0.capability == .textToVideo }
        #expect(s25?.t2vControls?.supportsInitAudio == true)
        #expect(s25?.parameters.contains { $0.name == "initAudio" } == true)
        #expect(s25?.controlsMatchCapability == true)
        let s23 = MLXLTX2Package.manifest.surfaces.first { $0.capability == .textToVideo }
        #expect(s23?.t2vControls == nil)
        #expect(s23?.parameters.contains { $0.name == "initAudio" } == false)
    }

    /// Two-layer declaration: weights = LTX-2 Community License; port code = Apache-2.0
    /// (our own implementation, mirroring Lightricks' Apache-2.0 inference code).
    @Test func licenseDeclaresCommunityWeightsApachePortCode() {
        let lic = MLXLTX2Package.manifest.license
        #expect(lic.weightLicense == .ltx2Community)
        #expect(lic.portCodeLicense == .apache2)
    }

    /// As of engine 0.6.0 LTX-2-Community is on the `permissiveAllowlist`, so BOTH layers
    /// clear the default `.permissiveOnly` gate — no eval-acknowledged relaxation needed.
    @Test func permissiveGateAdmits() {
        let result = LicensePolicy.permissiveOnly.evaluate(MLXLTX2Package.manifest.license)
        #expect(result.isAdmitted)
    }

    @Test func requirementsAreDeclared() {
        let r = MLXLTX2Package.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.footprints.contains { $0.quant == .bf16 })
    }

    @Test func registrationBuilds() {
        #expect(MLXLTX2Package.registration.manifest.capabilities.contains(.textToVideo))
    }

    /// bf16 (no `transformerPath`): prewarm the whole LTX dir + Gemma as-is.
    @Test func prewarmPathsBf16PagesWholeDirs() {
        let ltx = URL(fileURLWithPath: "/tmp/ltx"), gemma = URL(fileURLWithPath: "/tmp/gemma")
        let cfg = LTX2Configuration(ltxDirectory: ltx, gemmaDirectory: gemma)
        #expect(cfg.prewarmPaths == [ltx, gemma])
    }

    /// q8 (`transformerPath` set): the unused bf16 `transformer-distilled.safetensors` in
    /// `ltxDirectory` must be EXCLUDED; the q8 transformer + the other LTX weights + Gemma included.
    @Test func prewarmPathsQuantExcludesBf16Transformer() throws {
        let fm = FileManager.default
        let ltxDir = fm.temporaryDirectory.appendingPathComponent("ltx-prewarm-\(getpid())")
        try? fm.removeItem(at: ltxDir)
        try fm.createDirectory(at: ltxDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: ltxDir) }
        let bf16 = ltxDir.appendingPathComponent("transformer-distilled.safetensors")
        let connector = ltxDir.appendingPathComponent("connector.safetensors")
        for f in [bf16, connector] { try Data().write(to: f) }

        let q8 = URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/models/dgrauet/ltx-2.3-mlx-q8/transformer-distilled.safetensors")
        let gemma = URL(fileURLWithPath: "/tmp/gemma")
        let cfg = LTX2Configuration(ltxDirectory: ltxDir, transformerPath: q8, gemmaDirectory: gemma)
        let paths = cfg.prewarmPaths.map(\.standardizedFileURL)

        #expect(!paths.contains(bf16.standardizedFileURL))         // bf16 transformer excluded
        #expect(paths.contains(connector.standardizedFileURL))     // other LTX weights kept
        #expect(paths.contains(q8.standardizedFileURL))            // q8 override paged
        #expect(paths.contains(gemma.standardizedFileURL))         // Gemma paged
    }
}
