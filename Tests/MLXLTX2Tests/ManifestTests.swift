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

    @Test func licenseIsLTX2CommunityBothLayers() {
        let lic = MLXLTX2Package.manifest.license
        #expect(lic.weightLicense == SPDXLicense("LicenseRef-LTX-2-Community"))
        #expect(lic.portCodeLicense == SPDXLicense("LicenseRef-LTX-2-Community"))
    }

    /// Eval-only by design: the non-permissive LTX-2 Community License is REJECTED by the
    /// default `.permissiveOnly` gate (host relaxes it explicitly for capability evaluation).
    @Test func permissiveGateRejectsByDesign() {
        let result = LicensePolicy.permissiveOnly.evaluate(MLXLTX2Package.manifest.license)
        #expect(!result.isAdmitted)
        #expect(result == .rejectedWeight(SPDXLicense("LicenseRef-LTX-2-Community")))
    }

    @Test func requirementsAreDeclared() {
        let r = MLXLTX2Package.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.footprints.contains { $0.quant == .bf16 })
    }

    @Test func registrationBuilds() {
        #expect(MLXLTX2Package.registration.manifest.capabilities.contains(.textToVideo))
    }
}
