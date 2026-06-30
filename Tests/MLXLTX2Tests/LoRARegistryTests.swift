import Foundation
import Testing
@testable import MLXLTX2

/// Offline L2 gate: the bundled LoRA registry decodes via `Bundle.module`, entries resolve, and the
/// cache builds the expected local path. No MLX kernels / no network — runs under `swift test`.
struct LoRARegistryTests {
    @Test func bundledManifestDecodes() throws {
        let reg = try LoRARegistry.bundled()
        #expect(reg.schemaVersion == 1)
        #expect(reg.base.contains("LTX-2.3"))
        #expect(!reg.adapters.isEmpty)
    }

    @Test func transitionEntryResolves() throws {
        let reg = try LoRARegistry.bundled()
        let e = try #require(reg.entry(id: "transition"))
        #expect(e.repo == "joyfox/LTX-2.3-Transition-LORA")
        #expect(e.weightFile == "ltx2.3-transition.safetensors")
        #expect(e.defaultStrength == 1.0)
        #expect(e.trigger == "zhuanchang")
    }

    @Test func omnicineEntryResolves() throws {
        let reg = try LoRARegistry.bundled()
        let e = try #require(reg.entry(id: "omnicine"))
        #expect(e.repo == "WarmBloodAban/Singularity-LTX-2.3_OmniCine_V1")
        #expect(e.weightFile.hasSuffix(".safetensors"))
    }

    @Test func i2vAdapterEntryResolves() throws {
        let reg = try LoRARegistry.bundled()
        let e = try #require(reg.entry(id: "i2v-adapter"))
        #expect(e.repo == "MachineDelusions/LTX-2_Image2Video_Adapter_LoRa")
        #expect(e.weightFile.hasSuffix(".safetensors"))
        #expect(e.inputKind == .image)
    }

    @Test func plainEntriesDefaultToNoInput() throws {
        let reg = try LoRARegistry.bundled()
        #expect(reg.entry(id: "transition")?.inputKind == LoRAInputKind.none)
        #expect(reg.entry(id: "omnicine")?.inputKind == LoRAInputKind.none)
    }

    @Test func unknownEntryIsNil() throws {
        let reg = try LoRARegistry.bundled()
        #expect(reg.entry(id: "does-not-exist") == nil)
    }

    @Test func cacheLocalURLIsIdNamed() throws {
        let reg = try LoRARegistry.bundled()
        let e = try #require(reg.entry(id: "transition"))
        let cache = LoRACache(directory: URL(fileURLWithPath: "/tmp/ltx-lora-cache"))
        #expect(cache.localURL(for: e).lastPathComponent == "transition.safetensors")
    }
}
