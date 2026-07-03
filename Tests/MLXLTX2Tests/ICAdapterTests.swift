// ICAdapterTests.swift — offline checks for the IC intake (IC-LORA-PLAN P3/P4): the synced v2
// registry decodes with the wrapper's runtime fields, IC entries are distinguishable from plain
// ones, and the metaData keys are stable strings (they're an app-facing contract).

import Foundation
import LTX2
import XCTest
@testable import MLXLTX2

final class ICAdapterTests: XCTestCase {

    func testBundledRegistryCarriesICEntries() throws {
        let reg = try LoRARegistry.bundled()
        XCTAssertEqual(reg.schemaVersion, 2)
        // Plain entries unchanged.
        let plain = try XCTUnwrap(reg.entry(id: "transition"))
        XCTAssertFalse(plain.isIC)
        // Ingredients: ic kind, downscale 1, one-stage policy, community license (not gated).
        let ing = try XCTUnwrap(reg.entry(id: "ingredients"))
        XCTAssertTrue(ing.isIC)
        XCTAssertEqual(ing.referenceDownscale, 1)
        XCTAssertEqual(ing.stage2, "skip")
        XCTAssertEqual(ing.defaultStrength, 1.4, accuracy: 0.001)
        XCTAssertNotEqual(ing.licenseGated, true)
        // Cameraman: research-only ⇒ licenseGated.
        let cam = try XCTUnwrap(reg.entry(id: "cameraman-v2"))
        XCTAssertTrue(cam.isIC)
        XCTAssertEqual(cam.licenseGated, true)
    }

    func testICMetaKeysAreStable() {
        // App-facing contract (the interim transport until the engine ConditioningInput lands).
        XCTAssertEqual(ICMetaKeys.adapterId, "ic.adapterId")
        XCTAssertEqual(ICMetaKeys.adapterStrength, "ic.adapterStrength")
        XCTAssertEqual(ICMetaKeys.referencePath, "ic.referencePath")
        XCTAssertEqual(ICMetaKeys.referenceStrength, "ic.referenceStrength")
    }

    func testSnapFramesMatchesOracle() {
        XCTAssertEqual(ReferenceConditioning.snapFrames(121), 121)
        XCTAssertEqual(ReferenceConditioning.snapFrames(120), 113)
        XCTAssertEqual(ReferenceConditioning.snapFrames(1), 9)
    }
}
