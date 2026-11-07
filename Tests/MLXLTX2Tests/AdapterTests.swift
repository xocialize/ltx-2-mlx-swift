// AdapterTests.swift — LTX on the engine's typed adapters (contract 1.53.0 + 1.55.0's conditioning
// groups, AB-A-0197): the ADP-1..4 gate over both manifests and the bundled format-3 registry, the
// declarations, the qualified catalog, Ingredients' exactlyOne group, slot order, what the package
// itself refuses, and the sheet the subject_images alternative builds. Offline: no weights.

import CoreGraphics
import Foundation
import ImageIO
import MLXServeConformance
import MLXToolKit
import UniformTypeIdentifiers
import XCTest

@testable import MLXLTX2

final class AdapterTests: XCTestCase {

    // MARK: ADP gate

    func testADPGate23() throws {
        let report = AdapterConformance.check(manifest: MLXLTX2Package.manifest,
                                              configuration: LTX2Configuration(),
                                              catalog: try LTXAdapters.catalog(for: .ltx23))
        XCTAssertTrue(report.passed, report.summary)
        print(report.summary)
    }

    func testADPGate25() throws {
        let report = AdapterConformance.check(manifest: MLXLTX25Package.manifest,
                                              configuration: LTX2Configuration(family: .ltx25),
                                              catalog: try LTXAdapters.catalog(for: .ltx25))
        XCTAssertTrue(report.passed, report.summary)
        print(report.summary)
    }

    /// The family filter is load-bearing: the whole registry against 2.5 fails ADP-4 on exactly
    /// the adapter cleared only on 2.3 (Cameraman v2). So the gate can fail, and the per-family
    /// catalog is what a 2.5 host registers.
    func testWholeRegistryAgainst25FailsOnlyOn23OnlyAdapters() throws {
        let report = AdapterConformance.check(manifest: MLXLTX25Package.manifest,
                                              catalog: try LTXAdapters.catalog())
        let failed = report.checks.filter { !$0.passed }.map(\.name)
        XCTAssertEqual(failed, ["ADP-4 catalog (ltx:cameraman-v2)"], report.summary)
    }

    // MARK: Declarations

    func testDeclarations() throws {
        for (manifest, family) in [(MLXLTX2Package.manifest, LTXFamily.ltx23),
                                   (MLXLTX25Package.manifest, .ltx25)] {
            for surface in manifest.surfaces {
                guard surface.capability == .textToVideo else {
                    // retake/extend/a2v run no adapters, so the vedit surface declares none.
                    XCTAssertNil(surface.adapters, surface.name)
                    continue
                }
                let support = try XCTUnwrap(surface.adapters, surface.name)
                XCTAssertEqual(support.baseModel, LTXAdapters.baseModel(for: family))
                XCTAssertEqual(support.formats, [.diffusersPEFT])
                XCTAssertEqual(Set(support.kinds), [.plain, .referenceConditioned])
                XCTAssertEqual(support.binding, .perRequest)
                XCTAssertEqual(support.maxCount, 1)
                XCTAssertNil(support.quants)
            }
        }
        XCTAssertEqual(LTXAdapters.baseModel(for: .ltx23), "ltx-2.3")
        XCTAssertEqual(LTXAdapters.baseModel(for: .ltx25), "ltx-2.5")
    }

    // MARK: Catalog

    func testBundledRegistryIsFormat3() throws {
        let registry = try LTXAdapterRegistry.bundled()
        XCTAssertEqual(registry.schemaVersion, 3)
        XCTAssertEqual(registry.namespace, "ltx")
        let lipdub = try XCTUnwrap(registry.entry(id: "ltx:lipdub"))
        XCTAssertEqual(lipdub.runtime.audioReference, true)
        XCTAssertEqual(lipdub.slot("reference_video")?.ingest, "videoClip")
        XCTAssertEqual(registry.entry(id: "union-control")?.runtime.referenceDownscale, 2)
        XCTAssertEqual(registry.entry(id: "ingredients")?.slot("subject_images")?.ingest, "sheetBuilder")
        XCTAssertNil(registry.entry(id: "does-not-exist"))
    }

    func testCatalogIsQualifiedAndPinned() throws {
        let all = try LTXAdapters.catalog()
        XCTAssertEqual(all.count, try LTXAdapterRegistry.bundled().adapters.count)
        XCTAssertTrue(all.allSatisfy { $0.id.hasPrefix("ltx:") })
        XCTAssertTrue(all.allSatisfy { $0.validationProblems.isEmpty }, "\(all.map(\.validationProblems))")
        XCTAssertTrue(all.allSatisfy(\.source.isPinned))
        XCTAssertEqual(try LTXAdapters.catalog(for: .ltx23).count, all.count)
        XCTAssertEqual(try LTXAdapters.catalog(for: .ltx25).map(\.id).contains("ltx:cameraman-v2"), false)
        // IC adapters carry their measured footprint (AB-R-0600, registry v0.3.3): the multiplier
        // the engine charges on top of the base activation.
        for ic in all where ic.isReferenceConditioned {
            let footprint = try XCTUnwrap(ic.footprint, ic.id)
            XCTAssertGreaterThan(footprint.activationMultiplier, 1.3, ic.id)
            XCTAssertLessThan(footprint.activationMultiplier, 1.7, ic.id)
            XCTAssertEqual(footprint.residentBytes, ic.source.bytes, ic.id)
        }
        // Qualifying is idempotent.
        let one = try XCTUnwrap(all.first)
        XCTAssertEqual(LTXAdapters.qualified(one, namespace: "ltx").id, one.id)
    }

    // MARK: Pre-flight (the engine's checks, run the way it runs them)

    private func plan(_ refs: [AdapterRef], family: LTXFamily = .ltx25) throws -> [AdapterSupport.Plan] {
        let catalog = try LTXAdapters.catalog(for: family)
        return try LTXAdapters.support(for: family).plan(refs, capability: .textToVideo, quant: .int8) { id in
            catalog.first { $0.id == id }
        }
    }

    private let png = ConditioningMedia.image(Image(format: .png, data: Data([1])))

    func testIngredientsGroupIsExactlyOne() throws {
        func ingredients(_ roles: [String]) -> [AdapterRef] {
            [AdapterRef(id: "ltx:ingredients",
                        conditioning: roles.map { ConditioningInput(role: $0, media: png) })]
        }
        // The two alternatives pass.
        XCTAssertNoThrow(try plan(ingredients(["reference_sheet"])))
        XCTAssertNoThrow(try plan(ingredients(["subject_images"])))
        XCTAssertNoThrow(try plan(ingredients(["subject_images", "subject_images", "location_image"])))
        // Nothing bound, both alternatives, or a location alone are refused before admission.
        for roles in [[], ["reference_sheet", "subject_images"], ["location_image"],
                      ["reference_sheet", "location_image"]] {
            XCTAssertThrowsError(try plan(ingredients(roles)), "\(roles)") { error in
                guard case PackageError.unsupportedRequestFeature = error else {
                    return XCTFail("\(roles): \(error)")
                }
            }
        }
    }

    func testConditioningArrivesInSlotOrder() throws {
        let clip = ConditioningMedia.video(Video(format: .mp4, data: Data([2])))
        let wav = ConditioningMedia.audio(Audio(data: Data([3])))
        let ref = AdapterRef(id: "ltx:lipdub", conditioning: [
            ConditioningInput(role: "dub_audio", media: wav),
            ConditioningInput(role: "reference_video", media: clip),
        ])
        let planned = try XCTUnwrap(try plan([ref]).first)
        XCTAssertEqual(planned.conditioning.map(\.role), ["reference_video", "dub_audio"])
        XCTAssertEqual(planned.strength, 1.0)
    }

    func testSurfaceRefusals() throws {
        // A 2.3-only adapter on 2.5, two adapters, a strength outside 0…2, an unknown id.
        XCTAssertThrowsError(try plan([AdapterRef(id: "ltx:cameraman-v2", conditioning: [
            ConditioningInput(role: "reference_video", media: .video(Video(format: .mp4, data: Data())))])]))
        XCTAssertThrowsError(try plan([AdapterRef(id: "ltx:transition"), AdapterRef(id: "ltx:omnicine")]))
        XCTAssertThrowsError(try plan([AdapterRef(id: "ltx:transition", strength: 2.5)]))
        XCTAssertThrowsError(try plan([AdapterRef(id: "transition")]))
        // Cameraman v2 still fits 2.3.
        XCTAssertNoThrow(try plan([AdapterRef(id: "ltx:cameraman-v2", conditioning: [
            ConditioningInput(role: "reference_video", media: .video(Video(format: .mp4, data: Data())))])],
            family: .ltx23))
    }

    // MARK: What the package checks itself

    private func resolved(_ id: String, format: AdapterFormat? = nil, strength: Double = 1,
                          conditioning: [ConditioningInput] = []) throws -> ResolvedAdapter {
        let d = try XCTUnwrap(try LTXAdapters.catalog().first { $0.id == id })
        return ResolvedAdapter(id: d.id, kind: d.kind, format: format ?? d.format,
                               fileURL: URL(fileURLWithPath: "/dev/null"), source: d.source,
                               strength: strength, conditioning: conditioning,
                               footprint: d.resolvedFootprint)
    }

    func testAdapterRunSplit() throws {
        let registry = try LTXAdapterRegistry.bundled()
        let plain = try LTXAdapterRun([try resolved("ltx:transition", strength: 0.8)], registry: registry)
        XCTAssertEqual(plain.loras.map(\.strength), [0.8])
        XCTAssertNil(plain.ic)
        XCTAssertEqual(plain.key.count, 1)

        let sheet = ConditioningInput(role: "reference_sheet", media: png)
        let ic = try LTXAdapterRun([try resolved("ltx:ingredients", strength: 1.4, conditioning: [sheet])],
                                   registry: registry)
        XCTAssertEqual(ic.ic?.adapter.id, "ltx:ingredients")
        XCTAssertEqual(ic.ic?.entry?.descriptor.id, "ingredients")
        XCTAssertEqual(ic.loras.count, 1)   // the IC adapter's weights are applied too

        // A strength change is a different set; the same set is the same key.
        XCTAssertNotEqual(plain.key, try LTXAdapterRun([try resolved("ltx:transition")], registry: registry).key)
        XCTAssertEqual(plain.key, try LTXAdapterRun([try resolved("ltx:transition", strength: 0.8)],
                                                    registry: registry).key)
        XCTAssertTrue(LTXAdapterRun.empty.loras.isEmpty)
    }

    func testPackageRefusesWhatOnlyItKnows() throws {
        let registry = try LTXAdapterRegistry.bundled()
        // A dialect the DiT's low-rank ADD is not verified on.
        XCTAssertThrowsError(try LTXAdapterRun([try resolved("ltx:transition", format: .kohya)],
                                               registry: registry))
        // Two IC adapters (only reachable when a host stamps resolvedAdapters past maxCount).
        let clip = ConditioningInput(role: "reference_video", media: .video(Video(format: .mp4, data: Data())))
        XCTAssertThrowsError(try LTXAdapterRun([try resolved("ltx:lipdub", conditioning: [clip]),
                                                try resolved("ltx:cameraman-v2", conditioning: [clip])],
                                               registry: registry))
    }

    /// A host may register a descriptor under its own id; the pinned bytes still find the entry.
    func testEntryFallsBackToContent() throws {
        let registry = try LTXAdapterRegistry.bundled()
        let d = try XCTUnwrap(try LTXAdapters.catalog().first { $0.id == "ltx:union-control" })
        let renamed = ResolvedAdapter(id: "host:canny", kind: d.kind, format: d.format,
                                      fileURL: URL(fileURLWithPath: "/dev/null"), source: d.source,
                                      strength: 1, footprint: d.resolvedFootprint)
        XCTAssertEqual(registry.entry(for: renamed)?.runtime.referenceDownscale, 2)
    }

    // MARK: Ingredients sheet (the subject_images alternative)

    private func solidPNG(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, size: Int = 64) -> Image {
        let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: r, green: g, blue: b, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        let out = NSMutableData()
        let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
        CGImageDestinationFinalize(dest)
        return Image(format: .png, data: out as Data)
    }

    /// Subjects in a top row, the location in a band below, black around both — read back through
    /// the same still ingest the run uses (row 0 = top).
    func testSheetLayout() throws {
        let sheet = try IngredientsSheet.compose(subjects: [solidPNG(1, 0, 0)], location: solidPNG(0, 0, 1))
        XCTAssertEqual(sheet.width, 1456)
        XCTAssertEqual(sheet.height, 825)
        let t = try ImageInput.referenceStillFrames(sheet, width: 91, height: 52, frames: 1).asType(.float32)
        func rgb(_ y: Int, _ x: Int) -> [Float] { (0..<3).map { t[0, $0, 0, y, x].item(Float.self) } }
        XCTAssertEqual(rgb(15, 45).map { $0 > 0.9 }, [true, false, false], "subject (red) on top")
        XCTAssertEqual(rgb(41, 45).map { $0 > 0.9 }, [false, false, true], "location (blue) below")
        XCTAssertTrue(rgb(15, 2).allSatisfy { $0 < -0.9 }, "black beside the aspect-fit subject")
        XCTAssertThrowsError(try IngredientsSheet.compose(subjects: [], location: nil))
    }
}
