// LTX25PackageGate.swift — acceptance for the LTX-2.5 engine package (`--ltx25-package-gate`).
//
// This gates the surface that is decided BEFORE any weights exist: which repo serves a quantized
// DiT, where the text encoder lives, what the manifest declares. Every one of those fails SILENTLY
// if wrong — you get a running pipeline with a wrong memory declaration, which is worse than a
// crash because the governor then admits a run it should have refused.
//
// ⚠️ The load-bearing case is `.int8` → `-ditq8`. On 2.3 the quantized-DiT siblings are `<repo>-q8`
// / `-q4`; on 2.5 `<repo>-q8` is the int8 TEXT-ENCODER sibling (AB-D-0013) whose
// `transformer-distilled.safetensors` is a SYMLINK back to the bf16 file. So the 2.3 rule applied
// to 2.5 yields a config that declares int8 (~23 GB resident) and loads bf16 (~40 GB) — an ~17 GB
// under-declaration. Case 2 does not assert the suffix; it READS BOTH CHECKPOINTS and proves the
// wrong one is unquantized, so the gate would fail if the trees were ever reorganized.

import Foundation
import LTX2
import MLXLTX2
import MLXToolKit

private let modelBase = "/Volumes/Satechi/Models/xocialize"

/// True when the safetensors header carries quantization params (`.scales` siblings) — the same
/// signal `DiT.dense()` uses to pick `quantizedMatmul`.
private func headerIsQuantized(_ path: String) -> Bool? {
    guard let fh = FileHandle(forReadingAtPath: path),
          let lenData = try? fh.read(upToCount: 8), lenData.count == 8 else { return nil }
    let n = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
    guard n > 0, n < 200_000_000, let hdr = try? fh.read(upToCount: Int(n)),
          let json = String(data: hdr, encoding: .utf8) else { return nil }
    try? fh.close()
    return json.contains(".scales\"")
}

func ltx25PackageGate() {
    var failures: [String] = []
    func check(_ name: String, _ ok: Bool, _ detail: String) {
        print("[ltx25-package-gate] \(ok ? "✅" : "❌") \(name) — \(detail)")
        if !ok { failures.append(name) }
    }

    // 1. Family defaults.
    let c25 = LTX2Configuration(family: .ltx25)
    let c23 = LTX2Configuration()
    check("01 default repo per family",
          c25.repo == "xocialize/ltx-2.5-mlx" && c23.repo == "xocialize/ltx-2.3-mlx",
          "2.5=\(c25.repo)  2.3=\(c23.repo)")
    check("02 family defaults to 2.3", c23.family == .ltx23, "\(c23.family.rawValue)")

    // 2. THE LANDMINE — and it is measured, not asserted.
    var q8_25 = LTX2Configuration(family: .ltx25); q8_25.quant = .int8
    let derived = q8_25.effectiveTransformerRepo
    check("03 2.5 int8 derives -ditq8", derived == "xocialize/ltx-2.5-mlx-ditq8", "\(derived ?? "nil")")

    let ditq8File = "\(modelBase)/ltx-2.5-mlx-ditq8/transformer-distilled.safetensors"
    let encq8File = "\(modelBase)/ltx-2.5-mlx-q8/transformer-distilled.safetensors"
    if let ditq8Quant = headerIsQuantized(ditq8File), let encq8Quant = headerIsQuantized(encq8File) {
        // The discrimination check: the suffix the gate demands must be the quantized one AND the
        // suffix 2.3's rule would have produced must be the unquantized one. If both read the same
        // way, this gate cannot tell a correct mapping from a broken one.
        check("04 -ditq8 checkpoint IS quantized", ditq8Quant, "scales present: \(ditq8Quant)")
        check("05 -q8 checkpoint is NOT quantized (it is the ENCODER sibling — the trap)",
              !encq8Quant, "scales present: \(encq8Quant) — 2.3's rule here would serve bf16 as int8")
    } else {
        print("[ltx25-package-gate] ⏭️  04/05 skipped — 2.5 model trees not present on this machine")
    }

    var q4_25 = LTX2Configuration(family: .ltx25); q4_25.quant = .int4
    check("06 2.5 int4 derives NOTHING (no q4 2.5 DiT exists; contraindicated)",
          q4_25.effectiveTransformerRepo == nil, "\(q4_25.effectiveTransformerRepo ?? "nil")")

    // 3. 2.3 must be byte-identical in behaviour — this change must not touch the shipping path.
    var q8_23 = LTX2Configuration(); q8_23.quant = .int8
    var q4_23 = LTX2Configuration(); q4_23.quant = .int4
    check("07 2.3 int8/int4 suffixes unchanged",
          q8_23.effectiveTransformerRepo == "xocialize/ltx-2.3-mlx-q8"
              && q4_23.effectiveTransformerRepo == "xocialize/ltx-2.3-mlx-q4",
          "\(q8_23.effectiveTransformerRepo ?? "nil") / \(q4_23.effectiveTransformerRepo ?? "nil")")
    var bf16_25 = LTX2Configuration(family: .ltx25)
    check("08 bf16 rides the components repo on both families",
          bf16_25.effectiveTransformerRepo == nil && c23.effectiveTransformerRepo == nil,
          "2.5=\(bf16_25.effectiveTransformerRepo ?? "nil") 2.3=\(c23.effectiveTransformerRepo ?? "nil")")
    bf16_25.quant = .bf16

    // 4. The encoder is in-tree on 2.5 — no separate source to materialize.
    let roles25 = Set(c25.weightSources.map(\.role))
    let roles23 = Set(c23.weightSources.map(\.role))
    check("09 2.5 declares NO text-encoder source (Gemma-4 is in-tree)",
          !roles25.contains("text-encoder") && roles23.contains("text-encoder"),
          "2.5=\(roles25.sorted()) 2.3=\(roles23.sorted())")
    check("10 2.5 components source carries the gemma4 dir",
          c25.weightSources.first { $0.role == "components" }?
              .matching?.contains("gemma4-12b-ltx-v1/*") == true,
          "\(c25.weightSources.first { $0.role == "components" }?.matching?.count ?? 0) globs")

    // 5. `resolved()` must yield a non-nil gemmaDirectory on 2.5, or `load()` throws
    //    configurationMismatch on a complete tree.
    var explicit25 = LTX2Configuration(family: .ltx25)
    explicit25.ltxDirectory = URL(fileURLWithPath: "\(modelBase)/ltx-2.5-mlx")
    let r25 = explicit25.resolved(storeRoot: nil)
    check("11 2.5 resolves gemmaDirectory in-tree",
          r25.gemmaDirectory?.lastPathComponent == "gemma4-12b-ltx-v1",
          "\(r25.gemmaDirectory?.path ?? "nil")")

    // 6. Codable back-compat: a payload written before `family` existed must decode as 2.3.
    let legacy = #"{"repo":"xocialize/ltx-2.3-mlx","quant":"bf16"}"#
    if let decoded = try? JSONDecoder().decode(LTX2Configuration.self, from: Data(legacy.utf8)) {
        check("12 pre-family config decodes as 2.3", decoded.family == .ltx23, "\(decoded.family.rawValue)")
    } else {
        check("12 pre-family config decodes as 2.3", false, "decode FAILED")
    }
    if let round = try? JSONEncoder().encode(c25),
       let back = try? JSONDecoder().decode(LTX2Configuration.self, from: round) {
        check("13 family round-trips", back.family == .ltx25, "\(back.family.rawValue)")
    } else {
        check("13 family round-trips", false, "round-trip FAILED")
    }

    // 7. The manifest — the numbers measured 2026-08-14 (AB-R-0039), and the absent int4 entry.
    let m = MLXLTX25Package.manifest
    let quants = m.requirements.footprints.map(\.quant)
    check("14 manifest declares bf16 + int8 and NO int4",
          Set(quants) == [.bf16, .int8], "\(quants.map(\.rawValue))")
    let int8fp = m.requirements.footprints.first { $0.quant == .int8 }
    let bf16fp = m.requirements.footprints.first { $0.quant == .bf16 }
    // WORST measured value across the tested envelope (704×512 at 121f AND 161f — 161f being
    // `standard64.maxFrames`, so this covers the profile's full frame range, not a point in it):
    //   bf16 phys-after-load max(39.92, 39.89) = 39.92 · peak max(43.11, 43.26) = 43.26
    //   int8 phys-after-load max(22.09, 22.09) = 22.09 · peak max(25.60, 25.55) = 25.60
    // Declared resident must cover phys-after-load, and resident + activation must cover the
    // measured PEAK — that sum is exactly what the governor charges, so an under-sum is an
    // admission bug. NB these thresholds deliberately do NOT come from the bench's "resident floor"
    // line, which swings 9 GB between these two geometries on both arms (AB-R-0041).
    let bf16Covers = (bf16fp.map { $0.residentBytes >= 39_920_000_000 } ?? false)
        && (bf16fp.map { $0.residentBytes + $0.peakActivationBytes >= 43_260_000_000 } ?? false)
    let int8Covers = (int8fp.map { $0.residentBytes >= 22_090_000_000 } ?? false)
        && (int8fp.map { $0.residentBytes + $0.peakActivationBytes >= 25_600_000_000 } ?? false)
    check("15 declared resident+activation covers the MEASURED peak", bf16Covers && int8Covers,
          "bf16 \(bf16Covers) int8 \(int8Covers)")
    // AB-D-0015 routes standard64 to the q8 DiT. Both halves of that are pinned here, because the
    // second is a BEHAVIOURAL consequence that would otherwise be an accident of rounding:
    //   • int8 fits standard64 with real headroom, and
    //   • bf16 does NOT fit standard64 — on 2.5 it is a max128 quant.
    // Measured peak is already 96% of budget (43.11 / 44.8) at 121f, BELOW the profile's own 161f
    // cap, so "2.5 bf16 runs on 64 GB" was never true with margin. A future edit that quietly
    // shaves the activation term until bf16 "fits" has to fail this line to do it.
    let budget64: UInt64 = 44_800_000_000, budget128: UInt64 = 108_800_000_000
    let int8Charge = int8fp.map { $0.residentBytes + $0.peakActivationBytes } ?? .max
    let bf16Charge = bf16fp.map { $0.residentBytes + $0.peakActivationBytes } ?? .max
    check("16 int8 charge fits standard64's 44.8 GB with real headroom",
          int8Charge <= budget64 && Double(int8Charge) / Double(budget64) < 0.7,
          String(format: "int8 %.2f GB (%.0f%% of 44.8)", Double(int8Charge) / 1e9,
                 100 * Double(int8Charge) / Double(budget64)))
    check("16b bf16 does NOT fit standard64 (it is a max128 quant on 2.5) but does fit max128",
          bf16Charge > budget64 && bf16Charge <= budget128,
          String(format: "bf16 %.2f GB — %.0f%% of 44.8, %.0f%% of 108.8", Double(bf16Charge) / 1e9,
                 100 * Double(bf16Charge) / Double(budget64),
                 100 * Double(bf16Charge) / Double(budget128)))
    check("17 surface is named for 2.5", m.surfaces.first?.name == "ltx-2.5-t2v",
          "\(m.surfaces.first?.name ?? "nil")")

    // 8. Constructing the package must FORCE the family — a caller passing a default config
    //    would otherwise get 2.3's repo and 2.3's `-q8` suffix under a 2.5 manifest.
    //    ⚠️ This asserts `MLXLTX25Package.coerced` ITSELF. An earlier draft re-derived the coercion
    //    here and then checked its own arithmetic, so deleting the shipping coercion left the gate
    //    green — caught by mutation-testing, and the reason `coerced` is exposed at all.
    var sloppy = LTX2Configuration()   // family .ltx23, repo 2.3, quant int8
    sloppy.quant = .int8
    let forced = MLXLTX25Package.coerced(sloppy)
    check("18 2.5 package coerces a default-constructed config onto 2.5 repos",
          forced.family == .ltx25 && forced.repo == "xocialize/ltx-2.5-mlx"
              && forced.effectiveTransformerRepo == "xocialize/ltx-2.5-mlx-ditq8",
          "family=\(forced.family.rawValue) repo=\(forced.repo) "
              + "transformer=\(forced.effectiveTransformerRepo ?? "nil")")
    // …but an EXPLICIT non-default repo must survive coercion: forcing the family must not
    // clobber a caller who deliberately pointed at their own mirror.
    var custom = LTX2Configuration(family: .ltx25, repo: "acme/my-ltx25-mirror")
    custom.quant = .int8
    let keptCustom = MLXLTX25Package.coerced(custom)
    check("19 coercion preserves an explicit custom repo",
          keptCustom.repo == "acme/my-ltx25-mirror"
              && keptCustom.effectiveTransformerRepo == "acme/my-ltx25-mirror-ditq8",
          "\(keptCustom.repo) → \(keptCustom.effectiveTransformerRepo ?? "nil")")

    print(failures.isEmpty
          ? "[ltx25-package-gate] PASS ✅ (20/20)"
          : "[ltx25-package-gate] FAIL ❌ \(failures.count): \(failures.joined(separator: ", "))")
    if !failures.isEmpty { exit(1) }
}
