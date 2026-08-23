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
          c25.repo == "mlx-community/ltx-2.5-mlx" && c23.repo == "xocialize/ltx-2.3-mlx",
          "2.5=\(c25.repo)  2.3=\(c23.repo)")
    check("02 family defaults to 2.3", c23.family == .ltx23, "\(c23.family.rawValue)")

    // 2. THE LANDMINE — and it is measured, not asserted.
    var q8_25 = LTX2Configuration(family: .ltx25); q8_25.quant = .int8
    let derived = q8_25.effectiveTransformerRepo
    check("03 2.5 int8 derives -ditq8", derived == "mlx-community/ltx-2.5-mlx-ditq8", "\(derived ?? "nil")")

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
          forced.family == .ltx25 && forced.repo == "mlx-community/ltx-2.5-mlx"
              && forced.effectiveTransformerRepo == "mlx-community/ltx-2.5-mlx-ditq8",
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

    // ───── the TEXT-ENCODER axis (2026-08-19) ─────
    // 🔑 The whole point of these cases is that `quant` and `textEncoderQuant` are DIFFERENT axes
    // resolving to DIFFERENT sibling repos with different suffixes. Conflating them is the
    // documented 2.5 footgun, and it is silent: both produce a plausible repo name.
    var enc = LTX2Configuration(family: .ltx25, repo: "mlx-community/ltx-2.5-mlx")
    enc.quant = .int8
    enc.textEncoderQuant = .int8
    check("20 int8 DiT and int8 ENCODER resolve to DIFFERENT siblings",
          enc.effectiveTransformerRepo == "mlx-community/ltx-2.5-mlx-ditq8"
              && enc.effectiveComponentsRepo == "mlx-community/ltx-2.5-mlx-q8",
          "transformer=\(enc.effectiveTransformerRepo ?? "nil") components=\(enc.effectiveComponentsRepo)")

    // No profile ⇒ nothing to follow ⇒ bf16, the pre-existing behaviour.
    var bf = LTX2Configuration(family: .ltx25, repo: "mlx-community/ltx-2.5-mlx")
    check("21 no profile ⇒ encoder resolves bf16 and rides the components repo unchanged",
          bf.textEncoderQuant == nil && bf.effectiveTextEncoderQuant == .bf16
              && bf.effectiveComponentsRepo == "mlx-community/ltx-2.5-mlx",
          "override=\(bf.textEncoderQuant?.rawValue ?? "nil") → "
              + "\(bf.effectiveTextEncoderQuant.rawValue) → \(bf.effectiveComponentsRepo)")

    // int4 encoder was REJECTED by the gate (AB-D-0010, 0.996728 vs a 0.999879 bf16 floor).
    // Deriving `-q4` would name a tree that must never be built, let alone loaded.
    bf.textEncoderQuant = .int4
    check("22 int4 encoder derives NO sibling (rejected quant, must not be nameable)",
          bf.effectiveComponentsRepo == "mlx-community/ltx-2.5-mlx",
          "\(bf.effectiveComponentsRepo)")

    // 2.3's encoder is an EXTERNAL repo, so a components sibling would carry no encoder at all.
    var enc23 = LTX2Configuration(family: .ltx23, repo: "xocialize/ltx-2.3-mlx")
    enc23.textEncoderQuant = .int8
    check("23 2.3 never derives an encoder sibling (its encoder is an external repo)",
          enc23.effectiveComponentsRepo == "xocialize/ltx-2.3-mlx",
          "\(enc23.effectiveComponentsRepo)")

    // The declaration has to REACH weightSources, not just the computed property — the wiring is
    // what ships, and a gate on the property alone would pass while materialization used `repo`.
    let compSource = enc.weightSources.first { $0.role == "components" }
    check("24 weightSources materializes the ENCODER sibling, not the base repo",
          compSource?.repo == "mlx-community/ltx-2.5-mlx-q8",
          "components source repo=\(compSource?.repo ?? "nil")")

    // ───── gate policy + Codable back-compat ─────
    check("25 gate policy defaults to .auto (no silent behaviour change)",
          bf.streamingOptions.gatePolicy == .auto, "\(bf.streamingOptions.gatePolicy.rawValue)")
    // ⟲ The GATE half moved 2026-08-23 (every tier pins now); the ENCODER half did NOT and is
    // still the load-bearing distinction — int8 on the low tiers, bf16 on the high ones.
    check("26 profiles ADVISE the measured encoder settings (int8 low / bf16 high)",
          LTX2Profile.compact24.recommendedTextEncoderQuant == .int8
              && LTX2Profile.balanced32.recommendedTextEncoderQuant == .int8
              && LTX2Profile.standard64.recommendedTextEncoderQuant == .bf16
              && LTX2Profile.max128.recommendedTextEncoderQuant == .bf16,
          "compact24/balanced32=int8 · standard64/max128=bf16")

    // 🚨 THE PIN RULE IS "does the FALLBACK still fit", not "is N small". Every tier whose budget
    // is met ONLY while streaming must pin the gate, or an .auto decline busts it outright.
    // Measured resident (fallback) peaks: compact24 31.92/16.8 ✗, balanced32 33.13/22.4 ✗,
    // standard64 27.31/44.8 ✓. balanced32 streamed 6/6 in measurement — but 6/6 is a probability,
    // and a declaration has to hold on the run that goes the other way.
    // ⟲ max128 ADDED 2026-08-23 for the same reason, one axis over: at its raised 1920x1088 cap the
    // resident corner (481f) is UNMEASURED and extrapolates ABOVE its declaration, so `.auto` could
    // bust it. standard64 stays UNPINNED and is the load-bearing negative — its declaration (34.8)
    // covers BOTH its streamed 33.55 and its resident 31.80, so pinning would remove a lane for
    // nothing. Without a negative case this rule degrades into "pin everything".
    check("26b a tier pins IFF its fallback could bust the declaration",
          LTX2Profile.compact24.recommendedForcedStreamGate
              && LTX2Profile.balanced32.recommendedForcedStreamGate
              && LTX2Profile.max128.recommendedForcedStreamGate
              && !LTX2Profile.standard64.recommendedForcedStreamGate,
          "compact24=\(LTX2Profile.compact24.recommendedForcedStreamGate) "
              + "balanced32=\(LTX2Profile.balanced32.recommendedForcedStreamGate) "
              + "standard64=\(LTX2Profile.standard64.recommendedForcedStreamGate)")

    // A config persisted BEFORE this key existed must still decode — and as bf16, the only
    // encoder that existed then (the bernini d02cfa1 rule).
    let legacyEnc = #"{"repo":"mlx-community/ltx-2.5-mlx","family":"ltx25","quant":"int8"}"#
    let decoded = try? JSONDecoder().decode(LTX2Configuration.self, from: Data(legacyEnc.utf8))
    check("27 legacy config without the key decodes as FOLLOW-PROFILE (nil), not a pinned value",
          decoded != nil && decoded?.textEncoderQuant == nil
              && decoded?.effectiveTextEncoderQuant == .bf16,
          "override=\(decoded?.textEncoderQuant?.rawValue ?? "nil") "
              + "effective=\(decoded.map { $0.effectiveTextEncoderQuant.rawValue } ?? "DECODE FAILED")")

    let round = try? JSONDecoder().decode(
        LTX2Configuration.self, from: JSONEncoder().encode(enc))
    check("28 an EXPLICIT textEncoderQuant survives a Codable round-trip",
          round?.textEncoderQuant == Quant.int8
              && round?.effectiveComponentsRepo == "mlx-community/ltx-2.5-mlx-q8",
          "\(round?.textEncoderQuant?.rawValue ?? "DECODE FAILED")")

    // ───── AUTO-FOLLOW (operator decision 2026-08-21: save users from themselves) ─────
    // 🔑 The point of these cases is that picking a low TIER and touching nothing else yields a
    // configuration that FITS. Before auto-follow it yielded one that busts the governor, silently.
    var lowTier = LTX2Configuration(family: .ltx25, repo: "mlx-community/ltx-2.5-mlx", profile: .compact24)
    lowTier.quant = .int8
    check("29 compact24 AUTO-FOLLOWS to the int8 encoder + a pinned gate",
          lowTier.effectiveTextEncoderQuant == .int8
              && lowTier.effectiveComponentsRepo == "mlx-community/ltx-2.5-mlx-q8"
              && lowTier.resolvedStreamingOptions.gatePolicy == .forceStream,
          "enc=\(lowTier.effectiveTextEncoderQuant.rawValue) "
              + "repo=\(lowTier.effectiveComponentsRepo) "
              + "gate=\(lowTier.resolvedStreamingOptions.gatePolicy.rawValue)")

    // …and it must reach weightSources, not just the computed property.
    check("30 auto-follow reaches weightSources",
          lowTier.weightSources.first { $0.role == "components" }?.repo
              == "mlx-community/ltx-2.5-mlx-q8",
          "\(lowTier.weightSources.first { $0.role == "components" }?.repo ?? "nil")")

    // The ESCAPE HATCH: an explicit override beats the profile, in both directions. This is what
    // makes auto-follow safe to ship — a tester can always get back to the unadvised configuration.
    var override = lowTier
    override.textEncoderQuant = .bf16
    override.forceStreamGate = false
    check("31 explicit overrides BEAT the profile advice (the escape hatch)",
          override.effectiveTextEncoderQuant == .bf16
              && override.effectiveComponentsRepo == "mlx-community/ltx-2.5-mlx"
              && override.resolvedStreamingOptions.gatePolicy == .auto,
          "enc=\(override.effectiveTextEncoderQuant.rawValue) "
              + "gate=\(override.resolvedStreamingOptions.gatePolicy.rawValue)")

    // High tiers must NOT auto-follow into int8 — the advice there is bf16, the reproducible arm.
    var highTier = LTX2Configuration(family: .ltx25, repo: "mlx-community/ltx-2.5-mlx", profile: .standard64)
    // standard64 keeps bf16 AND `.auto` — its declaration covers both lanes at the raised cap
    // (streamed 33.55 / resident 31.80 vs 34.8 declared), so it needs no pin.
    check("32 standard64 auto-follows to bf16/auto — still unpinned at the raised cap",
          highTier.effectiveTextEncoderQuant == .bf16
              && highTier.effectiveComponentsRepo == "mlx-community/ltx-2.5-mlx"
              && highTier.resolvedStreamingOptions.gatePolicy == .auto,
          "enc=\(highTier.effectiveTextEncoderQuant.rawValue) "
              + "gate=\(highTier.resolvedStreamingOptions.gatePolicy.rawValue)")

    // ⚠️ streamingOptions.gatePolicy is NOT the knob — resolution overwrites it. If this ever
    // stops being true, callers setting it directly will silently get the profile's policy.
    var direct = lowTier
    direct.streamingOptions.gatePolicy = .auto
    check("33 setting streamingOptions.gatePolicy directly does NOT survive resolution",
          direct.resolvedStreamingOptions.gatePolicy == .forceStream,
          "still \(direct.resolvedStreamingOptions.gatePolicy.rawValue) — use forceStreamGate")

    // ───── streaming as the DEFAULT for advised tiers (operator decision 2026-08-21) ─────
    var stream = LTX2Configuration(family: .ltx25, repo: "mlx-community/ltx-2.5-mlx", profile: .compact24)
    stream.quant = .int8
    // ⟲ max128 flipped 2026-08-23: EVERY tier streams. Its envelope reaches 1920x1088x481 and the
    // only lane measured across that corner is streamed+tiled (74.39 GB); the resident corner is
    // unmeasured and extrapolates ABOVE the old declaration.
    check("34 EVERY tier streams by default (max128 flipped with the raised cap)",
          LTX2Profile.compact24.recommendedStreamedBlocks
              && LTX2Profile.balanced32.recommendedStreamedBlocks
              && LTX2Profile.standard64.recommendedStreamedBlocks
              && LTX2Profile.max128.recommendedStreamedBlocks,
          "compact24/balanced32/standard64=on · max128=off (fits resident; streamed is UNMEASURED there)")

    // 🔑 THE POINT OF THE WIRING: streaming REPLACES the transformer download. Fetching both would
    // cost 70 GB instead of 35 on bf16 — the 2× that made "default streaming" look expensive.
    let roles = Set(stream.weightSources.map(\.role))
    check("35 streaming REPLACES the transformer source with granules (not both)",
          roles.contains("granules") && !roles.contains(where: { $0.hasPrefix("transformer-") }),
          "roles=\(roles.sorted())")

    check("36 granules resolve to the PUBLISHED tree",
          stream.weightSources.first { $0.role == "granules" }?.repo
              == "xocialize/ltx-2.5-granules",
          "\(stream.weightSources.first { $0.role == "granules" }?.repo ?? "nil")")

    // …and the converse. ⟲ No tier is non-streaming by ADVICE any more, so this now exercises the
    // EXPLICIT override — which is the path that still matters: a caller who opts out must get the
    // checkpoint, not granules, or they would stream against weights they never fetched.
    var resident = LTX2Configuration(family: .ltx25, repo: "mlx-community/ltx-2.5-mlx", profile: .max128)
    resident.quant = .int8
    resident.streamedBlocks = false          // explicit opt-out, not profile advice
    let rroles = Set(resident.weightSources.map(\.role))
    check("37 an EXPLICITLY non-streaming config fetches the transformer and NOT granules",
          rroles.contains("transformer-int8") && !rroles.contains("granules"),
          "roles=\(rroles.sorted())")

    // An explicit override must flip it back, in both directions — same escape hatch as the others.
    var noStream = stream
    noStream.streamedBlocks = false
    check("38 explicit streamedBlocks=false beats the profile and restores the transformer source",
          !noStream.effectiveStreamedBlocks
              && Set(noStream.weightSources.map(\.role)).contains("transformer-int8"),
          "effective=\(noStream.effectiveStreamedBlocks)")

    // ───── FOOTPRINT HINTS (AB-T-0069: the governor must ADMIT the streamed low tiers) ─────
    // The engine charges persistentHint + transientHint (MemoryGovernor.footprintSplit); these
    // cases mirror that arithmetic against the tier budgets (0.7 × RAM) and the AB-R-0106
    // worst-case measurements, so the declaration cannot drift out of the honest corridor
    // [worst measured, budget] in either direction without failing here.
    let gb = { (x: UInt64) in Double(x) / 1_000_000_000 }
    var fpLow = LTX2Configuration(family: .ltx25, repo: "mlx-community/ltx-2.5-mlx", profile: .compact24)
    fpLow.quant = .int8
    let c24 = (fpLow.residentBytesHint ?? 23_000_000_000) &+ (fpLow.peakActivationBytesHint ?? 5_000_000_000)
    check("39 compact24 streamed charge sits in [worst measured 15.49, budget 16.8]",
          fpLow.residentBytesHint != nil && fpLow.peakActivationBytesHint != nil
              && c24 >= 15_490_000_000 && c24 <= 16_800_000_000,
          String(format: "charge %.2f GB (resident %@ + activation %@)", gb(c24),
                 fpLow.residentBytesHint.map { String(format: "%.1f", gb($0)) } ?? "nil",
                 fpLow.peakActivationBytesHint.map { String(format: "%.1f", gb($0)) } ?? "nil"))

    var fpMid = fpLow; fpMid.profile = .balanced32
    let b32 = (fpMid.residentBytesHint ?? 23_000_000_000) &+ (fpMid.peakActivationBytesHint ?? 5_000_000_000)
    check("40 balanced32 streamed charge sits in [worst measured 17.14, budget 22.4]",
          fpMid.residentBytesHint != nil && fpMid.peakActivationBytesHint != nil
              && b32 >= 17_140_000_000 && b32 <= 22_400_000_000,
          String(format: "charge %.2f GB", gb(b32)))

    // standard64 keeps `.auto`, so its honest envelope is the RESIDENT fallback (27.31, AB-R-0105):
    // both hints nil → the quant-keyed int8 23+5 = 28 covers it. max128 keeps the resident bf16
    // lane: resident nil (quant 40) + activation 36 → charge 76, unchanged (the AB-T-0069
    // acceptance's other half).
    var fpStd = fpLow; fpStd.profile = .standard64
    var fpMax = LTX2Configuration(family: .ltx25, repo: "mlx-community/ltx-2.5-mlx", profile: .max128)
    fpMax.quant = .bf16
    // ⟲ REPLACED 2026-08-23. Both high tiers now declare from the WORST corner of their RAISED
    // envelope, measured streamed+tiled — standard64 1280x704x241 -> 33.55, max128 1920x1088x481
    // -> 74.39. ⚠️ The corner is what binds: at 1080p the peak is NOT flat across frames (49.30 ->
    // 60.35 -> 74.39), so a 121f-based declaration would have under-charged max128 by ~25 GB.
    check("41 standard64 and max128 declare from their measured streamed+tiled corners",
          fpStd.residentBytesHint == 800_000_000
              && fpStd.peakActivationBytesHint == 34_000_000_000
              && fpMax.residentBytesHint == 800_000_000
              && fpMax.peakActivationBytesHint == 74_500_000_000,
          "std hints=(\(fpStd.residentBytesHint.map(String.init) ?? "nil"),"
              + "\(fpStd.peakActivationBytesHint.map(String.init) ?? "nil")) "
              + "max128 act=\(fpMax.peakActivationBytesHint.map(String.init) ?? "nil")")

    // 🚨 FAIL-CLOSED, both escape hatches: a low tier that will NOT stream (explicit gate-off or
    // blocks-off) must get RESIDENT numbers — quant 23+5 = 28 → refused on 16.8. A streamed hint
    // here would under-declare a 31.92 GB run: fail-open, the AB-A-0012 condition.
    var fpGateOff = fpLow; fpGateOff.forceStreamGate = false
    var fpBlocksOff = fpLow; fpBlocksOff.streamedBlocks = false
    check("42 explicit stream-off on a low tier gets RESIDENT numbers (refused, correctly)",
          fpGateOff.residentBytesHint == nil && fpGateOff.peakActivationBytesHint == nil
              && fpBlocksOff.residentBytesHint == nil && fpBlocksOff.peakActivationBytesHint == nil,
          "gateOff=(\(fpGateOff.residentBytesHint.map(String.init) ?? "nil")) "
              + "blocksOff=(\(fpBlocksOff.residentBytesHint.map(String.init) ?? "nil"))")

    // 🚨 THE FAMILY TRAP: the engine reads hints from the config AS HANDED — coerced() covers repo
    // defaults, NOT footprints. A family-defaulted config on compact24 must yield 2.3's numbers
    // (resident nil, activation 3 GB), never 2.5's streamed 15.4 — fail-closed for the caller who
    // forgot `family: .ltx25`, and unchanged behaviour for actual 2.3 registrations.
    var fp23 = LTX2Configuration(repo: "xocialize/ltx-2.3-mlx", profile: .compact24)
    fp23.quant = .int4
    check("43 a family-defaulted (2.3) compact24 keeps 2.3 hints — coerced() does not cover footprints",
          fp23.residentBytesHint == nil && fp23.peakActivationBytesHint == 3_000_000_000,
          "resident=\(fp23.residentBytesHint.map(String.init) ?? "nil") "
              + "act=\(fp23.peakActivationBytesHint.map(String.init) ?? "nil")")

    // ── 44-47 · STORAGE FLOOR (contract 1.34.0, AB-T-0075 / AB-D-0038) ──────────────────────
    // 🚨 The field means "below this the run CRASHES", not "below this it is slow". I9 measured
    // bf16-on-USB at 0/7 while int8/int4 on the SAME volume were fine, so WHICH variants carry a
    // floor is the entire content of the declaration. Case 45 is the one that can actually fail:
    // declaring a floor on int8 would make the engine refuse (or warn about) a crash that has never
    // been observed on that variant — a false positive that costs users hardware.
    // ── 44-47 · STORAGE FLOOR (contract 1.34.0, AB-T-0075 / AB-D-0038) ──────────────────────
    // 🚨 The field means "below this the run CRASHES", not "below this it is slow". I9 measured
    // bf16-on-USB at 0/7 while int8/int4 on the SAME volume were fine, so WHICH variants carry a
    // floor is the entire content of the declaration. Case 45 is the one that can actually fail:
    // declaring a floor on int8 would have the engine refuse (or warn about) a crash never observed
    // on that variant — a false positive that costs users hardware. Case 47 pins the ABSENCE on the
    // streamed path so a future session cannot "helpfully" add one without reading AB-D-0038:
    // streaming degrades in TIME, not safety, and that advisory is app-side off
    // volumeCharacterization().
    let fps25: [QuantFootprint] = m.requirements.footprints
    let fps23: [QuantFootprint] = MLXLTX2Package.manifest.requirements.footprints
    // ⚠️ Hoisted into plain lets deliberately: `.first { }?.field.map(String.init) ?? "nil"` inside
    // a string interpolation blows the Swift type-checker's budget here (measured, twice).
    let floor25bf16: UInt64? = fps25.first { $0.quant == .bf16 }?.minSustainedReadBytesPerSecond
    let floor25int8: UInt64? = fps25.first { $0.quant == .int8 }?.minSustainedReadBytesPerSecond
    let floor23bf16: UInt64? = fps23.first { $0.quant == .bf16 }?.minSustainedReadBytesPerSecond
    let nonBf16: [QuantFootprint] = (fps25 + fps23).filter { $0.quant != .bf16 }
    let offenders: Int = nonBf16.filter { $0.minSustainedReadBytesPerSecond != nil }.count
    func show(_ v: UInt64?) -> String { v.map(String.init) ?? "nil" }

    check("44 2.5 bf16 declares a storage floor (I9 class: sub-floor is a CRASH, not a slowdown)",
          floor25bf16 == 1_000_000_000, show(floor25bf16))
    check("45 2.5 int8 declares NO floor — int8/int4 survived the same USB volume bf16 died on",
          floor25int8 == nil, show(floor25int8))
    check("46 2.3 bf16 declares the floor too — it is the variant I9 was MEASURED on",
          floor23bf16 == 1_000_000_000, show(floor23bf16))
    check("47 no non-bf16 variant carries a floor (streaming degrades in TIME — app-side advisory)",
          offenders == 0, "\(offenders) offenders")

    // ── 48-51 · EXPECTED READ VOLUME (contract 1.35.0, AB-A-0013 option b) ──────────────────
    // 🔑 PERFORMANCE hint, never refuses — the counterpart to 44-47's crash floor. The engine
    // projects I/O time from its own measured B/s, so the number's job is to be RIGHT about the
    // lane, not conservative. Streamed lanes re-read the sweep every step; resident lanes read once.
    var rdC = LTX2Configuration(family: .ltx25, profile: .compact24)
    rdC.quant = .int8
    var rdS = LTX2Configuration(family: .ltx25, profile: .standard64)
    rdS.quant = .int8
    var rdM = LTX2Configuration(family: .ltx25, profile: .max128)
    rdM.quant = .bf16
    var rd23 = LTX2Configuration(profile: .standard64)
    rd23.quant = .int8
    let rC: UInt64? = rdC.expectedWeightReadBytesPerRunHint
    let rS: UInt64? = rdS.expectedWeightReadBytesPerRunHint
    let rM: UInt64? = rdM.expectedWeightReadBytesPerRunHint
    let r23: UInt64? = rd23.expectedWeightReadBytesPerRunHint
    check("48 compact24 streamed = one-stage 8 steps x 18.37 GiB sweep (~158 GB/clip)",
          rC == 19_724_000_000 * 8, show(rC))
    check("49 standard64 streamed = TWO-stage 11 steps — the sigma schedule, not a guess",
          rS == 19_724_000_000 * 11, show(rS))
    // ⟲ max128 now STREAMS by advice, so it sweeps per step rather than reading the checkpoint
    // once. The resident branch is still covered — case 37 exercises the explicit opt-out.
    check("50 max128 STREAMS by advice — 11 sweeps of the measured bf16 granule set",
          rM == 37_106_000_000 * 11, show(rM))
    // ⚠️ bf16 streamed is max128's OVERRIDE lane (its profile advises resident). The sweep here is
    // the MEASURED bind-line figure — 34.56 GiB — not the on-disk ratio, which was ~2% out.
    var rdMs = LTX2Configuration(family: .ltx25, profile: .max128)
    rdMs.quant = .bf16; rdMs.streamedBlocks = true
    let rMs: UInt64? = rdMs.expectedWeightReadBytesPerRunHint
    check("52 max128 STREAMED (explicit override) uses the measured 34.56 GiB bf16 sweep x 11",
          rMs == 37_106_000_000 * 11, show(rMs))

    check("51 2.3 declares no read hint — the sweep numbers are 2.5-measured (family-keyed)",
          r23 == nil, show(r23))

    // ── 53-56 · RUN STAGE PLAN (AB-R-0122; the pre-run list a stepper UI draws on first paint) ──
    // 🚨 The plan's ONLY value is matching what the pipeline actually emits. These cases pin it
    // against the MEASURED sequences (LTX_PROGRESS=1, 2026-08-22), so a pipeline change that adds
    // or drops a phase fails here instead of silently desynchronising every host's stepper.
    var planC = LTX2Configuration(family: .ltx25, profile: .compact24)
    planC.quant = .int8
    var planS = LTX2Configuration(family: .ltx25, profile: .standard64)
    planS.quant = .int8
    let idsOne: [String] = planC.plannedStages().map(\.id)
    let idsTwo: [String] = planS.plannedStages().map(\.id)
    // MEASURED one-stage: encode -> denoise -> decode -> postprocess
    check("53 one-stage (compact24) plan matches the measured 4-phase sequence",
          idsOne == ["encode", "denoise.1", "decode", "postprocess"], idsOne.joined(separator: ","))
    // MEASURED two-stage: encode -> denoise(1/2) -> upsample -> denoise(2/2) -> decode -> postprocess
    check("54 two-stage (standard64) plan matches the measured 6-phase sequence",
          idsTwo == ["encode", "denoise.1", "upsample", "denoise.2", "decode", "postprocess"],
          idsTwo.joined(separator: ","))
    // i2v with an UNCACHED adapter must warn about the 4.93 GB in-run fetch (AB-R-0114) — a UI
    // that omits it shows an unexplained stall.
    let idsI2V: [String] = planS.plannedStages(initImage: true, adapterCached: false).map(\.id)
    check("55 i2v with an uncached adapter prepends the fetch node",
          idsI2V.first == "adapter" && idsI2V.count == idsTwo.count + 1, idsI2V.joined(separator: ","))
    // ⚠️ Every phase the plan names must be one the progress plane can actually emit. An invented
    // phase string would leave its node permanently un-lit.
    let known: Set<String> = ["encode", "denoise", "upsample", "decode", "postprocess", "generate"]
    let named: [String] = (planC.plannedStages() + planS.plannedStages()).compactMap(\.phase)
    let unknown: [String] = named.filter { !known.contains($0) }
    check("56 every planned phase is one RunProgress can emit (no invented phase names)",
          unknown.isEmpty, unknown.isEmpty ? "all known" : unknown.joined(separator: ","))

    print(failures.isEmpty
          ? "[ltx25-package-gate] PASS ✅ (56/56)"
          : "[ltx25-package-gate] FAIL ❌ \(failures.count): \(failures.joined(separator: ", "))")
    if !failures.isEmpty { exit(1) }
}
