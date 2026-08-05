// BenchAnalysis.swift — the bench-e2e verdict logic, extracted PURE so it is testable without a
// single GPU second (`--bench-verdict-selftest` encodes the harness's own past failures as
// regression cases). BENCH.md "v2 verdict-logic items", implemented 2026-08-05.
//
// v1 checks (kept): Δmedian vs spread; sign consistency across block sets.
// v2 additions, each motivated by a receipt where v1 mis-called or under-called:
//   · LINEAR DRIFT — pooled OLS of wall on global run order. When the fitted span rivals Δmed,
//     the session trend can masquerade as an arm difference (or hide one); the verdict then uses
//     the DRIFT-ADJUSTED Δ (residualized walls). Receipt: the mux A/B, where whichever lane ran
//     second was slower and two extra legs were burned proving it by hand.
//   · NONSTATIONARITY — within-arm spans across block sets. A one-way thermal step lands inside
//     one arm's blocks and ABBA cannot cancel it; sign-consistency is blind when both blocks land
//     the same side. Receipt: the nullfloor-ssd false positive (+8.5 s on identical arms).
//   · THERMAL STRATA — Δ recomputed within matching thermalBefore states, plus an imbalance
//     warning when a stratum is single-arm (the nullfloor case: the only nominal runs were arm
//     A's). Coarse (nominal/fair/serious), so it annotates rather than decides.
//
// Annotations soften a MEASURABLE to "MEASURABLE ⚠"; hard NOISE still comes only from sign-flip,
// spread, or the drift-adjusted Δ collapsing. The harness flags; the reader decides.

import Foundation

struct BenchAnalysis {
    struct ArmStats {
        let name: String
        let med: Double, min: Double, max: Double
        let byBlock: [Int: Double]
        let peakMed: Double
        let adjustedMed: Double
        let adjustedRange: Double
        let blockSpan: Double        // max |byBlock_i − byBlock_j| — the nonstationarity measure
    }

    let stats: [ArmStats]
    let slopePerRun: Double          // pooled OLS slope, seconds per measured run
    let driftSpan: Double            // slope × (n−1) — the fitted trend across the session
    let strata: [(state: String, deltas: [String: Double], counts: [String: (Int, Int)])]
    private let blocks: Int

    init(runRecords: [RunRecord], armNames: [String], blocks: Int) {
        self.blocks = blocks
        let measured = runRecords.filter { $0.run > 0 }
        let n = measured.count

        // Pooled OLS of wall on global (chronological) index — drift is a SESSION property.
        var slope = 0.0
        if n >= 3 {
            let xs = (0 ..< n).map(Double.init)
            let ys = measured.map(\.wallSeconds)
            let mx = xs.reduce(0, +) / Double(n), my = ys.reduce(0, +) / Double(n)
            let sxx = zip(xs, xs).reduce(0) { $0 + ($1.0 - mx) * ($1.1 - mx) }
            let sxy = zip(xs, ys).reduce(0) { $0 + ($1.0 - mx) * ($1.1 - my) }
            slope = sxx > 0 ? sxy / sxx : 0
        }
        slopePerRun = slope
        driftSpan = slope * Double(max(n - 1, 0))
        let meanIdx = Double(max(n - 1, 0)) / 2.0

        var built: [ArmStats] = []
        for name in armNames {
            let idxWalls: [(Int, RunRecord)] = measured.enumerated()
                .filter { $0.element.arm == name }
                .map { ($0.offset, $0.element) }
            let walls = idxWalls.map { $0.1.wallSeconds }
            let adjusted = idxWalls.map { $0.1.wallSeconds - slope * (Double($0.0) - meanIdx) }
            var byBlock: [Int: Double] = [:]
            for b in 0 ..< blocks {
                let bw = idxWalls.filter { $0.1.block == b }.map { $0.1.wallSeconds }
                if !bw.isEmpty { byBlock[b] = benchMedian(bw) }
            }
            let blockMeds = byBlock.values.sorted()
            built.append(ArmStats(
                name: name,
                med: benchMedian(walls), min: walls.min() ?? 0, max: walls.max() ?? 0,
                byBlock: byBlock,
                peakMed: benchMedian(idxWalls.map { $0.1.peakPhysGB }),
                adjustedMed: benchMedian(adjusted),
                adjustedRange: (adjusted.max() ?? 0) - (adjusted.min() ?? 0),
                blockSpan: blockMeds.isEmpty ? 0 : (blockMeds.last! - blockMeds.first!)))
        }
        stats = built

        // Thermal strata: Δmed per thermalBefore state, only meaningful when both arms appear.
        var strataOut: [(String, [String: Double], [String: (Int, Int)])] = []
        let states = Array(Set(measured.map(\.thermalBefore))).sorted()
        for st in states {
            let inState = measured.filter { $0.thermalBefore == st }
            var deltas: [String: Double] = [:]
            var counts: [String: (Int, Int)] = [:]
            guard let refName = armNames.first else { break }
            let refWalls = inState.filter { $0.arm == refName }.map(\.wallSeconds)
            for name in armNames.dropFirst() {
                let armWalls = inState.filter { $0.arm == name }.map(\.wallSeconds)
                counts[name] = (refWalls.count, armWalls.count)
                if !refWalls.isEmpty && !armWalls.isEmpty {
                    deltas[name] = benchMedian(armWalls) - benchMedian(refWalls)
                }
            }
            strataOut.append((st, deltas, counts))
        }
        strata = strataOut
    }

    func verdict(ref: ArmStats, arm: ArmStats) -> String {
        let d = arm.med - ref.med
        let spread = Swift.max(ref.max - ref.min, arm.max - arm.min)

        // v1: sign consistency across block sets — still a hard NOISE.
        var signs: [Double] = []
        for b in 0 ..< blocks {
            if let r = ref.byBlock[b], let a = arm.byBlock[b], r > 0, a > 0 { signs.append(a - r) }
        }
        if signs.count > 1, Set(signs.map { $0 > 0 }).count > 1 {
            return String(format: "Δmed %+.1fs — NOISE (sign flips between block sets)", d)
        }

        // v2: when the fitted session trend rivals the raw delta, decide on the ADJUSTED delta.
        let driftDominant = abs(driftSpan) >= Swift.max(abs(d), 1e-9) * 0.75
        let dAdj = arm.adjustedMed - ref.adjustedMed
        let spreadAdj = Swift.max(ref.adjustedRange, arm.adjustedRange)
        let decisionD = driftDominant ? dAdj : d
        let decisionSpread = driftDominant ? spreadAdj : spread

        var annotations: [String] = []
        if driftDominant {
            annotations.append(String(format: "drift span %+.1fs rivals Δ — decided on drift-adjusted Δ %+.1fs", driftSpan, dAdj))
        }
        // v2: within-arm nonstationarity (the thermal-step class ABBA cannot cancel).
        let nonstat = Swift.max(ref.blockSpan, arm.blockSpan)
        if nonstat >= abs(decisionD) * 0.5, nonstat > 0 {
            annotations.append(String(format: "nonstationary: within-arm block span %.1fs vs Δ %.1fs", nonstat, abs(decisionD)))
        }
        // v2: thermal strata — disagreeing sign, or single-arm strata (imbalance).
        for (st, deltas, counts) in strata {
            if let sd = deltas[arm.name], sd.sign != decisionD.sign, abs(sd) > 0.05 * Swift.max(abs(decisionD), 1) {
                annotations.append(String(format: "thermal '%@' stratum disagrees (Δ %+.1fs)", st, sd))
            } else if let c = counts[arm.name], (c.0 == 0) != (c.1 == 0) {
                annotations.append("thermal '\(st)' stratum is single-arm — arms did not share this state")
            }
        }

        let note = annotations.isEmpty ? "" : "  ⚠ " + annotations.joined(separator: " · ")
        if abs(decisionD) <= decisionSpread {
            return String(format: "Δmed %+.1fs ≤ spread %.1fs — NOISE%@",
                          decisionD, decisionSpread, note)
        }
        // Annotated MEASURABLE = "the number cleared the bar, but the session has a confound".
        return String(format: "Δmed %+.1fs > spread %.1fs — MEASURABLE%@%@",
                      decisionD, decisionSpread, annotations.isEmpty ? "" : " ⚠", note)
    }

    func driftMarkdown() -> String {
        var md = "\n## Session drift & thermal strata (v2)\n\n"
        md += String(format: "- pooled drift: %+.3f s/run → fitted span %+.1f s across the session\n",
                     slopePerRun, driftSpan)
        for s in stats {
            md += String(format: "- `%@`: raw med %.1f s · drift-adjusted med %.1f s · within-arm block span %.1f s\n",
                         s.name, s.med, s.adjustedMed, s.blockSpan)
        }
        for (st, deltas, counts) in strata {
            let parts = deltas.map { String(format: "%@ Δ%+.1fs", $0.key, $0.value) }
                + counts.filter { deltas[$0.key] == nil }.map { "\($0.key): single-arm (\($0.value.0)/\($0.value.1))" }
            md += "- thermal `\(st)`: " + (parts.isEmpty ? "—" : parts.joined(separator: " · ")) + "\n"
        }
        return md
    }
}

func benchMedian(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    guard !s.isEmpty else { return 0 }
    return s.count % 2 == 1 ? s[s.count / 2] : 0.5 * (s[s.count / 2 - 1] + s[s.count / 2])
}

/// `--bench-verdict-selftest` — the v2 verdict logic against the harness's own past failures,
/// pure logic, zero GPU. Each case is a real receipt or a known failure class.
func benchVerdictSelfTest() {
    func rec(_ arm: String, _ block: Int, _ wall: Double, _ thermal: String) -> RunRecord {
        RunRecord(arm: arm, block: block, run: 1, wallSeconds: wall, peakPhysGB: 50,
                  mlxActiveGB: 40, mlxCacheGB: 2, thermalBefore: thermal, thermalAfter: thermal,
                  outputMean: 0, outputStd: 1, cosineVsArmFirst: nil)
    }
    var allPass = true
    func check(_ name: String, _ verdict: String, expectContains: [String], expectNotContains: [String] = []) {
        var ok = true
        for e in expectContains where !verdict.contains(e) { ok = false }
        for e in expectNotContains where verdict.contains(e) { ok = false }
        allPass = allPass && ok
        print("[bench-verdict-selftest] \(ok ? "✅" : "❌") \(name)")
        print("    → \(verdict)")
    }

    // 1. The nullfloor-ssd false positive (2026-08-03, receipt bench_e2e_nullfloor-ssd_…): a
    //    one-way nominal→fair step inside block 0 handed arm A the only cool runs. v1 said
    //    "MEASURABLE +8.5s" unannotated. v2 must ANNOTATE it (nonstationary within-arm spans
    //    and/or a single-arm thermal stratum) — the reader must see the confound.
    let nullfloor = [
        rec("A", 0, 16.6, "nominal"), rec("A", 0, 17.8, "nominal"),
        rec("B", 0, 29.2, "fair"), rec("B", 0, 30.8, "fair"),
        rec("B", 1, 28.0, "fair"), rec("B", 1, 26.6, "fair"),
        rec("A", 1, 22.4, "fair"), rec("A", 1, 22.8, "fair"),
    ]
    let a1 = BenchAnalysis(runRecords: nullfloor, armNames: ["A", "B"], blocks: 2)
    check("nullfloor thermal-step null must carry ⚠",
          a1.verdict(ref: a1.stats[0], arm: a1.stats[1]), expectContains: ["⚠"])

    // 2. Pure linear session drift under UNBALANCED ordering (blocks=1, A,A,B,B): raw Δ reads
    //    +2.0 > spread 1.0 = a clean false MEASURABLE in v1. The drift regression must take over
    //    and collapse it to NOISE with the drift annotation.
    let drifting = [
        rec("A", 0, 10.0, "fair"), rec("A", 0, 11.0, "fair"),
        rec("B", 0, 12.0, "fair"), rec("B", 0, 13.0, "fair"),
    ]
    let a2 = BenchAnalysis(runRecords: drifting, armNames: ["A", "B"], blocks: 1)
    check("pure linear drift (AABB) must collapse to NOISE via adjusted Δ",
          a2.verdict(ref: a2.stats[0], arm: a2.stats[1]),
          expectContains: ["NOISE", "drift"])

    // 3. A clean real difference under proper ABBA: stays MEASURABLE, no annotations.
    let clean = [
        rec("A", 0, 10.0, "fair"), rec("A", 0, 10.2, "fair"),
        rec("B", 0, 13.1, "fair"), rec("B", 0, 13.0, "fair"),
        rec("B", 1, 13.2, "fair"), rec("B", 1, 12.9, "fair"),
        rec("A", 1, 10.1, "fair"), rec("A", 1, 10.3, "fair"),
    ]
    let a3 = BenchAnalysis(runRecords: clean, armNames: ["A", "B"], blocks: 2)
    check("clean ABBA difference stays MEASURABLE without ⚠",
          a3.verdict(ref: a3.stats[0], arm: a3.stats[1]),
          expectContains: ["MEASURABLE"], expectNotContains: ["⚠"])

    // 4. The v1 sign-flip rule must survive v2 unchanged.
    let flipping = [
        rec("A", 0, 10.0, "fair"), rec("B", 0, 12.0, "fair"),
        rec("B", 1, 10.0, "fair"), rec("A", 1, 12.0, "fair"),
    ]
    let a4 = BenchAnalysis(runRecords: flipping, armNames: ["A", "B"], blocks: 2)
    check("sign flip between block sets stays hard NOISE",
          a4.verdict(ref: a4.stats[0], arm: a4.stats[1]),
          expectContains: ["NOISE", "sign flips"])

    print(allPass ? "[bench-verdict-selftest] PASS ✅" : "[bench-verdict-selftest] FAIL ❌")
    if !allPass { exit(1) }
}
