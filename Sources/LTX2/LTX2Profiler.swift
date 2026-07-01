// LTX2Profiler.swift — env-gated timing + memory instrumentation for the LTX-2.3 pipeline.
//
// Purpose: localize where wall-clock goes during generation (why does 48f cost ≫ 24f?) and tell
// COMPUTE-bound from MEMORY-bound (paging). MLX is lazy, so a region's true cost is only realized
// at an `eval` — every region below is measured around the pipeline's existing per-stage / per-step
// `eval`, so the ms are honest. Each region also snapshots:
//   • MLX active / cache / peak  (the buffer pool; `cache` growing unbounded is the classic
//     unified-memory footgun — an uncapped pool inflates phys_footprint until the OS pages)
//   • OS phys_footprint          (the real RAM the process holds — the number that, once it crosses
//     the Metal working-set ceiling, triggers paging → GPU stalls at <10% util with huge wall-clock)
//   • a ⚠PAGING flag when phys_footprint > maxRecommendedWorkingSetSize.
//
// Enable with the env var `LTX_PROFILE=1` (any non-empty value). `LTX_PROFILE=csv` also writes
// `/tmp/ltx-profile.csv`. Zero overhead when disabled (spans early-return before any syscall).
//
// Live-logs each region as it completes (a hang is visible AS it happens, not only in a post-mortem
// summary) + prints a grouped summary at end-of-run.

import Foundation
import Darwin
import MLX

public final class LTX2Profiler: @unchecked Sendable {
    public static let shared = LTX2Profiler()

    public let enabled: Bool
    private let csv: Bool
    private let lock = NSLock()
    private let workingSet: UInt64          // Metal's soft ceiling; phys past this = paging risk

    public struct Row {
        public let group: String   // "encode" / "denoise" / "upscale" / "vae-decode" / "audio-decode"
        public let label: String   // step index / stage tag
        public let ms: Double
        public let activeGB: Double
        public let cacheGB: Double
        public let peakGB: Double
        public let physGB: Double
        public let paging: Bool
        public let note: String
    }
    private var rows: [Row] = []

    private init() {
        let v = ProcessInfo.processInfo.environment["LTX_PROFILE"]
        enabled = (v?.isEmpty == false)
        csv = (v == "csv")
        workingSet = enabled ? GPU.deviceInfo().maxRecommendedWorkingSetSize : 0
    }

    // MARK: - phys_footprint (OS RAM the process holds — the truthful figure vs peakMemory)

    private static func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
    }

    private static func gb(_ b: Int) -> Double { Double(b) / 1_000_000_000.0 }
    private static func gb(_ b: UInt64) -> Double { Double(b) / 1_000_000_000.0 }

    // MARK: - Spans (measure a region whose output is eval'd at/just before `end`)

    public struct Span {
        let group: String, label: String, note: String
        let t0: Date
        let enabled: Bool
    }

    /// Begin a timed region. The caller MUST `eval` the region's output tensors before `end()` —
    /// otherwise MLX's laziness attributes the compute to a later `eval`. All the pipeline's regions
    /// already eval at their boundary, so wrap tightly around that.
    public func begin(_ group: String, _ label: String = "", note: String = "") -> Span {
        guard enabled else { return Span(group: group, label: label, note: note, t0: Date.distantPast, enabled: false) }
        return Span(group: group, label: label, note: note, t0: Date(), enabled: true)
    }

    public func end(_ span: Span) {
        guard span.enabled else { return }
        let ms = Date().timeIntervalSince(span.t0) * 1000.0
        let snap = Memory.snapshot()
        let phys = Self.physFootprint()
        let paging = workingSet > 0 && phys > workingSet
        let row = Row(group: span.group, label: span.label, ms: ms,
                      activeGB: Self.gb(snap.activeMemory), cacheGB: Self.gb(snap.cacheMemory),
                      peakGB: Self.gb(snap.peakMemory), physGB: Self.gb(phys),
                      paging: paging, note: span.note)
        lock.lock(); rows.append(row); lock.unlock()
        print(String(format: "[LTX-PROF] %-22@ %8.1fms  act=%.1f cache=%.1f phys=%.1f/%.0f GB%@  %@",
                     "\(span.group)/\(span.label)" as NSString, ms,
                     row.activeGB, row.cacheGB, row.physGB, Self.gb(workingSet),
                     (paging ? "  ⚠PAGING" : "") as NSString, span.note as NSString))
    }

    /// Free-form note line (geometry at run start, etc.).
    public func note(_ message: String) {
        guard enabled else { return }
        print("[LTX-PROF] \(message)")
    }

    // MARK: - Run lifecycle

    public func beginRun(_ header: String) {
        guard enabled else { return }
        lock.lock(); rows.removeAll(); lock.unlock()
        let ws = Self.gb(workingSet)
        print("[LTX-PROF] ===== \(header) =====")
        print(String(format: "[LTX-PROF] workingSet(recommended)=%.1f GB  cacheLimit=%.1f GB  (phys past workingSet ⇒ paging)",
                     ws, Self.gb(Memory.cacheLimit)))
    }

    /// Grouped end-of-run summary (total ms per group + worst phys + any paging).
    public func endRun() {
        guard enabled else { return }
        lock.lock(); let rs = rows; lock.unlock()
        guard !rs.isEmpty else { return }
        print("[LTX-PROF] ---------- summary (ms per group) ----------")
        var order: [String] = []
        var totals: [String: (ms: Double, n: Int, worstPhys: Double, paging: Bool)] = [:]
        for r in rs {
            if totals[r.group] == nil { order.append(r.group) }
            var t = totals[r.group] ?? (0, 0, 0, false)
            t.ms += r.ms; t.n += 1; t.worstPhys = max(t.worstPhys, r.physGB); t.paging = t.paging || r.paging
            totals[r.group] = t
        }
        let grand = rs.reduce(0.0) { $0 + $1.ms }
        for g in order {
            let t = totals[g]!
            print(String(format: "[LTX-PROF]   %-16@ %9.1fms  (%2d region(s), %4.1f%%)  worstPhys=%.1f GB%@",
                         g as NSString, t.ms, t.n, 100 * t.ms / grand, t.worstPhys,
                         (t.paging ? "  ⚠PAGING" : "") as NSString))
        }
        print(String(format: "[LTX-PROF]   %-16@ %9.1fms  (%.1fs total)", "TOTAL" as NSString, grand, grand / 1000))
        if rs.contains(where: { $0.paging }) {
            print("[LTX-PROF] ⚠ phys_footprint crossed the Metal working-set ceiling — the GPU is paging;")
            print("[LTX-PROF]   that is the <10% GPU / long-wall-clock signature. Levers: cap Memory.cacheLimit,")
            print("[LTX-PROF]   decode the VAE in temporal chunks, or lower frames/resolution for this tier.")
        }
        if csv { writeCSV(rs) }
    }

    private func writeCSV(_ rs: [Row]) {
        var out = "group,label,ms,activeGB,cacheGB,peakGB,physGB,paging,note\n"
        for r in rs {
            out += "\(r.group),\(r.label),\(String(format: "%.1f", r.ms)),\(String(format: "%.2f", r.activeGB)),"
            out += "\(String(format: "%.2f", r.cacheGB)),\(String(format: "%.2f", r.peakGB)),\(String(format: "%.2f", r.physGB)),"
            out += "\(r.paging),\"\(r.note)\"\n"
        }
        try? out.write(toFile: "/tmp/ltx-profile.csv", atomically: true, encoding: .utf8)
        print("[LTX-PROF] wrote /tmp/ltx-profile.csv")
    }
}
