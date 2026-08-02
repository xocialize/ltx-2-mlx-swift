// HV2 weight streaming, part 2: the runtime — LTX-2.3 adaptation of wan-core's
// BlockStreamer (commit b45b879; receipts probes/hv2_stream_proto.out +
// hv2_wan_blockstreamer.out; STREAMING-PLAN.md). Sibling implementation, not a
// wan-core dependency — see GranuleLayout.swift's provenance note.
//
// Streams the DiT's 48 transformer blocks from per-block granule files through
// TWO resident block-group slots, refilled by a background pread thread while
// the GPU computes the other slot — the pread double-buffer proven bit-exact at
// scale by the HV2 prototype over the P-C alias seam.
//
// The LTX delta vs wan-core: the DiT is a FUNCTIONAL port over a weight dict,
// not a Module tree. Binding therefore CONSTRUCTS the dict — every
// `transformer_blocks.<i>.<suffix>` key maps to a slot-backed array (slot
// `(i/G)%2`, local `i%G`) — and the resident fallback REPLACES those dict
// entries through the shared `WeightStore` reference (the LoRAStore idiom).
// There is exactly one block loop (`DiT.callAsFunction`), which routes through
// `acquireGroup`/`releaseGroup` when `store.streamer` is set; the hand-written-
// loop corruption trap that bit bernini's editing surfaces is structurally
// absent today — any FUTURE surface that walks blocks itself must route the
// same way or refuse while streamed.
//
// Mechanics carried verbatim from the prototype (do not re-derive):
//   - Slot-backed arrays are created ONCE at bind; refills mutate the backings
//     through the P-C alias (`asData(.noCopyIfContiguous)` — same pointer as
//     `asMTLBuffer(noCopy:true)`, works below page alignment).
//   - STEP-MAJOR loop order: the cyclic group sequence advances continuously
//     across every forward (warmup sweep included) — never block-major.
//   - Compute `eval`s at every group boundary BEFORE releasing the slot — the
//     handoff that makes refills invisible to the lazy graph (and the reason
//     streamed output is bit-identical to resident).
//   - IO runs on a plain nonisolated background thread and never touches MLX.
//   - Granules are read F_NOCACHE: bounded memory behavior is the point, and S
//     measurements stay honest (the prototype's cold-number recipe).
//
// Self-calibrating runtime gate (prototype phase G):
//   stream hides iff S ≥ C(N)  ⇔  N ≥ B·F/(2·S)
// with S measured on the first refill sweeps and F/C measured IN-REGIME during
// the first gating forward. Policy `.auto` falls back to fully-resident
// automatically (output-invisibly) when the arithmetic doesn't clear.
//
// LTX-specific gate detail (STREAMING-PLAN §4): the two-stage pipeline runs
// stage 1 at ~¼ the tokens of stage 2. `gateEvaluationThresholdTokens` defers
// the verdict until the first forward at ≥ that many tokens (the pipeline sets
// it to the LARGEST stage's tokens), so a small stage-1 forward doesn't condemn
// a run whose stage 2 would stream. Below-threshold forwards stream at the IO
// floor — bounded, and only paid where .auto would otherwise mis-gate.
// `DiT.warmup()`'s nv=1 sweep sets `gateSuspended` instead — it primes Metal
// JIT AND doubles as S calibration, but must never feed C (its N is
// pathological by design).

import Foundation
import MLX
import Metal
import os

public enum LTXBlockStreamerError: LocalizedError {
    case noMetalDevice
    case aliasUnavailable(String)
    case pointerInstability(String)
    case contract(String)
    case state(String)

    public var errorDescription: String? {
        switch self {
        case .noMetalDevice: return "LTXBlockStreamer: no Metal device"
        case .aliasUnavailable(let m):
            return "LTXBlockStreamer: asMTLBuffer(noCopy:true) unavailable for \(m) — seam broken"
        case .pointerInstability(let m):
            return "LTXBlockStreamer: slot backing pointer moved (\(m)) — refusing to stream"
        case .contract(let m): return "LTXBlockStreamer contract violation: \(m)"
        case .state(let m): return "LTXBlockStreamer state error: \(m)"
        }
    }
}

/// Safetensors dtype string → MLX DType (the subset LTX checkpoints use).
func mlxDType(safetensors dtype: String) throws -> DType {
    switch dtype {
    case "BF16": return .bfloat16
    case "F16": return .float16
    case "F32": return .float32
    case "U32": return .uint32
    case "U8": return .uint8
    case "I32": return .int32
    case "I64": return .int64
    default:
        throw LTXBlockStreamerError.contract("unsupported safetensors dtype \(dtype)")
    }
}

/// The raw backing pointer of an evaluated, contiguous MLXArray — the P-C alias
/// seam. A silent copy (non-contiguous backing) is detected by double
/// extraction: a wrapper returns the same address twice, a copy cannot.
func backingPointer(_ a: MLXArray, context: @autoclosure () -> String) throws
    -> UnsafeMutableRawPointer
{
    func extract() -> UnsafeMutableRawPointer? {
        let d = a.asData(access: .noCopyIfContiguous)
        return d.data.withUnsafeBytes { raw in
            raw.baseAddress.map { UnsafeMutableRawPointer(mutating: $0) }
        }
    }
    guard let p1 = extract(), let p2 = extract(), p1 == p2 else {
        throw LTXBlockStreamerError.aliasUnavailable(context())
    }
    return p1
}

/// Opt-in configuration for block streaming. Constructed by the consumer's
/// package configuration (per config, never ambient).
public struct BlockStreamingOptions: Sendable {
    /// Blocks per granule group (slot capacity). Must divide the block count
    /// (48). 2 keeps slots ~1.5 GB bf16 / ~0.8 GB q8 / ~0.4 GB q4.
    public var groupSize: Int
    /// `.auto` — evaluate the runtime gate on the first gating forward and fall
    /// back to fully-resident when it doesn't clear. `.forceStream` — never
    /// fall back (receipts, parity tests).
    public var gatePolicy: GatePolicy
    /// Extra safety margin on the gate: require S ≥ C·margin.
    public var gateMargin: Double
    /// Suppress the one-line lifecycle prints.
    public var quiet: Bool

    public enum GatePolicy: String, Sendable {
        case auto
        case forceStream
    }

    public init(
        groupSize: Int = 2,
        gatePolicy: GatePolicy = .auto,
        gateMargin: Double = 1.0,
        quiet: Bool = false
    ) {
        self.groupSize = groupSize
        self.gatePolicy = gatePolicy
        self.gateMargin = gateMargin
        self.quiet = quiet
    }
}

/// What the gate measured and decided — surfaced for receipts and logs.
public struct LTXStreamerGateReport: Sendable {
    public var n: Int  // tokens in the gating forward (batch × (videoN + audioN))
    public var sGiBs: Double  // measured refill bandwidth
    public var cGiBs: Double  // measured GPU weight-consumption rate C(N)
    public var nMin: Int  // N_min = N · C / S at the measured rates
    public var step1ComputeSeconds: Double
    public var step1StallSeconds: Double
    public var streaming: Bool  // the decision
}

public final class LTXBlockStreamer: @unchecked Sendable {

    public enum Verdict: String, Sendable {
        case undecided  // gate not yet evaluated
        case streaming  // gate cleared (or .forceStream)
        case fellBack  // gate failed under .auto — blocks were made resident
    }

    /// Tensors smaller than this stay RESIDENT (loaded once per block at bind)
    /// instead of slot-backed. Two reasons: (1) ⚠️ Swift `Data` stores payloads
    /// ≤14 bytes INLINE, so `asData(.noCopyIfContiguous)` on a tiny array can
    /// hand back a pointer into a transient inline copy instead of the MLX
    /// backing — refills would write into the void while the slot keeps its
    /// zeros (caught by the tiny gate: the [2]-shaped fp32 gate biases were the
    /// ONLY divergent tensors). (2) sub-alignment tensors cost one syscall each
    /// per refill for near-zero bytes. 256 keeps the real model's 64-byte gate
    /// biases resident (~37 KB total across 48 blocks) while everything wan
    /// receipted as alias-safe (≥10 KiB norms) stays streamed.
    static let smallTensorResidentBytes = 256

    // Immutable layout after init.
    let options: BlockStreamingOptions
    let granuleDir: URL
    let manifest: GranuleManifest
    public let blockCount: Int
    public let groupSize: Int
    public let numGroups: Int
    let allTemplate: [GranuleTensor]  // uniform per-block tensor table (manifest order)

    // Partitioned at bind (needs computeDtype): slot-STREAMED tensors vs
    // bind-time RESIDENTS. A tensor streams iff it is ≥ the small threshold AND
    // its granule bytes are usable as-is — i.e. its dtype equals computeDtype,
    // or it is a quant component (packed weight / scales / biases stay raw by
    // the same rule as `DiT.init`). Plain params stored in a different dtype
    // (the six fp32 AdaLN tables, ~390 KB/block) load resident WITH the cast —
    // streaming them raw would silently diverge from the resident path's
    // cast-to-computeDtype semantics (caught by the q4 parity gate).
    private(set) var template: [GranuleTensor] = []
    private(set) var residentTemplate: [GranuleTensor] = []
    private var streamedKeys: Set<String> = []
    private(set) var tensorsPerBlock = 0
    private var boundComputeDtype: DType = .bfloat16
    /// Raw streamed bytes per full sweep (bind-time residents excluded).
    public private(set) var sweepBytes = 0
    /// Resident slot footprint: 2 slots × groupSize blocks of parameters.
    public private(set) var slotResidentBytes = 0
    /// In-model key prefix for block parameters (`transformer_blocks.`) — the
    /// manifest's on-disk prefix with the outer `transformer.` stripped, exactly
    /// as `DiT.init` strips it.
    let modelBlockPrefix: String

    private let device: any MTLDevice
    private let lock = OSAllocatedUnfairLock()

    // Slot storage — allocated at bind, stable for the streamer's lifetime.
    // Flat layout: [slot][localBlock * tensorsPerBlock + t]. The arrays OWN the
    // backing memory; slotPtrs alias it (valid while the arrays live).
    private var slotArrays: [[MLXArray]] = []
    private var slotPtrs: [[UnsafeMutableRawPointer]] = []

    /// The bound DiT weight store (fallback swaps its entries; detach nils its
    /// streamer). Strong on purpose — mirrored by `store.streamer`; the cycle is
    /// broken by `fallBackResident`/`detach`, the same explicit-lifecycle
    /// contract as wan-core's expert bindings.
    private var store: DiTWeightStore?

    // IO machinery — the refill plan (raw slot pointers + granule offsets)
    // lives INSIDE this @unchecked Sendable box so the @Sendable Thread body
    // captures only it (pointers cross the thread behind the free/ready
    // semaphore protocol).
    private final class IOState: @unchecked Sendable {
        struct Refill {
            let fd: Int32  // granule file of the block
            let dst: UnsafeMutableRawPointer  // slot tensor backing
            let offset: Int
            let nbytes: Int
        }

        let fds: [Int32]
        let plan: [[[Refill]]]  // [slot][group][refill]
        let free: [DispatchSemaphore]
        let ready: [DispatchSemaphore]
        let done = DispatchSemaphore(value: 0)
        let lock = OSAllocatedUnfairLock()
        var cancelled = false
        var ioBusySeconds: Double = 0
        var ioBytes: Int = 0
        var failure: String? = nil

        init(fds: [Int32], plan: [[[Refill]]]) {
            self.fds = fds
            self.plan = plan
            // Created at 0 and primed by signal() so dealloc is legal at any
            // rest value (a DispatchSemaphore whose value sits below its
            // creation value at dealloc traps in libdispatch).
            self.free = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
            self.ready = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
            free[0].signal()
            free[1].signal()
        }

        func note(busy: Double, bytes: Int) {
            lock.lock()
            ioBusySeconds += busy
            ioBytes += bytes
            lock.unlock()
        }

        var isCancelled: Bool {
            lock.lock()
            defer { lock.unlock() }
            return cancelled
        }

        func cancel() {
            lock.lock()
            cancelled = true
            lock.unlock()
        }
    }

    private var io: IOState?
    private var computeSeq = 0  // compute-side cyclic sequence (compute thread only)

    // Gate state (compute thread only, except `verdict` reads).
    private var _verdict: Verdict = .undecided
    private var _gateReport: LTXStreamerGateReport? = nil
    private var forwardCompute: Double = 0
    private var forwardStall: Double = 0
    private var forwardTokens: Int = 0
    private var forwardGating = true
    private var groupOpenedAt: Double = 0

    /// Warmup / calibration sweeps set this so their pathological N never feeds
    /// the gate. (`DiT.warmup` manages it around its call.)
    public var gateSuspended = false

    /// Defer the gate verdict until the first forward with at least this many
    /// tokens (two-stage: the pipeline sets the LARGEST stage's tokens before
    /// stage 1 runs). 0 = the wan semantics: first un-suspended forward gates.
    public var gateEvaluationThresholdTokens = 0

    public var verdict: Verdict {
        lock.lock()
        defer { lock.unlock() }
        return _verdict
    }

    public var gateReport: LTXStreamerGateReport? {
        lock.lock()
        defer { lock.unlock() }
        return _gateReport
    }

    /// Cumulative refill bandwidth over the current activation (the measured S).
    public var measuredSGiBs: Double {
        guard let io else { return 0 }
        io.lock.lock()
        defer { io.lock.unlock() }
        return io.ioBusySeconds > 0
            ? Double(io.ioBytes) / io.ioBusySeconds / 1_073_741_824 : 0
    }

    // MARK: - Init / bind

    public init(granuleDir: URL, options: BlockStreamingOptions = .init()) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw LTXBlockStreamerError.noMetalDevice
        }
        self.device = device
        self.options = options
        self.granuleDir = granuleDir
        self.manifest = try GranuleManifest.load(from: granuleDir)

        self.blockCount = manifest.blockCount
        self.groupSize = options.groupSize
        guard groupSize > 0, blockCount % groupSize == 0 else {
            throw LTXBlockStreamerError.contract(
                "groupSize \(groupSize) must divide blockCount \(blockCount)")
        }
        self.numGroups = blockCount / groupSize
        // The dict binds block i to slot (i/G)%2 while compute acquires slot
        // computeSeq%2 with computeSeq advancing CONTINUOUSLY across forwards
        // (step-major). Those stay in phase only when numGroups is EVEN — an
        // odd group count flips the parity every sweep and compute would read
        // the slot the IO thread is refilling. 48/G is even for G ∈ {1,2,3,4,
        // 6,8,12,24}; the guard exists for tiny synthetic models.
        guard numGroups % 2 == 0 else {
            throw LTXBlockStreamerError.contract(
                "numGroups \(numGroups) must be even (slot parity) — adjust groupSize")
        }
        self.allTemplate = manifest.blocks[0].tensors
        let onDisk = manifest.blockPrefix
        self.modelBlockPrefix =
            onDisk.hasPrefix("transformer.")
            ? String(onDisk.dropFirst("transformer.".count)) : onDisk
    }

    public func blockRange(_ group: Int) -> Range<Int> {
        (group * groupSize)..<((group + 1) * groupSize)
    }

    /// Build the slot-backed weight dict for a streaming `DiT` (the LTX bind).
    ///
    /// - Provenance-checks the granule tree against `checkpoint` (stale-tree
    ///   refusal — the wan consumer-wiring lesson).
    /// - Allocates the two slots, verifies the alias pointers, and maps every
    ///   `transformer_blocks.<i>.<suffix>` key onto its slot array.
    /// - Loads the NON-block ("global") tensors resident from the checkpoint on
    ///   the CPU stream (cold-load watchdog discipline), applying `DiT.init`'s
    ///   exact prefix-strip and quant-aware dtype cast.
    ///
    /// Returns the store this streamer is now bound to (`store.streamer === self`).
    func bindStore(checkpoint: URL, computeDtype: DType) throws -> DiTWeightStore {
        guard store == nil else {
            throw LTXBlockStreamerError.state("bindStore() called twice")
        }
        try manifest.validateProvenance(against: checkpoint)
        boundComputeDtype = computeDtype

        // Partition the uniform block table (see the property comment). Quant
        // components are detected exactly as `DiT.init` does: `.scales`/
        // `.biases` suffixes, and a `.weight` with a `.scales` sibling.
        let keySet = Set(allTemplate.map(\.key))
        func isQuantComponent(_ key: String) -> Bool {
            key.hasSuffix(".scales") || key.hasSuffix(".biases")
                || (key.hasSuffix(".weight")
                    && keySet.contains(String(key.dropLast("weight".count)) + "scales"))
        }
        var streamed: [GranuleTensor] = []
        var resident: [GranuleTensor] = []
        for t in allTemplate {
            let raw = try mlxDType(safetensors: t.dtype)
            let usableAsIs = raw == computeDtype || isQuantComponent(t.key)
            if t.nbytes >= Self.smallTensorResidentBytes, usableAsIs {
                streamed.append(t)
            } else {
                resident.append(t)
            }
        }
        template = streamed
        residentTemplate = resident
        streamedKeys = Set(streamed.map(\.key))
        tensorsPerBlock = streamed.count
        sweepBytes = streamed.reduce(0) { $0 + $1.nbytes } * blockCount
        slotResidentBytes = 2 * groupSize * streamed.reduce(0) { $0 + $1.nbytes }
        // A checkpoint whose block params mostly need casting (e.g. a bf16
        // checkpoint streamed at fp32 compute) would quietly load nearly
        // everything resident — refuse instead of "streaming" in name only.
        let totalBlockBytes = allTemplate.reduce(0) { $0 + $1.nbytes }
        guard sweepBytes >= totalBlockBytes * blockCount / 2 else {
            throw LTXBlockStreamerError.contract(
                "only \(sweepBytes / max(blockCount, 1)) of \(totalBlockBytes) bytes/block "
                    + "are streamable at computeDtype \(computeDtype) — checkpoint/compute "
                    + "dtype mismatch defeats streaming")
        }

        // Allocate slot arrays (zeros, eval'd, aliased — the prototype recipe).
        // Where page alignment permits, cross-check the pointer against the
        // exact P-C-proven `asMTLBuffer(noCopy: true)` route.
        for _ in 0..<2 {
            var arrays: [MLXArray] = []
            var ptrs: [UnsafeMutableRawPointer] = []
            for _ in 0..<groupSize {
                for t in template {
                    let a = MLXArray.zeros(t.shape, dtype: try mlxDType(safetensors: t.dtype))
                    eval(a)
                    let ptr = try backingPointer(a, context: "slot \(t.key)")
                    if UInt(bitPattern: ptr) % 16384 == 0, t.nbytes % 16384 == 0,
                        let buf = a.asMTLBuffer(device: device, noCopy: true),
                        buf.contents() != ptr
                    {
                        throw LTXBlockStreamerError.pointerInstability(
                            "asData vs asMTLBuffer disagree for \(t.key)")
                    }
                    arrays.append(a)
                    ptrs.append(ptr)
                }
            }
            slotArrays.append(arrays)
            slotPtrs.append(ptrs)
        }

        // Block keys → slot arrays: block i uses slot (i/G)%2, local i%G.
        var w: [String: MLXArray] = [:]
        for i in 0..<blockCount {
            let slot = (i / groupSize) % 2
            let base = (i % groupSize) * tensorsPerBlock
            for (t, entry) in template.enumerated() {
                w["\(modelBlockPrefix)\(i).\(entry.key)"] = slotArrays[slot][base + t]
            }
        }

        // Bind-time residents (sub-threshold + cast-needing tensors): loaded via
        // the COPYING constructor (never the alias — see smallTensorResidentBytes),
        // with plain params cast to computeDtype exactly as `DiT.init` casts them.
        var blockResidentBytes = 0
        if !residentTemplate.isEmpty {
            for block in manifest.blocks {
                let fd = try GranuleIO.openRead(
                    granuleDir.appendingPathComponent(block.file).path, noCache: false)
                defer { close(fd) }
                for t in block.tensors where !streamedKeys.contains(t.key) {
                    var bytes = Data(count: t.nbytes)
                    try bytes.withUnsafeMutableBytes { raw in
                        try GranuleIO.preadFull(
                            fd: fd, into: raw.baseAddress!, count: t.nbytes, offset: t.offset)
                    }
                    let raw = try mlxDType(safetensors: t.dtype)
                    var a = MLXArray(bytes, t.shape, dtype: raw)
                    if !isQuantComponent(t.key), raw != computeDtype {
                        a = a.asType(computeDtype)
                    }
                    eval(a)
                    w["\(modelBlockPrefix)\(block.index).\(t.key)"] = a
                    blockResidentBytes += t.nbytes
                }
            }
        }

        // Globals: resident load off the checkpoint, CPU stream, then the same
        // strip + quant-aware cast as `DiT.init` (packed quantized weights and
        // their scales/biases stay raw; plain params cast to computeDtype).
        let globalSet = Set(manifest.globalKeys)
        let globals = try Device.withDefaultDevice(.cpu) {
            let all = try MLX.loadArrays(url: checkpoint)
            var picked: [String: MLXArray] = [:]
            for (k, v) in all where globalSet.contains(k) {
                let stripped =
                    k.hasPrefix("transformer.") ? String(k.dropFirst("transformer.".count)) : k
                picked[stripped] = v
            }
            for key in picked.keys {
                if key.hasSuffix(".scales") || key.hasSuffix(".biases") { continue }
                if key.hasSuffix(".weight"),
                    picked[String(key.dropLast("weight".count)) + "scales"] != nil
                { continue }
                picked[key] = picked[key]!.asType(computeDtype)
            }
            eval(Array(picked.values))
            return picked
        }
        for (k, v) in globals { w[k] = v }

        // Pointer stability re-verified through the dict (the plan's
        // requirement): every block entry must alias the recorded slot pointer.
        for i in 0..<blockCount {
            let slot = (i / groupSize) % 2
            let base = (i % groupSize) * tensorsPerBlock
            for (t, entry) in template.enumerated() {
                let key = "\(modelBlockPrefix)\(i).\(entry.key)"
                let ptr = try backingPointer(w[key]!, context: "\(key) post-bind")
                guard ptr == slotPtrs[slot][base + t] else {
                    throw LTXBlockStreamerError.pointerInstability("block \(i) \(entry.key)")
                }
            }
        }

        let bound = DiTWeightStore(w: w)
        bound.streamer = self
        store = bound
        lock.lock()
        _verdict = options.gatePolicy == .forceStream ? .streaming : .undecided
        lock.unlock()
        say(String(
            format: "bound %d blocks · group=%d · slots 2×%.0f MiB (%.2f GiB resident) "
                + "· sweep %.2f GiB · globals %d · block-resident %d KiB (%d tensors/block)",
            blockCount, groupSize,
            Double(slotResidentBytes) / 2 / 1_048_576,
            Double(slotResidentBytes) / 1_073_741_824,
            Double(sweepBytes) / 1_073_741_824,
            globals.count, blockResidentBytes / 1024, residentTemplate.count))
        return bound
    }

    // MARK: - Activation and the IO thread

    /// (Re)start the prefetch thread at group 0. Called automatically by the
    /// streamed forward path; idempotent while IO is live.
    func ensureActive() {
        if io != nil { return }
        startIO()
    }

    private func startIO() {
        precondition(io == nil)
        let fds: [Int32] = manifest.blocks.map { block in
            let path = granuleDir.appendingPathComponent(block.file).path
            guard let fd = try? GranuleIO.openRead(path) else {
                fatalError("LTXBlockStreamer: cannot open granule \(path)")
            }
            return fd
        }
        // Precompute the full refill plan: [slot][group] → the preads that fill
        // that group's blocks into that slot's tensor backings.
        var plan: [[[IOState.Refill]]] = []
        for slot in 0..<2 {
            var groups: [[IOState.Refill]] = []
            for g in 0..<numGroups {
                var refills: [IOState.Refill] = []
                for lb in 0..<groupSize {
                    let bi = g * groupSize + lb
                    // Streamed tensors only — the rest are bind-time residents.
                    // The filtered order matches `template` (uniform tables),
                    // so index t aligns with the slot layout.
                    let streamed = manifest.blocks[bi].tensors.filter {
                        streamedKeys.contains($0.key)
                    }
                    for (t, entry) in streamed.enumerated() {
                        refills.append(
                            IOState.Refill(
                                fd: fds[bi],
                                dst: slotPtrs[slot][lb * tensorsPerBlock + t],
                                offset: entry.offset,
                                nbytes: entry.nbytes))
                    }
                }
                groups.append(refills)
            }
            plan.append(groups)
        }
        let state = IOState(fds: fds, plan: plan)
        io = state
        computeSeq = 0

        let numGroups = self.numGroups
        let thread = Thread {
            var seq = 0
            while !state.isCancelled {
                let slot = seq % 2
                state.free[slot].wait()
                if state.isCancelled { break }
                let g = seq % numGroups
                let t0 = CFAbsoluteTimeGetCurrent()
                var bytes = 0
                for refill in state.plan[slot][g] {
                    do {
                        try GranuleIO.preadFull(
                            fd: refill.fd, into: refill.dst,
                            count: refill.nbytes, offset: refill.offset)
                    } catch {
                        state.lock.lock()
                        state.failure = "\(error)"
                        state.cancelled = true
                        state.lock.unlock()
                        // Signal ready so a waiting compute thread can observe
                        // the failure instead of deadlocking.
                        state.ready[slot].signal()
                        state.done.signal()
                        return
                    }
                    bytes += refill.nbytes
                }
                state.note(busy: CFAbsoluteTimeGetCurrent() - t0, bytes: bytes)
                state.ready[slot].signal()
                seq += 1
            }
            state.done.signal()
        }
        thread.name = "ltx2.LTXBlockStreamer.io"
        thread.qualityOfService = .userInitiated
        thread.start()
        say("streaming from \(granuleDir.lastPathComponent)")
    }

    private func stopIO() {
        guard let state = io else { return }
        state.cancel()
        // Unblock a thread parked on either free semaphore, then wait it out.
        state.free[0].signal()
        state.free[1].signal()
        state.done.wait()
        for fd in state.fds { close(fd) }
        io = nil
    }

    /// Stop the prefetch thread and close granule files. Slots and the dict
    /// binding stay — a later forward reactivates. Call at the end of a
    /// generation (or before releasing the streamer).
    public func finish() {
        stopIO()
    }

    deinit {
        stopIO()
    }

    // MARK: - Group window (compute side)

    /// Wait until the current group's slot is refilled. Returns the slot index.
    @discardableResult
    func acquireGroup() -> Int {
        guard let state = io else {
            fatalError("LTXBlockStreamer: acquireGroup with no active IO")
        }
        let slot = computeSeq % 2
        let t0 = CFAbsoluteTimeGetCurrent()
        state.ready[slot].wait()
        if let failure = state.failure {
            fatalError("LTXBlockStreamer IO failed: \(failure)")
        }
        if let poison = pendingPoison, poison.target == acquireCount {
            // Negative-control hook: corrupt the slot AFTER its prefetch
            // completed and BEFORE compute reads it (the prototype's control).
            let entry = template[poison.tensor]
            memset(
                slotPtrs[slot][poison.localBlock * tensorsPerBlock + poison.tensor],
                0x55, min(poison.bytes, entry.nbytes))
            pendingPoison = nil
            say("poisoned slot \(slot) at acquire #\(acquireCount) (negative control)")
        }
        acquireCount += 1
        let now = CFAbsoluteTimeGetCurrent()
        forwardStall += now - t0
        groupOpenedAt = now
        return slot
    }

    /// Hand the group's slot back to the IO thread. The caller MUST have
    /// `eval`'d everything that reads the slot's arrays before calling this.
    func releaseGroup() {
        guard let state = io else { return }
        let slot = computeSeq % 2
        forwardCompute += CFAbsoluteTimeGetCurrent() - groupOpenedAt
        state.free[slot].signal()
        computeSeq += 1
    }

    // MARK: - Gate

    func beginForward(tokens: Int) {
        forwardCompute = 0
        forwardStall = 0
        forwardTokens = tokens
        forwardGating = !gateSuspended
    }

    /// Per-forward wrap-up: on the first GATING forward at or above the token
    /// threshold, evaluate the runtime gate from in-regime measurements and
    /// (policy `.auto`) fall back to fully resident when it doesn't clear.
    func endForward() {
        lastForwardComputeSeconds = forwardCompute
        lastForwardStallSeconds = forwardStall
        guard verdict == .undecided, forwardGating,
            forwardTokens >= gateEvaluationThresholdTokens
        else { return }

        let s = measuredSGiBs
        let c = forwardCompute > 0
            ? Double(sweepBytes) / forwardCompute / 1_073_741_824 : .infinity
        let nMin = s > 0 ? Int((Double(forwardTokens) * c / s).rounded(.up)) : Int.max
        let clears = s >= c * options.gateMargin
        let report = LTXStreamerGateReport(
            n: forwardTokens, sGiBs: s, cGiBs: c, nMin: nMin,
            step1ComputeSeconds: forwardCompute,
            step1StallSeconds: forwardStall,
            streaming: clears)
        say(String(
            format: "gate: N=%d · S=%.2f GiB/s vs C(N)=%.2f GiB/s → N_min≈%d → %@",
            report.n, s, c, nMin,
            clears ? "STREAM (hidden)" : "FALL BACK (resident)"))

        lock.lock()
        _gateReport = report
        _verdict = clears ? .streaming : .fellBack
        lock.unlock()

        if !clears {
            do {
                try fallBackResident()
            } catch {
                // Fallback is a strictly-safer path; failing it is fatal-worthy,
                // but streaming remains correct — keep streaming and say so.
                say("fallback FAILED (\(error)) — continuing streamed")
                lock.lock()
                _verdict = .streaming
                lock.unlock()
            }
        }
    }

    public private(set) var lastForwardComputeSeconds: Double = 0
    public private(set) var lastForwardStallSeconds: Double = 0

    // MARK: - Fully-resident fallback

    /// Load every block resident from the granule files, swap the dict entries
    /// through the bound store, and detach. The already-computed streamed
    /// artifacts (earlier steps' latents) are bit-exact and remain valid.
    /// After this the streamer holds ~slotResidentBytes of dead slots — release
    /// it (the bernini slot-stranding lesson).
    public func fallBackResident() throws {
        guard let store else {
            throw LTXBlockStreamerError.state("fallBackResident with no bound store")
        }
        stopIO()
        let t0 = CFAbsoluteTimeGetCurrent()
        var loadedBytes = 0
        for (i, granule) in manifest.blocks.enumerated() {
            let fd = try GranuleIO.openRead(
                granuleDir.appendingPathComponent(granule.file).path)
            defer { close(fd) }
            var arrays: [MLXArray] = []
            for t in granule.tensors {
                // One-time bulk load: the COPYING constructor, never the alias
                // (which is both unnecessary here and unsafe for tiny tensors —
                // see smallTensorResidentBytes). Bind-time residents are
                // already in the dict, but reloading them is harmless and
                // keeps this loop total. Plain params get the same cast to
                // computeDtype as `DiT.init` (and as bind applied).
                var bytes = Data(count: t.nbytes)
                try bytes.withUnsafeMutableBytes { raw in
                    try GranuleIO.preadFull(
                        fd: fd, into: raw.baseAddress!, count: t.nbytes, offset: t.offset)
                }
                let raw = try mlxDType(safetensors: t.dtype)
                var a = MLXArray(bytes, t.shape, dtype: raw)
                let quantComponent =
                    t.key.hasSuffix(".scales") || t.key.hasSuffix(".biases")
                    || (t.key.hasSuffix(".weight")
                        && granule.tensors.contains {
                            $0.key == String(t.key.dropLast("weight".count)) + "scales"
                        })
                if !quantComponent, raw != boundComputeDtype {
                    a = a.asType(boundComputeDtype)
                }
                store.w["\(modelBlockPrefix)\(i).\(t.key)"] = a
                arrays.append(a)
                loadedBytes += t.nbytes
            }
            eval(arrays)
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        say(String(
            format: "fell back resident: %.2f GiB in %.1fs (%.2f GiB/s)",
            Double(loadedBytes) / 1_073_741_824, dt,
            Double(loadedBytes) / dt / 1_073_741_824))
        store.streamer = nil
        self.store = nil
        lock.lock()
        _verdict = .fellBack
        lock.unlock()
    }

    // MARK: - Test / receipt hooks

    private struct PendingPoison {
        let target: Int  // absolute acquire index (never resets)
        let localBlock: Int
        let tensor: Int
        let bytes: Int
    }
    private var pendingPoison: PendingPoison? = nil
    /// Monotonic count of group acquisitions across the streamer's lifetime —
    /// the poison hook's clock. Compute-thread only.
    private var acquireCount = 0

    /// Arm a one-shot poisoned-slot negative control (a parity compare that
    /// cannot see a bad refill is not evidence): on the `afterAcquires`-th
    /// group acquisition from now (0 = the very next one), `bytes` of the given
    /// tensor are corrupted after the prefetch and before compute reads them.
    public func armPoison(
        afterAcquires: Int, localBlock: Int = 0, tensor: Int = 0, bytes: Int = 1 << 20
    ) {
        pendingPoison = PendingPoison(
            target: acquireCount + afterAcquires, localBlock: localBlock,
            tensor: tensor, bytes: bytes)
    }

    /// Stop IO and detach from the bound store WITHOUT loading anything
    /// resident — the receipts' between-arms reset. Block dict entries keep
    /// aliasing this streamer's slots, so no forward may run between `detach()`
    /// and releasing the DiT.
    public func detach() {
        stopIO()
        store?.streamer = nil
        store = nil
    }

    private func say(_ message: String) {
        if !options.quiet {
            print("[LTXBlockStreamer] \(message)")
            fflush(stdout)
        }
    }
}
