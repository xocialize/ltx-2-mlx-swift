// LTXStreaming.swift — the LTX-2.3 adapter over BlockStreamKit
// (mlx-block-stream-swift; NEUROSTREAM-ACTIONS HV2, extraction phase).
//
// The in-tree GranuleLayout/BlockStreamer implementation moved into the shared
// substrate after this second consumer pinned the seam (extraction gated by
// the receipts in probes/hv2_ltx_blockstreamer.out — the stream gates re-ran
// green over this adapter, byte-identical behavior). What stays HERE is the
// irreducibly LTX-specific ~20%:
//   - the checkpoint dialect (outer `transformer.` strip, `ltxBlockPrefix`,
//     the BasicAVTransformerBlock forward-order ranking for granule locality),
//   - the DICT-shaped bind: constructing `DiTWeightStore.w` over the kit's
//     slot-backed arrays + loading the 58 resident globals with `DiT.init`'s
//     exact quant-aware cast,
//   - fallback/detach wiring through the store (swap dict entries, nil the
//     streamer),
//   - and the block-loop routing itself, which lives in `DiT.callAsFunction`.

import Foundation
import MLX
import BlockStreamKit

/// Re-exported so gates/consumers keep one import.
public typealias BlockStreamingOptions = BlockStreamKit.BlockStreamingOptions
public typealias LTXStreamerGateReport = BlockStreamKit.StreamGateReport
public typealias GranuleManifest = BlockStreamKit.GranuleManifest

public enum LTXGranuleLayout {
    /// The LTX-2.3 checkpoints' on-disk per-block prefix (`DiT.init` strips
    /// the outer `transformer.`; granules are laid out from on-disk keys).
    public static let ltxBlockPrefix = "transformer.transformer_blocks."

    /// BasicAVTransformerBlock forward order (DiT.swift `block(_:)`): AdaLN
    /// tables first, then the eight attention/FF stages in execution order.
    /// Locality only — correctness never depends on it.
    static let forwardOrderPrefixes: [String] = [
        "scale_shift_table_a2v_ca_video",
        "scale_shift_table_a2v_ca_audio",
        "scale_shift_table",
        "audio_scale_shift_table",
        "prompt_scale_shift_table",
        "audio_prompt_scale_shift_table",
        "attn1.",
        "audio_attn1.",
        "attn2.",
        "audio_attn2.",
        "audio_to_video_attn.",
        "video_to_audio_attn.",
        "ff.",
        "audio_ff.",
    ]

    public static func forwardRank(_ key: String) -> Int {
        for (i, p) in forwardOrderPrefixes.enumerated() where key.hasPrefix(p) { return i }
        return forwardOrderPrefixes.count
    }

    @discardableResult
    public static func write(
        safetensors: URL, outputDir: URL,
        blockPrefix: String = ltxBlockPrefix,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> GranuleLayout.Result {
        try GranuleLayout.write(
            safetensors: safetensors, outputDir: outputDir, blockPrefix: blockPrefix,
            orderRank: forwardRank, progress: progress)
    }
}

/// The LTX streaming plane: a thin adapter that owns the kit core plus the
/// dict-shaped bind. API-compatible with the pre-extraction in-tree
/// implementation (DiT, pipeline, and the stream gates are unchanged).
public final class LTXBlockStreamer: @unchecked Sendable {

    public typealias Verdict = BlockStreamer.Verdict

    public let core: BlockStreamer
    /// In-model key prefix (`transformer_blocks.`) — the on-disk prefix with
    /// the outer `transformer.` stripped, exactly as `DiT.init` strips it.
    let modelBlockPrefix: String
    private weak var boundStore: DiTWeightStore?

    public init(granuleDir: URL, options: BlockStreamingOptions = .init()) throws {
        self.core = try BlockStreamer(granuleDir: granuleDir, options: options)
        let onDisk = core.manifest.blockPrefix
        self.modelBlockPrefix =
            onDisk.hasPrefix("transformer.")
            ? String(onDisk.dropFirst("transformer.".count)) : onDisk
    }

    // MARK: forwarded surface (kept 1:1 with the pre-extraction API)

    public var blockCount: Int { core.blockCount }
    public var numGroups: Int { core.numGroups }
    public var sweepBytes: Int { core.sweepBytes }
    public var slotResidentBytes: Int { core.slotResidentBytes }
    public var verdict: Verdict { core.verdict }
    public var gateReport: LTXStreamerGateReport? { core.gateReport }
    public var measuredSGiBs: Double { core.measuredSGiBs }
    public var lastForwardComputeSeconds: Double { core.lastForwardComputeSeconds }
    public var lastForwardStallSeconds: Double { core.lastForwardStallSeconds }
    public var gateSuspended: Bool {
        get { core.gateSuspended }
        set { core.gateSuspended = newValue }
    }
    public var gateEvaluationThresholdTokens: Int {
        get { core.gateEvaluationThresholdTokens }
        set { core.gateEvaluationThresholdTokens = newValue }
    }

    public func blockRange(_ group: Int) -> Range<Int> { core.blockRange(group) }
    public func ensureActive() { core.ensureActive() }
    @discardableResult public func acquireGroup() -> Int { core.acquireGroup() }
    public func releaseGroup() { core.releaseGroup() }
    public func beginForward(tokens: Int) { core.beginForward(tokens: tokens) }
    public func endForward() { core.endForward() }
    public func finish() { core.finish() }
    public func armPoison(
        afterAcquires: Int, localBlock: Int = 0, tensor: Int = 0, bytes: Int = 1 << 20
    ) {
        core.armPoison(
            afterAcquires: afterAcquires, localBlock: localBlock, tensor: tensor, bytes: bytes)
    }
    public func fallBackResident() throws { try core.fallBackResident() }
    public func detach() { core.detach() }

    // MARK: - The LTX bind (dict construction + globals)

    /// Build the slot-backed weight dict for a streaming `DiT`: kit `bind`
    /// (provenance check, slots, bind-resident/cast partition) → dict entries
    /// under `transformer_blocks.<i>.` → resident globals with `DiT.init`'s
    /// exact prefix-strip and quant-aware cast → kit pointer re-verification
    /// through the dict. Returns the store this streamer is bound to.
    func bindStore(checkpoint: URL, computeDtype: DType) throws -> DiTWeightStore {
        guard boundStore == nil else {
            throw BlockStreamError.state("bindStore() called twice")
        }
        var w: [String: MLXArray] = [:]
        let prefix = modelBlockPrefix
        let store = DiTWeightStore(w: [:])

        let blocks = try core.bind(
            computeDtype: computeDtype,
            provenanceCheckpoint: checkpoint,
            installResident: { [weak store] i, pairs in
                // .auto fallback: swap the dict entries for block i resident.
                guard let store else { return }
                for (key, array) in pairs {
                    store.w["\(prefix)\(i).\(key)"] = array
                }
            },
            onDetach: { [weak store] in
                store?.streamer = nil
            })
        for block in blocks {
            for (key, array) in block.arrays {
                w["\(prefix)\(block.block).\(key)"] = array
            }
        }

        // Globals: resident load off the checkpoint, CPU stream (cold-load
        // watchdog discipline), same strip + quant-aware cast as `DiT.init`.
        let globalSet = Set(core.manifest.globalKeys)
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

        store.w = w
        store.streamer = self
        boundStore = store

        try core.verifyInstalled { [weak store] i, key in
            store?.w["\(prefix)\(i).\(key)"]
        }
        return store
    }
}
