import Foundation
import LTX2
import MLXToolKit

/// MLXEngine package: Lightricks **LTX-2.5** distilled — a SEPARATE `ModelPackage` from
/// `MLXLTX2Package` (2.3), registered under the same `.textToVideo` capability so the engine picks
/// between them by `PackageID`.
///
/// **Why separate rather than a revision of the 2.3 package.** The two share every line of run
/// logic — which is why this type owns none of it and forwards to an inner `MLXLTX2Package` — but
/// they share no *numbers*. 2.5's bf16 DiT floor is ~40 GB against 2.3's ~38.85 with a bf16 12B
/// encoder rather than a 4-bit one, and the tier scopes differ: 2.3 ships four profiles, 2.5 ships
/// `standard64` + `max128` only (AB-D-0014, re-confirmed AB-D-0015). One footprint table cannot
/// state both honestly, and a manifest that over-declares is how a tier gets admitted that should
/// have been refused.
///
/// The GENERATION itself is still detected from the checkpoint at load time
/// (`LTX2Pipeline.isLTX25` keys off the in-dir `gemma4-12b-ltx-v1/`, never a path-name parse). The
/// `family` axis on the configuration only decides what cannot wait for the weights: default repo,
/// which sibling repo carries a quantized DiT, and where the encoder lives.
///
/// LICENSE — identical two-layer declaration to 2.3: weights LTX-2 Community, port code Apache-2.0.
@InferenceActor
public final class MLXLTX25Package: ModelPackage {
    public typealias Configuration = LTX2Configuration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(
                weightLicense: .ltx2Community,
                portCodeLicense: .apache2),
            provenance: Provenance(
                // Origin, not fetch location. ⟲ **CORRECTED 2026-08-21**: this read
                // `dgrauet/ltx-2.5-mlx`, describing a mirror-of-a-conversion. That repo does not
                // exist (404 with a valid token), and the description was stale — **our 2.5 port
                // is FROM ORIGIN**, which the published tree states itself:
                // `mlx-community/ltx-2.5-mlx` declares `base_model: Lightricks/LTX-2.5`.
                // The origin/fetch distinction the old comment defended is still right and still
                // enforced — origin is Lightricks, fetch is `LTX2Configuration.repo` — it was
                // simply pointed at the wrong origin, which under-credited our own work AND named
                // an artifact nobody can retrieve.
                sourceRepo: "Lightricks/LTX-2.5",
                revision: "main",
                tier: 3),  // multi-component pipeline (Gemma-4 + connector + DiT + VAEs + vocoder)
            requirements: RequirementsManifest(
                // SPLIT FOOTPRINT (contract 1.14.0) — MEASURED 2026-08-14 at 704×512, two-stage,
                // in the regime the engine actually runs: UNPROFILED + `LTX_EVICT_DIT=1` +
                // `LTX_CACHE_LIMIT_GB=2`. Receipts AB-R-0039 / AB-R-0041; harness `--mem-bench25`.
                //
                //   quant   phys-after-load      PEAK              → declared resident / activation
                //           121f     161f        121f     161f
                //   bf16    39.92    39.89       43.11    43.26      40 GB / 6 GB
                //   int8    22.09    22.09       25.60    25.55      23 GB / 5 GB
                //
                // 161f is `standard64.maxFrames`, so the declaration now covers that profile's FULL
                // envelope rather than a point inside it.
                //
                // 🔑 **Declared in the EVICTED regime on purpose.** The footprint SHAPE INVERTS
                // under eviction — unevicted, resident is 40.34 GB and activation 22.45; evicted it
                // is 2.16 / 40.26 (AB-R-0034). Eviction is default-on for every tier
                // (`LTX2Profile.evictDiTBeforeDecode`), so declaring from an unevicted run would
                // overstate the resident floor ~19×. These numbers are the shipping regime.
                //
                // 🔑 **PEAK IS FLAT ACROSS THE WHOLE TESTED ENVELOPE** — 9f→121f is +0.57 GB on
                // BOTH quants, 121f→161f is +0.15 (bf16) and −0.05 (int8), and a 2.4× area increase
                // is +0.28. 2.5 is weight-dominated, not activation-dominated: the transient is
                // small and the weight floor is the whole story.
                // ⚠️ Tested envelope is ≤704×512×161f. **`max128.maxFrames` is 481 and is NOT
                // covered** — flatness over 9→161f is strong evidence, but 481f is a 3× extension
                // beyond anything measured on 2.5 and this file has just finished replacing exactly
                // that kind of extrapolation with a measurement. bf16's charge leaves 62 GB of
                // max128 headroom, so the risk is low; the CLAIM is still unmade.
                //
                // 🚨 **`residentBytes` comes from phys-after-load, NOT from the bench's "resident
                // floor" line — and that line is actively misleading.** It reads ~11.4 GB for BOTH
                // quants at 121f and ~2.4 GB for both at 161f: a **9 GB swing between adjacent
                // geometries**, on both arms, while phys-after-load moves 0.00–0.03 GB. It is one
                // post-`clearCache` sample of retained mmap pages under eviction, not a floor. The
                // harness's own `DECLARE →` line is computed from it, so following that suggestion
                // would declare a 2.4 GB resident for a 40 GB checkpoint. Only PEAK and
                // phys-after-load are arm-attributable (AB-R-0039, AB-R-0041).
                //
                // 🔑 **CONSEQUENCE, and it is intended: bf16 does NOT fit `standard64`.** The
                // declared charge (resident + activation) is 46 GB against that tier's 44.8 GB
                // budget, so the governor refuses it there. That is honest rather than unfortunate
                // — the MEASURED peak is **43.26 GB = 97% of budget at 161f**, the profile's own
                // frame cap, so "2.5 bf16 runs on 64 GB" is not true with any margin. `standard64`
                // runs the **int8** DiT (28 GB charge, 62%); bf16 is a
                // `max128` quant on 2.5 and stays the choice wherever bit-reproducibility against
                // bf16 is required (q8 is a LEAN tier — it diverges from bf16 and was judged
                // equally good, AB-R-0038, not a faithful reproduction of it).
                // `--ltx25-package-gate` case 16b pins this so it cannot be rounded away.
                //
                // ⚠️ NO int4 ENTRY, deliberately. No q4 2.5 DiT exists and building one is
                // contraindicated (weight-only int4 on a DiT is measurable quality damage that buys
                // no speed — AB-D-0013). Declaring an int4 footprint would invite a checkpoint that
                // must not be produced. `LTXFamily.transformerRepoSuffix` returns nil for it too.
                footprints: [
                    // ⚠️ STORAGE FLOOR ON bf16 ONLY (contract 1.34.0, AB-T-0075/AB-D-0038). The
                    // field means "below this the run CRASHES", not "below this it is slow" — I9:
                    // safetensors load lazily inside live Metal command buffers, and when the bf16
                    // working set outgrows what page cache absorbs on a slow volume the fault storm
                    // trips the GPU watchdog. Measured bf16-on-USB (~250-475 MB/s) = 0/7 across
                    // three sessions; int8/int4 on the SAME volume were fine, which is why int8
                    // carries no floor. Prewarm does not save it — the run's own working set evicts
                    // the cache mid-generation.
                    // ⚠️ 1.0 GB/s is CONSERVATIVE, not measured: the band between ~475 MB/s (0/7
                    // crash) and PCI-E is untested, so this sits clearly above the known-fatal range
                    // and below any real NVMe. Narrow it only with data from that band.
                    // 🚨 This is NOT the streaming floor. Streamed tiers degrade in TIME, not
                    // safety, so declaring one here would make the engine assert a crash that does
                    // not happen — that advisory is app-side off volumeCharacterization().
                    QuantFootprint(quant: .bf16, residentBytes: 40_000_000_000,
                                   peakActivationBytes: 6_000_000_000,
                                   minSustainedReadBytesPerSecond: 1_000_000_000),
                    QuantFootprint(quant: .int8, residentBytes: 23_000_000_000,
                                   peakActivationBytes: 5_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max),
            specialties: [
                SpecialtyWeight(.general, strength: 0.6),
            ],
            surfaces: [
                T2VContract.descriptor(
                    name: "ltx-2.5-t2v",
                    summary: "Lightricks LTX-2.5 distilled text/image-to-video (MLX, research port). "
                        + "Joint audio-video DiT with a Gemma-4 12B text encoder, a 128-ch video VAE, "
                        + "and a BigVGAN/BWE audio vocoder; two-stage distilled with a latent "
                        + "upsampler. Supports i2v via `initImage` (first-frame conditioning). "
                        + "Output is an MP4 with synced 48kHz audio. Runs on 64 GB and 128 GB tiers "
                        + "only — the bf16 DiT floor puts 2.5 out of reach of 24/32 GB (AB-D-0015).")
            ]
        )
    }

    /// All behaviour is 2.3's — the pipeline is generation-agnostic and branches on the checkpoint,
    /// so duplicating `load`/`run`/`unload` here would fork a ~250-line shipping path for nothing
    /// and let the two drift. This type exists for the MANIFEST.
    private let inner: MLXLTX2Package

    /// Force the family regardless of how the configuration was built: registering this package IS
    /// the statement "this is 2.5", and a caller passing a default-constructed config would
    /// otherwise silently get 2.3's repo defaults and 2.3's `-q8` suffix — which on 2.5 resolves to
    /// the text-encoder sibling and serves a bf16 DiT under an int8 declaration.
    ///
    /// Exposed (rather than inlined into `init`) so `--ltx25-package-gate` can assert THIS function
    /// instead of re-implementing it. A gate that re-derives the behaviour it is checking passes
    /// when the shipping path is deleted — which is exactly what mutation-testing this caught.
    public nonisolated static func coerced(_ configuration: Configuration) -> Configuration {
        var cfg = configuration
        guard cfg.family != .ltx25 else { return cfg }
        let wasDefault23 = cfg.repo == LTXFamily.ltx23.defaultRepo
        cfg.family = .ltx25
        if wasDefault23 { cfg.repo = LTXFamily.ltx25.defaultRepo }
        return cfg
    }

    public nonisolated init(configuration: Configuration) {
        self.inner = MLXLTX2Package(configuration: Self.coerced(configuration))
    }

    public func load() async throws { try await inner.load() }
    public func unload() async { await inner.unload() }
    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try await inner.run(request)
    }
}
