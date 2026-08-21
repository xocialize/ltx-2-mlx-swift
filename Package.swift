// swift-tools-version: 6.2
// ltx-2-mlx-swift — Swift/MLX port of Lightricks LTX-2.5 (PRIMARY since
// 2026-08-13) and LTX-2.3 (prior work, still loaded and gated), DiT-based joint
// audio+video foundation models. Mirrors the Python oracle `ltx-2-mlx`
// (github.com/dgrauet/ltx-2-mlx). Version is resolved from the CHECKPOINT
// (`LTX2Pipeline.isLTX25` keys off the in-dir `gemma4-12b-ltx-v1/`), never a
// path-name parse, so 2.3 output stays byte-identical. Unlike the Wan family
// this is a STANDALONE substrate (Gemma-3 / Gemma-4 text encoder, 128-ch VAE,
// joint-AV DiT, BigVGAN+BWE audio) — NOT a wan-core consumer.
//
// LICENSE POSTURE (reversed 2026-06-16 — permitted by default): TWO LAYERS.
// Weights = LTX-2 Community License, on the engine `permissiveAllowlist` since
// mlx-engine-swift 0.6.0 → admitted by the default `MLXServeEngine()`. Port code
// = Apache-2.0 (Lightricks license their OWN inference code Apache-2.0, so
// inference code is not a §3 weight-derivative); repo published to
// xocialize/ltx-2-mlx-swift. Distribution ≠ admission: the weights' §2 revenue
// gate + §A.20 non-compete still bind anyone SHIPPING output. Converted weights
// and parity/goldens/ are weight-derivatives — gitignored, never committed.
// Authoritative: ../CLAUDE.md §License.
//
// Gemma text encoder = REUSE of mlx-swift-lm's Gemma3TextModel (Path A). The
// 49-layer hidden-state extraction lives in THIS package
// (Sources/LTX2/Gemma3+AllHiddenStates.swift), written against the official
// `@_spi(GemmaEncoder)` surface upstream merged in ml-explore/mlx-swift-lm#387
// (2026-07-21, `6608a35`). No mlx-swift-lm patch is carried any more — the
// checkout at ../mlx-swift-lm is a plain upstream revision, not a fork.
//
// ⚠️ TRANSITIONAL PIN. No mlx-swift-lm RELEASE TAG contains #387 yet (latest is
// 3.31.4, 2026-06-30, which predates the merge), so this is still a local path
// dep pointing at an upstream-`main` checkout. A main-SHA pin is less stable
// than a tag: DO NOT cut or ship an LTX release off this. When a tag ships,
// adoption is the ONE-LINE edit marked below — nothing else in this package
// changes, because the tap no longer depends on any local patch.

import PackageDescription

let package = Package(
    name: "LTX2",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "LTX2", targets: ["LTX2"]),
        // The MLXEngine wrapper: a conformant `ModelPackage` over the LTX2 pipeline.
        .library(name: "MLXLTX2", targets: ["MLXLTX2"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMinor(from: "0.31.3")),
        // ⬇️ THE ONE-LINE SWAP: replace with
        //      .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "<tag>")
        //    once a release tag containing #387 (`6608a35`) exists. Carries no local
        //    patches — ../mlx-swift-lm is checked out at plain upstream `main`.
        // ⚠️ Traits disabled deliberately. mlx-swift-lm's default-on
        // `FoundationModelsIntegration` trait builds MLXFoundationModels, the adapter for
        // APPLE's FoundationModels framework — which LTX does not use (we consume only
        // MLXLLM / MLXLMCommon / MLXHuggingFace, and no source here imports it).
        // Apple changed that framework's API in the macOS 27 SDK (`capabilities:` label
        // removed; `ConvertibleToGeneratedContent` replaced the old dictionary type) and
        // upstream has not caught up, so leaving the trait on fails the build of an
        // adapter we never link. Upstream anticipated exactly this and made it a trait.
        .package(path: "../mlx-swift-lm", traits: []),
        // mlx-swift-lm 3.x decoupled the HF stack — the consumer provides these for
        // the #huggingFaceLoadModel macro (same pins as mlx-qwen-llm-swift).
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.2.1"),
        // MLXEngine contract (MLXToolKit) for the wrapper target. `from: "0.9.1"` (resolves 0.14.0):
        //  • 0.6.0 revised the license stance — LTX-2-Community is on the `permissiveAllowlist`
        //    (admitted by the default `.permissiveOnly`), since Lightricks licenses their own
        //    inference code (ltx-core/ltx-pipelines) as Apache-2.0; only the weights are Community.
        //  • 0.7.0 added cold-start weight prewarm (the engine pages weight files into the OS
        //    cache before load()'s GPU evals → fixes the I5 cold-load watchdog abort). Opt-in via
        //    `LTX2Configuration: WeightPrewarming` (see LTX2Configuration.swift).
        //  • 0.14.0 (contract 1.14.0) added the split footprint (`QuantFootprint.peakActivationBytes`
        //    + `FootprintConfigured.peakActivationBytesHint`) and `BudgetAware`: the engine reserves
        //    ONE transient activation across residents (serialized inference). Adopted in the manifest
        //    (residentBytes = weight floor, peakActivationBytes = peak − floor) + the per-stage evict.
        //  • 0.25.0 (contract 1.18.0) added the run-phase progress plane (`RunPhase`/`RunProgress`,
        //    ENGINE-NEEDS V2); the wrapper forwards the core's `LTX2Progress` events into it, so
        //    this package now REQUIRES ≥0.25.0.
        //  • 0.27.0 added the CAN gate (`CancellationConformance`, CAN-1..3): entry checkpoint
        //    first act of run(), no laundering, cadence declared in Tests/CancellationTests.swift
        //    (the LTX-proven per-step/per-chunk placements, RunProgress-evidenced).
        //  • 0.32.0 (contract 1.24.0) moved first-run materialization ENGINE-side: resident()/
        //    prepare() downloads a WeightSourcing configuration's missing sources before load()
        //    runs, replacing this package's own WeightMaterializer (the original of the four
        //    per-package copies). load() now only resolves directories off the store.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.47.0"),
        // HV2 weight streaming substrate (BlockStreamKit) — the model-agnostic core
        // extracted from this package's in-tree implementation (and wan-core's) once the
        // second consumer pinned the seam. Versioned dep since 2026-08-14: the kit repo
        // went public + tagged v0.5.0, which is the condition the old path dep was
        // waiting on (a URL dep on a private repo needs git creds at SPM resolve).
        .package(url: "https://github.com/xocialize/mlx-block-stream-swift", from: "0.5.0"),
        // Shared env-gated profiling harness (timing + phys_footprint/paging instrumentation).
        // Faithful superset of the old in-tree LTX2Profiler — same manual span API + Row fields +
        // ⚠PAGING flag + CSV export, plus region/barrier closures. Env var is MLX_PROFILE (not
        // LTX_PROFILE); MLX_PROFILE=csv writes MLX_PROFILE_CSV (default /tmp/mlx-profile.csv).
        .package(url: "https://github.com/xocialize/mlx-profiling", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "LTX2",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                // Gemma 3 text encoder (Path A reuse). The allHiddenStates tap is ours
                // (Sources/LTX2/Gemma3+AllHiddenStates.swift) via @_spi(GemmaEncoder).
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
                .product(name: "BlockStreamKit", package: "mlx-block-stream-swift"),
            ],
            path: "Sources/LTX2"
        ),
        .target(
            name: "MLXLTX2",
            dependencies: [
                "LTX2",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
                // Auto-materialization (BRIDGE M4): HubClient.downloadSnapshot for first-run
                // weight downloads into the engine ModelStore layout.
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXLTX2",
            resources: [.process("Resources")]  // ltx-lora-registry.json → Bundle.module
        ),
        // HV2 weight streaming: lays a transformer checkpoint out as per-block
        // granule files for LTXBlockStreamer (STREAMING-PLAN.md). The layout code
        // itself lives in LTX2 (Sources/LTX2/Streaming/); this is the thin CLI.
        .executableTarget(
            name: "ltx-granule-layout",
            dependencies: ["LTX2"],
            path: "Sources/GranuleLayoutCLI"
        ),
        .executableTarget(
            name: "RunLTX2",
            dependencies: [
                "LTX2",
                "MLXLTX2",  // for the --encode-stress gate (calls the real encodeMP4)
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/RunLTX2"
        ),
        .testTarget(
            name: "MLXLTX2Tests",
            dependencies: [
                "MLXLTX2",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                // The engine's executable MAT gate (MaterializationConformance) — run from this
                // package's own conformance suite per the per-package convention.
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXLTX2Tests"
        ),
    ]
)
