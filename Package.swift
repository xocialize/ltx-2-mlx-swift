// swift-tools-version: 6.2
// ltx-2-mlx-swift — Swift/MLX port of Lightricks LTX-2.3 (DiT-based joint
// audio+video foundation model). Mirrors the Python oracle `ltx-2-mlx`
// (github.com/dgrauet/ltx-2-mlx). Unlike the Wan family this is a STANDALONE
// substrate (Gemma-3 text encoder, 128-ch VAE, joint-AV DiT, BigVGAN+BWE audio)
// — NOT a wan-core consumer.
//
// LICENSE POSTURE: LTX-2 Community License → EVAL-ONLY GATED SPECIALTY. Not
// shippable; never wired into a commercial product. See the memory note
// `ltx-2.3-swift-port-active` and EngineeringDocs/apple_native_berlini_opportunities.md.
//
// Gemma text encoder = REUSE of mlx-swift-lm's Gemma3TextModel (Path A). The
// 49-layer hidden-state extraction needs an `allHiddenStates` method that the
// stock model doesn't expose, added in our fork
// (github.com/xocialize/mlx-swift-lm @ ltx/gemma-all-hidden-states) — wired here
// as a local path dep during development.

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
        // Fork with the Gemma `allHiddenStates` text-encoder extension.
        .package(path: "../mlx-swift-lm"),
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
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.9.1"),
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
                // Gemma 3 text encoder (Path A reuse) + the allHiddenStates seam.
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
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
            ],
            path: "Sources/MLXLTX2",
            resources: [.process("Resources")]  // ltx-lora-registry.json → Bundle.module
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
            ],
            path: "Tests/MLXLTX2Tests"
        ),
    ]
)
