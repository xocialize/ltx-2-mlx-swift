# CLAUDE.md — ltx-2-mlx-swift (the Swift port)

Package-level navigation. **Methodology, quirks, license, and the determinism doctrine are
in the parent [`../CLAUDE.md`](../CLAUDE.md)** (auto-loads) — don't duplicate them here.


## 🔑 LTX-2.5 IS PRIMARY (2026-08-13)

Default to 2.5. The Swift port is at FULL PARITY with the oracle — DiT keyframes embedding,
ancestral-Euler stage 1, Gemma-4 encoder, keyframe slots, DFR layout + pipeline with temporal
rounds — and generates end to end (`RunLTX2 --e2e25`, `--dfr25`). Gate board below.

2.3 still loads and still gates; every 2.5 addition is CHECKPOINT-driven (`isLTX25` keys off
the in-dir `gemma4-12b-ltx-v1/`, never a path-name parse), so 2.3 output is byte-identical —
verified: `--e2e25` reproduces its documented baseline exactly after the audio-optional
refactor touched every transformer block. Do not start new feature work on 2.3.

### The 2.5 gate board (all green)

| gate | covers |
|---|---|
| `--dit-tiny-kf25` | keyframes_abs_pos_embedding + `ff_bias:false`; fixture seeds a NON-ZERO embedding (a zero one makes the delta a no-op) |
| `--ancestral-step` | stage-1 ancestral Euler, INJECTED noise (cross-binding RNG is not bit-identical) |
| `--denoise-wiring` | that the two deltas REACH the DiT through `run`/`runConditioned` — components were once gated while nothing called them |
| `--keyframe-slots` | slot geometry, `(t+0.5)/fps` spans, seed survival at σ 0.05 vs 1.0 |
| `--dfr-layout` | canvas/tile/stitch geometry, bit-exact vs the Python port |
| `--dfr` | DFR orchestration: frame contract, trims, fps split, tile-local conditioning, per-tile seeds, clean re-blend |
| `--pipeline-25` | 2.5 detection against the REAL model dirs; 2.3 must read false |
| `--gemma-tokenizer` | tokenization by INTEGER equality (was covered by nothing) |
| `--gemma4-gate` | 49-state encoder parity, mean 0.999985 (needs the 23.8 GB encoder) |
| **`--diffvae-gate`** | the DiffVAE decoder (claim C3 / bench arm C), 6 stages in dependency order, all **cos 1.000000** vs the MLX oracle (0.999999–1.000000 vs torch) on both NA backends. ⚠️ Stage [F] injects the fixture's NOISE CANVAS — this decoder is stochastic and must NEVER be gated on pixel equality from a fresh draw (RNG streams are not bit-identical across bindings). Stage [G] closes the loophole that creates by re-decoding from a second canvas and requiring the pixels to MOVE |
| **`--duration-gate`** | the duration head (claim C6 / bench arm D), all 3 input modes on the REAL banked connector encodings. Two bars on purpose: seconds at 1e-3 relative (the 1-query/2048-key pooler amplifies fp32 accumulation to ~2e-4), the snapped FRAME COUNT at exact equality (one 24 fps grid step is 0.333 s — four orders above the residual) |
| **`--ltx25-package-gate`** | the ENGINE-side 2.5 surface, 20/20 — everything decided before any weight exists, where being wrong is silent. Load-bearing case: **2.5 int8 → `-ditq8`, NOT `-q8`** (on 2.5 that name is the int8 *text-encoder* sibling whose transformer symlinks back to bf16, so 2.3's rule declares int8 ≈23 GB and loads bf16 ≈40 GB). Case 05 READS both checkpoint headers rather than asserting the suffix. Mutation-tested ×3 — and the third mutation initially PASSED because the gate re-derived the coercion it was checking, hence `MLXLTX25Package.coerced` being exposed |

⚠️ Rows in the table below without a 2.5 marker were measured on 2.3. The doctrine carries;
the numbers do not (2.5's encoder is a bf16 12B vs 2.3's 4-bit Gemma-3).

## Source map (`Sources/LTX2/` — engine-agnostic functional cores)

| File | What | Gate |
|---|---|---|
| `RoPE.swift` | split-type RoPE (log-spaced freq grid, fractional positions) — shared by connector + DiT | (via others) |
| `GemmaEncoder.swift` | Gemma-3 load (stock mlx-swift-lm) + combined causal+padding mask + tokenize (+ `loadTokenizer` — tokenizer only, no 12B). `tokenize` is a static over a bare tokenizer so the gate can pin it cheaply | `--gemma-gate` (49 states), **`--gemma-tokenizer-gate`** (tokenization: INTEGER equality, 7 cases, no weights loaded) |
| `Gemma3+AllHiddenStates.swift` | the 49-state encoder tap itself — uniform mask on every layer, per-layer `eval` (watchdog), per-layer `Task.checkCancellation()`. Ours, via `@_spi(GemmaEncoder) import MLXLLM` | `--gemma-gate` |
| `Connector.swift` | `GemmaFeaturesExtractorV2` + `Embeddings1DConnector` (49-layer RMS, dual project, gated attn, GEGLU, registers) — **fp32** | `--connector-gate` |
| `DiT.swift` | joint-AV Diffusion Transformer (48 blocks, AdaLN ×4 kinds, self/text-cross/AV-cross attn) — **bf16**. Quant-aware `dense()` (q8/q4, bits auto). Optional **per-token timesteps** (i2v). ⚠️ `audioLatent: nil` is the **AUDIO-FREE forward** (oracle `run_ax`; upstream's `ax.numel() > 0`) that DFR rounds need — audio SA/text-CA/FF and BOTH AV-cross directions are skipped, so the VIDEO genuinely loses its A2V residual. That is upstream's behaviour, not an approximation. | `--dit-tiny`, `--dit-full`, `--dit-q8`, `--dit-q4`, `--dit-pertoken`, `--dfr-gate` (audio-free arm) |
| `DenoiseLoop.swift` | distilled Euler (X0Model + euler_step). `run` = uniform-mask t2v; **`runConditioned`** = i2v (per-token σ + clean-latent re-blend). Both take `audioLatent0: nil` for the audio-free loop, and the ancestral key stream then SKIPS the audio split so it advances as a video-only stage's would. ⚠️ `runConditioned` re-applies the clean blend AFTER the ancestral renoise (oracle `post_process_latent`) — unreachable before DFR (i2v/icT2V both run plain Euler), which is exactly why it was missing. | `--denoise-gate` (+ i2v via `--dit-pertoken`), `--denoise-wiring-gate`, `--dfr-gate` |
| `VideoVAE.swift` | 128-ch video VAE decoder + encoder (pixel-shuffle, PixelNorm, causal/non-causal) + `denormalizeLatent`/`normalizeLatent` — **fp32**. Widths come from the weights, so a pruned decoder (PrunaVAED) loads through the same code; its channel adapters switch on when `conv_in.norm3` is present. Decode tiling on BOTH axes: temporal `decodeChunked` (chunk+halo) × spatial `decodeSpatialTiled` (halo+crop, uniform windows; spatial RF 15.12 cells/side, halo 16 bit-exact, default 5 ≈ 74 dB; 4K-only lever — see BLOCKSTREAM-EXPANSION-EVAL §1.4d). | `--vae-decode`, `--vae-encode`, `--vae-decode-pruna`, `--vae-tile-gate` (+ `--vae-rf-probe`, `--vae-tile-bench`; TileGates.swift), bench `--vae-decode-bench` |
| `NA3D.swift` + `DiffusionVideoDecoder.swift` | **DiffVAE** — the 2.5 diffusion video decoder (claim C3), the second decoder for the same latent. 4 deterministic NA stages + pixel-shuffle upsamples → a stage-4 context volume, then 8 context-injected NA blocks that turn a pure-noise canvas into pixels in ONE x0 step at t=1.0 — **fp32**, COMBINED (full-volume) semantics, `LTX2_NA_IMPL=box\|gather` picking the NA backend (box default: shared key box + fused SDPA with a cached BOOLEAN window mask; gather: per-query window gather, more precise, ~6× slower and memory-hungry at full-decode size). ⚠️ **Stochastic**: `decode(noise:)` takes the canvas as a real input, and the oracle's verdict is that CONV is the more faithful decoder (42.6 vs 36.8 dB PSNR) at ~39× the speed — this port supplies the Swift arm of that comparison, it does not re-open it. | `--diffvae-gate` (6 stages) |
| `DurationHead.swift` | **duration head** (claim C6) — 1.9 M params over the CONNECTOR outputs (not the latents), predicting shot duration in log-seconds via a 1-query attention pooler, then clamped and snapped to the causal frame grid (8k+1) by `secondsToNumFrames`. **fp32**. ⚠️ Upstream's fused `in_proj_weight` is already split into q/k/v by the converter, so the shipped 19-tensor checkpoint keys the split names. Oracle verdict on C6 is PARTIAL (responds to pacing, not to explicit durations) and stands. | `--duration-gate` |
| `AudioVAE.swift` | audio VAE decoder (Conv2d, causal-height, latent→mel) + encoder (waveform→Slaney-mel→latent, LipDub reference audio) — **fp32** | `--audio-vae-decode-gate`, `--audio-vae-encode-gate` |
| `Vocoder.swift` | BigVGAN v2 + Hann-sinc resampler + MelSTFT + BWE (mel→48kHz) — **fp32** | `--vocoder-gate` |
| `Upsampler.swift` | latent upsampler, ALL THREE variants config-driven (sidecar `<stem>_config.json` → x2 / x1.5-rational / temporal-x2; `peekVariant` resolves without weight IO) — **fp32**. Two-stage wiring is VARIANT-AWARE: `LTX2Pipeline.upsamplerFile` (or `LTX_UPSAMPLER=`) selects the checkpoint and t2vTwoStage derives stage-1 geometry from it — x2 target ÷64, x1.5 target ÷96, temporal pixel-frames ≡ 1 (mod 16) + stage 1 at fps/2 (x1.5/temporal two-stage = OUR composition; no oracle/upstream consumer exists, verified 2026-08-05 vs both repos — acceptance = pre-flight geometry rejects + exact output shape + x2 bit-identity to the pre-wiring pipeline) | `--upsampler`, `--upscale-step`, `--upsampler-variants-gate`, `--two-stage-variants-gate` |
| `Positions.swift` | pixel-space video/audio positions + `distilledSigmas`/`stage2Sigmas` + LipDub audio-ref patchify/negative-time positions | (via `--audio-vae-encode-gate`) |
| `LTX2Pipeline.swift` | assembles all of the above → `t2v` (one-stage) / `t2vTwoStage` / **`i2v`** (first-frame conditioning); loads + decodes audio | `--e2e-gate`, `--audio-decode` |
| `KeyframeConditioning.swift` | clean keyframe at a chosen pixel frame (oracle `VideoConditionByKeyframeIndex`): appended clean tokens, denoise mask `1−strength`, single-frame positions offset by `frameIdx`. ⚠️ `frameIdx` is TILE-LOCAL. The oracle's `update_attention_mask(None, …)` returns nil throughout this path, so strength lives ENTIRELY in the denoise mask — no (B,N,N) machinery needed. | `--dfr-gate` |
| `DFRPlan.swift` | every DFR frame/fps/trim number, resolved before any weight load — delivered frames `(requested−1)·2^rounds+1`, the canvas trim, the AUDIO trim, playback fps `= fps·2^rounds`, conditioning fps capped at 60 **independently**. `generateDFR` reads nothing else, so gating the plan gates the shipping path. | `--dfr-gate` |
| `DFRPipeline.swift` | **DFR** (`generateDFR` → latents, `dfr` → decoded): stage-1 half-res + slots → spatial upsample of video AND slots → stage-2 detailing with slots re-appended SEEDED by their upsampled stage-1 content → optional temporal rounds (`2^round` keyframe-seam tiles, per-tile slots + `0.95`-strength seam anchors, ancestral Euler η=0.5, stitch, carry-forward). Tiles denoise **VIDEO-ONLY**. ⚠️ Rounds are a QUALITY/parity feature, NOT a speed lever (Python: x2.767 r=1 / x3.542 r=2, operator judged NATIVE better — `ltx-2-mlx/docs/claims/C2-dfr.md`). | `--dfr-gate` (orchestration, 6/6 mutation-tested), `--dfr25 W H F R` (real 2.5 checkpoint) |
| `Streaming/GranuleLayout.swift` + `Streaming/BlockStreamer.swift` | **HV2 weight streaming** (`LTX_DEV/STREAMING-PLAN.md`; sibling adaptation of wan-core `b45b879`, receipts `mlxengine-todo/probes/hv2_ltx_blockstreamer.out`): per-block granule files → two slot-backed groups, background pread refill, self-calibrating gate `N ≥ B·F/(2·S)` with output-invisible resident fallback. **LTX deltas:** dict-shaped bind (`DiTWeightStore` — the LoRAStore idiom), sub-256 B tensors are bind-time residents (⚠️ Swift `Data` inlines ≤14 B payloads — the alias seam silently misses them), plain params whose granule dtype ≠ computeDtype load resident WITH the cast (the six fp32 AdaLN tables), numGroups must be EVEN (slot parity), `warmup()` doubles as the S sweep (gate suspended), two-stage arms `gateEvaluationThresholdTokens` with stage-2's N. **Acceptance is CONDITIONAL, not per-quant** (⟲ corrected 2026-08-16, AB-T-0042 — the old per-quant wording had bf16/q8/q4 characterized backwards): the gate runs the resident reference TWICE first, and the bar it then applies depends on what the resident path itself reproduced *that run*. Resident self-repeat bit-exact → memcmp; otherwise the quant band **cosine ≥ 0.999** (the repo-wide gate bar, `StreamGates.swift:321`) + poison control. ⚠️ **q8 is the unstable one, not q4** — measured on real weights (AB-R-0052): bf16 memcmp ✅, **q8's own resident reference failed to self-repeat in 3 of 7 runs** (cos 0.9971–1.0000), **q4 hit memcmp exact**, stronger than any band. So "q8 self-repeats bit-exactly in-process" was false as an unconditional claim, and q4 never needed the band it was documented under. 🔑 **Exit 2 = INCONCLUSIVE, not regression.** ⚠️ The trigger is `residentSelfExact || parity` (`StreamGates.swift:374`), NOT "the reference wobbled" — an unstable reference whose streamed arm still lands in the band is a normal PASS (observed 2026-08-16 at kit 0.5.0: q8 self-repeat 0.999968 ✗, streamed 0.999968 → band ✅, exit 0). Exit 2 fires only when the reference failed to self-repeat **and** the streamed arm missed the band — the one case where "streaming broke" and "the reference is unstable" cannot be told apart. Even then the reference-independent checks (determinism, poison, golden) are still asserted: exit **2** if they are green (RETRY, not a streaming break — don't wire CI to bank it as a regression), exit **1** if they are not. Opt-in: `LTX2Configuration.streamedBlocks` + `granuleRootDirectory` (Codable-excluded) or `LTX_STREAM_GRANULES=<dir>`; granules via `ltx-granule-layout` → `/Volumes/Satechi/Models/ltx-granules/{bf16,q8,q4}` (2.3) and **`ltx-granules-25/{bf16,ditq8}`** (2.5). 🔑 **2.5 STREAMS, and the substrate needed no change to do it** (2026-08-16, AB-R-0088): the tree was 2.3-only by accident of the gate's hardcoded paths, not by design — `ensureDiT()` already passes the same `DiTConfig()` down both paths and 2.5's deltas are checkpoint-key-driven. Read off the headers, not assumed: 2.5 uses the **identical dialect** (`transformer.transformer_blocks.`, 48 blocks), differing only in 84 keys/block vs 86 (`ff_bias:false`) and 59 globals vs 58 (the keyframes embedding), so `ltx-granule-layout` laid both trees out first try. Gate arms carry a **`25-` prefix** (`25-bf16`, `25-ditq8`); ⚠️ there is deliberately **no `25-q4`** (2.5 has no q4 DiT) and **never a `25-q8`** (on 2.5 that name is the int8 TEXT-ENCODER sibling whose transformer symlinks back to bf16 — an arm so named would stream bf16 against a bf16 reference and report the null delta as a result). ⚠️ 2.5 inputs are NOT the 2.3 `dit_full` golden — `DiTInputs.forLTX25` rebuilds them the way `t2vTwoStage` does (Gemma-4 golden → 2.5 connector, first-latent-frame keyframes mask, stage-2 sigma); feeding 2.5 the 2.3 golden runs the 2.5 forward with **no keyframes mask**, a different forward from the shipping one, silently. ⚠️ Any future hand-written block loop must route `acquireGroup`/`releaseGroup` or refuse-while-streamed (the bernini trap). | `--stream-tiny-gate` (synthetic, CI), `--stream-parity-gate [bf16\|q8\|q4\|25-bf16\|25-ditq8]`, `--stream-auto-gate [quant] [videoN]` (ONE rung per process), `--stream-budget-gate [quant] [videoN] [GB] [steps]`. ⚠️ **Never run two streaming gates at once** — S is a shared-resource measurement and contention silently depresses it (measured: a q4 arm read S=2.15 vs its clean 3.56 GiB/s when a second gate overlapped, moving the auto rung's N_min 3629→4154). Verdicts survive; every S/stall number from an overlapped run is void. |

`Sources/MLXLTX2/` — the engine wrapper: `MLXLTX2Package` (ModelPackage, `.textToVideo` incl. i2v
via `initImage`), **`MLXLTX25Package`** (the 2.5 sibling — a SEPARATE package under the same
capability, owning no run logic and forwarding to the 2.3 one; the two share every line of
behaviour and **no numbers**, so it exists for the manifest), `LTX2Configuration` (+ the
**`LTXFamily`** axis and `WeightPrewarming` conformance), `FrameCodec` (frames→H.264
+ AAC-muxed MP4), `ImageInput` (decode/preprocess the i2v init frame).
⚠️ **2.5's declared footprints make bf16 INADMISSIBLE on `standard64` — that is intended.** Charge
46 GB vs a 44.8 budget, off a measured peak already at **96%** of budget at 121f, *below* the
profile's own 161f cap. `standard64` runs **int8** (28 GB, 62%); **bf16 is a `max128` quant on 2.5**
and stays the pick wherever bit-reproducibility matters, since q8 is a LEAN tier (AB-R-0038). `Sources/RunLTX2/` — the
parity-gate CLI, plus **`--bench-e2e`** (`BenchE2E.swift`) — the protocol A/B harness for any
timing/memory lever claim: ABBA arm alternation, excluded warmups, prewarm, cooldowns, 25 ms phys
sampling, MEASURABLE/NOISE verdicts, receipts → `probes/`. **Any "X is faster/lighter" claim about
an e2e lever should cite one of its receipts — single-pair timings are how the ±15 s desktop drift
manufactures findings (see `BENCH.md`).**
Arm axes: `model=ltx23|ltx25` · `quant` · `decoder` · **`dfr=<rounds>`** · `cache` · `env.KEY`, plus
**`--dry-run`** (resolve + print the whole plan, no weights, no GPU — use it before every real
invocation). ⚠️ **The quant-sibling suffix is family-dependent and wrong silently**: 2.3 uses
`-q8`/`-q4`, but on 2.5 `-q8` is the int8 TEXT-ENCODER tree whose transformer symlinks back to
bf16, so an arm labelled q8 there benchmarks bf16 and reports the null delta as a finding — 2.5's
quantized DiT is `-ditq8` and it has no q4. ⚠️ **Timing noise SCALES with geometry** — `BENCH.md`'s
"≤8.5 s is noise" was taken at 9f; the cold-start artifact is ~30 s at 121f and 47–107 s at 241f, so
a single-block arm at ≥121f is uninterpretable.
**`--gen-gap-probe`** (`GenGapProbe.swift`) is the enhancer/port diagnostic arm. Two settled results
live there, both of which began as WRONG claims of mine: AB-R-0080's "2.6× mlx-swift-lm gap" was a
debug build plus a tokenizer cost summing to a convincing uniform factor (retracted, AB-L-0046 —
release Swift is at parity: decode 61.6 vs 65 tok/s, prefill 0.56 s vs 0.71 s), and the tokenizer
half is **not** a general `swift-transformers` slowness. `Tokenizer.encode` is **quadratic in a
checkpoint's added-token count** (one `NSRegularExpression` capture group per added token,
`Tokenizer.swift:517-523`), charged once per input character that can BEGIN an added token — so
Gemma-3's 6,415 added tokens cost ~29 ms per newline, and the enhancer prompt's 39 newlines are its
whole 1.14 s. ⚠️ **LTX-2.5 does not pay this**: `gemma4-12b-ltx-v1` has 24 added tokens and prices
`\n` at 0.003 ms, so only the Gemma-3 **prompt enhancer** is affected. Filed upstream with a verified
behaviour-preserving fix (huggingface/swift-transformers#383; 40-line prompt 1123 → 1.89 ms, token
IDs identical). Receipts AB-R-0086, method trap AB-L-0047 — don't re-derive either.
Claim-bench arms A–F live in **`probes/CLAIMS-BENCH-MATRIX.md`** (A ✅ parity · B ✅ DFR dominated ·
C–F reachable). `parity/` — Python golden dumpers + (gitignored) `goldens/`.

## Conventions

- Build: `DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun swift build`.
- Cores are functional (`[String: MLXArray]` + explicit ops keyed by oracle weight-key strings);
  fp32 except the DiT (bf16). New component → port from oracle → `parity/dump_*` golden →
  `RunLTX2 --*-gate` (cosine ≥0.999) → commit.
- `mlx-swift-lm` is a **local path-dep** (`../mlx-swift-lm`) checked out at plain **upstream
  `main`** — no fork, no local patches (upstream #387 / `6608a35` exposes `@_spi(GemmaEncoder)`;
  the tap is ours). **Transitional**: no release tag contains #387 yet, so don't ship off this
  pin; adopting a tag is a one-line `Package.swift` edit. Goldens are gitignored + regenerable —
  `parity/dump_text_encode_goldens.py` writes both the `.npy` set and the packed
  `goldens.safetensors` the Swift gates read. **Page the weights in first**
  (`cat <model>/*.safetensors > /dev/null`) or the oracle dump trips the GPU watchdog.
- **Repo is Apache-2.0 and published** (`xocialize/ltx-2-mlx-swift`) — the port code is our own
  implementation (license stance reversed 2026-06-16, see `../CLAUDE.md` §License). **Never commit**
  the converted weights or `parity/goldens/` — those are LTX-2 weight-derivatives (Community-licensed)
  and stay gitignored.
