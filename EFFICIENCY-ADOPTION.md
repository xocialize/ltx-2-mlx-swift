# Efficiency Adoption Brief — `ltx-2-mlx-swift` (LTX-2.3, `textToVideo`)

> **For a session-specific agent.** Self-contained: audit + prioritized tasks to adopt the MLXEngine
> library-efficiency contract (engine 1.14.0). Load the `mlx-swift-integration` skill and read
> `references/package-efficiency.md` + `references/memory-harness.md` first. This is the **template** for
> the per-package sweep — later packages get a brief in the same shape. Audited 2026-06-30.

## Package at a glance
- **Wrapper:** `MLXLTX2Package` (`Sources/MLXLTX2/`), core `LTX2` (`Sources/LTX2/`). Capability `textToVideo`.
- **Pipeline (multi-component, tier 3):** Gemma-3 text encoder → Connector → joint-AV DiT → 128-ch Video VAE → Audio VAE → Vocoder (+ two-stage: VAE encoder + Upsampler). Synchronized audio-video MP4.
- **Why it's the first sweep target:** the heaviest, most-staged pipeline in the library — biggest upside, and a good end-to-end validation of the adoption process itself.

## Engine dependency status (prerequisite)
- `Package.swift` declares `mlx-engine-swift` **`from: "0.9.1"`** (the inline "Pinned at 0.7.0" comment is **stale — fix it**). `Package.resolved` currently resolves **0.13.0**.
- 0.13.0 already has config-aware footprint (`QuantConfigured` ✓ — the config conforms) but **not the 1.14.0 split** (`peakActivationBytes` / `FootprintConfigured.peakActivationBytesHint` / `BudgetAware`).
- **P0 is just `swift package update`** to move 0.13.0 → 0.14.0 — `from: "0.9.1"` already admits it, no manifest edit needed beyond fixing the stale comment.

## Audit vs. the four levers

| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | resolves 0.13.0; re-resolve to 0.14.0 to get the split + BudgetAware | **P0** |
| 1. Split footprint | ❌ | manifest folds activation INTO `residentBytes` (84/64/54 GB = weights + activation peak); single-number model | **P1** |
| 2. mmap/lazy load | 🟢 mostly | `MLX.loadArrays` mmaps; casts are lazy. Minor: fp32 casts double a couple of small components | P3 |
| 3. Per-stage load→use→evict | ❌ | `LTX2Pipeline.load` loads ALL 8 components up front and holds them resident for the whole run | **P2 (headline)** |
| 4. BudgetAware adaptive dtype | ➖ | has a quant lever, but int4 "diverges the sample" (quality) — adaptive precision has a real cost | P4 (defer) |

---

## P1 — Declare the split footprint  (effort S; data mostly exists)

The manifest (`MLXLTX2Package.swift:46–70`) declares `residentBytes` as the **measured peak** and explicitly notes *"activation working set is dtype-independent (same bf16 compute) so it dominates the peak."* That is the textbook case for the split: the ~27–31 GB activation is roughly constant across quants; only the weights change. Re-declare as `residentBytes` = **weight residency** + `peakActivationBytes` = **peak − weights**:

| Quant | Measured peak | Weight residency (stated) | → `residentBytes` | → `peakActivationBytes` |
|---|---|---|---|---|
| bf16 | 82.81 GB | ~52 GB | ~52 GB | ~31 GB |
| int8 | 62.39 GB | ~35 GB (bf16 − ~17) | ~35 GB | ~27 GB |
| int4 | 53.33 GB | ~28 GB (q4 DiT 11.3 vs 35) | ~28 GB | ~26 GB |

- **Do:** re-run the memory harness (`references/memory-harness.md`) to get a clean step-1 resident floor (weights) and step-2 peak per quant at the **704×512×9f** envelope, then `residentBytes = floor`, `peakActivationBytes = peak − floor`. The table above are starting estimates — measure, don't trust them.
- **Payoff:** the engine reserves ONE ~31 GB activation across residents instead of baking it into each. LTX is usually the only heavy resident, but the split lets it co-reside with smaller models (and the optimizer chain) far better, and makes the per-quant numbers honest.
- Keep the existing measurement comments (provenance) — just move the activation term into `peakActivationBytes`.

## P2 — Per-stage load → use → evict  (effort M–L; the headline win)

`LTX2Pipeline.load` (`LTX2Pipeline.swift:62–81`) loads Gemma + Connector + DiT + Video VAE + Audio VAE + Vocoder + VAE encoder + Upsampler **all up front** and holds them as non-optional `let`s for the pipeline lifetime. But the flow uses them in phases (`t2v`/`t2vTwoStage`/`i2v`):

1. **Gemma text-encode** — used ONCE at the start (`states = gemma.allHiddenStates(...)`), then **idle through the entire denoise + decode**. Gemma-3 is multi-GB. This is the Wan-T5 pattern exactly → **evict Gemma after encode, before the DiT denoise peak.**
2. **VAE encoder + Upsampler** — two-stage only, used between stage-1 and stage-2; evictable after the upscale.
3. **Video/Audio VAE decoder + Vocoder** — used at the very end; can load lazily after denoise.

The memory peak is the DiT denoise loop. Cutting co-resident weights during it (drop Gemma ≈ several GB; defer decoder load) lowers the peak → lowers declared `residentBytes`/`peakActivationBytes` → fits smaller tiers and co-resides better.

- **Refactor:** make staged components load-on-demand / evictable (optionals + a small `ensureX()`/`dropX()` around each phase), generalizing Wan's `withTextEncoder` helper. Drop a stage with the ref set to `nil` + `GPU.clearCache()` before the next.
- **Tradeoff (accept + document):** evicting Gemma means re-loading it on the next request (encode is cheap vs denoise; same call as Wan's +reload/−residency trade). A keep-resident flag for big-RAM tiers is the natural refinement, but **evict-between-stages is the default.**
- **After this lands, re-measure (P1)** — the declared footprint should drop.

## P3 — fp32 cast review  (effort S; modest)

`AudioVAE.swift:26` and `Connector.swift:52` cast weights to **fp32** (`v.asType(.float32)`), doubling those components' resident size vs the bf16 on-disk weights. `DiT.swift:57` casts to `computeDtype` (bf16) — a near-no-op for bf16 on-disk and correctly skips quantized layers, so DiT is fine. Evaluate whether AudioVAE/Connector can run **bf16** compute without breaking parity (they may need fp32 for numerical reasons — check the parity gates before changing). Small components, so this is a modest, parity-gated cleanup, not a priority.

## P4 — BudgetAware  (defer)

The config is `QuantConfigured` (quant chosen at registration). `BudgetAware` could pick int8/int4 under pressure, but the manifest notes **int4 diverges the distilled sample** (quality cost), so adaptive precision isn't free here. Defer until a memory-constrained tier specifically needs it; if adopted, gate the downgrade on a quality/DOVER check, don't silently drop precision.

---

## Already good (don't regress)
- mmap baseline (`MLX.loadArrays`) + watchdog-disciplined per-block `eval` in DiT/Connector/DenoiseLoop.
- Runtime-LoRA **hot-swap** on the resident DiT (`setLoRAs`/`clearLoRAs`) — no reload; keep it.
- Cancellation honored before/after generation; `ModelStorable` + `WeightPrewarming` already wired.

## Definition of done
- [x] `swift package update` → engine 0.14.0; fix the stale "Pinned at 0.7.0" comment.
- [x] Split footprint declared per quant (`residentBytes` weights + `peakActivationBytes` activation), re-measured at 704×512×9f; provenance comments kept.
- [x] Pipeline stages Gemma+Connector (load→encode→evict) and defers/evicts VAE-decode + (two-stage) encoder/upsampler; `clearCache()` between stages.
- [x] Re-measured footprint reflects the lower denoise-peak residency (bf16 peak 82.81 → 51.95 GB).
- [x] (Optional) AudioVAE/Connector fp32→bf16 evaluated — **kept fp32** (see below); P2 already evicts both off-peak.
- [x] Parity gates (`--e2e-gate` cosine 0.999971) + in-app validation (LTXVideoTesting, all 3 quants, valid MP4s) green.
- [x] BudgetAware: explicitly deferred (note why) — below.

---

## Adoption outcome (executed 2026-06-30)

**P0 — engine 0.14.0.** `swift package update` → 0.14.0 (package + workspace `Package.resolved`); stale Package.swift comment fixed.

**P2 — per-stage load→use→evict (the headline).** `LTX2Pipeline` now holds ONLY the DiT resident; Gemma+Connector load for text-encode then evict (`eval` embeds → ref=nil → `Memory.clearCache()`) *before* the denoise; the VAE-decode stack loads after denoise; the two-stage VAE-encoder/upsampler load only around the upscale step. Public async paths take `isolation: isolated (any Actor)? = #isolation` so the now-async methods stay on the wrapper's `@InferenceActor` (no `Sendable` violation). LoRA hot-swap + cancellation untouched (DiT stays resident). **Insight:** because weights are lazy-mmap, the DiT isn't even materialized during encode → encoder and DiT are *never co-resident*, so the peak is the DiT denoise alone.

**P1 — split footprint (MEASURED in LTXVideoTesting, M5 Max/128 GB, two-stage 704×512×9f, seed 42):**

| Quant | Resident floor (declared) | Activation (declared) | Measured peak | OLD single-number | New reserve | Peak drop |
|---|---|---|---|---|---|---|
| bf16 | 38.85 → **40 GB** | 13.10 → **16 GB** | 51.95 GB | 84 GB | 56 GB | **−31 GB** |
| int8 | 21.49 → **22 GB** | 12.10 → **15 GB** | 33.59 GB | 64 GB | 37 GB | −29 GB |
| int4 | 12.19 → **13 GB** | 15.06 → **18 GB** | 27.25 GB | 54 GB | 31 GB | −26 GB |

Activation is ~dtype-independent (12–15 GB) — the engine now reserves ONE such transient across residents instead of baking the full peak into each quant. bf16 LTX + a ~15 GB optimizer model now co-reside under one shared 16 GB activation reserve where before LTX alone claimed 84 GB.

**P3 — fp32 casts: kept (both).** Connector (`Connector.swift:46`) *requires* fp32 — its own comment documents the 188160-wide projection overflowing in bf16 matmul (libmlx divergence) → correctness, not a choice. AudioVAE (`AudioVAE.swift`) is intrinsically fp32 (input cast + fp32 per-channel stats; precision-sensitive spectral decoder) and only 106 MB. Critically, **P2 already evicts both before/after the denoise peak**, so their fp32 size no longer inflates the *declared* footprint — the lever P3 targeted is subsumed by P2. Not worth the parity risk for ~0 off-peak gain.

**P4 — BudgetAware: deferred.** The config is `QuantConfigured` (quant chosen at registration). A memory-adaptive downgrade would have to drop bf16→int4, but **int4 diverges the distilled sample** (measured: q4 is artifact-free but shifts the trajectory) — so adaptive precision carries a real quality cost, not a free win. Defer until a constrained tier specifically needs it; if adopted, gate the downgrade on a quality check, don't silently drop precision.

## Validation note
LTX validated the sweep process: the brief was executable cold, the split materialized the co-residency win (−31 GB bf16 peak; one shared transient), and the memory-harness re-measure flow works (added `RunLTX2 --mem-bench` + an in-app split readout in LTXVideoTesting). Replicate for the most-consumed packages, then the optimizer family. (Wan deferred for a dedicated deep-dive.)

> **Watchdog note (process calibration):** the standalone `RunLTX2 --mem-bench` CLI trips the Metal GPU watchdog (`kIOGPUCommandBufferCallbackErrorTimeout`) on the full two-stage run on this beta OS — the gate is kept for component-scale measurement, but the **reliable measurement surface is the app autorun** (`LTX_AUTORUN=1 LTX_QUANT=bf16|q8|q4`), which has the engine's `WeightPrewarmer` + governor. Same methodology, stable.
