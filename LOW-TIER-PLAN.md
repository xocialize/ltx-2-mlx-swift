# Low-Tier Plan — bringing the LTX-2.3 floor to 32 GB (target: 24 GB) Macs

> **Self-contained, executable cold** (same shape as `EFFICIENCY-ADOPTION.md`). Goal: make LTX-2.3
> admissible + usable on 32 GB Macs, stretch **24 GB** (the M5 MacBook Pro default). Written
> 2026-07-01 after the ecosystem evaluation (`wildminder/awesome-ltx2`) and the profiling deep-dive
> (`PROFILING.md`).

## Why these levers (evaluation outcome)

**There is no smaller LTX-2 checkpoint** — every LTX-2.3 variant (dev/distilled/fp8/nvfp4/GGUF) is the
same 22B DiT, and we already run the distilled (8-step + 3-step stage-2). The ecosystem's memory story
reduces to quant depth + component handling, and we're already ahead of it:

| Ecosystem | Theirs | Ours |
|---|---|---|
| DiT fp8/int8 | ~29 GB | int8 **20.6 GB** ✓ |
| DiT nvfp4 / GGUF Q4 | 14–21 GB | int4 **11.3 GB** ✓ (GGUF/nvfp4 don't run on MLX) |
| Gemma TE fp8/fp4 | 9.5 GB | gemma-3-12b **4-bit ≈ 6.5 GB** ✓ |
| GGUF Q2/Q3 | 12.4 GB | REJECTED — int4 already shifts the distilled trajectory; sub-4-bit is a quality cliff |
| Block-swap / LowVram offload | ComfyUI | REJECTED — frees nothing on unified memory (standing rule) |

So the floor levers are **local engineering**, targeting our measured per-stage peaks (per-stage evict
means the peak is `max(encode, denoise, decode)`):

**Measured baselines (M5 Max/128 GB, 704×512×9f two-stage, seed 42):**

| Stage | bf16 | int8 | int4 | dominated by |
|---|---|---|---|---|
| encode | ~27 GB phys | ~27 | ~27 | Gemma 4-bit 6.5 + **connector fp32 12.7 (materialized 2× its bf16 weights)** |
| denoise | 38.9 + 13.1 act | 21.5 + 12.1 | 12.2 + 15.1 | DiT resident + seqLen-scaled activation |
| decode | grows with frames | — | — | **un-chunked VAE: 240f bf16 run peaked 110 GB**; cache ~17–20 GB at 41f |

**Budget math:** governor ≈ 0.7× unified (eval apps run 0.85×).
- 32 GB → **22.4 GB** budget (0.85× → 27.2)
- 24 GB → **16.8 GB** budget (0.85× → 20.4) ← every stage must fit under this

At 24 GB, ALL THREE levers are mandatory: chunked decode (T1), connector residency (T2), and small-envelope
tier profiles (T3). Ordering below starts with the runway to the chunked-decode headline.

---

## T0 — Decode attribution + whole-frame parity harness (S) — the runway to T1

1. **Per-stage decode spans:** add profiler spans inside `VideoVAEDecoder.decode` (per up-block /
   upsample stage) so the decode peak is attributed, and capture the **decode-peak-vs-frames curve**
   via the app autorun (`LTX_FRAMES=9/48/120/240`, the `SPLIT`/profiler lines) — the baseline T1 must
   beat.
2. **`RunLTX2 --vae-chunk-gate`:** real-weights gate comparing **whole-frame decode vs temporally
   chunked decode** of the same latent (cosine + maxAbs + per-boundary-frame PSNR, the RIFE
   seam-eval pattern). Whole-frame at a mid envelope (e.g. 41–113 output frames @704×512) is the
   exact reference — it fits on this box. The gate is written FIRST so T1 iterates against it.

**Acceptance:** decode curve captured; gate runs green in trivial mode (chunk = whole).

> **T0 RESULTS (2026-07-01, DONE):** gate + `decodeChunked` landed together. Trim math
> (`startTrim = a==0 ? 0 : 8·hl−7`, `endTrim = b==F ? 0 : 8·hr`) exact on first try (shape + totals).
> Decode-peak-vs-frames is LINEAR at ~2.2 GB/latent-frame @704×512: F_lat 15 → 34.5 GB, F_lat 30
> → **67.8 GB** (233 frames — this is what drove the 240f run's 110 GB total). Chunked is
> window-bound. Halo sweep @F_lat 30, chunk 5 (seam-min PSNR): halo 2 → 38.5 dB ❌ · 3 → 51.4 ❌ ·
> 4 → 59.1 ❌ (just under) · **5 → 66.2 dB / cosine 1.000000 ✅**; chunk 10 halo 5 → 67.1 dB ✅.
> Exact temporal receptive field is **13.5 latent frames** (bit-exactness would defeat chunking);
> conv influence decays, so **halo 5 is the perceptually-exact minimum — LOCKED**. Peaks: chunk 5
> h5 (window 15) = 36.9 GB · chunk 10 h5 (window 20) = 47.3 GB vs 67.8 whole. Time cost = window/chunk
> (chunk 5: ~2.9× whole; chunk 10: ~1.7×) — T3 picks per-tier chunk sizes on that tradeoff.

## T1 — Temporal-chunked VAE decode (M) — the headline

Decode the (1,128,F,H,W) latent in **temporal chunks with halo overlap**, concatenating pixel outputs.
Structure that matters (`VideoVAE.swift`): NON-causal temporal convs (k=3, symmetric replicate pad,
`tpad=1` per conv), 3 × temporal-2× pixel-shuffle upsamples (8× total) each followed by
**drop-first-frame**, res stages `[2,2,4,6,4]` blocks × 2 convs.

- **Derive the exact temporal receptive field** per side in latent frames (sum of conv halos at each
  cumulative temporal scale); start with a conservative halo (e.g. 4 latent frames/side) and shrink
  against the T0 gate. Interior chunks: replicate-pad only at true clip edges; trim halo in PIXEL space
  (accounting for the per-stage drop-first).
- **Accept near-exact** (cosine ≥ 0.9999 / boundary PSNR ≥ ~60 dB) if bit-exact proves fiddly through
  the drop-first boundaries — perceptual output, RIFE precedent.
- **Auto chunk-size from budget** (frames per chunk chosen so decode-stage peak ≤ a target), plus a
  config override. Default ON.
- i2v's single-frame VAE *encode* is unaffected; audio VAE/vocoder are small (skip/keep as-is).

**Acceptance:** decode-stage peak ~flat vs frame count (tile-bound, the SeedVR2 property);
240f bf16 total peak drops well below today's 110 GB; `--vae-chunk-gate` + `--vae-decode-gate` +
`--e2e-gate` green; one in-app 240f run confirms output unchanged perceptually.

> **T1 STATUS (2026-07-01): DONE — closed by an in-app 240f run.** `decodeChunked` (gate-validated
> above) is wired into all three pipeline paths via `decodePixels`: defaults **chunk 8 / halo 5**
> (window ≤ 18), engaging only when `fLat > chunk + 2·halo`; env overrides `LTX_VAE_CHUNK` /
> `LTX_VAE_HALO` (0 disables).
> **Close-out (240f i2v-adapter bf16 @704×512, identical settings to the pre-chunk run):**
> peak **110.22 → 92.23 GB** (act 65.7 → 47.9), run 414.5 → 463.8 s (+12%, decode-phase only),
> output valid. The remaining act peak IS the decode window (18 latent ≈ 43 GB @704×512) — i.e. the
> knob now controls the decode peak: chunk 5 (window 15 ≈ 37 GB) would put this run near ~81 GB.
> **Budget-driven window sizing folds into T3's tier profiles.**

## T2 — Connector residency (S)

`Connector.init` materializes ALL weights fp32 (12.7 GB; the bf16→fp32 upcast is a **compute**
requirement — the 188160-wide projection overflows bf16 matmul — not a storage one).
- Step 1: keep weights **bf16-resident, upcast fp32 per-op** in the forward → 12.7 → **6.3 GB**.
- Step 2 (if 24 GB needs it): quantize connector Linears to **int8 with fp32 compute** → ~**3.2 GB**.
Gates: `--connector-gate` + `--text-encode-gate` cosines unchanged (≥ 0.999).

**Acceptance:** encode-stage phys drops ~6–9 GB (from ~27 toward ~13–19 measured); gates green.

> **T2 RESULTS (2026-07-01, Step 1 DONE):** Connector weights now stay **bf16-resident** with lazy
> `f32()` views at every use site (the two 188160-wide projection weights hoisted once per forward —
> one transient fp32 materialization each, shared across the watchdog chunk loop; small block
> weights upcast per-op). Math is bit-identical to the old materialize-at-init. Gates:
> `--connector-gate` VIDEO 0.999988 / AUDIO 0.999666 · `--text-encode-gate` VIDEO 0.999989 /
> AUDIO 0.999651 — **PASS**. Connector resident 12.7 → ~6.3 GB. Step 2 (int8 weights + fp32
> compute, → ~3.2 GB) remains available if the 24 GB tier's encode-stage measurement needs it (T3).

## T3 — Tier profiles + `FootprintConfigured` hints + honest admission (S–M)

Activation is seqLen-scaled, so small envelopes shrink the denoise term — declare it per tier instead
of charging every tier the 704×512 numbers. Define + MEASURE (app autorun `SPLIT` lines) profiles like:

| Tier | Quant | Path | Envelope (start point, tune on measurement) |
|---|---|---|---|
| 24 GB | int4 | one-stage (no upsampler/stage-2) | ≤ 512×288, short clips |
| 32 GB | int4/int8 | one-stage | ≤ 576×320 |
| 64 GB | int8 | two-stage | 704×512 |
| 128 GB | bf16 | two-stage | up to ~240f (proven) |

- Wire per-profile `residentBytesHint` + `peakActivationBytesHint` on `LTX2Configuration`
  (`FootprintConfigured` — the 1.14 machinery exists for exactly this); the app/tier picker selects the
  profile; requests above the profile's envelope are clamped or rejected honestly.
- **16 GB: declared ineligible** — int4 DiT alone (11.3 GB) ≈ a 16 GB governor budget (11.2); no smaller
  checkpoint exists. Record, don't chase.
- int4's known trajectory divergence is ACCEPTED for low tiers (artifact-free; documented).

**Acceptance:** every declared profile's measured stage-max ≤ 0.7× its tier; the admissibility panel
shows truthful ✅/❌ per tier; registry/manifest re-baselined from the measured `SPLIT` lines.

> **T3 RESULTS (2026-07-01 — plumbing DONE + measured; budgets NOT yet met → T3b below):**
> `LTX2Profile` (envelope clamp + one/two-stage + VAE window + `peakActivationBytesHint`) landed in
> config/wrapper/pipeline/app (`LTX_TIER=` env). All three runs requested 704×512×240 and were
> correctly clamped (headers prove it); one-stage path exercised on low tiers.
>
> | profile | geometry ran | peak | floor | act | budget 0.7× | peak STAGE (profiler) |
> |---|---|---|---|---|---|---|
> | compact24 (q4, chunk4) | 512×288×121 1-stage | **35.9** | 11.9 | 23.9 | 16.8 ❌ | **encode 36.1** (denoise 15.3 ✓, decode 26) |
> | balanced32 (q4, chunk6) | 576×320×161 1-stage | **35.6** | 12.1 | 23.5 | 22.4 ❌ | **encode** (envelope-independent — same stage) |
> | standard64 (q8, chunk8) | 704×512×161 2-stage | **60.7** | 21.7 | 39.1 | 44.8 ❌ | **decode chunk[8,16): act 23.7 + CACHE 36.7** (encode 45.4 #2, denoise 26 ✓) |
> | max128 (bf16) | 704×512×240 i2v | 92.2 | 44.4 | 47.9 | 89.6 ✓ | decode window (T1 close-out) |
>
> Hints are declared at the MEASURED values (honest now), to be tightened by **T3b — the three
> levers the attribution names precisely:**
> 1. **Sequential Gemma→connector inside the encode stage** (drop Gemma after `allHiddenStates`,
>    BEFORE the connector forward — they never need co-residency; hidden states are ~0.8 GB
>    materialized). ~−6.5 GB off the encode peak on every tier.
> 2. **Connector int8 (T2 step 2) via `quantizedMatmul` with fp32 activations** — kills BOTH the
>    6.3 GB bf16 residency (→ ~3.2) AND the hoisted fp32 projection views (~4.6 GB transient),
>    since quantized matmul needs no fp32 weight materialization. Gates: connector/text-encode.
> 3. **Scoped `Memory.cacheLimit` during VAE decode** (the Wan `cacheLimit` lever): standard64's
>    60.7 is act 23.7 + pool 36.7 — the pool retains window intermediates; a decode-scoped cap
>    forces reuse (restore after). Expected: standard64 decode → ~25, compact24 decode 26 → ~15.
> Post-T3b projections: compact24 ≈ 15–16 ✓ (16.8) · balanced32 ≈ 16–17 ✓ (22.4) · standard64
> ≈ 34 (encode-bound after fixes) ✓ (44.8). If compact24 still misses, the remaining lever is
> DiT-evict-before-decode on low tiers (reload cost accepted there).

## T4 — Validation + docs

- In-app runs per profile on this box, peaks recorded against budgets (can't shrink RAM, CAN verify
  peaks); int4 quality spot-check at the 24/32 GB envelopes.
- Update `PROFILING.md`, `README.md` (Memory table gains tier rows), `EFFICIENCY-ADOPTION.md` pointer,
  registry row, and the `mlx-swift-integration` skill if new durable lessons surface (the halo math
  likely is one — it generalizes to Wan vae22 halo-tiling).

## Deferred / rejected (don't revisit without new facts)
- GGUF/nvfp4/fp8-CUDA checkpoints (format/kernels don't apply to MLX; our int8/int4 are ahead).
- Sub-4-bit DiT (quality cliff past int4's already-shifted trajectory).
- CPU offload / block swapping (unified memory — frees nothing).
- Smaller checkpoint (doesn't exist for LTX-2).
- Step-count reduction below the distilled 8 (+3) — a latency lever, not a floor lever.
