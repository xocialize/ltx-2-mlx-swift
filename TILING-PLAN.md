# TILING-PLAN.md — modality tiling (the DiT-side attention wall)

**Opened 2026-08-04.** Port `ltx_core.modality_tiling` (upstream Lightricks; MLX port in the
oracle at `components/modality_tiling.py`) into Swift. This is the **DiT-side** lever and is
distinct from everything we already ship:

| Lever | Caps | Where | Status |
|---|---|---|---|
| Block streaming (`BlockStreamer.swift`) | **weight** memory | DiT | ✅ shipped (HV2) |
| VAE temporal chunking (`decodeChunked`) | decode activation, temporal | VAE | ✅ shipped |
| VAE spatial tiling (`decodeSpatialTiled`) | decode activation, spatial | VAE | ✅ shipped (v0.2.0, 4K-only) |
| **Modality tiling** | **DiT forward activation — the O(N²) attention scores** | **DiT** | ⬜ **absent** |

🔑 **This is the wall `BLOCKSTREAM-EXPANSION-EVAL` §2 named and left open** (its table lists
"per-block attention transients at big N" with the lever as *"check what `scaledDotProductAttention`
already gives us"*). Upstream had already solved it at the **token** level, and the oracle ported
it — we just never consumed it. Oracle's own scaling table (attention scores per layer):

| geometry | Nv | scores/layer |
|---|---|---|
| 480×704×33 | 1650 | ~350 MB |
| 480×704×97 | 3168 | ~1.3 GB |
| 720×1280×97 | ~9000 | **~10 GB** |
| 1080p 8s+ | — | **"doesn't fit any current Apple Silicon"** |

Splitting N into N/k per tile drops peak attention memory by **k²**. `--tile-spatial 2` (4 tiles)
= 4× cut. **Our Swift port cannot generate 1080p 8s+ at all today; the oracle can.**

## Design (mirror the oracle, which mirrors upstream)

- `Modality` — value type bundling latent + sigma + timesteps + positions + context + masks.
  Isomorphic with upstream's dataclass; the canonical in/out type for the tiler.
- `VideoModalityTiler(tiling:latentShape:)` — built once per generation from the token-grid
  `(F,H,W)`. Owns the tile list (`splitByCount` per axis + overlap).
  - `tileModality(_:tile:normalizePositions:) -> (Modality, TileContext)` — slice token state to
    the tile; keep-mask selects generated tokens by grid position, conditioning tokens by
    point-in-range on every axis **or negative time** (IC-LoRA refs are kept in every tile).
  - `blend(tileOutput:tile:ctx:into:)` — accumulate with the tile's **trapezoidal** blend mask;
    conditioning tokens weighted `1/(number of tiles that keep them)` so contributions sum to 1.
- `TiledDiT` — wraps `DiT` (and composes with `BlockStreamer` in either order), iterates tiles,
  blends video output, **averages audio output across tiles**.

⚠️ **Position-layout divergence to preserve:** upstream uses `(B, num_axes, T, 2)` *interval*
positions; the oracle uses `(B, T, num_axes)` *point* coords, so the cond-overlap test is
point-in-range rather than interval-overlap ("mathematically equivalent for non-degenerate
intervals"). **Our Swift port already uses point coords**, so we inherit the oracle's convention —
match it, and gate against the oracle, not against upstream's interval math.

## 🔑 Optimization findings — seams the oracle left on the table

Found while reading `modality_tiling.py`. These are the "what could run in parallel / on GPU that
was skipped" answers for this component. **Port correct first, gate, then apply these** (the
porting doctrine: match first, optimize with a stated reason).

**1. `cond_blend_weights` is computed O(T²) per call, and it is loop-invariant.** (`:238–241`)
For each tile, `tileModality` iterates **every other tile** re-running `_keep_mask` to count how
many tiles keep each conditioning token. With T tiles that is T² mask builds *per forward*. But the
counts depend only on `positions` + the tile layout — **both fixed for the whole denoise loop**.
Hoist to a one-time `precomputeCondCounts()` at tiler construction. For 8 tiles × 8 steps × 2 CFG
passes this is **1024 mask builds → 8**.

**2. `_bool_to_indices` round-trips through NumPy — a GPU→host sync in the hot loop.** (`:56–63`)
`mx.array(np.flatnonzero(np.asarray(mask)))`, because "MLX doesn't support boolean indexing yet."
Called once in `tileModality` and once in `blend`, per tile per forward. 🚨 **Swift-MLX has no
NumPy escape hatch at all**, so this is a porting blocker *and* an optimization opportunity with
the same fix: the keep-indices are **also loop-invariant** — precompute them once per tile at
construction (generated indices are already arithmetic in `_generated_token_indices`; only the
cond-token set needs a materialization, and it is small and fixed). Result: **zero host syncs in
the denoise loop.**

**3. Tiles are forwarded strictly serially, and the oracle names the cost.** (`:352`, and its
"Tradeoff" note: *"each tile is a separate model forward + kernel dispatch… ~2-3x latency"*.)
⚠️ **This is a genuine trade, not a free win — do not oversell it.** Tiles are independent
forwards of the *same* weights, so they cannot run concurrently on one GPU without contending; the
real lever is **micro-batching** — uniform splitting makes every tile the same token count, so
*b* tiles can go through as one batch-*b* forward, converting *b* kernel dispatches into 1. But
batching multiplies activation memory by *b*, which is precisely what tiling exists to cut, so the
benefit is `k²/b`, not `k²`. **Ship it as a dial (`tileBatch`, default 1) and measure the
speed/memory curve** — this is exactly a `--bench-e2e` arm.

**4. Blend accumulation is read-modify-write scatter.** (`:297`, `:305`)
`output[:, genIdx, :] = output[:, genIdx, :] + genPart` — MLX has no in-place mutation, so this
copy-emulates a full `(B, numTotal, D)` buffer per tile per forward. Check whether MLX-Swift
exposes a scatter-add primitive before transcribing the read-modify-write literally.

**5. Audio is recomputed in EVERY tile and averaged.** (`:365`, `:370`)
Audio compute scales linearly with tile count. The oracle's rationale is semantic — *"per-tile
outputs differ only in the joint audio↔video cross-attention contribution"* — so the average is
meaningful, not redundant. ⚠️ **Do not "optimize" this away without understanding the cross-attention
semantics**; note it, measure its share, and revisit only with a parity gate in hand. Flagged
because for an audio+video model at high tile counts it is a real and growing cost.

## ✅ Port surface — mapped and verified 2026-08-04

**The seam is a single call.** `DenoiseLoop.x0` (`DenoiseLoop.swift:38–42`) is the **only** production
DiT invocation; `run` and `runConditioned` both funnel through it, and `Sources/MLXLTX2/` never
touches the DiT directly. `TiledDiT` intercepts exactly there.

`DiT` is a **`public struct`** whose forward is `callAsFunction(videoLatent:audioLatent:sigma:
videoText:audioText:videoPositions:audioPositions:videoTimesteps:audioTimesteps:) -> (video:, audio:)`
(`DiT.swift:141–146`). Weights hang off a reference (`DiTWeightStore`), so a wrapper holding it by
value shares weights and LoRA with no copy.

**Token geometry is confirmed compatible.** `patchify` is a pure transpose+reshape
(`LTX2Pipeline.swift:399–402`), so the token grid **is** the latent grid at patch size 1, flat index
`f·(H·W) + h·W + w`. `Positions.video` builds `(1, F·H·W, 3)` with the **same F-major/W-minor
flattening** (`Positions.swift:30–43`) — so slicing positions by the same flat indices as the latent
is valid, which the whole tiler depends on.

**Type plumbing (the only edit to existing `Sources/LTX2/` files):** `DenoiseLoop` takes
`dit: DiT` concretely at `:34`, `:67`, `:106`. Introduce `protocol LTXDenoiser` with the identical
signature, conform both `DiT` and `TiledDiT`, and widen those three to `any LTXDenoiser`. ⚠️ Prefer
this over storing an optional tiler *inside* `DiT` — that would entangle tiling with the streaming
branch at `DiT.swift:183` and lose the compose-in-either-order property.

### 🚨 Three hazards found before writing any code

**1. `armStreamingGate` is a bug-in-waiting under tiling — and tiling makes streaming HARDER.**
`LTX2Pipeline.armStreamingGate` (`:58–60`, called at `:441`, `:504`, `:666`) sets
`gateEvaluationThresholdTokens` from the **untiled** `nv + audioT`. But the count the kit actually
measures comes from `beginForward` inside the DiT (`DiT.swift:194–195`) and would be the **per-tile**
count. The kit only evaluates when `forwardTokens >= gateEvaluationThresholdTokens`
(`BlockStreamKit/BlockStreamer.swift:732–734`), so **no forward would ever reach the threshold, the
verdict would stay `.undecided` forever, and the run would stream unconditionally no matter how bad
the arithmetic.** Fix: arm with the **largest tile's** token count (`maxTileGen + audioT +
maxKeptCond`). 🔑 And note the deeper coupling — the gate's `N_min ≈ N·C/S` is computed from that
same N, so **tiling genuinely makes the streaming gate harder to clear**. Two memory levers that
each look free in isolation partly cancel; TILING-PLAN's earlier "composes with block streaming"
line was too optimistic. ⚠️ Separately: `icT2V` **never arms the gate at all** — a pre-existing gap,
worth fixing in the same change.

**2. Our video IC-reference tokens have STRICTLY POSITIVE time, so the oracle's "keep in every tile"
branch is dead code on our video path.** `Positions.video` clamps time at 0 (`Positions.swift:32–33`)
and `ReferenceConditioning.scaledPositions` multiplies by `[1, downscale, downscale]` — the temporal
axis is never scaled and never negative (`ReferenceConditioning.swift:46–50`). So the oracle's
`has_negative_time` escape (`modality_tiling.py:178–180`) never fires for us; IC refs fall through to
the point-in-range test and get distributed across tiles with `1/count` blending. The **only**
negative-time producer in the port is `Positions.patchifyLipdubAudioReference`
(`Positions.swift:61–72`), which is on the **audio** stream — and audio is passed whole to every tile
and averaged, so the video tiler never sees it. **Port the branch for fidelity, but do not assume it
keeps IC refs intact.**

**3. ✅ Timesteps: `nil` must stay `nil` to the model — verified, and it contradicts a plausible
misreading.** The oracle *does* broadcast scalar sigma to per-token timesteps
(`modality_tiling.py:393–397`), but **only into its internal `Modality` so the tiler can slice them
alongside the latent**. At the forward it re-checks `if "video_timesteps" in kwargs and … is not
None` (`:359–360`), so a t2v caller still gets `video_timesteps=None` and the **scalar AdaLN path is
preserved**. 🚨 Broadcasting all the way through to the model would silently move our *shipping* t2v
path from scalar to per-token AdaLN — a real numerical divergence, and a consistent one that a
tiled-vs-untiled gate might not flag as a *tiling* bug. Slice internally; forward `nil` as `nil`.

### Two smaller confirmations

- **No attention-mask parameter exists on our DiT** (`DiT.swift:141–146`; it hardcodes `mask: .none`
  at `:290`). The oracle's `attention_mask` slicing has no counterpart — **drop it, don't invent one.**
- **Do NOT extract a shared tile primitive with the VAE tiler.** They differ on every axis: 5-D
  latent grid vs flat token sequence; disjoint `tileBounds` + halo + **crop** vs overlapping
  **trapezoidal-weighted accumulate**; exactness-by-locality vs blend-because-attention-is-global.
  The only shared idea is "n near-equal parts", and the modality version needs the *overlap-aware*
  formula. Write a fresh `TokenTileGeometry`; leave `VideoVAEDecoder.tileBounds` alone. ✅ Two things
  from the VAE work that DO transfer as doctrine: uniform window shapes (one compiled graph shape
  across tiles) and the per-tile `eval` + `Memory.clearCache()` cadence that keeps the pool
  tile-bound.

## Port plan

1. **Read the primitives** the tiler stands on — `create_tiles`, `split_by_count`,
   `identity_mapping_operation`, `TileCountConfig`, `Tile.blend_mask` in
   `model/video_vae/tiling.py`. ⚠️ Note our Swift VAE tiler was written independently and does
   **not** share these; decide deliberately whether to introduce a shared Swift tile primitive or
   keep the two separate (they solve different problems at different granularities).
2. **Port `Modality` + `VideoModalityTiler` + `TiledDiT`** into `Sources/LTX2/`, structurally
   isomorphic with the oracle (same names, same decomposition — the porting doctrine, and the
   thing the oracle's own history says it lost when it drifted).
3. **Gate** — `--modality-tile-gate` in a new `Sources/RunLTX2/ModalityTileGates.swift` (own file,
   one dispatch line; the TileGates precedent). Golden from the oracle's 8 unit tests
   (`tests/test_modality_tiling.py`, bit-exact at 1e-6): tile/blend round-trips, position
   normalization, cond overlap, wrapper-vs-baseline. **Acceptance: the oracle's own validation bar
   — conservative config (`tile-frames 2`, `overlap 4`) must reproduce the untiled baseline
   bit-identically** (it reports PSNR 228 dB), because saturating overlap makes blend collapse to
   identity. That is a strong, cheap correctness gate. ⚠️ It is explicitly *not* a stress test of
   tile boundaries — add a real 2×2 spatial arm with the seam metrics we already use.
4. **Then** apply optimizations 1, 2, 4 (all pure wins), and measure 3 as a dial.
5. **Receipt** — `--bench-e2e` arms at a geometry that today OOMs or is infeasible: peak phys,
   wall, and output cosine vs untiled where untiled can still run.

---

## 🚨 MEASURED 2026-08-04 — the lever works, but **its stated premise does not hold on our path**

The tiler is built, gated and wired (`--modality-tile-gate` 9/9 exact; `LTX_TILE_SPATIAL` /
`LTX_TILE_FRAMES` / `LTX_TILE_OVERLAP`). First end-to-end numbers, 704×512 bf16, `--speed-bench`:

| geometry | arm | run2 | peak phys |
|---|---|---|---|
| 9f (nv=704) | untiled | **20.6 s** | 52.49 GB |
| 9f | tiled 1×2×2 ov2 (4 tiles, 216 tok) | 38.6 s | 52.50 GB |
| 121f (nv=5632) | untiled | **173.3 s** (ladder median) | 67.76 GB |
| 121f | tiled 1×2×2 ov2 | 200.0 s | **66.11 GB** |

**At 121f tiling buys 1.65 GB (2.4%) for +15% wall.** At 9f it buys nothing for +87%.

### Why — and this is the finding that matters

🔑 **The DiT's activation grows LINEARLY with token count on our path, not quadratically — so the
O(N²) attention wall the oracle built this lever to break does not exist here.** Evidence:

- **Profiler, denoise phase, nv=704:** `act=38.0 cache=3.4 phys=41.8 GB` — i.e. ~3.8 GB of
  activation above the 38 GB of bf16 weights.
- **Whole-run peak at nv=5632 is 67.76 GB**, i.e. ~30 GB of activation. That is **8× the tokens for
  8× the activation — linear.** Quadratic would predict 3.8 × 64 ≈ **243 GB**.
- **Directly corroborated by `--sdpa-mask-probe`:** at N=5632 the call-attributed peak is
  **0.280 GB**, against **2.03 GB** if `[1,H,N,N]` were materialized. MLX's fused SDPA is genuine
  flash attention — it never forms the scores tensor.

The oracle's own justification (*"per-layer attention scores… ~10 GB/layer at 720×1280×97; 1080p 8s+
doesn't fit any current Apple Silicon"*) is **computed from `B·H·N²·2` bytes, i.e. it assumes a
materialized scores tensor.** On MLX with the fused kernel that tensor is never allocated. Tiling
still cuts the *per-token* activations (hidden states across 48 blocks, residuals) — but by **k**,
not **k²**, and that term was never the wall.

⚠️ **And the run peak is often not the DiT at all.** At 9f the profiler's worst-phys by phase is
**encode 52.5 GB** (Gemma) · denoise 41.8 · vae-decode 42.0 — so tiling the DiT cannot move the run
peak by construction at that size. Only past ~65f does denoise become the peak phase.

### Disposition

🟡 **Keep it, default OFF, and stop advertising it as the 1080p-8s+ unlock until that claim is
re-measured on our stack.** It is correct, cheap when unused (`isIdentity` ⇒ the plain path, byte
for byte), and it is the only lever that reduces DiT-phase activation at all. But:

- **Do NOT repeat the oracle's k² framing in any of our docs.** Ours is a ~k reduction of a linear
  term, measured at 2.4% of run peak at 121f.
- **The honest use case is narrower than advertised:** geometries where (a) denoise is the peak
  phase — past ~65f at 704×512 — and (b) the 15%+ wall-clock is acceptable to fit a budget that
  would otherwise OOM. That is a *tier-fitting* tool, not a speed or general-memory tool.
- ⚠️ **`BLOCKSTREAM-EXPANSION-EVAL` §2's open row ("per-block attention transients at big N, Q/KV
  ~12–16 GB at 500k tokens") should be re-derived**, since it rests on the same materialized-scores
  arithmetic. The 20s-4K attention wall may be a *compute* wall (O(N²) time, which is real and
  unchanged) rather than a *memory* wall.
- **The next real question is Gemma**, not the DiT: it sets the run peak at small geometries
  (52.5 GB at 9f) and nothing in this plan touches it.

## Open questions

- Does `--tile-spatial` compose with our **spatial VAE tiling** cleanly? They are independent
  (DiT tokens vs decode activations) but both cost wall-clock; the 4K path would use both.
- ~~Does a mask drop SDPA off the fused kernel onto the materialized `[B,H,Tq,Tk]` path?~~
  ✅ **ANSWERED 2026-08-04 by measurement — `RunLTX2 --sdpa-mask-probe`, 3/3 reproducible.**
  **It does NOT. Masks stay fused.** At the production shape N=5632 (704×512×121f), H=32, D=128,
  bf16, a materialized fallback would allocate **2.03 GB**; measured call-attributed peak:

  | mask | peak attributable to the call | time | vs `.none` |
  |---|---|---|---|
  | `.none` | 0.280 GB | 10.9–11.2 ms | — |
  | `.array` additive bf16 | **0.326 GB** | 19.0–19.7 ms | **1.74–1.77×** |
  | `.array` bool | **0.372 GB** | 19.7–20.0 ms | **1.76–1.83×** |

  The mask arms add only **+0.046 / +0.092 GB** over `.none` — i.e. ≈ the mask tensor itself
  (0.063 GB), nowhere near 2.03 GB. Memory figures were byte-identical across all three runs.
  Correctness bonus: an all-zero additive mask and an all-true bool mask are semantic no-ops and
  both returned **cos = 1.000000** vs `.none`, confirming float⇒additive / bool⇒keep semantics.
  **So the memory math behind tiling holds.**

  🔑 **But masks cost ~1.75× in attention TIME** (stable to ±0.03× across runs at N=5632). That
  is a real tax to carry in any cost model. ⚠️ N=2112 timings are NOT trustworthy — 2.5–4.8 ms
  baselines with ratios swinging 0.97–1.40× across runs, i.e. small-N measurement noise; only the
  N=5632 column should be quoted.

  ⚠️ **Scope correction while establishing this:** modality tiling **does not itself require
  attention masks.** The tiler *slices* an optional `attention_mask` when one is present, but our
  port passes `.none` and has no mask system at all — masks arrive with the **conditioning** work
  (`GAP-ANALYSIS` #6, `mask_utils`). So this finding de-risks two items at once: tiling can proceed
  with `.none` and no attention-memory surprise, and when the mask system lands it stays fused but
  pays the ~1.75× attention tax.

  ⚠️ **Measurement trap banked (v1 of this probe was wrong).** The first version reset the
  peak-memory high-water *before* the warmup call, so the warmup's own peak swallowed the measured
  call and **every arm — including `.none` — read a confident `peak+0.000 GB`**. `GPU.resetPeakMemory()`
  must come *after* warmup and after `Memory.clearCache()`, immediately before the measured call.
  A peak-delta of exactly zero for an arm that provably allocates a 46 MB output is the tell.
- Upstream wires `ic-lora` on the same primitive but the oracle "isn't yet" — check whether our
  IC path would want it.
