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

## Open questions

- Does `--tile-spatial` compose with our **spatial VAE tiling** cleanly? They are independent
  (DiT tokens vs decode activations) but both cost wall-clock; the 4K path would use both.
- Our DiT passes `mask: .none` to `MLXFast.scaledDotProductAttention` (fused path). Tiling
  introduces per-tile attention masks — ⚠️ **an additive mask can drop SDPA off the fused kernel
  onto the materialized `[B,H,Tq,Tk]` path** (`mlx-porting`, framework constraints), which would
  *recreate* the very O(N²) tensor tiling exists to avoid. **Verify the fused path still fires
  with the tiled mask before trusting the memory math.** This is the single highest-risk unknown
  in the port.
- Upstream wires `ic-lora` on the same primitive but the oracle "isn't yet" — check whether our
  IC path would want it.
