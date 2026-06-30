# LTX-2.3 Runtime LoRA — Implementation Plan

**Capability:** load + hot-switch LoRA adapters on the resident LTX-2.3 DiT, **by extending the
forward pass (runtime low-rank add), not by swapping modules.** Lighter than a module-swap,
survives q8/q4 (low-rank factors stay full precision), and hot-swaps with zero base reload.

**Skills:** `mlx-swift-integration` (engine seam + contract). `mlx-porting` is NOT needed — there
is no new architecture; we extend the already-ported `DiT`.

**Precedent:** `qwen-image-edit-swift` (`QwenImageEditLoRA` + `QwenImageEditLoRASwapper` +
`LoRARegistry`/`LoRACache`). The registry/cache and the dialect parser port nearly verbatim; only
the *application seam* and *key remap* are LTX-specific (Qwen swaps `Linear` modules; LTX has no
module tree, so we add inside `dense`).

---

## Why the approach differs from Qwen (architecture)

LTX-2.3's DiT (`Sources/LTX2/DiT.swift`) is a `struct` with a flat weight dict `w: [String: MLXArray]`
and a functional `dense(x, key) = x @ w[key].T`. There are **no `Linear` modules** to replace with
`LoRALinear`. So:

- **Apply = runtime low-rank add inside `dense`:** `dense(x, key) → x@w[key].T + scale·(x@A)@B`
  when `key` has a registered adapter. Hot-swap = swap a sidecar dict. (The alternative — fusing
  `w[key] += scale·B·A` at load — is simpler but loses sub-ULP deltas in bf16/quant and makes
  swap a reload. Use runtime-add.)

## Target keys & dialect (confirmed from the official trainer)

From `Lightricks/LTX-2` `packages/ltx-trainer/configs/t2v_lora.yaml` (**rank 32, alpha 32 → scale
alpha/rank = 1.0**) and `packages/ltx-core/.../fuse_loras.py`:

- **Dialect:** diffusers-PEFT — `<module>.lora_A.weight` / `.lora_B.weight` (the exact suffixes the
  Qwen `factors()` already handles; community files may also use comfy/kohya — Qwen's multi-suffix
  list covers them).
- **Trained target modules** (per `transformer_blocks.<i>`):
  - video `attn1`/`attn2`, audio `audio_attn1`/`audio_attn2`, AV-cross `audio_to_video_attn` /
    `video_to_audio_attn` × {`to_q`, `to_k`, `to_v`, **`to_out.0`**}
  - `ff` / `audio_ff` × {**`net.0.proj`**, **`net.2`**}
- **Remap (trainer/diffusers naming → this port's `dense` keys):**
  - strip leading `transformer.` / `diffusion_model.`
  - `.to_out.0` → `.to_out`   (port uses a single `to_out`)
  - `.ff.net.0.proj` → `.ff.proj_in`, `.ff.net.2` → `.ff.proj_out` (and `audio_ff.*` likewise)
  - all `attn*` submodule names already match the port — pass through
  - (this mirrors Qwen's `.net.0.proj→proj_in` / `.net.2→proj_out` remap almost exactly)

## Scope boundary

- **IN (v1):** plain **style / motion / likeness** LoRAs — a pure weight delta.
- **OUT (defer to L4):** **IC-LoRA** (Detailer / Water-Sim / Motion-Track / Ingredients). Those need
  a reference-video conditioning input path (`ic_lora.py` / `iclora_utils.py`), i.e. extra aligned
  reference tokens at inference — a separate feature, not a weight add.

---

## Phases (each ends at a gate)

### L0 — De-risk: probe a real LoRA file (no engine code)
- Download `joyfox/LTX-2.3-Transition-LORA` `.safetensors`; dump its tensor keys (python or a tiny
  Swift gate). Confirm: `.lora_A/.lora_B.weight` suffixes, the `to_out.0` / `ff.net.*` module names,
  rank, presence of per-layer `.alpha`, **video-only vs also audio keys**, and which base
  (distilled vs full 22b) the card claims.
- **Gate:** a printed key list + a finalized remap table (adjust the rules above to the real keys).

### L0 — RESULTS (✅ done 2026-06-30, probed `joyfox/LTX-2.3-Transition-LORA/ltx2.3-transition.safetensors`)
- **Dialect:** `diffusion_model.` prefix + `.lora_A.weight`/`.lora_B.weight` (clean diffusers-PEFT; 576 module pairs / 1152 tensors). **No `.alpha`** → scale = strength (alpha-less path). **rank 32**, bf16.
- **Shapes:** `lora_A = [rank, in] = [32, 4096]`, `lora_B = [out, rank] = [4096, 32]` → hook factors `A' = lora_A.T [in,rank]`, `B' = (strength·lora_B).T [rank,out]` (identical to Qwen `factors()`).
- **48 blocks** (matches port `numLayers = 48`); in-dim 4096 matches the port video stream.
- **Targets per block:** `attn1` + `attn2` × {to_q,to_k,to_v,to_out.0}; `ff` × {net.0.proj,net.2}; `audio_ff` × {net.0.proj,net.2}. **No** `audio_attn*` / AV-cross (this is a video-stream + audio-FFN LoRA; keys-present apply leaves the rest base).
- **`dense()` keys on `prefix`** (`w["<prefix>.weight"]`) → sidecar keys on the bare `prefix`; `dense` checks `lora[prefix]`.

**FINAL remap (LoRA key → port `prefix`):** strip `diffusion_model.` (and `transformer.`/`diffusion_model.transformer.` if seen) → then:
| LoRA module | port `prefix` |
|---|---|
| `…attn1.to_q/to_k/to_v` | unchanged |
| `…attn1.to_out.0` | `…attn1.to_out` |
| `…attn2.*` (same as attn1, incl. `to_out.0`→`to_out`) | |
| `…ff.net.0.proj` | `…ff.proj_in` |
| `…ff.net.2` | `…ff.proj_out` |
| `…audio_ff.net.0.proj` | `…audio_ff.proj_in` |
| `…audio_ff.net.2` | `…audio_ff.proj_out` |

(Keep the Qwen multi-suffix + de-kohya handling for community files that use comfy/kohya dialects — this particular file is plain diffusers-PEFT.)

### L1 — DiT `dense` hook + `LoRA.swift` + single-LoRA apply (core)
- `DiT`: add a sidecar `lora: [String: (a: MLXArray, b: MLXArray, scale: Float)]`; in `dense`, when
  `lora[key]` exists, return `base + scale·(x@a)@b`. (Factor `a` = `[in, rank]`, `b` = `[rank, out]`.)
- New `Sources/LTX2/LoRA.swift`: port Qwen's `factors()` (suffix detection, alpha÷rank scale,
  dtype) + `combined()` (multi-LoRA rank-stack); add LTX `remap()`; `apply([(url,strength)], to: dit)`
  populates the sidecar; `detach()` clears it. **Keys-present-driven** (never an all-Linear default).
- **Gate (`RunLTX2 --lora <file> [strength]`):**
  1. **lora-off must be bit-identical to current base** (no regression when the sidecar is empty).
  2. lora-on renders coherent T2V (no NaN) and is visibly different from base.

### L1 — RESULTS (✅ done 2026-06-30)
- `DiT.swift`: added reference-type `LoRAStore` (held by `let lora`) + a 3-line add in `dense`
  (`y += (x·A)·B` when `lora.adapters[prefix]` exists); `hasDenseWeight()` + public `loraTargetCount`.
- `Sources/LTX2/LoRA.swift`: `LoRAStore` + `LTX2LoRA` (factors/combined/remap/apply/detach), ported
  from the Qwen dialect logic; keys-present (skips unmatched targets).
- `RunLTX2 --lora-gate` on the **real 22b distilled DiT** + joyfox transition LoRA:
  - lora-OFF vs golden **0.999912** (no regression — hook is skipped when empty)
  - **576 / 576 LoRA targets resolved, 0 skipped** (remap table exactly right)
  - lora-ON vs base **cosine 0.860, finite** (strong, clean effect)
  - detach vs base **1.000000** (exact restore → hot-swap proven)
- **Distilled-vs-full risk retired:** the LoRA's keys+dims match the distilled DiT and produce a
  finite, coherent-magnitude change. (Full t2v-render eyeball rides L3's app path.)

### L2 — Registry + cache + per-request `metaData` + hot-swap + multi-LoRA
- Port `LoRARegistry` (JSON manifest) + `LoRACache` (lazy HF `resolve/main`, atomic) into
  `Sources/MLXLTX2/`; add `Resources/ltx-lora-registry.json` (entries below).
- `MLXLTX2Package`:
  - `run()`: read `metaData["loraId"]` / `["loraStrength"]`; `cache.ensure(entry)`; apply to the
    resident DiT (hot-swap, no pipeline reload); `detach()` when none selected.
  - optional always-on base LoRA in `LTX2Configuration` (mirrors Qwen Lightning) — leave nil for v1.
  - multi-LoRA: accept an array → `combined()` rank-stack.
- **Contract:** document `loraId`/`loraStrength` on the `T2VContract` descriptor's metaData; this is
  a genuinely package-specific extra (**C5 clean**, not a canonical smuggle). Additive → minor
  package contract-version bump; **no enum change (C12 unaffected)**.
- **Gate:** switch LoRAs across two requests with no reload; footprint delta ≈ Σ rank factors only.

### L2 — RESULTS (✅ code done 2026-06-30; live engine run rides L3)
- `Sources/MLXLTX2/LoRARegistry.swift` — `LoRAEntry`/`LoRARegistry`/`LoRACache`/`LoRAMetaKeys` +
  `MetaValue` ext, ported from Qwen. `Resources/ltx-lora-registry.json` (verified `transition`
  entry: joyfox/LTX-2.3-Transition-LORA, strength 1.0, trigger `zhuanchang`). Target gained
  `resources: [.process("Resources")]` → `Bundle.module`.
- `LTX2Pipeline`: public `setLoRAs([(url,strength)])` / `clearLoRAs()` / `activeLoRATargets`
  (the `dit` is module-internal, so the wrapper drives LoRA through these).
- `MLXLTX2Package`: `registry`/`cache`/`appliedLoRA` state; `load()` builds them (cache under
  `modelsRootDirectory/ltx-lora-cache`); `run()` reads `metaData[loraId|loraStrength]` →
  `cache.ensure` → `pipeline.setLoRAs`, dedups unchanged selections, clears on none. T2V descriptor
  documents the keys (C5/C11). No engine `contractVersion` bump (package tracks `.current`).
- **Offline gate `swift test --filter LoRARegistryTests` → 4/4 PASS** (Bundle.module decode, entry
  resolve, unknown→nil, id-named cache path). MLXLTX2 builds.
- Live per-request hot-swap + footprint-delta across two engine requests = exercised in L3 (needs
  the engine + GPU); the hot-swap *mechanism* was already proven in L1's `--lora-gate`.

### L3 — App dropdown (video validation harness)
- Add an LTX LoRA effect picker to the video testing app (mirror the Qwen Turbo dropdown);
  per-request `loraId`/`loraStrength` through the engine.
- **Gate:** in-app pick → generate → coherent; hot-swap between effects without reload.

### L3 — RESULTS (✅ code done 2026-06-30; live render = user GPU step)
- `LTXVideoTesting/LTXTestView.swift`: `loraOptions` (from `LoRARegistry.bundled()`), `loraId`,
  `loraStrength` state; a **"LoRA (runtime extend)"** GroupBox (effect picker None+entries, strength
  field, trigger-word hint); `T2VRequest` now passes `metaData[loraId|loraStrength]` when selected.
- App consumes the LTX package via `XCLocalSwiftPackageReference ../LTX_DEV/ltx-2-mlx-swift`, so the
  L1/L2 changes flow in. **`xcodebuild -scheme LTXVideoTesting` → BUILD SUCCEEDED.**
- **LIVE RUN 2026-06-30 (704×512×9f, bf16, seed 42):** base run 138.6s / **peak 81.13 GB**; LoRA run
  195.2s / **peak 81.39 GB** → **footprint delta ≈ 0.26 GB = the rank-32 factors only** ✅ (the L2
  criterion). Clean completion, no NaN/crash — the full engine path (metaData → registry → cache →
  hot-swap → render) works end to end.
- **Perf note:** LoRA run ~40% slower (the runtime low-rank add runs rank-32 matmuls on ~576 Linears
  every denoise step). Expected for *extend*; a future optional **fuse tier** (one-time bake for a
  single static LoRA) would remove it, at the cost of hot-swap + precision (the bf16-fuse lesson).
- **Fixed:** output filename now includes the LoRA id+strength tag (`…-transition@1.mp4`) — base and
  LoRA runs were overwriting one shared name, so they couldn't be A/B'd.
- **OPEN (user visual verdict):** does the transition effect read + stay coherent? (Can't assess from
  logs; the two prior MP4s collided to one file — re-run base + transition now that names differ.)

### L4 — (optional, separate) IC-LoRA conditioning path
- Port the reference-video conditioning (`ic_lora.py`) — aligned reference tokens. Unlocks the
  Lightricks Detailer / Control / Ingredients adapters. Scope as its own feature later.

---

## Conformance / contract checklist (C-levels touched)
- **C5** metaData hygiene — `loraId`/`loraStrength` are package-specific extras, ✅ not canonical.
- **C9** `PackageConfiguration` — optional base-LoRA path stays init-time + Codable + defaultable.
- **C12** forward-compat — no capability/quant enum change; switches untouched.
- **C13** IoC — registry/cache/swapper are owned by the package instance the engine constructs;
  apply/detach run on the `@InferenceActor`; nothing self-caches outside the engine's model store.
- **License** — base = LTX-2-Community (allowlisted). Each registry entry records its **own** LoRA
  license; gate/flag non-permissive ones (don't auto-bundle).

## Risks
- **Distilled vs full base.** Port runs `transformer-distilled.safetensors`; community LoRAs may be
  trained on full 22b. Plain style LoRAs usually transfer, but verify per-LoRA at L0/L1.
- **Joint-AV.** A full LoRA carries audio-side weights too; the `dense` hook applies them naturally
  (keys-present). Confirm whether the test LoRAs include `audio_*` keys.
- **Precision.** Runtime-add survives bf16/q8/q4 (factors full precision) — the reason we add, not fuse.
- **Dialect variance.** comfy/kohya key flattening — reuse Qwen's multi-suffix + de-kohya handling.

## Test LoRAs
| LoRA | Type | Source | Role |
|---|---|---|---|
| `joyfox/LTX-2.3-Transition-LORA` | plain motion/transition | HuggingFace | **Primary** (registry/lazy-cache) |
| Anime Style / Gurren Lagann (LTX-2.3) | plain style + likeness | Civitai | style + likeness check |
| valiantcat LTX-2.3 Transition | plain motion | Civitai (HF mirror = joyfox) | second motion sample |
| `Lightricks/LTX-2.3-22b-IC-LoRA-Detailer` (+ Water-Sim) | IC-LoRA | HuggingFace (gated) | **L4 only** |
