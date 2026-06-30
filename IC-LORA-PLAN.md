# LTX-2.3 IC-LoRA — Scoping Doc (L4, dedicated session)

**Status:** scoped, NOT started. This is the focused follow-on to `LORA-PLAN.md` (the runtime/plain
LoRA capability, which is DONE + validated). Read `LORA-PLAN.md` first — IC builds on its hook,
registry, and cache.

**One-line framing:** an IC-LoRA is the *same* low-rank weight delta as a plain LoRA **plus** a
runtime requirement to feed an aligned **reference signal** into the DiT. So IC = (the L1 weight
hook we already have) + (a new reference-conditioning input path) + (a way to *obtain* each LoRA's
particular reference). The weight side is solved; L4 is about the reference side.

---

## What's already in place (IC reuses, doesn't rebuild)
- **Weight application** — `DiT.dense` runtime low-rank add + `LTX2LoRA` (apply/detach/remap) +
  `LoRAStore`. An IC-LoRA's *weights* load through this unchanged (same diffusers-PEFT dialect).
- **Registry + lazy HF cache** — `LoRARegistry`/`LoRACache`. IC extends the *schema* (below), not
  the download machinery (HF resolve/main; a Civitai scheme is a separate small add if needed).
- **Per-request selection** — `metaData` works for scalars; it does NOT carry a reference video
  (see Contract, below).

## The core insight: generalize the INJECTION, not the ACQUISITION
- **Injection (generalizes → port once):** VAE-encode the reference → positionally align + concat it
  into the DiT latent sequence so the adapted attention reads it in-context. This is the
  `ic_lora.py` / `iclora_utils.py` logic in `Lightricks/LTX-2` (`ltx-pipelines`). ONE code path,
  reused by every IC-LoRA. (`hdr_ic_lora.py` is a worked variant to study.)
- **Acquisition (varies → typed per-entry, NOT one generic UI):** where the reference comes from
  differs per LoRA. This is what the registry must capture and what drives the UI.

## Acquisition taxonomy → registry `ReferenceRequirement` (the schema IC adds)
Each IC registry entry declares a typed requirement; the type drives the UI affordance and the
acquisition code. Best-guess classification (CONFIRM per card/trainer config in the session):

| Requirement type | UI affordance | Candidate IC-LoRAs |
|---|---|---|
| `userVideo` (1 ref clip) | reference-video picker | Upscale_IC, v2v / ICEdit |
| `userVideo` (N ref clips) | N pickers | Ingredients (multi-reference compose) |
| `autoDerivedFromInput\|Output` | none (or a checkbox) | Detailer (refine pass over input/base output), HDR (SDR→HDR) |
| `extractedControl(kind)` | "extract from source" step + an extractor model | Motion-Track-Control (motion/flow), Cameraman (camera path), depth/pose variants, Water-Simulation (control video) |

Proposed schema (extends `LoRAEntry`):
```jsonc
{
  "id": "...", "displayName": "...", "repo": "...", "weightFile": "...",
  "defaultStrength": 1.0, "trigger": "",
  "reference": {
    "type": "userVideo | userVideoMulti | autoDerivedInput | autoDerivedOutput | extractedControl",
    "count": 1,                      // for userVideoMulti
    "controlKind": "motion|camera|depth|pose|none",
    "align": { "tokenBudget": 0, "downscale": 1.0 },   // positional-concat params
    "preprocess": "none|degrade|downscale|flow|trackpts|camerapath|depth"
  }
}
```

## UI variations to support (driven by `reference.type`)
- `userVideo[Multi]` → 1..N reference-clip pickers (reuse the `initImage`/video-picker pattern).
- `autoDerived*` → no extra input; maybe a strength/intensity slider.
- `extractedControl` → a source picker + an **extract** action backed by an extractor model
  (tracker / camera-path / depth). This is the heavy tail — those extractors are separate models
  and may not exist in-engine yet (scope them as their own deps).
- Always: strength; show trigger if present; show the reference-token budget (cost).

## Contract implications (the real difference from plain LoRAs)
- A reference **video is a real artifact**, not a metaData scalar → it needs a **typed request
  input** (e.g. `referenceVideo` / `controlVideo` / `[referenceVideos]`), not `metaData`.
- **Surface mapping** — several IC-LoRAs map better onto an existing capability than onto a
  `textToVideo` dropdown:
  - Upscale_IC → `videoUpscale`
  - v2v / ICEdit → `videoEdit`
  - Detailer → a refine pass (videoEdit-ish, or a post-step)
  - Motion/Camera/Depth-controlled generation → control-conditioned `textToVideo` with a control input
  - Ingredients (multi-ref compose) → possibly its own surface
- **Conclusion:** IC-LoRAs are NOT just "more dropdown effects." Some are effectively their **own
  capability**. Decide per-entry whether it's (a) a control input on an existing surface, or (b) a
  distinct capability. The registry's `reference.type` + a `surface` field should encode this.

## Phasing (build the generalizable core first, long tail last)
1. **P1 — user-supplied reference (the injection path, proven once).** Pick ONE `userVideo` target
   (Upscale_IC or a v2v IC-LoRA). Port `ic_lora.py` injection into the pipeline; add a typed
   `referenceVideo` input; wire one surface (videoUpscale or videoEdit). Gate: reference in →
   coherent, conditioned output; reference absent → graceful base.
2. **P2 — auto-derived.** Detailer / HDR: the package derives the reference (degrade / SDR→HDR
   transform) — no new user input, reuses P1's injection.
3. **P3 — extracted control.** Motion-Track / Cameraman / depth: add control-signal extraction
   (separate extractor models) + the control-conditioned path. Heaviest; do last.

## Candidate IC-LoRAs to enumerate/classify in the session
- `Lightricks/LTX-2.3-22b-IC-LoRA-{Water-Simulation, Motion-Track-Control, Ingredients}` (22b, right base)
- `Lightricks/LTX-2-19b-IC-LoRA-Detailer` (**19b — base mismatch**; find/await a 2.3-22b detailer)
- Community: `Zlikwid/LTX_2.3_Upscale_IC_Lora`, `joyfox/LTX2.3-ICEdit-Insight`, Cameraman (Civitai)
- Trainer configs to mirror: `packages/ltx-trainer/configs/{v2v_ic_lora,av2av_ic_lora,a2a_ic_lora}.yaml`;
  pipeline refs: `packages/ltx-pipelines/src/ltx_pipelines/{ic_lora,iclora_utils,hdr_ic_lora}.py`

## Open questions for the dedicated session
1. Confirm each candidate's reference requirement from its card + trainer config (the table is best-guess).
2. Surface decision per IC-LoRA: control-input-on-existing-surface vs its-own-capability.
3. Reference encoding/alignment params: token budget, downscale (smaller ref = fewer tokens),
   positional-concat layout — read from `iclora_utils.py`.
4. Which extractor models are needed for `extractedControl`, and whether any already exist in-engine.
5. License per IC-LoRA (base = LTX-2-Community allowlisted; each adapter has its own).
6. First target to build in P1 (recommend Upscale_IC → `videoUpscale`, the cleanest userVideo case).

## NOT in scope for L4 / deferred
- Civitai download scheme (only if a needed IC-LoRA is Civitai-only).
- The plain-LoRA fuse perf-tier (separate, deferred — see LORA-PLAN.md).
