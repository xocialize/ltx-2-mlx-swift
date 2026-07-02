# LTX-2.3 IC-LoRA — Integration Plan (L4, ACTIVE)

**Status:** REVISED 2026-07-02 — oracle mechanics read, candidates confirmed, open questions from
the original scoping (kept at bottom) answered. Follow-on to `LORA-PLAN.md` (plain/runtime LoRA,
DONE + validated). IC builds on its weight hook, registry, and cache.

**One-line framing (unchanged):** an IC-LoRA is the *same* low-rank weight delta as a plain LoRA
**plus** a runtime requirement to feed an aligned **reference signal** into the DiT. The weight
side is solved (L1); L4 is the reference side.

**Design requirement (operator, 2026-07-02):** attachments must be a **generic role-tagged
surface** — new IC-LoRAs are registry DATA (declared slots + a weight file), never new request
fields, bespoke UI, or per-adapter pipeline code.

Oracle reference (ALL of it exists in `LTX_DEV/ltx-2-mlx` v0.14.12, our parity source):
`ltx_pipelines_mlx/ic_lora.py` (generic two-stage), `lipdub.py` (audio variant),
`iclora_utils.py`, `ltx_core_mlx/conditioning/types/reference_video_cond.py`,
`conditioning/mask_utils.py`, `conditioning/types/reference_audio_cond.py`.

---

## 1. Mechanics (read from the oracle — answers original Q3)

1. **Weights:** diffusers/comfy dialect, rank 64–128, same `dense()` targets our `LoRAStore`
   already serves (attn q/k/v/out + ff proj). Rank is free (store is rank-agnostic). One extra
   datum: the safetensors **file-header metadata** carries `reference_downscale_factor` (int,
   default 1) — MLX `loadArrays` doesn't expose it; parse the header JSON directly (trivial).
2. **Injection** (`VideoConditionByReferenceLatent.apply` — ONE code path for every IC-LoRA):
   VAE-encode the reference at `output_res / downscale_factor` (frames snapped to 8k+1), patchify,
   **APPEND to the video token sequence**:
   - `latent = concat(latent, ref)`; `clean = concat(clean, ref)` (refs enter clean);
   - `denoise_mask` extended with `1 − strength` for ref tokens (1.0 ⇒ preserved, never denoised);
   - RoPE **positions appended**, spatial axes × `downscale_factor` (temporal unscaled);
   - **attention mask is `None` when strength == 1.0 and no pixel mask** — the basic path needs
     NO DiT mask support. The (B, N+M, N+M) block mask (`mask_utils.build_attention_mask`) is only
     for sub-1.0 attention strength / masked conditioning → deferred slice.
3. **Denoise:** the per-token-timestep path (ported for i2v, M20) over the extended sequence;
   post-loop **slice the first F·H·W tokens** — ref tokens are never decoded.
4. **Two-stage:** stage 1 = half-res **with LoRA + refs**; stage 2 = upsample + refine **WITHOUT
   the LoRA, no refs** (clean distilled model). Our runtime add makes the detach free
   (`LoRAStore.clear()` between stages); the oracle fuses/unfuses — we deliberately don't.
5. **LipDub audio variant:** decode audio from the source video → audio-VAE encode (bit-exact
   already) → patchify → append to the **audio** stream with **negative-time positions**
   (`patchify_lipdub_audio_reference_latent`); plus the standard IC video reference. Same
   injection pattern, second modality.

## 2. Candidates (2026-07-02 review — answers original Q1/Q5)

| Adapter | Conditioning (CONFIRMED from cards) | Rank | License | Verdict |
|---|---|---|---|---|
| `Lightricks/LTX-2.3-22b-IC-LoRA-LipDub` (0.9) | source video (visual + audio track); downscale 1 | n/s | **LTX-2-community** (= base, engine-admitted) | ✅ incorporate (P3 — adds the audio append) |
| `Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients` (0.9) | ONE composite reference sheet as looped still video (≥121f, matches output length), LoRA strength **1.4**, dual-part prompt ("Reference sheet: … / Generated video: …") | 128 | **LTX-2-community** | ✅ incorporate — **P1 primary** (official, single ref, pure `ic_lora.py` path). Card says dev-trained → perceptual verify on distilled |
| `Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v2` | raw reference video carrying camera motion + optional init image; ≥960×512 recommended; **no extractor** | 64 | ⚠️ "research purposes only, not intended for commercial use" | 🔬 EVAL-ONLY second sample (license-gated registry entry, never default-bundled) |

**Taxonomy corrections vs the original scoping:** Cameraman is `userVideo`, NOT
`extractedControl` (the LoRA itself learned the motion transfer — no camera-path extractor
needed); Ingredients is a SINGLE sheet (`userImage`-as-looped-video), not `userVideoMulti`. The
`extractedControl` tail (Motion-Track, depth/pose) remains real but moves entirely to a later
phase — none of the current candidates need it.

All three target **LTX-2.3-22B = our base** (`dgrauet/ltx-2.3-mlx` is the 22b conversion; the
oracle's IC pipelines run against it). P0 verifies weight-shape fit regardless.

## 3. The generic attachment surface (answers original Q2 + contract question)

Three layers, each generic; "generalize the INJECTION, type the ACQUISITION" survives, but
acquisition typing lives in the registry, not the contract:

**(a) Engine contract (MLXToolKit) — the durable seam.** ONE canonical addition instead of
per-adapter fields (improve-mlxengine mandate):

```swift
/// Role-tagged conditioning input. Roles are free-form strings DEFINED BY the consuming
/// package/adapter ("reference_video", "dub_audio", "reference_sheet", …). The contract
/// standardizes only the envelope — new adapter types never touch the contract again.
public struct ConditioningInput: Sendable, Codable, Equatable {
    public let role: String
    public let media: ConditioningMedia   // .video(Video) / .audio(Audio) / .image(Image)
    public let strength: Double?          // nil = adapter default
}
// T2VRequest gains: public let conditioning: [ConditioningInput]?
```

Generic beyond LTX (VACE control videos, SCAIL pose refs, Phantom subject sets). **Interim:** the
package accepts the same payloads via package-specific `metaData` (`ltx.conditioning`) so the port
doesn't block on the engine tag; swap when the contract lands (P5).

**(b) Registry v2 — adapters as data.** LIVES IN ITS OWN REPO as of 2026-07-02:
**`xocialize/ltx-lora-registry`** (private; local `LTX_DEV/ltx-lora-registry/`) — the editing/
review surface where schema + entries are shaped BEFORE app integration; `MLXLTX2` vendors a copy
(sync on change; remote-fetch consumption optional later). Schema v2 (see the repo README for the
full field table) — `LoRAEntry` grows (supersedes the original `ReferenceRequirement` sketch;
same idea, role-shaped):

```jsonc
{ "id": "ingredients", "kind": "ic",                 // "plain" (default) | "ic"
  "license": "LTX-2-community",                       // per-entry; non-permissive ⇒ gated
  "conditioning": [                                   // declared attachment SLOTS → drive the UI
    { "role": "reference_sheet", "media": "image", "required": true,
      "ingest": "loopedStillVideo", "note": "composite sheet, black bg, no text" }
  ],
  "loraStrength": 1.4, "stage2": "clean",
  "promptConvention": "ingredients-dual-part",        // hooks the prompt enhancer (P7)
  "surface": "textToVideo" }                          // per-entry surface mapping (original Q2)
```

UI renders slots FROM declarations; pipeline routes roles generically. Adding Cameraman later =
one JSON entry + its license gate, zero code.

**(c) Package pipeline — one IC path.** `LTX2Pipeline` gains a single `icConditioned` two-stage
variant consuming `[role: encodedRef]`; LipDub's audio append keys off the `dub_audio` role, not a
bespoke pipeline. `ingest` hints (`loopedStillVideo`, `videoClip`, `audioTrack`) are the only
per-kind code, shared across adapters.

## 4. Phasing (each gated; original P1-first-injection strategy kept)

- **P0 — scoping gates: ✅ DONE 2026-07-02 (LipDub + Cameraman; Ingredients pending one click).**
  Weights cached at `/Volumes/DEV_ARCHIVE/models/loras/ltx-ic/`. Header probes: both diffusers-PEFT
  (`diffusion_model.*`), all 48 blocks, `reference_downscale_factor: 1` in metadata (embedded
  license text too — LipDub carries the full LTX-2 Community agreement). **LipDub = rank 128,
  2688 tensors, FULL joint-AV coverage** (video attn1/attn2/ff + audio branch + both cross-modal
  attentions). **Cameraman = rank 64, 960 tensors, video-only.** Live `--lora-gate` on the real
  22b distilled DiT: LipDub **1344/1344 applied, PASS** (on 0.809 / off 0.99991 / detach 1.0,
  finite); Cameraman **480/480, PASS** (0.786 / detach 1.0). ⇒ the weight half of IC is fully
  proven; ONLY the injection path remains. **Ingredients:** HF gate (`gated: auto`) not yet
  accepted for `xocialize` — accept on the repo page once, then rerun the recipe.
- **P1 — injection core (LTX2), Ingredients as the proving adapter:** extend
  `DenoiseLoop.runConditioned` to appended tokens (concat latent/clean/mask/positions w/ spatial
  scaling; per-token σ exists; post-loop slice). **Gate:** tiny-scale oracle parity
  (`--ic-tiny-gate`, goldens from a seeded `ic_lora.py` run) + full-scale 1-step cosine ≥ 0.999
  (M9 doctrine).
- **P2 — reference ingestion:** 8k+1 frame snap; VAE-encode ref at downscaled res (encoder
  bit-exact); looped-still tiling for sheets. **Gate:** encode-path parity vs `iclora_utils`.
- **P3 — two-stage IC pipeline + LipDub audio:** stage 1 LoRA+refs half-res → upsample → stage 2
  clean; audio-VAE ref + negative-time positions. **Gate:** e2e perceptual live runs (Xcode
  agent): Ingredients consistency-vs-sheet; LipDub lip-sync readable.
- **P4 — MLXLTX2 wrapper + registry v2 + tiers:** conditioning intake (metaData interim), role
  routing, per-entry license gate, footprint re-measure. Ref tokens ≈ target tokens at downscale 1
  ⇒ stage-1 seq ~2× (attention FLOPs ~4× in stage 1) — measure, then likely **standard64+ only**
  initially; compact tiers **reject-with-reason** (first envelope rejection — clamping away a
  required ref would be silently wrong).
- **P5 — engine contract PR:** `ConditioningInput` on `T2VRequest` (minor contract bump); retire
  the metaData interim.
- **P6 — app (BRIDGE ticket, Xcode agent):** generic attachment-slot UI from registry
  declarations (picker per declared role/media + strength where declared). Acceptance: Ingredients
  + LipDub end-to-end from the GUI; Cameraman behind the license gate.
- **P7 — enhancer synergy (optional):** `promptConvention` → PromptEnhanceKit template
  (Ingredients' dual-part format; the UPE extensible-profile pattern fits exactly).

**Deferred (own slices):** attention-strength < 1.0 / pixel-mask conditioning (needs the (B,N,N)
additive mask plumbed into attn1 — currently `mask: .none`); `extractedControl` adapters
(Motion-Track, depth/pose — extractor models are their own deps); HDR IC-LoRA; auto-derived refs
(Detailer-class — still no 22b Detailer anyway); Civitai download scheme.

## 5. Risks

- **Distilled vs dev base:** oracle runs official 0.9 IC-LoRAs on distilled two-stage; Ingredients
  card says dev-trained — P3's perceptual gate decides (LORA-PLAN risk carried forward).
- **Stage-1 memory:** the 2× token count sets the IC tier floor — measure before any envelope
  promise (LOW-TIER-PLAN discipline).
- **LipDub input semantics:** oracle takes ONE source video (visual + its audio track). True
  "dub with new audio" = the `dub_audio` role overriding the track — default UX decided in P6.
- **Cameraman license:** research-only ⇒ eval-only, gated, never in a shipped manifest.

---

## Appendix — original open questions → answers
1. Reference requirements per candidate → §2 (confirmed from cards; two taxonomy corrections).
2. Surface per IC-LoRA → registry `surface` field; all three current candidates = `textToVideo`.
3. Encoding/alignment params → §1 (downscale from safetensors header; positions spatial-scaled;
   token budget = ref F·H·W at downscaled res).
4. Extractors needed → NONE for current candidates; `extractedControl` deferred wholesale.
5. Licenses → §2 (two community, one research-only-gated).
6. First P1 target → **Ingredients** (official, permissive-as-base, single ref, pure `ic_lora.py`
   path) — supersedes the earlier Upscale_IC suggestion; Upscale_IC remains a fine later add via
   the same path (`surface: videoUpscale`).
