# a2v three-arm comparison harness (AB-A-0027 → AB-T-0097)

Rescued from a session scratchpad (`/private/tmp`, does not survive reboot). These are the drivers
for the three arms the deferred phoneme-shape study needs.

## The three arms

| arm | driver | interpreter | steps | ~runtime @704×512×241 |
|---|---|---|---|---|
| vendor **dev+CFG** | `vendor-a2v-devcfg-pinned.sh OUT` | `LTX-2/.venv-mps` | 30 | 909 s |
| vendor **distilled** | `vendor-distilled-a2v.py OUT` | **Desktop's own** py3.13 | 8 | 292 s |
| **ours** distilled | `RunLTX2 --a2v-control` | — | 8 | 213 s |

- The distilled arm runs **Desktop's own `DistilledA2VPipeline` under Desktop's own interpreter**
  (`/Applications/LTX Desktop.app/Contents/Resources/python/bin/python3.13`, torch 2.11 + MPS) —
  their shipping code, not a reconstruction.
- `--a2v-control` takes `A2V_AUDIO` / `A2V_IMAGE` / `A2V_OUT`, and `A2V_TIER` (**default max128**).

## Traps that cost real time here

1. ⚠️ **`A2V_TIER=standard64` CLAMPS to 161 frames.** A 241-frame request comes back at 161 with the
   audio truncated, and the clip still looks fine — it is simply not comparable. The smoke now prints
   a loud `⚠️ CLAMPED` line; do not ignore it.
2. ⚠️ **The vendor's audio VAE REJECTS MONO** (`expected input to have 2 channels`). Our reader
   upmixes silently, theirs errors. Convert fixtures: `ffmpeg -i in.wav -ac 2 -ar 48000 out.wav`.
3. ⚠️ **Prompt alone does not reliably produce a talking face.** The first dev+CFG run returned a
   beach scene with no visible face — 16 minutes for an unevaluable clip. **Pin the subject with
   `--image <portrait> 0 1.0`** (vendor) / `kf.initPath` (ours), and LOOK AT A FRAME before drawing
   conclusions.
4. Verify **geometry AND content AND duration** match before comparing. "Evaluable" is not the same
   as "comparable".

## Dub-It — the vendor's dedicated pipeline, and what it actually does

`vendor-dubit.sh OUT` runs first-party `dubit.py` with our `lipdub.safetensors` IC-LoRA
(`reference_downscale_factor=1`). 708 s at 704×512×241.

🔑 **Dub-It does NOT lip-sync to a supplied track — it REPLACES the voice.** Measured: its output
audio correlates with the supplied source at **r = +0.0031**. The code says the same thing: stage-1
audio is DENOISED (not frozen), `s1_audio_latent` is cloned, stage 2 freezes audio at that latent,
and the delivered track is `audio_decoder(s1_audio_latent)` — the generated audio. The reference
video's audio only conditions it, appended at negative RoPE positions.

So its tight mouth agreement is between its OWN generated audio and its own video; nothing had to hit
a fixed target. **Do not read it as "the vendor can lip-sync and we cannot."**

Capability summary for LTX-2.5: **preserve your audio OR get tight mouth agreement, not both.**
a2v preserves (0.999x) with rhythm-level agreement; Dub-It agrees tightly by replacing the voice.
Corroborated externally by ComfyUI #15643 (AB-R-0148) — custom audio, CFG tried, "the mouth moves,
but no synced at all".

⚠️ n=1, on GENERATED reference video, `--reference-strength` untested (defaulted 1.0).

## `FAILED-av-offset.py` — kept deliberately

An objective AV-offset estimator (per-frame motion energy × audio envelope). **It does not work** —
validated against a ladder with known 200 ms steps and the steps did not track (later-100/200/300 all
returned −333 ms), r ≤ 0.09. Kept so nobody rebuilds it. A working version needs mouth landmarks.

Two other instruments also failed here: a dark-fraction mouth aperture (correlated NEGATIVELY with
audio; could not separate a frame pair that was visibly open-vs-closed), and a correlation quoted
without a reference scale — which produced a wrong conclusion that had to be retracted.

🔑 **Compute the vendor-vs-vendor baseline FIRST.** Two vendor arms agree with each other only
r=+0.125 at 0 ms (best +0.527 at +83 ms). Any ours-vs-vendor number is uninterpretable without that
reference distribution.

## Results so far (n=1 clip — not sufficient for a verdict)

- **SYNC:** vendor dev ✅, vendor distilled ✅, ours ✗ ~200 ms → the offset is OURS, and the tier,
  the audio encoder (ours and the oracle place a burst in the same token at the same +70 ms),
  position registration and content are all excluded. Frozen-track alignment is the remaining
  suspect. **This part is solid.**
- **SHAPE:** inconclusive; deferred (AB-T-0097). Ours sits BETWEEN the two vendor arms on the
  clearest vowels, and the vendor arms disagree with each other as much as we disagree with either.
