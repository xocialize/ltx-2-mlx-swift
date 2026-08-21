# LTX-2.5 tier-admissibility matrix — raw receipts (2026-08-16)

Raw `--t2v-spot25` logs behind **AB-R-0090**. Fresh boot, quiet box (mactop only), release build,
**serial — one process per arm, never two at once**. Requested `704×512×161` in every arm and let
each profile clamp itself (the T3 methodology); the acceptance number is the 25 ms `PhysSampler`
high-water, never a profiler span.

## The result: neither lever clears a low tier alone

**compact24** — clamped 512×288×121, budget **16.8 GB**

| arm | encoder | DiT | peak | | log |
|---|---|---|---|---|---|
| baseline | bf16 | resident | 31.92 | ❌ | `compact24-int8-resident.log` |
| streaming only | bf16 | **streamed** | 24.79 | ❌ | `compact24-int8-streamed.log` |
| encoder only | **int8** | resident | 31.88 | ❌ | `compact24-encq8-resident.log` |
| **both** | **int8** | **streamed** | **14.57** | ✅ | `compact24-encq8.log` |

**balanced32** — clamped 576×320×161, budget **22.4 GB**: 33.13 ❌ · 24.81 ❌ · 33.46 ❌ ·
**15.38 ✅**

**standard64** — 704×512×161, budget **44.8 GB**: int8 resident 27.31 ✅ / streamed 24.81 ✅;
**bf16 resident 45.27 ❌ → streamed 24.81 ✅**

Constraint-peeling, and the order IS the explanation: the resident peak is **DiT-bound at ~31.9**
(so swapping the encoder there moves nothing, 31.92 → 31.88); streaming removes the DiT floor and
exposes the **encoder floor at ~24.8**; only removing both clears the tier.

## The ~24.8 GB floor is the encoder, and the profiler could not prove it

`inv-256x256x9.log` vs `inv-512x288x121.log`: **24.79** vs **24.80 GB** — a ~50× difference in pixel
volume agreeing to 0.01 GB. A decode- or activation-bound peak scales with geometry; a constant one
is a fixed resident term (the bf16 12B Gemma-4 encoder, 22 GB on disk).

⚠️ `attribution.log` is the **MLX_PROFILE=1** pass and is NOT evidence for the acceptance numbers.
The profiler breaks fusion and inflates decode — it reads `vae-decode/up8` at **26.3 GB**, above the
clean run's whole-run peak, i.e. it would have pointed at the wrong stage. Attribution came from the
clean geometry sweep above.

## Reproducing

```bash
LTX_TIER=compact24 LTX_QUANT=int8 LTX_STREAM_25=1 LTX_ENC=q8 \
  ./.build/release/RunLTX2 --t2v-spot25 704 512 161
```

Granule trees (54 GB, not in git): `/Volumes/Satechi/Models/ltx-granules-25/{bf16,q8}`, laid out by
`ltx-granule-layout <transformer.safetensors> <dir>` — no flags, 2.5 shares 2.3's key dialect.
⚠️ The subdir is keyed by QUANT (`q8`), because `LTX2Configuration.resolvedGranuleDirectory` appends
it to `granuleRootDirectory`; naming it `ditq8` binds in the gates and fails in the real pipeline.

## ⚠️ Admissible ≠ shippable — ⟲ ALL THREE CAVEATS BELOW ARE NOW CLOSED (see the 2026-08-20 section)

~~The int8 encoder's perceptual A/B is not done~~ → **PASSED BLIND**, 4 pairs (AB-R-0104).
~~Single run, one geometry per tier~~ → **3 reps × 3 tiers × 2 modes** (AB-R-0106).
~~DFR/i2v untested~~ → **i2v measured**; **DFR formally DEFERRED** for the base release (AB-D-0034 —
dominated ×1.6–1.8, operator prefers native, and unreachable from the package, so no tier
declaration covers it).

⚠️ One caveat SURVIVES and one is NEW: the numbers here were `.auto`-gated, and at compact24's N
that gate flips run-to-run (AB-R-0105) — the low tiers require `.forceStream`. And **compact24 i2v
sits at 92% of budget**, which is thin (see below).

---

# i2v + corrected footprint — 18 runs, fresh boot, serial (2026-08-20, AB-R-0106)

Closes AB-T-0066, the last measurement gap. Each profile clamps its own envelope from a
704×512×161 request; **compact24 takes `.forceStream` from its profile advisory** (no env override),
balanced32/standard64 stay `.auto`.

**WORST-CASE per tier+mode** — worst, never mean, because compact24's `.auto` headroom once ranged
1.33×…0.59× and a mean would have hidden the failure:

| tier | mode | resident | activation | peak | budget | % of budget |
|---|---|---|---|---|---|---|
| compact24 | t2v | 0.49 | 14.15 | **14.58** | 16.8 | 87% ✅ |
| compact24 | i2v | 0.49 | 15.02 | **15.49** | 16.8 | **92% ⚠️** |
| balanced32 | t2v | 0.63 | 14.86 | **15.42** | 22.4 | 69% ✅ |
| balanced32 | i2v | 0.61 | 16.65 | **17.14** | 22.4 | 77% ✅ |
| standard64 | t2v | 0.63 | 18.80 | **19.43** | 44.8 | 43% ✅ |
| standard64 | i2v | 0.58 | 18.47 | **19.03** | 44.8 | 42% ✅ |

**18/18 ✅ WITHIN.** Every `.auto` run STREAMED on the measured pass (the "fell back resident" lines
present in balanced32/standard64 logs are the **9f warmup**, which falls back at small N by design —
read the LAST gate line, not any of them). compact24 never fell back at all under `.forceStream`.

**i2v costs 0.9–1.7 GB over t2v** and had to be measured: it adds the ~4.9 GB i2v-adapter LoRA, and
the resident/activation split shows why the peak does not move by that much — the adapter is inside
the evicted set, so it lands in activation, not residency.

**The split reproduces the eviction signature** — resident 0.41–0.63 GB against 14–19 GB of
activation. That inversion (tiny resident, activation-dominated) is what the low-water instrument
was adopted to capture, and its consistency across 18 runs is the evidence it is measuring the
right thing.

⚠️ **compact24 i2v at 92% of budget is THIN and should not be declared casually.** This project has
already called out a 96%-of-budget declaration as unsafe (bf16 on standard64, AB-R-0038) — and that
one at least sat *below* its profile's frame cap, where this sits *at* compact24's own clamped
envelope (512×288×121), so there is no headroom hiding in an untested geometry. One prompt was
tested; activation is content-sensitive.

---

# AUTO-FOLLOW shipped (2026-08-21) — and the compact24 i2v note the UI should carry

**Operator decision: the profile's advice is now FOLLOWED automatically**, "to save users from
themselves". Picking a low tier and touching nothing else yields a configuration that FITS; before
this it yielded one that busts the governor, silently.

| axis | default | resolves to |
|---|---|---|
| `LTX2Configuration.textEncoderQuant` | `nil` | profile advice → `.bf16` |
| `LTX2Configuration.forceStreamGate` | `nil` | profile advice → `.auto` |

**Escape hatch (explicit override always wins, both directions):**

```swift
cfg.textEncoderQuant = .bf16     // ignore the tier's int8 advice
cfg.forceStreamGate  = false     // ignore the tier's pinned-gate advice
```

Harness equivalents: `LTX_ENC=bf16|q8`, `LTX_STREAM_GATE=auto|force`. Unset ⇒ follow the profile.

⚠️ **`streamingOptions.gatePolicy` is NOT the knob** — resolution overwrites it. Use
`forceStreamGate`. Package-gate case 33 pins this so it cannot rot silently.

⚠️ **Nothing changes for existing configs.** The advice is `.bf16`/`.auto` on `standard64`/`max128`
— the old defaults — and 2.5 was never admissible on the low tiers, so no persisted 2.5 config can
be sitting on a profile whose advice differs.

Gate: **33/33**, mutation-tested ×2 — disabling auto-follow fails cases 29/30; letting the profile
beat an explicit override fails case 31, the case that exists for it.

## 📋 RECOMMENDED UI ITEM — warn on compact24 + i2v

**compact24 i2v measures 15.49 GB against a 16.8 GB budget: 92%.** It PASSES and is allowed
(operator decision), but the margin is lean enough that the UI should say so.

Suggested behaviour: when the selected tier is `compact24` **and** the request carries an init image
(i2v), surface a non-blocking advisory — something like *"Image-to-video on a 24 GB machine runs
close to the memory limit; close other apps or choose a smaller output if generation fails."*

Why it is worth a warning rather than silence:
- 92% leaves ~1.3 GB. This repo already treats **96%** as unsafe to declare (bf16/standard64,
  AB-R-0038) — and *that* case sat below its frame cap, whereas this sits **at** compact24's clamped
  envelope (512×288×121), so no headroom is hiding in an untested geometry.
- **One prompt was measured, and activation is content-sensitive.** The figure is a sample, not a
  bound.
- t2v on the same tier is 87% (14.58 GB), so the warning is specific to i2v — do not warn on both.
