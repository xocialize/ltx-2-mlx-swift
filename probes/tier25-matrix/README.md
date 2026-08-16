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

## ⚠️ Admissible ≠ shippable

The int8 encoder's **perceptual A/B is not done** — its sample MOVES (mean −0.1421/std 0.4253 vs
bf16 −0.1274/0.4222) and it is gated on numbers only. Single run, one geometry per tier. DFR/i2v
untested in this config. Do not turn these into a tier declaration without the perceptual pass.
