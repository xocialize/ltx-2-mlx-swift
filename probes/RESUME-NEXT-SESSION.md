# RESUME — next session starts here

> ✅ **2026-08-16 — the tier matrix is DONE (AB-R-0090). LTX-2.5 reaches `compact24` and
> `balanced32`, but only with BOTH levers: DiT streaming AND the int8 text encoder.**
> Everything is committed, pushed and gate-green. The single thing standing between that
> measurement and a tier expansion is the **perceptual A/B on the int8 encoder** — below.

Previous entries (all closed): the `dfr=2` probe → AB-R-0073 (C2 finished on cost, ×1.756 at the
481f frame cap); the claims matrix → 6/6 complete, `probes/CLAIMS-BENCH-MATRIX.md`.

## Where things stand

| | |
|---|---|
| **Granule trees** | `/Volumes/Satechi/Models/ltx-granules-25/{bf16,q8}` — 54 GB, on disk, survive reboot. Not in git. |
| **Gate board** | 2.5 streaming green: `--stream-parity-gate 25-{bf16,ditq8}` memcmp-exact, `--stream-auto-gate` 0.0% stall at real N. 2.3 arms unregressed. |
| **Raw receipts** | `probes/tier25-matrix/` (committed — the originals were in `/private/tmp`, which a reboot clears). |
| **Records** | AB-R-0087 (kit 0.5.0 re-verify) · AB-R-0088/0089 (2.5 streams) · AB-R-0090 (tier matrix) · AB-T-0042 done. |

The measured tier picture, so it is not re-derived:

| tier | budget | resident | streamed only | **streamed + int8 enc** |
|---|---|---|---|---|
| compact24 | 16.8 | 31.92 ❌ | 24.79 ❌ | **14.57 ✅** |
| balanced32 | 22.4 | 33.13 ❌ | 24.81 ❌ | **15.38 ✅** |
| standard64 | 44.8 | int8 27.31 ✅ / **bf16 45.27 ❌** | 24.81 ✅ | — |

## The next run: perceptual A/B on the int8 Gemma-4 encoder

**Why it is the blocker.** `AB-D-0014` scoped 2.5 to `standard64` + `max128`. The matrix above
reopens that on memory evidence — but the int8 encoder is **gated on numbers only** and its sample
MOVES (mean −0.1421/std 0.4253 vs bf16 −0.1274/0.4222). CLAUDE.md has flagged "perceptual A/B is
NOT done" since 2026-08-13. No tier declaration can rest on it until an operator has looked.

### 🚨 THE TRAP: this is NOT the pruna A/B, and SSIM/PSNR is the WRONG metric here

The PrunaVAED A/B swapped a **decoder**: same latents in, so both arms encode the *same content* and
SSIM 0.9825 / PSNR 39.87 dB meant "faithful reconstruction". **Swapping the text ENCODER changes the
conditioning**, which changes the denoise trajectory, which produces a **legitimately different
video**. A low SSIM between these two arms would mean nothing at all — and reporting one as a
fidelity failure would be a category error.

**Acceptance is therefore operator preference, not a threshold**: is the int8 arm's output as *good*
— prompt adherence, coherence, artifacts — not as *similar*. Judge several prompts, not one, because
a single pair cannot distinguish "int8 is worse" from "this seed suited bf16".

### Commands

```bash
cd LTX_DEV/ltx-2-mlx-swift
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
xcrun swift build -c release --product RunLTX2

# Same seed, same geometry, same DiT — ONLY the encoder differs.
for enc in "" q8; do
  for p in "a fox running down a beach at sunset, waves rolling in" \
           "a woman's face in close-up, soft window light, she turns and smiles" \
           "a busy night market, neon reflections in wet pavement, handheld camera"; do
    LTX_ENC=$enc LTX_TIER=standard64 LTX_QUANT=int8 LTX_STREAM_25=1 \
      LTX_T2V_PROMPT="$p" LTX_T2V_SAVE=~/Desktop/ltx25-ab/$(echo "${enc:-bf16}-${p:0:20}" | tr ' ,' '__').mp4 \
      ./.build/release/RunLTX2 --t2v-spot25 704 512 121
  done
done
```

Include a **face prompt** — that is what caught the pruna decoder's worst case, and it is where
conditioning error shows first.

### Protocol notes (earned, do not rediscover)

- **Burn the box in** with 2–3 throwaway generations before any *timing* claim; a fresh boot is the
  worst case for timing stability (AB-R-0067). Memory numbers are fine cold.
- **Never run two streaming gates at once** — S is a shared-resource measurement (a q4 arm read
  S=2.15 vs its clean 3.56 GiB/s under overlap; verdicts survive, S/stall numbers do not).
- **Never use `MLX_PROFILE=1` for an acceptance number** — it breaks fusion and inflated
  `vae-decode/up8` to 26.3 GB, *above* the clean run's whole-run peak. Attribution only.
- **`LTX_ENC=q8` swaps the ENCODER, not the DiT.** `ltx-2.5-mlx-q8`'s transformer symlinks back to
  bf16, which is why that tree is a trap for a DiT arm and correct for an encoder arm.

## If the A/B passes

Then — and only then — AB-D-0014's two-tier scope is reopenable. That needs, beyond the A/B:
repeats at each profile's own envelope (these were single runs), plus DFR and i2v under the
streamed+int8-encoder config, neither of which has been measured.

## If it fails

The `compact24`/`balanced32` result does not survive: streaming alone leaves the ~24.8 GB
geometry-independent encoder floor, which is over both budgets. `standard64` bf16 (45.27 ❌ →
24.81 ✅ on streaming alone) is unaffected either way and stands on its own.
