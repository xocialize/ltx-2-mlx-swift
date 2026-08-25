# MPS comparison harness (AB-P-0004 → AB-R-0145)

Ours (Swift-MLX) vs the vendor's own PyTorch-MPS path, same machine, one instrument.
**Rescued from a session scratchpad — that lives in `/private/tmp` and does not survive a reboot.**

## The environment already exists — do NOT rebuild it blindly

`/Volumes/Satechi/Development/LTX-2/.venv-mps` (on the Satechi volume, survives reboot).
Verify before assuming it is gone:

```bash
/Volumes/Satechi/Development/LTX-2/.venv-mps/bin/python -c \
  "import torch, ltx_pipelines, mps_sdpa; from ltx_core.devices import get_preferred_device; \
   print(torch.__version__, get_preferred_device())"
# expect: 2.13.0 mps
```

`mps-env.sh` rebuilds it if needed. Two workarounds are load-bearing and NOT obvious:

1. **Use the venv's own `pip`, never `uv pip`.** `ltx-core/pyproject.toml` points *every* platform's
   torch at `download.pytorch.org/whl/cu132`, which has **no macOS wheels**. `pip` ignores
   `[tool.uv.index]` entirely; `uv` does not.
2. **`torchvision` is required** even though nothing declares it here: `Gemma4UnifiedProcessor`'s
   backing module imports it, and transformers' lazy loader reports only
   `Could not import module 'Gemma4UnifiedProcessor'`. transformers 5.14.1 is *in range*
   (`>=5.8,<5.15`) — the version is a red herring. Import
   `transformers.models.gemma4.processing_gemma4` directly to see the real cause.

## Scripts

| file | what |
|---|---|
| `vendor-run.sh W H F OUT` | arm B — first-party `DistilledPipeline` on MPS, split checkpoints under `Models/Lightricks/LTX-2.5` |
| `mps-matrix.sh SD W H F ROUNDS TAG` | interleaved A1/A2/B matrix with discarded warm-ups |
| `a1prime.sh` / `a3.sh` | the two arms added mid-flight to break a confound (below) |
| `g2.sh` | the 1280×704×121 matrix (A1/A3/B) |

⚠️ `g2.sh` uses `env VAR=val …`, **not** an assignment prefix. `${stream:+VAR=val}` in prefix
position is parsed by zsh as a COMMAND word, so the next assignment becomes the command name
(`command not found: LTX_T2V_PROMPT=…`). This bit twice in one session.

## Why five arms and not two

The two-arm reading said *"vendor 1.78× faster at matched bf16"* — **wrong**. max128 pins the stream
gate, so our bf16 arm streamed 34.56 GiB per forward-set while the vendor's default `--offload none`
stays resident. It compared our disk-streaming path against their resident path and called the
difference "binding". `A3` (bf16 **resident**, `LTX_STREAM_25=0`) is the real like-for-like and it
reverses the sign.

Levers: `LTX_TIER`, `LTX_QUANT`, `LTX_STREAM_25=0` (resident), `LTX_STREAM_GATE=auto|force`.

## Results (medians; raw CSVs beside this file)

| geometry | ours int8 std64 (ships) | ours bf16 resident | vendor bf16 MPS |
|---|---|---|---|
| 704×512×33 | 113.99 s / 24.6 GB | **74.73 s** / 41.3 GB | 94.53 s / 39.8 GB |
| 1280×704×121 | 414.1 s / 27.9 GB | 390.7 s / 42.7 GB | 401.4 s / 41.2 GB |

(footprint column = peak memory footprint; RSS is in the CSVs and is ~2–3.5× lower for us.)

**Speed parity at HD, 3.5× less RSS.** The G1 lead is their fixed overhead amortizing away, not a
durable win — do not quote G1 alone.

## Traps for the next run

- **Profiled totals are NOT timings.** `MLX_PROFILE=1` gave 610 s where the clean run was 390 s; it
  syncs per region and there are 45 decode regions against 11 denoise, so overhead lands unevenly.
  Use it for proportions only, and never in the timing table.
- Their non-denoise cost is **not** a fixed load penalty — it grew 83 → 215 s, because decode scales
  with pixels too.
- Both arms must emit **video + audio**; check the vendor output carries an AAC stream, or one side
  is doing less work.
- Not a pixel comparison — RNG streams differ across bindings.
