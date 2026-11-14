# Feed-forward chunking and the stage-2 AdaLN table (AB-T-0484, 2026-11-14)

Two exact changes NVlabs/Sana applies to LTX-2.5 (`sol-engine` @ `cf67a3a`,
`models/ltx25/RTX5090/memory.py` and `exact_adaln.py`), measured on the Swift port.

**Verdicts.** Feed-forward chunking ships, on by default (`DiT.ffChunking`), with one change from
upstream: equal-size chunks. The AdaLN table is not built: the work it would save is 0.006–0.02%
of a forward.

Build: Release (`xcrun swift build -c release --product RunLTX2`), Swift 6.4.2, macOS 27.3,
mlx-swift 0.32.3 (`19601207`), Apple M5 Max 128 GB, branch `ffn-chunking` off `main` @ `1ec183f`.
Weights on `/Volumes/DEV_VOL1/Models/Models` (`LTX_MODELS_ROOT`). Memory figures are bytes / 1e9.

## What was measured

One DiT forward at a stage-2 geometry (`--ffn-chunk-probe`): no encode, no decode. All arms run in
one process on the same weights and inputs, interleaved with the order reversed on the second rep,
`Memory.clearCache()` before each forward. `phys peak` is the `phys_footprint` high-water from a
25 ms sampler; `mlx active` is mlx's own peak active memory, reset per forward.

Arms: `off` = unchunked; `lazy` = chunked, no `eval`; `eval` = chunked, `eval` per chunk (ships).

### max128 — 1920x1088x481, bf16, streamed (gate pinned), 124,440 video tokens, 8 chunks of 15,555

| arm | phys peak (2 reps) | mlx active | step time (2 reps) | max\|Δ\| vs off |
|---|---|---|---|---|
| off | 60.80 / 60.86 GB | 34.18 / 34.24 GB | 841.3 / 842.5 s | — |
| lazy | 55.19 / 55.19 GB | 34.71 GB | 855.1 / 853.7 s | 0 |
| **eval** | **45.97 / 45.97 GB** | 35.87 GB | 844.9 / 843.2 s | **0** |

Per-chunk `eval` lowers the forward's peak by **14.9 GB (−24%)** for +0.2% step time. Lazy
chunking keeps most of the peak and is 1.5% slower, so the `eval` is what does the work.
The drop is in mlx's buffer pool, not in active memory (which is 1.6 GB higher with `eval`):
unchunked, the pool ends up holding 4 GB-class buffers; chunked, it recycles 0.5 GB-class ones.
The pipeline does not cap the pool during denoise, so the probe's pool policy is the pipeline's.

### standard64 — 1280x704x161, int8, streamed (gate pinned), 18,480 video tokens

At the shipped thresholds chunking does not engage here (18,480 < 65,536), so before = after:

| arm | phys peak | mlx active | step time | max\|Δ\| |
|---|---|---|---|---|
| off | 12.31 / 12.32 GB | 7.28 / 7.29 GB | 33.42 / 33.78 s | — |
| eval (same path) | 12.32 / 12.32 GB | 7.29 GB | 33.29 / 33.44 s | 0 |

Probe-only thresholds (`LTX_FFN_MIN_TOKENS=16384 LTX_FFN_CHUNK_TOKENS=9240`, 2 chunks) show what a
lower threshold would do. Not shipped:

| lane | arm | phys peak | step time | max\|Δ\| |
|---|---|---|---|---|
| int8 streamed | off | 12.31 / 12.32 GB | 33.25 / 33.29 s | — |
| int8 streamed | lazy | 12.36 / 12.35 GB | 33.38 / 33.45 s | 0 |
| int8 streamed | eval | 11.03 / 11.03 GB | 33.49 / 33.48 s | 0 |
| bf16 resident | off | 48.54 / 48.55 GB | 32.78 / 32.18 s | — |
| bf16 resident | lazy | 49.10 / 49.10 GB | 33.43 / 32.77 s | 0 |
| bf16 resident | eval | 47.39 / 47.34 GB | 33.47 / 33.53 s | 0 |

### AdaLN prelude (scalar-timestep path)

Timed: both timestep embeddings plus all 8 AdaLN MLPs (upstream's table replaces 2 of them).

| geometry / lane | prelude (median, n=32) | warm forward | share |
|---|---|---|---|
| 704x512x121 bf16 resident | 1.826 ms | 8.23 s | 0.0222% |
| 1280x704x161 bf16 resident | 2.135 ms | 32.66 s | 0.0065% |
| 1280x704x161 int8 streamed | 1.908 / 2.045 ms | 33.88 / 32.43 s | 0.0056% / 0.0063% |
| 1920x1088x481 bf16 streamed | 1.833 ms | 842 s (off arm) | 0.0002% |

Far under the 1% bar. The scalar path already embeds once per forward, so there is nothing for a
table to remove.

## Why the chunks are equal, not upstream's 16,384 + remainder

The first chunked forward was not bit-identical (`smoke-704x512x121-bf16-resident-min1024.out`,
taken with a remainder split). `--ffn-chunk-micro` isolates it to `proj_out`
(`micro-124440-bf16-block0.out`): mlx picks the GEMM kernel from the shape, and for K = 16,384 in,
4,096 out, it takes the split-K kernel when `K >= 3 * max(M, N)`, i.e. at M <= 5,461 rows. The row
sweep shows the switch exactly there: rows computed at M = 5,461 and M = 5,462 differ, and are
M-independent on either side. `proj_in` (K = 4,096) is bit-exact at any M. A short remainder chunk
lands on the other kernel (chunk 20,000 at N = 124,440 leaves 4,440 rows: max|Δ| 4.0 on 3,185
rows). `FFChunking.ranges` therefore splits into `ceil(N / 16384)` equal chunks, which are never
under 8,192 rows. int8 (`quantizedMatmul`) is bit-identical under either split.

## Gate

`RunLTX2 --ffn-chunk-gate` (`gate.out`, PASS): range arithmetic; wiring on the tiny fixture;
the kernel regime with a remainder-split poison control; block 0 of the real bf16 and int8
checkpoints at 70,000 tokens, lazy and eval, max|Δ| must be 0. Mutation-tested twice: upstream's
remainder split in `FFChunking.ranges` fails 5 checks (`gate-mutation-remainder-split.out`),
reversed concat order fails 6 (`gate-mutation-reversed-concat.out`).

## What this does not show

- **Whole-run peak.** The probe measures the DiT forward only. The max128 corner's declared
  whole-run peak is 74.39 GB and the unchunked forward peaks at 60.86 GB, so another phase
  (decode) most likely binds there and the declaration would not move. That is an inference, not a
  measurement; `--t2v-spot25` owns it (AB-T-0340).
- **Where it engages.** Only streams of 65,536+ video tokens: 1920x1088 from 257 frames up. No
  `standard64`, `balanced32` or `compact24` geometry reaches it.
- **GPU sharing.** Another session's worker used the GPU in short bursts during the session. Step
  times were stable within each arm (max128: under 2 s spread on ~843 s), but the box was not
  guaranteed idle.
