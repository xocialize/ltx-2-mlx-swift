# Profiling the LTX-2.3 pipeline

An env-gated harness (`LTX2Profiler`) instruments generation so you can see **where wall-clock goes**
and tell **compute-bound** from **memory-bound (paging)** from **a stalled encoder**.

## Usage

Set `LTX_PROFILE=1` in the environment of whatever drives the pipeline:

```bash
# Headless app autorun (LTXVideoTesting) — reproduce a specific envelope:
LTX_AUTORUN=1 LTX_QUANT=bf16 LTX_FRAMES=48 LTX_PROFILE=1 <LTXVideoTesting.app binary>
#   LTX_QUANT = bf16 | q8 | q4        LTX_FRAMES / LTX_WIDTH / LTX_HEIGHT override the envelope
```

Or set `LTX_PROFILE=1` in the Xcode scheme's environment and run normally (GUI). `LTX_PROFILE=csv`
also writes `/tmp/ltx-profile.csv`.

Each region logs live as it completes:

```
[LTX-PROF] denoise/s1-step0  162045.8ms  act=38.4 cache=2.5 phys=41.3/115 GB  vN=264 aN=25 σ=1.000
                             ^wall-ms      ^MLX active/cache  ^phys/workingSet   ^geometry
```

`⚠PAGING` appears when `phys_footprint` crosses the Metal working-set ceiling — the GPU is paging
(the classic "<10% GPU, huge wall-clock" signature). A grouped summary prints at end-of-run.

## Diagnostic levers

- `LTX_CACHE_LIMIT_GB=N` — cap the (otherwise unbounded) MLX buffer pool. Test whether an
  overgrown cache is inflating `phys_footprint` into paging.

## Findings (bf16, 704×512, M5 Max / 128 GB, macOS 27 beta)

Two distinct costs dominate — **neither is the DiT compute**, which is small:

### 1. First-forward Metal kernel compilation (denoise side)

The **first** DiT forward pays a large one-time cost; every subsequent step is ~1.5–3.5 s.

| Run | s1-step0 | s1-step1..7 | s2-step (nv=2112) | why |
|---|---|---|---|---|
| 24f, cold shader cache | **162 s** | 1.5 s | 2.1 s | compiles the whole kernel set |
| 48f, warm shader cache | **48.6 s** | 1.5 s | 3.5 s | reuses macOS's on-disk shader cache |

A *larger* shape compiling *faster* proves it's **compilation** (a weight-fault wouldn't shrink across
launches). It is **single-threaded** → the GPU sits <10% and only one CPU core is busy (invisible on a
multi-core monitor), so it **looks like a hang**. MLX specializes kernels per shape, so **each new
frame count recompiles** — incrementing sizes pays it every time, and a cold OS shader cache is the
worst case.

Levers (not yet applied — need parity re-validation): warm the kernels during `load()` (moves the cost
to the expected "Loading" phase); replace the manual `rms0`/`layerNormAffineFree`/`pixelNorm`
(mean/rsqrt chains, hundreds of kernels across 48 layers) with fused `MLXFast.rmsNorm`/`layerNorm`
(fewer kernels → faster compile *and* faster steps).

### 2. H.264 MP4-encode stall (post-generation — the ">1000s, looks like a loop")

After denoise+decode finish (~100 s for 48f), the wrapper's `encodeMP4` can **spin-wait forever** in
`while !input.isReadyForMoreMediaData { … }` at ~0 % CPU — the VideoToolbox H.264 encoder stops
draining at higher frame counts (GPU/memory contention with the 38 GB resident model + ~20 GB MLX
cache). **24f encodes fine; 41 frames hung indefinitely** (confirmed by sampling the process at
`FrameCodec.swift:93`). This is literally a loop and matches "slows to the point it looks like a loop,
GPU <10%, may be processing." It was previously **uninstrumented** (post-pipeline), hence invisible.

Applied here: `Memory.clearCache()` before handing frames to the encoder, an `encode-mp4` profiler
span + per-frame progress, and a **90 s stall timeout** that throws a clear error instead of hanging
forever. With the per-frame trace the stall is now pinpointed: the encoder accepts frames 0–31 then
**hangs at frame 32/41** — `isReadyForMoreMediaData` never recovers.

**`clearCache()` alone does NOT fix it** (measured — still stalls at frame 32): freeing the buffer
pool isn't enough because the **38 GB DiT stays wired** and keeps contending with the VideoToolbox
media engine. The next lever to try (a design change, hence not yet applied) is **evicting the DiT
before `encodeMP4`** so the hardware encoder has the whole GPU — at the cost of a reload on the next
request (weigh against the resident-DiT/LoRA design). Fallbacks: a software/AVFoundation encoder path,
or chunked encoding with explicit drains.
