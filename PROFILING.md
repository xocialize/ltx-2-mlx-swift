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

**Fix (applied): `DiT.warmup()` — a tiny nv=1 forward during `load()`** pre-compiles the block
kernels in the "Loading" phase. Measured 24f: s1-step0 **162 s → 4.1 s**, generation total **218.8 s →
60.4 s**; the compile now shows as a `DiT kernel warmup: …s` line in load, not a mid-generation hang.
(On a cold machine the warmup itself takes the full compile time — but it lands where slowness is
expected. Disable with `LTX_NO_WARMUP=1`.)

Further lever (not applied — needs parity re-validation): replace the manual
`rms0`/`layerNormAffineFree`/`pixelNorm` (mean/rsqrt chains → hundreds of kernels across 48 layers)
with fused `MLXFast.rmsNorm`/`layerNorm` — fewer kernels → faster compile *and* faster steps.

### 2. H.264 MP4-encode stall (post-generation — the ">1000s, looks like a loop")

After denoise+decode finish (~100 s for 48f), the wrapper's `encodeMP4` can **spin-wait forever** in
`while !input.isReadyForMoreMediaData { … }` at ~0 % CPU — the VideoToolbox H.264 encoder stops
draining at higher frame counts (GPU/memory contention with the 38 GB resident model + ~20 GB MLX
cache). **24f encodes fine; 41 frames hung indefinitely** (confirmed by sampling the process at
`FrameCodec.swift:93`). This is literally a loop and matches "slows to the point it looks like a loop,
GPU <10%, may be processing." It was previously **uninstrumented** (post-pipeline), hence invisible.

The `encode-mp4` span + per-frame progress + a **90 s stall timeout** (fail loud, not hang) pinpoint
it: the encoder accepts frames 0–31 then **hangs at frame 32/41** — `isReadyForMoreMediaData` never
recovers.

**Isolated with `RunLTX2 --encode-stress N [--hog] [--software]`** (encodes N synthetic frames, no
4-min generation):

| Experiment | Result |
|---|---|
| hardware, 41 frames, no pressure | ✅ 1.2 s |
| hardware, 41 frames, +38 GB idle allocation | ✅ 1.2 s |
| hardware, right after a real LTX generation | ❌ hangs at frame 32/41 |
| **software, right after a real LTX generation** | ✅ **3.7 s, valid MP4** |

So it is **not** frame count and **not** memory pressure — the **hardware VideoToolbox media engine
contends with MLX's active post-compute GPU context** (`AVAssetWriter` *is* VideoToolbox; it doesn't
offload it). `Memory.clearCache()` before encode does **not** fix it.

**Fix (applied): `encodeMP4` defaults to the SOFTWARE H.264 encoder.** It bypasses the hardware media
engine — 41 frames in ~3.7 s, no stall, and the full 48f run completes. ~2.5 s slower than hardware's
best case, trivial next to the ~100 s generation, and reliable (LTX output *always* follows heavy MLX
compute). `LTX_ENCODE=hardware` (or `software: false`) opts back into hardware for callers that don't.
A future hardware-speed option is driving `VTCompressionSession` directly with explicit
`VTCompressionSessionCompleteFrames` drains (the pattern in `h00mankind/MetalVideoEngine`), plus a
zero-copy IOSurface frame handoff to replace the current per-frame GPU→CPU readback + Swift pixel
copy.
