# Profiling the LTX-2.3 pipeline

An env-gated harness (the shared **`MLXProfiling`** package — `MLXProfiler.shared`) instruments
generation so you can see **where wall-clock goes** and tell **compute-bound** from **memory-bound
(paging)** from **a stalled encoder**.

## Usage

Set `MLX_PROFILE=1` in the environment of whatever drives the pipeline:

```bash
# Headless app autorun (LTXVideoTesting) — reproduce a specific envelope:
LTX_AUTORUN=1 LTX_QUANT=bf16 LTX_FRAMES=48 MLX_PROFILE=1 <LTXVideoTesting.app binary>
#   LTX_QUANT = bf16 | q8 | q4        LTX_FRAMES / LTX_WIDTH / LTX_HEIGHT override the envelope
```

Or set `MLX_PROFILE=1` in the Xcode scheme's environment and run normally (GUI). `MLX_PROFILE=csv`
also writes the CSV to `MLX_PROFILE_CSV` (default `/tmp/mlx-profile.csv`).

Each region logs live as it completes:

```
[MLXPROF] denoise/s1-step0  162045.8ms  act=38.4 cache=2.5 phys=41.3/115 GB  vN=264 aN=25 σ=1.000
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

### 2. MP4-encode stall (post-generation — the ">1000s, looks like a loop")

**ROOT CAUSE (corrected 2026-07-01): the AVAssetWriter two-track INTERLEAVE deadlock — not GPU
contention.** After denoise+decode finish, `encodeMP4` **spin-waits forever** in
`while !input.isReadyForMoreMediaData { … }` at ~0 % CPU. With TWO inputs (video + LTX's audio track)
and `expectsMediaDataInRealTime = false`, AVAssetWriter interleaves tracks: once the appended video
gets **~1.8 s ahead** of the still-empty audio track, it parks the video input's readiness *waiting
for audio* — and the old code appended audio only **after** all video frames. Deadlock. This matches
"slows to the point it looks like a loop, GPU <10 %, may be processing" exactly, and it was
previously uninstrumented (post-pipeline), hence invisible.

**Decisive isolation with `RunLTX2 --encode-stress N [--hog] [--software] [--audio]`** (synthetic
frames, no generation, no model resident):

| Experiment | Result |
|---|---|
| hardware, 41 frames, **no audio**, no pressure | ✅ 1.2 s |
| hardware, 41 frames, **no audio**, +38 GB idle allocation | ✅ 1.2 s |
| software, 113 frames, **no audio** | ✅ 3.1 s |
| software, 113 frames, **+audio** | ❌ **stalls at frame 43/113** — zero MLX work involved |
| software, 113 frames, +audio, **audio-first fix** | ✅ 3.2 s |
| hardware, 113 frames, +audio, **audio-first fix** | ✅ 3.0 s |

So it is **not** frame count, **not** memory pressure, and **not** the hardware media engine — it is
the second (audio) track. 43 frames @24 fps = 1.79 s = the writer's interleave window.

**History of the misdiagnosis (kept honestly):** the first isolation pass tested hardware-vs-software
and idle-vs-post-generation, but every stress run passed `audio: nil` while every real run muxes
audio — so "hardware after a real generation" (stall at 32/41, WITH audio) vs "hardware idle" (pass,
NO audio) looked like GPU contention. The "software fixes it" validation then passed only because
that clip was **41 frames — just under the ~43-frame software threshold** (hardware's internal
queueing trips a bit earlier, at 32). The user's 120-frame run crossed the threshold and re-stalled
at exactly 43, exposing the misdiagnosis. Lesson: **the stress gate must reproduce ALL tracks the
real path writes — a missing second track silently removes the failure mode.**

**Fix (applied): append the audio sample buffer and `markAsFinished()` the audio input BEFORE the
video frame loop.** The audio track is then complete and the writer never blocks video on interleave.
Validated 113 frames + audio on BOTH encoders (table above). The wait loop also now surfaces
`writer.status == .failed` immediately (a real writer error no longer masquerades as "not ready") and
keeps the 90 s fail-loud timeout. The software default (`software: true` / `LTX_ENCODE`) is retained
as a harmless belt-and-suspenders — note its `RequireSoftwareOnlyVideoEncoder` spec key does not
exist in the SDK (only `EnableHardwareAcceleratedVideoEncoder: false` is real and effective).
Cross-package note: single-track (video-only) writers — e.g. `frame-stream-native` used by
RIFE/SeedVR2 — **cannot** hit this deadlock; their preemptive software-default is likely unnecessary
(the bounded stall-timeout remains worthwhile everywhere).
