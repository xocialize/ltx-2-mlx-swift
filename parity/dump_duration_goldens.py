#!/usr/bin/env python
"""LTX-2.5 duration-head parity fixture (arm D / claim C6).

Feeds the banked REAL connector encodings through the Python-MLX oracle's ``DurationHead``
and stores inputs + predictions, so the Swift port is gated on the production token
distribution rather than synthetic normals.

Three input modes are covered because the head accepts any of them and the concatenation
order (video first) is a silent-failure axis: both, video-only, audio-only.

Also carries the TORCH reference predictions when the oracle's ``.work`` bank has them
(``scripts/dump_ltx25_duration_golden.py``), so the Swift gate can report against upstream
as well as against the MLX oracle it was ported from. The torch numbers are reported, not
gated: the oracle already measured ~2-6e-4 relative agreement, an amplification of ordinary
fp32 accumulation through a 1-query/2048-key softmax, not a math difference.

    cd .../ltx-2-mlx && uv run --no-sync python .../ltx-2-mlx-swift/parity/dump_duration_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "goldens" / "duration"
MODEL_DIR = Path("/Volumes/Satechi/Models/xocialize/ltx-2.5-mlx")
REF_BANK = Path("/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx/.work/ltx25-ref/goldens")


def main() -> None:
    from ltx_core_mlx.duration_head import DurationHead, seconds_to_num_frames
    from ltx_core_mlx.utils.weights import load_split_safetensors

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    head = DurationHead()
    w = load_split_safetensors(MODEL_DIR / "duration_head.safetensors", prefix="duration_head.")
    head.load_weights(list(w.items()), strict=True)
    mx.eval(head.parameters())
    print(f"  strict load OK ({len(w)} tensors)")

    v = mx.array(np.load(REF_BANK / "connector_video_encoding_fp32.npy"))
    a = mx.array(np.load(REF_BANK / "connector_audio_encoding_fp32.npy"))
    print(f"  video tokens {v.shape}  audio tokens {a.shape}")

    out: dict[str, mx.array] = {"video_tokens": v, "audio_tokens": a}
    cases = [("both", v, a), ("video_only", v, None), ("audio_only", None, a)]

    for name, vt, at in cases:
        y = head(vt, at)
        mx.eval(y)
        out[f"seconds_{name}"] = y
        secs = float(np.array(y)[0])
        print(f"  {name:11s}: {secs:9.6f} s -> {seconds_to_num_frames(secs, 24.0):3d} frames @24fps, "
              f"{seconds_to_num_frames(secs, 30.0):3d} @30fps")

    # torch reference, if the oracle's bank has it — reported by the gate, not gated on.
    for name, key in (("both", "duration_both"),
                      ("video_only", "duration_video_only"),
                      ("audio_only", "duration_audio_only")):
        p = REF_BANK / f"{key}.npy"
        if p.exists():
            ref = mx.array(np.load(p))
            out[f"torch_seconds_{name}"] = ref
            rel = abs(float(np.array(ref)[0]) - float(np.array(out[f"seconds_{name}"])[0])) \
                / abs(float(np.array(ref)[0]))
            print(f"  {name:11s}: torch ref {float(np.array(ref)[0]):9.6f} s  rel {rel:.2e}")

    # Frame-grid expectations, so the Swift snap is checked against numbers the oracle
    # produced rather than ones the Swift side re-derives from its own prediction.
    grid = []
    for name, _, _ in cases:
        secs = float(np.array(out[f"seconds_{name}"])[0])
        grid.append([float(seconds_to_num_frames(secs, 24.0)), float(seconds_to_num_frames(secs, 30.0))])
    out["frames_24_30"] = mx.array(grid)  # rows in `cases` order

    # Clamp boundaries: the head's raw output is unbounded, so the pipeline's clamp is what
    # keeps a wild prediction on the grid. Pinned here at both ends.
    out["clamp_probe_seconds"] = mx.array([0.2, 99.0])
    out["clamp_probe_frames"] = mx.array(
        [float(seconds_to_num_frames(s, 24.0)) for s in (0.2, 99.0)])

    mx.save_safetensors(str(OUT_DIR / "io.safetensors"),
                        {k: val.astype(mx.float32) for k, val in out.items()})
    print("wrote", OUT_DIR / "io.safetensors")


if __name__ == "__main__":
    main()
