#!/usr/bin/env python
"""LTX-2.5 ancestral-Euler step fixture (pure sampler math, injected noise).

The step is gated with a FIXED noise tensor rather than a seeded draw: MLX-Swift and
MLX-Python RNG streams are not bit-identical (determinism doctrine, ISSUES I1), so a
seeded gate would measure the RNG, not the step. Injecting noise isolates the math,
which is the part being ported.

Covers the real distilled sigma pairs plus the two boundary cases: sigma_next == 0
(must return x0 exactly) and eta == 0 (must reduce to the plain Euler step).

    cd .../ltx-2-mlx && uv run python .../ltx-2-mlx-swift/parity/dump_ancestral_step_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.samplers import euler_ancestral_step, euler_step

OUT_DIR = Path(__file__).resolve().parent / "goldens" / "ancestral_step"
B, N, C = 1, 16, 8


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mx.random.seed(3)
    x = mx.random.normal((B, N, C)).astype(mx.float32)
    x0 = mx.random.normal((B, N, C)).astype(mx.float32)
    noise = mx.random.normal((B, N, C)).astype(mx.float32)

    out: dict[str, mx.array] = {"x": x, "x0": x0, "noise": noise}
    sigmas: list[float] = []
    cases: list[tuple[float, float, float]] = []

    pairs = list(zip(DISTILLED_SIGMAS[:-1], DISTILLED_SIGMAS[1:], strict=False))
    for i, (s, sn) in enumerate(pairs):
        y = euler_ancestral_step(x, x0, float(s), float(sn), noise, eta=1.0, s_noise=1.0)
        out[f"eta1_{i}"] = y
        cases.append((float(s), float(sn), 1.0))
        sigmas.extend([float(s), float(sn)])

    # eta = 0 must reduce to the plain Euler step — asserted against euler_step itself.
    s, sn = float(DISTILLED_SIGMAS[1]), float(DISTILLED_SIGMAS[2])
    out["eta0"] = euler_ancestral_step(x, x0, s, sn, noise, eta=0.0, s_noise=1.0)
    out["eta0_plain_euler"] = euler_step(x, x0, s, sn)
    out["meta_eta0_sigma"] = mx.array([s, sn])

    out["n_cases"] = mx.array([float(len(pairs))])
    out["case_sigmas"] = mx.array([[c[0], c[1], c[2]] for c in cases])
    mx.save_safetensors(str(OUT_DIR / "io.safetensors"), {k: v.astype(mx.float32) for k, v in out.items()})
    print(f"wrote {OUT_DIR}/io.safetensors  ({len(pairs)} eta=1 cases + eta=0 reduction)")
    delta = float(mx.abs(out["eta0"] - out["eta0_plain_euler"]).max())
    print(f"  eta=0 vs plain euler maxAbs = {delta:g} (oracle self-check; must be ~0)")


if __name__ == "__main__":
    main()
