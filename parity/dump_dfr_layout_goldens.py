#!/usr/bin/env python
"""DFR canvas-layout golden for the Swift port.

Emits the geometry from `ltx_pipelines_mlx.dfr_layout`, which is itself gated
BIT-EXACTLY against upstream (`gate_ltx25_dfr_layout.py`: 399 segment lengths,
18 canvases, 72 tile partitions, all `==`). So the Swift gate transitively
inherits upstream parity.

    cd .../ltx-2-mlx && uv run python .../ltx-2-mlx-swift/parity/dump_dfr_layout_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ltx_pipelines_mlx.dfr_layout import choose_segment_length, resolve_canvas, tile_ranges

OUT = Path(__file__).resolve().parent / "goldens" / "dfr_layout" / "golden.json"
FRAME_COUNTS = [9, 17, 25, 33, 41, 49, 65, 73, 97, 121, 129, 161, 193, 201, 241, 289, 385, 481]
TILE_COUNTS = [1, 2, 4, 8]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out: dict = {"segment_lengths": {}, "canvases": {}, "tiles": {}}
    for content in range(1, 400):
        out["segment_lengths"][str(content)] = choose_segment_length(content)
    for n in FRAME_COUNTS:
        padded, segment, positions = resolve_canvas(n)
        out["canvases"][str(n)] = {"padded": padded, "segment": segment, "positions": positions}
        for t in TILE_COUNTS:
            out["tiles"][f"{n}:{t}"] = [list(r) for r in tile_ranges(positions, padded, t)]
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT}  ({len(out['segment_lengths'])} lengths, "
          f"{len(out['canvases'])} canvases, {len(out['tiles'])} partitions)")


if __name__ == "__main__":
    main()
