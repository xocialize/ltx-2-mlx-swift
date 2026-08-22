#!/bin/bash
# AB-T-0078 — two questions, cheapest first.
#
#  A. 720p on standard64 (a 64 GB Mac). ⚠️ THE BIGGER PRIZE and NOT blocked by anything.
#     Our 44.94 GB 720p figure (AB-R-0118) was measured at max128 with the **bf16** encoder.
#     standard64 AUTO-FOLLOWS to int8 — 13 GB vs 22 — so it may already fit. HD on a 64 GB Mac is a
#     far larger product win than 1080p on a 128 GB one.
#     ⚠️ Compare against BOTH 44.8 (0.7x nominal, what the harness prints) and the Metal-derived
#     budget (~51 GB on a real 64 GB machine) — AB-R-0119: the engine asks the OS, it does not use
#     a flat rate. The harness's denominator is known wrong at max128 (AB-T-0081).
#
#  B. 1080p spatial-tiling A/B. ⚠️ OUR OWN CODE PREDICTS MARGINAL: VideoVAE.swift:85 says tiling
#     "only pays once window << grid — 1080p is marginal, 4K (120x68) is the target". The 1080p
#     latent grid is 60x34; at halo 5, two tiles across width still spans ~67% of it, and at the
#     BIT-EXACT halo 16 the window (62) EXCEEDS the grid (60) — worse than not tiling.
#     Be ready to CLOSE this lever, not to rescue it.
#
# 🚨 NONE OF THESE ARE ACCEPTANCE NUMBERS — all use LTX_ENVELOPE_OVERRIDE=1, and no profile permits
# these geometries. They decide whether the CAP should move. And moving it is BLOCKED on
# re-declaring max128's footprint in the same change (AB-T-0077): the cap is currently the only
# thing keeping the 76 GB declaration honest (48.59 inside it, 97.54 outside).
#
# ⚠️ Memory is the deliverable. Fresh boot = right for memory, WORST for timing (AB-R-0116).
set -u
cd "$(dirname "$0")/../.." || exit 1
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
OUT=probes/hd-viability
BIN=./.build/release/RunLTX2
xcrun swift build -c release --product RunLTX2 2>&1 | grep -E "error:|Build complete" | tail -1
if pgrep -f "RunLTX2" >/dev/null; then echo "🚨 another RunLTX2 is live — ABORTING"; exit 2; fi

arm() { # tag  tier  W H F  extra-env...
  local tag="$1" tier="$2" w="$3" h="$4" f="$5"; shift 5
  echo "── $tag  (${tier}, ${w}x${h}x${f}f)"
  env LTX_TIER="$tier" LTX_STREAM_25=1 LTX_ENVELOPE_OVERRIDE=1 "$@" \
    "$BIN" --t2v-spot25 "$w" "$h" "$f" > "$OUT/$tag.log" 2>&1
  grep -E "request|SPLIT|ACCEPTANCE" "$OUT/$tag.log" | tail -2
  grep -E "\[BlockStreamer\] gate:" "$OUT/$tag.log" | tail -1
  echo
}

echo "═══ A · 720p on standard64 — int8 auto-followed, 2 reps ═══"
arm 720p-std64-r1 standard64 1280 704 241
arm 720p-std64-r2 standard64 1280 704 241
# Control: the SAME geometry on max128 was 44.94 GB with the bf16 encoder (AB-R-0118). If std64
# lands materially lower, the int8 encoder is why — which is the whole hypothesis.

echo "═══ B · 1080p tiling A/B — same geometry, one variable ═══"
arm 1080p-tiles-off max128 1920 1088 241 LTX_VAE_TILES=1
arm 1080p-tiles-2x2 max128 1920 1088 241 LTX_VAE_TILES=2
# ⚠️ halo stays at the default 5 (~74 dB, NOT bit-exact). If tiling pays, a perceptual pass is
# required before any declaration — quality verdicts are the operator's.

echo "READ:"
echo " 1. CONFIRM each request line shows the geometry ASKED FOR — 704x512 means the override failed."
echo " 2. CONFIRM the LAST BlockStreamer gate line says STREAM — lane= reports the CONFIG, not the gate."
echo " 3. A: 720p-std64 vs 44.8 AND vs ~51 (Metal-derived). Under both = HD on a 64 GB Mac."
echo " 4. B: if the tiling delta is small, CLOSE the lever and say so — the code already predicts it."
