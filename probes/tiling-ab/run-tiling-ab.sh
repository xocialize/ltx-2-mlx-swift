#!/bin/bash
# SPATIAL TILING PERCEPTUAL A/B at 1080p — the gate on AB-R-0125's 38% memory win.
#
# 🔑 THIS IS A SIMILARITY QUESTION, NOT A PREFERENCE ONE — and the distinction decides the metric.
# Tiling changes only the DECODER path: identical latents in, so the two arms SHOULD reconstruct the
# same picture. That makes SSIM/PSNR meaningful, exactly as in the PrunaVAED decoder A/B (SSIM
# 0.9825 / PSNR 39.87 dB). ⚠️ Contrast the ENCODER A/B (AB-R-0104), where conditioning changed and
# a low SSIM would have meant nothing — reporting one there would have been a category error.
#
# ⚠️ WHAT TO LOOK FOR IS SEAMS, NOT "QUALITY". Tiling decodes overlapping windows and crops; the
# failure mode is a visible discontinuity at tile boundaries — a 2x2 split puts one vertical seam
# down the middle and one horizontal across it. General softness or a different-looking clip is NOT
# the failure mode here and would indicate something else went wrong.
#
# ⚠️ HALO 5 IS NOT BIT-EXACT (~74 dB seam). And bit-exact is UNAVAILABLE at 1080p: at halo 16 the
# window (62 cells) exceeds the 60-cell grid axis. So the real choice is halo 5 or no tiling —
# there is no lossless tiled option at this resolution.
#
# 121f (~5 s), not 241f: seams are SPATIAL, so frame count does not bear on the question, and this
# halves the slot. Same seed (pinned 42 in the harness), same geometry, one variable.
set -u
cd "$(dirname "$0")/../.." || exit 1
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
OUT=~/Desktop/ltx25-tiling-ab
mkdir -p "$OUT"
BIN=./.build/release/RunLTX2
xcrun swift build -c release --product RunLTX2 2>&1 | grep -E "error:|Build complete" | tail -1
if pgrep -f "RunLTX2" >/dev/null; then echo "🚨 another RunLTX2 is live — ABORTING"; exit 2; fi

# A prompt with FINE DETAIL and STRONG STRAIGHT EDGES across the frame — seams hide in soft or
# busy content and show against continuous structure spanning a tile boundary.
P="a slow pan across a modern glass office tower at dusk, sharp straight window frames, thin cables, city lights reflecting"

arm() { # tag  tiles
  local tag="$1" tiles="$2" t0=$(date +%s)
  echo "── $tag (LTX_VAE_TILES=$tiles)"
  env LTX_TIER=max128 LTX_QUANT=bf16 LTX_STREAM_25=1 LTX_ENVELOPE_OVERRIDE=1 \
      LTX_VAE_TILES="$tiles" LTX_T2V_PROMPT="$P" LTX_T2V_SAVE="$OUT/$tag.mp4" \
    "$BIN" --t2v-spot25 1920 1088 121 > "$OUT/$tag.log" 2>&1
  grep -E "SPLIT|ACCEPTANCE" "$OUT/$tag.log" | tail -1
  printf "   wall %ss\n\n" "$(( $(date +%s) - t0 ))"
}

arm tiles-off 1
arm tiles-2x2 2

echo "═══ objective leg — SSIM/PSNR (legitimate here: same latents, decoder-only change) ═══"
ffmpeg -v error -i "$OUT/tiles-2x2.mp4" -i "$OUT/tiles-off.mp4" \
  -lavfi "[0:v][1:v]ssim=stats_file=$OUT/ssim.log" -f null - 2>&1 | tail -2
ffmpeg -v error -i "$OUT/tiles-2x2.mp4" -i "$OUT/tiles-off.mp4" \
  -lavfi "[0:v][1:v]psnr=stats_file=$OUT/psnr.log" -f null - 2>&1 | tail -2
echo "⚠️ Measured over H.264-encoded frames, so both numbers are CONSERVATIVE — codec noise is"
echo "   counted against tiling. Same caveat the pruna A/B carried."
