#!/bin/bash
# TILE-COUNT SWEEP + 720p tiling — the measurements that unblock AB-T-0077's re-declaration.
#
# 🔑 WHY THIS, ON A FRESH BOOT. Everything else open is desk work. The cap change is blocked on
# re-declaring max128's footprint, and that declaration needs numbers we do not have:
#   (A) a tile COUNT chosen from more than one sample — 2x2 is our only measured point;
#   (B) whether tiling pays at 720p, where standard64 sits at 42.03/44.8 = 94% of budget — thin by
#       this repo's own standards (92% already earns a UI advisory, 96% is called unsafe).
#
# 🚨 THE DECLARATION RULE THIS MUST RESPECT (AB-R-0107, learned the hard way on streaming):
# TILED NUMBERS MAY ONLY BE DECLARED WHERE TILING IS GUARANTEED. Tiling is opt-in via env today, so
# a declaration would have to cover the UNTILED peak. Both arms are measured at every candidate so
# the fail-closed number exists alongside the optimistic one.
#
# ⚠️ More tiles is not strictly better: peak falls, but seams multiply and halo regions are
# re-decoded per tile. Looking for the KNEE, not the minimum.
# ⚠️ 1080p latent grid is 60x34. 2x2 -> 30x17 tiles; 3x3 -> 20x12; 4x4 -> 15x9 (+ halo 5 each side).
#    At 4x4 the halo starts to dominate the tile — expect diminishing returns and rising wall.
# ⚠️ MEMORY is the deliverable. Fresh boot = right for memory, worst for timing (AB-R-0116).
set -u
cd "$(dirname "$0")/../.." || exit 1
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
OUT=probes/tiling-ab
BIN=./.build/release/RunLTX2
xcrun swift build -c release --product RunLTX2 2>&1 | grep -E "error:|Build complete" | tail -1
if pgrep -f "RunLTX2" >/dev/null; then echo "🚨 another RunLTX2 is live — ABORTING"; exit 2; fi

arm() { # tag tier W H F tiles
  local tag="$1" tier="$2" w="$3" h="$4" f="$5" tiles="$6" t0=$(date +%s)
  printf "── %-22s tiles=%-4s " "$tag" "$tiles"
  env LTX_TIER="$tier" LTX_STREAM_25=1 LTX_ENVELOPE_OVERRIDE=1 LTX_VAE_TILES="$tiles" \
    "$BIN" --t2v-spot25 "$w" "$h" "$f" > "$OUT/$tag.log" 2>&1
  local pk=$(grep "DECLARE" "$OUT/$tag.log" | tail -1 | grep -oE "peakActivationBytes=[0-9.]+ GB" | grep -oE "[0-9.]+")
  local rs=$(grep "DECLARE" "$OUT/$tag.log" | tail -1 | grep -oE "residentBytes=[0-9.]+ GB" | grep -oE "[0-9.]+")
  local g=$(grep -E "\[BlockStreamer\] gate:" "$OUT/$tag.log" | tail -1 | grep -oE "STREAM|FALL BACK")
  printf "peak=%-7s gate=%-6s wall=%ss\n" \
    "$(python3 -c "print(f'{${rs:-0}+${pk:-0}:.2f}')" 2>/dev/null)" "$g" "$(( $(date +%s) - t0 ))"
}

echo "═══ A · 1080p tile-count sweep (1920x1088x121f, max128) ═══"
arm sweep-1080p-t3 max128 1920 1088 121 3
arm sweep-1080p-t4 max128 1920 1088 121 4
echo "   (tiles=1 → 93.68 and tiles=2 → 49.30 already measured, AB-R-0126)"

echo
echo "═══ B · does tiling relieve standard64's thin 720p margin? ═══"
arm sweep-720p-t2 standard64 1280 704 241 2
arm sweep-720p-t3 standard64 1280 704 241 3
echo "   (tiles=1 → 42.03 already measured = 94% of the 0.7x figure, AB-R-0125)"

echo
echo "READ: pick the KNEE, not the minimum — seams and halo re-decode both grow with tile count."
echo "⚠️ Any tiled number is declarable ONLY if tiling becomes GUARANTEED (profile-advised/pinned),"
echo "   exactly as compact24's streamed numbers required .forceStream. Otherwise declare untiled."
