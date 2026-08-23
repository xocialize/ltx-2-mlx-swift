#!/bin/bash
# The FALLBACK-LANE measurements a declaration at the new caps requires.
#
# 🚨 WHY. A tier must be declared on the WORST lane it can actually take, not the best one measured.
#   standard64 runs `.auto` — it may FALL BACK to resident on any run (AB-R-0105 measured that flip).
#   max128 defaults to RESIDENT outright (`recommendedStreamedBlocks` is false for it).
# Every 720p/1080p number so far is STREAMED. Declaring from those would repeat the exact error
# AB-R-0107 was created to fix, one axis over: an optimistic lane backing a declaration that the
# fallback busts. Fails OPEN.
set -u
cd "$(dirname "$0")/../.." || exit 1
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
OUT=probes/tiling-ab
BIN=./.build/release/RunLTX2
while pgrep -f "RunLTX2 --t2v-spot25 1920" >/dev/null; do sleep 30; done   # let the 481f corner finish

arm() { # tag tier W H F stream
  local tag="$1" tier="$2" w="$3" h="$4" f="$5" st="$6" t0=$(date +%s)
  printf "── %-26s " "$tag"
  env LTX_TIER="$tier" LTX_STREAM_25="$st" LTX_ENVELOPE_OVERRIDE=1 \
    "$BIN" --t2v-spot25 "$w" "$h" "$f" > "$OUT/$tag.log" 2>&1
  local r=$(grep DECLARE "$OUT/$tag.log"|tail -1|grep -oE "residentBytes=[0-9.]+"|grep -oE "[0-9.]+")
  local a=$(grep DECLARE "$OUT/$tag.log"|tail -1|grep -oE "peakActivationBytes=[0-9.]+"|grep -oE "[0-9.]+")
  local g=$(grep -E "\[BlockStreamer\] gate:" "$OUT/$tag.log"|tail -1|grep -oE "STREAM|FALL BACK")
  printf "peak=%-8s lane=%-10s wall=%ss\n" \
    "$(python3 -c "print(f'{${r:-0}+${a:-0}:.2f}')" 2>/dev/null)" "${g:-resident}" "$(( $(date +%s)-t0 ))"
}

# ⚠️ Tiling is now AUTOMATIC at these grids, so both arms are tiled — that is the point: the
# declaration must hold on the resident lane WITH tiling, which is the shipping configuration.
echo "═══ standard64 @ its new cap, streaming OFF — the .auto fallback it may take ═══"
arm fb-720p-std64-resident standard64 1280 704 161 0
echo "═══ max128 @ its new cap, RESIDENT — its DEFAULT lane, never measured at 1080p ═══"
arm fb-1080p-max128-resident max128 1920 1088 121 0
echo
echo "READ: declare each tier from the WORST of {streamed, resident} at its cap."
