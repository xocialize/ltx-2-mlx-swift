#!/bin/bash
# PHASE 2d — max128 STREAMED. The last unmeasured tier+mode, and the on-ramp to Phase 3.
#
# 🚨 THIS NEEDS A DEDICATED SLOT. It is a memory ACCEPTANCE measurement, so every protocol rule this
# project earned applies:
#   * SERIAL ONLY — S is a shared-resource measurement; any overlap voids every S/stall number.
#   * NO MLX_PROFILE — it breaks fusion and once inflated a decode span ABOVE the clean whole-run peak.
#   * NO burn-in — burn-in is for TIMING claims. A fresh boot is the WORST case for timing stability
#     but the RIGHT one for memory safety (I9 is a paging ABORT, not a slowdown).
#   * WORST-CASE, never mean — compact24's .auto headroom once ranged 1.33x...0.59x.
#
# WHY IT IS OPEN: max128 is the one tier whose profile advises RESIDENT (recommendedStreamedBlocks
# returns false for it), so it has never been measured streamed. Two things ride on it:
#   1. Its declared charge is 76 GB (40 resident + 36 activation). If streaming lands where the other
#      tiers did, that number is enormously conservative and max128 could host far more co-residency
#      (LTX + enhancer + upscalers) than it currently admits.
#   2. 481f@96 = max128.maxFrames is the LONGEST clip any profile permits and measured 66.01 GB
#      NATIVE (AB-R-0073). Whether streaming moves that is the Phase 3 on-ramp.
#
# ⚠️ 481f IS 3x THE LARGEST FRAME COUNT EVER MEASURED ON 2.5 STREAMED. The flat 9->161f result is
# strong evidence but does NOT extend automatically; that is the point of measuring.
#
# ⚠️ max128 advises RESIDENT, so streaming here is an EXPLICIT OVERRIDE (LTX_STREAM_25=1). That is
# the escape hatch working as designed (AB-D-0035) — not a profile change. Do NOT "fix" the profile
# from this run; a measurement is not a decision.
set -u
cd "$(dirname "$0")/../.." || exit 1
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
OUT=probes/max128-streamed
BIN=./.build/release/RunLTX2
xcrun swift build -c release --product RunLTX2 2>&1 | grep -E "error:|Build complete" | tail -1

# Guard the one rule that silently ruins the run.
if pgrep -f "RunLTX2" | grep -qv "^$$\$"; then
  echo "🚨 another RunLTX2 is live — S is a shared-resource measurement. ABORTING."; exit 2
fi

run() { # tag  frames  streamflag  extra-env
  local tag="$1" frames="$2" stream="$3"; shift 3
  echo "── $tag (${frames}f, stream=$stream)"
  env LTX_TIER=max128 LTX_QUANT=bf16 LTX_STREAM_25="$stream" "$@" \
    "$BIN" --t2v-spot25 704 512 "$frames" > "$OUT/$tag.log" 2>&1
  grep -E "PEAK|DECLARE|gate:|fell back|phys-after-load" "$OUT/$tag.log" | tail -4
}

# 3 reps per arm, worst-case wins. Resident first so the pair is comparable within one boot.
for rep in 1 2 3; do run "resident-161f-r$rep" 161 0; done
for rep in 1 2 3; do run "streamed-161f-r$rep" 161 1; done

# The frame cap. ⚠️ 481f resident measured 66.01 GB / 533 s NATIVE — expect a LONG run.
for rep in 1 2 3; do run "streamed-481f-r$rep" 481 1; done
run "resident-481f-ctl" 481 0     # control: the AB-R-0073 number, re-measured on THIS boot

echo
echo "READ: worst-case PEAK per arm, from the 25 ms PhysSampler high-water — never a profiler span."
echo "⚠️ Read the LAST gate line per log: the 9f warmup falls back at small N BY DESIGN."
echo "🚨 THE SPLIT LINE lane= REPORTS THE CONFIG, NOT THE GATE OUTCOME. Verified in the 9f smoke"
echo "   test: the gate FELL BACK on both passes (N=713 vs N_min 4140) yet the line still read"
echo "   lane=STREAMED, peak 44.10 GB — a RESIDENT number under a STREAMED label. Read the LAST"
echo "   [BlockStreamer] gate line in each log and confirm it says STREAM before quoting any peak."
