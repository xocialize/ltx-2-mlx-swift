#!/bin/bash
# HD VIABILITY — 10s@720p (the falsification test) then 10s@1080p (the number). AB-T-0077.
#
# 🚨 NOT ACCEPTANCE NUMBERS. Every arm uses LTX_ENVELOPE_OVERRIDE=1; no 2.5 profile permits either
# geometry (all tiers cap at 704x512). These decide whether the CAP SHOULD MOVE — the evidence path
# maxFrames took from 241 to 481. A number from an overridden run can never be an acceptance number.
#
# ORDER IS DELIBERATE — the 720p arm goes FIRST because it can prove me WRONG:
#   AB-R-0117 predicted 10s@720p ~= 43.7 GB, i.e. the SAME as the 5 s watermark, on the model that
#   activation is resolution-driven while frames amortise through temporal decode chunking. If 241f
#   costs materially more than 121f, that model is dead AND the 1080p projection built on it dies
#   with it. Run the falsifier before the expensive arm it licenses.
#
# ⟲ 1080p is NOT blocked by the budget-rate question (AB-T-0076), and my own earlier note saying so
# was wrong. The RATE decides the VERDICT; it does not affect the MEASUREMENT. And measuring may
# DISSOLVE the dependency: ~85 GB is within both rates, ~115 GB is over both; only the middle needs
# the rate. Right now we are arguing about a projection when a fact is one run away.
#
# ⚠️ NO resident 1080p control. 2.5 bf16 resident at 1080p would be ~40 GB DiT + ~94 GB activation
# = ~134 GB on a 137 GB box (2.3 measured 132.71). That is the I9 boundary, unattended, for a
# nice-to-have — the streaming comparison is made against 2.3's existing receipt instead. The
# resident control is taken at 720p, where ~83 GB is safe.
#
# ⚠️ MEMORY IS THE DELIVERABLE. Fresh boot = right for memory, WORST for timing (AB-R-0116).
# Arms are interleaved so the two 720p streamed reps straddle the resident control rather than
# sitting on systematically different thermal states — that does not rescue timing on a cold box,
# it just stops the ordering adding a second confound. Walls here are INDICATIVE ONLY.
set -u
cd "$(dirname "$0")/../.." || exit 1
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
OUT=probes/hd-viability
BIN=./.build/release/RunLTX2
xcrun swift build -c release --product RunLTX2 2>&1 | grep -E "error:|Build complete" | tail -1

if pgrep -f "RunLTX2" >/dev/null; then echo "🚨 another RunLTX2 is live — S is shared. ABORTING."; exit 2; fi

arm() { # tag W H F stream
  local tag="$1" w="$2" h="$3" f="$4" st="$5" t0=$(date +%s)
  echo "── $tag  (${w}x${h}x${f}f, stream=$st)"
  env LTX_TIER=max128 LTX_QUANT=bf16 LTX_STREAM_25="$st" LTX_ENVELOPE_OVERRIDE=1 \
    "$BIN" --t2v-spot25 "$w" "$h" "$f" > "$OUT/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "   ❌ exit $rc — see $OUT/$tag.log"; fi
  grep -E "request|SPLIT|ACCEPTANCE" "$OUT/$tag.log" | tail -2
  grep -E "\[BlockStreamer\] gate:" "$OUT/$tag.log" | tail -1
  printf "   wall %ss (INDICATIVE — fresh boot)\n\n" "$(( $(date +%s) - t0 ))"
}

echo "═══ PHASE A — 10s@720p: the falsification test ═══"
arm 720p-241f-streamed-r1 1280 704 241 1
arm 720p-241f-resident    1280 704 241 0
arm 720p-241f-streamed-r2 1280 704 241 1

echo "═══ PHASE B — 10s@1080p: the number ═══"
echo "free before 1080p: $(memory_pressure 2>/dev/null | tail -1)"
arm 1080p-241f-streamed-r1 1920 1088 241 1
arm 1080p-241f-streamed-r2 1920 1088 241 1

echo "READ:"
echo " 1. CONFIRM each request line shows the geometry ASKED FOR — 704x512 means the override failed."
echo " 2. CONFIRM the LAST BlockStreamer gate line says STREAM — lane= reports the CONFIG, not the gate."
echo " 3. 720p ~= 43.7 GB CONFIRMS frame-amortisation; materially higher REFUTES it and voids the"
echo "    1080p projection (though not the 1080p MEASUREMENT)."
echo " 4. 1080p vs 89.6 (0.7x) and 108.8 (0.85x) — if it clears or busts BOTH, AB-T-0076 stops"
echo "    blocking the verdict."
