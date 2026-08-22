#!/bin/bash
# 5s @ 720p WATERMARK — a cheap first data point before committing to the long HD runs.
#
# 🔑 WHY A WATERMARK FIRST (operator, 2026-08-22): the real viability runs (10s@720p, 10s@1080p) are
# LONG — LTX-2.3 spent 962 s on 10s@720p and 3112 s (52 min) on 10s@1080p. Measuring 5s@720p first
# costs ~1/4 of that and tells us whether the activation scaling is anywhere near the 2.3-derived
# estimate before a slot is committed.
#
# 🚨 THIS USES LTX_ENVELOPE_OVERRIDE=1 AND IS NOT AN ACCEPTANCE NUMBER. No 2.5 profile permits
# 1280x704 — every tier caps at 704x512 (`maxWidth`/`maxHeight`, and the `default:` branch means
# max128 shares standard64's ceiling). The override exists so the cap can be moved on EVIDENCE, the
# same way maxFrames went 241 -> 481 ("the old cap left ~60 GB of a 128 GB budget unused").
#
# WHAT WE EXPECT, and why it is only an estimate:
#   LTX-2.3 measured 1280x704x241f = 80.36 GB with a ~40 GB RESIDENT bf16 DiT, so ~40 GB was
#   activation. 2.5 streams that DiT away (0.7 GB resident), so the estimate is ~43 GB at 241f.
#   At 121f expect LESS — but ⚠️ NOT half: our 704x512 streamed peak was FLAT from 161f to 481f
#   because it is ENCODER-BOUND at 24.81 GB. The question this run answers is exactly where the
#   crossover sits: at 720p, does activation finally exceed the ~24.8 GB encoder floor?
#
# ⚠️ MEMORY IS THE DELIVERABLE, NOT TIME. Single arm on whatever thermal state the box is in.
set -u
cd "$(dirname "$0")/../.." || exit 1
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
OUT=probes/hd-viability
BIN=./.build/release/RunLTX2
xcrun swift build -c release --product RunLTX2 2>&1 | grep -E "error:|Build complete" | tail -1

if pgrep -f "RunLTX2" >/dev/null; then echo "🚨 another RunLTX2 is live — aborting"; exit 2; fi

# 1280x704 = 720p-class, both axes on the /32 latent grid (40 x 22). 121f = 5.04 s at 24 fps,
# and 121 is on the 8n+1 causal frame grid.
echo "── 720p watermark: 1280x704x121f, max128, STREAMED"
env LTX_TIER=max128 LTX_QUANT=bf16 LTX_STREAM_25=1 LTX_ENVELOPE_OVERRIDE=1 \
  "$BIN" --t2v-spot25 1280 704 121 > "$OUT/720p-121f-streamed.log" 2>&1
grep -E "ENVELOPE_OVERRIDE|request|gate:|fell back|SPLIT|DECLARE|ACCEPTANCE|run " "$OUT/720p-121f-streamed.log" | tail -8

echo
echo "⚠️ CONFIRM the request line says 1280x704 — if it says 704x512 the override did not take and"
echo "   the number is a re-measurement of a geometry we already have."
echo "⚠️ CONFIRM the LAST BlockStreamer gate line says STREAM — lane= reports the CONFIG, not the gate."
