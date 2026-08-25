#!/bin/zsh
# AB-P-0004 execution. Three arms, interleaved, warm-ups discarded.
#   A1 ours int8  @ standard64   (what we ship)
#   A2 ours bf16  @ max128       (isolates BINDING from QUANTIZATION — see plan)
#   B  vendor bf16 on PyTorch-MPS
# ONE instrument for all three: /usr/bin/time -l max RSS + peak footprint.
SD=$1; W=$2; H=$3; F=$4; ROUNDS=$5; TAG=$6
SW=/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift
PROMPT="a woman turns toward the camera and smiles, warm afternoon light"
CSV="$SD/mps-$TAG.csv"
echo "arm,round,geometry,seconds,max_rss_bytes,peak_footprint_bytes" > "$CSV"

run_ours () {  # tier quant round
  local tier=$1 quant=$2 rnd=$3 lbl=$4
  local log="$SD/run-$TAG-$lbl-$rnd.log"
  ( cd "$SW" && LTX_TIER=$tier LTX_QUANT=$quant LTX_T2V_PROMPT="$PROMPT" \
    /usr/bin/time -l ./.build/release/RunLTX2 --t2v-spot25 $W $H $F ) > "$log" 2>&1
  emit "$lbl" "$rnd" "$log"
}
run_vendor () {  # round
  local rnd=$1
  local log="$SD/run-$TAG-B-$rnd.log"
  "$SD/vendor-run.sh" $W $H $F "$SD/out-$TAG-B-$rnd.mp4" > "$log" 2>&1
  emit "B" "$rnd" "$log"
}
emit () {
  local arm=$1 rnd=$2 log=$3
  local sec=$(grep -E "^ *[0-9.]+ real" "$log" | tail -1 | awk '{print $1}')
  local rss=$(grep "maximum resident set size" "$log" | tail -1 | awk '{print $1}')
  local fp=$(grep "peak memory footprint" "$log" | tail -1 | awk '{print $1}')
  echo "$arm,$rnd,${W}x${H}x${F},${sec:-NA},${rss:-NA},${fp:-NA}" >> "$CSV"
  echo "[matrix] $arm round$rnd  ${sec:-NA}s  rss=${rss:-NA}  fp=${fp:-NA}"
}

echo "[matrix] === WARM-UPS (discarded: metallib + MPS kernel compile + weight materialization) ==="
run_ours standard64 int8 0 A1warm
run_ours max128     bf16 0 A2warm
run_vendor 0
sed -i '' '/,0,/d' "$CSV"       # drop warm-ups from the CSV

for r in $(seq 1 $ROUNDS); do
  echo "[matrix] === round $r/$ROUNDS (interleaved so thermal drift cannot alias onto an arm) ==="
  run_ours standard64 int8 $r A1
  run_ours max128     bf16 $r A2
  run_vendor $r
done
echo "[matrix] DONE -> $CSV"
