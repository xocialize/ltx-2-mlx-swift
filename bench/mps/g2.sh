#!/bin/zsh
# G2 = 1280x704x121, the realistic target. Only the three arms that carry meaning at scale:
#   A1 ours int8 std64 (WHAT WE SHIP) · A3 ours bf16 resident (like-for-like) · B vendor.
# The quant/streaming isolations are already established at G1; no need to repeat them here.
SD=$1; SW=/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift
W=1280; H=704; F=121
PROMPT="a woman turns toward the camera and smiles, warm afternoon light"
CSV="$SD/mps-g2.csv"; echo "arm,round,geometry,seconds,max_rss_bytes,peak_footprint_bytes" > "$CSV"
emit () { local arm=$1 rnd=$2 log=$3
  local sec=$(grep -E "^ *[0-9.]+ real" "$log" | tail -1 | awk '{print $1}')
  local rss=$(grep "maximum resident set size" "$log" | tail -1 | awk '{print $1}')
  local fp=$(grep "peak memory footprint" "$log" | tail -1 | awk '{print $1}')
  [ "$rnd" != "0" ] && echo "$arm,$rnd,${W}x${H}x${F},${sec:-NA},${rss:-NA},${fp:-NA}" >> "$CSV"
  echo "[g2] $arm round$rnd ${sec:-NA}s rss=${rss:-NA} fp=${fp:-NA}"; }
ours () { local tier=$1 quant=$2 stream=$3 rnd=$4 lbl=$5
  local log="$SD/run-g2-$lbl-$rnd.log"
  # ⚠️ `env`, not an assignment prefix. ${stream:+VAR=val} in prefix position is parsed as a
  # COMMAND word, so the next assignment became the command name ("command not found:
  # LTX_T2V_PROMPT=..."). Same trap as the LTX_ENC_TREE one earlier this session.
  ( cd "$SW" && env LTX_TIER=$tier LTX_QUANT=$quant ${stream:+LTX_STREAM_25=$stream} \
    LTX_T2V_PROMPT="$PROMPT" /usr/bin/time -l ./.build/release/RunLTX2 --t2v-spot25 $W $H $F ) > "$log" 2>&1
  emit "$lbl" "$rnd" "$log"; }
vendor () { local rnd=$1; local log="$SD/run-g2-B-$rnd.log"
  "$SD/vendor-run.sh" $W $H $F "$SD/out-g2-B-$rnd.mp4" > "$log" 2>&1; emit "B" "$rnd" "$log"; }
for r in 0 1 2; do
  echo "[g2] === round $r (0 = discarded warm-up) ==="
  ours standard64 int8 ""  $r A1
  ours max128     bf16 "0" $r A3
  vendor $r
done
echo "[g2] DONE"
