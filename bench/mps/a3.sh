#!/bin/zsh
SD=$1; SW=/Volumes/Satechi/Development/mlxengine-video-ltx/LTX_DEV/ltx-2-mlx-swift
while pgrep -f "a1prime.sh" >/dev/null; do sleep 15; done   # never overlap arms
PROMPT="a woman turns toward the camera and smiles, warm afternoon light"
echo "arm,round,geometry,seconds,max_rss_bytes,peak_footprint_bytes" > "$SD/mps-g1-a3.csv"
for r in 0 1 2 3; do
  log="$SD/run-g1-A3-$r.log"
  ( cd "$SW" && LTX_TIER=max128 LTX_QUANT=bf16 LTX_STREAM_25=0 LTX_T2V_PROMPT="$PROMPT" \
    /usr/bin/time -l ./.build/release/RunLTX2 --t2v-spot25 704 512 33 ) > "$log" 2>&1
  sec=$(grep -E "^ *[0-9.]+ real" "$log" | tail -1 | awk '{print $1}')
  rss=$(grep "maximum resident set size" "$log" | tail -1 | awk '{print $1}')
  fp=$(grep "peak memory footprint" "$log" | tail -1 | awk '{print $1}')
  [ "$r" != "0" ] && echo "A3,$r,704x512x33,$sec,$rss,$fp" >> "$SD/mps-g1-a3.csv"
  echo "[a3] round$r ${sec}s rss=$rss"
done
