#!/bin/zsh
# AB-T-0297 step 1 on the M5 Pro 24 GB: compact24's i2v row, stock vs Pruna decoder, AB-R-0561's protocol.
# `--t2v-spot25` with LTX_I2V=1 (synthetic clean first frame; the ~4.9 GB i2v-adapter LoRA resident), one fresh
# process per run under `/usr/bin/time -l`, swap and thermal state around each run, cooldown between runs, two
# burn-in generations and one warm-up excluded, measured arms in ABBA + BAAB order at the tier's own envelope.
set -u
cd "$(dirname "$0")/../../.." || exit 1
OUT="probes/pruna-m5pro-24gb/i2v"; LOG="$OUT/run.log"
BIN=./.build/release/RunLTX2
COOLDOWN=${COOLDOWN:-45}
THERM=/private/tmp/claude-501/-Users-mvsadmin-Downloads/5fead774-c1ba-417a-adf7-a21fc12d9fcb/scratchpad/thermstate
export LTX_MODELS_ROOT="$HOME/Models"
export LTX_NO_CHECKPOINT=1
export LTX_TIER=compact24 LTX_QUANT=int8 LTX_STREAM_25=1 LTX_ENC=q8 LTX_I2V=1
export LTX_T2V_PROMPT="a cat playing piano" LTX_SEED=42
unset MLX_PROFILE
{
  echo "# $(git log --oneline -1) · $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion) · $(date)"
  echo "# env: LTX_TIER=$LTX_TIER LTX_QUANT=$LTX_QUANT LTX_STREAM_25=$LTX_STREAM_25 LTX_ENC=$LTX_ENC LTX_I2V=$LTX_I2V LTX_NO_CHECKPOINT=$LTX_NO_CHECKPOINT prompt='$LTX_T2V_PROMPT' seed=$LTX_SEED cooldown=${COOLDOWN}s"
} > "$LOG"
therm() { [ -x "$THERM" ] && "$THERM" 2>/dev/null || echo "n/a"; }
run() {  # name decoder W H F
  local name=$1 dec=$2 W=$3 H=$4 F=$5 slug
  slug="${name// /_}"
  echo "=== $name" | tee -a "$LOG"
  echo "swap before: $(sysctl -n vm.swapusage) · therm: $(therm)" >> "$LOG"
  LTX_VAE_DECODER="$dec" LTX_T2V_SAVE="$OUT/logs/$slug.mp4" /usr/bin/time -l "$BIN" --t2v-spot25 "$W" "$H" "$F" > "$OUT/logs/$slug.full.log" 2>&1
  echo "exit=$?" >> "$LOG"
  grep -E 'request|granules|prewarm|resident floor|fell back|\] run |SPLIT|ACCEPTANCE|DECLARE|maximum resident|peak memory footprint| real ' "$OUT/logs/$slug.full.log" | tee -a "$LOG"
  echo "swap after:  $(sysctl -n vm.swapusage) · therm: $(therm)" >> "$LOG"
  sleep "$COOLDOWN"
}
W=512; H=288; F=121; g="${W}x${H}x${F}"
for i in 1 2; do run "burn-in $i stock i2v $g" stock "$W" "$H" "$F"; done
run "warmup stock i2v $g"  stock "$W" "$H" "$F"
run "stock i2v $g rep1"    stock "$W" "$H" "$F"
run "pruna i2v $g rep1"    pruna "$W" "$H" "$F"
run "pruna i2v $g rep2"    pruna "$W" "$H" "$F"
run "stock i2v $g rep2"    stock "$W" "$H" "$F"
run "pruna i2v $g rep3"    pruna "$W" "$H" "$F"
run "stock i2v $g rep3"    stock "$W" "$H" "$F"
run "stock i2v $g rep4"    stock "$W" "$H" "$F"
run "pruna i2v $g rep4"    pruna "$W" "$H" "$F"
echo "=== DONE $(date)" | tee -a "$LOG"
