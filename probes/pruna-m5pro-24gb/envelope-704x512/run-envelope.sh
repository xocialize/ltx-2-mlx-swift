#!/bin/zsh
# AB-T-0298 on the M5 Pro 24 GB: compact24's widened envelope, 704x512x121, under the lean decoder (the tier's
# default since v0.19.0) — acceptance-grade runs (NO LTX_ENVELOPE_OVERRIDE: the profile allows the geometry),
# t2v and i2v, AB-R-0561's protocol: fresh process per run under `time -l`, swap + thermal state around each
# run, burn-in and warm-up excluded, cooldown between runs. Stock-lever runs are DIAGNOSTIC rows (why the
# envelope is lean-only), never acceptance numbers.
set -u
cd "$(dirname "$0")/../../.." || exit 1
OUT="probes/pruna-m5pro-24gb/envelope-704x512"; LOG="$OUT/run.log"
BIN=./.build/release/RunLTX2
COOLDOWN=${COOLDOWN:-45}
THERM=/private/tmp/claude-501/-Users-mvsadmin-Downloads/5fead774-c1ba-417a-adf7-a21fc12d9fcb/scratchpad/thermstate
export LTX_MODELS_ROOT="$HOME/Models"
export LTX_NO_CHECKPOINT=1
export LTX_TIER=compact24 LTX_QUANT=int8 LTX_STREAM_25=1 LTX_ENC=q8
export LTX_T2V_PROMPT="a cat playing piano" LTX_SEED=42
unset MLX_PROFILE LTX_ENVELOPE_OVERRIDE LTX_VAE_DECODER
{
  echo "# $(git log --oneline -1) · $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion) · $(date)"
  echo "# env: LTX_TIER=$LTX_TIER LTX_QUANT=$LTX_QUANT LTX_STREAM_25=$LTX_STREAM_25 LTX_ENC=$LTX_ENC LTX_NO_CHECKPOINT=$LTX_NO_CHECKPOINT prompt='$LTX_T2V_PROMPT' seed=$LTX_SEED cooldown=${COOLDOWN}s · decoder: profile default (lean) unless 'stock' rows (LTX_VAE_DECODER=stock, diagnostic)"
} > "$LOG"
therm() { [ -x "$THERM" ] && "$THERM" 2>/dev/null || echo "n/a"; }
run() {  # name mode(t2v|i2v) decoder(default|stock) W H F
  local name=$1 mode=$2 dec=$3 W=$4 H=$5 F=$6 slug i2v=0
  slug="${name// /_}"; [ "$mode" = "i2v" ] && i2v=1
  echo "=== $name" | tee -a "$LOG"
  echo "swap before: $(sysctl -n vm.swapusage) · therm: $(therm)" >> "$LOG"
  if [ "$dec" = "stock" ]; then
    LTX_I2V=$i2v LTX_VAE_DECODER=stock LTX_T2V_SAVE="$OUT/logs/$slug.mp4" /usr/bin/time -l "$BIN" --t2v-spot25 "$W" "$H" "$F" > "$OUT/logs/$slug.full.log" 2>&1
  else
    LTX_I2V=$i2v LTX_T2V_SAVE="$OUT/logs/$slug.mp4" /usr/bin/time -l "$BIN" --t2v-spot25 "$W" "$H" "$F" > "$OUT/logs/$slug.full.log" 2>&1
  fi
  echo "exit=$?" >> "$LOG"
  grep -q 'OVERRIDE=1' "$OUT/logs/$slug.full.log" && echo "!! override warning present — NOT acceptance-grade" >> "$LOG"
  grep -E 'request|granules|prewarm|resident floor|fell back|\] run |SPLIT|ACCEPTANCE|DECLARE|maximum resident|peak memory footprint| real ' "$OUT/logs/$slug.full.log" | tee -a "$LOG"
  echo "swap after:  $(sysctl -n vm.swapusage) · therm: $(therm)" >> "$LOG"
  sleep "$COOLDOWN"
}
W=704; H=512; F=121; g="${W}x${H}x${F}"
run "burn-in 1 t2v lean $g" t2v default "$W" "$H" "$F"
run "burn-in 2 t2v lean $g" t2v default "$W" "$H" "$F"
# t2v block: warm-up excluded; lean reps are the acceptance rows; stock rows diagnostic
run "warmup t2v lean $g"  t2v default "$W" "$H" "$F"
run "t2v lean $g rep1"    t2v default "$W" "$H" "$F"
run "t2v stock $g diag1"  t2v stock   "$W" "$H" "$F"
run "t2v lean $g rep2"    t2v default "$W" "$H" "$F"
run "t2v lean $g rep3"    t2v default "$W" "$H" "$F"
run "t2v stock $g diag2"  t2v stock   "$W" "$H" "$F"
run "t2v lean $g rep4"    t2v default "$W" "$H" "$F"
# i2v block
run "warmup i2v lean $g"  i2v default "$W" "$H" "$F"
run "i2v lean $g rep1"    i2v default "$W" "$H" "$F"
run "i2v stock $g diag1"  i2v stock   "$W" "$H" "$F"
run "i2v lean $g rep2"    i2v default "$W" "$H" "$F"
run "i2v lean $g rep3"    i2v default "$W" "$H" "$F"
run "i2v stock $g diag2"  i2v stock   "$W" "$H" "$F"
run "i2v lean $g rep4"    i2v default "$W" "$H" "$F"
echo "=== DONE $(date)" | tee -a "$LOG"
