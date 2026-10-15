#!/bin/zsh
# AB-A-0005 on the M5 Pro 24 GB: stock vs Pruna decoder at the compact24 tier, alternating repeats,
# wall-clock + peak + DECLARE lines from `--t2v-spot25` (the staged instrument; AB-T-0081 budget).
# One fresh process per run under `/usr/bin/time -l`; swap and thermal state sampled around each run;
# one excluded warm-up per geometry (shape-specific kernel compile lands there); cooldown between runs.
# Geometries: 512x288x121 = compact24's own clamped envelope; 704x512x121 = the desktop Pruna receipt's
# geometry (bench_e2e_ladder-pruna-121f), where decode is a larger share of the run.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT="probes/pruna-m5pro-24gb"; LOG="$OUT/run.log"
BIN=./.build/release/RunLTX2
COOLDOWN=${COOLDOWN:-45}
export LTX_MODELS_ROOT="$HOME/Models"          # local staging copy of the fleet archive (same names)
export LTX_NO_CHECKPOINT=1                     # granule tree v2: globals + integrity from the manifest (hosted shape)
export LTX_TIER=compact24 LTX_QUANT=int8 LTX_STREAM_25=1 LTX_ENC=q8
export LTX_T2V_PROMPT="a cat playing piano" LTX_SEED=42
unset MLX_PROFILE                              # the profiler breaks fusion and inflates decode (never on a measured run)
{
  echo "# $(git log --oneline -1) · $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion) · $(date)"
  echo "# env: LTX_TIER=$LTX_TIER LTX_QUANT=$LTX_QUANT LTX_STREAM_25=$LTX_STREAM_25 LTX_ENC=$LTX_ENC LTX_NO_CHECKPOINT=$LTX_NO_CHECKPOINT prompt='$LTX_T2V_PROMPT' seed=$LTX_SEED cooldown=${COOLDOWN}s"
} > "$LOG"
therm() { pmset -g therm 2>/dev/null | grep -E 'CPU_Speed_Limit|CPU_Scheduler_Limit' | tr -s ' ' | paste -sd ';' - ; }
run() {  # name decoder W H F
  local name=$1 dec=$2 W=$3 H=$4 F=$5 slug
  slug="${name// /_}"
  echo "=== $name" | tee -a "$LOG"
  echo "swap before: $(sysctl -n vm.swapusage) · therm: $(therm)" >> "$LOG"
  LTX_VAE_DECODER="$dec" LTX_T2V_SAVE="$OUT/logs/$slug.mp4" /usr/bin/time -l "$BIN" --t2v-spot25 "$W" "$H" "$F" > "$OUT/logs/$slug.full.log" 2>&1
  echo "exit=$?" >> "$LOG"
  grep -E 'LTX_VAE_DECODER|granules|prewarm|resident floor|BlockStreamer\] gate|fell back|\] run |SPLIT|ACCEPTANCE|DECLARE|maximum resident|peak memory footprint| real ' "$OUT/logs/$slug.full.log" | tee -a "$LOG"
  echo "swap after:  $(sysctl -n vm.swapusage) · therm: $(therm)" >> "$LOG"
  sleep "$COOLDOWN"
}
mkdir -p "$OUT/logs"
# Burn-in (AB-R-0067): a freshly started box ramps monotonically across a session, which ABBA cannot
# cancel; throwaway generations first, then the measured arms. Excluded from every stat.
for i in 1 2 3; do run "burn-in $i stock 512x288x121" stock 512 288 121; done
# MEASURED: compact24's own clamped envelope (512x288x121 — what the profile resolves any request to).
# block 1: ABBA · block 2: BAAB — symmetric under linear drift. The warm-up carries the shape's kernel compile.
W=512; H=288; F=121; g="${W}x${H}x${F}"
run "warmup stock $g"  stock "$W" "$H" "$F"
run "stock $g rep1"    stock "$W" "$H" "$F"
run "pruna $g rep1"    pruna "$W" "$H" "$F"
run "pruna $g rep2"    pruna "$W" "$H" "$F"
run "stock $g rep2"    stock "$W" "$H" "$F"
run "pruna $g rep3"    pruna "$W" "$H" "$F"
run "stock $g rep3"    stock "$W" "$H" "$F"
run "stock $g rep4"    stock "$W" "$H" "$F"
run "pruna $g rep4"    pruna "$W" "$H" "$F"
# DIAGNOSTIC, never an acceptance number: the desktop Pruna receipt's geometry (704x512x121, where decode
# is a larger share of the run) forced past the profile clamp with LTX_ENVELOPE_OVERRIDE=1. Shows where
# the Pruna time win sits when decode dominates, and what a 24 GB box does above its envelope.
export LTX_ENVELOPE_OVERRIDE=1
W=704; H=512; F=121; g="${W}x${H}x${F}-OVERRIDE"
run "diag warmup stock $g" stock "$W" "$H" "$F"
run "diag pruna $g rep1"   pruna "$W" "$H" "$F"
run "diag stock $g rep1"   stock "$W" "$H" "$F"
run "diag stock $g rep2"   stock "$W" "$H" "$F"
run "diag pruna $g rep2"   pruna "$W" "$H" "$F"
unset LTX_ENVELOPE_OVERRIDE
echo "=== DONE $(date)" | tee -a "$LOG"
