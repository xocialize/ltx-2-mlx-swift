#!/bin/zsh
# Per-forward q8-vs-bf16 spread probe for the LTX-2.5 DiT, with LTX-2.3 as the comparison arm.
#
# N SEPARATE PROCESSES PER ARM — that is the whole design. 2.3's `--dit-q8-gate` spread
# (0.996388–0.999915 over five runs) showed up ACROSS process invocations; in-process the int8
# path self-repeats bit-exactly. One process reporting one cosine is not a result here.
#
# Comparisons produced:
#   bf16[i] vs bf16[j]   — MUST-FAIL CONTROL. On 2.3 this arm was exactly deterministic (0 spread).
#                          If it moves, the instrument is what is being measured, not the quant.
#   q8[i]   vs q8[j]     — the 2.3 nondeterminism probe, re-run on 2.5.
#   bf16[i] vs q8[j]     — the quant delta a tier declaration rests on (all cross-pairs).
#   the same three on 2.3 — so 2.5's delta has a same-metric comparison point instead of being
#                          read against doctrine figures that were measured a different way.
#
# ⚠️ `seq $((i+1)) $REPS` is WRONG on macOS: BSD seq counts DOWN when start > end, so the last
# iteration emitted a nonexistent index AND a self-pair, and a file compared with itself reports
# bitExact=yes unconditionally. Pairs are enumerated with an explicit guard below, and
# `--dit25-probe-compare` now refuses a self-comparison outright.
#
# usage: probes/dit25-quant-probe.sh [reps] [arms...]      (default: 3, "bf16 ditq8")
#   GEOM="W H F" to override the 704 512 121 default; TAG to keep runs in their own directory.
set -u
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-beta.app/Contents/Developer}
BIN=./.build/debug/RunLTX2
REPS=${1:-3}
shift 2>/dev/null || true
if [ $# -gt 0 ]; then ARMS=("$@"); else ARMS=(bf16 ditq8); fi
GEOM=${GEOM:-}
TAG=${TAG:-}
# ⚠️ zsh does NOT word-split an unquoted parameter expansion, so `$GEOM` would reach the binary as
# ONE argument "448 320 9" and be silently ignored — the exact collapse that once turned a geometry
# sweep into two identical runs. Split it explicitly, and the probe echoes its RESOLVED geometry.
if [ -n "$GEOM" ]; then GEOMARGS=(${=GEOM}); else GEOMARGS=(); fi
OUT=probes/dit25probe${TAG:+-$TAG}
mkdir -p "$OUT"
LOG="$OUT/runs.log"
: > "$LOG"

echo "[driver] reps=$REPS  arms=${ARMS[*]}  geom=${GEOMARGS[*]:-default(704 512 121)}  out=$OUT  binary=$BIN"

for i in $(seq 1 "$REPS"); do
  for arm in "${ARMS[@]}"; do
    echo "[driver] === arm=$arm rep=$i ==="
    "$BIN" --dit25-probe "$arm" "$OUT/$arm-$i.safetensors" ${GEOMARGS[@]:-} 2>&1 | tee -a "$LOG" || exit 1
  done
done

# Every unordered pair (x-i, y-j) with x-i strictly before y-j in (arm, rep) order — no self-pairs,
# no duplicates, and cross-arm pairs enumerated in full.
cmp_pairs() {
  local -a files
  files=()
  for arm in "${ARMS[@]}"; do
    for i in $(seq 1 "$REPS"); do files+=("$OUT/$arm-$i.safetensors"); done
  done
  local n=${#files[@]}
  for ((x = 1; x <= n; x++)); do
    for ((y = x + 1; y <= n; y++)); do
      echo "${files[$x]} ${files[$y]}"
    done
  done
}

echo
echo "[driver] ================= ALL PAIRS ================="
cmp_pairs | while read -r f1 f2; do
  "$BIN" --dit25-probe-compare "$f1" "$f2" 2>&1 | tee -a "$LOG"
done

echo
echo "[driver] SPREAD SUMMARY"
grep 'RESULT' "$LOG" | sed 's/^\[dit25-cmp\] RESULT //'
