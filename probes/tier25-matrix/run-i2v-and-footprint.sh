#!/bin/bash
# LTX-2.5 low-tier closure — the LAST measurement gap before AB-D-0014's tier table is rewritable.
# Written 2026-08-20, to be run in a dedicated GPU slot. ~45-60 min, unattended.
#
# Measures, under the SHIPPING low-tier candidate (streamed DiT + int8 encoder):
#   1. i2v  — a materially different resident set (adds the ~4.9 GB i2v-adapter LoRA). Never
#             measured under this config; must not be inferred from the t2v row.
#   2. t2v footprint re-declaration with the CORRECTED split (see the instrument note below).
#
# ⚠️ PROTOCOL, all of it earned:
#   - SERIAL. Never two streaming arms at once: S is a shared-resource measurement and contention
#     silently depresses it (a q4 arm read 2.15 vs its clean 3.56 GiB/s under overlap).
#   - No MLX_PROFILE. It breaks fusion and inflated vae-decode above the clean whole-run peak.
#   - No burn-in needed: these are MEMORY numbers. Burn-in is for TIMING claims only (AB-R-0067).
#   - compact24 gets .forceStream from its profile advisory automatically — do NOT set LTX_STREAM_GATE.
#     .auto flips run-to-run at that N (peak 14.6 <-> 33.6 GB, 1 in 3), which is why the advisory exists.
#
# 🔑 INSTRUMENT: the DECLARE line uses resident = LOW-WATER DURING THE RUN, not phys-after-load and
# not the post-run sample. Under eviction after-load reads a transient the run never returns to
# (22.13 GB against a 14.59 GB peak — a resident larger than the peak); the post-run sample is
# unstable run-to-run. The harness prints all three so the choice stays checkable.
set -u
cd "$(dirname "$0")/../.." || exit 1
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
OUT="probes/tier25-matrix"
BIN=./.build/release/RunLTX2
xcrun swift build -c release --product RunLTX2 2>&1 | grep -E "error:|Build complete" | tail -1

run() {  # mode tier rep
  local mode=$1 tier=$2 rep=$3 tag="$2-$1-rep$3"
  local i2v=0; [ "$mode" = "i2v" ] && i2v=1
  LTX_I2V=$i2v LTX_TIER="$tier" LTX_QUANT=int8 LTX_STREAM_25=1 LTX_ENC=q8 \
    "$BIN" --t2v-spot25 704 512 161 > "$OUT/$tag.log" 2>&1
  printf "  %-26s %s\n     %s\n" "$tag" \
    "$(grep -oE 'ACCEPTANCE .*' "$OUT/$tag.log" | head -1)" \
    "$(grep -oE 'DECLARE .*'    "$OUT/$tag.log" | head -1)"
}

for rep in 1 2 3; do
  for tier in compact24 balanced32 standard64; do
    run i2v "$tier" "$rep"
  done
done
echo "── t2v footprint re-declaration (corrected split) ──"
for rep in 1 2 3; do
  for tier in compact24 balanced32 standard64; do
    run t2v "$tier" "$rep"
  done
done
echo
echo "Next: read the DECLARE lines, take the WORST-case resident and activation per tier, and only"
echo "then edit MLXLTX25Package's QuantFootprint + AB-D-0014's tier table. Worst case, never mean —"
echo "compact24's .auto headroom ranged 1.33x..0.59x and the mean would have hidden the failure."
