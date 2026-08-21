#!/bin/bash
# ENCODER LADDER × PROMPT ADHERENCE — AB-T-0074 precondition 1.
#
# 🔑 WHY THIS EXISTS. The blind audio ladder (AB-R-0111/AB-D-0037) could not separate ANY encoder
# arm — not q8, not the REJECTED q4, not a POISON encoder at 0.829 connector cosine. But its
# prompts were *"a person looks directly at the camera and says clearly: <line>"*: essentially NO
# visual content to adhere to. Both bf16 and poison produced a talking head square to camera,
# differing only in background and face. **That is the easiest possible adherence test.**
#
# 🚨 A DEGRADED TEXT ENCODER'S FAILURE MODE IS IGNORING THE PROMPT, NOT LOOKING BAD. A
# quality-PREFERENCE judgement structurally cannot see that — the same preference-vs-similarity
# distinction AB-R-0104 drew. So this probe asks a different question, and it is the only question
# that can justify reviving q4:
#
#     NOT "which looks better"   ->  every arm looked great; that told us nothing
#     BUT "does it contain what was asked for", element by element
#
# DESIGN: each prompt carries 4-5 INDEPENDENTLY CHECKABLE elements — named objects, colours bound
# to specific objects, a COUNT, and a spatial relation. Colour-object binding and counting are the
# first things to go when conditioning degrades, which is exactly why they are here.
#
# 🚨 THE POISON ARM IS THE CALIBRATION, AGAIN, AND IT IS WHAT MAKES THE RESULT READABLE. If poison
# scores like bf16 on adherence TOO, then this test is also blind and NOTHING here licenses q4 —
# report that, do not report a q4 pass. (AB-L-0017: a metric that passes a deliberately-broken arm
# is not a metric.)
#
# ⚠️ Scoring is per-ELEMENT, and the comparison is BETWEEN ARMS, never against an absolute bar.
# Counts are hard for every diffusion model; if bf16 also misses the count, that element simply is
# not discriminating — it is not a q4 failure.
set -u
cd "$(dirname "$0")/../.." || exit 1
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
OUT=~/Desktop/ltx25-enc-adherence
mkdir -p "$OUT"
BIN=./.build/release/RunLTX2
xcrun swift build -c release --product RunLTX2 2>&1 | grep -E "error:|Build complete" | tail -1

MANIFEST=$OUT/manifest.tsv; : > "$MANIFEST"

p1="a red bicycle leaning against a blue wooden door, a black cat sitting on the ground beside it"
p2="three yellow rubber ducks floating in a white bathtub filled with water"
p3="a woman in a green raincoat holding a red umbrella, walking past a brick wall"

gen() { # arm tree encquant pid prompt
  local arm="$1" tree="$2" encq="$3" pid="$4" prompt="$5"
  local tag="${pid}-${arm}" t0=$(date +%s)
  { echo "ARM:      $arm"; echo "ENC_TREE: ${tree:-<resolved from LTX_ENC=$encq>}"
    echo "PROMPT:   $prompt"; } > "$OUT/$tag.log"
  LTX_TIER=standard64 LTX_QUANT=int8 LTX_ENC="$encq" LTX_ENC_TREE="$tree" \
  LTX_T2V_PROMPT="$prompt" LTX_T2V_SAVE="$OUT/$tag.mp4" \
    "$BIN" --t2v-spot25 704 512 121 >> "$OUT/$tag.log" 2>&1
  printf '%s\t%s\t%s\n' "$tag" "$arm" "$prompt" >> "$MANIFEST"
  printf "  %-22s %4ss\n" "$tag" "$(( $(date +%s) - t0 ))"
}

for pair in "p1:$p1" "p2:$p2" "p3:$p3"; do
  pid="${pair%%:*}"; prompt="${pair#*:}"
  echo "═══ $pid — \"$prompt\""
  gen bf16   ""                 bf16 "$pid" "$prompt"   # reference
  gen q8     ""                 q8   "$pid" "$prompt"   # shipping on low tiers
  gen q4     ltx-2.5-mlx-q4     q8   "$pid" "$prompt"   # THE CANDIDATE (rejected on the gate)
  gen poison ltx-2.5-mlx-poison q8   "$pid" "$prompt"   # CALIBRATION — must fail, or the test is blind
done

# Contact sheet per prompt: one frame per arm, arms UNLABELLED in the image itself so the
# scorer reads pixels rather than filenames.
echo; echo "═══ extracting frames for element scoring ═══"
for pid in p1 p2 p3; do
  for arm in bf16 q8 q4 poison; do
    ffmpeg -v error -y -i "$OUT/$pid-$arm.mp4" -vf "select=eq(n\,60),scale=420:-1" \
      -vframes 1 "$OUT/frame-$pid-$arm.png" 2>/dev/null
  done
done
ls "$OUT"/frame-*.png | wc -l | xargs echo "frames extracted:"
echo
echo "SCORE PER ELEMENT, ARM VS ARM — not against an absolute bar."
echo "⚠️ If poison scores like bf16, this test is blind too. Report THAT, not a q4 pass."
