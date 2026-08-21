#!/bin/bash
# DOES THE int8 TEXT ENCODER DEGRADE SPEECH? — the gap Phase 1 left open.
#
# 🔑 WHY THIS IS THE LOAD-BEARING ONE. Every Phase 1 audio arm ran the **bf16** encoder, but the
# low tiers AUTO-FOLLOW to int8 (AB-D-0035) — so if int8 conditioning costs speech, it costs it on
# exactly the machines where joint audio+video is most of the product's value. And AB-R-0104's
# blind A/B, which passed, judged **VIDEO**. Speech under int8 is untested.
#
# ⚠️ WHY WER IS LEGITIMATE HERE THOUGH SSIM WOULD NOT BE. Swapping the TEXT ENCODER changes
# conditioning -> trajectory -> a different (equally valid) video, so cross-arm pixel metrics are a
# category error (AB-R-0104). But the ground truth here is the SCRIPTED TEXT, not the other arm's
# pixels: both arms should say the same WORDS. That comparison is well-posed.
#
# 🚨 LOW-PRIOR SENTENCES ARE MANDATORY (AB-R-0110). Phase 1's arms were famous stock sentences, and
# the phrase's prior contaminated the model, the ASR, AND the human judge in the same direction —
# the operator completed "the rain in Spain..." to "Spain" from audio that genuinely says "spine".
# When every judge shares the prior there is no independent check left. So: no proverbs, no
# pangrams, no stock phrases, no homophones, everyday vocabulary (so a failure is not just
# rare-word difficulty).
#
# 🚨 THE POISON CONTROL IS THE POINT. A quality test whose metric cannot fail on a known-bad
# encoder proves nothing (AB-L-0017, the trap this repo has hit five times). Two known-bad arms:
#   -q4      REJECTED by the encoder gate (connector 0.996728 vs a 0.999879 bf16 floor)
#   -poison  DELIBERATELY broken (0.829329 on real tokens)
# If those transcribe as cleanly as bf16, the instrument is BLIND and a q8 pass means nothing.
# Read the controls FIRST; the q8 verdict is only interpretable if they moved.
set -u
cd "$(dirname "$0")/../.." || exit 1
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
OUT=~/Desktop/ltx25-enc-audio-ab
mkdir -p "$OUT"
BIN=./.build/release/RunLTX2
xcrun swift build -c release --product RunLTX2 2>&1 | grep -E "error:|Build complete" | tail -1

MANIFEST=$OUT/manifest.tsv; : > "$MANIFEST"

# Low-prior, homophone-free, everyday vocabulary.
s1="the yellow truck left the depot at seven"
s2="my sister keeps her bicycle behind the garden shed"
s3="we ordered nine boxes of paper on Thursday morning"

gen() { # arm  tree-or-empty  encquant  sid  sentence
  local arm="$1" tree="$2" encq="$3" sid="$4" sent="$5"
  local tag="${sid}-${arm}" t0=$(date +%s)
  { echo "ARM:      $arm"; echo "ENC_TREE: ${tree:-<resolved from LTX_ENC=$encq>}"
    echo "PROMPT:   a person looks directly at the camera and says clearly: $sent"
    echo "EXPECT:   $sent"; } > "$OUT/$tag.log"
  # ⚠️ LTX_ENC_TREE is passed UNCONDITIONALLY (empty = ignored by the Swift guard). A
  # `${tree:+LTX_ENC_TREE=$tree}` here is silently broken: bash decides assignment-vs-command at
  # PARSE time, so a word starting with `$` is marked the COMMAND, and when it expands to nothing
  # the next assignment gets run as a command. Caught in seconds only because the log records the
  # resolved parameters — the same rule that fixed the Phase 1 harness.
  LTX_TIER=standard64 LTX_QUANT=int8 LTX_ENC="$encq" LTX_ENC_TREE="$tree" \
  LTX_T2V_PROMPT="a person looks directly at the camera and says clearly: $sent" \
  LTX_T2V_SAVE="$OUT/$tag.mp4" \
    "$BIN" --t2v-spot25 704 512 121 >> "$OUT/$tag.log" 2>&1
  printf '%s\t%s\t%s\n' "$tag" "$arm" "$sent" >> "$MANIFEST"
  printf "  %-26s %4ss\n" "$tag" "$(( $(date +%s) - t0 ))"
}

# Interleaved BY SENTENCE so any session drift lands on every arm equally.
for pair in "s1:$s1" "s2:$s2" "s3:$s3"; do
  sid="${pair%%:*}"; sent="${pair#*:}"
  echo "═══ $sid — \"$sent\""
  gen bf16   ""                    bf16 "$sid" "$sent"   # reference
  gen q8     ""                    q8   "$sid" "$sent"   # UNDER TEST — what low tiers auto-follow to
  gen q4     ltx-2.5-mlx-q4        q8   "$sid" "$sent"   # CONTROL: gate-rejected
  gen poison ltx-2.5-mlx-poison    q8   "$sid" "$sent"   # CONTROL: deliberately broken
done

echo
echo "═══ objective leg — WER per arm, against each arm's OWN scripted line ═══"
while IFS=$'\t' read -r tag arm sent; do
  [ -n "${tag:-}" ] || continue
  printf "── %-26s " "$tag"
  ( cd ../ltx-2-mlx && uv run --with mlx-whisper python ../LTX_TESTING/tools/stt_verify.py \
      "$OUT/$tag.mp4" --expect "$sent" 2>&1 | tail -3 ) || echo "(stt_verify unavailable)"
done < "$MANIFEST"

echo
echo "READ THE CONTROLS FIRST:"
echo "  q4/poison WER ~ bf16  -> instrument is BLIND; the q8 result means nothing. Redesign."
echo "  q4/poison WER >> bf16 -> instrument discriminates; q8 vs bf16 is then interpretable."
echo
echo "⚠️ WER is corroboration. The verdict is the operator's ears, and it should be BLIND —"
echo "   the filenames name the arm, so shuffle before listening."
