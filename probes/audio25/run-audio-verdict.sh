#!/bin/bash
# LTX-2.5 AUDIO VERDICT (Phase 1) — does 2.5 generate intelligible speech, or 2.3's babble?
#
# 🔑 WHY THIS IS OPEN. LTX-2.3 distilled audio is prosodic/phonetic, NOT intelligible speech — a
# MODEL property, oracle A/B'd (I7). AB-R-0002 found 2.5's audio STACK (VAE + vocoder) byte-identical
# to 2.3's, which is often misread as settling this. It does not: the stack is the DECODER. Audio
# GENERATION comes from the joint-AV DiT, and 2.5's DiT is a different model. The 2.3 finding
# therefore neither transfers nor is refuted — it is simply unmeasured on 2.5.
#
# ⚠️ WHY IT MATTERS BEFORE APP SCAFFOLDING. 2.5's headline is JOINT AUDIO+VIDEO. Whether the product
# promises speech, ambient/foley only, or audio-off-by-default is a different app — and the UI is
# cheaper to design before it assumes speech than after.
#
# PROTOCOL: two arms on purpose.
#   SPEECH arms   — prompts that imply a person talking. If 2.5 inherits 2.3's behaviour these come
#                   out as phonetic babble with speech-like prosody.
#   AMBIENT arms  — rain, traffic, a workshop. These are the CONTROL: non-speech audio can be
#                   perfectly good while speech is babble, and judging only speech prompts would
#                   condemn the whole feature on its hardest case.
#
# Objective leg: stt_verify.py (mlx-whisper) transcribes each clip. High WER on a known phrase is
# evidence of unintelligibility; it is NOT evidence of bad audio for the ambient arms, where a
# transcript SHOULD be empty or nonsense. Read WER only for the speech arms.
#
# ⚠️ The operator's ears are the verdict. WER is corroboration, not the decision — the same posture
# every quality call in this project has taken.
set -u
cd "$(dirname "$0")/../.." || exit 1
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
OUT=~/Desktop/ltx25-audio-verdict
mkdir -p "$OUT"
BIN=./.build/release/RunLTX2
xcrun swift build -c release --product RunLTX2 2>&1 | grep -E "error:|Build complete" | tail -1

gen() { # tag prompt
  local tag="$1" prompt="$2" t0=$(date +%s)
  LTX_TIER=standard64 LTX_QUANT=int8 \
  LTX_T2V_PROMPT="$prompt" LTX_T2V_SAVE="$OUT/$tag.mp4" \
    "$BIN" --t2v-spot25 704 512 121 > "$OUT/$tag.log" 2>&1
  printf "  %-22s %4ss  %s\n" "$tag" "$(( $(date +%s) - t0 ))" \
    "$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,duration -of csv=p=0 "$OUT/$tag.mp4" 2>/dev/null)"
}

echo "═══ SPEECH arms — the case 2.3 fails ═══"
gen speech-01-piece-to-camera "a woman looks directly at the camera and says clearly: the quick brown fox jumps over the lazy dog"
gen speech-02-newsreader      "a news anchor at a desk reading the evening headlines, clear studio audio"
gen speech-03-street-busker   "a street musician singing a simple folk song to a small crowd"
echo "═══ AMBIENT arms — the CONTROL (non-speech audio can be fine while speech is babble) ═══"
gen ambient-01-rain           "heavy rain on a tin roof at night, thunder in the distance"
gen ambient-02-traffic        "a busy city intersection at rush hour, traffic and horns"
gen ambient-03-workshop       "a woodworking shop, a hand plane moving across timber"

echo
echo "═══ objective leg — SPEECH arms only ═══"
for f in "$OUT"/speech-*.mp4; do
  [ -e "$f" ] || continue
  echo "── $(basename "$f")"
  ( cd ../ltx-2-mlx && uv run python ../LTX_TESTING/tools/stt_verify.py "$f" \
      --expect "the quick brown fox jumps over the lazy dog" 2>&1 | tail -3 ) || \
    echo "   (stt_verify unavailable — operator listen is the verdict regardless)"
done
echo
echo "clips in $OUT — LISTEN to all six. The question is not 'is it similar to 2.3' but:"
echo "  speech arms  → intelligible words, or phonetic babble with speech-like prosody?"
echo "  ambient arms → is non-speech audio actually good? (this decides ambient-only as a product)"
