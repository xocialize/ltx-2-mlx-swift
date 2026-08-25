#!/bin/zsh
# AB-P-0004 arm B — vendor first-party DistilledPipeline on PyTorch-MPS.
# /usr/bin/time -l gives max RSS: ONE instrument, identical for both arms (our Metal numbers and
# their psutil numbers have complementary blind spots, AB-R-0131, so neither side's own is usable).
M=/Volumes/Satechi/Models/Lightricks/LTX-2.5
W=$1; H=$2; F=$3; OUT=$4
cd /Volumes/Satechi/Development/LTX-2 || exit 1
exec /usr/bin/time -l .venv-mps/bin/python -m ltx_pipelines.distilled \
  --transformer-path        "$M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
  --text-encoder-path       "$M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path          "$M/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --audio-vae-path          "$M/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --duration-head-path      "$M/model_patches/ltx-2.5-duration-head-bf16.safetensors" \
  --spatial-upsampler-path  "$M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --prompt "a woman turns toward the camera and smiles, warm afternoon light" \
  --seed 42 --width "$W" --height "$H" --num-frames "$F" --frame-rate 24 \
  --output-path "$OUT"
