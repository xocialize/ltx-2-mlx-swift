#!/bin/zsh
# AB-A-0027 — the VENDOR's own local a2v (first-party A2VidPipelineTwoStage = FULL/dev + distilled
# LoRA, CFG in stage 1) on the operator's speech fixture. This is the CEILING arm: if it is
# phoneme-tight and our distilled path is not, the distilled TIER is the limitation, not the port.
M=/Volumes/Satechi/Models/Lightricks/LTX-2.5
S=/Volumes/Satechi/Models/_smoke
cd /Volumes/Satechi/Development/LTX-2 || exit 1
exec /usr/bin/time -l .venv-mps/bin/python -m ltx_pipelines.a2vid_two_stage \
  --transformer-path        "$M/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --distilled-lora          "$M/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  --text-encoder-path       "$M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path          "$M/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --audio-vae-path          "$M/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --spatial-upsampler-path  "$M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --audio-path              "/private/tmp/claude-501/-Volumes-Satechi-Development-mlxengine-video-ltx/933a325c-be65-4a90-b506-69f1c8450c89/scratchpad/roxy-voice-stereo.wav" \
  --prompt "a woman speaking directly to the camera, clear facial expression, natural lip movement" \
  --seed 42 --width 704 --height 512 --num-frames 241 --frame-rate 24 \
  --output-path "$1"
