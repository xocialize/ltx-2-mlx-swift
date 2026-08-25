#!/bin/zsh
# AB-A-0027 finding 2, take 2. First attempt produced an off-prompt beach scene with no visible
# face — unevaluable for lip sync. Pin the subject with the OPERATOR'S OWN frame 0 so both lanes
# share audio, opening frame and geometry, and any lip-sync difference is the pipeline's.
M=/Volumes/Satechi/Models/Lightricks/LTX-2.5
SD="$(dirname "$0")"
cd /Volumes/Satechi/Development/LTX-2 || exit 1
exec /usr/bin/time -l .venv-mps/bin/python -m ltx_pipelines.a2vid_two_stage \
  --transformer-path        "$M/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --distilled-lora          "$M/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
  --text-encoder-path       "$M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path          "$M/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --audio-vae-path          "$M/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --spatial-upsampler-path  "$M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --audio-path              "$SD/roxy-voice-stereo.wav" \
  --image                   "$SD/portrait-f0.png" 0 1.0 \
  --prompt "close-up portrait of a red-haired woman talking to the camera against a plain tan background, mouth moving as she speaks" \
  --seed 42 --width 704 --height 512 --num-frames 241 --frame-rate 24 \
  --output-path "$1"
