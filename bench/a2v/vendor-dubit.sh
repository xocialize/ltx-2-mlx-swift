#!/bin/zsh
# AB-A-0027 follow-up: the vendor's DEDICATED lip-sync-adjacent pipeline on the roxy fixture.
# ⚠️ Dub-It GENERATES its audio (stage-1 audio modality is denoised, frozen only for stage 2, and
# the delivered track is decoded from that generated latent). It does NOT lip-sync to a supplied
# track. Video and audio are therefore mutually consistent by construction — a different regime
# from a2v, where the audio is fixed and external.
M=/Volumes/Satechi/Models/Lightricks/LTX-2.5
SD="$(dirname "$0")"
cd /Volumes/Satechi/Development/LTX-2 || exit 1
exec /usr/bin/time -l .venv-mps/bin/python -m ltx_pipelines.dubit \
  --transformer-path        "$M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
  --text-encoder-path       "$M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --video-vae-path          "$M/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --audio-vae-path          "$M/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  --spatial-upsampler-path  "$M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --lora                    /Volumes/Satechi/Models/ltx-lora-cache/lipdub.safetensors 1.0 \
  --reference-video         "$SD/vendor-a2v-distilled.mp4" \
  --prompt "close-up portrait of a red-haired woman speaking to the camera against a plain tan background" \
  --seed 42 --width 704 --height 512 \
  --output-path "$1"
