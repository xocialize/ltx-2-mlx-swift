"""AB-A-0027 — vendor DISTILLED a2v (Desktop's DistilledA2VPipeline), the missing 2x2 cell.

Runs Desktop's OWN class under Desktop's OWN interpreter and site-packages, so this is their
shipping distilled a2v, not a reconstruction. Same audio / geometry / seed / init frame as the
dev+CFG arm, so the only variable is the tier.
"""
import sys, os, time
R = "/Applications/LTX Desktop.app/Contents/Resources"
sys.path.insert(0, f"{R}/backend")

import torch
from ltx_pipelines.utils.model_paths import ModelPaths
from services.a2v_pipeline.distilled_a2v_pipeline import DistilledA2VPipeline, resolve_image_conditionings  # noqa
from ltx_pipelines.utils.media_io import encode_video

M   = "/Volumes/Satechi/Models/Lightricks/LTX-2.5"
SD  = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]

paths = ModelPaths.from_split(
    transformer_path   = f"{M}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    text_encoder_path  = f"{M}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    video_vae_path     = f"{M}/vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    audio_vae_path     = f"{M}/vae/ltx-2.5-audio-vae-bf16.safetensors",
    duration_head_path = f"{M}/model_patches/ltx-2.5-duration-head-bf16.safetensors",
)

pipe = DistilledA2VPipeline(
    model_paths=paths,
    spatial_upsampler_path=f"{M}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    device=torch.device("mps"),
)

t0 = time.time()
with torch.inference_mode():
    video, audio = pipe(
        prompt="close-up portrait of a red-haired woman talking to the camera against a plain tan "
               "background, mouth moving as she speaks",
        seed=42, height=512, width=704, num_frames=241, frame_rate=24.0,
        images=[(f"{SD}/portrait-f0.png", 0, 1.0)],     # same init frame as the dev+CFG arm
        audio_path=f"{SD}/roxy-voice-stereo.wav",
    )
    encode_video(video=video, fps=24.0, audio=audio, output_path=OUT, video_chunks_number=None)
print(f"[vendor-distilled] wrote {OUT} in {time.time()-t0:.0f}s")
