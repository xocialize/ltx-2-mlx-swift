# LTX-2.3 Video VAE vs the Wan VAE — comparison notes

Running notes from porting the LTX-2.3 video VAE, kept as a **sounding board for Wan
VAE development** (wan-core 16-ch WanVAE / 48-ch vae22). Updated as the port proceeds.

## Headline differences

| Axis | LTX-2.3 video VAE | Wan (2.1 WanVAE / 2.2 vae22) |
|---|---|---|
| Latent channels | **128** | 16 (2.1) / 48 (vae22) |
| Compression | **8× temporal, 32× spatial** | 4× temporal, 8× spatial (2.1) |
| Up/downsample | **pixel-shuffle (DepthToSpace / SpaceToDepth)** — rearrange channels↔space | conv stride / nearest-interp + conv |
| Normalization | **PixelNorm** — parameterless RMS over channels (`x/√(mean(x²)+1e-8)`), NO norm weights | GroupNorm (learnable γ/β) |
| ResBlock | pre-activation: PixelNorm→SiLU→Conv ×2 + skip | GroupNorm→SiLU→Conv + learnable conv-shortcut |
| I/O boundary | **spatial patchify/unpatchify 4×4** (RGB 3 ⇄ 48 ch) packed into channels | direct conv to/from 3 ch |
| Causality | decoder **causal=False** (symmetric replicate temporal pad); encoder causal=True | causal throughout (feat_cache streaming) |
| Latent (de)norm | **per-channel mean/std baked into the VAE** (`per_channel_statistics`); encoder uses `_mean_of_means`/`_std_of_means` | fixed `latents_mean`/`latents_std` constant lists |
| Temporal seam | drops first frame after each temporal pixel-shuffle upsample | CausalConv3d feat_cache "Rep" sentinel |

## Why this matters for Wan

- **128-ch vs 16/48-ch** is the biggest divergence: LTX trades far more aggressive spatial
  compression (32× vs 8×) for a fat 128-ch latent. Fewer spatial tokens per frame → the DiT
  sequence length scales differently than Wan's. (LTX H_lat = H/32 vs Wan H/8.)
- **PixelNorm (parameterless) vs GroupNorm**: LTX's VAE carries zero norm weights — the entire
  norm is `rms_norm(weight=None)` over channels. This is the #1 silent-killer axis for Wan VAE
  ports (groupnorm_eps / num_groups); LTX sidesteps it entirely. A useful contrast when chasing
  Wan VAE color-tint/gray bugs.
- **Pixel-shuffle up/downsampling**: LTX never uses interpolation — it's pure channel↔space
  rearrange + Conv3d. The exact channel split order is load-bearing:
  - `pixel_shuffle_3d`: `(c, p1=temporal, p2=H, p3=W)` — c outermost.
  - final `unpatchify_spatial`: `(c, p=1, r=W, q=H)` — **width before height** (differs from
    pixel_shuffle_3d's H-before-W; using the wrong one → checkerboard / H-W swap).
- **Decoder is NON-causal** in LTX-2.3 (zeros spatial pad, symmetric replicate temporal pad) —
  unlike Wan's fully-causal streaming decode. The encoder IS causal. Worth remembering when
  reasoning about the shared streaming-decode memory lever.

## Decoder architecture (latent → pixels)

`conv_in (128→1024)` → 9 `up_blocks` → `PixelNorm→SiLU→conv_out (128→48)` → `unpatchify_spatial(4)` (48→3).

up_blocks (ResStage = N pre-act ResBlock3d; DTS = DepthToSpaceUpsample conv then pixel_shuffle):
| idx | block | detail |
|---|---|---|
| 0 | ResStage 1024 ×2 | |
| 1 | DTS 1024→4096 | pixel_shuffle sf=2,tf=2 → 512ch, **drop first frame** (tf>1) |
| 2 | ResStage 512 ×2 | |
| 3 | DTS 512→4096 | sf=2,tf=2 → 512ch, drop first frame |
| 4 | ResStage 512 ×4 | |
| 5 | DTS 512→512 | sf=1,tf=2 → 256ch, drop first frame |
| 6 | ResStage 256 ×6 | |
| 7 | DTS 256→512 | sf=2,tf=1 → 128ch (spatial only) |
| 8 | ResStage 128 ×4 | |

8× temporal = 2·2·2 (blocks 1,3,5); 32× spatial = 2·2·2(DTS 1,3,7) · 4(unpatchify).

## Encoder (deferred; needed for i2v + upsampler denorm/renorm)

Mirror: `patchify_spatial(4)` → `conv_in (48→128)` → 9 `down_blocks` (ResStage / SpaceToDepthDownsample
with group-mean skip) → `PixelNorm→SiLU→conv_out (1024→129)` → take first 128 ch → normalize. Always causal.

## Port status

- [x] **decoder (latent→pixels) — BIT-EXACT** (`RunLTX2 --vae-decode-gate`, cosine 1.000000,
      maxAbs 1e-5 fp32). latent (1,128,2,4,4) → pixels (1,3,9,128,128) confirms 8×temporal/32×spatial.
      All pixel-shuffle channel orders + first-frame drops + PixelNorm + non-causal pad correct.
- [x] **encoder (pixels→latent) — BIT-EXACT** (`RunLTX2 --vae-encode-gate`, cosine 1.000000,
      maxAbs 1e-5 fp32). pixels (1,3,9,128,128) → latent (1,128,2,4,4). CAUSAL (replicate first
      frame); SpaceToDepthDownsample group-mean skip; patchify(4×4) channel order all correct.
- [ ] streaming / temporal tiling (LTX2_VAE_DECODE_BUDGET_GB) — defer; bit-exact non-tiled first

## Wan-dev takeaways (so far)

1. LTX proves a **parameterless-norm VAE works at quality** — PixelNorm (RMS over channels) replaces
   GroupNorm entirely, removing the groupnorm_eps/num_groups silent-killer class. If a Wan VAE port
   ever fights GroupNorm numerics, LTX is the existence proof for the simpler norm.
2. **Pixel-shuffle (channel↔space) up/downsampling is bit-exact and cheap** vs conv-stride/interp —
   no learned upsample filters beyond the pre-shuffle Conv3d. Channel split order is the only gotcha
   (and it differs between the temporal-aware `pixel_shuffle_3d` and the final `unpatchify_spatial`).
3. 128-ch / 32×-spatial means LTX latents are **spatially tiny but channel-fat** — opposite tradeoff
   from Wan's 16-ch / 8×. Relevant when comparing DiT seq-len / attention-memory scaling.
