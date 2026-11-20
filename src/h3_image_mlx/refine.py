"""Refine a render: encode it, upscale its latent, then denoise a few steps.

This is the pass the app's pixel upscaler cannot do. A draft's latent carries the
composition; the learned upscaler doubles its H/W without a VAE round-trip; and a
short denoise from a start sigma (0.5-0.7 is the useful band) puts real texture
back at the larger canvas instead of interpolating it.

    image = refine("draft.png", models=..., upscaler=".../2d_conv_v1_bf16.safetensors",
                   prompt="...", scale=2, tier=3, sigma=0.5, steps=3, seed=11)

The step list is linear in sigma from `sigma` to 0; `initial` reaches
`dit.denoise`, which interpolates x = (1 - s) x0 + s eps at the first sigma and
runs the engine's own Euler update from there.
"""

import json
from pathlib import Path

from .core import Weights
from .dit import denoise
from .latent_upscale import load as load_upscaler
from .models import model_paths
from .ref_vae import encode_reference
from .text import encode as encode_text
from .vae import decode


def refine(image, models, upscaler, *, prompt, scale=2.0, tier=3, sigma=0.5, steps=3,
           seed=42, precision="int8", progress=None, cancel=None, on_stage=None):
    root = Path(models).expanduser()
    dit, text_path, vae, tokenizer, config = model_paths(precision)
    stats = json.loads((root / config).read_text())
    stage = on_stage or (lambda _name: None)

    if not 0.0 < sigma <= 1.0:
        raise ValueError(f"a refine sigma must be in (0, 1], got {sigma}")
    if steps < 1:
        raise ValueError("a refine needs at least one step")

    stage("text")
    embedding, tags, _ = encode_text(root / text_path, root / tokenizer, prompt, ())
    stage("encode")
    latent = encode_reference(Weights(root / vae, True), image, stats, sample=False)
    stage("upscale")
    upscaled = load_upscaler(upscaler)(latent, scale=scale, tier=tier)
    width = int(upscaled.shape[2]) * 16
    height = int(upscaled.shape[1]) * 16
    grid = [sigma * (1.0 - k / steps) for k in range(steps + 1)]
    stage("refine")
    refined, _ = denoise(str(root / dit), embedding, width, height, seed,
                         resident=False, tags=tags, progress=progress,
                         precision=precision, cancel=cancel,
                         initial=upscaled, sigmas_override=grid)
    stage("decode")
    return decode(str(root / vae), refined, stats)