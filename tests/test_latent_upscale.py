"""The 2D latent upscaler against a PyTorch reference, on the released weights.

The golden output was produced by the ComfyUI node's own architecture (the
ResBlock/Fourier/temporal classes, on CPU, float32) with this checkpoint and the
same seed-3 latent; the port must agree to float noise, not just "look right".

Requires the release (0.36 GB), which is not committed:

    H3_UPSCALER=/path/to/minimax_h3_latent_upscaler_2d_conv_v1_bf16.safetensors \
        uv run --with pytest --python 3.13 pytest tests/test_latent_upscale.py
"""

import os
from pathlib import Path

import numpy as np
import pytest

from h3_image_mlx import latent_upscale

GOLDEN = Path(__file__).with_name("data") / "upscaler2d_v1_golden.npz"
CHECKPOINT = os.environ.get("H3_UPSCALER")


@pytest.mark.skipif(not CHECKPOINT, reason="set H3_UPSCALER to the 2d_conv_v1 checkpoint")
def test_matches_the_pytorch_reference():
    golden = np.load(GOLDEN)
    port = latent_upscale.load(CHECKPOINT)
    out = np.asarray(port(golden["latent"], scale=2.0, tier=3))
    assert out.shape == golden["output"].shape
    error = np.abs(out - golden["output"])
    assert error.max() < 2e-4, error.max()
    assert error.mean() < 2e-5, error.mean()


def test_a_latent_must_be_three_dimensional():
    port = latent_upscale.Upscaler({})
    with pytest.raises(ValueError, match="C, H, W"):
        port(np.zeros((4, 4), dtype=np.float32))