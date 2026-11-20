"""MiniMax H3's latent upscaler (2d_conv_v1) in MLX.

The release (LBH-123-AI/Minimax_h3_latent_Upscaler, Apache-2.0, 182M parameters)
resizes a 24-channel H3 latent by 1.1x-4x in H/W while keeping the frame count,
so a draft can reach a delivery canvas without the decode -> pixel upscale ->
encode round trip. This port covers the 2d_conv_v1 checkpoint: 12 ResBlocks on
each side of the mid bilinear upsample at 640 channels, a Fourier scale
embedding with its 6-step sharpness tier, and the two depthwise temporal convs.
A still image is one frame, so each temporal conv collapses to its centre tap
plus a 1x1 projection -- exactly what the padding makes of a one-frame clip.

Latents are (channels, H, W) and already normalised the way the VAE stores them;
this module applies the upscaler's own per-channel normalisation around the net
and returns a latent of the same channel count, `scale` times larger.
"""

import json
import struct

import mlx.core as mx
import mlx.nn as nn
import numpy as np

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]

GROUPS = 32
BLOCKS = 12
CHANNELS = 640


def load(path):
    """The 2d_conv_v1 checkpoint, with every conv weight moved to MLX's layout."""
    raw = dict(mx.load(str(path), format="safetensors"))
    weights = {}
    for key, value in raw.items():
        if key.endswith(".weight") and value.ndim == 4:      # Conv2d (out, in, kh, kw)
            value = value.transpose(0, 2, 3, 1)
        elif key.endswith(".weight") and value.ndim == 5:    # Conv3d (out, in, kt, kh, kw)
            value = value.transpose(0, 2, 3, 4, 1)
        weights[key] = value.astype(mx.float32)
    return Upscaler(weights)


class Upscaler:
    def __init__(self, weights):
        self.w = weights
        self.mean = mx.array(np.asarray(LATENTS_MEAN, dtype=np.float32)).reshape(1, 1, 1, -1)
        self.std = mx.array(np.asarray(LATENTS_STD, dtype=np.float32)).reshape(1, 1, 1, -1)

    def __call__(self, latent, scale=2.0, tier=1):
        """(C, H, W) -> (C, round(H*scale), round(W*scale))."""
        x = mx.array(latent).astype(mx.float32)
        if x.ndim != 3:
            raise ValueError(f"expected a (C, H, W) latent, got {x.shape}")
        x = x.transpose(1, 2, 0)[None]                       # (1, H, W, C)
        height, width = int(x.shape[1]), int(x.shape[2])
        target = (int(round(height * scale)), int(round(width * scale)))
        x = (x - self.mean) / self.std

        emb = self._embed(target[0] / height, target[1] / width, tier)
        h = self._conv(x, "resizer.conv_in")
        for i in range(BLOCKS):
            h = self._resblock(h, emb, f"resizer.in_blocks.{i}")
            if i % 2 == 0:
                h = self._temporal(h, "temporal_blocks.0")
        h = nn.Upsample(scale_factor=target[0] / height, mode="linear")(h)
        if (h.shape[1], h.shape[2]) != target:
            raise ValueError(f"bilinear resize produced {h.shape[1:3]}, wanted {target}")
        for i in range(BLOCKS):
            h = self._resblock(h, emb, f"resizer.out_blocks.{i}")
            if i % 2 == 0:
                h = self._temporal(h, "temporal_blocks.1")
        h = self._conv(nn.silu(self._gn(h, "resizer.norm_out")), "resizer.conv_out")
        out = (h * self.std + self.mean)[0].transpose(2, 0, 1)  # (C, H', W')
        mx.eval(out)
        return out

    # -- pieces -------------------------------------------------------------------------------------------------------

    def _gn(self, x, prefix, eps=1e-5):
        n, h, w, c = x.shape
        y = x.reshape(n, h, w, GROUPS, c // GROUPS)
        mean = y.mean(axis=(1, 2, 4), keepdims=True)
        var = ((y - mean) ** 2).mean(axis=(1, 2, 4), keepdims=True)
        y = ((y - mean) / mx.sqrt(var + eps)).reshape(n, h, w, c)
        weight = self.w[prefix + ".weight"].reshape(1, 1, 1, c)
        bias = self.w[prefix + ".bias"].reshape(1, 1, 1, c)
        return y * weight + bias

    def _conv(self, x, prefix):
        return mx.conv2d(x, self.w[prefix + ".weight"], stride=1, padding=1) + self.w[prefix + ".bias"]

    def _linear(self, x, prefix):
        return x @ self.w[prefix + ".weight"].T + self.w[prefix + ".bias"]

    def _resblock(self, x, emb, prefix):
        h = self._conv(nn.silu(self._gn(x, prefix + ".in_layers.0")), prefix + ".in_layers.2")
        e = self._linear(nn.silu(emb), prefix + ".emb_layers.1")
        scale, shift = mx.split(e, 2, axis=-1)
        h = self._gn(h, prefix + ".out_norm") * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
        h = self._conv(nn.silu(h), prefix + ".out_layers.2")
        return x + h

    def _temporal(self, x, prefix):
        """One frame: the depthwise temporal kernel contributes only its centre tap."""
        dw = self.w[prefix + ".dwconv.weight"]               # (C, kt, 1, 1, 1)
        centre = dw[:, dw.shape[1] // 2].reshape(-1)
        y = (nn.silu(self._gn(x, prefix + ".norm")) * centre.reshape(1, 1, 1, -1)
             + self.w[prefix + ".dwconv.bias"].reshape(1, 1, 1, -1))
        pw = self.w[prefix + ".pwconv.weight"][:, 0]
        y = mx.conv2d(y, pw, stride=1, padding=0) + self.w[prefix + ".pwconv.bias"]
        return x + y

    def _embed(self, scale_h, scale_w, tier):
        freqs = (2.0 ** np.arange(16)).astype(np.float32)
        angle = lambda value: np.concatenate([np.sin(value * freqs), np.cos(value * freqs)])
        t = float(tier) - 1.0
        vector = np.concatenate([angle(scale_h), angle(scale_w), angle(t)]).astype(np.float32)
        e = mx.array(vector)[None]
        return self._linear(nn.silu(self._linear(e, "resizer.embed.linear.0")), "resizer.embed.linear.2")