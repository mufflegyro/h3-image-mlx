"""LoRA adapters for the transformer, applied beside the quantized weights.

Our Linears are fused: `attn.qkv_proj` holds Q|K|V in one projection and
`mlp.fc1` holds [gate; up]. diffusers' adapters target `attn.to_q`, `to_k`, `to_v`,
`to_out.0`, `ff.net.0.proj` and `ff.net.2`, so one adapter module becomes one or
more slices of an engine projection: Q, K and V concatenated in that order, and
`ff.net.0.proj`'s rows moved from [value; gate] to [gate; up].

Files: diffusers keys (`transformer.<module>...` or bare `<module>...`), PEFT's
`.default.` infix, the fork's already-fused `diffusion_model.<engine module>` form,
and a fused `attn.to_qkv` target. Adapters trained against the original
checkpoint's own module names (`lora_unet_...`, `lora_down` / `lora_up`) are
refused with a message that names the conversion rather than applied wrongly.

The adapter is never merged into the weights: int8 ConvRot weights are rotated
and quantized, so the delta runs beside the quantized GEMM on the unrotated
activations, at full precision, exactly as the hook did before it read real files.

`H3_LORA` names the file; `H3_LORA_SCALE` is its strength (1.0 is as trained, 0
is the run without it).
"""

import json
import math
import re
import struct
from dataclasses import dataclass, field

import mlx.core as mx

# The six Linears of each block a diffusers adapter can target, and where each
# lands in our fused projection: (engine module, slot). The slots of a fused
# projection are concatenated in the order listed here.
FUSED_TARGETS = {
    "attn.to_q": ("attn.qkv_proj", "q"),
    "attn.to_k": ("attn.qkv_proj", "k"),
    "attn.to_v": ("attn.qkv_proj", "v"),
    "attn.to_qkv": ("attn.qkv_proj", "qkv"),
    "attn.to_out.0": ("attn.out_proj", None),
    "ff.net.0.proj": ("mlp.fc1", None),
    "ff.net.2": ("mlp.fc2", None),
}
QKV_SLOTS = ("q", "k", "v")
LAYER_PREFIXES = (
    (re.compile(r"^transformer_blocks\.(\d+)"), "blocks"),
    (re.compile(r"^token_refiner\.refiner_blocks\.(\d+)"), "token_refiner.blocks"),
)
ENGINE_LAYER_PREFIXES = (
    (re.compile(r"^blocks\.(\d+)"), "blocks"),
    (re.compile(r"^token_refiner\.blocks\.(\d+)"), "token_refiner.blocks"),
)
KEY_RE = re.compile(r"^(?P<module>.+)\.lora_(?P<matrix>[AB])\.weight$")
METADATA_KEY = "lora_adapter_metadata"
UNSUPPORTED = ("use_dora", "use_rslora", "lora_bias", "fan_in_fan_out")


@dataclass
class Part:
    """One module of the adapter, as a slice of one engine projection."""

    a: mx.array
    b: mx.array
    scaling: float
    halves: bool = False  # ff.net.0.proj: [value; gate] -> [gate; up]

    @property
    def rank(self):
        return int(self.a.shape[0])


@dataclass
class Adapter:
    """Every module of one file, grouped by the engine projection it feeds."""

    path: str
    scale: float
    parts: dict = field(default_factory=dict)  # engine prefix -> [Part]

    @property
    def modules(self):
        return sorted(self.parts)

    def delta(self, prefix, x):
        """The adapter's contribution to `x @ W.T`, or None when it does not touch this projection."""
        parts = self.parts.get(prefix)
        if not parts:
            return None
        chunks = []
        for part in parts:
            if x.shape[-1] != part.a.shape[1]:
                raise ValueError(
                    f"{self.path}: {prefix} takes {part.a.shape[1]}-wide input, "
                    f"the transformer feeds {x.shape[-1]}"
                )
            low = x.astype(mx.float32) @ part.a.astype(mx.float32).T
            chunk = (low @ part.b.astype(mx.float32).T) * part.scaling
            if part.halves:
                half = chunk.shape[-1] // 2
                chunk = mx.concatenate([chunk[:, half:], chunk[:, :half]], axis=-1)
            chunks.append(chunk)
        return chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=-1)


def load(path, scale=1.0):
    """Read a LoRA file into an Adapter, refusing anything it cannot apply faithfully."""
    check_scale(scale)
    tensors = dict(mx.load(str(path)))
    metadata = file_metadata(path)
    return read(tensors, metadata, scale=scale, path=str(path))


def read(tensors, metadata=None, scale=1.0, path="<adapter>"):
    check_scale(scale)
    _refuse_dora(tensors)
    tensors = _normalize(tensors, path)
    config = _metadata_config(metadata or {}, path)
    modules = _pairs(tensors, path)
    normalized = {}
    for module, (a_name, b_name) in modules.items():
        engine, slot = _engine_target(module, path)
        alpha = _alpha(module, tensors, a_name, config, path)
        normalized.setdefault(engine, {})[slot] = Part(
            a=tensors[a_name],
            b=tensors[b_name],
            scaling=scale * alpha / int(tensors[a_name].shape[0]),
            halves=engine.endswith(".mlp.fc1"),
        )
    parts = {}
    for engine, slots in normalized.items():
        if engine.endswith(".attn.qkv_proj") and "qkv" not in slots:
            missing = [slot for slot in QKV_SLOTS if slot not in slots]
            if missing:
                raise ValueError(
                    f"{path}: {engine} is a fused projection, so an adapter for it must carry "
                    f"all of Q, K and V; this one is missing {', '.join(missing)}"
                )
            parts[engine] = [slots[slot] for slot in QKV_SLOTS]
        else:
            if len(slots) != 1:
                raise ValueError(f"{path}: {engine} got more than one adapter module: {sorted(slots)}")
            parts[engine] = [next(iter(slots.values()))]
    if not parts:
        raise ValueError(f"{path}: no LoRA tensors in this file")
    return Adapter(path=path, scale=scale, parts=parts)


# -- reading a file ---------------------------------------------------------------------------------------------------


def file_metadata(path):
    """The safetensors header's `__metadata__` (mx.load returns tensors only)."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return header.get("__metadata__") or {}


def _normalize(tensors, path):
    """One component's keys without their prefixes, or a refusal that says why."""
    if any(key.startswith("lora_unet_") or key.endswith(".lora_down.weight") for key in tensors):
        raise NotImplementedError(
            f"{path}: this adapter was trained against the original checkpoint's module names "
            "(`lora_unet_...` / `lora_down`); export it to diffusers names first"
        )
    normalized = {}
    for key, value in tensors.items():
        name = key.removeprefix("diffusion_model.")
        name = name.removeprefix("transformer.")
        name = name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
        normalized[name.removeprefix("transformer.")] = value
    if not any(k.endswith(".lora_A.weight") for k in normalized):
        raise ValueError(f"{path}: no lora_A/lora_B tensors in this file")
    return normalized


def _pairs(tensors, path):
    modules = {}
    alphas = set()
    for key in tensors:
        match = KEY_RE.match(key)
        if match:
            modules.setdefault(match["module"], {})[match["matrix"]] = key
        elif key.endswith(".alpha"):
            alphas.add(key[: -len(".alpha")])
        else:
            raise ValueError(f"{path}: unexpected tensor {key!r}; only lora_A / lora_B / alpha are supported")
    stray = sorted(module for module in alphas if module not in modules)
    if stray:
        raise ValueError(f"{path}: an alpha with no lora_A / lora_B pair: {stray[:4]}")
    missing = sorted(m for m, v in modules.items() if set(v) != {"A", "B"})
    if missing:
        raise ValueError(f"{path}: modules with only one of lora_A / lora_B: {missing[:8]}")
    return {m: (v["A"], v["B"]) for m, v in modules.items()}


def _engine_target(module, path):
    for pattern, replacement in LAYER_PREFIXES:
        match = pattern.match(module)
        if match:
            rest = module[match.end():]
            if not rest.startswith("."):
                continue
            target = FUSED_TARGETS.get(rest[1:])
            if target is None:
                raise ValueError(
                    f"{path}: {module} is not one of the block Linears this engine can take "
                    f"({', '.join(sorted(FUSED_TARGETS))})"
                )
            return f"{replacement}.{match.group(1)}.{target[0]}", target[1]
    for pattern, replacement in ENGINE_LAYER_PREFIXES:
        match = pattern.match(module)
        if match:
            rest = module[match.end():]
            fused = {
                "attn.qkv_proj": "qkv",
                "attn.out_proj": None,
                "mlp.fc1": None,
                "mlp.fc2": None,
            }
            name = rest[1:] if rest.startswith(".") else rest
            if name not in fused:
                raise ValueError(f"{path}: {module} is not a block Linear this engine can take")
            return f"{replacement}.{match.group(1)}.{name}", fused[name]
    raise ValueError(f"{path}: {module} is not a block Linear this engine can take")


def _metadata_config(metadata, path):
    raw = metadata.get(METADATA_KEY)
    if not raw:
        return None
    try:
        config = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {METADATA_KEY} is not JSON: {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"{path}: {METADATA_KEY} is not a JSON object")
    on = [name for name in UNSUPPORTED if config.get(name)]
    if config.get("bias") not in (None, "none"):
        on.append(f"bias={config['bias']!r}")
    if on:
        raise NotImplementedError(f"{path}: this adapter was saved with {on}, which are not implemented")
    if "r" not in config or "lora_alpha" not in config:
        raise ValueError(f"{path}: {METADATA_KEY} has no `r` / `lora_alpha`")
    return config


def _alpha(module, tensors, a_name, config, path):
    """PEFT's rule: the module's own alpha, else the file's network alpha, else the rank."""
    rank = int(tensors[a_name].shape[0])
    alpha = tensors.get(module + ".alpha")
    if alpha is not None:
        value = float(alpha.item() if hasattr(alpha, "item") else alpha)
    elif config is not None:
        value = float(config["lora_alpha"])
    else:
        value = float(rank)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}: {module}'s lora_alpha is {value!r}; it must be positive and finite")
    return value


def check_scale(scale):
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError(f"a LoRA scale must be a finite number, got {scale!r}")


def _refuse_dora(tensors):
    if any("dora_scale" in key or "lora_magnitude_vector" in key for key in tensors):
        raise NotImplementedError("DoRA adapters are not supported")