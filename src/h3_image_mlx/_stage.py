"""Private stage entry point. Each child releases its weights on exit."""
import json
import os
import re
import sys
import time
from pathlib import Path
from .events import Cancelled, Heartbeat, SignalCanceller, check_cancel, emit
from .models import model_paths, manifest_for

CANCELLED_EXIT = 2  # distinct from a failure (1) so the parent can tell them apart


def select_backend(requested, device, precision="int8"):
    manifest_for(precision)
    if precision == "bf16":
        if requested == "w8a8":
            raise ValueError("W8A8 requires INT8 weights; use --backend auto or portable with BF16")
        return "portable"
    native = bool(re.search(r"\bApple M[56]\b", device))
    if requested == "auto":
        return "w8a8" if native else "portable"
    if requested == "w8a8" and not native:
        raise ValueError(f"The W8A8 TensorOps kernel requires an M5/M6 GPU; found {device}")
    return requested


def main(stage, work, cancel=None):
    import mlx.core as mx
    from . import core
    request = json.loads((work / "request.json").read_text())
    info = mx.device_info()
    precision = request.get("precision", "int8")
    check_cancel(cancel)
    dit, text, vae, tokenizer, vae_config = model_paths(precision)
    backend = select_backend(request["backend"], info["device_name"], precision)
    core.backend.set(backend)
    from ._portable import _legacy_metal
    core.legacy_int8_reads.set(precision == "int8" and backend == "portable" and _legacy_metal())
    mx.set_cache_limit(96 << 20)
    models = Path(request["models"])
    refs = request["references"]
    embedding, latents, result = (str(work / (key + ".safetensors")) for key in ("embedding", "references", "latent"))
    mode = request.get("dit_mode")
    auto_resident = mode == "auto" and backend == "w8a8" and info["memory_size"] >= 32 * 1024**3
    # The exact DiT sequence length becomes available after encoding.
    m5 = backend == "w8a8" and bool(re.search(r"\bApple M5\b", info["device_name"]))
    likely_resident = auto_resident and not refs and request["width"] * request["height"] // 1024 < 8192
    core.m5_resident.set(m5 and (mode == "resident" or likely_resident))
    if stage == "text":
        from .text import encode
        # H3_TEXT / H3_TOKENIZER / H3_VAE / H3_DIT: use another checkpoint (e.g. P-H3-Image) instead of the manifest's.
        end = float(os.environ.get("H3_EMPTY_PICTURE") or 0) / 100.0
        end = min(max(end, 0.0), 1.0)
        text_path = os.environ.get("H3_TEXT") or models / text
        tokenizer_path = os.environ.get("H3_TOKENIZER") or models / tokenizer
        if 0.0 < end < 1.0:
            x, tags, _ = encode(text_path, tokenizer_path, request["prompt"], refs,
                                empty_picture=1.0)
            alt_x, alt_tags, _ = encode(text_path, tokenizer_path, request["prompt"], refs,
                                        empty_picture=0.0)
            mx.save_safetensors(embedding, {"embedding": x, "tags": tags,
                                            "embedding_text_only": alt_x,
                                            "tags_text_only": alt_tags})
        else:
            x, tags, _ = encode(text_path, tokenizer_path, request["prompt"], refs,
                                empty_picture=end)
            mx.save_safetensors(embedding, {"embedding": x, "tags": tags})
        emit("progress", stage=stage, step=1, total=1)
    elif stage == "references":
        from .ref_vae import encode_reference
        weights = core.Weights(models / vae, True)
        stats = json.loads((models / vae_config).read_text())
        zs = {}
        for i, ref in enumerate(refs):
            check_cancel(cancel)
            zs[str(i)] = encode_reference(weights, ref, stats)
            mx.eval(zs[str(i)])
            emit("progress", stage=stage, step=i + 1, total=len(refs))
        mx.save_safetensors(latents, zs)
    elif stage == "dit":
        from .dit import denoise
        inputs = mx.load(embedding)
        images = mx.load(latents) if refs else {}
        zs = [images[str(i)] for i in range(len(refs))]
        width, height = request["width"], request["height"]
        rows = inputs["embedding"].shape[0] + width * height // 1024 + sum(z.shape[1] * z.shape[2] // 4 for z in zs)
        resident = mode == "resident" or (auto_resident and rows <= 8192)
        core.m5_resident.set(m5 and resident)

        def progress(step, elapsed, total=None):
            emit("progress", stage=stage, step=step, total=total, seconds=round(elapsed, 3))

        # H3_DIT=<file>: use another DiT checkpoint with the same layout (e.g. one with a LoRA merged in).
        alt_x = inputs.get("embedding_text_only")
        alt_tags = inputs.get("tags_text_only")
        end = min(max(float(os.environ.get("H3_EMPTY_PICTURE") or 0) / 100.0, 0.0), 1.0)
        z, _ = denoise(os.environ.get("H3_DIT") or models / dit, inputs["embedding"], width, height, request["seed"], resident, zs, inputs["tags"], precision=precision, progress=progress, cancel=cancel, text_alt=alt_x, tags_alt=alt_tags, empty_picture_end=end)
        mx.save_safetensors(result, {"latent": z})
    elif stage == "vae":
        from .vae import decode
        check_cancel(cancel)
        stats = json.loads((models / vae_config).read_text())
        decode(os.environ.get("H3_VAE") or models / vae, mx.load(result)["latent"], stats).save(work / "image.png")
        emit("progress", stage=stage, step=1, total=1)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    mx.synchronize()


def run(argv):
    """Run one stage, reporting its lifecycle on stderr as JSON lines.

    The start line is written before MLX is imported so the parent learns the
    stage began before the interpreter finishes loading.
    """
    stage, work = argv[1], Path(argv[2])
    started = time.monotonic()
    emit("stage", stage=stage, phase="start", pid=os.getpid())
    cancel = SignalCanceller().install()
    try:
        with Heartbeat(stage):
            main(stage, work, cancel)
    except Cancelled as error:
        emit("cancelled", stage=stage, reason=str(error) or "Cancelled")
        return CANCELLED_EXIT
    except BaseException as error:
        emit("stage", stage=stage, phase="error", error=f"{type(error).__name__}: {error}")
        raise
    emit("stage", stage=stage, phase="end", seconds=round(time.monotonic() - started, 3))
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv))
