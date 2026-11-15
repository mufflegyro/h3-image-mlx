"""Text-to-image and ordered multi-reference editing in isolated MLX stages."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Sequence

from PIL import Image
from .events import Cancelled, check_cancel, is_cancelled, parse_event
from .models import check_models, manifest_for, model_paths

# How long a cancelled stage gets to exit on SIGTERM before SIGKILL. MLX blocks
# inside one kernel call, where Python cannot run its handler, so a graceful
# exit is best-effort and the kill is what actually frees the GPU.
CANCEL_GRACE = float(os.environ.get("H3_CANCEL_GRACE", "1.5"))
LOG_TAIL = 400


def embedding_cache(directory, prompt, refs, backend, precision="int8"):
    """Opt-in (H3_EMBED_CACHE=dir): reuse the text-stage output for a repeated prompt + references."""
    root = os.environ.get("H3_EMBED_CACHE")
    if not root:
        return None
    text = model_paths(precision)[1]
    encoder = Path(os.environ.get("H3_TEXT") or directory / text).resolve()
    stat = encoder.stat()
    tokenizer = Path(os.environ.get("H3_TOKENIZER") or directory / model_paths(precision)[3])
    key = [prompt, backend, precision, str(encoder), stat.st_size, stat.st_mtime_ns,
           hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
           [hashlib.sha256(p.read_bytes()).hexdigest() for p in refs]]
    name = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    Path(root).expanduser().mkdir(parents=True, exist_ok=True)
    return Path(root).expanduser() / f"{name}.safetensors"


def _stage_command(stage, work):
    """The argv of one isolated stage. Tests replace this to run without weights."""
    return [sys.executable, "-m", "h3_image_mlx._stage", stage, str(work)]


def _run_stage(stage, work, *, on_stage=None, on_progress=None, cancel=None, on_log=None):
    """Run one stage child, forwarding its stderr events and stopping it on cancel.

    The child owns the stage lifecycle: it writes ``stage start`` before MLX is
    imported and ``stage end`` when the files are on disk, so a parent never has
    to guess when a stage began or how long it took.
    """
    check_cancel(cancel)
    tail = deque(maxlen=LOG_TAIL)
    failed = threading.Event()
    cancelled = threading.Event()

    def pump(stream):
        for raw in stream:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            event = parse_event(line)
            if event is None:
                tail.append(line)
                if on_log is not None:
                    on_log(line)
                continue
            if event["event"] == "cancelled":
                cancelled.set()
                failed.set()
            elif event.get("phase") == "error":
                failed.set()
            if event["event"] in ("stage", "cancelled"):
                if on_stage is not None:
                    on_stage(event)
            elif on_progress is not None:
                on_progress(event)

    # Stage children write protocol and logs to stderr; stdout is folded in so a
    # library's own output still reaches the log instead of filling a pipe.
    proc = subprocess.Popen(_stage_command(stage, work), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    reader = threading.Thread(target=pump, args=(proc.stdout,), daemon=True,
                              name=f"h3-{stage}-stderr")
    reader.start()
    stopped = False
    while True:
        try:
            code = proc.wait(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if is_cancelled(cancel):
                stopped = True
                proc.terminate()
                try:
                    code = proc.wait(timeout=CANCEL_GRACE)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    code = proc.wait()
                break
    reader.join(timeout=5)
    proc.stdout.close()
    if stopped or cancelled.is_set():
        raise Cancelled(f"H3 {stage} stage cancelled")
    if code:
        detail = "\n".join(tail) or f"exit code {code}"
        if not failed.is_set() and on_stage is not None:
            on_stage({"event": "stage", "stage": stage, "phase": "error",
                      "error": detail.splitlines()[-1]})
        raise RuntimeError(f"H3 {stage} stage exited with {code}: {detail}")


def generate(prompt: str, *, models: str | Path, width: int = 1024, height: int = 1024,
             seed: int = 42, references: Sequence[str | Path] = (),
             backend: str = "auto", dit_mode: str = "auto", precision: str = "int8",
             on_stage=None, on_progress=None, cancel=None, on_log=None) -> Image.Image:
    """Render a PIL RGB image. Refer to ordered reference files as <Picture 1>, etc.

    ``on_stage`` receives each stage's lifecycle event (start, end with seconds,
    error), ``on_progress`` receives step and heartbeat events, ``cancel`` is a
    predicate checked between stages and while a stage runs: when it turns true
    the child is terminated and ``Cancelled`` is raised instead of an image.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Provide a nonempty prompt")
    if not all(type(v) is int and 256 <= v <= 4096 and v % 32 == 0 for v in (width, height)):
        raise ValueError("Width and height must be multiples of 32 from 256 through 4096")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if backend not in ("auto", "portable", "w8a8") or dit_mode not in ("auto", "stream", "resident"):
        raise ValueError("Unknown compute backend or DiT mode")
    manifest_for(precision)
    if precision == "bf16" and backend == "w8a8":
        raise ValueError("W8A8 requires INT8 weights; use --backend auto or portable with BF16")
    if isinstance(references, (str, Path)) or len(references) > 16:
        raise ValueError("Supply an ordered list of up to 16 reference files")
    refs = [Path(ref).expanduser().resolve() for ref in references]
    for path in refs:
        with Image.open(path) as im:
            if im.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError(f"Unsupported reference image format: {path}")
            if max(im.size) / min(im.size) * 1024**2 > 12_845_056:
                raise ValueError(f"Reference exceeds the vision encoder aspect ratio budget: {path}")
    directory = check_models(models, precision=precision)
    with tempfile.TemporaryDirectory(prefix="h3-image-mlx-") as tmp:
        work = Path(tmp)
        args = dict(prompt=prompt, models=str(directory), width=width, height=height, seed=seed,
                    references=[str(p) for p in refs], backend=backend, dit_mode=dit_mode, precision=precision)
        (work / "request.json").write_text(json.dumps(args))
        stages = ["text", *(["references"] if refs else []), "dit", "vae"]
        cached = embedding_cache(directory, prompt, refs, backend, precision)
        if cached and cached.is_file():
            shutil.copy(cached, work / "embedding.safetensors")
            stages.remove("text")
        for stage in stages:
            _run_stage(stage, work, on_stage=on_stage, on_progress=on_progress,
                       cancel=cancel, on_log=on_log)
            if stage == "text" and cached:
                shutil.copy(work / "embedding.safetensors", cached.with_suffix(".tmp"))
                cached.with_suffix(".tmp").replace(cached)
        with Image.open(work / "image.png") as image:
            return image.convert("RGB")
