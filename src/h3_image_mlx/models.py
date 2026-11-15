"""Pinned INT8 and BF16 inference components, sharing the VAE and tokenizer."""
import hashlib
import json
import os
from pathlib import Path

MANIFEST = json.loads(Path(__file__).with_name("weights.json").read_text())
BF16_MANIFEST = json.loads(Path(__file__).with_name("weights-bf16.json").read_text())
DIT, TEXT, VAE, TOKENIZER, VAE_CONFIG = (entry["path"] for entry in MANIFEST["files"])


def manifest_for(precision="int8"):
    if precision == "int8":
        return pinned_manifest()
    if precision == "bf16":
        return BF16_MANIFEST
    raise ValueError(f"Unknown weight precision: {precision}")


def pinned_manifest():
    """H3_MANIFEST: run another pinned INT8 checkpoint family.

    The file has the shape of weights.json — one entry per component, each with
    a path, size, SHA-256 and both download URLs — so paths, verification and
    fetch all follow it unchanged. H3 Studio points it at the manifest sitting
    beside a model's own files (e.g. P-H3-Image), which keeps that family in its
    own directory instead of over H3's.
    """
    path = os.environ.get("H3_MANIFEST")
    if not path:
        return MANIFEST
    return json.loads(Path(path).expanduser().read_text())


def model_paths(precision="int8"):
    return tuple(entry["path"] for entry in manifest_for(precision)["files"])


def check_models(directory, *, hashes=False, precision="int8"):
    directory = Path(directory).expanduser().resolve()
    for entry in manifest_for(precision)["files"]:
        path = directory / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing model file: {path}. Run h3-image-mlx fetch --models '{directory}' --precision {precision} to complete this directory.")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Unexpected file size: {path}")
        if hashes and sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
    return directory


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
