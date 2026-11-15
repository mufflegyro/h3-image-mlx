"""Download only the pinned H3 components; resume interrupted transfers."""
import fcntl
import shutil
from pathlib import Path
import httpx

from .models import manifest_for, sha256


def _download_file(client, url, dest, entry, on_bytes=None, on_phase=None):
    """Fetch one pinned file, reporting bytes as they arrive.

    ``on_bytes(done, total)`` receives this file's progress, including the
    resume offset at the start, and ``on_phase`` receives ``"verify"`` before
    the hash check so a caller can distinguish transferring from checking a
    multi-gigabyte file.
    """
    partial = dest.with_name(dest.name + ".part")
    size = partial.stat().st_size if partial.exists() else 0
    if size > entry["bytes"]:
        raise ValueError(f"Oversized partial file: {partial}")
    if shutil.disk_usage(dest.parent).free < entry["bytes"] - size:
        raise OSError(f"Insufficient free disk space for {dest.name}")
    if size < entry["bytes"]:
        headers = {"Range": f"bytes={size}-"} if size else {}
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            if response.status_code == 206:
                if not response.headers.get("content-range", "").startswith(f"bytes {size}-"):
                    raise ValueError(f"Server returned an invalid byte range for {dest.name}")
            else:
                size = 0
            if on_bytes:
                on_bytes(size, entry["bytes"])
            with partial.open("ab" if size else "wb") as out:
                for chunk in response.iter_bytes(1024**2):
                    size += len(chunk)
                    if size > entry["bytes"]:
                        raise ValueError(f"Response exceeded expected size for {dest.name}")
                    out.write(chunk)
                    if on_bytes:
                        on_bytes(size, entry["bytes"])
    if partial.stat().st_size != entry["bytes"]:
        raise OSError(f"Incomplete download of {dest.name}. Run fetch again to resume.")
    if on_phase:
        on_phase("verify")
    if sha256(partial) != entry["sha256"]:
        partial.unlink()
        raise ValueError(f"SHA-256 mismatch downloading {dest.name}")
    partial.replace(dest)
    if on_bytes:
        on_bytes(entry["bytes"], entry["bytes"])


def fetch(directory: Path, source="modelscope", *, precision="int8", on_progress=None):
    return fetch_manifest(directory, manifest_for(precision), source, on_progress=on_progress)


def fetch_manifest(directory: Path, manifest, source="modelscope", *, on_progress=None):
    """Shared pinned-asset transport for model and adapter manifests.

    ``on_progress`` receives one dict per step: ``phase`` is ``"start"``,
    ``"file"``, ``"bytes"``, ``"verify"``, ``"skip"`` or ``"end"``; ``path``
    names the file in flight; ``done``/``total`` count bytes for this whole
    fetch and ``files_done``/``files_total`` count finished files.
    """
    if source not in ("modelscope", "huggingface"):
        raise ValueError(f"Unknown model source: {source}")
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    entries = manifest["files"]
    with (directory / ".h3-fetch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing_bytes = 0
        for entry in entries:
            dest = directory / entry["path"]
            if not dest.exists():
                partial = dest.with_name(dest.name + ".part")
                missing_bytes += max(0, entry["bytes"] - (partial.stat().st_size if partial.exists() else 0))
        if missing_bytes > shutil.disk_usage(directory).free:
            raise OSError(f"Need another {missing_bytes:,} available bytes to fetch this model")
        state = {"done": 0, "files": 0, "path": None}

        def report(phase, path=None, file_done=None, file_total=None):
            if on_progress is None:
                return
            on_progress({"phase": phase, "path": path or state["path"],
                         "file_done": file_done, "file_total": file_total,
                         "done": state["done"], "total": missing_bytes,
                         "files_done": state["files"], "files_total": len(entries)})

        report("start")
        with httpx.Client(follow_redirects=True, timeout=90) as client:
            for entry in entries:
                dest = directory / entry["path"]
                state["path"] = entry["path"]
                if dest.exists():
                    if dest.stat().st_size != entry["bytes"] or sha256(dest) != entry["sha256"]:
                        raise ValueError(f"Existing file differs from pinned weights: {dest}")
                    state["files"] += 1
                    report("skip")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                partial = dest.with_name(dest.name + ".part")
                resume = partial.stat().st_size if partial.exists() else 0
                baseline, finish = state["done"], entry["bytes"] - resume
                report("file")

                def on_bytes(done, total, baseline=baseline, resume=resume, finish=finish):
                    state["done"] = baseline + min(max(0, done - resume), finish)
                    report("bytes", file_done=done, file_total=total)

                def on_phase(phase):
                    report(phase, file_done=entry["bytes"], file_total=entry["bytes"])

                _download_file(client, entry[source], dest, entry,
                               on_bytes=on_bytes, on_phase=on_phase)
                state["files"] += 1
        report("end")
    return directory
