"""Progress and cancellation shared by isolated stages and their parent.

A stage child writes one JSON object per line to stderr. Every object carries an
``event`` key, so any other stderr line stays a plain log line and library
output is never mistaken for protocol data.

Events written by a child:

``{"event": "stage", "stage": "dit", "phase": "start" | "end" | "error", ...}``
``{"event": "progress", "stage": "dit", "step": 3, "total": 6, ...}``
``{"event": "heartbeat", "stage": "dit", "elapsed": 30.0}``
``{"event": "cancelled", "stage": "dit", "reason": "..."}``
"""
import json
import os
import signal
import sys
import threading
import time


class Cancelled(RuntimeError):
    """Raised when a caller asks work in progress to stop before it finishes.

    A cancellation is not a failure: parents report it separately and no output
    is published for it.
    """


def is_cancelled(cancel):
    """Whether ``cancel`` asks to stop: a predicate or an Event-like object."""
    if cancel is None:
        return False
    return bool(cancel() if callable(cancel) else cancel.is_set())


def check_cancel(cancel):
    """Raise Cancelled when ``cancel`` is set.

    ``cancel`` may be a zero-argument predicate or a ``threading.Event``-like
    object; ``None`` means the work cannot be cancelled.
    """
    if is_cancelled(cancel):
        raise Cancelled("Cancelled")


def emit(event, *, stream=None, **fields):
    """Write one protocol line, ignoring a closed or redirected stream."""
    stream = stream or sys.stderr
    try:
        stream.write(json.dumps({"event": event, **fields}, default=str) + "\n")
        stream.flush()
    except (BrokenPipeError, ValueError):
        return None
    return {"event": event, **fields}


def parse_event(line):
    """Return the protocol object on a line of stderr, or None for a log line."""
    text = line.strip() if isinstance(line, str) else line.decode("utf-8", "replace").strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("event"), str):
        return None
    return payload


class Heartbeat(threading.Thread):
    """Report that the current stage is alive while it computes.

    MLX evaluates in long blocks and a stage can be quiet for a minute at a
    time; the parent uses these to tell "working" from "wedged".
    """

    def __init__(self, stage, interval=None, stream=None):
        super().__init__(daemon=True, name=f"heartbeat-{stage}")
        self.stage = stage
        self.interval = float(os.environ.get("H3_HEARTBEAT", "10") or 0) if interval is None else interval
        self.stream = stream
        self.started = time.monotonic()
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(self.interval):
            emit("heartbeat", stream=self.stream, stage=self.stage,
                 elapsed=round(time.monotonic() - self.started, 1))

    def stop(self):
        self._stop.set()

    def __enter__(self):
        if self.interval > 0:
            self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


class SignalCanceller:
    """Turn SIGTERM/SIGINT into a predicate a stage can check between blocks.

    The first signal asks for cancellation; a second one is handed back to the
    default handler so a stage wedged in a single kernel still dies.
    """

    def __init__(self):
        self.event = threading.Event()
        self._previous = {}

    def __call__(self):
        return self.event.is_set()

    def install(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous[signum] = signal.signal(signum, self._handle)
        return self

    def _handle(self, signum, frame):
        if self.event.is_set():
            signal.signal(signum, self._previous.get(signum, signal.SIG_DFL))
            os.kill(os.getpid(), signum)
            return
        self.event.set()
