"""Stage progress, cancellation and download progress without any weights.

The stage children are replaced by a small script that speaks the same JSON
lines, so these tests run on any Python with Pillow and httpx installed and
never load MLX or a checkpoint.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from h3_image_mlx.api import generate_result
from h3_image_mlx.download import _download_file, fetch_manifest
from h3_image_mlx.events import Cancelled, check_cancel, parse_event
from h3_image_mlx.pipeline import generate

FAKE_STAGE = '''\
import json, os, sys, time
from pathlib import Path

stage, work = sys.argv[1], Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
settings = json.loads((work / "settings.json").read_text())
delay = settings.get("delay", 0)

def emit(event, **fields):
    sys.stderr.write(json.dumps({"event": event, **fields}) + "\\n")
    sys.stderr.flush()

emit("stage", stage=stage, phase="start", pid=os.getpid())
print(f"{stage} working", flush=True)
if settings.get("fail") == stage:
    print("out of memory", flush=True)
    emit("stage", stage=stage, phase="error", error="RuntimeError: out of memory")
    sys.exit(3)
if settings.get("ignore_term") == stage:
    signal_ignored = True
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if settings.get("hang") == stage:
    time.sleep(60)
if stage == "dit":
    for step in range(1, 7):
        if delay:
            time.sleep(delay)
        emit("progress", stage=stage, step=step, total=6, seconds=0.5)
        (work / f"step-{step}.txt").write_text("done")
if stage == "references":
    request = json.loads((work / "request.json").read_text())
    for index in range(len(request["references"])):
        emit("progress", stage=stage, step=index + 1, total=len(request["references"]))
if stage == "vae":
    from PIL import Image
    Image.new("RGB", (64, 64), (10, 20, 30)).save(work / "image.png")
emit("stage", stage=stage, phase="end", seconds=0.25)
'''


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.script = self.root / "fake_stage.py"
        self.script.write_text(FAKE_STAGE)
        self.calls = []
        self.stages = []
        self.progress = []
        self.logs = []
        settings = self.root / "settings.json"
        self.settings_file = settings

    def tearDown(self):
        self.tmp.cleanup()

    def command(self, settings=None):
        (self.root / "settings.json").write_text(json.dumps(settings or {}))

        def command(stage, work):
            self.calls.append(stage)
            (Path(work) / "settings.json").write_text(json.dumps(settings or {}))
            return [sys.executable, str(self.script), stage, str(work)]

        return patch("h3_image_mlx.pipeline._stage_command", side_effect=command)

    def run_generate(self, settings=None, cancel=None, **kwargs):
        settings = settings or {}
        with patch("h3_image_mlx.pipeline.check_models", return_value=self.root), self.command(settings):
            return generate(
                "rusty robot",
                models=self.root,
                width=256,
                height=256,
                on_stage=self.stages.append,
                on_progress=self.progress.append,
                on_log=self.logs.append,
                cancel=cancel,
                **kwargs,
            )

    def test_event_lines_separate_protocol_from_logs(self):
        self.assertEqual(parse_event('{"event": "progress", "step": 1}'),
                         {"event": "progress", "step": 1})
        self.assertIsNone(parse_event("loading weights"))
        self.assertIsNone(parse_event('{"step": 1}'))
        self.assertIsNone(parse_event("{not json}"))

    def test_stage_callbacks_in_order_with_timings(self):
        image = self.run_generate()
        self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(self.calls, ["text", "dit", "vae"])
        self.assertEqual([(e["stage"], e["phase"]) for e in self.stages],
                         [("text", "start"), ("text", "end"),
                          ("dit", "start"), ("dit", "end"),
                          ("vae", "start"), ("vae", "end")])
        self.assertTrue(all(e.get("seconds") is not None for e in self.stages if e["phase"] == "end"))
        self.assertIn("text working", self.logs)

    def test_step_counts_reach_the_total(self):
        self.run_generate()
        steps = [e["step"] for e in self.progress if e["event"] == "progress"]
        self.assertEqual(steps, [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(e["total"] == 6 for e in self.progress if e["event"] == "progress"))

    def test_references_stage_announces_each_image(self):
        refs = []
        for index in range(2):
            path = self.root / f"reference-{index}.png"
            Image.new("RGB", (64, 64), (index, 0, 0)).save(path)
            refs.append(path)
        self.run_generate(references=refs)
        self.assertEqual(self.calls, ["text", "references", "dit", "vae"])
        self.assertEqual([(e["stage"], e["step"], e["total"]) for e in self.progress],
                         [("references", 1, 2), ("references", 2, 2),
                          ("dit", 1, 6), ("dit", 2, 6), ("dit", 3, 6),
                          ("dit", 4, 6), ("dit", 5, 6), ("dit", 6, 6)])

    def test_cancel_before_a_stage_spawns_nothing(self):
        with self.assertRaises(Cancelled):
            self.run_generate(cancel=lambda: True)
        self.assertEqual(self.calls, [])

    def test_cancel_mid_denoise_stops_the_stage_and_keeps_no_image(self):
        started = time.monotonic()
        with self.assertRaises(Cancelled):
            self.run_generate({"delay": 0.4}, cancel=lambda: len(self.progress) >= 2)
        self.assertEqual(self.calls, ["text", "dit"])
        self.assertLess(time.monotonic() - started, 20)
        self.assertIn(("dit", "start"), [(e["stage"], e["phase"]) for e in self.stages])
        self.assertNotIn(("dit", "end"), [(e["stage"], e["phase"]) for e in self.stages])

    def test_cancel_kills_a_stage_that_ignores_sigterm(self):
        event = threading.Event()
        timer = threading.Timer(0.5, event.set)
        timer.start()
        started = time.monotonic()
        try:
            with patch("h3_image_mlx.pipeline.CANCEL_GRACE", 0.5):
                with self.assertRaises(Cancelled):
                    self.run_generate({"ignore_term": "dit", "hang": "dit"}, cancel=event)
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 20)

    def test_failed_stage_reports_the_log_tail(self):
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.run_generate({"fail": "dit"})
        self.assertEqual(self.stages[-1]["phase"], "error")

    def test_api_h3_accepts_callbacks_and_records_stage_timings(self):
        image = Image.new("RGB", (64, 64), "red")
        with patch("h3_image_mlx.pipeline.generate", return_value=image) as h3:
            result = generate_result("prompt", models=self.root, on_progress=self.progress.append,
                                     on_stage=self.stages.append, cancel=lambda: False)
        self.assertIs(result.image, image)
        self.assertEqual(h3.call_args.kwargs["cancel"](), False)
        h3.call_args.kwargs["on_stage"]({"event": "stage", "stage": "dit", "phase": "end", "seconds": 2.5})
        self.assertEqual(result.metrics["stages"], {"dit": 2.5})
        self.assertEqual(self.stages[-1]["stage"], "dit")

    def test_stage_entry_point_emits_start_before_failing(self):
        tmp = self.root / "stage"
        tmp.mkdir()
        (tmp / "request.json").write_text("{}")
        proc = subprocess.run([sys.executable, "-m", "h3_image_mlx._stage", "dit", str(tmp)],
                              capture_output=True, text=True, timeout=120)
        self.assertNotEqual(proc.returncode, 0)
        events = [event for event in map(parse_event, proc.stderr.splitlines()) if event]
        self.assertEqual((events[0]["event"], events[0]["stage"], events[0]["phase"]),
                         ("stage", "dit", "start"))
        self.assertEqual(events[-1].get("phase"), "error")


class DownloadProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.events = []

    def tearDown(self):
        self.tmp.cleanup()

    def test_resumed_download_reports_bytes_and_overall_totals(self):
        first = b"0123456789"
        second = b"abcdefghij"
        manifest = {"files": [
            {"path": "one.bin", "bytes": len(first), "sha256": hashlib.sha256(first).hexdigest(),
             "modelscope": "https://example.test/one"},
            {"path": "two.bin", "bytes": len(second), "sha256": hashlib.sha256(second).hexdigest(),
             "modelscope": "https://example.test/two"},
        ]}
        part = self.root / "two.bin.part"
        part.write_bytes(second[:4])  # resume five-sixths of the way through the second file

        def respond(request):
            url = request.url.path
            raw = first if url.endswith("one") else second
            offset = int(request.headers.get("range", "bytes=0-").split("=")[1].rstrip("-"))
            self.assertEqual(offset, 0 if url.endswith("one") else 4)
            if offset:
                return httpx.Response(206, content=raw[offset:],
                                      headers={"Content-Range": f"bytes {offset}-{len(raw) - 1}/{len(raw)}"})
            return httpx.Response(200, content=raw)

        real_client = httpx.Client

        def client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(respond)
            return real_client(*args, **kwargs)

        with patch("h3_image_mlx.download.httpx.Client", side_effect=client):
            fetch_manifest(self.root, manifest, "modelscope", on_progress=self.events.append)

        phases = [event["phase"] for event in self.events]
        self.assertEqual(phases[0], "start")
        self.assertEqual(phases[-1], "end")
        self.assertIn("verify", phases)
        self.assertEqual(phases.count("file"), 2)
        totals = {event["total"] for event in self.events}
        self.assertEqual(totals, {len(first) + len(second) - 4})
        self.assertEqual(max(event["done"] for event in self.events), len(first) + len(second) - 4)
        self.assertEqual(self.events[-1]["files_done"], 2)
        self.assertTrue((self.root / "two.bin").read_bytes() == second)

    def test_byte_callback_starts_at_the_resume_offset(self):
        raw = b"model-weights"
        entry = {"path": "model.bin", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        dest = self.root / entry["path"]
        dest.with_name(dest.name + ".part").write_bytes(raw[:5])
        seen = []

        def respond(request):
            self.assertEqual(request.headers["range"], "bytes=5-")
            return httpx.Response(206, content=raw[5:],
                                  headers={"Content-Range": f"bytes 5-{len(raw) - 1}/{len(raw)}"})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            _download_file(client, "https://example.test/model", dest, entry,
                           on_bytes=lambda done, total: seen.append((done, total)),
                           on_phase=seen.append)
        self.assertEqual(seen[0], (5, len(raw)))
        self.assertIn("verify", seen)
        self.assertEqual(seen[-1], (len(raw), len(raw)))

    def test_cancel_helper_accepts_predicates_and_events(self):
        import threading

        event = threading.Event()
        check_cancel(None)
        check_cancel(event)
        with self.assertRaises(Cancelled):
            check_cancel(lambda: True)
        event.set()
        with self.assertRaises(Cancelled):
            check_cancel(event)


if __name__ == "__main__":
    unittest.main()
