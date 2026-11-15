"""Distribution and dispatch regressions, runnable against an installed wheel."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from h3_image_mlx import generate
from h3_image_mlx._stage import select_backend
from h3_image_mlx.download import _download_file
from h3_image_mlx.models import check_models, manifest_for, model_paths


class PackageTests(unittest.TestCase):
    def test_packaged_runtime_resources(self):
        package = files("h3_image_mlx")
        self.assertFalse(package.joinpath("vae_stats.json").is_file())
        for name in ("MODEL_LICENSE.txt", "weights.json", "weights-bf16.json", "NOTICE.txt",
                     "kernels/LICENSE", "kernels/mpp_tensor_ops.metal",
                     "kernels/w8a8_gemm_body.metal", "kernels/w8a8_epi_plain.metal"):
            self.assertTrue(package.joinpath(name).is_file(), name)

    def test_device_selection(self):
        for device in ("M1", "M2 Max", "M3 Ultra", "M4 Pro"):
            self.assertEqual(select_backend("auto", f"Apple {device}"), "portable")
            with self.assertRaises(ValueError):
                select_backend("w8a8", f"Apple {device}")
        self.assertEqual(select_backend("auto", "Apple M5 Pro"), "w8a8")
        self.assertEqual(select_backend("portable", "Apple M5 Pro"), "portable")

    def test_order_and_stage_boundaries(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            paths = [root / f"reference-{i}.png" for i in reversed(range(16))]
            for i, path in enumerate(paths):
                Image.new("RGB", (32, 32), (i, 0, 0)).save(path)
            stage_script = root / "fake_stage.py"
            stage_script.write_text(
                "import sys\nfrom pathlib import Path\nfrom PIL import Image\n"
                "stage, work = sys.argv[1], Path(sys.argv[2])\n"
                "if stage == 'vae':\n"
                "    Image.new('RGB', (256, 256), 'blue').save(work / 'image.png')\n"
            )
            def command(stage, work):
                calls.append(stage)
                request = json.loads((Path(work) / "request.json").read_text())
                self.assertEqual(request["references"], list(map(str, paths)))
                return [sys.executable, str(stage_script), stage, str(work)]
            with patch("h3_image_mlx.pipeline.check_models", return_value=root), patch("h3_image_mlx.pipeline._stage_command", side_effect=command):
                image = generate("Use <Picture 16>", models=root, references=paths, width=256, height=256)
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
            self.assertEqual(calls, ["text","references","dit","vae"])
            with self.assertRaisesRegex(ValueError, "16"):
                generate("Edit", models=root, references=paths+[paths[0]])

    def test_manifest_override_runs_another_checkpoint_family(self):
        """H3_MANIFEST: a second pinned family in its own directory."""
        manifest = {"total_bytes": 6, "files": [
            {"path": "diffusion_models/other-dit.safetensors", "bytes": 3,
             "sha256": hashlib.sha256(b"abc").hexdigest(),
             "modelscope": "https://example.test/dit", "huggingface": "https://example.test/dit"},
            {"path": "text_encoders/other-text.safetensors", "bytes": 1,
             "sha256": hashlib.sha256(b"d").hexdigest(),
             "modelscope": "https://example.test/text", "huggingface": "https://example.test/text"},
            {"path": "vae/other-vae.safetensors", "bytes": 1,
             "sha256": hashlib.sha256(b"e").hexdigest(),
             "modelscope": "https://example.test/vae", "huggingface": "https://example.test/vae"},
            {"path": "tokenizer.json", "bytes": 1,
             "sha256": hashlib.sha256(b"f").hexdigest(),
             "modelscope": "https://example.test/tokenizer", "huggingface": "https://example.test/tokenizer"},
            {"path": "vae/config.json", "bytes": 0,
             "sha256": hashlib.sha256(b"").hexdigest(),
             "modelscope": "https://example.test/config", "huggingface": "https://example.test/config"},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "weights.json").write_text(json.dumps(manifest))
            for entry in manifest["files"]:
                path = root / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x" * entry["bytes"])
                # Keep the hashes honest for the verification call.
                manifest_entry = next(f for f in manifest["files"] if f["path"] == entry["path"])
                manifest_entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "weights.json").write_text(json.dumps(manifest))

            with patch.dict(os.environ, {"H3_MANIFEST": str(root / "weights.json")}):
                self.assertEqual(model_paths("int8"), tuple(f["path"] for f in manifest["files"]))
                self.assertEqual(manifest_for("int8")["total_bytes"], 6)
                self.assertEqual(check_models(root, hashes=True), root)
                # BF16 is unaffected: the override names one INT8 family.
                self.assertEqual(len(manifest_for("bf16")["files"]), 5)
            # Without the override the pinned H3 manifest is back.
            self.assertIn("minimax_h3_image", model_paths("int8")[0])

            with patch.dict(os.environ, {"H3_MANIFEST": str(root / "missing.json")}):
                with self.assertRaises(FileNotFoundError):
                    manifest_for("int8")

    def test_invalid_request(self):
        with self.assertRaisesRegex(ValueError, "multiples"):
            generate("Prompt", models="missing", width=999)
        with self.assertRaisesRegex(FileNotFoundError, "Missing model file"):
            generate("Prompt", models="missing")

    def test_verify_and_range_resume(self):
        raw = b"test model data"
        entry = {"path":"model.bin", "bytes":len(raw), "sha256":hashlib.sha256(raw).hexdigest()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            dest = root / entry["path"]
            dest.with_name(dest.name+".part").write_bytes(raw[:4])
            def respond(request):
                self.assertEqual(request.headers["range"], "bytes=4-")
                return httpx.Response(206, content=raw[4:], headers={"Content-Range":f"bytes 4-{len(raw)-1}/{len(raw)}"})
            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                _download_file(client, "https://example.test/model", dest, entry)
            with patch("h3_image_mlx.models.MANIFEST", {"files":[entry]}):
                self.assertEqual(check_models(root, hashes=True), root)
                dest.write_bytes(b"x"*len(raw))
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    check_models(root, hashes=True)


if __name__ == "__main__":
    unittest.main()
