from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from io import BytesIO
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_worker = None
_jobs = {}
_jobs_lock = threading.Lock()
_generation_lock = threading.Lock()


class Spar3DWorker:
    def __init__(
        self,
        *,
        repo_path: str,
        python_bin: str,
        device: str,
        low_vram_mode: bool,
        pretrained_model: str,
        texture_resolution: int,
        remesh_option: str,
        runtime_dir: str,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.python_bin = python_bin
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.pretrained_model = pretrained_model
        self.texture_resolution = texture_resolution
        self.remesh_option = remesh_option
        self.runtime_dir = Path(runtime_dir)
        self.jobs_dir = self.runtime_dir / "jobs"
        self.results_dir = self.runtime_dir / "results"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def health_payload(self) -> dict:
        with _jobs_lock:
            queued = sum(1 for item in _jobs.values() if item.get("status") == "queued")
            processing = sum(1 for item in _jobs.values() if item.get("status") == "processing")
        return {
            "status": "ok",
            "message": "SPAR3D local single-image service is ready.",
            "jobs": len(_jobs),
            "queued_jobs": queued,
            "processing_jobs": processing,
            "repo_path": str(self.repo_path),
            "python_bin": self.python_bin,
            "device": self.device,
            "low_vram_mode": self.low_vram_mode,
            "pretrained_model": self.pretrained_model,
        }

    def generate(self, uid: str, payload: dict) -> None:
        try:
            with _jobs_lock:
                _jobs[uid] = {"status": "queued"}
            with _generation_lock:
                with _jobs_lock:
                    _jobs[uid] = {"status": "processing"}

                job_dir = self.jobs_dir / uid
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
                job_dir.mkdir(parents=True, exist_ok=True)

                image = _load_image_from_base64(payload["image"])
                input_path = job_dir / "input.png"
                image.save(input_path)

                command = [
                    self.python_bin,
                    "run.py",
                    str(input_path),
                    "--output-dir",
                    str(job_dir),
                    "--device",
                    self.device,
                    "--pretrained-model",
                    self.pretrained_model,
                    "--texture-resolution",
                    str(int(payload.get("texture_resolution", self.texture_resolution))),
                    "--remesh_option",
                    str(payload.get("remesh_option", self.remesh_option)),
                ]
                if self.low_vram_mode or bool(payload.get("low_vram_mode", False)):
                    command.append("--low-vram-mode")

                env = os.environ.copy()
                if self.device == "cpu":
                    env["SPAR3D_USE_CPU"] = "1"
                if self.device == "mps":
                    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

                proc = subprocess.run(
                    command,
                    cwd=self.repo_path,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(_condense_error(proc.stderr or proc.stdout))

                mesh_path = job_dir / "0" / "mesh.glb"
                if not mesh_path.exists():
                    raise FileNotFoundError(f"SPAR3D did not produce expected GLB: {mesh_path}")

                result_path = self.results_dir / f"{uid}.glb"
                shutil.copy2(mesh_path, result_path)
                with _jobs_lock:
                    _jobs[uid] = {"status": "completed", "path": str(result_path)}
        except Exception as exc:
            traceback.print_exc()
            with _jobs_lock:
                _jobs[uid] = {"status": "error", "message": str(exc)}


@app.get("/health")
async def health():
    return _worker.health_payload()


@app.post("/send")
async def send(request: Request):
    payload = await request.json()
    image_b64 = payload.get("image")
    if not image_b64:
        return JSONResponse({"error": "image is required"}, status_code=400)
    uid = str(uuid.uuid4())
    threading.Thread(target=_worker.generate, args=(uid, payload), daemon=True).start()
    return {"uid": uid}


@app.get("/status/{uid}")
async def status(uid: str):
    with _jobs_lock:
        job = _jobs.get(uid)
    if job is None:
        return {"status": "processing"}
    if job["status"] == "completed":
        model_base64 = base64.b64encode(Path(job["path"]).read_bytes()).decode("utf-8")
        return {"status": "completed", "model_base64": model_base64}
    if job["status"] == "error":
        return {"status": "error", "message": job.get("message", "unknown")}
    return {"status": job["status"]}


def _load_image_from_base64(image_b64: str) -> Image.Image:
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


def _condense_error(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "SPAR3D subprocess failed without stderr output."
    return "\n".join(lines[-20:])


def _default_repo_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "third_party" / "stable-point-aware-3d")


def _default_python_bin() -> str:
    root = Path(__file__).resolve().parents[1]
    return str(root / ".venv-spar3d" / "bin" / "python3")


def _default_runtime_dir() -> str:
    return str(Path(__file__).resolve().parents[1] / ".runtime" / "spar3d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local SPAR3D single-image bridge service.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--repo-path", type=str, default=os.environ.get("SPAR3D_REPO_PATH", _default_repo_path()))
    parser.add_argument("--python-bin", type=str, default=os.environ.get("SPAR3D_PYTHON_BIN", _default_python_bin()))
    parser.add_argument("--device", type=str, default=os.environ.get("SPAR3D_DEVICE", "cpu"))
    parser.add_argument("--low-vram-mode", action="store_true")
    parser.add_argument("--pretrained-model", type=str, default=os.environ.get("SPAR3D_MODEL_ID", "stabilityai/stable-point-aware-3d"))
    parser.add_argument("--texture-resolution", type=int, default=512)
    parser.add_argument("--remesh-option", type=str, default="none")
    parser.add_argument("--runtime-dir", type=str, default=os.environ.get("SPAR3D_RUNTIME_DIR", _default_runtime_dir()))
    args = parser.parse_args()

    global _worker
    _worker = Spar3DWorker(
        repo_path=args.repo_path,
        python_bin=args.python_bin,
        device=args.device,
        low_vram_mode=args.low_vram_mode,
        pretrained_model=args.pretrained_model,
        texture_resolution=args.texture_resolution,
        remesh_option=args.remesh_option,
        runtime_dir=args.runtime_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
