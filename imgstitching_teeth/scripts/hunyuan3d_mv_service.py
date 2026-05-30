from __future__ import annotations

import argparse
import base64
import gc
import tempfile
import threading
import time
import traceback
import uuid
from io import BytesIO
from pathlib import Path

import torch
import trimesh
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_worker = None
_jobs = {}
_jobs_lock = threading.Lock()
_generation_lock = threading.Lock()
_save_dir = Path("gradio_cache")
_save_dir.mkdir(exist_ok=True)

class MultiviewWorker:
    def __init__(self, model_path, subfolder, device, enable_flashvdm, keepalive_seconds):
        self.model_path = model_path
        self.subfolder = subfolder
        self.device = device
        self.enable_flashvdm = enable_flashvdm
        self.keepalive_seconds = max(0.0, float(keepalive_seconds))
        self.pipeline = None
        self._pipeline_lock = threading.RLock()
        self._unload_timer = None

    def is_pipeline_loaded(self):
        with self._pipeline_lock:
            return self.pipeline is not None

    def _ensure_pipeline_loaded(self):
        self._cancel_unload_timer()
        with self._pipeline_lock:
            if self.pipeline is not None:
                return self.pipeline

            kwargs = {"subfolder": self.subfolder, "device": self.device}
            if self.device.startswith("cuda"):
                kwargs["variant"] = "fp16"

            print(
                f"[{_timestamp()}] Loading Hunyuan3D pipeline "
                f"({self.model_path}/{self.subfolder}) on {self.device}"
            )
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model_path, **kwargs)
            if self.enable_flashvdm and hasattr(pipeline, "enable_flashvdm"):
                pipeline.enable_flashvdm(mc_algo="mc")
            self.pipeline = pipeline
            return pipeline

    def _cancel_unload_timer(self):
        timer = None
        with self._pipeline_lock:
            timer = self._unload_timer
            self._unload_timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_unload(self):
        self._cancel_unload_timer()
        if not self.is_pipeline_loaded():
            return
        if self.keepalive_seconds <= 0:
            self.unload_pipeline(reason="job finished")
            return
        timer = threading.Timer(self.keepalive_seconds, self._unload_if_idle)
        timer.daemon = True
        with self._pipeline_lock:
            self._unload_timer = timer
        timer.start()

    def _unload_if_idle(self):
        if _generation_lock.locked():
            return
        self.unload_pipeline(reason=f"idle for {self.keepalive_seconds:g}s")

    def unload_pipeline(self, reason="manual"):
        self._cancel_unload_timer()
        with self._pipeline_lock:
            if self.pipeline is None:
                return
            print(f"[{_timestamp()}] Unloading Hunyuan3D pipeline ({reason})")
            self.pipeline = None
        gc.collect()
        _clear_device_cache(self.device)

    @torch.inference_mode()
    def generate(self, uid, payload):
        try:
            with _jobs_lock:
                _jobs[uid] = {"status": "queued"}
            with _generation_lock:
                try:
                    with _jobs_lock:
                        _jobs[uid] = {"status": "processing"}
                    image_dict = {tag: _load_image_from_base64(b64) for tag, b64 in payload["images"].items()}
                    seed = int(payload.get("seed", 1234))
                    steps = int(payload.get("num_inference_steps", 24))
                    guidance = float(payload.get("guidance_scale", 5.0))
                    octree = int(payload.get("octree_resolution", 256))
                    mesh_type = str(payload.get("type", "glb"))
                    gen_device = self.device if self.device.startswith("cuda") else "cpu"
                    pipeline = self._ensure_pipeline_loaded()
                    mesh = pipeline(image=image_dict, num_inference_steps=steps, guidance_scale=guidance, octree_resolution=octree, generator=torch.Generator(device=gen_device).manual_seed(seed), output_type="trimesh")[0]
                    mesh = _keep_largest_mesh_component(mesh)
                    save_path = _save_dir / f"{uid}.{mesh_type}"
                    with tempfile.NamedTemporaryFile(suffix=f".{mesh_type}", delete=False) as temp_file:
                        mesh.export(temp_file.name)
                        mesh_loaded = trimesh.load(temp_file.name)
                        mesh_loaded = _keep_largest_mesh_component(mesh_loaded)
                        mesh_loaded.export(save_path)
                    with _jobs_lock:
                        _jobs[uid] = {"status": "completed", "path": str(save_path)}
                finally:
                    self._schedule_unload()
                    _clear_device_cache(self.device)
        except Exception as exc:
            traceback.print_exc()
            message = _format_runtime_error(exc)
            with _jobs_lock:
                _jobs[uid] = {"status": "error", "message": message}

@app.get("/health")
async def health():
    with _jobs_lock:
        queued = sum(1 for item in _jobs.values() if item.get("status") == "queued")
        processing = sum(1 for item in _jobs.values() if item.get("status") == "processing")
    return {
        "status": "ok",
        "message": "Hunyuan3D multiview bridge is ready.",
        "jobs": len(_jobs),
        "queued_jobs": queued,
        "processing_jobs": processing,
        "model_loaded": _worker.is_pipeline_loaded() if _worker else False,
        "keepalive_seconds": _worker.keepalive_seconds if _worker else 0,
    }

@app.post("/send_multiview")
async def send_multiview(request: Request):
    payload = await request.json()
    images = payload.get("images") or {}
    if "front" not in images:
        return JSONResponse({"error": "front view is required"}, status_code=400)
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

def _load_image_from_base64(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGBA")

def _keep_largest_mesh_component(mesh):
    if isinstance(mesh, trimesh.Scene):
        geometries = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geometries:
            raise ValueError("Generated mesh scene does not contain any mesh geometry.")
        mesh = trimesh.util.concatenate(geometries)
    components = mesh.split(only_watertight=False)
    if not components:
        return mesh
    largest = max(components, key=lambda item: len(item.faces))
    largest.remove_unreferenced_vertices()
    return largest


def _timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _format_runtime_error(exc):
    text = str(exc)
    if "incomplete metadata" in text or "file not fully covered" in text:
        return (
            "本地 Hunyuan3D 模型文件不完整或已损坏，请在启动器里重新执行“仅下载模型”后重试。"
        )
    return text


def _clear_device_cache(device):
    try:
        device_type = torch.device(device).type
    except (RuntimeError, TypeError, ValueError):
        device_type = str(device)

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return

    if device_type == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "empty_cache"):
            try:
                mps_module.empty_cache()
            except RuntimeError:
                pass

def main():
    parser = argparse.ArgumentParser(description="Local Hunyuan3D-2mv bridge service for pseudo multiview inputs.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2mv")
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-mv-turbo")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable_flashvdm", action="store_true")
    parser.add_argument("--keepalive-seconds", type=float, default=0.0)
    args = parser.parse_args()
    global _worker
    _worker = MultiviewWorker(
        args.model_path,
        args.subfolder,
        args.device,
        args.enable_flashvdm,
        args.keepalive_seconds,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
