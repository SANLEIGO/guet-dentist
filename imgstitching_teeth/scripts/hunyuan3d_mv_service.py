from __future__ import annotations

import argparse
import base64
import tempfile
import threading
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
    def __init__(self, model_path, subfolder, device, enable_flashvdm):
        self.device = device
        kwargs = {"subfolder": subfolder, "device": device}
        if device.startswith("cuda"):
            kwargs["variant"] = "fp16"
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, **kwargs)
        if enable_flashvdm and hasattr(self.pipeline, "enable_flashvdm"):
            self.pipeline.enable_flashvdm(mc_algo="mc")

    @torch.inference_mode()
    def generate(self, uid, payload):
        try:
            with _jobs_lock:
                _jobs[uid] = {"status": "queued"}
            with _generation_lock:
                with _jobs_lock:
                    _jobs[uid] = {"status": "processing"}
                image_dict = {tag: _load_image_from_base64(b64) for tag, b64 in payload["images"].items()}
                seed = int(payload.get("seed", 1234))
                steps = int(payload.get("num_inference_steps", 24))
                guidance = float(payload.get("guidance_scale", 5.0))
                octree = int(payload.get("octree_resolution", 256))
                mesh_type = str(payload.get("type", "glb"))
                gen_device = self.device if self.device.startswith("cuda") else "cpu"
                mesh = self.pipeline(image=image_dict, num_inference_steps=steps, guidance_scale=guidance, octree_resolution=octree, generator=torch.Generator(device=gen_device).manual_seed(seed), output_type="trimesh")[0]
                mesh = _keep_largest_mesh_component(mesh)
                save_path = _save_dir / f"{uid}.{mesh_type}"
                with tempfile.NamedTemporaryFile(suffix=f".{mesh_type}", delete=False) as temp_file:
                    mesh.export(temp_file.name)
                    mesh_loaded = trimesh.load(temp_file.name)
                    mesh_loaded = _keep_largest_mesh_component(mesh_loaded)
                    mesh_loaded.export(save_path)
                with _jobs_lock:
                    _jobs[uid] = {"status": "completed", "path": str(save_path)}
        except Exception as exc:
            traceback.print_exc()
            with _jobs_lock:
                _jobs[uid] = {"status": "error", "message": str(exc)}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

@app.get("/health")
async def health():
    with _jobs_lock:
        queued = sum(1 for item in _jobs.values() if item.get("status") == "queued")
        processing = sum(1 for item in _jobs.values() if item.get("status") == "processing")
    return {"status": "ok", "message": "Hunyuan3D multiview bridge is ready.", "jobs": len(_jobs), "queued_jobs": queued, "processing_jobs": processing}

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

def main():
    parser = argparse.ArgumentParser(description="Local Hunyuan3D-2mv bridge service for pseudo multiview inputs.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2mv")
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-mv-turbo")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable_flashvdm", action="store_true")
    args = parser.parse_args()
    global _worker
    _worker = MultiviewWorker(args.model_path, args.subfolder, args.device, args.enable_flashvdm)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
