from __future__ import annotations

import base64
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import dotenv_values


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_ENV_CONFIG = dotenv_values(ENV_PATH)


@dataclass
class HunyuanServiceConfig:
    service_url: str
    model_path: str
    subfolder: str
    device: str
    service_mode: str = "single_image"
    request_timeout_sec: float = 20.0
    poll_interval_sec: float = 3.0

    @classmethod
    def from_env(cls) -> "HunyuanServiceConfig":
        return cls(
            service_url=str(_ENV_CONFIG.get("HUNYUAN3D_SERVICE_URL", "http://127.0.0.1:8081")).rstrip("/"),
            model_path=str(_ENV_CONFIG.get("HUNYUAN3D_MODEL_PATH", "tencent/Hunyuan3D-2.1")),
            subfolder=str(_ENV_CONFIG.get("HUNYUAN3D_SUBFOLDER", "hunyuan3d-dit-v2-1")),
            device=str(_ENV_CONFIG.get("HUNYUAN3D_DEVICE", "auto")),
            service_mode=str(_ENV_CONFIG.get("HUNYUAN3D_SERVICE_MODE", "single_image")),
            request_timeout_sec=float(_ENV_CONFIG.get("HUNYUAN3D_REQUEST_TIMEOUT_SEC", 20.0)),
            poll_interval_sec=float(_ENV_CONFIG.get("HUNYUAN3D_POLL_INTERVAL_SEC", 3.0)),
        )


@dataclass
class HunyuanProbeResult:
    reachable: bool
    status: str
    message: str
    url: str


@dataclass
class HunyuanJobStatus:
    uid: str
    status: str
    model_bytes: Optional[bytes] = None
    message: Optional[str] = None


class HunyuanServiceClient:
    def __init__(self, config: Optional[HunyuanServiceConfig] = None):
        self.config = config or HunyuanServiceConfig.from_env()

    def probe(self) -> HunyuanProbeResult:
        try:
            data = self._request_json(f"{self.config.service_url}/health", timeout=1.5)
            return HunyuanProbeResult(
                reachable=True,
                status=str(data.get("status", "ok")),
                message=str(data.get("message", "Service responded successfully.")),
                url=self.config.service_url,
            )
        except Exception:
            pass

        try:
            data = self._request_json(f"{self.config.service_url}/status/codex-probe", timeout=1.5)
        except urllib.error.HTTPError as exc:
            return HunyuanProbeResult(
                reachable=False,
                status="http_error",
                message=f"HTTP {exc.code}: {exc.reason}",
                url=self.config.service_url,
            )
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, socket.timeout):
                message = "Connection timed out."
            else:
                message = str(reason)
            return HunyuanProbeResult(
                reachable=False,
                status="unreachable",
                message=message,
                url=self.config.service_url,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return HunyuanProbeResult(
                reachable=False,
                status="error",
                message=str(exc),
                url=self.config.service_url,
            )

        status = str(data.get("status", "unknown"))
        return HunyuanProbeResult(
            reachable=True,
            status=status,
            message="Service responded successfully.",
            url=self.config.service_url,
        )

    def submit_image_async(
        self,
        image: np.ndarray,
        *,
        seed: int = 1234,
        num_inference_steps: int = 5,
        guidance_scale: float = 5.0,
        octree_resolution: int = 128,
        texture: bool = False,
    ) -> str:
        image_b64 = _encode_image_to_base64(image)
        payload = {
            "image": image_b64,
            "seed": int(seed),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "octree_resolution": int(octree_resolution),
            "texture": bool(texture),
        }
        response = self._request_json(
            f"{self.config.service_url}/send",
            payload=payload,
            timeout=self.config.request_timeout_sec,
        )
        uid = response.get("uid")
        if not uid:
            raise ValueError("Hunyuan service did not return a uid.")
        return str(uid)

    def submit_multiview_async(
        self,
        images: dict[str, np.ndarray],
        *,
        seed: int = 1234,
        num_inference_steps: int = 5,
        guidance_scale: float = 5.0,
        octree_resolution: int = 128,
        texture: bool = False,
        mesh_type: str = "glb",
    ) -> str:
        payload = {
            "images": {tag: _encode_image_to_base64(image) for tag, image in images.items()},
            "seed": int(seed),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "octree_resolution": int(octree_resolution),
            "texture": bool(texture),
            "type": mesh_type,
        }
        endpoint = (
            f"{self.config.service_url}/send_multiview"
            if self.config.service_mode == "mv_bridge"
            else f"{self.config.service_url}/send"
        )
        response = self._request_json(
            endpoint,
            payload=payload,
            timeout=self.config.request_timeout_sec,
        )
        uid = response.get("uid")
        if not uid:
            raise ValueError("Hunyuan multiview service did not return a uid.")
        return str(uid)

    def get_job_status(self, uid: str) -> HunyuanJobStatus:
        response = self._request_json(
            f"{self.config.service_url}/status/{uid}",
            timeout=self.config.request_timeout_sec,
        )
        status = str(response.get("status", "unknown"))
        model_b64 = response.get("model_base64")
        model_bytes = base64.b64decode(model_b64) if model_b64 else None
        return HunyuanJobStatus(
            uid=uid,
            status=status,
            model_bytes=model_bytes,
            message=response.get("message"),
        )

    def _request_json(
        self,
        url: str,
        payload: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read().decode("utf-8")
        return json.loads(content)


def _encode_image_to_base64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image for Hunyuan service request.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")
