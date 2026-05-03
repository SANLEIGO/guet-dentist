from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .hunyuan_client import HunyuanServiceConfig, HunyuanServiceClient


ROOT_DIR = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT_DIR / ".runtime" / "hunyuan3d"
LOG_DIR = RUNTIME_DIR / "logs"
REPO_DIR = ROOT_DIR / "third_party" / "Hunyuan3D-2"
MODELS_DIR = RUNTIME_DIR / "models"
DEFAULT_MODEL_DIR = MODELS_DIR / "Hunyuan3D-2mv"
TASK_LOG_PATH = LOG_DIR / "task.log"
SERVICE_LOG_PATH = LOG_DIR / "service.log"
TASK_PID_PATH = RUNTIME_DIR / "task.pid"
SERVICE_PID_PATH = RUNTIME_DIR / "service.pid"
INSTALL_STAMP_PATH = RUNTIME_DIR / "install.ok"
RUNTIME_SCRIPT_PATH = ROOT_DIR / "scripts" / "hunyuan3d_runtime.py"


@dataclass
class HunyuanRuntimeStatus:
    repo_exists: bool
    install_ready: bool
    model_ready: bool
    task_running: bool
    task_pid: Optional[int]
    service_running: bool
    service_pid: Optional[int]
    service_healthy: bool
    service_message: str
    service_url: str
    repo_dir: str
    model_dir: str
    task_log_tail: str
    service_log_tail: str


class HunyuanRuntimeManager:
    def __init__(self) -> None:
        self.service_config = HunyuanServiceConfig.from_env()
        self._ensure_dirs()

    def read_status(self) -> HunyuanRuntimeStatus:
        task_pid = _read_pid(TASK_PID_PATH)
        service_pid = _read_pid(SERVICE_PID_PATH)
        task_running = _is_pid_alive(task_pid)
        service_running = _is_pid_alive(service_pid)

        if not task_running and TASK_PID_PATH.exists():
            TASK_PID_PATH.unlink(missing_ok=True)
        if not service_running and SERVICE_PID_PATH.exists():
            SERVICE_PID_PATH.unlink(missing_ok=True)

        probe = HunyuanServiceClient(self.service_config).probe()
        return HunyuanRuntimeStatus(
            repo_exists=REPO_DIR.exists(),
            install_ready=INSTALL_STAMP_PATH.exists(),
            model_ready=_model_weights_exist(DEFAULT_MODEL_DIR),
            task_running=task_running,
            task_pid=task_pid if task_running else None,
            service_running=service_running,
            service_pid=service_pid if service_running else None,
            service_healthy=probe.reachable,
            service_message=probe.message,
            service_url=self.service_config.service_url,
            repo_dir=str(REPO_DIR),
            model_dir=str(DEFAULT_MODEL_DIR),
            task_log_tail=_tail_text(TASK_LOG_PATH, max_chars=6000),
            service_log_tail=_tail_text(SERVICE_LOG_PATH, max_chars=6000),
        )

    def launch_action(self, action: str) -> tuple[bool, str]:
        if action in {"setup", "download-model", "bootstrap-and-start"} and _is_pid_alive(_read_pid(TASK_PID_PATH)):
            return False, "已有后台任务在运行，请等待当前任务结束。"

        self._ensure_dirs()
        command = [sys.executable, str(RUNTIME_SCRIPT_PATH), action]
        with TASK_LOG_PATH.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
        TASK_PID_PATH.write_text(str(process.pid), encoding="utf-8")
        return True, f"已启动后台任务：{action}（PID {process.pid}）"

    def stop_service(self) -> tuple[bool, str]:
        pid = _read_pid(SERVICE_PID_PATH)
        if not _is_pid_alive(pid):
            SERVICE_PID_PATH.unlink(missing_ok=True)
            return False, "当前没有运行中的 Hunyuan3D 服务。"

        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            os.kill(pid, signal.SIGTERM)
        SERVICE_PID_PATH.unlink(missing_ok=True)
        return True, f"已请求停止 Hunyuan3D 服务（PID {pid}）。"

    def clear_task_log(self) -> None:
        self._ensure_dirs()
        TASK_LOG_PATH.write_text("", encoding="utf-8")

    def _ensure_dirs(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _read_pid(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (TypeError, ValueError):
        return None


def _is_pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if len(content) <= max_chars:
        return content
    return content[-max_chars:]


def _model_weights_exist(model_root: Path) -> bool:
    candidate_files = [
        model_root / "hunyuan3d-dit-v2-mv-turbo" / "model.fp16.safetensors",
        model_root / "hunyuan3d-dit-v2-mv-turbo" / "model.fp16.ckpt",
        model_root / "hunyuan3d-dit-v2-mv-fast" / "model.fp16.safetensors",
        model_root / "hunyuan3d-dit-v2-mv-fast" / "model.fp16.ckpt",
        model_root / "hunyuan3d-dit-v2-mv" / "model.safetensors",
        model_root / "hunyuan3d-dit-v2-mv" / "model.ckpt",
    ]
    return any(path.exists() for path in candidate_files)
