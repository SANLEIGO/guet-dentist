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
    model_status: str
    model_message: str
    task_running: bool
    task_pid: Optional[int]
    task_message: str
    service_running: bool
    service_pid: Optional[int]
    service_healthy: bool
    service_message: str
    recommended_action: Optional[str]
    recommended_label: str
    user_hint: str
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

        task_log_tail = _preview_text(TASK_LOG_PATH, head_chars=2200, tail_chars=3800)
        service_log_tail = _tail_text(SERVICE_LOG_PATH, max_chars=6000)
        model_status, model_message = _inspect_model_state(DEFAULT_MODEL_DIR)
        if task_running and _task_log_indicates_model_download(task_log_tail):
            model_status = "downloading"
            model_message = "正在下载或修复模型文件，请耐心等待。"
        probe = HunyuanServiceClient(self.service_config).probe()
        recommended_action, recommended_label, user_hint = _build_user_guidance(
            repo_exists=REPO_DIR.exists(),
            install_ready=INSTALL_STAMP_PATH.exists(),
            model_status=model_status,
            task_running=task_running,
            task_log_tail=task_log_tail,
            service_running=service_running,
            service_healthy=probe.reachable,
            service_message=probe.message,
        )
        return HunyuanRuntimeStatus(
            repo_exists=REPO_DIR.exists(),
            install_ready=INSTALL_STAMP_PATH.exists(),
            model_ready=(model_status == "ready"),
            model_status=model_status,
            model_message=model_message,
            task_running=task_running,
            task_pid=task_pid if task_running else None,
            task_message=_build_running_task_hint(task_log_tail) if task_running else "",
            service_running=service_running,
            service_pid=service_pid if service_running else None,
            service_healthy=probe.reachable,
            service_message=probe.message,
            recommended_action=recommended_action,
            recommended_label=recommended_label,
            user_hint=user_hint,
            service_url=self.service_config.service_url,
            repo_dir=str(REPO_DIR),
            model_dir=str(DEFAULT_MODEL_DIR),
            task_log_tail=task_log_tail,
            service_log_tail=service_log_tail,
        )

    def launch_action(self, action: str) -> tuple[bool, str]:
        if action in {"setup", "download-model", "bootstrap-and-start"} and _is_pid_alive(_read_pid(TASK_PID_PATH)):
            return False, "已有后台任务在运行，请等待当前任务结束。"

        if action == "restart-service":
            return self.restart_service()

        self._ensure_dirs()
        command = [sys.executable, str(RUNTIME_SCRIPT_PATH), action]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with TASK_LOG_PATH.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                env=env,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
        TASK_PID_PATH.write_text(str(process.pid), encoding="utf-8")
        return True, _launch_message(action, process.pid)

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

    def restart_service(self) -> tuple[bool, str]:
        self.stop_service()
        return self.launch_action("start-service")

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


def _preview_text(path: Path, head_chars: int = 2000, tail_chars: int = 3000) -> str:
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if len(content) <= head_chars + tail_chars + 64:
        return content
    return content[:head_chars] + "\n...\n" + content[-tail_chars:]


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


def _inspect_model_state(model_root: Path) -> tuple[str, str]:
    candidate_files = [
        model_root / "hunyuan3d-dit-v2-mv-turbo" / "model.fp16.safetensors",
        model_root / "hunyuan3d-dit-v2-mv-turbo" / "model.fp16.ckpt",
        model_root / "hunyuan3d-dit-v2-mv-fast" / "model.fp16.safetensors",
        model_root / "hunyuan3d-dit-v2-mv-fast" / "model.fp16.ckpt",
        model_root / "hunyuan3d-dit-v2-mv" / "model.safetensors",
        model_root / "hunyuan3d-dit-v2-mv" / "model.ckpt",
    ]

    for path in candidate_files:
        if not path.exists():
            continue
        if path.suffix == ".safetensors" and not _is_safetensors_file_valid(path):
            return "invalid", "模型文件不完整或已损坏，请重新下载模型。"
        return "ready", f"模型已就绪：{path.name}（{_format_size(path.stat().st_size)}）"

    return "missing", "还没有下载 3D 模型文件。"


def _is_safetensors_file_valid(path: Path) -> bool:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            handle.keys()
        return True
    except Exception:
        return False


def _format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0


def _launch_message(action: str, pid: int) -> str:
    mapping = {
        "bootstrap-and-start": f"已开始自动准备 Hunyuan3D（PID {pid}）。首次使用会比较久，完成后服务会自动启动。",
        "setup": f"已开始初始化运行环境（PID {pid}）。完成后再下载模型即可。",
        "download-model": f"已开始下载或修复模型文件（PID {pid}）。下载完成后就可以启动服务。",
        "start-service": f"已开始启动 Hunyuan3D 服务（PID {pid}）。几秒后刷新状态即可。",
    }
    return mapping.get(action, f"已启动后台任务：{action}（PID {pid}）")


def _summarize_task_message(task_log_tail: str) -> str:
    lines = [line.strip() for line in task_log_tail.splitlines() if line.strip()]
    for line in reversed(lines):
        if _is_noisy_progress_line(line):
            continue
        if line.startswith("[") and "]" in line:
            return line.split("]", 1)[1].strip()
        if line.startswith("$"):
            continue
        return line
    return "后台任务进行中，请稍等。"


def _build_running_task_hint(task_log_tail: str) -> str:
    if _task_log_indicates_model_download(task_log_tail):
        return "正在下载或修复模型文件，模型较大，请耐心等待。"
    if "Step 2/3: 安装 Hunyuan3D 兼容运行依赖" in task_log_tail:
        return "正在安装 Hunyuan3D 运行环境，请稍等。"
    if "准备 Hunyuan3D 代码仓库" in task_log_tail or "Step 1/3" in task_log_tail:
        return "正在准备 Hunyuan3D 代码和运行环境。"
    if "准备启动本地 Hunyuan3D bridge 服务" in task_log_tail or "服务启动命令已提交" in task_log_tail:
        return "正在启动 Hunyuan3D 服务。"
    return _summarize_task_message(task_log_tail)


def _task_log_indicates_model_download(task_log_tail: str) -> bool:
    return (
        "开始下载 Hunyuan3D-2mv 模型" in task_log_tail
        or "下载 model.fp16.safetensors" in task_log_tail
        or "Resuming transfer" in task_log_tail
        or "% Total" in task_log_tail
        or "Dload  Upload" in task_log_tail
    )


def _is_noisy_progress_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("% Total") or stripped.startswith("** Resuming transfer"):
        return True
    if stripped[0].isdigit() and "Time" in stripped and "Current" in stripped:
        return True
    if stripped[0].isdigit() and "M" in stripped and ":" in stripped and "k" in stripped:
        return True
    return False


def _build_user_guidance(
    *,
    repo_exists: bool,
    install_ready: bool,
    model_status: str,
    task_running: bool,
    task_log_tail: str,
    service_running: bool,
    service_healthy: bool,
    service_message: str,
) -> tuple[Optional[str], str, str]:
    if task_running:
        return None, "后台任务进行中", _build_running_task_hint(task_log_tail)

    if service_healthy:
        return None, "服务已就绪", "Hunyuan3D 服务已经准备好，可以直接提交 3D 任务。"

    if not repo_exists or not install_ready:
        return (
            "bootstrap-and-start",
            "一键准备并启动",
            "第一次使用时，点这一个按钮就够了。系统会自动安装环境、下载模型并启动服务。",
        )

    if model_status == "missing":
        return "download-model", "下载模型", "运行环境已经就绪，下一步只需要下载 3D 模型文件。"

    if model_status == "invalid":
        return "download-model", "重新下载模型", "检测到模型文件不完整，请重新下载模型后再启动服务。"

    if service_running and not service_healthy:
        return "restart-service", "重启服务", f"服务进程已在运行，但当前未就绪：{service_message}"

    return "start-service", "启动服务", "模型已经准备好，启动服务后就可以提交 3D 任务。"
