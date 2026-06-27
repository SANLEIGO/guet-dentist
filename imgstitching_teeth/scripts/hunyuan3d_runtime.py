from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from typing import Optional

from dotenv import dotenv_values


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
ENV_CONFIG = dotenv_values(ENV_PATH)

RUNTIME_DIR = ROOT_DIR / ".runtime" / "hunyuan3d"
LOG_DIR = RUNTIME_DIR / "logs"
REPO_DIR = ROOT_DIR / "third_party" / "Hunyuan3D-2"
MODELS_DIR = RUNTIME_DIR / "models"
MODEL_DIR = MODELS_DIR / "Hunyuan3D-2.1"
INSTALL_STAMP_PATH = RUNTIME_DIR / "install.ok"
SERVICE_PID_PATH = RUNTIME_DIR / "service.pid"
SERVICE_LOG_PATH = LOG_DIR / "service.log"
SERVICE_SCRIPT_PATH = ROOT_DIR / "scripts" / "hunyuan3d_mv_service.py"
MINIMAL_RUNTIME_REQUIREMENTS = [
    "ninja",
    "pybind11",
    "diffusers",
    "einops",
    "transformers",
    "omegaconf",
    "tqdm",
    "trimesh",
    "accelerate",
    "fastapi",
    "uvicorn",
    "huggingface_hub>=0.30",
    "pyyaml",
    "scikit-image",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Background runtime actions for local Hunyuan3D setup.")
    parser.add_argument(
        "action",
        choices=["setup", "download-model", "start-service", "bootstrap-and-start", "stop-service"],
    )
    args = parser.parse_args()

    _ensure_dirs()

    if args.action == "setup":
        setup_runtime()
    elif args.action == "download-model":
        download_model()
    elif args.action == "start-service":
        start_service()
    elif args.action == "bootstrap-and-start":
        setup_runtime()
        download_model()
        start_service()
    elif args.action == "stop-service":
        stop_service()


def setup_runtime() -> None:
    print(f"[{_now()}] Step 1/3: 准备 Hunyuan3D 代码仓库")
    if not REPO_DIR.exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
                str(REPO_DIR),
            ]
        )
    else:
        print(f"[{_now()}] 代码仓库已存在：{REPO_DIR}")
    _patch_shapegen_init_for_optional_postprocessors()

    print(f"[{_now()}] Step 2/3: 安装 Hunyuan3D 兼容运行依赖到当前 Python 环境")
    print(f"[{_now()}] 当前模式: Hunyuan3D-2.1 单图 shape-only bridge（跳过纹理和重型可选组件）")
    python = sys.executable
    _run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    _run([python, "-m", "pip", "install", *MINIMAL_RUNTIME_REQUIREMENTS])
    _run([python, "-m", "pip", "install", "-e", str(REPO_DIR), "--no-deps"])

    INSTALL_STAMP_PATH.write_text(
        (
            f"installed_at={datetime.now().isoformat()}\n"
            f"python={python}\n"
            "profile=shape_only_single_image_bridge\n"
        ),
        encoding="utf-8",
    )
    print(f"[{_now()}] Step 3/3: 依赖安装完成")


def download_model() -> None:
    subfolder = str(ENV_CONFIG.get("HUNYUAN3D_SUBFOLDER", "hunyuan3d-dit-v2-1"))
    print(f"[{_now()}] 开始下载 Hunyuan3D-2.1 模型到 {MODEL_DIR}")
    print(f"[{_now()}] 目标子目录: {subfolder}")
    _cleanup_partial_model_downloads(subfolder)
    target_filename = "model.fp16.ckpt" if subfolder == "hunyuan3d-dit-v2-1" else (
        "model.fp16.safetensors" if "turbo" in subfolder or "fast" in subfolder else "model.safetensors"
    )
    target_dir = MODEL_DIR / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        (f"https://huggingface.co/tencent/Hunyuan3D-2.1/resolve/main/{subfolder}/config.yaml", target_dir / "config.yaml"),
        (f"https://huggingface.co/tencent/Hunyuan3D-2.1/resolve/main/{subfolder}/{target_filename}", target_dir / target_filename),
        ("https://huggingface.co/tencent/Hunyuan3D-2.1/resolve/main/README.md", MODEL_DIR / "README.md"),
        ("https://huggingface.co/tencent/Hunyuan3D-2.1/resolve/main/LICENSE", MODEL_DIR / "LICENSE"),
        ("https://huggingface.co/tencent/Hunyuan3D-2.1/resolve/main/Notice.txt", MODEL_DIR / "Notice.txt"),
    ]

    for url, path in files_to_download:
        path.parent.mkdir(parents=True, exist_ok=True)
        _download_file_with_validation(url, path)
    print(f"[{_now()}] 模型下载完成")


def start_service() -> None:
    print(f"[{_now()}] 准备启动本地 Hunyuan3D 单图 bridge 服务")
    existing_pid = _read_pid(SERVICE_PID_PATH)
    if _is_pid_alive(existing_pid):
        print(f"[{_now()}] 服务已在运行，PID={existing_pid}")
        return

    python = sys.executable
    host = str(ENV_CONFIG.get("HUNYUAN3D_SERVICE_HOST", "127.0.0.1"))
    port = str(ENV_CONFIG.get("HUNYUAN3D_SERVICE_PORT", "8081"))
    device = _resolve_device(str(ENV_CONFIG.get("HUNYUAN3D_DEVICE", "auto")))
    subfolder = str(ENV_CONFIG.get("HUNYUAN3D_SUBFOLDER", "hunyuan3d-dit-v2-1"))
    keepalive_seconds = _resolve_keepalive_seconds(device)
    model_path = str(MODEL_DIR if MODEL_DIR.exists() else ENV_CONFIG.get("HUNYUAN3D_MODEL_PATH", "tencent/Hunyuan3D-2.1"))
    if subfolder == "hunyuan3d-dit-v2-1":
        model_file = MODEL_DIR / subfolder / "model.fp16.ckpt"
        model_ckpt = model_file
    else:
        model_file = MODEL_DIR / subfolder / ("model.fp16.safetensors" if "turbo" in subfolder or "fast" in subfolder else "model.safetensors")
        model_ckpt = MODEL_DIR / subfolder / ("model.fp16.ckpt" if "turbo" in subfolder or "fast" in subfolder else "model.ckpt")
    if not model_file.exists() and not model_ckpt.exists():
        raise FileNotFoundError(
            f"模型权重未找到：{model_file}。请先执行 download-model 或一键下载并启动。"
        )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HOME"] = str(RUNTIME_DIR / "hf_home")

    command = [
        python,
        str(SERVICE_SCRIPT_PATH),
        "--host",
        host,
        "--port",
        port,
        "--model_path",
        model_path,
        "--subfolder",
        subfolder,
        "--device",
        device,
        "--keepalive-seconds",
        str(keepalive_seconds),
    ]
    if _should_enable_flashvdm(subfolder, device):
        command.append("--enable_flashvdm")

    with SERVICE_LOG_PATH.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=env,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )

    SERVICE_PID_PATH.write_text(str(process.pid), encoding="utf-8")
    print(f"[{_now()}] 服务启动命令已提交，PID={process.pid}")
    print(f"[{_now()}] 服务地址预期为 http://{host}:{port}")
    print(f"[{_now()}] 当前推理设备: {device}")
    print(f"[{_now()}] 模型空闲保留时长: {keepalive_seconds:g} 秒")


def stop_service() -> None:
    pid = _read_pid(SERVICE_PID_PATH)
    if not _is_pid_alive(pid):
        print(f"[{_now()}] 当前没有运行中的 Hunyuan3D 服务")
        SERVICE_PID_PATH.unlink(missing_ok=True)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        os.kill(pid, signal.SIGTERM)
    SERVICE_PID_PATH.unlink(missing_ok=True)
    print(f"[{_now()}] 已请求停止服务，PID={pid}")


def _run(command: list[str]) -> None:
    print(f"[{_now()}] $ {' '.join(command)}")
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def _cleanup_partial_model_downloads(target_subfolder: str) -> None:
    model_root = MODEL_DIR
    if not model_root.exists():
        return

    removed_count = 0
    download_root = model_root / ".cache" / "huggingface" / "download"
    if download_root.exists():
        for path in download_root.rglob("*.incomplete"):
            path.unlink(missing_ok=True)
            removed_count += 1
        for lock_file in download_root.rglob("*.lock"):
            if target_subfolder not in str(lock_file):
                lock_file.unlink(missing_ok=True)

    for child in model_root.iterdir():
        if child.is_dir() and child.name.startswith("hunyuan3d-dit-v2-") and child.name != target_subfolder:
            shutil.rmtree(child, ignore_errors=True)
            removed_count += 1

    if removed_count:
        print(f"[{_now()}] 已清理旧的模型残留/未完成下载，共处理 {removed_count} 项")


def _download_file_with_validation(url: str, path: Path) -> None:
    expected_size = _get_remote_file_size(url)
    if path.exists() and expected_size is not None:
        local_size = path.stat().st_size
        if local_size == expected_size and _is_model_file_valid(path):
            print(f"[{_now()}] 已复用完整文件 {path.name} ({_format_size(local_size)})")
            return
        if local_size > expected_size:
            print(f"[{_now()}] 检测到本地文件大于远端，删除后重下：{path.name}")
            path.unlink(missing_ok=True)

    print(f"[{_now()}] 下载 {path.name}")
    if expected_size is not None:
        print(f"[{_now()}] 远端大小: {_format_size(expected_size)}")

    _run(
        [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "-C",
            "-",
            "-o",
            str(path),
            url,
        ]
    )

    if expected_size is not None:
        actual_size = path.stat().st_size if path.exists() else 0
        if actual_size != expected_size:
            raise RuntimeError(
                f"下载后的文件大小不匹配：{path.name} 本地 {_format_size(actual_size)}，远端 {_format_size(expected_size)}"
            )

    if not _is_model_file_valid(path):
        raise RuntimeError(f"下载后的模型文件校验失败：{path}")
    print(f"[{_now()}] 已完成 {path.name}（{_format_size(path.stat().st_size)}）")


def _get_remote_file_size(url: str) -> Optional[int]:
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=30) as response:
        content_length = response.headers.get("Content-Length")
    if not content_length:
        return None
    return int(content_length)


def _is_model_file_valid(path: Path) -> bool:
    if not path.exists() or path.suffix != ".safetensors":
        return True
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


def _resolve_device(configured_device: str) -> str:
    if configured_device and configured_device.lower() != "auto":
        return configured_device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_keepalive_seconds(device: str) -> float:
    configured = ENV_CONFIG.get("HUNYUAN3D_KEEPALIVE_SEC")
    if configured not in (None, ""):
        return float(configured)
    if device == "mps":
        return 0.0
    return 900.0


def _should_enable_flashvdm(subfolder: str, device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    return subfolder != "hunyuan3d-dit-v2-1"


def _patch_shapegen_init_for_optional_postprocessors() -> None:
    init_path = REPO_DIR / "hy3dgen" / "shapegen" / "__init__.py"
    if not init_path.exists():
        return

    content = init_path.read_text(encoding="utf-8")
    fixed_content = (
        "# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT\n"
        "# except for the third-party components listed below.\n"
        "# Hunyuan 3D does not impose any additional limitations beyond what is outlined\n"
        "# in the repsective licenses of these third-party components.\n"
        "# Users must comply with all terms and conditions of original licenses of these third-party\n"
        "# components and must ensure that the usage of the third party components adheres to\n"
        "# all relevant laws and regulations.\n\n"
        "# For avoidance of doubts, Hunyuan 3D means the large language models and\n"
        "# their software and algorithms, including trained model weights, parameters (including\n"
        "# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,\n"
        "# fine-tuning enabling code and other elements of the foregoing made publicly available\n"
        "# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.\n\n"
        "from .pipelines import Hunyuan3DDiTPipeline, Hunyuan3DDiTFlowMatchingPipeline\n"
        "try:\n"
        "    from .postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier\n"
        "except Exception:\n"
        "    FaceReducer = None\n"
        "    FloaterRemover = None\n"
        "    DegenerateFaceRemover = None\n"
        "    MeshSimplifier = None\n"
        "from .preprocessors import ImageProcessorV2, IMAGE_PROCESSORS, DEFAULT_IMAGEPROCESSOR\n"
    )
    if content != fixed_content:
        init_path.write_text(fixed_content, encoding="utf-8")
        print(f"[{_now()}] 已应用 shapegen optional-postprocessor 兼容补丁")


def _ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    (RUNTIME_DIR / "hf_home").mkdir(parents=True, exist_ok=True)


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _is_pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


if __name__ == "__main__":
    main()
