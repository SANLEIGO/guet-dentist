"""
标定参数持久化存储

保存和加载标定参数到JSON文件
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from dental_stitcher_v1.calibration.camera_estimator import CameraParameters


def save_calibration(
    camera_params: CameraParameters,
    output_path: Optional[str] = None,
    quality_score: float = 0.0,
    confidence: str = "unknown"
) -> str:
    """保存标定参数到JSON文件"""

    if output_path is None:
        # 默认路径
        output_path = get_default_calibration_path()

    # 准备数据
    calibration_data = {
        "fx": float(camera_params.fx),
        "fy": float(camera_params.fy),
        "cx": float(camera_params.cx),
        "cy": float(camera_params.cy),
        "k1": float(camera_params.k1),
        "k2": float(camera_params.k2),
        "p1": float(camera_params.p1),
        "p2": float(camera_params.p2),
        "k3": float(camera_params.k3),
        "image_width": camera_params.image_width,
        "image_height": camera_params.image_height,
        "calibration_date": datetime.now().isoformat(),
        "quality_score": quality_score,
        "confidence": confidence
    }

    # 写入文件
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(calibration_data, f, indent=2)

    return str(path)


def load_calibration(input_path: Optional[str] = None) -> Optional[CameraParameters]:
    """从JSON文件加载标定参数"""

    if input_path is None:
        input_path = get_default_calibration_path()

    path = Path(input_path)

    if not path.exists():
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        camera_params = CameraParameters(
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            k1=data["k1"],
            k2=data["k2"],
            p1=data["p1"],
            p2=data["p2"],
            k3=data["k3"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            rotation_vectors=[],  # 外参不保存（每张图像不同）
            translation_vectors=[],
            reprojection_error=0.0,
            confidence=data.get("confidence", "unknown")
        )

        return camera_params

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to load calibration: {e}")
        return None


def get_default_calibration_path() -> str:
    """获取默认标定参数文件路径"""
    # 存储在项目根目录的config文件夹
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    config_dir = project_root / "config"
    return str(config_dir / "calibration_cache.json")


def get_cache_key(image_shapes: list[tuple[int, int]], num_images: int) -> str:
    """生成缓存键（基于图像尺寸和数量）"""
    # 简化：使用图像尺寸hash
    size_hash = hash(tuple(image_shapes))
    return f"calib_{num_images}_{size_hash}"