from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dental_stitcher.models import ImageRecord, MatchResult


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image_records(paths: list[str], arch: str, segment: str) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        records.append(ImageRecord(path=path, arch=arch, segment=segment, image=image))
    return records


def load_uploaded_records(files: list, arch: str, segment: str) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for uploaded in files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            continue
        records.append(ImageRecord(path=Path(uploaded.name), arch=arch, segment=segment, image=image))
    return records


def segment_display_name(segment: str) -> str:
    mapping = {
        "left": "左侧段",
        "right": "右侧段",
        "full": "完整牙弓",
    }
    return mapping.get(segment, segment)


def arch_display_name(arch: str) -> str:
    mapping = {
        "upper": "上牙",
        "lower": "下牙",
    }
    return mapping.get(arch, arch)


def segment_guidance(arch: str, segment: str) -> dict[str, str]:
    if segment == "full":
        if arch == "upper":
            return {
                "region": "上牙完整牙弓",
                "anchor_hint": "优先让中切牙到尖牙附近的中央视角作为基准候选。",
                "order_hint": "建议按左后牙 -> 左前牙 -> 中前牙 -> 右前牙 -> 右后牙，或反向连续采集。",
            }
        return {
            "region": "下牙完整牙弓",
            "anchor_hint": "优先让下前牙到尖牙附近的中央视角作为基准候选。",
            "order_hint": "建议按左后牙 -> 左前牙 -> 中前牙 -> 右前牙 -> 右后牙，或反向连续采集，并尽量降低舌体遮挡。",
        }
    if arch == "upper" and segment == "left":
        return {
            "region": "上牙左侧段",
            "anchor_hint": "优先包含上前牙偏左到左尖牙附近的居中视角。",
            "order_hint": "建议从前牙偏左开始，逐步移动到左尖牙和左前磨牙，保持连续重叠。",
        }
    if arch == "upper" and segment == "right":
        return {
            "region": "上牙右侧段",
            "anchor_hint": "优先包含上前牙偏右到右尖牙附近的居中视角。",
            "order_hint": "建议从前牙偏右开始，逐步移动到右尖牙和右前磨牙，保持连续重叠。",
        }
    if arch == "lower" and segment == "left":
        return {
            "region": "下牙左侧段",
            "anchor_hint": "优先包含下前牙偏左到左尖牙附近、无遮挡较少的稳定视角。",
            "order_hint": "建议从下前牙偏左开始，逐步移动到左尖牙和左前磨牙，避免舌体遮挡。",
        }
    return {
        "region": "下牙右侧段",
        "anchor_hint": "优先包含下前牙偏右到右尖牙附近、无遮挡较少的稳定视角。",
        "order_hint": "建议从下前牙偏右开始，逐步移动到右尖牙和右前磨牙，避免舌体遮挡。",
    }


def resize_for_display(image: np.ndarray, max_width: int = 900, max_height: int = 700) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compute_quality_score(image: np.ndarray) -> float:
    sharpness, exposure = compute_image_metrics(image)
    return combine_quality_score(sharpness, exposure)


def compute_image_metrics(image: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_32F).var()
    mean_intensity = float(gray.mean())
    exposure_score = 1.0 - min(abs(mean_intensity - 127.5) / 127.5, 1.0)
    return float(sharpness), float(exposure_score)


def combine_quality_score(sharpness: float, exposure_score: float) -> float:
    return 0.65 * np.log1p(sharpness) + 0.35 * exposure_score


def render_match_visualization(
    src: np.ndarray,
    dst: np.ndarray,
    match: MatchResult,
    max_points: int = 50,
) -> np.ndarray:
    src_rgb = bgr_to_rgb(src)
    dst_rgb = bgr_to_rgb(dst)

    src_h, src_w = src_rgb.shape[:2]
    dst_h, dst_w = dst_rgb.shape[:2]
    canvas_h = max(src_h, dst_h)
    canvas_w = src_w + dst_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:src_h, :src_w] = src_rgb
    canvas[:dst_h, src_w : src_w + dst_w] = dst_rgb

    pts0 = match.matched_points0
    pts1 = match.matched_points1
    if pts0 is None or pts1 is None or len(pts0) == 0:
        return canvas

    limit = min(max_points, len(pts0))
    indices = np.linspace(0, len(pts0) - 1, limit, dtype=int)
    for idx in indices:
        p0 = tuple(np.round(pts0[idx]).astype(int))
        p1 = tuple(np.round(pts1[idx]).astype(int) + np.array([src_w, 0]))
        color = (
            int(40 + (idx * 17) % 180),
            int(90 + (idx * 37) % 150),
            int(120 + (idx * 23) % 120),
        )
        cv2.circle(canvas, p0, 4, color, -1)
        cv2.circle(canvas, p1, 4, color, -1)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
    return canvas
