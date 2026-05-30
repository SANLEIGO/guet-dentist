"""简单的拍摄区域识别。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dental_stitcher_v1.photo_quality import BLACK_BORDER_THRESHOLD


@dataclass
class RegionAssessment:
    predicted_region: str              # "left" | "center" | "right" | "unknown"
    confidence: float                  # 0-1
    center_x: float                    # 0-1
    span_ratio: float                  # 0-1
    symmetry_score: float              # 0-1
    left_ratio: float                  # 0-1
    center_ratio: float                # 0-1
    right_ratio: float                 # 0-1
    tooth_area_ratio: float            # 0-1
    status_text: str


def infer_simple_capture_region(image: np.ndarray) -> RegionAssessment:
    """基于亮牙区域分布，粗略判断当前更像左侧段/前牙段/右侧段。"""
    if image is None or image.size == 0:
        return RegionAssessment(
            predicted_region="unknown",
            confidence=0.0,
            center_x=0.5,
            span_ratio=0.0,
            symmetry_score=0.0,
            left_ratio=0.0,
            center_ratio=0.0,
            right_ratio=0.0,
            tooth_area_ratio=0.0,
            status_text="区域识别失败：空画面",
        )

    tooth_mask = _build_tooth_candidate_mask(image)
    h, w = tooth_mask.shape
    tooth_area = int(np.count_nonzero(tooth_mask))
    tooth_area_ratio = tooth_area / float(h * w)
    if tooth_area_ratio < 0.015:
        return RegionAssessment(
            predicted_region="unknown",
            confidence=0.18,
            center_x=0.5,
            span_ratio=0.0,
            symmetry_score=0.0,
            left_ratio=0.0,
            center_ratio=0.0,
            right_ratio=0.0,
            tooth_area_ratio=tooth_area_ratio,
            status_text="牙齿亮区太少，区域判断不稳定",
        )

    ys, xs = np.nonzero(tooth_mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox_w = max(1, x1 - x0 + 1)
    span_ratio = bbox_w / float(w)
    center_x = float(xs.mean() / max(w - 1, 1))

    thirds = np.array_split(tooth_mask, 3, axis=1)
    band_counts = np.array([np.count_nonzero(part) for part in thirds], dtype=np.float32)
    band_total = float(np.sum(band_counts)) if np.sum(band_counts) > 0 else 1.0
    left_ratio, center_ratio, right_ratio = (band_counts / band_total).tolist()

    symmetry_score = _compute_horizontal_symmetry(tooth_mask[y0:y1 + 1, x0:x1 + 1])
    left_bias = float(np.clip((0.5 - center_x) / 0.24, 0.0, 1.0))
    right_bias = float(np.clip((center_x - 0.5) / 0.24, 0.0, 1.0))
    band_bias = float(np.clip((right_ratio - left_ratio) / 0.30, -1.0, 1.0))
    side_strength = float(np.clip(
        0.45 * abs(band_bias)
        + 0.20 * max(left_bias, right_bias)
        + 0.20 * np.clip((0.62 - symmetry_score) / 0.30, 0.0, 1.0)
        + 0.15 * np.clip((0.60 - span_ratio) / 0.25, 0.0, 1.0),
        0.0,
        1.0,
    ))
    center_strength = float(np.clip(
        0.38 * np.clip((symmetry_score - 0.45) / 0.40, 0.0, 1.0)
        + 0.34 * np.clip((span_ratio - 0.42) / 0.28, 0.0, 1.0)
        + 0.18 * np.clip((0.22 - abs(center_x - 0.5)) / 0.22, 0.0, 1.0)
        + 0.10 * np.clip((center_ratio - 0.24) / 0.20, 0.0, 1.0),
        0.0,
        1.0,
    ))

    if center_strength >= max(side_strength * 0.92, 0.46):
        predicted_region = "center"
        confidence = center_strength
    else:
        predicted_region = "left" if (band_bias < 0 or left_bias > right_bias) else "right"
        confidence = side_strength

    if confidence < 0.36:
        predicted_region = "unknown"

    status_text = _format_region_status(
        predicted_region=predicted_region,
        confidence=confidence,
        center_x=center_x,
        symmetry_score=symmetry_score,
        span_ratio=span_ratio,
    )

    return RegionAssessment(
        predicted_region=predicted_region,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        center_x=center_x,
        span_ratio=span_ratio,
        symmetry_score=symmetry_score,
        left_ratio=left_ratio,
        center_ratio=center_ratio,
        right_ratio=right_ratio,
        tooth_area_ratio=tooth_area_ratio,
        status_text=status_text,
    )


def _build_tooth_candidate_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    effective_mask = gray >= BLACK_BORDER_THRESHOLD
    effective_pixels = gray[effective_mask] if np.any(effective_mask) else gray.reshape(-1)

    if effective_pixels.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    threshold_value = max(75.0, float(np.percentile(effective_pixels, 72)))
    tooth_mask = (gray >= threshold_value).astype(np.uint8) * 255
    tooth_mask[gray < BLACK_BORDER_THRESHOLD] = 0

    kernel = np.ones((5, 5), dtype=np.uint8)
    tooth_mask = cv2.morphologyEx(tooth_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    tooth_mask = cv2.morphologyEx(tooth_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    tooth_mask = _keep_largest_components(tooth_mask, max_components=4)
    return tooth_mask


def _keep_largest_components(mask: np.ndarray, max_components: int = 4) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    component_ids = sorted(
        range(1, num_labels),
        key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]),
        reverse=True,
    )[:max_components]

    filtered = np.zeros_like(mask)
    for idx in component_ids:
        filtered[labels == idx] = 255
    return filtered


def _compute_horizontal_symmetry(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0

    resized = cv2.resize(mask, (120, 80), interpolation=cv2.INTER_NEAREST)
    left = resized[:, :60]
    right = cv2.flip(resized[:, 60:], 1)
    diff = np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))) / 255.0
    return float(np.clip(1.0 - diff, 0.0, 1.0))


def _format_region_status(
    *,
    predicted_region: str,
    confidence: float,
    center_x: float,
    symmetry_score: float,
    span_ratio: float,
) -> str:
    region_label = {
        "left": "左侧段",
        "center": "前牙段",
        "right": "右侧段",
        "unknown": "未知区域",
    }[predicted_region]

    if predicted_region == "unknown":
        return "区域识别不稳定，系统暂时无法判断左/中/右"

    return (
        f"简单区域识别：疑似{region_label}"
        f"（置信度 {confidence * 100:.0f}% / 居中 {center_x:.2f} / 对称 {symmetry_score:.2f} / 横向覆盖 {span_ratio:.2f}）"
    )
