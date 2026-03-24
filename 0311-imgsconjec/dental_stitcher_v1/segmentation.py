from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    mask: np.ndarray
    overlay: np.ndarray
    method: str
    fallback_reason: Optional[str] = None


_SEG_MODEL: Any = None
_SEG_MODEL_ERROR: Optional[str] = None


def segment_teeth(image: np.ndarray) -> SegmentationResult:
    """Segment teeth region using AlphaDent + GrabCut.

    Deep model (AlphaDent) is required; when unavailable returns full mask.
    """
    deep_result = _segment_alphadent(image)
    if deep_result is not None:
        return deep_result
    return fallback_full_mask(image)


def _segment_alphadent(image: np.ndarray) -> Optional[SegmentationResult]:
    global _SEG_MODEL_ERROR

    model = _get_alphadent_model()
    if model is None:
        print(f"[DEBUG] AlphaDent model is None, error: {_SEG_MODEL_ERROR}")
        return None
    if image.ndim != 3 or image.shape[2] != 3:
        _SEG_MODEL_ERROR = f"alphadent_expected_bgr_image, got shape={image.shape}"
        print(f"[DEBUG] {_SEG_MODEL_ERROR}")
        return None

    # 打印图像信息
    h, w = image.shape[:2]
    print(f"[DEBUG] Input image: {w}x{h}, dtype={image.dtype}")

    # 图像增强：CLAHE（使用更高的 clip_limit 5.0）
    enhanced = _enhance_image_clahe(image, clip_limit=5.0)

    # 尝试多个置信度阈值（最后一个 0.001 作为最后手段）
    conf_thresholds = [0.1, 0.05, 0.02, 0.01, 0.001]

    for idx_conf, conf in enumerate(conf_thresholds):
        try:
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, imgsz=960, conf=conf, verbose=False)
            print(f"[DEBUG] Model prediction (conf={conf}): {len(results) if results else 0} results")

            if results and len(results) > 0:
                # 打印检测结果
                r = results[0]
                if hasattr(r, 'boxes') and r.boxes is not None:
                    num_boxes = len(r.boxes)
                    print(f"[DEBUG]   Detected {num_boxes} boxes at conf={conf}")
                    if num_boxes > 0:
                        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else r.boxes.conf
                        print(f"[DEBUG]   Confidence range: {confs.min():.4f} - {confs.max():.4f}")

                        # 如果是最后一个阈值 (0.001) 且检测到的框置信度极低，发出警告
                        if conf == 0.001 and confs.max() < 0.01:
                            print(f"[WARNING] 检测到的目标置信度极低 (max={confs.max():.4f})，可能是误检")
                else:
                    print(f"[DEBUG]   boxes attribute: {r.boxes if hasattr(r, 'boxes') else 'N/A'}")
        except Exception as exc:  # pragma: no cover - defensive
            _SEG_MODEL_ERROR = f"alphadent_inference_failed: {exc}"
            print(f"[DEBUG] {_SEG_MODEL_ERROR}")
            import traceback
            traceback.print_exc()
            return None

        if not results:
            print(f"[DEBUG] No results with conf={conf}, trying lower threshold...")
            continue

        masks = results[0].masks
        if masks is None or masks.data is None or masks.data.shape[0] == 0:
            print(f"[DEBUG] No masks found with conf={conf}: masks={masks}, data={masks.data if masks else None}")
            continue

        # 找到了有效的 mask
        mask_data = masks.data
        if hasattr(mask_data, "detach"):
            mask_data = mask_data.detach().cpu().numpy()
        mask_bool = np.any(mask_data > 0.5, axis=0)
        combined = (mask_bool.astype(np.uint8) * 255)
        pixel_count = cv2.countNonZero(combined)
        print(f"[DEBUG] Mask generated (conf={conf}): {pixel_count} pixels")

        # 如果检测到的像素太少（< 1000），继续尝试更低的阈值
        if pixel_count < 1000 and conf != conf_thresholds[-1]:
            print(f"[DEBUG] Mask too small ({pixel_count} pixels), trying lower threshold...")
            continue

        # 将掩膜调整回原图尺寸（如果增强时改变了尺寸）
        if enhanced.shape != image.shape:
            combined = cv2.resize(combined, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        alphadent_mask = _normalize_mask(combined, image.shape[:2])
        refined = _grabcut_refine(image, alphadent_mask)
        mask = _fill_mask_holes(refined)
        overlay = _overlay_mask(image, mask)
        return SegmentationResult(mask=mask, overlay=overlay, method="alphadent_grabcut_enhanced", fallback_reason=None)

    # 所有阈值都失败
    _SEG_MODEL_ERROR = "alphadent_no_detections_at_any_confidence"
    print(f"[DEBUG] {_SEG_MODEL_ERROR}")
    return None



def _get_alphadent_model() -> Optional[Any]:
    global _SEG_MODEL, _SEG_MODEL_ERROR
    if _SEG_MODEL is not None:
        return _SEG_MODEL
    if _SEG_MODEL_ERROR is not None:
        return None
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - import guard
        _SEG_MODEL_ERROR = f"alphadent_import_failed: {exc}"
        return None
    weights_path = os.getenv("DENTAL_SEG_WEIGHTS")
    if not weights_path:
        _SEG_MODEL_ERROR = "alphadent_weights_missing"
        return None
    if not os.path.exists(weights_path):
        _SEG_MODEL_ERROR = f"alphadent_weights_not_found: {weights_path}"
        return None
    try:
        _SEG_MODEL = YOLO(weights_path)
    except Exception as exc:
        _SEG_MODEL_ERROR = f"alphadent_load_failed: {exc}"
        return None
    return _SEG_MODEL



def _normalize_mask(mask: Any, target_shape: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    if mask_np.shape[:2] != target_shape:
        mask_np = cv2.resize(mask_np, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask_np.dtype != np.uint8:
        mask_min, mask_max = float(mask_np.min()), float(mask_np.max())
        if mask_max <= 1.0:
            mask_np = (mask_np * 255.0).astype(np.uint8)
        else:
            mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
    else:
        mask_np = mask_np.copy()
    _, mask_bin = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    return mask_bin



def _coarse_tooth_candidate(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    bright = val >= 140
    low_sat = sat <= 95
    yellowish = b_channel >= 125
    not_red = a_channel < 145

    base = bright & low_sat & yellowish & not_red
    candidate = base.astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return candidate


def _enhance_image_clahe(image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """使用 CLAHE 增强图像对比度，特别适合暗图像和低对比度图像。

    Args:
        image: BGR 图像
        clip_limit: CLAHE 对比度限制，默认 3.0（范围 0-10，越高对比度越强）

    Returns:
        增强后的 BGR 图像
    """
    # 计算原图亮度
    original_brightness = image.mean()
    print(f"[DEBUG] Original brightness: {original_brightness:.2f}")

    # 转换到 LAB 颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 对 L 通道应用 CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # 转回 BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 计算增强后亮度
    enhanced_brightness = enhanced.mean()
    print(f"[DEBUG] Enhanced brightness: {enhanced_brightness:.2f} (gain: {(enhanced_brightness/(original_brightness+1e-6)):.2f}x)")

    return enhanced


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    h, w = mask_u8.shape[:2]
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, holes)
    return filled


def _grabcut_refine(image: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    h, w = seed_mask.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.erode(seed_mask, kernel)
    mask[fg > 0] = cv2.GC_FGD

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    a_channel = lab[:, :, 1]
    red_soft = (a_channel >= 150) & (sat >= 120)
    dark_soft = (val <= 60) & (sat >= 80)
    bg = red_soft | dark_soft
    mask[bg] = cv2.GC_BGD

    border = 10
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return seed_mask
    refined = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return refined


def fallback_full_mask(image: np.ndarray) -> SegmentationResult:
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    return SegmentationResult(mask=mask, overlay=_overlay_mask(image, mask), method="full", fallback_reason="segmentation_failed")


def _overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color = np.zeros_like(image)
    color[:, :, 1] = 200
    alpha = 0.35
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    overlay = (overlay * (1 - alpha * mask_3) + color * (alpha * mask_3)).astype(np.uint8)
    return overlay
