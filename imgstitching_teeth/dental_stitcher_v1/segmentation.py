from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from dotenv import dotenv_values

# 仅读取 .env 文件，不依赖系统环境变量
ENV_PATH = Path(__file__).parent.parent / ".env"
_ENV_CONFIG = dotenv_values(ENV_PATH)


@dataclass
class SegmentationResult:
    mask: np.ndarray
    overlay: np.ndarray
    method: str
    fallback_reason: Optional[str] = None


_SEG_MODEL: Any = None
_SEG_MODEL_ERROR: Optional[str] = None

_UNET_MODEL: Any = None
_UNET_DEVICE: Optional[torch.device] = None
_UNET_MODEL_ERROR: Optional[str] = None


def segment_teeth(
    image: np.ndarray,
    method: str = "alphadent",
    use_grabcut: bool = True,
    use_enhancement: bool = False,
    enhancement_level: float = 3.0
) -> SegmentationResult:
    """Segment teeth region using selected method.

    Args:
        image: Input BGR image
        method: Segmentation method - "alphadent" or "unet"
        use_grabcut: Whether to apply GrabCut refinement
        use_enhancement: Whether to apply CLAHE enhancement before segmentation
        enhancement_level: CLAHE clip limit for enhancement

    Returns:
        SegmentationResult with mask, overlay, and method info

    Raises:
        RuntimeError: If the selected segmentation method is not available
    """
    # 保存原始图像用于显示掩膜
    original_image = image.copy()

    # Apply CLAHE enhancement if requested (仅用于分割，不影响显示)
    if use_enhancement:
        image = _apply_clahe(image, enhancement_level)

    if method == "unet":
        unet_result = _segment_unet(image, use_grabcut, original_image)
        if unet_result is not None:
            return unet_result
        # U-Net 不可用时抛出异常，不回退
        global _UNET_MODEL_ERROR
        error_msg = _UNET_MODEL_ERROR if _UNET_MODEL_ERROR else "unknown_error"
        raise RuntimeError(f"U-Net segmentation failed: {error_msg}. Please check the model file and configuration.")
    else:
        deep_result = _segment_alphadent(image, use_grabcut, original_image)
        if deep_result is not None:
            return deep_result
        # AlphaDent 不可用时也抛出异常
        global _SEG_MODEL_ERROR
        error_msg = _SEG_MODEL_ERROR if _SEG_MODEL_ERROR else "unknown_error"
        raise RuntimeError(f"AlphaDent segmentation failed: {error_msg}. Please check the model file and configuration.")


def _segment_alphadent(
    image: np.ndarray,
    use_grabcut: bool = True,
    original_image: Optional[np.ndarray] = None
) -> Optional[SegmentationResult]:
    """Segment teeth using AlphaDent (YOLOv8) model.

    Args:
        image: Input BGR image (可能已增强)
        use_grabcut: Whether to apply GrabCut refinement
        original_image: 原始图像（用于显示掩膜），如果为 None 则使用 image
    """
    display_image = original_image if original_image is not None else image

    model = _get_alphadent_model()
    if model is None:
        return None
    if image.ndim != 3 or image.shape[2] != 3:
        global _SEG_MODEL_ERROR
        _SEG_MODEL_ERROR = "alphadent_expected_bgr_image"
        return None
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, imgsz=960, conf=0.1, verbose=False)
    except Exception as exc:  # pragma: no cover - defensive
        _SEG_MODEL_ERROR = f"alphadent_inference_failed: {exc}"
        return None
    if not results:
        _SEG_MODEL_ERROR = "alphadent_no_results"
        return None
    masks = results[0].masks
    if masks is None or masks.data is None or masks.data.shape[0] == 0:
        combined = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        mask_data = masks.data
        if hasattr(mask_data, "detach"):
            mask_data = mask_data.detach().cpu().numpy()
        mask_bool = np.any(mask_data > 0.5, axis=0)
        combined = (mask_bool.astype(np.uint8) * 255)
    alphadent_mask = _normalize_mask(combined, image.shape[:2])

    if use_grabcut:
        refined = _grabcut_refine(image, alphadent_mask)
    else:
        refined = alphadent_mask

    mask = _fill_mask_holes(refined)
    overlay = _overlay_mask(display_image, mask)  # 使用原始图像显示
    return SegmentationResult(
        mask=mask,
        overlay=overlay,
        method="alphadent" + ("_grabcut" if use_grabcut else ""),
        fallback_reason=None
    )



def _get_alphadent_model() -> Optional[Any]:
    global _SEG_MODEL, _SEG_MODEL_ERROR
    if _SEG_MODEL is not None:
        return _SEG_MODEL
    # 文件不存在类的错误允许重试（用户可能修改了 .env）
    if _SEG_MODEL_ERROR is not None and "not_found" not in _SEG_MODEL_ERROR:
        return None
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - import guard
        _SEG_MODEL_ERROR = f"alphadent_import_failed: {exc}"
        return None

    # 仅从 .env 文件读取模型路径
    weights_path = _ENV_CONFIG.get("DENTAL_SEG_WEIGHTS")
    if not weights_path:
        # .env 未配置时使用默认路径
        weights_path = str(Path(__file__).parent.parent / "pts" / "alphadent_9cls_960.pt")

    if not os.path.exists(weights_path):
        _SEG_MODEL_ERROR = f"alphadent_weights_not_found: {weights_path}"
        return None
    try:
        _SEG_MODEL = YOLO(weights_path)
        _SEG_MODEL_ERROR = None
    except Exception as exc:
        _SEG_MODEL_ERROR = f"alphadent_load_failed: {exc}"
        return None
    return _SEG_MODEL


def _get_unet_model() -> Optional[Any]:
    """Get or load U-Net model."""
    global _UNET_MODEL, _UNET_DEVICE, _UNET_MODEL_ERROR

    if _UNET_MODEL is not None:
        return _UNET_MODEL

    if _UNET_MODEL_ERROR is not None:
        return None

    # 从 .env 文件读取 U-Net 模型路径
    model_path = _ENV_CONFIG.get("UNET_WEIGHTS")
    if not model_path:
        # .env 未配置时使用默认路径
        model_path = str(Path(__file__).parent.parent / "pts" / "best_model_1.pth")

    if not os.path.exists(model_path):
        _UNET_MODEL_ERROR = f"unet_weights_not_found: {model_path}"
        return None

    try:
        # 设置设备
        _UNET_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 导入 U-Net 模型
        from dental_stitcher_v1.model.unet_resnet import Unet

        # 创建模型（2类：背景 + 牙齿）
        _UNET_MODEL = Unet(num_classes=2)

        # 加载 state_dict
        state_dict = torch.load(model_path, map_location=_UNET_DEVICE, weights_only=False)
        _UNET_MODEL.load_state_dict(state_dict)
        _UNET_MODEL.eval()
        _UNET_MODEL.to(_UNET_DEVICE)

        _UNET_MODEL_ERROR = None

    except Exception as exc:
        _UNET_MODEL_ERROR = f"unet_load_failed: {exc}"
        return None

    return _UNET_MODEL


def _segment_unet(
    image: np.ndarray,
    use_grabcut: bool = True,
    original_image: Optional[np.ndarray] = None
) -> Optional[SegmentationResult]:
    """Segment teeth using U-Net model.

    Args:
        image: Input BGR image (可能已增强)
        use_grabcut: Whether to apply GrabCut refinement
        original_image: 原始图像（用于显示掩膜），如果为 None 则使用 image
    """
    display_image = original_image if original_image is not None else image

    model = _get_unet_model()
    if model is None:
        return None

    try:
        # 预处理图像
        input_shape = [480, 480]
        original_h, original_w = image.shape[:2]

        # 转换为 RGB PIL Image
        from PIL import Image
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 调整图像大小（保持宽高比）
        from dental_stitcher_v1.segmentation_unet_utils import cvtColor, preprocess_input, resize_image
        image_pil = cvtColor(image_pil)
        image_data, nw, nh = resize_image(image_pil, (input_shape[1], input_shape[0]))

        # 预处理
        image_data = np.array(image_data, np.float32)
        image_data = np.transpose(preprocess_input(image_data), (2, 0, 1))
        image_data = np.expand_dims(image_data, 0)

        # 转换为 tensor
        images = torch.from_numpy(image_data).to(_UNET_DEVICE)

        # 推理
        with torch.no_grad():
            pr = model(images)[0]
            pr = torch.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

        # 恢复图像尺寸
        pr = pr[
            int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh),
            int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)
        ]
        pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        # 获取预测类别（取前景类的概率）
        if pr.shape[-1] > 1:
            foreground_prob = pr[:, :, 1]
        else:
            foreground_prob = pr[:, :, 0]

        # 应用阈值
        unet_mask = (foreground_prob >= 0.5).astype(np.uint8) * 255

        # 填充孔洞
        unet_mask = _fill_mask_holes(unet_mask)

        # 形态学操作平滑边界
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        unet_mask = cv2.morphologyEx(unet_mask, cv2.MORPH_CLOSE, kernel)
        unet_mask = cv2.morphologyEx(unet_mask, cv2.MORPH_OPEN, kernel)

        if use_grabcut:
            refined = _grabcut_refine(image, unet_mask)
        else:
            refined = unet_mask

        mask = _fill_mask_holes(refined)
        overlay = _overlay_mask(display_image, mask)  # 使用原始图像显示

        return SegmentationResult(
            mask=mask,
            overlay=overlay,
            method="unet" + ("_grabcut" if use_grabcut else ""),
            fallback_reason=None
        )

    except Exception as exc:
        global _UNET_MODEL_ERROR
        _UNET_MODEL_ERROR = f"unet_inference_failed: {exc}"
        return None


def _apply_clahe(image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image.

    Args:
        image: Input BGR image
        clip_limit: Contrast limit for CLAHE (higher = more enhancement)

    Returns:
        Enhanced BGR image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel_enhanced = clahe.apply(l_channel)

    # Merge channels and convert back to BGR
    lab_enhanced = cv2.merge([l_channel_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced_bgr



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
