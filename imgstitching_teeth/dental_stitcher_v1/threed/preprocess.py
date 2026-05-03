from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class Prepared3DAsset:
    bgr_image: np.ndarray
    bgra_image: np.ndarray
    mask: np.ndarray
    source_bbox: tuple[int, int, int, int]
    placed_bbox: tuple[int, int, int, int]
    metadata: dict[str, Any]

    def png_bytes(self, transparent: bool = False) -> bytes:
        image = self.bgra_image if transparent else self.bgr_image
        ext = ".png"
        success, encoded = cv2.imencode(ext, image)
        if not success:
            raise ValueError("Failed to encode prepared 3D asset as PNG.")
        return encoded.tobytes()

    def metadata_bytes(self) -> bytes:
        return json.dumps(self.metadata, ensure_ascii=False, indent=2).encode("utf-8")


def derive_mask_from_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        binary = image > 0
    else:
        binary = np.any(image > 0, axis=2)
    return (binary.astype(np.uint8) * 255)


def prepare_image_for_hunyuan3d(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    target_size: int = 512,
    padding_ratio: float = 0.12,
    keep_largest_component: bool = True,
    white_background: bool = True,
) -> Prepared3DAsset:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    normalized_mask = _normalize_mask(mask, image.shape[:2])
    if normalized_mask is None or cv2.countNonZero(normalized_mask) == 0:
        normalized_mask = derive_mask_from_image(image)

    if keep_largest_component:
        normalized_mask = _keep_largest_component(normalized_mask)

    if cv2.countNonZero(normalized_mask) == 0:
        raise ValueError("Prepared 3D mask is empty after cleanup.")

    source_bbox = _compute_bbox(normalized_mask)
    padded_bbox = _pad_bbox(source_bbox, image.shape[:2], padding_ratio)

    x0, y0, x1, y1 = padded_bbox
    cropped_image = image[y0:y1, x0:x1]
    cropped_mask = normalized_mask[y0:y1, x0:x1]

    canvas_bgr, canvas_bgra, canvas_mask, placed_bbox = _place_on_square_canvas(
        cropped_image,
        cropped_mask,
        target_size=target_size,
        white_background=white_background,
    )

    source_pixels = int(cv2.countNonZero(normalized_mask))
    prepared_pixels = int(cv2.countNonZero(canvas_mask))
    metadata = {
        "source_shape": [int(image.shape[0]), int(image.shape[1])],
        "target_size": int(target_size),
        "padding_ratio": float(padding_ratio),
        "source_bbox_xyxy": list(map(int, padded_bbox)),
        "placed_bbox_xyxy": list(map(int, placed_bbox)),
        "source_mask_pixels": source_pixels,
        "prepared_mask_pixels": prepared_pixels,
        "source_mask_coverage": float(source_pixels / normalized_mask.size),
        "prepared_mask_coverage": float(prepared_pixels / canvas_mask.size),
        "keep_largest_component": bool(keep_largest_component),
        "background_mode": "white" if white_background else "transparent_only",
    }

    return Prepared3DAsset(
        bgr_image=canvas_bgr,
        bgra_image=canvas_bgra,
        mask=canvas_mask,
        source_bbox=padded_bbox,
        placed_bbox=placed_bbox,
        metadata=metadata,
    )


def _normalize_mask(mask: Optional[np.ndarray], target_shape: tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    if mask_np.shape[:2] != target_shape:
        mask_np = cv2.resize(mask_np, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask_np.dtype != np.uint8:
        mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
    _, mask_bin = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned = np.zeros_like(mask)
    cleaned[labels == largest_label] = 255
    return cleaned


def _compute_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    height, width = image_shape
    box_w = max(x1 - x0, 1)
    box_h = max(y1 - y0, 1)
    pad = int(round(max(box_w, box_h) * padding_ratio))

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)
    return (x0, y0, x1, y1)


def _place_on_square_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int,
    white_background: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    crop_h, crop_w = image.shape[:2]
    max_dim = max(crop_h, crop_w, 1)
    content_scale = (target_size * 0.86) / max_dim
    content_scale = min(content_scale, 1.0) if max_dim <= target_size else content_scale

    resized_w = max(1, int(round(crop_w * content_scale)))
    resized_h = max(1, int(round(crop_h * content_scale)))

    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)

    if white_background:
        canvas_bgr = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    else:
        canvas_bgr = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas_bgra = np.zeros((target_size, target_size, 4), dtype=np.uint8)
    canvas_mask = np.zeros((target_size, target_size), dtype=np.uint8)

    y0 = (target_size - resized_h) // 2
    x0 = (target_size - resized_w) // 2
    y1 = y0 + resized_h
    x1 = x0 + resized_w

    fg_mask = resized_mask > 0
    canvas_mask[y0:y1, x0:x1] = resized_mask

    target_region = canvas_bgr[y0:y1, x0:x1]
    target_region[fg_mask] = resized_image[fg_mask]
    canvas_bgr[y0:y1, x0:x1] = target_region

    alpha = resized_mask.copy()
    rgba_region = np.dstack([resized_image, alpha])
    canvas_bgra[y0:y1, x0:x1] = rgba_region

    return canvas_bgr, canvas_bgra, canvas_mask, (x0, y0, x1, y1)
