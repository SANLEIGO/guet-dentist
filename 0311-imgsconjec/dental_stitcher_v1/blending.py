from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def blend_images(
    base: np.ndarray,
    overlay: np.ndarray,
    homography: Optional[np.ndarray],
    feather_width: int = 30,
    base_mask: Optional[np.ndarray] = None,
    overlay_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if homography is None:
        fallback_mask = _normalize_mask_input(base_mask, base.shape[:2])
        if fallback_mask is None:
            fallback_mask = _nonzero_content_mask(base)
        return base.copy(), fallback_mask

    base_h, base_w = base.shape[:2]
    overlay_h, overlay_w = overlay.shape[:2]

    corners = np.array(
        [[0, 0, 1], [overlay_w, 0, 1], [overlay_w, overlay_h, 1], [0, overlay_h, 1]], dtype=np.float32
    )
    warped_corners = (homography @ corners.T).T
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2, np.newaxis]

    min_x = int(np.floor(min(0, warped_corners[:, 0].min())))
    min_y = int(np.floor(min(0, warped_corners[:, 1].min())))
    max_x = int(np.ceil(max(base_w, warped_corners[:, 0].max())))
    max_y = int(np.ceil(max(base_h, warped_corners[:, 1].max())))

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
    warp_matrix = translation @ homography

    base_mask_input = _normalize_mask_input(base_mask, base.shape[:2])
    overlay_mask_input = _normalize_mask_input(overlay_mask, overlay.shape[:2])
    if base_mask_input is None:
        base_mask_input = _nonzero_content_mask(base)
    if overlay_mask_input is None:
        overlay_mask_input = _nonzero_content_mask(overlay)

    warped_overlay = cv2.warpPerspective(
        overlay,
        warp_matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    ).astype(np.float32)

    warped_overlay_mask = cv2.warpPerspective(
        overlay_mask_input,
        warp_matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    base_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    base_canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    y0 = -min_y
    x0 = -min_x
    base_canvas[y0 : y0 + base_h, x0 : x0 + base_w] = base.astype(np.float32)
    base_canvas_mask[y0 : y0 + base_h, x0 : x0 + base_w] = base_mask_input

    base_binary = base_canvas_mask > 0
    overlay_binary = warped_overlay_mask > 0
    overlap = np.logical_and(base_binary, overlay_binary)
    base_only = np.logical_and(base_binary, ~overlay_binary)
    overlay_only = np.logical_and(overlay_binary, ~base_binary)

    blended = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    blended[base_only] = base_canvas[base_only]
    blended[overlay_only] = warped_overlay[overlay_only]

    if np.any(overlap):
        base_distance = cv2.distanceTransform((base_binary.astype(np.uint8) * 255), cv2.DIST_L2, 5)
        overlay_distance = cv2.distanceTransform((overlay_binary.astype(np.uint8) * 255), cv2.DIST_L2, 5)

        overlay_preferred = (overlay_distance > base_distance).astype(np.float32)
        seam_blur = max(3, feather_width // 3)
        seam_kernel = seam_blur * 2 + 1
        overlay_alpha = cv2.GaussianBlur(overlay_preferred, (seam_kernel, seam_kernel), seam_blur / 2)
        overlay_alpha = np.clip(overlay_alpha, 0.0, 1.0)
        overlay_alpha = np.where(overlap, overlay_alpha, 0.0)
        base_alpha = np.where(overlap, 1.0 - overlay_alpha, 0.0)

        blended[overlap] = (
            base_canvas[overlap] * base_alpha[overlap, None]
            + warped_overlay[overlap] * overlay_alpha[overlap, None]
        )

    occupancy = np.logical_or(base_binary, overlay_binary)
    if not np.any(occupancy):
        return np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)

    ys, xs = np.where(occupancy)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    blended = blended[y1:y2, x1:x2]
    occupancy_mask = (occupancy[y1:y2, x1:x2].astype(np.uint8) * 255)

    return np.clip(blended, 0, 255).astype(np.uint8), occupancy_mask


def _normalize_mask_input(mask: Optional[np.ndarray], target_shape: tuple[int, int]) -> Optional[np.ndarray]:
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


def _nonzero_content_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        binary = image > 0
    else:
        binary = np.any(image > 0, axis=2)
    return (binary.astype(np.uint8) * 255)
