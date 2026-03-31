from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from dental_stitcher_v1.registration import RegistrationResult


def render_mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color = np.zeros_like(image)
    color[:, :, 1] = 220
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    alpha = 0.35
    return (overlay * (1 - alpha * mask_3) + color * (alpha * mask_3)).astype(np.uint8)


def render_matches(
    src: np.ndarray,
    dst: np.ndarray,
    src_pts: Optional[np.ndarray],
    dst_pts: Optional[np.ndarray],
    max_points: int = 50,
) -> np.ndarray:
    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    src_h, src_w = src_rgb.shape[:2]
    dst_h, dst_w = dst_rgb.shape[:2]
    canvas_h = max(src_h, dst_h)
    canvas_w = src_w + dst_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:src_h, :src_w] = src_rgb
    canvas[:dst_h, src_w : src_w + dst_w] = dst_rgb

    if src_pts is None or dst_pts is None or len(src_pts) == 0:
        return canvas

    limit = min(max_points, len(src_pts))
    indices = np.linspace(0, len(src_pts) - 1, limit, dtype=int)
    for idx in indices:
        p0 = tuple(np.round(src_pts[idx]).astype(int))
        p1 = tuple(np.round(dst_pts[idx]).astype(int) + np.array([src_w, 0]))
        color = (
            int(40 + (idx * 17) % 180),
            int(90 + (idx * 37) % 150),
            int(120 + (idx * 23) % 120),
        )
        cv2.circle(canvas, p0, 4, color, -1)
        cv2.circle(canvas, p1, 4, color, -1)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
    return canvas


def render_registration_overlay(
    base: np.ndarray,
    overlay: np.ndarray,
    transform: Optional[np.ndarray],
) -> np.ndarray:
    if transform is None:
        return base
    base_h, base_w = base.shape[:2]
    warped = cv2.warpPerspective(overlay, transform, (base_w, base_h))
    blended = cv2.addWeighted(base, 0.6, warped, 0.4, 0)
    return blended
