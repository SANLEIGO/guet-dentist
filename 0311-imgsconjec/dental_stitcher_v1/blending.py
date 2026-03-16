from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def blend_images(
    base: np.ndarray,
    overlay: np.ndarray,
    homography: Optional[np.ndarray],
    feather_width: int = 30,
) -> np.ndarray:
    if homography is None:
        return base.copy()

    base_h, base_w = base.shape[:2]
    overlay_h, overlay_w = overlay.shape[:2]

    corners = np.array(
        [[0, 0, 1], [overlay_w, 0, 1], [overlay_w, overlay_h, 1], [0, overlay_h, 1]], dtype=np.float32
    )
    warped_corners = (homography @ corners.T).T
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2, np.newaxis]

    min_x = int(min(0, warped_corners[:, 0].min()))
    min_y = int(min(0, warped_corners[:, 1].min()))
    max_x = int(max(base_w, warped_corners[:, 0].max()))
    max_y = int(max(base_h, warped_corners[:, 1].max()))

    canvas_w = max_x - min_x
    canvas_h = max_y - min_y

    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
    warped_overlay = cv2.warpPerspective(
        overlay,
        translation @ homography,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    warped_mask = cv2.warpPerspective(
        np.ones((overlay_h, overlay_w), dtype=np.uint8) * 255,
        translation @ homography,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas[-min_y : -min_y + base_h, -min_x : -min_x + base_w] = base.astype(np.float32)

    feather = max(5, feather_width)
    kernel = feather * 2 + 1
    feathered = cv2.GaussianBlur(warped_mask.astype(np.float32) / 255.0, (kernel, kernel), feather / 3)

    canvas = canvas * (1 - feathered[..., None]) + warped_overlay.astype(np.float32) * feathered[..., None]
    return np.clip(canvas, 0, 255).astype(np.uint8)
