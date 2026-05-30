from __future__ import annotations

import warnings

import cv2
import numpy as np

from dental_stitcher_v1.blending import blend_images


def _make_feature_rich_canvas() -> np.ndarray:
    image = np.zeros((220, 320, 3), dtype=np.uint8)
    image[:] = (18, 28, 36)
    for idx in range(7):
        center = (45 + idx * 36, 110 + (idx % 2) * 8)
        cv2.circle(image, center, 15 + (idx % 3), (190, 210, 230), thickness=-1)
        cv2.circle(image, center, 10, (120, 150, 175), thickness=2)
    cv2.line(image, (20, 60), (300, 80), (220, 120, 90), 3, cv2.LINE_AA)
    cv2.line(image, (25, 170), (295, 150), (90, 200, 130), 3, cv2.LINE_AA)
    cv2.putText(image, "ARCH", (95, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 220, 150), 2, cv2.LINE_AA)
    return image


def test_piecewise_blending_handles_local_deformation() -> None:
    overlay = _make_feature_rich_canvas()
    base = overlay.copy()
    control_src = np.float32(
        [
            [0, 0],
            [319, 0],
            [319, 219],
            [0, 219],
            [70, 110],
            [150, 100],
            [240, 118],
            [120, 155],
            [220, 70],
        ]
    )
    control_dst = control_src.copy()
    control_dst[4:, 1] += np.array([18, -12, 14, -10, 16], dtype=np.float32)
    control_dst[4:, 0] += np.array([6, -8, 10, -6, 4], dtype=np.float32)

    from skimage.transform import PiecewiseAffineTransform, warp

    transform = PiecewiseAffineTransform()
    assert transform.estimate(control_src, control_dst)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = warp(
            overlay.astype(np.float32),
            inverse_map=transform.inverse,
            output_shape=overlay.shape[:2],
            order=1,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        ).astype(np.uint8)

    mask = np.ones(overlay.shape[:2], dtype=np.uint8) * 255
    result = blend_images(
        base=base,
        overlay=overlay,
        homography=np.eye(3, dtype=np.float32),
        base_mask=mask,
        overlay_mask=mask,
        base_points=control_dst,
        overlay_points=control_src,
    )

    assert result.warp_mode == "piecewise_affine"
    assert result.control_point_count >= len(control_src)
    diff = np.mean(np.abs(result.image.astype(np.float32) - base.astype(np.float32)))
    assert diff < 8.0
