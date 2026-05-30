from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np

from dental_stitcher_v1.pipeline import _evaluate_pair_quality
from dental_stitcher_v1.registration import RegistrationResult, estimate_transform
from dental_stitcher_v1.segmentation import SegmentationResult


def _make_keypoints_and_matches():
    src_points = [
        (40.0, 40.0),
        (80.0, 42.0),
        (120.0, 45.0),
        (160.0, 47.0),
        (200.0, 50.0),
        (240.0, 54.0),
        (280.0, 58.0),
        (320.0, 62.0),
    ]
    kp1 = [cv2.KeyPoint(x=float(x), y=float(y), size=1) for x, y in src_points]
    kp2 = [cv2.KeyPoint(x=float(x + 36.0), y=float(y + 4.0), size=1) for x, y in src_points]
    matches = [
        cv2.DMatch(_queryIdx=index, _trainIdx=index, _imgIdx=0, _distance=0.1 + index * 0.01)
        for index in range(len(src_points))
    ]
    return kp1, kp2, matches


def test_estimate_transform_prefers_affine_when_homography_is_unstable() -> None:
    kp1, kp2, matches = _make_keypoints_and_matches()
    bad_homography = np.array(
        [[1.0, 0.0, 36.0], [0.0, 1.0, 4.0], [0.0035, 0.0, 1.0]],
        dtype=np.float32,
    )
    affine = np.array([[1.0, 0.0, 36.0], [0.0, 1.0, 4.0]], dtype=np.float32)
    mask = np.ones((len(matches), 1), dtype=np.uint8)

    with patch("dental_stitcher_v1.registration.cv2.findHomography", return_value=(bad_homography, mask)), patch(
        "dental_stitcher_v1.registration.cv2.estimateAffine2D",
        return_value=(affine, mask),
    ), patch(
        "dental_stitcher_v1.registration.cv2.estimateAffinePartial2D",
        return_value=(affine, mask),
    ):
        result = estimate_transform(kp1, kp2, matches)

    assert result.homography is not None
    assert result.method.startswith("affine")
    assert result.fallback_reason is not None
    assert "homography_unstable" in result.fallback_reason


def test_quality_gate_rejects_perspective_explosion() -> None:
    mask = np.ones((180, 320), dtype=np.uint8) * 255
    overlay = np.zeros((180, 320, 3), dtype=np.uint8)
    seg_results = [
        SegmentationResult(mask=mask, overlay=overlay, method="mask"),
        SegmentationResult(mask=mask, overlay=overlay, method="mask"),
    ]
    per_image = [
        {"mask_coverage": 1.0, "used_fallback_mask": False},
        {"mask_coverage": 1.0, "used_fallback_mask": False},
    ]
    reg_result = RegistrationResult(
        homography=np.eye(3, dtype=np.float32),
        inlier_mask=np.ones(20, dtype=np.uint8),
        method="homography",
        inlier_count=20,
        inlier_ratio=1.0,
        reprojection_error=0.5,
    )
    unstable_warp = np.array(
        [[1.0, 0.0, 12.0], [0.0, 1.0, 3.0], [0.004, 0.0, 1.0]],
        dtype=np.float32,
    )

    quality_gate = _evaluate_pair_quality(
        seg_results=seg_results,
        per_image=per_image,
        match_count=20,
        reg_result=reg_result,
        blend_homography=unstable_warp,
    )

    assert quality_gate["gate_passed"] is False
    assert any(
        reason in quality_gate["fail_reasons"]
        for reason in {"warp_area_ratio_out_of_range", "warp_canvas_too_large", "warp_perspective_too_strong"}
    )
