from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class RegistrationResult:
    homography: Optional[np.ndarray]
    inlier_mask: np.ndarray
    method: str
    inlier_count: int
    inlier_ratio: float
    reprojection_error: float
    fallback_reason: Optional[str] = None


@dataclass
class MatchDiagnostics:
    src_points: Optional[np.ndarray]
    dst_points: Optional[np.ndarray]


def estimate_transform(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    use_affine_fallback: bool = True,
) -> RegistrationResult:
    if len(matches) < 8:
        return RegistrationResult(
            homography=None,
            inlier_mask=np.array([], dtype=np.uint8),
            method="homography",
            inlier_count=0,
            inlier_ratio=0.0,
            reprojection_error=float("inf"),
            fallback_reason="insufficient_matches",
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    if homography is not None and mask is not None:
        inlier_mask = mask.astype(np.uint8).flatten()
        inliers = int(inlier_mask.sum())
        ratio = inliers / max(len(matches), 1)
        reproj = _reprojection_error(src_pts, dst_pts, homography, inlier_mask)
        return RegistrationResult(
            homography=homography,
            inlier_mask=inlier_mask,
            method="homography",
            inlier_count=inliers,
            inlier_ratio=ratio,
            reprojection_error=reproj,
        )

    if use_affine_fallback:
        affine, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if affine is not None and mask is not None:
            homography = np.vstack([affine, [0, 0, 1]])
            inlier_mask = mask.astype(np.uint8).flatten()
            inliers = int(inlier_mask.sum())
            ratio = inliers / max(len(matches), 1)
            reproj = _reprojection_error(src_pts, dst_pts, homography, inlier_mask)
            return RegistrationResult(
                homography=homography,
                inlier_mask=inlier_mask,
                method="affine",
                inlier_count=inliers,
                inlier_ratio=ratio,
                reprojection_error=reproj,
                fallback_reason="homography_failed",
            )

    return RegistrationResult(
        homography=None,
        inlier_mask=np.array([], dtype=np.uint8),
        method="homography",
        inlier_count=0,
        inlier_ratio=0.0,
        reprojection_error=float("inf"),
        fallback_reason="transform_failed",
    )


def _reprojection_error(
    src_pts: np.ndarray, dst_pts: np.ndarray, homography: np.ndarray, mask: np.ndarray
) -> float:
    if mask.sum() == 0:
        return float("inf")
    src = src_pts.reshape(-1, 2)
    dst = dst_pts.reshape(-1, 2)
    src_h = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    diff = src_h - dst
    errors = np.sqrt((diff ** 2).sum(axis=1))
    return float(errors[mask.astype(bool)].mean())
