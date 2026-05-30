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

    homography_result = _estimate_homography_candidate(src_pts, dst_pts, match_count=len(matches))
    affine_result = None
    affine_partial_result = None
    if use_affine_fallback:
        affine_result = _estimate_affine_candidate(src_pts, dst_pts, match_count=len(matches), full_affine=True)
        affine_partial_result = _estimate_affine_candidate(src_pts, dst_pts, match_count=len(matches), full_affine=False)
    translation_result = _estimate_translation_candidate(src_pts, dst_pts, match_count=len(matches))

    fallback_candidates = [candidate for candidate in [affine_result, affine_partial_result, translation_result] if candidate is not None]

    if homography_result is not None and _is_homography_plausible(homography_result.homography, src_pts, dst_pts):
        preferred_fallback = _select_preferred_fallback(homography_result, fallback_candidates, src_pts, dst_pts)
        if preferred_fallback is not None:
            preferred_fallback.fallback_reason = _merge_fallback_reasons(
                preferred_fallback.fallback_reason,
                "homography_unstable",
            )
            return preferred_fallback
        return homography_result

    best_fallback = _best_usable_candidate(fallback_candidates)
    if best_fallback is not None:
        best_fallback.fallback_reason = _merge_fallback_reasons(
            best_fallback.fallback_reason,
            "homography_unstable",
        )
        return best_fallback

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


def _estimate_homography_candidate(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    *,
    match_count: int,
) -> Optional[RegistrationResult]:
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    return _build_result(homography, mask, "homography", src_pts, dst_pts, match_count=match_count)


def _estimate_affine_candidate(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    *,
    match_count: int,
    full_affine: bool,
) -> Optional[RegistrationResult]:
    if full_affine:
        affine, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        method_name = "affine_full"
    else:
        affine, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        method_name = "affine_partial"
    if affine is None:
        return None
    homography = np.vstack([affine, [0, 0, 1]])
    result = _build_result(homography, mask, method_name, src_pts, dst_pts, match_count=match_count)
    if result is not None:
        result.fallback_reason = "homography_failed"
    return result


def _estimate_translation_candidate(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    *,
    match_count: int,
) -> Optional[RegistrationResult]:
    src = src_pts.reshape(-1, 2)
    dst = dst_pts.reshape(-1, 2)
    deltas = dst - src
    dx, dy = np.median(deltas, axis=0)
    homography = np.array(
        [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    errors = _point_reprojection_errors(src_pts, dst_pts, homography)
    inlier_mask = (errors <= 5.0).astype(np.uint8)
    return _build_result(homography, inlier_mask, "translation", src_pts, dst_pts, match_count=match_count)


def _build_result(
    homography: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    method: str,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    *,
    match_count: int,
) -> Optional[RegistrationResult]:
    if homography is None or mask is None:
        return None
    if not np.isfinite(homography).all():
        return None
    inlier_mask = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if inlier_mask.size != match_count:
        return None
    inliers = int(inlier_mask.sum())
    ratio = inliers / max(match_count, 1)
    reproj = _reprojection_error(src_pts, dst_pts, homography, inlier_mask)
    return RegistrationResult(
        homography=homography,
        inlier_mask=inlier_mask,
        method=method,
        inlier_count=inliers,
        inlier_ratio=ratio,
        reprojection_error=reproj,
    )


def _candidate_is_usable(
    result: RegistrationResult,
    *,
    min_inliers: int = 8,
    min_inlier_ratio: float = 0.2,
    max_reprojection_error: float = 12.0,
) -> bool:
    if result.homography is None:
        return False
    if result.inlier_count < min_inliers:
        return False
    if result.inlier_ratio < min_inlier_ratio:
        return False
    if not np.isfinite(result.reprojection_error):
        return False
    return result.reprojection_error <= max_reprojection_error


def _should_prefer_simpler_transform(
    homography_result: RegistrationResult,
    affine_result: RegistrationResult,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> bool:
    spread_ratio = min(_point_spread_ratio(src_pts), _point_spread_ratio(dst_pts))
    if spread_ratio < 0.05:
        return True
    if affine_result.inlier_count >= homography_result.inlier_count and (
        affine_result.reprojection_error <= homography_result.reprojection_error * 1.2
    ):
        return True
    if affine_result.inlier_ratio >= homography_result.inlier_ratio + 0.08 and (
        affine_result.reprojection_error <= homography_result.reprojection_error * 1.5
    ):
        return True
    return False


def _select_preferred_fallback(
    homography_result: RegistrationResult,
    candidates: list[RegistrationResult],
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> Optional[RegistrationResult]:
    usable = [candidate for candidate in candidates if _candidate_is_usable_for_method(candidate)]
    if not usable:
        return None

    spread_ratio = min(_point_spread_ratio(src_pts), _point_spread_ratio(dst_pts))
    ranked = sorted(usable, key=_candidate_rank_key)
    best = ranked[0]
    if spread_ratio < 0.05:
        return best
    if _should_prefer_simpler_transform(homography_result, best, src_pts, dst_pts):
        return best
    return None


def _best_usable_candidate(candidates: list[RegistrationResult]) -> Optional[RegistrationResult]:
    usable = [candidate for candidate in candidates if _candidate_is_usable_for_method(candidate)]
    if not usable:
        return None
    return sorted(usable, key=_candidate_rank_key)[0]


def _candidate_is_usable_for_method(result: RegistrationResult) -> bool:
    if result.method == "translation":
        return _candidate_is_usable(
            result,
            min_inliers=6,
            min_inlier_ratio=0.3,
            max_reprojection_error=8.0,
        )
    return _candidate_is_usable(result)


def _candidate_rank_key(result: RegistrationResult) -> tuple[float, float, float]:
    method_priority = {
        "affine_full": 0.0,
        "affine_partial": 1.0,
        "translation": 2.0,
    }.get(result.method, 3.0)
    return (
        method_priority,
        -float(result.inlier_ratio),
        float(result.reprojection_error),
    )


def _is_homography_plausible(
    homography: Optional[np.ndarray],
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> bool:
    if homography is None or not np.isfinite(homography).all():
        return False

    spread_ratio = min(_point_spread_ratio(src_pts), _point_spread_ratio(dst_pts))
    if spread_ratio < 0.012:
        return False

    src = src_pts.reshape(-1, 2)
    warped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    if not np.isfinite(warped).all():
        return False

    src_w = max(float(src[:, 0].max() - src[:, 0].min()), 1.0)
    src_h = max(float(src[:, 1].max() - src[:, 1].min()), 1.0)
    warped_w = max(float(warped[:, 0].max() - warped[:, 0].min()), 1.0)
    warped_h = max(float(warped[:, 1].max() - warped[:, 1].min()), 1.0)
    bbox_area_ratio = (warped_w * warped_h) / (src_w * src_h)
    if bbox_area_ratio < 0.15 or bbox_area_ratio > 6.0:
        return False

    perspective_strength = abs(float(homography[2, 0])) * src_w + abs(float(homography[2, 1])) * src_h
    if perspective_strength > 0.18:
        return False

    bbox_corners = np.array(
        [
            [src[:, 0].min(), src[:, 1].min(), 1.0],
            [src[:, 0].max(), src[:, 1].min(), 1.0],
            [src[:, 0].max(), src[:, 1].max(), 1.0],
            [src[:, 0].min(), src[:, 1].max(), 1.0],
        ],
        dtype=np.float32,
    )
    denominators = (homography @ bbox_corners.T).T[:, 2]
    if np.any(np.abs(denominators) < 0.35):
        return False

    return True


def _point_spread_ratio(points: np.ndarray) -> float:
    pts = points.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 3:
        return 0.0
    centered = pts - pts.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(pts) - 1, 1)
    eigvals = np.linalg.eigvalsh(covariance)
    major = float(max(eigvals.max(), 1e-6))
    minor = float(max(eigvals.min(), 0.0))
    return minor / major


def _point_reprojection_errors(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    src = src_pts.reshape(-1, 2)
    dst = dst_pts.reshape(-1, 2)
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    diff = projected - dst
    return np.sqrt((diff ** 2).sum(axis=1))


def _merge_fallback_reasons(existing: Optional[str], new_reason: str) -> str:
    if not existing:
        return new_reason
    if new_reason in existing.split("|"):
        return existing
    return f"{existing}|{new_reason}"
