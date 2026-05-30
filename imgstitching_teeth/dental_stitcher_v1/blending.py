from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import cv2
import numpy as np


@dataclass
class BlendResult:
    image: np.ndarray
    mask: np.ndarray
    warp_mode: str
    control_point_count: int = 0
    fallback_reason: Optional[str] = None


def blend_images(
    base: np.ndarray,
    overlay: np.ndarray,
    homography: Optional[np.ndarray],
    feather_width: int = 30,
    base_mask: Optional[np.ndarray] = None,
    overlay_mask: Optional[np.ndarray] = None,
    base_points: Optional[np.ndarray] = None,
    overlay_points: Optional[np.ndarray] = None,
) -> BlendResult:
    if homography is None:
        fallback_mask = _normalize_mask_input(base_mask, base.shape[:2])
        if fallback_mask is None:
            fallback_mask = _nonzero_content_mask(base)
        return BlendResult(
            image=base.copy(),
            mask=fallback_mask,
            warp_mode="identity",
            fallback_reason="missing_transform",
        )

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
    y0 = -min_y
    x0 = -min_x

    base_mask_input = _normalize_mask_input(base_mask, base.shape[:2])
    overlay_mask_input = _normalize_mask_input(overlay_mask, overlay.shape[:2])
    if base_mask_input is None:
        base_mask_input = _nonzero_content_mask(base)
    if overlay_mask_input is None:
        overlay_mask_input = _nonzero_content_mask(overlay)

    warped_overlay, warped_overlay_mask, warp_mode, control_point_count, fallback_reason = _warp_overlay(
        overlay=overlay,
        overlay_mask=overlay_mask_input,
        warp_matrix=warp_matrix,
        canvas_size=(canvas_w, canvas_h),
        base_offset=(x0, y0),
        base_points=base_points,
        overlay_points=overlay_points,
    )

    base_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    base_canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
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
        return BlendResult(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            mask=np.zeros((1, 1), dtype=np.uint8),
            warp_mode=warp_mode,
            control_point_count=control_point_count,
            fallback_reason=fallback_reason,
        )

    ys, xs = np.where(occupancy)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    blended = blended[y1:y2, x1:x2]
    occupancy_mask = (occupancy[y1:y2, x1:x2].astype(np.uint8) * 255)

    return BlendResult(
        image=np.clip(blended, 0, 255).astype(np.uint8),
        mask=occupancy_mask,
        warp_mode=warp_mode,
        control_point_count=control_point_count,
        fallback_reason=fallback_reason,
    )


def _warp_overlay(
    overlay: np.ndarray,
    overlay_mask: np.ndarray,
    warp_matrix: np.ndarray,
    canvas_size: tuple[int, int],
    base_offset: tuple[int, int],
    base_points: Optional[np.ndarray],
    overlay_points: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, str, int, Optional[str]]:
    canvas_w, canvas_h = canvas_size
    global_overlay = cv2.warpPerspective(
        overlay,
        warp_matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    ).astype(np.float32)
    global_mask = cv2.warpPerspective(
        overlay_mask,
        warp_matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if base_points is None or overlay_points is None:
        return global_overlay, global_mask, "global_projective", 0, None

    piecewise = _warp_overlay_piecewise(
        overlay=overlay,
        overlay_mask=overlay_mask,
        warp_matrix=warp_matrix,
        canvas_size=(canvas_h, canvas_w),
        base_offset=base_offset,
        base_points=base_points,
        overlay_points=overlay_points,
    )
    if piecewise is None:
        return global_overlay, global_mask, "global_projective", 0, "piecewise_unavailable"

    warped_overlay, warped_mask, control_point_count = piecewise
    return warped_overlay, warped_mask, "piecewise_affine", control_point_count, None


def _warp_overlay_piecewise(
    overlay: np.ndarray,
    overlay_mask: np.ndarray,
    warp_matrix: np.ndarray,
    canvas_size: tuple[int, int],
    base_offset: tuple[int, int],
    base_points: np.ndarray,
    overlay_points: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
    control_src, control_dst = _prepare_control_points(
        overlay_shape=overlay.shape[:2],
        overlay_mask=overlay_mask,
        warp_matrix=warp_matrix,
        base_offset=base_offset,
        base_points=base_points,
        overlay_points=overlay_points,
    )
    if control_src is None or control_dst is None or len(control_src) < 12:
        return None
    if min(_point_spread_ratio(control_src), _point_spread_ratio(control_dst)) < 0.002:
        return None

    try:
        from skimage.transform import PiecewiseAffineTransform, warp
    except Exception:
        return None

    transform = PiecewiseAffineTransform()
    if not transform.estimate(control_src, control_dst):
        return None

    canvas_h, canvas_w = canvas_size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warped_overlay = warp(
            overlay.astype(np.float32),
            inverse_map=transform.inverse,
            output_shape=(canvas_h, canvas_w),
            order=1,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        ).astype(np.float32)
        warped_mask = warp(
            overlay_mask.astype(np.float32),
            inverse_map=transform.inverse,
            output_shape=(canvas_h, canvas_w),
            order=0,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        )
    warped_mask = np.where(warped_mask > 127.5, 255, 0).astype(np.uint8)
    return warped_overlay, warped_mask, int(len(control_src))


def _prepare_control_points(
    overlay_shape: tuple[int, int],
    overlay_mask: np.ndarray,
    warp_matrix: np.ndarray,
    base_offset: tuple[int, int],
    base_points: np.ndarray,
    overlay_points: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if base_points is None or overlay_points is None:
        return None, None

    src = np.asarray(overlay_points, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(base_points, dtype=np.float32).reshape(-1, 2)
    if len(src) != len(dst) or len(src) < 8:
        return None, None

    finite_mask = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite_mask]
    dst = dst[finite_mask]
    if len(src) < 8:
        return None, None

    if len(src) > 96:
        keep = np.linspace(0, len(src) - 1, 96, dtype=int)
        src = src[keep]
        dst = dst[keep]

    dst = dst + np.array(base_offset, dtype=np.float32)
    anchor_src = _build_anchor_points(overlay_shape, overlay_mask)
    anchor_dst = cv2.perspectiveTransform(anchor_src.reshape(-1, 1, 2), warp_matrix).reshape(-1, 2)

    src = np.vstack([src, anchor_src])
    dst = np.vstack([dst, anchor_dst])

    unique_src: list[np.ndarray] = []
    unique_dst: list[np.ndarray] = []
    seen: set[tuple[int, int, int, int]] = set()
    for src_point, dst_point in zip(src, dst):
        key = (
            int(round(float(src_point[0]))),
            int(round(float(src_point[1]))),
            int(round(float(dst_point[0]))),
            int(round(float(dst_point[1]))),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_src.append(src_point)
        unique_dst.append(dst_point)

    if len(unique_src) < 12:
        return None, None

    return np.asarray(unique_src, dtype=np.float32), np.asarray(unique_dst, dtype=np.float32)


def _build_anchor_points(overlay_shape: tuple[int, int], overlay_mask: np.ndarray) -> np.ndarray:
    height, width = overlay_shape
    bbox = _mask_bbox(overlay_mask)
    if bbox is None:
        left, top, right, bottom = 0, 0, width - 1, height - 1
    else:
        left, top, right, bottom = bbox

    cx = (left + right) * 0.5
    cy = (top + bottom) * 0.5
    points = np.array(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
            [cx, top],
            [cx, bottom],
            [left, cy],
            [right, cy],
            [cx, cy],
        ],
        dtype=np.float32,
    )
    return points


def _mask_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _point_spread_ratio(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    centered = pts - pts.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(pts) - 1, 1)
    eigvals = np.linalg.eigvalsh(covariance)
    major = float(max(eigvals.max(), 1e-6))
    minor = float(max(eigvals.min(), 0.0))
    return minor / major


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
