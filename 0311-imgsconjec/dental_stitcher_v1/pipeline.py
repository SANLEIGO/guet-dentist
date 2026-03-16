from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from dental_stitcher_v1.blending import blend_images
from dental_stitcher_v1.diagnostics import StepDiagnostics, StitchDiagnostics
from dental_stitcher_v1.features import extract_features, match_features
from dental_stitcher_v1.io_utils import compute_image_metrics, normalize_image
from dental_stitcher_v1.registration import MatchDiagnostics, estimate_transform
from dental_stitcher_v1.segmentation import fallback_full_mask, segment_teeth
from dental_stitcher_v1.visualization import render_matches


@dataclass
class StitchOutputs:
    stitched: Optional[np.ndarray]
    diagnostics: StitchDiagnostics
    mask_overlay: list[np.ndarray]
    match_visualization: np.ndarray


PIPELINE_VERSION = "v1"


def run_pipeline(
    images: list[np.ndarray],
    feature_method: str = "orb",
    use_deep_segmentation: bool = False,
) -> StitchOutputs:
    logs: list[str] = []

    if len(images) < 2:
        diagnostics = StitchDiagnostics(
            pipeline_version=PIPELINE_VERSION,
            segmentation=StepDiagnostics(False, {"reason": "insufficient_images"}),
            features=StepDiagnostics(False, {"reason": "insufficient_images"}),
            registration=StepDiagnostics(False, {"reason": "insufficient_images"}),
            blending=StepDiagnostics(False, {"reason": "insufficient_images"}),
            logs=["Need at least two images."],
        )
        placeholder = images[0] if images else np.zeros((100, 100, 3), dtype=np.uint8)
        return StitchOutputs(
            stitched=None,
            diagnostics=diagnostics,
            mask_overlay=[placeholder],
            match_visualization=placeholder,
        )

    normalized = [normalize_image(img) for img in images]
    metrics = [compute_image_metrics(img) for img in normalized]
    logs.append(f"Loaded {len(normalized)} images.")

    seg_results = []
    per_image = []
    for idx, img in enumerate(normalized):
        seg_result = segment_teeth(img, use_deep=use_deep_segmentation)
        if cv2.countNonZero(seg_result.mask) == 0:
            seg_result = fallback_full_mask(img)
        seg_results.append(seg_result)
        sharpness, exposure = metrics[idx]
        per_image.append(
            {
                "index": idx,
                "sharpness": sharpness,
                "exposure": exposure,
                "mask_coverage": float(cv2.countNonZero(seg_result.mask) / seg_result.mask.size),
                "segmentation_method": seg_result.method,
                "segmentation_fallback": seg_result.fallback_reason,
            }
        )

    seg_diag = StepDiagnostics(
        True,
        {"method": seg_results[0].method, "count": len(seg_results)},
        seg_results[0].fallback_reason,
    )

    gray0 = cv2.cvtColor(normalized[0], cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(normalized[1], cv2.COLOR_BGR2GRAY)

    features_result = extract_features(gray0, seg_results[0].mask, feature_method)
    features_fallback = None
    if features_result.descriptors is None or len(features_result.keypoints) < 8:
        features_result = extract_features(gray0, seg_results[0].mask, "orb")
        features_fallback = "feature_fallback_orb"

    feat_diag = StepDiagnostics(True, {"method": features_result.method}, features_fallback or features_result.fallback_reason)

    features_target = extract_features(gray1, seg_results[1].mask, features_result.method)
    matches = match_features(features_result.descriptors, features_target.descriptors, features_result.method)
    match_pts0, match_pts1 = _collect_match_points(features_result, features_target, matches.matches)

    reg_result = estimate_transform(features_result.keypoints, features_target.keypoints, matches.matches)
    reg_diag = StepDiagnostics(
        reg_result.homography is not None,
        {
            "method": reg_result.method,
            "inliers": reg_result.inlier_count,
            "inlier_ratio": round(reg_result.inlier_ratio, 3),
            "reprojection_error": round(reg_result.reprojection_error, 3),
        },
        reg_result.fallback_reason,
    )

    per_pair = [
        {
            "pair": [0, 1],
            "matches": len(matches.matches),
            "inliers": reg_result.inlier_count,
            "inlier_ratio": round(reg_result.inlier_ratio, 3),
            "reprojection_error": round(reg_result.reprojection_error, 3),
        }
    ]

    stitched = None
    blend_diag = StepDiagnostics(False, {"reason": "registration_failed"})
    if reg_result.homography is not None:
        stitched = blend_images(normalized[0], normalized[1], reg_result.homography)
        blend_diag = StepDiagnostics(True, {"method": "feather"})

    diagnostics = StitchDiagnostics(
        pipeline_version=PIPELINE_VERSION,
        segmentation=seg_diag,
        features=feat_diag,
        registration=reg_diag,
        blending=blend_diag,
        metrics={"inputs": [{"sharpness": m[0], "exposure": m[1]} for m in metrics]},
        per_image=per_image,
        per_pair=per_pair,
        logs=logs,
    )

    mask_overlay = [seg.overlay for seg in seg_results]
    match_vis = render_matches(normalized[0], normalized[1], match_pts0, match_pts1)

    return StitchOutputs(
        stitched=stitched,
        diagnostics=diagnostics,
        mask_overlay=mask_overlay,
        match_visualization=match_vis,
    )


def _collect_match_points(
    features_a, features_b, matches: list[cv2.DMatch]
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not matches:
        return None, None
    pts0 = np.float32([features_a.keypoints[m.queryIdx].pt for m in matches])
    pts1 = np.float32([features_b.keypoints[m.trainIdx].pt for m in matches])
    return pts0, pts1
