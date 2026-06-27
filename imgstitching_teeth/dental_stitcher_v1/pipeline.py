from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from dental_stitcher_v1.blending import blend_images
from dental_stitcher_v1.diagnostics import StepDiagnostics, StitchDiagnostics
from dental_stitcher_v1.features import extract_features, match_features
from dental_stitcher_v1.io_utils import compute_image_metrics, normalize_image
from dental_stitcher_v1.registration import estimate_transform
from dental_stitcher_v1.segmentation import SegmentationResult, fallback_full_mask, segment_teeth
from dental_stitcher_v1.visualization import render_matches


@dataclass
class StitchOutputs:
    stitched: Optional[np.ndarray]
    stitched_mask: Optional[np.ndarray]
    diagnostics: StitchDiagnostics
    mask_overlay: list[np.ndarray]
    match_visualization: np.ndarray


PIPELINE_VERSION = "v1"
OUTPUT_MODE = "teeth_only"


def run_pipeline(
    images: list[np.ndarray],
    feature_method: str = "sift",
    seg_results: Optional[list[SegmentationResult]] = None,
    enable_auto_calibration: bool = True  # 新增参数
) -> StitchOutputs:
    logs: list[str] = []

    if len(images) == 0:
        diagnostics = StitchDiagnostics(
            pipeline_version=PIPELINE_VERSION,
            segmentation=StepDiagnostics(False, {"reason": "insufficient_images"}),
            features=StepDiagnostics(False, {"reason": "insufficient_images"}),
            registration=StepDiagnostics(False, {"reason": "insufficient_images"}),
            blending=StepDiagnostics(False, {"reason": "insufficient_images"}),
            output_mode=OUTPUT_MODE,
            logs=["Need at least one image."],
        )
        placeholder = images[0] if images else np.zeros((100, 100, 3), dtype=np.uint8)
        return StitchOutputs(
            stitched=None,
            stitched_mask=None,
            diagnostics=diagnostics,
            mask_overlay=[placeholder],
            match_visualization=placeholder,
        )

    # === 新增：自动标定前置模块 ===
    calibration_result = None
    if enable_auto_calibration and len(images) >= 2:
        logs.append("Starting auto-calibration...")

        from dental_stitcher_v1.calibration.auto_calibrator import auto_calibrate_pipeline

        try:
            calibration_result = auto_calibrate_pipeline(
                images,
                seg_method="alphadent",
                enable_iterative=True  # 启用迭代标定（两轮优化）
            )

            if calibration_result.success:
                logs.append(f"Calibration successful. RMSE improvement: {calibration_result.quality_metrics['rmse_improvement']:.2f}px")
                images = calibration_result.undistorted_images  # 使用校正后的图像
            else:
                # 标定失败，详细提示原因
                failure_reason = calibration_result.diagnostics.failure_reason or "unknown"

                # 解析实例提取失败原因
                instance_diag = calibration_result.diagnostics.instance_extraction
                if not instance_diag.success and instance_diag.fallback_reason:
                    failure_detail = instance_diag.fallback_reason

                    # 解析详细消息（格式：reason|message）
                    if "|" in failure_detail:
                        reason_code, detailed_msg = failure_detail.split("|", 1)
                        logs.append(f"Calibration failed: {reason_code}")
                        logs.append(f"Details: {detailed_msg}")
                    else:
                        logs.append(f"Calibration failed: {failure_detail}")
                else:
                    logs.append(f"Calibration failed: {failure_reason}")

                logs.append("Auto-calibration requirements:")
                logs.append("  1. Use AlphaDent (YOLOv8) segmentation (not U-Net)")
                logs.append("  2. Each image must have ≥4 detected tooth instances")
                logs.append("  3. Images should not have severe distortion")
                logs.append("Suggestion: Disable auto-calibration and try direct stitching")

                # 不降级，返回失败诊断
                diagnostics = StitchDiagnostics(
                    pipeline_version=PIPELINE_VERSION,
                    segmentation=StepDiagnostics(False, {"reason": "calibration_failed"}),
                    features=StepDiagnostics(False, {"reason": "calibration_failed"}),
                    registration=StepDiagnostics(False, {"reason": "calibration_failed"}),
                    blending=StepDiagnostics(False, {"reason": "calibration_failed"}),
                    output_mode=OUTPUT_MODE,
                    quality_gate={"calibration": calibration_result.diagnostics.to_dict()},
                    logs=logs,
                )
                placeholder = images[0] if images else np.zeros((100, 100, 3), dtype=np.uint8)
                return StitchOutputs(
                    stitched=None,
                    stitched_mask=None,
                    diagnostics=diagnostics,
                    mask_overlay=[placeholder],
                    match_visualization=placeholder,
                )
        except Exception as e:
            # 标定过程异常，停止流程
            logs.append(f"Calibration exception: {str(e)}")
            logs.append("Auto-calibration encountered an error and cannot continue.")
            logs.append("Suggestion: Disable auto-calibration and try direct stitching")

            diagnostics = StitchDiagnostics(
                pipeline_version=PIPELINE_VERSION,
                segmentation=StepDiagnostics(False, {"reason": "calibration_exception", "error": str(e)}),
                features=StepDiagnostics(False, {"reason": "calibration_exception"}),
                registration=StepDiagnostics(False, {"reason": "calibration_exception"}),
                blending=StepDiagnostics(False, {"reason": "calibration_exception"}),
                output_mode=OUTPUT_MODE,
                logs=logs,
            )
            placeholder = images[0] if images else np.zeros((100, 100, 3), dtype=np.uint8)
            return StitchOutputs(
                stitched=None,
                stitched_mask=None,
                diagnostics=diagnostics,
                mask_overlay=[placeholder],
                match_visualization=placeholder,
            )

    # === 原有流程（使用校正后的图像） ===
    normalized = [normalize_image(img) for img in images]
    metrics = [compute_image_metrics(img) for img in normalized]
    logs.append(f"Loaded {len(normalized)} images.")

    prepared_seg_results, segmentation_source = _prepare_segmentation_inputs(normalized, seg_results, logs)
    extracted_images = [_extract_teeth_image(img, seg.mask) for img, seg in zip(normalized, prepared_seg_results)]
    mask_overlay = [seg.overlay for seg in prepared_seg_results]

    if len(normalized) == 1:
        return _run_single_image_pipeline(
            images=normalized,
            extracted_images=extracted_images,
            seg_results=prepared_seg_results,
            metrics=metrics,
            segmentation_source=segmentation_source,
            logs=logs,
            mask_overlay=mask_overlay,
        )

    if len(normalized) == 2:
        return _run_pair_pipeline(
            images=normalized,
            extracted_images=extracted_images,
            seg_results=prepared_seg_results,
            metrics=metrics,
            feature_method=feature_method,
            segmentation_source=segmentation_source,
            logs=logs,
            mask_overlay=mask_overlay,
            pair_indices=[0, 1],
        )

    return _run_multi_pipeline(
        images=normalized,
        extracted_images=extracted_images,
        seg_results=prepared_seg_results,
        metrics=metrics,
        feature_method=feature_method,
        segmentation_source=segmentation_source,
        logs=logs,
        mask_overlay=mask_overlay,
    )


def _run_single_image_pipeline(
    images: list[np.ndarray],
    extracted_images: list[np.ndarray],
    seg_results: list[SegmentationResult],
    metrics: list[tuple[float, float]],
    segmentation_source: str,
    logs: list[str],
    mask_overlay: list[np.ndarray],
) -> StitchOutputs:
    per_image = _build_per_image_diagnostics(seg_results, metrics, segmentation_source)
    seg_result = seg_results[0]
    stitched = extracted_images[0]
    stitched_mask = seg_result.mask.copy()
    mask_coverage = float(cv2.countNonZero(stitched_mask) / stitched_mask.size) if stitched_mask.size else 0.0
    used_fallback_mask = bool(per_image[0].get("used_fallback_mask"))

    quality_gate = {
        "gate_passed": True,
        "confidence_level": "medium" if used_fallback_mask else "high",
        "single_image_panorama_mode": True,
        "accepted_indices": [0],
        "skipped_indices": [],
        "fail_reasons": [],
        "degrade_reasons": ["segmentation_fallback_mask"] if used_fallback_mask else [],
        "stitched_mask_coverage": mask_coverage,
        "mask_overlap_ratio": 1.0,
        "match_count": 0,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "reprojection_error": 0.0,
        "warp_area_ratio": 1.0,
        "warp_canvas_ratio": 1.0,
        "warp_perspective_strength": 0.0,
        "region_mode": "single_image_panorama",
        "output_mode": OUTPUT_MODE,
        "strict_teeth_only": not used_fallback_mask,
    }

    diagnostics = StitchDiagnostics(
        pipeline_version=PIPELINE_VERSION,
        segmentation=StepDiagnostics(True, {"method": seg_result.method, "count": 1, "output_mode": OUTPUT_MODE}),
        features=StepDiagnostics(True, {"method": "single_image_passthrough", "match_count": 0}),
        registration=StepDiagnostics(True, {"method": "identity", "reason": "single_image_panorama"}),
        blending=StepDiagnostics(True, {"method": "single_image_passthrough", "output_mode": OUTPUT_MODE}),
        segmentation_source=segmentation_source,
        output_mode=OUTPUT_MODE,
        quality_gate=quality_gate,
        metrics={
            "inputs": [{"sharpness": m[0], "exposure": m[1]} for m in metrics],
            "sequence_mode": "single_image_panorama",
        },
        per_image=per_image,
        per_pair=[],
        logs=logs + ["Single image panorama mode: stitching skipped."],
    )

    return StitchOutputs(
        stitched=stitched,
        stitched_mask=stitched_mask,
        diagnostics=diagnostics,
        mask_overlay=mask_overlay,
        match_visualization=images[0],
    )


def _run_multi_pipeline(
    images: list[np.ndarray],
    extracted_images: list[np.ndarray],
    seg_results: list[SegmentationResult],
    metrics: list[tuple[float, float]],
    feature_method: str,
    segmentation_source: str,
    logs: list[str],
    mask_overlay: list[np.ndarray],
) -> StitchOutputs:
    current_image = images[0]
    current_extracted = extracted_images[0]
    current_seg = seg_results[0]
    current_mask = seg_results[0].mask.copy()
    current_index = 0
    accepted_indices = [0]
    skipped_indices: list[int] = []
    step_summaries: list[dict] = []
    pair_diagnostics: list[dict] = []
    final_match_vis = current_image

    overall_confidence = "high"
    overall_gate_passed = True
    overall_fail_reasons: list[str] = []
    overall_degrade_reasons: list[str] = []

    for idx in range(1, len(images)):
        stage_output = _run_pair_pipeline(
            images=[current_image, images[idx]],
            extracted_images=[current_extracted, extracted_images[idx]],
            seg_results=[current_seg, seg_results[idx]],
            metrics=[compute_image_metrics(current_image), metrics[idx]],
            feature_method=feature_method,
            segmentation_source=segmentation_source,
            logs=[f"Sequential stage: {current_index} -> {idx}"],
            mask_overlay=[current_seg.overlay, seg_results[idx].overlay],
            pair_indices=[current_index, idx],
        )

        stage_diag = stage_output.diagnostics.to_dict()
        stage_gate = stage_diag.get("quality_gate", {})
        stage_pair = stage_diag.get("per_pair", [{}])[0] if stage_diag.get("per_pair") else {}
        stage_pair["accepted"] = stage_output.stitched is not None
        pair_diagnostics.append(stage_pair)
        final_match_vis = stage_output.match_visualization

        step_summary = {
            "reference_index": current_index,
            "candidate_index": idx,
            "accepted": stage_output.stitched is not None,
            "confidence_level": stage_gate.get("confidence_level", "low"),
            "gate_passed": stage_gate.get("gate_passed", False),
            "fail_reasons": stage_gate.get("fail_reasons", []),
            "degrade_reasons": stage_gate.get("degrade_reasons", []),
            "stitched_mask_coverage": stage_gate.get("stitched_mask_coverage", 0.0),
        }
        step_summaries.append(step_summary)

        if stage_output.stitched is not None:
            current_image = stage_output.stitched
            current_mask = (
                stage_output.stitched_mask.copy()
                if stage_output.stitched_mask is not None
                else _extract_mask_from_image(current_image)
            )
            current_seg = _segmentation_from_mask(current_image, current_mask, method="stitched_teeth_only")
            current_extracted = current_image
            current_index = idx
            accepted_indices.append(idx)
        else:
            skipped_indices.append(idx)

        if not stage_gate.get("gate_passed", False):
            overall_gate_passed = False
        overall_fail_reasons.extend(stage_gate.get("fail_reasons", []))
        overall_degrade_reasons.extend(stage_gate.get("degrade_reasons", []))
        overall_confidence = _merge_confidence(overall_confidence, stage_gate.get("confidence_level", "low"))

    per_image = _build_per_image_diagnostics(seg_results, metrics, segmentation_source)
    overall_quality_gate = {
        "gate_passed": overall_gate_passed,
        "confidence_level": overall_confidence,
        "accepted_indices": accepted_indices,
        "skipped_indices": skipped_indices,
        "step_count": len(step_summaries),
        "steps": step_summaries,
        "fail_reasons": sorted(set(overall_fail_reasons)),
        "degrade_reasons": sorted(set(overall_degrade_reasons)),
        "region_mode": "single_region_sequence",
        "output_mode": OUTPUT_MODE,
    }

    diagnostics = StitchDiagnostics(
        pipeline_version=PIPELINE_VERSION,
        segmentation=StepDiagnostics(True, {"method": seg_results[0].method, "count": len(seg_results)}),
        features=StepDiagnostics(True, {"method": feature_method, "mode": "sequential_multi_image"}),
        registration=StepDiagnostics(len(accepted_indices) > 1, {"accepted_steps": len(accepted_indices) - 1, "total_steps": len(images) - 1}),
        blending=StepDiagnostics(
            len(accepted_indices) > 1,
            {"accepted_images": len(accepted_indices), "skipped_images": len(skipped_indices), "output_mode": OUTPUT_MODE},
        ),
        segmentation_source=segmentation_source,
        output_mode=OUTPUT_MODE,
        quality_gate=overall_quality_gate,
        metrics={
            "inputs": [{"sharpness": m[0], "exposure": m[1]} for m in metrics],
            "sequence_mode": "manual_order_multi",
        },
        per_image=per_image,
        per_pair=pair_diagnostics,
        logs=logs + [f"Accepted images: {accepted_indices}", f"Skipped images: {skipped_indices}"],
    )

    return StitchOutputs(
        stitched=current_image if len(accepted_indices) > 1 else None,
        stitched_mask=current_mask if len(accepted_indices) > 1 else None,
        diagnostics=diagnostics,
        mask_overlay=mask_overlay,
        match_visualization=final_match_vis,
    )


def _run_pair_pipeline(
    images: list[np.ndarray],
    extracted_images: list[np.ndarray],
    seg_results: list[SegmentationResult],
    metrics: list[tuple[float, float]],
    feature_method: str,
    segmentation_source: str,
    logs: list[str],
    mask_overlay: list[np.ndarray],
    pair_indices: list[int],
) -> StitchOutputs:
    per_image = _build_per_image_diagnostics(seg_results, metrics, segmentation_source, indices=pair_indices)

    seg_diag = StepDiagnostics(
        True,
        {"method": seg_results[0].method, "count": len(seg_results), "output_mode": OUTPUT_MODE},
        seg_results[0].fallback_reason,
    )

    gray0 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)

    features_result, features_target, effective_method, features_fallback = _select_feature_pair_for_matching(
        gray0,
        seg_results[0].mask,
        gray1,
        seg_results[1].mask,
        feature_method,
    )

    matches = match_features(features_result.descriptors, features_target.descriptors, effective_method)
    match_pts0, match_pts1 = _collect_match_points(features_result, features_target, matches.matches)

    feat_diag = StepDiagnostics(
        features_result.descriptors is not None and features_target.descriptors is not None,
        {
            "method": effective_method,
            "preferred_method": feature_method,
            "keypoints_image_a": len(features_result.keypoints),
            "keypoints_image_b": len(features_target.keypoints),
            "match_count": len(matches.matches),
        },
        features_fallback or features_result.fallback_reason or features_target.fallback_reason,
    )

    reg_result = estimate_transform(features_result.keypoints, features_target.keypoints, matches.matches)
    blend_homography = _invert_homography(reg_result.homography)
    quality_gate = _evaluate_pair_quality(
        seg_results=seg_results,
        per_image=per_image,
        match_count=len(matches.matches),
        reg_result=reg_result,
        blend_homography=blend_homography,
    )
    reg_diag = StepDiagnostics(
        reg_result.homography is not None,
        {
            "method": reg_result.method,
            "inliers": reg_result.inlier_count,
            "inlier_ratio": round(reg_result.inlier_ratio, 3),
            "reprojection_error": round(reg_result.reprojection_error, 3),
            "mask_overlap_ratio": round(quality_gate["mask_overlap_ratio"], 3),
            "confidence_level": quality_gate["confidence_level"],
            "gate_passed": quality_gate["gate_passed"],
        },
        reg_result.fallback_reason,
    )

    per_pair = [
        {
            "pair": pair_indices,
            "matches": len(matches.matches),
            "inliers": reg_result.inlier_count,
            "inlier_ratio": round(reg_result.inlier_ratio, 3),
            "reprojection_error": round(reg_result.reprojection_error, 3),
            "mask_overlap_ratio": round(quality_gate["mask_overlap_ratio"], 3),
            "warp_area_ratio": round(quality_gate["warp_area_ratio"], 3),
            "warp_canvas_ratio": round(quality_gate["warp_canvas_ratio"], 3),
            "warp_perspective_strength": round(quality_gate["warp_perspective_strength"], 3),
            "gate_passed": quality_gate["gate_passed"],
            "confidence_level": quality_gate["confidence_level"],
            "degrade_reasons": quality_gate["degrade_reasons"],
            "fail_reasons": quality_gate["fail_reasons"],
        }
    ]

    stitched = None
    stitched_mask = None
    blend_diag = StepDiagnostics(False, {"reason": "registration_failed", "output_mode": OUTPUT_MODE})
    if blend_homography is not None and quality_gate["gate_passed"]:
        inlier_match_pts0, inlier_match_pts1 = _filter_inlier_points(
            match_pts0,
            match_pts1,
            reg_result.inlier_mask,
        )
        blend_result = blend_images(
            extracted_images[0],
            extracted_images[1],
            blend_homography,
            base_mask=seg_results[0].mask,
            overlay_mask=seg_results[1].mask,
            base_points=inlier_match_pts0,
            overlay_points=inlier_match_pts1,
        )
        stitched = blend_result.image
        stitched_mask = blend_result.mask
        quality_gate["stitched_mask_coverage"] = float(cv2.countNonZero(stitched_mask) / stitched_mask.size) if stitched_mask.size else 0.0
        if any(item.get("used_fallback_mask") for item in per_image):
            quality_gate.setdefault("degrade_reasons", []).append("degraded_teeth_only_fallback")
        blend_diag = StepDiagnostics(
            True,
            {
                "method": "masked_feather",
                "warp_mode": blend_result.warp_mode,
                "control_point_count": blend_result.control_point_count,
                "confidence_level": quality_gate["confidence_level"],
                "gate_passed": quality_gate["gate_passed"],
                "output_mode": OUTPUT_MODE,
                "strict_teeth_only": not any(item.get("used_fallback_mask") for item in per_image),
                "stitched_mask_coverage": round(quality_gate["stitched_mask_coverage"], 3),
            },
            blend_result.fallback_reason if blend_result.fallback_reason else (None if quality_gate["gate_passed"] else "low_confidence_blend"),
        )
        per_pair[0]["stitched_mask_coverage"] = round(quality_gate["stitched_mask_coverage"], 3)
        per_pair[0]["strict_teeth_only"] = not any(item.get("used_fallback_mask") for item in per_image)
        per_pair[0]["warp_mode"] = blend_result.warp_mode
        per_pair[0]["warp_control_points"] = blend_result.control_point_count
    elif blend_homography is not None:
        blend_diag = StepDiagnostics(
            False,
            {
                "reason": "quality_gate_rejected",
                "confidence_level": quality_gate["confidence_level"],
                "gate_passed": quality_gate["gate_passed"],
                "output_mode": OUTPUT_MODE,
            },
            "low_confidence_blend",
        )
        per_pair[0]["strict_teeth_only"] = not any(item.get("used_fallback_mask") for item in per_image)

    diagnostics = StitchDiagnostics(
        pipeline_version=PIPELINE_VERSION,
        segmentation=seg_diag,
        features=feat_diag,
        registration=reg_diag,
        blending=blend_diag,
        segmentation_source=segmentation_source,
        output_mode=OUTPUT_MODE,
        quality_gate=quality_gate,
        metrics={"inputs": [{"sharpness": m[0], "exposure": m[1]} for m in metrics]},
        per_image=per_image,
        per_pair=per_pair,
        logs=logs,
    )

    match_vis = render_matches(images[0], images[1], match_pts0, match_pts1)

    return StitchOutputs(
        stitched=stitched,
        stitched_mask=stitched_mask,
        diagnostics=diagnostics,
        mask_overlay=mask_overlay,
        match_visualization=match_vis,
    )


def _prepare_segmentation_inputs(
    normalized: list[np.ndarray],
    seg_results: Optional[list[SegmentationResult]],
    logs: list[str],
) -> tuple[list[SegmentationResult], str]:
    input_seg_results = seg_results
    provided_seg_results = input_seg_results is not None and len(input_seg_results) == len(normalized)
    segmentation_source = "frontend_session" if provided_seg_results else "pipeline_runtime"
    if input_seg_results is not None and len(input_seg_results) != len(normalized):
        logs.append("Provided seg_results length mismatch; falling back to runtime segmentation.")

    prepared_seg_results: list[SegmentationResult] = []
    for idx, img in enumerate(normalized):
        if provided_seg_results:
            seg_result = _prepare_segmentation_result(seg_results_input=input_seg_results, index=idx, image=img)
        else:
            seg_result = segment_teeth(img)
            if cv2.countNonZero(seg_result.mask) == 0:
                seg_result = fallback_full_mask(img)
        prepared_seg_results.append(seg_result)
    return prepared_seg_results, segmentation_source


def _select_feature_pair_for_matching(
    gray0: np.ndarray,
    mask0: np.ndarray,
    gray1: np.ndarray,
    mask1: np.ndarray,
    preferred_method: str,
) -> tuple:
    attempted_methods: list[str] = []
    for method in _candidate_feature_methods(preferred_method):
        attempted_methods.append(method)
        features_a = extract_features(gray0, mask0, method)
        features_b = extract_features(gray1, mask1, method)
        if features_a.descriptors is not None and features_b.descriptors is not None:
            fallback_reason = None
            if method != preferred_method:
                fallback_reason = f"feature_fallback_{preferred_method}_to_{method}"
            return features_a, features_b, method, fallback_reason

    fallback_method = _candidate_feature_methods(preferred_method)[-1]
    return (
        extract_features(gray0, mask0, fallback_method),
        extract_features(gray1, mask1, fallback_method),
        fallback_method,
        f"feature_unresolved_attempts_{'_'.join(attempted_methods)}",
    )


def _candidate_feature_methods(preferred_method: str) -> list[str]:
    ordered: list[str] = []
    for method in [preferred_method, "sift", "akaze", "orb"]:
        if method not in ordered:
            ordered.append(method)
    return ordered


def _build_per_image_diagnostics(
    seg_results: list[SegmentationResult],
    metrics: list[tuple[float, float]],
    segmentation_source: str,
    indices: Optional[list[int]] = None,
) -> list[dict]:
    if indices is None:
        indices = list(range(len(seg_results)))

    per_image = []
    for idx, seg_result, metric in zip(indices, seg_results, metrics):
        sharpness, exposure = metric
        mask_coverage = float(cv2.countNonZero(seg_result.mask) / seg_result.mask.size)
        per_image.append(
            {
                "index": idx,
                "sharpness": sharpness,
                "exposure": exposure,
                "mask_coverage": mask_coverage,
                "segmentation_method": seg_result.method,
                "segmentation_fallback": seg_result.fallback_reason,
                "segmentation_source": segmentation_source,
                "used_fallback_mask": bool(seg_result.fallback_reason),
                "output_mode": OUTPUT_MODE,
                "strict_teeth_only": not bool(seg_result.fallback_reason),
                "teeth_pixels": int(cv2.countNonZero(seg_result.mask)),
            }
        )
    return per_image


def _merge_confidence(current: str, new_value: str) -> str:
    order = {"high": 0, "medium": 1, "low": 2}
    return current if order.get(current, 2) >= order.get(new_value, 2) else new_value


def _evaluate_pair_quality(
    seg_results: list[SegmentationResult],
    per_image: list[dict],
    match_count: int,
    reg_result,
    blend_homography: Optional[np.ndarray],
) -> dict:
    degrade_reasons: list[str] = []
    fail_reasons: list[str] = []

    min_coverage = min(item["mask_coverage"] for item in per_image)
    if min_coverage < 0.08:
        degrade_reasons.append("low_mask_coverage")

    if any(item["used_fallback_mask"] for item in per_image):
        degrade_reasons.append("segmentation_fallback_mask")

    if match_count < 12:
        fail_reasons.append("very_low_match_count")
    elif match_count < 24:
        degrade_reasons.append("low_match_count")

    if reg_result.homography is None or blend_homography is None:
        fail_reasons.append("registration_failed")
        mask_overlap_ratio = 0.0
    else:
        if reg_result.inlier_count < 8:
            fail_reasons.append("low_inlier_count")
        elif reg_result.inlier_count < 16:
            degrade_reasons.append("limited_inlier_count")

        if reg_result.inlier_ratio < 0.2:
            fail_reasons.append("very_low_inlier_ratio")
        elif reg_result.inlier_ratio < 0.35:
            degrade_reasons.append("low_inlier_ratio")

        if np.isfinite(reg_result.reprojection_error):
            if reg_result.reprojection_error > 12.0:
                fail_reasons.append("high_reprojection_error")
            elif reg_result.reprojection_error > 6.0:
                degrade_reasons.append("elevated_reprojection_error")
        else:
            fail_reasons.append("invalid_reprojection_error")

        mask_overlap_ratio = _compute_mask_overlap_ratio(seg_results[0].mask, seg_results[1].mask, blend_homography)
        if mask_overlap_ratio < 0.02:
            fail_reasons.append("mask_overlap_too_small")
        elif mask_overlap_ratio < 0.08:
            degrade_reasons.append("limited_mask_overlap")

        geometry_metrics = _compute_warp_geometry_metrics(
            seg_results[0].mask.shape,
            seg_results[1].mask.shape,
            blend_homography,
        )
        if geometry_metrics["invalid"]:
            fail_reasons.append("invalid_warp_geometry")
        else:
            if geometry_metrics["area_ratio"] < 0.15 or geometry_metrics["area_ratio"] > 6.0:
                fail_reasons.append("warp_area_ratio_out_of_range")
            elif geometry_metrics["area_ratio"] < 0.33 or geometry_metrics["area_ratio"] > 3.0:
                degrade_reasons.append("warp_area_ratio_suspicious")

            if geometry_metrics["canvas_ratio"] > 9.0:
                fail_reasons.append("warp_canvas_too_large")
            elif geometry_metrics["canvas_ratio"] > 5.0:
                degrade_reasons.append("warp_canvas_expanded")

            if geometry_metrics["perspective_strength"] > 0.22:
                fail_reasons.append("warp_perspective_too_strong")
            elif geometry_metrics["perspective_strength"] > 0.10:
                degrade_reasons.append("warp_perspective_strong")
    if reg_result.homography is None or blend_homography is None:
        geometry_metrics = {
            "invalid": True,
            "area_ratio": 0.0,
            "canvas_ratio": 0.0,
            "perspective_strength": 0.0,
        }

    gate_passed = len(fail_reasons) == 0
    if fail_reasons:
        confidence_level = "low"
    elif degrade_reasons:
        confidence_level = "medium"
    else:
        confidence_level = "high"

    return {
        "gate_passed": gate_passed,
        "confidence_level": confidence_level,
        "degrade_reasons": degrade_reasons,
        "fail_reasons": fail_reasons,
        "mask_overlap_ratio": float(mask_overlap_ratio),
        "match_count": int(match_count),
        "inlier_count": int(reg_result.inlier_count),
        "inlier_ratio": float(reg_result.inlier_ratio),
        "reprojection_error": float(reg_result.reprojection_error),
        "warp_area_ratio": float(geometry_metrics["area_ratio"]),
        "warp_canvas_ratio": float(geometry_metrics["canvas_ratio"]),
        "warp_perspective_strength": float(geometry_metrics["perspective_strength"]),
        "output_mode": OUTPUT_MODE,
        "strict_teeth_only": not any(item["used_fallback_mask"] for item in per_image),
    }


def _compute_mask_overlap_ratio(base_mask: np.ndarray, overlay_mask: np.ndarray, homography: np.ndarray) -> float:
    base_binary = (base_mask > 0).astype(np.uint8)
    overlay_binary = (overlay_mask > 0).astype(np.uint8)
    warped_overlay = cv2.warpPerspective(
        overlay_binary,
        homography,
        (base_mask.shape[1], base_mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    intersection = np.logical_and(base_binary > 0, warped_overlay > 0).sum()
    union = np.logical_or(base_binary > 0, warped_overlay > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _compute_warp_geometry_metrics(
    base_shape: tuple[int, int],
    overlay_shape: tuple[int, int],
    homography: np.ndarray,
) -> dict:
    base_h, base_w = base_shape[:2]
    overlay_h, overlay_w = overlay_shape[:2]
    corners = np.array(
        [[0, 0], [overlay_w, 0], [overlay_w, overlay_h], [0, overlay_h]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    if not np.isfinite(warped).all():
        return {"invalid": True, "area_ratio": 0.0, "canvas_ratio": 0.0, "perspective_strength": 0.0}

    polygon_area = abs(float(cv2.contourArea(warped.astype(np.float32))))
    overlay_area = max(float(overlay_w * overlay_h), 1.0)

    min_x = min(0.0, float(warped[:, 0].min()))
    min_y = min(0.0, float(warped[:, 1].min()))
    max_x = max(float(base_w), float(warped[:, 0].max()))
    max_y = max(float(base_h), float(warped[:, 1].max()))
    canvas_area = max((max_x - min_x) * (max_y - min_y), 1.0)
    base_area = max(float(base_w * base_h), 1.0)

    perspective_strength = abs(float(homography[2, 0])) * overlay_w + abs(float(homography[2, 1])) * overlay_h

    return {
        "invalid": False,
        "area_ratio": polygon_area / overlay_area,
        "canvas_ratio": canvas_area / base_area,
        "perspective_strength": perspective_strength,
    }


def _invert_homography(homography: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if homography is None:
        return None
    try:
        return np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None


def _prepare_segmentation_result(
    seg_results_input: list[SegmentationResult],
    index: int,
    image: np.ndarray,
) -> SegmentationResult:
    source = seg_results_input[index]
    mask = source.mask
    overlay = source.overlay
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = _build_overlay(image, mask)
    elif overlay.shape[:2] != image.shape[:2]:
        overlay = _build_overlay(image, mask)
    return SegmentationResult(
        mask=mask.copy(),
        overlay=overlay.copy(),
        method=source.method,
        fallback_reason=source.fallback_reason,
    )


def _build_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(image)
    color[:, :, 1] = 200
    alpha = 0.35
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    return (image * (1 - alpha * mask_3) + color * (alpha * mask_3)).astype(np.uint8)


def _collect_match_points(
    features_a, features_b, matches: list[cv2.DMatch]
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not matches:
        return None, None
    pts0 = np.float32([features_a.keypoints[m.queryIdx].pt for m in matches])
    pts1 = np.float32([features_b.keypoints[m.trainIdx].pt for m in matches])
    return pts0, pts1


def _filter_inlier_points(
    points_a: Optional[np.ndarray],
    points_b: Optional[np.ndarray],
    inlier_mask: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if points_a is None or points_b is None or len(points_a) == 0:
        return None, None
    if inlier_mask is None or len(inlier_mask) != len(points_a):
        return points_a, points_b
    keep = np.asarray(inlier_mask, dtype=bool)
    if keep.sum() < 8:
        return points_a, points_b
    return points_a[keep], points_b[keep]


def _extract_teeth_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_binary = (mask > 0).astype(image.dtype)
    if image.ndim == 3:
        mask_binary = mask_binary[:, :, None]
    return (image * mask_binary).astype(np.uint8)


def _extract_mask_from_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        binary = image > 0
    else:
        binary = np.any(image > 0, axis=2)
    return (binary.astype(np.uint8) * 255)


def _segmentation_from_mask(image: np.ndarray, mask: np.ndarray, method: str) -> SegmentationResult:
    return SegmentationResult(
        mask=mask.copy(),
        overlay=_build_overlay(image, mask),
        method=method,
        fallback_reason=None,
    )
