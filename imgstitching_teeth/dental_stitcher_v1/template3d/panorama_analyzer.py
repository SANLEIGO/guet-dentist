from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np

from dental_stitcher_v1.template3d.caries_detector import (
    CariesDetection,
    CariesDetectionResult,
    detect_caries,
)
from dental_stitcher_v1.template3d.schema import (
    ArchLabel,
    FeatureSeverity,
    FeatureType,
    ResolvedToothFeature,
    ToothSurface,
)


PANORAMA_TOOTH_COUNT = 16
_MIN_TOOTH_PIXELS = 220


@dataclass(frozen=True)
class ArchPrediction:
    arch_label: ArchLabel
    confidence: float
    status_text: str
    metrics: dict


@dataclass(frozen=True)
class ToothSlotAnalysis:
    slot_index: int
    tooth_id: int
    x0: int
    x1: int
    occupancy_score: float
    missing_confidence: float
    is_missing_candidate: bool


@dataclass(frozen=True)
class PanoramaAnalysisResult:
    image_id: str
    predicted_arch: ArchPrediction
    arch_label: ArchLabel
    features: list[ResolvedToothFeature]
    tooth_slots: list[ToothSlotAnalysis]
    caries_result: CariesDetectionResult
    tooth_bbox_xyxy: tuple[int, int, int, int]
    slot_scores: list[float]
    missing_slot_indices: list[int]
    notes: list[str]
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


def analyze_panorama_image(
    image: np.ndarray,
    image_id: str,
    *,
    arch_override: Optional[ArchLabel] = None,
    caries_conf_threshold: float = 0.10,
    caries_mode: str = "sensitive",
    missing_sensitivity: float = 0.22,
) -> PanoramaAnalysisResult:
    arch_prediction = infer_panorama_arch(image)
    manual_arch = arch_override if arch_override in {ArchLabel.UPPER, ArchLabel.LOWER} else None
    resolved_arch = manual_arch if manual_arch is not None else arch_prediction.arch_label
    if resolved_arch not in {ArchLabel.UPPER, ArchLabel.LOWER}:
        resolved_arch = ArchLabel.UPPER

    caries_result = detect_caries(image, conf_threshold=caries_conf_threshold, mode=caries_mode)
    bbox, slot_scores = estimate_tooth_slot_scores(image)
    missing_slot_indices = estimate_missing_slot_indices(slot_scores, missing_sensitivity)

    result = PanoramaAnalysisResult(
        image_id=image_id,
        predicted_arch=arch_prediction,
        arch_label=resolved_arch,
        features=[],
        tooth_slots=[],
        caries_result=caries_result,
        tooth_bbox_xyxy=bbox,
        slot_scores=slot_scores,
        missing_slot_indices=missing_slot_indices,
        notes=_build_notes(arch_prediction, caries_result, missing_slot_indices, manual_arch),
        error=None if caries_result.success else caries_result.error,
    )
    return remap_panorama_analysis(result, resolved_arch)


def remap_panorama_analysis(result: PanoramaAnalysisResult, arch_label: ArchLabel) -> PanoramaAnalysisResult:
    if arch_label not in {ArchLabel.UPPER, ArchLabel.LOWER}:
        arch_label = result.arch_label
    tooth_slots = _build_tooth_slots(
        arch_label=arch_label,
        bbox=result.tooth_bbox_xyxy,
        slot_scores=result.slot_scores,
        missing_slot_indices=result.missing_slot_indices,
    )
    features = _build_features(
        image_id=result.image_id,
        arch_label=arch_label,
        caries_detections=result.caries_result.detections,
        tooth_slots=tooth_slots,
    )
    notes = list(result.notes)
    if arch_label in {ArchLabel.UPPER, ArchLabel.LOWER}:
        if notes:
            notes[0] = f"已按人工指定的{_arch_label_display(arch_label)}进行牙位映射。"
        else:
            notes.append(f"已按人工指定的{_arch_label_display(arch_label)}进行牙位映射。")
    return replace(result, arch_label=arch_label, features=features, tooth_slots=tooth_slots, notes=notes)


def infer_panorama_arch(image: np.ndarray) -> ArchPrediction:
    mask = build_tooth_candidate_mask(image)
    h, w = image.shape[:2]
    tooth_pixels = int(np.count_nonzero(mask))
    if tooth_pixels < _MIN_TOOTH_PIXELS:
        return ArchPrediction(
            arch_label=ArchLabel.UNKNOWN,
            confidence=0.0,
            status_text="牙齿候选区域太少，无法稳定判断上/下牙弓。",
            metrics={"tooth_pixels": tooth_pixels},
        )

    ys, xs = np.nonzero(mask)
    center_y = float(ys.mean() / max(h - 1, 1))
    center_x = float(xs.mean() / max(w - 1, 1))
    upper_ratio = float(np.count_nonzero(ys < h * 0.48) / max(tooth_pixels, 1))
    lower_ratio = float(np.count_nonzero(ys > h * 0.52) / max(tooth_pixels, 1))
    vertical_gap = abs(center_y - 0.5)

    if center_y >= 0.52:
        arch_label = ArchLabel.LOWER
        directional_strength = lower_ratio - upper_ratio
    elif center_y <= 0.48:
        arch_label = ArchLabel.UPPER
        directional_strength = upper_ratio - lower_ratio
    elif lower_ratio >= upper_ratio:
        arch_label = ArchLabel.LOWER
        directional_strength = lower_ratio - upper_ratio
    else:
        arch_label = ArchLabel.UPPER
        directional_strength = upper_ratio - lower_ratio

    confidence = float(np.clip(0.38 + vertical_gap * 1.2 + directional_strength * 0.35, 0.18, 0.86))
    if confidence < 0.45:
        status = "上/下牙弓判断比较不稳定，请人工确认。"
    else:
        status = f"疑似{'上牙弓' if arch_label == ArchLabel.UPPER else '下牙弓'}，仍需人工复核。"

    return ArchPrediction(
        arch_label=arch_label,
        confidence=confidence,
        status_text=status,
        metrics={
            "tooth_pixels": tooth_pixels,
            "center_x": round(center_x, 4),
            "center_y": round(center_y, 4),
            "upper_ratio": round(upper_ratio, 4),
            "lower_ratio": round(lower_ratio, 4),
        },
    )


def build_tooth_candidate_mask(image: np.ndarray) -> np.ndarray:
    if image is None or image.ndim != 3:
        return np.zeros((1, 1), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    effective = gray > 22
    effective_pixels = gray[effective] if np.any(effective) else gray.reshape(-1)
    threshold = max(72.0, float(np.percentile(effective_pixels, 68)))

    bright = gray >= threshold
    tooth_like_color = (saturation <= 168) | (value >= 188)
    mask = (bright & tooth_like_color & effective).astype(np.uint8) * 255

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _keep_largest_components(mask, max_components=10)
    return mask


def estimate_tooth_slot_scores(image: np.ndarray) -> tuple[tuple[int, int, int, int], list[float]]:
    mask = build_tooth_candidate_mask(image)
    h, w = mask.shape[:2]
    if int(np.count_nonzero(mask)) < _MIN_TOOTH_PIXELS:
        return (0, 0, max(w - 1, 0), max(h - 1, 0)), [0.0] * PANORAMA_TOOTH_COUNT

    ys, xs = np.nonzero(mask)
    x0 = int(np.clip(np.percentile(xs, 1.5), 0, max(w - 1, 0)))
    x1 = int(np.clip(np.percentile(xs, 98.5), x0 + 1, max(w - 1, 0)))
    y0 = int(np.clip(np.percentile(ys, 1.0), 0, max(h - 1, 0)))
    y1 = int(np.clip(np.percentile(ys, 99.0), y0 + 1, max(h - 1, 0)))

    raw_scores: list[float] = []
    for index in range(PANORAMA_TOOTH_COUNT):
        sx0 = int(round(x0 + (x1 - x0 + 1) * index / PANORAMA_TOOTH_COUNT))
        sx1 = int(round(x0 + (x1 - x0 + 1) * (index + 1) / PANORAMA_TOOTH_COUNT))
        sx1 = max(sx1, sx0 + 1)
        slot = mask[y0 : y1 + 1, sx0:sx1]
        raw_scores.append(float(np.count_nonzero(slot) / max(slot.size, 1)))

    positive_scores = [score for score in raw_scores if score > 0.002]
    normalizer = float(np.percentile(positive_scores, 70)) if positive_scores else 1.0
    normalizer = max(normalizer, 0.01)
    slot_scores = [float(np.clip(score / normalizer, 0.0, 1.0)) for score in raw_scores]
    return (x0, y0, x1, y1), slot_scores


def estimate_missing_slot_indices(slot_scores: list[float], sensitivity: float = 0.22) -> list[int]:
    if len(slot_scores) != PANORAMA_TOOTH_COUNT:
        return []

    threshold = float(np.clip(sensitivity, 0.08, 0.45))
    missing: list[int] = []
    for index, score in enumerate(slot_scores):
        left = slot_scores[index - 1] if index > 0 else 0.0
        right = slot_scores[index + 1] if index < len(slot_scores) - 1 else 0.0
        neighbor_support = (left + right) * 0.5
        internal_gap = 0 < index < len(slot_scores) - 1 and neighbor_support >= threshold * 1.8
        edge_gap = index in {0, len(slot_scores) - 1} and max(left, right) >= threshold * 2.3
        if score < threshold and (internal_gap or edge_gap):
            missing.append(index)
    return missing


def draw_panorama_analysis_overlay(image: np.ndarray, result: PanoramaAnalysisResult) -> np.ndarray:
    overlay = image.copy()
    x0, y0, x1, y1 = result.tooth_bbox_xyxy
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (55, 118, 172), 2)

    for slot in result.tooth_slots:
        color = (92, 92, 92) if slot.is_missing_candidate else (120, 154, 172)
        thickness = 2 if slot.is_missing_candidate else 1
        cv2.line(overlay, (slot.x0, y0), (slot.x0, y1), color, thickness)
        if slot.is_missing_candidate:
            cv2.rectangle(overlay, (slot.x0, y0), (slot.x1, y1), (92, 92, 92), 2)
        label_y = max(18, y0 - 8)
        cv2.putText(
            overlay,
            str(slot.tooth_id),
            (slot.x0 + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for detection in result.caries_result.detections:
        dx1, dy1, dx2, dy2 = [int(round(value)) for value in detection.bbox_xyxy]
        cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (44, 72, 190), 3)
        tooth_id = _tooth_id_for_detection(detection, result.tooth_slots)
        label = f"{tooth_id} caries {detection.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(dy1 - text_h - baseline - 8, 0)
        cv2.rectangle(overlay, (dx1, top), (dx1 + text_w + 8, top + text_h + baseline + 8), (44, 72, 190), -1)
        cv2.putText(
            overlay,
            label,
            (dx1 + 4, top + text_h + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    arch_label = "upper" if result.arch_label == ArchLabel.UPPER else "lower"
    cv2.putText(
        overlay,
        f"confirmed arch: {arch_label}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (40, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _build_tooth_slots(
    *,
    arch_label: ArchLabel,
    bbox: tuple[int, int, int, int],
    slot_scores: list[float],
    missing_slot_indices: list[int],
) -> list[ToothSlotAnalysis]:
    x0, _, x1, _ = bbox
    fdi_order = _fdi_order_for_arch(arch_label)
    slots: list[ToothSlotAnalysis] = []
    for index, tooth_id in enumerate(fdi_order):
        sx0 = int(round(x0 + (x1 - x0 + 1) * index / PANORAMA_TOOTH_COUNT))
        sx1 = int(round(x0 + (x1 - x0 + 1) * (index + 1) / PANORAMA_TOOTH_COUNT))
        score = slot_scores[index] if index < len(slot_scores) else 0.0
        is_missing = index in set(missing_slot_indices)
        slots.append(
            ToothSlotAnalysis(
                slot_index=index,
                tooth_id=tooth_id,
                x0=sx0,
                x1=max(sx1, sx0 + 1),
                occupancy_score=score,
                missing_confidence=float(np.clip(1.0 - score, 0.0, 0.92)),
                is_missing_candidate=is_missing,
            )
        )
    return slots


def _build_features(
    *,
    image_id: str,
    arch_label: ArchLabel,
    caries_detections: list[CariesDetection],
    tooth_slots: list[ToothSlotAnalysis],
) -> list[ResolvedToothFeature]:
    features: list[ResolvedToothFeature] = []
    for detection in caries_detections:
        tooth_id = _tooth_id_for_detection(detection, tooth_slots)
        features.append(
            ResolvedToothFeature(
                tooth_id=tooth_id,
                feature_type=FeatureType.CARIES_SUSPECTED,
                confidence=detection.confidence,
                evidence_image_ids=[image_id],
                surface=ToothSurface.UNKNOWN,
                severity=_severity_from_confidence(detection.confidence),
                review_required=True,
                notes=(
                    f"单张全景自动映射到 FDI {tooth_id}；"
                    f"检测来源 {detection.source}，需人工确认牙位和龋齿框。"
                ),
            )
        )

    for slot in tooth_slots:
        if not slot.is_missing_candidate:
            continue
        features.append(
            ResolvedToothFeature(
                tooth_id=slot.tooth_id,
                feature_type=FeatureType.MISSING,
                confidence=slot.missing_confidence,
                evidence_image_ids=[image_id],
                surface=ToothSurface.WHOLE_TOOTH,
                severity=FeatureSeverity.REVIEW,
                review_required=True,
                notes=(
                    f"单张全景牙齿亮区在该槽位较少（score={slot.occupancy_score:.2f}），"
                    "仅作为缺牙候选，需要人工复核。"
                ),
            )
        )
    return features


def _tooth_id_for_detection(detection: CariesDetection, tooth_slots: list[ToothSlotAnalysis]) -> int:
    if not tooth_slots:
        return 11
    center_x, _ = detection.center
    return min(
        tooth_slots,
        key=lambda slot: abs(center_x - ((slot.x0 + slot.x1) * 0.5)),
    ).tooth_id


def _fdi_order_for_arch(arch_label: ArchLabel) -> list[int]:
    if arch_label == ArchLabel.LOWER:
        return [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
    return [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]


def _severity_from_confidence(confidence: float) -> FeatureSeverity:
    if confidence >= 0.8:
        return FeatureSeverity.HIGH
    if confidence >= 0.55:
        return FeatureSeverity.MEDIUM
    return FeatureSeverity.REVIEW


def _build_notes(
    arch_prediction: ArchPrediction,
    caries_result: CariesDetectionResult,
    missing_slot_indices: list[int],
    manual_arch: Optional[ArchLabel] = None,
) -> list[str]:
    if manual_arch in {ArchLabel.UPPER, ArchLabel.LOWER}:
        notes = [f"已按人工指定的{_arch_label_display(manual_arch)}进行牙位映射。"]
    else:
        notes = [arch_prediction.status_text]
    if not caries_result.success:
        notes.append(f"龋齿模型推理失败：{caries_result.error}")
    else:
        notes.append(f"龋齿模型检出 {len(caries_result.detections)} 个候选框。")
    notes.append(f"缺牙启发式检出 {len(missing_slot_indices)} 个候选牙位。")
    notes.append("所有自动结果都只作为候选，必须由人工复核后确认。")
    return notes


def _arch_label_display(arch_label: ArchLabel) -> str:
    return "上牙弓" if arch_label == ArchLabel.UPPER else "下牙弓"


def _keep_largest_components(mask: np.ndarray, max_components: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    component_ids = sorted(
        range(1, num_labels),
        key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]),
        reverse=True,
    )[:max_components]
    filtered = np.zeros_like(mask)
    for idx in component_ids:
        filtered[labels == idx] = 255
    return filtered
