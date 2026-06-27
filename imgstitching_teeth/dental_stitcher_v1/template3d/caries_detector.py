from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from dotenv import dotenv_values

from dental_stitcher_v1.template3d.schema import (
    ArchLabel,
    FeatureObservation,
    FeatureSeverity,
    FeatureType,
    ResolvedToothFeature,
    ToothSurface,
)


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_ENV_CONFIG = dotenv_values(ENV_PATH)

DEFAULT_CARIES_MODEL_PATH = Path(__file__).resolve().parents[2] / "pts" / "curries_best_model.onnx"

_CARIES_MODEL: Any = None
_CARIES_MODEL_ERROR: Optional[str] = None
_CARIES_MODEL_PATH: Optional[str] = None


@dataclass
class CariesDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    source: str = "original"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass
class CariesDetectionResult:
    detections: list[CariesDetection]
    model_path: str
    mode: str = "balanced"
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


def detect_caries(
    image: np.ndarray,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    model_path: Optional[str] = None,
    mode: str = "balanced",
) -> CariesDetectionResult:
    resolved_model_path = _resolve_model_path(model_path)
    model = _get_caries_model(resolved_model_path)
    if model is None:
        return CariesDetectionResult(
            detections=[],
            model_path=resolved_model_path,
            mode=mode,
            error=_CARIES_MODEL_ERROR or "caries_model_unavailable",
        )

    if image.ndim != 3 or image.shape[2] != 3:
        return CariesDetectionResult(
            detections=[],
            model_path=resolved_model_path,
            mode=mode,
            error="caries_expected_bgr_image",
        )

    try:
        detections: list[CariesDetection] = []
        for source_name, candidate in _build_inference_images(image, mode):
            # Ultralytics expects numpy images in OpenCV/BGR order. Converting to RGB here
            # makes this caries model miss color-sensitive lesions that it detects elsewhere.
            results = model.predict(candidate, imgsz=imgsz, conf=conf_threshold, verbose=False)
            for detection in _extract_detections(results):
                detection.source = source_name
                detections.append(detection)
    except Exception as exc:  # pragma: no cover - defensive guard around model runtime
        return CariesDetectionResult(
            detections=[],
            model_path=resolved_model_path,
            mode=mode,
            error=f"caries_inference_failed: {exc}",
        )

    return CariesDetectionResult(
        detections=_merge_duplicate_detections(detections),
        model_path=resolved_model_path,
        mode=mode,
    )


def overlay_caries_detections(image: np.ndarray, result: CariesDetectionResult) -> np.ndarray:
    overlay = image.copy()
    for detection in result.detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (44, 72, 190), 3)
        label = f"{detection.class_name} {detection.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(y1 - text_h - baseline - 8, 0)
        cv2.rectangle(overlay, (x1, top), (x1 + text_w + 8, top + text_h + baseline + 8), (44, 72, 190), -1)
        cv2.putText(
            overlay,
            label,
            (x1 + 4, top + text_h + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def detections_to_feature_observations(
    result: CariesDetectionResult,
    arch_label: ArchLabel,
    image_id: str,
    candidate_tooth_ids: list[int],
) -> list[FeatureObservation]:
    observations: list[FeatureObservation] = []
    for index, detection in enumerate(result.detections):
        observations.append(
            FeatureObservation(
                observation_id=f"{image_id}_caries_{index:02d}",
                image_id=image_id,
                arch_label=arch_label,
                feature_type=FeatureType.CARIES_SUSPECTED,
                confidence=detection.confidence,
                candidate_tooth_ids=list(candidate_tooth_ids),
                surface=ToothSurface.UNKNOWN,
                severity=_severity_from_confidence(detection.confidence),
                bbox_xyxy=detection.bbox_xyxy,
                notes=f"{detection.class_name} detected by YOLOv8 caries model.",
            )
        )
    return observations


def observations_to_resolved_features(
    observations: list[FeatureObservation],
    tooth_id: int,
) -> list[ResolvedToothFeature]:
    resolved: list[ResolvedToothFeature] = []
    for observation in observations:
        resolved.append(
            ResolvedToothFeature(
                tooth_id=tooth_id,
                feature_type=observation.feature_type,
                confidence=observation.normalized_confidence(),
                evidence_image_ids=[observation.image_id],
                surface=observation.surface,
                severity=observation.severity,
                review_required=True,
                notes="模型已检测到疑似龋病区域；当前原型需人工确认牙位绑定。",
            )
        )
    return resolved


def _resolve_model_path(model_path: Optional[str] = None) -> str:
    configured_path = model_path or _ENV_CONFIG.get("DENTAL_CARIES_WEIGHTS")
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return str(path)
    return str(DEFAULT_CARIES_MODEL_PATH)


def _get_caries_model(model_path: str) -> Optional[Any]:
    global _CARIES_MODEL, _CARIES_MODEL_ERROR, _CARIES_MODEL_PATH
    if _CARIES_MODEL is not None and _CARIES_MODEL_PATH == model_path:
        return _CARIES_MODEL

    if not Path(model_path).exists():
        _CARIES_MODEL_ERROR = f"caries_weights_not_found: {model_path}"
        return None

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - environment dependent
        _CARIES_MODEL_ERROR = f"caries_import_failed: {exc}"
        return None

    try:
        _CARIES_MODEL = YOLO(model_path, task="detect")
        _CARIES_MODEL_PATH = model_path
        _CARIES_MODEL_ERROR = None
    except Exception as exc:  # pragma: no cover - environment dependent
        _CARIES_MODEL = None
        _CARIES_MODEL_ERROR = f"caries_load_failed: {exc}"
        return None
    return _CARIES_MODEL


def _extract_detections(results: Any) -> list[CariesDetection]:
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = getattr(result, "names", {}) or {}
    xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
    confs = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
    classes = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)

    detections: list[CariesDetection] = []
    for bbox, confidence, class_id_raw in zip(xyxy, confs, classes):
        class_id = int(class_id_raw)
        detections.append(
            CariesDetection(
                bbox_xyxy=tuple(float(value) for value in bbox[:4]),
                confidence=float(confidence),
                class_id=class_id,
                class_name=str(names.get(class_id, f"class_{class_id}")),
            )
        )
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections


def _build_inference_images(image: np.ndarray, mode: str) -> list[tuple[str, np.ndarray]]:
    normalized_mode = mode.lower()
    images = [("original", image)]
    if normalized_mode in {"sensitive", "review"}:
        images.append(("clahe", _apply_clahe_bgr(image)))
        images.append(("sharpen", _apply_sharpen(image)))
    if normalized_mode == "review":
        images.append(("clahe_sharpen", _apply_sharpen(_apply_clahe_bgr(image))))
    return images


def _apply_clahe_bgr(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def _apply_sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1.45, blurred, -0.45, 0)


def _merge_duplicate_detections(
    detections: list[CariesDetection],
    iou_threshold: float = 0.45,
) -> list[CariesDetection]:
    kept: list[CariesDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(_bbox_iou(detection.bbox_xyxy, item.bbox_xyxy) < iou_threshold for item in kept):
            kept.append(detection)
    return kept


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _severity_from_confidence(confidence: float) -> FeatureSeverity:
    if confidence >= 0.8:
        return FeatureSeverity.HIGH
    if confidence >= 0.55:
        return FeatureSeverity.MEDIUM
    return FeatureSeverity.REVIEW
