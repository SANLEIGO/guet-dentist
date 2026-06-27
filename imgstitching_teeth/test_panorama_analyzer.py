from __future__ import annotations

import numpy as np

from dental_stitcher_v1.template3d import ArchLabel
from dental_stitcher_v1.template3d.caries_detector import CariesDetection, CariesDetectionResult
from dental_stitcher_v1.template3d.panorama_analyzer import (
    ArchPrediction,
    PanoramaAnalysisResult,
    infer_panorama_arch,
    remap_panorama_analysis,
)
from dental_stitcher_v1.template3d.schema import FeatureType


def _synthetic_arch_image(center_y: int) -> np.ndarray:
    image = np.zeros((220, 420, 3), dtype=np.uint8)
    for center_x in range(70, 360, 38):
        image[center_y - 12 : center_y + 12, center_x - 10 : center_x + 10] = (235, 235, 225)
    return image


def test_infer_panorama_arch_uses_vertical_tooth_distribution() -> None:
    upper = infer_panorama_arch(_synthetic_arch_image(66))
    lower = infer_panorama_arch(_synthetic_arch_image(158))

    assert upper.arch_label == ArchLabel.UPPER
    assert lower.arch_label == ArchLabel.LOWER


def test_remap_panorama_analysis_rebuilds_fdi_features_for_confirmed_arch() -> None:
    result = PanoramaAnalysisResult(
        image_id="panorama_001",
        predicted_arch=ArchPrediction(
            arch_label=ArchLabel.UPPER,
            confidence=0.72,
            status_text="test",
            metrics={},
        ),
        arch_label=ArchLabel.UPPER,
        features=[],
        tooth_slots=[],
        caries_result=CariesDetectionResult(
            detections=[
                CariesDetection(
                    bbox_xyxy=(90.0, 40.0, 120.0, 70.0),
                    confidence=0.81,
                    class_id=0,
                    class_name="caries",
                )
            ],
            model_path="pts/curries_best_model.onnx",
        ),
        tooth_bbox_xyxy=(0, 0, 320, 160),
        slot_scores=[0.8, 0.8, 0.8, 0.8, 0.05, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        missing_slot_indices=[4],
        notes=[],
    )

    upper = remap_panorama_analysis(result, ArchLabel.UPPER)
    lower = remap_panorama_analysis(result, ArchLabel.LOWER)

    assert any(feature.tooth_id == 14 and feature.feature_type == FeatureType.MISSING for feature in upper.features)
    assert any(feature.tooth_id == 44 and feature.feature_type == FeatureType.MISSING for feature in lower.features)
    assert all(feature.review_required for feature in lower.features)
