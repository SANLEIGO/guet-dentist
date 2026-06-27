from __future__ import annotations

from dental_stitcher_v1.template3d.schema import (
    ArchLabel,
    FeatureSeverity,
    FeatureType,
    ResolvedToothFeature,
    ToothSurface,
)


def build_demo_features(arch_label: ArchLabel, scenario: str) -> list[ResolvedToothFeature]:
    if scenario == "blank":
        return []
    if arch_label == ArchLabel.LOWER:
        return _lower_demo_features()
    return _upper_demo_features()


def _upper_demo_features() -> list[ResolvedToothFeature]:
    return [
        ResolvedToothFeature(
            tooth_id=14,
            feature_type=FeatureType.MISSING,
            confidence=0.88,
            evidence_image_ids=["upper_left_0003", "upper_center_0004"],
            surface=ToothSurface.WHOLE_TOOTH,
            severity=FeatureSeverity.HIGH,
            notes="相邻牙位连续可见，14 牙位缺失概率较高。",
        ),
        ResolvedToothFeature(
            tooth_id=16,
            feature_type=FeatureType.CARIES_SUSPECTED,
            confidence=0.76,
            evidence_image_ids=["upper_right_0007", "upper_right_0008"],
            surface=ToothSurface.OCCLUSAL,
            severity=FeatureSeverity.MEDIUM,
            notes="咬合面暗色区域，建议复核。",
        ),
    ]


def _lower_demo_features() -> list[ResolvedToothFeature]:
    return [
        ResolvedToothFeature(
            tooth_id=36,
            feature_type=FeatureType.MISSING,
            confidence=0.84,
            evidence_image_ids=["lower_left_0004", "lower_left_0005"],
            surface=ToothSurface.WHOLE_TOOTH,
            severity=FeatureSeverity.HIGH,
        ),
        ResolvedToothFeature(
            tooth_id=46,
            feature_type=FeatureType.CARIES_SUSPECTED,
            confidence=0.79,
            evidence_image_ids=["lower_right_0007"],
            surface=ToothSurface.OCCLUSAL,
            severity=FeatureSeverity.MEDIUM,
            review_required=True,
        ),
    ]
