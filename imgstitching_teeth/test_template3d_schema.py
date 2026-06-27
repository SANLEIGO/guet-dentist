from __future__ import annotations

from dental_stitcher_v1.template3d import (
    ArchLabel,
    CariesDetection,
    CariesDetectionResult,
    FeatureSeverity,
    FeatureType,
    ResolvedToothFeature,
    TemplateRenderState,
    ToothRenderState,
    ToothSurface,
    build_template_render_state,
    build_adult_template_teeth,
    detections_to_feature_observations,
    observations_to_resolved_features,
)


def test_build_adult_template_teeth_uses_fdi_upper_arch() -> None:
    teeth = build_adult_template_teeth(ArchLabel.UPPER)

    assert [tooth.tooth_id for tooth in teeth] == [
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
    ]
    assert teeth[0].mesh_node_name == "tooth_11"


def test_template_render_state_exports_feature_payload() -> None:
    feature = ResolvedToothFeature(
        tooth_id=16,
        feature_type=FeatureType.CARIES_SUSPECTED,
        confidence=0.76,
        evidence_image_ids=["upper_0007"],
        surface=ToothSurface.OCCLUSAL,
        severity=FeatureSeverity.MEDIUM,
    )
    state = TemplateRenderState(
        template_id="adult_upper_v1",
        arch_label=ArchLabel.UPPER,
        teeth=[ToothRenderState(tooth_id=16, features=[feature], highlight_color="#a83b32")],
    )

    payload = state.to_dict()

    assert payload["arch_label"] == "upper"
    assert payload["teeth"][0]["features"][0]["feature_type"] == "caries_suspected"
    assert payload["teeth"][0]["features"][0]["surface"] == "occlusal"


def test_build_template_render_state_marks_missing_tooth() -> None:
    feature = ResolvedToothFeature(
        tooth_id=14,
        feature_type=FeatureType.MISSING,
        confidence=0.88,
        evidence_image_ids=["upper_0003"],
        surface=ToothSurface.WHOLE_TOOTH,
        severity=FeatureSeverity.HIGH,
    )

    state = build_template_render_state(ArchLabel.UPPER, [feature])
    tooth_14 = next(tooth for tooth in state.teeth if tooth.tooth_id == 14)

    assert tooth_14.visible is False
    assert tooth_14.opacity < 0.5
    assert tooth_14.label == "缺牙"
    assert state.evidence_summary["affected_tooth_count"] == 1


def test_caries_detections_convert_to_resolved_features() -> None:
    result = CariesDetectionResult(
        detections=[
            CariesDetection(
                bbox_xyxy=(10.0, 20.0, 44.0, 58.0),
                confidence=0.82,
                class_id=0,
                class_name="caries",
            )
        ],
        model_path="pts/curries_best_model.onnx",
        mode="balanced",
    )

    observations = detections_to_feature_observations(
        result,
        arch_label=ArchLabel.UPPER,
        image_id="upper_0001",
        candidate_tooth_ids=[16],
    )
    resolved = observations_to_resolved_features(observations, tooth_id=16)

    assert observations[0].feature_type == FeatureType.CARIES_SUSPECTED
    assert observations[0].candidate_tooth_ids == [16]
    assert resolved[0].tooth_id == 16
    assert resolved[0].review_required is True
    assert resolved[0].severity == FeatureSeverity.HIGH
