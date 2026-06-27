from __future__ import annotations

from collections import defaultdict

from dental_stitcher_v1.template3d.schema import (
    ArchLabel,
    FeatureSeverity,
    FeatureType,
    ResolvedToothFeature,
    TemplateRenderState,
    ToothRenderState,
)
from dental_stitcher_v1.template3d.tooth_map import build_adult_template_teeth


FEATURE_LABELS = {
    FeatureType.MISSING: "缺牙",
    FeatureType.CARIES_SUSPECTED: "龋齿疑似",
}

FEATURE_COLORS = {
    FeatureType.MISSING: "#7d8790",
    FeatureType.CARIES_SUSPECTED: "#b94435",
}

_FEATURE_PRIORITY = {
    FeatureType.MISSING: 0,
    FeatureType.CARIES_SUSPECTED: 1,
}


def build_template_render_state(
    arch_label: ArchLabel,
    features: list[ResolvedToothFeature],
    template_id: str | None = None,
) -> TemplateRenderState:
    template_teeth = build_adult_template_teeth(arch_label)
    features_by_tooth: dict[int, list[ResolvedToothFeature]] = defaultdict(list)
    for feature in features:
        features_by_tooth[feature.tooth_id].append(feature)

    render_teeth: list[ToothRenderState] = []
    for template_tooth in template_teeth:
        tooth_features = sorted(
            features_by_tooth.get(template_tooth.tooth_id, []),
            key=_feature_rank,
        )
        primary = tooth_features[0] if tooth_features else None
        is_missing = any(feature.feature_type == FeatureType.MISSING for feature in tooth_features)
        render_teeth.append(
            ToothRenderState(
                tooth_id=template_tooth.tooth_id,
                visible=not is_missing,
                opacity=0.22 if is_missing else 1.0,
                highlight_color=_feature_color(primary) if primary else None,
                label=_feature_label(primary) if primary else None,
                features=tooth_features,
            )
        )

    evidence_summary = {
        "feature_count": len(features),
        "affected_tooth_count": len(features_by_tooth),
        "review_required_count": sum(1 for feature in features if feature.review_required),
    }
    return TemplateRenderState(
        template_id=template_id or f"adult_{arch_label.value}_v1",
        arch_label=arch_label,
        teeth=render_teeth,
        evidence_summary=evidence_summary,
    )


def _feature_rank(feature: ResolvedToothFeature) -> tuple[int, int, float]:
    return (
        _FEATURE_PRIORITY.get(feature.feature_type, 99),
        _severity_rank(feature.severity),
        -float(feature.confidence),
    )


def _severity_rank(severity: FeatureSeverity) -> int:
    return {
        FeatureSeverity.HIGH: 0,
        FeatureSeverity.MEDIUM: 1,
        FeatureSeverity.LOW: 2,
        FeatureSeverity.REVIEW: 3,
        FeatureSeverity.UNKNOWN: 4,
    }.get(severity, 4)


def _feature_color(feature: ResolvedToothFeature | None) -> str | None:
    if feature is None:
        return None
    return FEATURE_COLORS.get(feature.feature_type, "#675f99")


def _feature_label(feature: ResolvedToothFeature | None) -> str | None:
    if feature is None:
        return None
    return FEATURE_LABELS.get(feature.feature_type, "异常")
