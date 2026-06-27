"""Template-based 3D dental visualization contracts."""

from dental_stitcher_v1.template3d.schema import (
    ArchLabel,
    FeatureObservation,
    FeatureSeverity,
    FeatureType,
    ResolvedToothFeature,
    TemplateRenderState,
    TemplateTooth,
    ToothRenderState,
    ToothSurface,
)
from dental_stitcher_v1.template3d.render_state import build_template_render_state
from dental_stitcher_v1.template3d.tooth_map import build_adult_template_teeth
from dental_stitcher_v1.template3d.assets import (
    DentalTemplateAsset,
    get_default_dental_arch_asset,
    load_model_data_uri,
    read_asset_license,
)
from dental_stitcher_v1.template3d.fdi_assignment import (
    AssignedToothCandidate,
    FDIAssignmentResult,
    FDISequenceConfig,
    assign_tooth_candidates_sequentially,
    build_expected_fdi_sequence,
)
from dental_stitcher_v1.template3d.panorama_analyzer import (
    ArchPrediction,
    PanoramaAnalysisResult,
    ToothSlotAnalysis,
    analyze_panorama_image,
    draw_panorama_analysis_overlay,
    infer_panorama_arch,
    remap_panorama_analysis,
)
from dental_stitcher_v1.template3d.caries_detector import (
    CariesDetection,
    CariesDetectionResult,
    detect_caries,
    detections_to_feature_observations,
    observations_to_resolved_features,
    overlay_caries_detections,
)

__all__ = [
    "ArchLabel",
    "FeatureObservation",
    "FeatureSeverity",
    "FeatureType",
    "ResolvedToothFeature",
    "TemplateRenderState",
    "TemplateTooth",
    "ToothRenderState",
    "ToothSurface",
    "CariesDetection",
    "CariesDetectionResult",
    "DentalTemplateAsset",
    "AssignedToothCandidate",
    "FDIAssignmentResult",
    "FDISequenceConfig",
    "ArchPrediction",
    "PanoramaAnalysisResult",
    "ToothSlotAnalysis",
    "assign_tooth_candidates_sequentially",
    "build_template_render_state",
    "build_adult_template_teeth",
    "build_expected_fdi_sequence",
    "analyze_panorama_image",
    "draw_panorama_analysis_overlay",
    "get_default_dental_arch_asset",
    "infer_panorama_arch",
    "load_model_data_uri",
    "remap_panorama_analysis",
    "read_asset_license",
    "detect_caries",
    "detections_to_feature_observations",
    "observations_to_resolved_features",
    "overlay_caries_detections",
]
