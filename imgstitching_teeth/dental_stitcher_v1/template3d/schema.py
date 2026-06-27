from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class ArchLabel(str, Enum):
    UPPER = "upper"
    LOWER = "lower"
    UNKNOWN = "unknown"


class FeatureType(str, Enum):
    MISSING = "missing"
    CARIES_SUSPECTED = "caries_suspected"


class ToothSurface(str, Enum):
    WHOLE_TOOTH = "whole_tooth"
    OCCLUSAL = "occlusal"
    INCISAL = "incisal"
    BUCCAL_LABIAL = "buccal_labial"
    LINGUAL_PALATAL = "lingual_palatal"
    MESIAL = "mesial"
    DISTAL = "distal"
    UNKNOWN = "unknown"


class FeatureSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    REVIEW = "review"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TemplateTooth:
    tooth_id: int
    arch_label: ArchLabel
    quadrant: int
    position: int
    display_name: str
    mesh_node_name: str


@dataclass
class FeatureObservation:
    observation_id: str
    image_id: str
    arch_label: ArchLabel
    feature_type: FeatureType
    confidence: float
    candidate_tooth_ids: list[int] = field(default_factory=list)
    surface: ToothSurface = ToothSurface.UNKNOWN
    severity: FeatureSeverity = FeatureSeverity.UNKNOWN
    bbox_xyxy: Optional[tuple[float, float, float, float]] = None
    mask_path: Optional[str] = None
    notes: str = ""

    def normalized_confidence(self) -> float:
        return max(0.0, min(1.0, float(self.confidence)))


@dataclass
class ResolvedToothFeature:
    tooth_id: int
    feature_type: FeatureType
    confidence: float
    evidence_image_ids: list[str]
    surface: ToothSurface = ToothSurface.UNKNOWN
    severity: FeatureSeverity = FeatureSeverity.UNKNOWN
    review_required: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["feature_type"] = self.feature_type.value
        payload["surface"] = self.surface.value
        payload["severity"] = self.severity.value
        return payload


@dataclass
class ToothRenderState:
    tooth_id: int
    visible: bool = True
    opacity: float = 1.0
    highlight_color: Optional[str] = None
    label: Optional[str] = None
    features: list[ResolvedToothFeature] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tooth_id": self.tooth_id,
            "visible": self.visible,
            "opacity": max(0.0, min(1.0, float(self.opacity))),
            "highlight_color": self.highlight_color,
            "label": self.label,
            "features": [feature.to_dict() for feature in self.features],
        }


@dataclass
class TemplateRenderState:
    template_id: str
    arch_label: ArchLabel
    teeth: list[ToothRenderState]
    evidence_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "arch_label": self.arch_label.value,
            "teeth": [tooth.to_dict() for tooth in self.teeth],
            "evidence_summary": self.evidence_summary,
        }
