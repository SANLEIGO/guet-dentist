from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StepDiagnostics:
    success: bool
    details: dict[str, Any] = field(default_factory=dict)
    fallback_reason: Optional[str] = None


@dataclass
class StitchDiagnostics:
    pipeline_version: str
    segmentation: StepDiagnostics
    features: StepDiagnostics
    registration: StepDiagnostics
    blending: StepDiagnostics
    metrics: dict[str, Any] = field(default_factory=dict)
    per_image: list[dict[str, Any]] = field(default_factory=list)
    per_pair: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "segmentation": _step_to_dict(self.segmentation),
            "features": _step_to_dict(self.features),
            "registration": _step_to_dict(self.registration),
            "blending": _step_to_dict(self.blending),
            "metrics": self.metrics,
            "per_image": self.per_image,
            "per_pair": self.per_pair,
            "logs": self.logs,
        }


def _step_to_dict(step: StepDiagnostics) -> dict[str, Any]:
    return {
        "success": step.success,
        "details": step.details,
        "fallback_reason": step.fallback_reason,
    }
