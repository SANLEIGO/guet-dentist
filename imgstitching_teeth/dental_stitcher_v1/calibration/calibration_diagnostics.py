"""
标定诊断系统

记录标定过程的详细诊断信息
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class StepDiagnostics:
    """单个步骤的诊断信息"""
    success: bool
    details: dict[str, Any]
    fallback_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "details": self.details,
            "fallback_reason": self.fallback_reason
        }


@dataclass
class CalibrationDiagnostics:
    """标定诊断数据"""
    # 各步骤诊断
    instance_extraction: StepDiagnostics
    geometry_constraints: StepDiagnostics
    camera_estimation: StepDiagnostics
    bundle_adjustment: StepDiagnostics
    distortion_correction: StepDiagnostics
    quality_validation: StepDiagnostics

    # 失败原因（如果有）
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_extraction": self.instance_extraction.to_dict(),
            "geometry_constraints": self.geometry_constraints.to_dict(),
            "camera_estimation": self.camera_estimation.to_dict(),
            "bundle_adjustment": self.bundle_adjustment.to_dict(),
            "distortion_correction": self.distortion_correction.to_dict(),
            "quality_validation": self.quality_validation.to_dict(),
            "failure_reason": self.failure_reason
        }


def generate_calibration_diagnostics(
    instance_extraction_success: bool,
    instance_extraction_details: dict,
    geometry_constraints_success: bool,
    geometry_constraints_details: dict,
    camera_estimation_success: bool,
    camera_estimation_details: dict,
    bundle_adjustment_success: bool,
    bundle_adjustment_details: dict,
    distortion_correction_success: bool,
    distortion_correction_details: dict,
    quality_validation_success: bool,
    quality_validation_details: dict,
    failure_reason: Optional[str] = None
) -> CalibrationDiagnostics:
    """生成标定诊断"""

    return CalibrationDiagnostics(
        instance_extraction=StepDiagnostics(
            success=instance_extraction_success,
            details=instance_extraction_details
        ),
        geometry_constraints=StepDiagnostics(
            success=geometry_constraints_success,
            details=geometry_constraints_details
        ),
        camera_estimation=StepDiagnostics(
            success=camera_estimation_success,
            details=camera_estimation_details
        ),
        bundle_adjustment=StepDiagnostics(
            success=bundle_adjustment_success,
            details=bundle_adjustment_details
        ),
        distortion_correction=StepDiagnostics(
            success=distortion_correction_success,
            details=distortion_correction_details
        ),
        quality_validation=StepDiagnostics(
            success=quality_validation_success,
            details=quality_validation_details
        ),
        failure_reason=failure_reason
    )