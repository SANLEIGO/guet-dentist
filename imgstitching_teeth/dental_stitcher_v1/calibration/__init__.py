"""
自动标定模块

利用牙齿几何特性自动估计相机参数和畸变校正
"""

from dental_stitcher_v1.calibration.auto_calibrator import (
    auto_calibrate_pipeline,
    CalibrationResult,
)
from dental_stitcher_v1.calibration.instance_extractor import (
    ToothInstance,
    InstanceSegmentationResult,
    extract_teeth_instances_from_yolo,
)
from dental_stitcher_v1.calibration.camera_estimator import CameraParameters
from dental_stitcher_v1.calibration.calibration_diagnostics import CalibrationDiagnostics

__all__ = [
    "auto_calibrate_pipeline",
    "CalibrationResult",
    "ToothInstance",
    "InstanceSegmentationResult",
    "extract_teeth_instances_from_yolo",
    "CameraParameters",
    "CalibrationDiagnostics",
]