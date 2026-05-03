"""
畸变校正模块

应用畸变校正到图像并评估质量
"""

from __future__ import annotations

import cv2
import numpy as np

from dental_stitcher_v1.calibration.camera_estimator import CameraParameters, get_camera_matrix, get_distortion_coefficients
from dental_stitcher_v1.calibration.geometry_constraints import fit_dental_arch_curve, compute_spacing_constraints
from dental_stitcher_v1.calibration.instance_extractor import ToothInstance


def undistort_images(
    images: list[np.ndarray],
    camera_params: CameraParameters
) -> list[np.ndarray]:
    """应用畸变校正到所有图像"""

    # 构建OpenCV相机矩阵和畸变系数
    K = get_camera_matrix(camera_params)
    D = get_distortion_coefficients(camera_params)

    undistorted_images = []

    for image in images:
        # 计算新相机矩阵（保持全视野）
        new_K = cv2.getOptimalNewCameraMatrix(
            K, D,
            (camera_params.image_width, camera_params.image_height),
            alpha=0  # 0=去除所有无效像素
        )

        # 应用畸变校正
        undistorted = cv2.undistort(
            image, K, D, None, new_K
        )

        undistorted_images.append(undistorted)

    return undistorted_images


def evaluate_calibration_quality(
    original_instances: list[list[ToothInstance]],
    undistorted_instances: list[list[ToothInstance]],
    camera_params: CameraParameters
) -> dict:
    """评估标定质量"""

    # 指标1：校正后牙弓曲线拟合改善度
    original_rmse = []
    undistorted_rmse = []

    for orig_inst, undist_inst in zip(original_instances, undistorted_instances):
        orig_centers = np.array([inst.center for inst in orig_inst])
        undist_centers = np.array([inst.center for inst in undist_inst])

        # 拟合抛物线并计算RMSE
        if len(orig_centers) >= 3:
            x = orig_centers[:, 0]
            y = orig_centers[:, 1]
            A = np.vstack([x**2, x, np.ones_like(x)]).T
            params = np.linalg.lstsq(A, y, rcond=None)[0]
            y_pred = A @ params
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            original_rmse.append(rmse)

        if len(undist_centers) >= 3:
            x = undist_centers[:, 0]
            y = undist_centers[:, 1]
            A = np.vstack([x**2, x, np.ones_like(x)]).T
            params = np.linalg.lstsq(A, y, rcond=None)[0]
            y_pred = A @ params
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            undistorted_rmse.append(rmse)

    rmse_improvement = np.mean(original_rmse) - np.mean(undistorted_rmse) if original_rmse and undistorted_rmse else 0.0

    # 指标2：间距一致性改善度
    original_spacing_std = []
    undistorted_spacing_std = []

    for orig_inst in original_instances:
        mean, std = compute_spacing_constraints(orig_inst)
        if mean > 0:
            original_spacing_std.append(std / mean)

    for undist_inst in undistorted_instances:
        mean, std = compute_spacing_constraints(undist_inst)
        if mean > 0:
            undistorted_spacing_std.append(std / mean)

    spacing_improvement = np.mean(original_spacing_std) - np.mean(undistorted_spacing_std) if original_spacing_std and undistorted_spacing_std else 0.0

    # 指标3：畸变参数合理性
    distortion_reasonable = (
        abs(camera_params.k1) < 0.5 and
        abs(camera_params.k2) < 0.1 and
        abs(camera_params.p1) < 0.01 and
        abs(camera_params.p2) < 0.01 and
        abs(camera_params.k3) < 0.05
    )

    # 成功标准
    calibration_success = (
        rmse_improvement > 5.0 and
        distortion_reasonable
    )

    return {
        "rmse_improvement": rmse_improvement,
        "spacing_improvement": spacing_improvement,
        "original_rmse_mean": np.mean(original_rmse) if original_rmse else 0.0,
        "undistorted_rmse_mean": np.mean(undistorted_rmse) if undistorted_rmse else 0.0,
        "original_spacing_std_mean": np.mean(original_spacing_std) if original_spacing_std else 0.0,
        "undistorted_spacing_std_mean": np.mean(undistorted_spacing_std) if undistorted_spacing_std else 0.0,
        "distortion_reasonable": distortion_reasonable,
        "calibration_success": calibration_success
    }