"""
Bundle Adjustment联合优化模块

优化相机参数和牙齿3D位置
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr

from dental_stitcher_v1.calibration.camera_estimator import CameraParameters
from dental_stitcher_v1.calibration.instance_extractor import ToothInstance


def bundle_adjustment_optimization(
    all_instance_sets: list[list[ToothInstance]],
    camera_params: CameraParameters,
    arch_models: list
) -> tuple[CameraParameters, list[list[np.ndarray]]]:
    """
    Bundle Adjustment联合优化

    Args:
        all_instance_sets: 所有图像的实例列表
        camera_params: 初始相机参数
        arch_models: 牙弓几何模型

    Returns:
        (optimized_camera, optimized_positions): 优化后的相机参数和牙齿3D位置
    """
    # 初始化牙齿3D位置（简化：使用第一张图像的2D位置作为初始估计）
    tooth_3d_positions = initialize_tooth_3d_positions_simple(
        all_instance_sets, camera_params
    )

    # 简化优化：仅优化畸变参数，保持内参固定
    # （完整版本需要优化所有参数，包括旋转和平移）

    def objective(params):
        """优化目标函数"""
        k1, k2, p1, p2, k3 = params

        total_error = 0.0

        # 更新相机参数
        camera_params.k1 = k1
        camera_params.k2 = k2
        camera_params.p1 = p1
        camera_params.p2 = p2
        camera_params.k3 = k3

        # 重投影误差（简化）
        for img_idx, instances in enumerate(all_instance_sets):
            if img_idx >= len(camera_params.rotation_vectors):
                continue

            R = camera_params.rotation_vectors[img_idx]
            t = camera_params.translation_vectors[img_idx] if img_idx < len(camera_params.translation_vectors) else np.zeros(3)

            for inst_idx, instance in enumerate(instances):
                # 简化：使用估计的3D位置
                if inst_idx >= len(tooth_3d_positions[img_idx]):
                    continue

                point_3d = tooth_3d_positions[img_idx][inst_idx]

                # 投影到2D（简化）
                projected = project_point_simple(point_3d, camera_params, R, t)

                # 重投影误差
                error = np.linalg.norm(projected - instance.center)
                total_error += error**2

        # 几何约束违反（简化）
        for positions_img in tooth_3d_positions:
            if len(positions_img) < 2:
                continue

            # 间距约束（简化）
            spacings = compute_3d_spacings(positions_img)
            if len(spacings) > 0:
                spacing_std = np.std(spacings)
                total_error += 0.1 * spacing_std**2

        return total_error

    # 初始参数
    initial_params = np.array([
        camera_params.k1,
        camera_params.k2,
        camera_params.p1,
        camera_params.p2,
        camera_params.k3
    ])

    # 优化求解
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5), (-0.1, 0.1), (-0.01, 0.01), (-0.01, 0.01), (-0.05, 0.05)],
        options={'maxiter': 200, 'disp': False}
    )

    # 更新相机参数
    camera_params.k1 = result.x[0]
    camera_params.k2 = result.x[1]
    camera_params.p1 = result.x[2]
    camera_params.p2 = result.x[3]
    camera_params.k3 = result.x[4]

    return camera_params, tooth_3d_positions


def initialize_tooth_3d_positions_simple(
    all_instance_sets: list[list[ToothInstance]],
    camera_params: CameraParameters
) -> list[list[np.ndarray]]:
    """
    简化初始化牙齿3D位置（使用2D位置作为估计）

    Args:
        all_instance_sets: 所有图像的实例列表
        camera_params: 相机参数

    Returns:
        每张图像的牙齿3D位置列表
    """
    all_positions = []

    for img_idx, instances in enumerate(all_instance_sets):
        positions_img = []

        # 假设：牙齿在相机前方50mm处（简化）
        assumed_depth = 50.0  # mm

        for inst in instances:
            # 从2D中心点反投影到3D
            x_2d = inst.center[0]
            y_2d = inst.center[1]

            # 归一化坐标
            x_norm = (x_2d - camera_params.cx) / camera_params.fx
            y_norm = (y_2d - camera_params.cy) / camera_params.fy

            # 3D位置（简化）
            x_3d = x_norm * assumed_depth
            y_3d = y_norm * assumed_depth
            z_3d = assumed_depth

            point_3d = np.array([x_3d, y_3d, z_3d])
            positions_img.append(point_3d)

        all_positions.append(positions_img)

    return all_positions


def project_point_simple(
    point_3d: np.ndarray,
    camera_params: CameraParameters,
    rot_vec: np.ndarray,
    trans_vec: np.ndarray
) -> np.ndarray:
    """
    简化的3D点投影到2D（仅考虑内参）

    Args:
        point_3d: 3D点坐标
        camera_params: 相机参数
        rot_vec: 旋转向量（Rodrigues）
        trans_vec: 平移向量

    Returns:
        2D投影点坐标
    """
    # 简化：忽略旋转和平移（假设所有图像在同一视角）
    x_3d, y_3d, z_3d = point_3d

    # 投影（针孔相机模型）
    x_2d = (x_3d / z_3d) * camera_params.fx + camera_params.cx
    y_2d = (y_3d / z_3d) * camera_params.fy + camera_params.cy

    return np.array([x_2d, y_2d])


def compute_3d_spacings(tooth_positions: list[np.ndarray]) -> list[float]:
    """
    计算3D间距

    Args:
        tooth_positions: 牙齿3D位置列表

    Returns:
        间距列表（mm）
    """
    if len(tooth_positions) < 2:
        return []

    spacings = []
    for i in range(len(tooth_positions) - 1):
        dist = np.linalg.norm(tooth_positions[i+1] - tooth_positions[i])
        spacings.append(dist)

    return spacings


def compute_curvature_error(tooth_positions: list[np.ndarray]) -> float:
    """
    计算牙弓曲率误差（抛物线拟合RMSE）

    Args:
        tooth_positions: 牙齿3D位置列表

    Returns:
        曲率拟合误差
    """
    if len(tooth_positions) < 3:
        return 0.0

    # 使用x和y坐标拟合抛物线（忽略z）
    positions = np.array(tooth_positions)
    x = positions[:, 0]
    y = positions[:, 1]

    # 拟合抛物线
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    params = np.linalg.lstsq(A, y, rcond=None)[0]

    # RMSE
    y_pred = A @ params
    rmse = np.sqrt(np.mean((y - y_pred)**2))

    return rmse