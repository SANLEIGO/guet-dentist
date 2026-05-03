"""
相机参数估计模块

估计相机内参（焦距、主点）和畸变系数
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import minimize

from dental_stitcher_v1.calibration.instance_extractor import ToothInstance


@dataclass
class CameraParameters:
    """相机参数"""
    # 内参
    fx: float              # 焦距x（像素）
    fy: float              # 焦距y（像素）
    cx: float              # 主点x（像素）
    cy: float              # 主点y（像素）

    # 畸变系数
    k1: float              # 径向畸变k1
    k2: float              # 径向畸变k2
    p1: float              # 切向畸变p1
    p2: float              # 切向畸变p2
    k3: float              # 径向畸变k3（可选）

    # 图像尺寸
    image_width: int
    image_height: int

    # 外参（每张图像）
    rotation_vectors: list[np.ndarray]   # Rodrigues向量
    translation_vectors: list[np.ndarray]

    # 质量指标
    reprojection_error: float = 0.0
    confidence: str = "unknown"


def initialize_camera_intrinsics(image_shape: tuple[int, int]) -> CameraParameters:
    """初始化相机内参（假设或从图像尺寸估计）"""
    h, w = image_shape[:2]

    # 假设：焦距≈图像宽度（常见口腔内窥镜）
    fx = w
    fy = w  # 假设方形像素

    # 主点：图像中心
    cx = w / 2.0
    cy = h / 2.0

    return CameraParameters(
        fx=fx, fy=fy, cx=cx, cy=cy,
        k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0,
        image_width=w, image_height=h,
        rotation_vectors=[], translation_vectors=[]
    )


def estimate_camera_extrinsics(
    instances_img1: list[ToothInstance],
    instances_img2: list[ToothInstance],
    intrinsics: CameraParameters
) -> tuple[np.ndarray, np.ndarray]:
    """估计两张图像间的相对位姿（本质矩阵方法）"""
    # 匹配牙齿实例（基于类别和相对位置）
    matched_pairs = match_tooth_instances(instances_img1, instances_img2)

    if len(matched_pairs) < 4:
        raise ValueError("Insufficient matched instances for pose estimation (need >=4)")

    # 构建2D-2D对应点
    pts1 = np.array([inst1.center for inst1, inst2 in matched_pairs])
    pts2 = np.array([inst2.center for inst1, inst2 in matched_pairs])

    # 估计本质矩阵
    E, mask = cv2.findEssentialMat(
        pts1, pts2,
        focal=intrinsics.fx,
        pp=(intrinsics.cx, intrinsics.cy),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    # 从本质矩阵恢复位姿
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # 选择正确的R和t（简化：使用R1）
    R = R1

    # 转换为Rodrigues向量
    rot_vec, _ = cv2.Rodrigues(R)

    return rot_vec, t


def match_tooth_instances(
    instances1: list[ToothInstance],
    instances2: list[ToothInstance]
) -> list[tuple[ToothInstance, ToothInstance]]:
    """匹配相邻图像的牙齿实例（基于类别+位置相似性）"""
    matched = []

    # 策略：匹配相同类别的实例，优先选择距离最近的
    for inst1 in instances1:
        best_match = None
        best_score = float('inf')

        for inst2 in instances2:
            # 类别必须相同
            if inst1.class_id != inst2.class_id:
                continue

            # 计算位置相似性（相对位置比例）
            # 使用相对位置而非绝对位置（因为视角变化）
            pos_diff = np.linalg.norm(
                inst1.center / np.array([inst1.width, inst1.height]) -
                inst2.center / np.array([inst2.width, inst2.height])
            )

            # 尺寸相似性
            size_diff = abs(inst1.area - inst2.area) / max(inst1.area, inst2.area)

            # 综合评分
            score = pos_diff + 0.5 * size_diff

            if score < best_score:
                best_score = score
                best_match = inst2

        if best_match is not None:
            matched.append((inst1, best_match))

    return matched


def estimate_distortion_parameters(
    all_instance_sets: list[list[ToothInstance]],
    intrinsics: CameraParameters,
    arch_models: list
) -> CameraParameters:
    """估计畸变参数（利用牙弓曲率约束）"""
    # 收集所有图像的牙弓中心点
    all_centers = []
    for instances in all_instance_sets:
        centers = np.array([inst.center for inst in instances])
        all_centers.append(centers)

    # 优化目标：最小化校正后牙弓曲线拟合误差
    def objective(dist_params):
        k1, k2, p1, p2, k3 = dist_params
        total_error = 0.0

        for centers in all_centers:
            # 应用畸变校正（逆过程）
            undistorted_points = undistort_points_simple(
                centers, intrinsics, dist_params
            )

            # 拟合抛物线
            x = undistorted_points[:, 0]
            y = undistorted_points[:, 1]

            if len(x) < 3:
                continue

            A = np.vstack([x**2, x, np.ones_like(x)]).T
            params = np.linalg.lstsq(A, y, rcond=None)[0]

            # 计算拟合误差
            y_pred = A @ params
            error = np.sum((y - y_pred)**2)
            total_error += error

        return total_error

    # 初始参数
    initial_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # 优化求解
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5), (-0.1, 0.1), (-0.01, 0.01), (-0.01, 0.01), (-0.05, 0.05)],
        options={'maxiter': 100, 'disp': False}
    )

    # 更新参数
    intrinsics.k1 = result.x[0]
    intrinsics.k2 = result.x[1]
    intrinsics.p1 = result.x[2]
    intrinsics.p2 = result.x[3]
    intrinsics.k3 = result.x[4]

    return intrinsics


def undistort_points_simple(
    points: np.ndarray,
    intrinsics: CameraParameters,
    dist_params: np.ndarray
) -> np.ndarray:
    """简单的点去畸变（仅径向畸变）"""
    k1, k2, p1, p2, k3 = dist_params

    # 转换到归一化坐标
    normalized = points.copy()
    normalized[:, 0] = (points[:, 0] - intrinsics.cx) / intrinsics.fx
    normalized[:, 1] = (points[:, 1] - intrinsics.cy) / intrinsics.fy

    # 计算径向畸变（简化）
    r2 = normalized[:, 0]**2 + normalized[:, 1]**2
    r4 = r2**2
    r6 = r2**3

    # 畸变校正因子
    factor = 1 + k1 * r2 + k2 * r4 + k3 * r6

    # 应用校正
    undistorted = normalized.copy()
    undistorted[:, 0] = normalized[:, 0] * factor
    undistorted[:, 1] = normalized[:, 1] * factor

    # 转换回像素坐标
    result = undistorted.copy()
    result[:, 0] = undistorted[:, 0] * intrinsics.fx + intrinsics.cx
    result[:, 1] = undistorted[:, 1] * intrinsics.fy + intrinsics.cy

    return result


def get_camera_matrix(intrinsics: CameraParameters) -> np.ndarray:
    """构建OpenCV相机矩阵K"""
    return np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1]
    ], dtype=np.float64)


def get_distortion_coefficients(intrinsics: CameraParameters) -> np.ndarray:
    """构建OpenCV畸变系数向量"""
    return np.array([
        intrinsics.k1,
        intrinsics.k2,
        intrinsics.p1,
        intrinsics.p2,
        intrinsics.k3
    ], dtype=np.float64)