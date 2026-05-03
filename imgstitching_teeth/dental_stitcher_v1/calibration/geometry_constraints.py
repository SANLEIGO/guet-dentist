"""
牙弓几何约束建模模块

建立牙弓几何约束模型（抛物线曲率、间距规律）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from dental_stitcher_v1.calibration.instance_extractor import ToothInstance


@dataclass
class DentalArchModel:
    """牙弓几何模型"""
    arch_type: str                     # "upper" or "lower" or "unknown"
    curvature_params: np.ndarray      # [a, b, c] 抛物线参数：y = ax² + bx + c
    spacing_mean: float                # 平均间距（像素）
    spacing_std: float                 # 间距标准偏差（像素）
    curvature_rmse: float              # 抛物线拟合RMSE（像素）
    num_instances: int                 # 实例数量
    consistency_score: float           # 几何一致性评分（0-1）


def fit_dental_arch_curve(instances: list[ToothInstance]) -> DentalArchModel:
    """
    拟合牙弓曲率模型（抛物线）

    Args:
        instances: 牙齿实例列表（已排序）

    Returns:
        DentalArchModel包含抛物线参数和拟合质量
    """
    if len(instances) < 3:
        # 至少需要3个点才能拟合抛物线
        return DentalArchModel(
            arch_type="unknown",
            curvature_params=np.zeros(3),
            spacing_mean=0.0,
            spacing_std=0.0,
            curvature_rmse=float('inf'),
            num_instances=len(instances),
            consistency_score=0.0
        )

    # 提取中心点坐标
    centers = np.array([inst.center for inst in instances])
    x = centers[:, 0]
    y = centers[:, 1]

    # 抛物线拟合：y = ax² + bx + c
    # 使用最小二乘法
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    params, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

    # 计算拟合RMSE
    y_pred = A @ params
    rmse = np.sqrt(np.mean((y - y_pred)**2))

    # 计算间距约束
    spacing_mean, spacing_std = compute_spacing_constraints(instances)

    # 计算一致性评分
    consistency = compute_consistency_score(rmse, spacing_std, spacing_mean)

    # 推断牙弓类型（基于抛物线凹凸性）
    # 上牙弓：抛物线向下开口（a > 0）
    # 下牙弓：抛物线向上开口（a < 0）
    arch_type = "unknown"
    if params[0] > 0.0001:
        arch_type = "upper"
    elif params[0] < -0.0001:
        arch_type = "lower"

    return DentalArchModel(
        arch_type=arch_type,
        curvature_params=params,
        spacing_mean=spacing_mean,
        spacing_std=spacing_std,
        curvature_rmse=rmse,
        num_instances=len(instances),
        consistency_score=consistency
    )


def compute_spacing_constraints(instances: list[ToothInstance]) -> tuple[float, float]:
    """
    计算牙齿间距约束（相邻实例中心点距离）

    Args:
        instances: 牙齿实例列表（已排序）

    Returns:
        (spacing_mean, spacing_std): 平均间距和标准偏差
    """
    if len(instances) < 2:
        return 0.0, 0.0

    centers = np.array([inst.center for inst in instances])
    spacings = []

    for i in range(len(centers) - 1):
        dist = np.linalg.norm(centers[i+1] - centers[i])
        spacings.append(dist)

    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)

    return mean_spacing, std_spacing


def check_geometry_consistency(
    instances: list[ToothInstance],
    arch_model: DentalArchModel,
    strict: bool = True
) -> dict:
    """
    验证几何一致性

    Args:
        instances: 牙齿实例列表
        arch_model: 牙弓几何模型
        strict: 是否使用严格约束阈值

    Returns:
        包含一致性检查结果的字典
    """
    violations = []

    # 阈值设置（严格vs宽松）
    spacing_std_threshold = 0.3 if strict else 0.5  # 标准偏差占比阈值
    curvature_rmse_threshold = 20.0 if strict else 50.0  # RMSE阈值（像素）
    aspect_ratio_std_threshold = 0.15 if strict else 0.25  # 长宽比方差阈值

    # 1. 间距约束：标准偏差不应过大
    if arch_model.spacing_mean > 0:
        spacing_ratio = arch_model.spacing_std / arch_model.spacing_mean
        if spacing_ratio > spacing_std_threshold:
            violations.append({
                "type": "spacing_variance_high",
                "value": spacing_ratio,
                "threshold": spacing_std_threshold,
                "severity": "warning"
            })

    # 2. 曲率约束：拟合RMSE不应过大
    if arch_model.curvature_rmse > curvature_rmse_threshold:
        violations.append({
            "type": "curvature_rmse_high",
            "value": arch_model.curvature_rmse,
            "threshold": curvature_rmse_threshold,
            "severity": "warning"
        })

    # 3. 尺寸比例约束：同类牙齿长宽比应相似
    class_groups = _group_instances_by_class(instances)
    for class_id, group in class_groups.items():
        if len(group) > 1:
            ratios = [inst.aspect_ratio for inst in group]
            ratio_std = np.std(ratios)
            if ratio_std > aspect_ratio_std_threshold:
                violations.append({
                    "type": f"aspect_ratio_variance_high_class_{class_id}",
                    "value": ratio_std,
                    "threshold": aspect_ratio_std_threshold,
                    "severity": "warning",
                    "class_name": group[0].class_name
                })

    # 4. 实例数量约束：至少4个实例才能进行标定
    if len(instances) < 4:
        violations.append({
            "type": "insufficient_instances",
            "value": len(instances),
            "threshold": 4,
            "severity": "critical"
        })

    # 计算一致性评分
    critical_count = sum(1 for v in violations if v["severity"] == "critical")
    warning_count = sum(1 for v in violations if v["severity"] == "warning")

    consistent = (critical_count == 0)

    return {
        "consistent": consistent,
        "violations": violations,
        "critical_count": critical_count,
        "warning_count": warning_count,
        "spacing_mean": arch_model.spacing_mean,
        "spacing_std": arch_model.spacing_std,
        "spacing_ratio": arch_model.spacing_std / max(arch_model.spacing_mean, 1.0),
        "curvature_rmse": arch_model.curvature_rmse,
        "consistency_score": arch_model.consistency_score
    }


def filter_outlier_instances(
    instances: list[ToothInstance],
    arch_model: DentalArchModel,
    max_spacing_deviation: float = 2.0
) -> list[ToothInstance]:
    """
    过滤异常实例（基于间距约束）

    Args:
        instances: 牙齿实例列表（已排序）
        arch_model: 牙弓几何模型
        max_spacing_deviation: 最大间距偏离倍数

    Returns:
        过滤后的实例列表
    """
    if len(instances) < 3:
        return instances

    # 计算每个实例与相邻实例的间距
    centers = np.array([inst.center for inst in instances])
    spacings = []
    for i in range(len(centers) - 1):
        dist = np.linalg.norm(centers[i+1] - centers[i])
        spacings.append(dist)

    # 使用中位数和MAD（Median Absolute Deviation）检测异常
    median_spacing = np.median(spacings)
    mad = np.median(np.abs(spacings - median_spacing))

    # 异常阈值
    threshold = median_spacing + max_spacing_deviation * max(mad, 1.0)

    # 保留正常实例
    filtered_instances = [instances[0]]  # 保留第一个
    for i, spacing in enumerate(spacings):
        if spacing <= threshold:
            filtered_instances.append(instances[i+1])

    return filtered_instances


def compute_consistency_score(
    curvature_rmse: float,
    spacing_std: float,
    spacing_mean: float
) -> float:
    """
    计算几何一致性评分（0-1）

    Args:
        curvature_rmse: 曲线拟合RMSE
        spacing_std: 间距标准偏差
        spacing_mean: 平均间距

    Returns:
        一致性评分（0-1，越高越好）
    """
    # RMSE评分（0-20px区间）
    rmse_score = max(0, min(1, 1 - curvature_rmse / 20.0))

    # 间距评分（标准偏差占比0-50%区间）
    if spacing_mean > 0:
        spacing_ratio = spacing_std / spacing_mean
        spacing_score = max(0, min(1, 1 - spacing_ratio / 0.5))
    else:
        spacing_score = 0.0

    # 综合评分（加权平均）
    consistency_score = 0.6 * rmse_score + 0.4 * spacing_score

    return consistency_score


def _group_instances_by_class(
    instances: list[ToothInstance]
) -> dict[int, list[ToothInstance]]:
    """
    按类别ID分组实例

    Args:
        instances: 实例列表

    Returns:
        {class_id: [instances]}字典
    """
    groups = {}
    for inst in instances:
        if inst.class_id not in groups:
            groups[inst.class_id] = []
        groups[inst.class_id].append(inst)
    return groups


def estimate_real_spacing_mm(
    spacing_pixels: float,
    image_width: int,
    assumed_focal_length: float = 10.0,  # mm（口腔内窥镜典型焦距）
    assumed_distance: float = 50.0  # mm（拍摄距离）
) -> float:
    """
    估计真实间距（毫米）

    Args:
        spacing_pixels: 像素间距
        image_width: 图像宽度（像素）
        assumed_focal_length: 假设焦距（mm）
        assumed_distance: 假设拍摄距离（mm）

    Returns:
        真实间距（mm）
    """
    # 传感器宽度假设（5mm，口腔内窥镜典型值）
    sensor_width_mm = 5.0

    # 像素大小（mm/pixel）
    pixel_size = sensor_width_mm / image_width

    # 比例因子（基于针孔相机模型）
    scale_factor = assumed_distance / assumed_focal_length

    # 真实间距（mm）
    real_spacing = spacing_pixels * pixel_size * scale_factor

    return real_spacing