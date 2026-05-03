"""
拼接结果去噪模块

去除拼接后图像中的小块软组织噪点
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class NoiseRemovalConfig:
    """去噪配置参数"""
    min_area_threshold: int = 2000  # 最小面积阈值（像素）
    morph_kernel_size: int = 7  # 形态学核大小
    use_morphology: bool = True  # 是否使用形态学开运算
    use_connected_components: bool = True  # 是否使用连通域分析
    fill_holes: bool = True  # 是否填充内部孔洞


def remove_noise_from_stitched_result(
    stitched_image: np.ndarray,
    config: NoiseRemovalConfig = None
) -> tuple[np.ndarray, dict]:
    """
    去除拼接结果中的小块噪点

    Args:
        stitched_image: 拼接后的图像（BGR格式）
        config: 去噪配置参数，如果为None则使用默认配置(method4)

    Returns:
        tuple: (去噪后的图像, 去噪统计信息)
    """
    if config is None:
        # 使用method4的默认配置
        config = NoiseRemovalConfig(
            min_area_threshold=2000,
            morph_kernel_size=7,
            use_morphology=True,
            use_connected_components=True,
            fill_holes=True
        )

    # 统计信息
    stats = {
        "original_pixels": 0,
        "final_pixels": 0,
        "num_components_before": 0,
        "num_components_after": 0,
        "removed_noise_count": 0,
        "kept_regions": []
    }

    # 步骤1: 提取非黑色区域（拼接结果的有效区域）
    gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    stats["original_pixels"] = np.count_nonzero(binary)

    # 步骤2: 形态学开运算（先侵蚀后膨胀）去除小噪点
    if config.use_morphology:
        kernel = np.ones((config.morph_kernel_size, config.morph_kernel_size), np.uint8)
        binary_morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    else:
        binary_morphed = binary

    # 步骤3: 连通域分析，保留大区域
    if config.use_connected_components:
        num_labels, labels, stats_cc, centroids = cv2.connectedComponentsWithStats(
            binary_morphed, connectivity=8
        )

        stats["num_components_before"] = num_labels - 1  # 不包括背景

        # 获取各个连通域的面积
        areas = stats_cc[:, cv2.CC_STAT_AREA]

        # 创建掩膜，只保留大区域
        final_mask = np.zeros(binary_morphed.shape, dtype=np.uint8)

        # 找出大面积区域并保留
        kept_count = 0
        for i in range(1, num_labels):  # 0是背景
            if areas[i] >= config.min_area_threshold:
                kept_count += 1
                final_mask[labels == i] = 255
                stats["kept_regions"].append({
                    "area": int(areas[i]),
                    "center": (float(centroids[i][0]), float(centroids[i][1]))
                })

        stats["num_components_after"] = kept_count
        stats["removed_noise_count"] = num_labels - 1 - kept_count
    else:
        final_mask = binary_morphed

    # 步骤4: 可选的闭运算（填充牙齿内部的小孔洞）
    if config.fill_holes and config.use_morphology:
        kernel_close = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 步骤5: 应用掩膜到原始图像
    result = cv2.bitwise_and(stitched_image, stitched_image, mask=final_mask)

    stats["final_pixels"] = np.count_nonzero(final_mask)

    return result, stats


def get_noise_removal_method_config(method_name: str) -> NoiseRemovalConfig:
    """
    根据方法名称获取去噪配置

    Args:
        method_name: 方法名称 ("method1", "method2", "method3", "method4")

    Returns:
        NoiseRemovalConfig配置对象
    """
    configs = {
        "method1": NoiseRemovalConfig(
            min_area_threshold=1000,
            morph_kernel_size=0,
            use_morphology=False,
            use_connected_components=True,
            fill_holes=False
        ),
        "method2": NoiseRemovalConfig(
            min_area_threshold=1000,
            morph_kernel_size=3,
            use_morphology=True,
            use_connected_components=True,
            fill_holes=True
        ),
        "method3": NoiseRemovalConfig(
            min_area_threshold=1000,
            morph_kernel_size=5,
            use_morphology=True,
            use_connected_components=True,
            fill_holes=True
        ),
        "method4": NoiseRemovalConfig(  # 用户推荐的最佳方法
            min_area_threshold=2000,
            morph_kernel_size=7,
            use_morphology=True,
            use_connected_components=True,
            fill_holes=True
        ),
    }

    return configs.get(method_name, configs["method4"])  # 默认使用method4


def format_noise_removal_stats(stats: dict) -> str:
    """
    格式化去噪统计信息为可读文本

    Args:
        stats: 去噪统计字典

    Returns:
        格式化的文本字符串
    """
    lines = []

    if stats["num_components_before"] > 0:
        lines.append(f"原始连通域数: {stats['num_components_before']}个")
        lines.append(f"保留大区域数: {stats['num_components_after']}个")
        lines.append(f"去除噪点数: {stats['removed_noise_count']}个")

    if stats["kept_regions"]:
        lines.append("\n保留的主要区域:")
        for i, region in enumerate(stats["kept_regions"][:5], 1):  # 只显示前5个
            lines.append(f"  区域{i}: 面积={region['area']}px, 中心=({region['center'][0]:.1f}, {region['center'][1]:.1f})")

    if stats["original_pixels"] > 0:
        reduction = stats["original_pixels"] - stats["final_pixels"]
        reduction_pct = (reduction / stats["original_pixels"]) * 100
        lines.append(f"\n像素变化: {stats['original_pixels']} → {stats['final_pixels']} (减少{reduction_pct:.1f}%)")

    return "\n".join(lines)