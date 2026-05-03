"""
牙齿实例提取模块

从AlphaDent YOLOv8输出提取完整实例信息
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class ToothInstance:
    """单个牙齿实例信息"""
    instance_id: int           # 实例ID（排序后）
    class_id: int              # 类别ID（0-8，AlphaDent 9类）
    class_name: str            # 类别名称
    bbox: np.ndarray           # 边界框 [x1, y1, x2, y2]
    center: np.ndarray         # 中心点 [cx, cy]
    mask: np.ndarray           # 单实例mask (H×W)，uint8类型，255=实例区域
    area: int                  # 面积（像素）
    aspect_ratio: float        # 长宽比 (width/height)
    confidence: float          # 检测置信度
    width: int                 # bbox宽度（像素）
    height: int                # bbox高度（像素）


@dataclass
class InstanceSegmentationResult:
    """实例分割结果"""
    instances: list[ToothInstance]         # 所有牙齿实例（已排序）
    combined_mask: np.ndarray              # 合并mask（兼容现有流程）
    overlay: np.ndarray                    # 可视化overlay（不同颜色区分实例）
    method: str                            # 方法标识
    fallback_reason: Optional[str] = None  # 失败原因


def extract_teeth_instances_from_yolo(
    results: Any,
    image: np.ndarray,
    apply_grabcut: bool = False
) -> InstanceSegmentationResult:
    """
    从YOLOv8结果提取所有牙齿实例信息

    Args:
        results: YOLOv8 predict结果对象
        image: 原始图像（BGR格式）
        apply_grabcut: 是否对每个实例应用GrabCut精细化

    Returns:
        InstanceSegmentationResult包含所有实例信息
    """
    # 检查是否有检测结果
    if not results or len(results) == 0:
        return _empty_instance_result(image, "no_results")

    result = results[0]

    # 检查是否有mask
    masks = result.masks
    if masks is None or masks.data is None or masks.data.shape[0] == 0:
        return _empty_instance_result(image, "no_masks")

    # 提取数据
    masks_data = masks.data.detach().cpu().numpy()  # [N, H, W] 实例掩膜
    boxes = result.boxes                           # YOLOv8 Boxes对象

    # 提取boxes数据（归一化坐标）
    if hasattr(boxes, 'xyxyn'):
        boxes_xyxyn = boxes.xyxyn.detach().cpu().numpy()  # [N, 4] 归一化坐标
    else:
        boxes_xyxy = boxes.xyxy.detach().cpu().numpy()    # [N, 4] 像素坐标
        h, w = image.shape[:2]
        boxes_xyxyn = boxes_xyxy / np.array([w, h, w, h])

    # 提取类别和置信度
    classes = boxes.cls.detach().cpu().numpy()     # [N] 类别ID
    confidences = boxes.conf.detach().cpu().numpy()  # [N] 置信度
    names = result.names                            # 类别名称字典 {id: name}

    # 图像尺寸
    h, w = image.shape[:2]

    # 构建ToothInstance列表
    instances = []
    for i in range(masks_data.shape[0]):
        # 提取单实例mask
        mask_i = (masks_data[i] > 0.5).astype(np.uint8) * 255

        # 边界框（归一化坐标转像素坐标）
        box_norm = boxes_xyxyn[i]  # [x1_n, y1_n, x2_n, y2_n]
        bbox_pixel = np.array([
            box_norm[0] * w,  # x1
            box_norm[1] * h,  # y1
            box_norm[2] * w,  # x2
            box_norm[3] * h   # y2
        ])

        # 计算中心点
        center = compute_mask_center(mask_i)

        # 面积和尺寸
        area = cv2.countNonZero(mask_i)
        width = int(bbox_pixel[2] - bbox_pixel[0])
        height = int(bbox_pixel[3] - bbox_pixel[1])
        aspect_ratio = width / max(height, 1)

        # GrabCut精细化（可选）
        if apply_grabcut and area > 100:
            mask_i = _grabcut_refine_instance(image, mask_i)
            center = compute_mask_center(mask_i)
            area = cv2.countNonZero(mask_i)

        # 创建实例对象
        instance = ToothInstance(
            instance_id=i,  # 暂时使用原始ID，后续会重新编号
            class_id=int(classes[i]),
            class_name=names[int(classes[i])],
            bbox=bbox_pixel,
            center=center,
            mask=mask_i,
            area=area,
            aspect_ratio=aspect_ratio,
            confidence=float(confidences[i]),
            width=width,
            height=height
        )
        instances.append(instance)

    # 按x坐标排序（从左到右）
    instances = sort_instances_by_position(instances)

    # 重新分配instance_id（排序后）
    for idx, inst in enumerate(instances):
        inst.instance_id = idx

    # 生成合并mask（兼容现有流程）
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for inst in instances:
        combined_mask = np.maximum(combined_mask, inst.mask)

    # 生成可视化overlay
    overlay = visualize_instances_with_labels(image, instances)

    return InstanceSegmentationResult(
        instances=instances,
        combined_mask=combined_mask,
        overlay=overlay,
        method="alphadent_instances",
        fallback_reason=None
    )


def compute_mask_center(mask: np.ndarray) -> np.ndarray:
    """
    计算mask的中心点

    Args:
        mask: 二值mask (H×W), uint8类型

    Returns:
        中心点坐标 [cx, cy]
    """
    # 使用 moments 计算质心
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return np.array([cx, cy], dtype=np.float32)
    else:
        # 如果moments失败，使用bbox中心
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            cx = np.mean(xs)
            cy = np.mean(ys)
            return np.array([cx, cy], dtype=np.float32)
        else:
            # 空mask
            return np.array([0.0, 0.0], dtype=np.float32)


def sort_instances_by_position(instances: list[ToothInstance]) -> list[ToothInstance]:
    """
    按中心点x坐标排序实例（从左到右）

    Args:
        instances: 未排序的实例列表

    Returns:
        排序后的实例列表
    """
    return sorted(instances, key=lambda inst: inst.center[0])


def visualize_instances_with_labels(
    image: np.ndarray,
    instances: list[ToothInstance],
    alpha: float = 0.4
) -> np.ndarray:
    """
    可视化所有实例（不同颜色+类别标签）

    Args:
        image: 原始图像（BGR格式）
        instances: 实例列表
        alpha: overlay透明度

    Returns:
        可视化overlay图像
    """
    overlay = image.copy()

    # 预定义颜色（9种）
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
        (128, 0, 255),  # 橙色
        (255, 128, 0),  # 深蓝
        (0, 128, 255),  # 浅蓝
    ]

    for inst in instances:
        # 选择颜色（基于类别ID）
        color = colors[inst.class_id % len(colors)]

        # 绘制mask区域
        mask_3ch = cv2.cvtColor(inst.mask, cv2.COLOR_GRAY2BGR) / 255.0
        colored_mask = np.zeros_like(overlay, dtype=np.float32)
        colored_mask[:, :] = color
        mask_region = (inst.mask > 0)

        # 混合颜色
        overlay[mask_region] = (
            overlay[mask_region] * (1 - alpha) +
            colored_mask[mask_region] * alpha
        ).astype(np.uint8)

        # 绘制边界框
        x1, y1, x2, y2 = inst.bbox.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # 绘制中心点
        cx, cy = inst.center.astype(int)
        cv2.circle(overlay, (cx, cy), 5, color, -1)

        # 绘制标签（类别名称+置信度）
        label = f"{inst.class_name} {inst.confidence:.2f}"
        label_pos = (x1, y1 - 10)
        cv2.putText(
            overlay, label, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

        # 绘制实例ID（排序后的编号）
        id_label = f"#{inst.instance_id}"
        id_pos = (cx - 10, cy + 20)
        cv2.putText(
            overlay, id_label, id_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    return overlay


def _grabcut_refine_instance(
    image: np.ndarray,
    initial_mask: np.ndarray,
    iterations: int = 3
) -> np.ndarray:
    """
    对单个实例应用GrabCut精细化

    Args:
        image: 原始图像
        initial_mask: 初始mask
        iterations: GrabCut迭代次数

    Returns:
        精细化后的mask
    """
    h, w = image.shape[:2]

    # GrabCut需要特定的mask标记
    # 0: 确定背景, 1: 确定前景, 2: 可能背景, 3: 可能前景
    grabcut_mask = np.zeros((h, w), dtype=np.uint8)

    # 初始mask作为可能前景
    grabcut_mask[initial_mask > 0] = cv2.GC_PR_FGD

    # 扩展bbox区域（边界区域标记为可能背景）
    ys, xs = np.where(initial_mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        # 扩展20像素
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        # bbox外标记为确定背景
        grabcut_mask[:y1, :] = cv2.GC_BGD
        grabcut_mask[y2:, :] = cv2.GC_BGD
        grabcut_mask[:, :x1] = cv2.GC_BGD
        grabcut_mask[:, x2:] = cv2.GC_BGD

    # 运行GrabCut
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    rect = (0, 0, w, h)  # 全图（已通过mask标记）
    cv2.grabCut(
        image, grabcut_mask, rect,
        bgd_model, fgd_model,
        iterations, cv2.GC_INIT_WITH_MASK
    )

    # 提取前景mask
    refined_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)

    # 如果GrabCut失败（面积显著减少），返回原始mask
    refined_area = cv2.countNonZero(refined_mask)
    original_area = cv2.countNonZero(initial_mask)
    if refined_area < 0.5 * original_area:
        return initial_mask

    return refined_mask


def _empty_instance_result(
    image: np.ndarray,
    reason: str,
    detailed_message: str = ""
) -> InstanceSegmentationResult:
    """
    创建空的实例分割结果（失败时）

    Args:
        image: 原始图像
        reason: 失败原因代码
        detailed_message: 详细的失败说明（可选）

    Returns:
        空的InstanceSegmentationResult
    """
    h, w = image.shape[:2]
    empty_mask = np.zeros((h, w), dtype=np.uint8)

    # 组合失败原因（代码+详细说明）
    fallback_reason = reason
    if detailed_message:
        fallback_reason = f"{reason}|{detailed_message}"

    return InstanceSegmentationResult(
        instances=[],
        combined_mask=empty_mask,
        overlay=image.copy(),
        method="failed",
        fallback_reason=fallback_reason
    )