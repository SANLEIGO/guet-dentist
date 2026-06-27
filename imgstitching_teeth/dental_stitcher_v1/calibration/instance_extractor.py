"""
牙齿实例提取模块

从AlphaDent YOLOv8输出提取完整实例信息
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    source_instance_ids: list[int] = field(default_factory=list)
    source_labels: list[str] = field(default_factory=list)


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
    apply_grabcut: bool = False,
    merge_overlaps: bool = False,
    merge_overlap_threshold: float = 0.55,
    merge_bbox_iou_threshold: float = 0.20,
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

    # 提取类别和置信度
    classes = boxes.cls.detach().cpu().numpy()     # [N] 类别ID
    confidences = boxes.conf.detach().cpu().numpy()  # [N] 置信度
    names = result.names                            # 类别名称字典 {id: name}

    # 图像尺寸
    h, w = image.shape[:2]
    boxes_xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes, "xyxy") else None

    # 构建ToothInstance列表
    instances = []
    for i in range(masks_data.shape[0]):
        # 提取单实例mask
        mask_i = _normalize_instance_mask((masks_data[i] > 0.5).astype(np.uint8) * 255, (h, w))

        # 边界框直接使用原图像素坐标，避免和低分辨率mask错位
        if boxes_xyxy is not None:
            x1, y1, x2, y2 = boxes_xyxy[i][:4]
            bbox_pixel = np.array([
                float(np.clip(x1, 0, w - 1)),
                float(np.clip(y1, 0, h - 1)),
                float(np.clip(x2, 0, w - 1)),
                float(np.clip(y2, 0, h - 1)),
            ])
        else:
            bbox_pixel = _bbox_from_mask(mask_i)

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
            height=height,
            source_instance_ids=[i],
            source_labels=[str(names[int(classes[i])])],
        )
        instances.append(instance)

    if merge_overlaps:
        instances = merge_overlapping_instances(
            instances,
            overlap_threshold=merge_overlap_threshold,
            bbox_iou_threshold=merge_bbox_iou_threshold,
        )

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


def merge_overlapping_instances(
    instances: list[ToothInstance],
    overlap_threshold: float = 0.55,
    bbox_iou_threshold: float = 0.20,
) -> list[ToothInstance]:
    """Merge raw AlphaDent masks that strongly overlap into tooth candidates."""
    if len(instances) < 2:
        return instances

    parent = list(range(len(instances)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            if _instance_overlap_ratio(instances[i], instances[j]) >= overlap_threshold:
                union(i, j)
                continue
            if _bbox_iou(instances[i].bbox, instances[j].bbox) >= bbox_iou_threshold:
                union(i, j)

    groups: dict[int, list[ToothInstance]] = {}
    for idx, instance in enumerate(instances):
        groups.setdefault(find(idx), []).append(instance)

    merged: list[ToothInstance] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        merged.append(_merge_instance_group(members))

    return merged


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


def _normalize_instance_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32)


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _instance_overlap_ratio(a: ToothInstance, b: ToothInstance) -> float:
    mask_a = a.mask > 0
    mask_b = b.mask > 0
    intersection = int(np.count_nonzero(mask_a & mask_b))
    if intersection <= 0:
        return 0.0
    return float(intersection / max(1, min(a.area, b.area)))


def _merge_instance_group(members: list[ToothInstance]) -> ToothInstance:
    union_mask = np.zeros_like(members[0].mask, dtype=np.uint8)
    source_instance_ids: list[int] = []
    source_labels: list[str] = []
    best_member = max(members, key=lambda item: item.confidence)
    for member in members:
        union_mask = np.maximum(union_mask, member.mask)
        source_instance_ids.extend(member.source_instance_ids or [member.instance_id])
        source_labels.extend(member.source_labels or [member.class_name])

    bbox = _bbox_from_mask(union_mask)
    center = compute_mask_center(union_mask)
    area = cv2.countNonZero(union_mask)
    width = int(max(0.0, bbox[2] - bbox[0]))
    height = int(max(0.0, bbox[3] - bbox[1]))

    return ToothInstance(
        instance_id=best_member.instance_id,
        class_id=best_member.class_id,
        class_name=best_member.class_name,
        bbox=bbox,
        center=center,
        mask=union_mask,
        area=area,
        aspect_ratio=width / max(height, 1),
        confidence=max(member.confidence for member in members),
        width=width,
        height=height,
        source_instance_ids=sorted(dict.fromkeys(source_instance_ids)),
        source_labels=sorted(dict.fromkeys(source_labels)),
    )


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
