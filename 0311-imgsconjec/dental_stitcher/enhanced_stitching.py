from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

from dental_stitcher.models import ImageRecord, MatchResult, StitchResult, PrecheckItem, PrecheckReport, CandidateScore
from dental_stitcher.utils import combine_quality_score, compute_image_metrics

logger = logging.getLogger(__name__)


class DentalImagePreprocessor:
    """专门的牙齿图像预处理类"""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预处理图像，返回(增强图像, 牙齿掩膜)"""
        # 创建简单的全白掩膜（用于特征检测）
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        # 轻微增强图像
        enhanced = self._enhance_image(image, mask)

        return enhanced, mask

    def _extract_teeth_mask(self, image: np.ndarray) -> np.ndarray:
        """提取牙齿区域掩膜 - 简化版本"""
        # 简单的亮度掩膜，排除黑色背景
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 30).astype(np.uint8) * 255
        return mask

    def _enhance_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """轻微增强图像质量"""
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 轻微CLAHE增强
        enhanced_l = self.clahe.apply(l)

        # 轻微融合（只保留30%的增强效果）
        blended_l = (l.astype(np.float32) * 0.7 +
                    enhanced_l.astype(np.float32) * 0.3).astype(np.uint8)

        # 轻微降噪
        blended_l = cv2.bilateralFilter(blended_l, 5, 50, 50)

        # 合并通道
        enhanced_lab = cv2.merge([blended_l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


class DentalFeatureMatcher:
    """专门针对牙齿图像的特征匹配器"""

    def __init__(self):
        # 使用SIFT作为主要特征检测器
        self.sift = cv2.SIFT_create(
            nfeatures=5000,  # 增加特征点数量
            nOctaveLayers=8,
            contrastThreshold=0.01,  # 降低对比度阈值，检测更多特征
            edgeThreshold=10,
            sigma=1.6
        )
        # 使用AKAZE作为备用 - 对纹理少的图像效果好
        self.akaze = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=0.001,  # 降低阈值
            nOctaves=4,
            nOctaveLayers=4
        )
        # 使用ORB作为最后备用
        self.orb = cv2.ORB_create(
            nfeatures=5000,  # 增加特征点
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10  # 降低阈值
        )

    def detect_and_compute(self, image: np.ndarray, mask: np.ndarray) -> Tuple[List, np.ndarray]:
        """检测特征点并计算描述子 - 多检测器策略"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        best_result = ([], None)
        best_score = 0

        # 尝试SIFT
        try:
            keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
            if descriptors is not None and len(keypoints) > best_score:
                best_result = (keypoints, descriptors)
                best_score = len(keypoints)
                logger.info(f"SIFT检测到 {len(keypoints)} 个特征点")
        except Exception as e:
            logger.warning(f"SIFT检测失败: {e}")

        # 尝试AKAZE - 特别适合纹理少的牙齿图像
        try:
            keypoints, descriptors = self.akaze.detectAndCompute(gray, mask)
            if descriptors is not None and len(keypoints) > best_score:
                best_result = (keypoints, descriptors)
                best_score = len(keypoints)
                logger.info(f"AKAZE检测到 {len(keypoints)} 个特征点")
        except Exception as e:
            logger.warning(f"AKAZE检测失败: {e}")

        # 如果SIFT和AKAZE都不够好，尝试ORB
        if best_score < 100:
            try:
                keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
                if descriptors is not None and len(keypoints) > best_score:
                    best_result = (keypoints, descriptors)
                    best_score = len(keypoints)
                    logger.info(f"ORB检测到 {len(keypoints)} 个特征点")
            except Exception as e:
                logger.warning(f"ORB检测失败: {e}")

        kp, desc = best_result
        if desc is not None:
            logger.info(f"最终使用检测器，特征点数: {len(kp)}")
        else:
            logger.warning("所有特征检测器都失败")

        return best_result

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray,
                      method: str = 'sift') -> List:
        """匹配两组特征描述子"""
        if desc1 is None or desc2 is None:
            return []

        if method == 'orb':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # KNN匹配
        matches = matcher.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test - 更宽松的阈值
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:  # 从0.75放宽到0.8
                good_matches.append(m)

        return good_matches

    def match_features_robust(self, desc1: np.ndarray, desc2: np.ndarray,
                             kp1: List, kp2: List, method: str = 'sift',
                             img_shape: Tuple[int, int] = None) -> List:
        """鲁棒的特征匹配 - 双向检查 + 空间过滤"""

        if desc1 is None or desc2 is None:
            return []

        # 前向匹配
        if method == 'orb':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        forward_matches = matcher.knnMatch(desc1, desc2, k=2)

        # 反向匹配
        backward_matches = matcher.knnMatch(desc2, desc1, k=2)

        # Lowe's ratio test + 双向检查
        good_matches = []
        match_dict_forward = {}
        match_dict_backward = {}

        # 处理前向匹配
        for m, n in forward_matches:
            if m.distance < 0.8 * n.distance:
                match_dict_forward[m.queryIdx] = m.trainIdx

        # 处理反向匹配
        for m, n in backward_matches:
            if m.distance < 0.8 * n.distance:
                match_dict_backward[m.queryIdx] = m.trainIdx

        # 只保留双向一致的匹配
        for query_idx in match_dict_forward:
            train_idx = match_dict_forward[query_idx]
            if train_idx in match_dict_backward and match_dict_backward[train_idx] == query_idx:
                good_matches.append(cv2.DMatch(query_idx, train_idx, 0))

        logger.info(f"双向匹配后保留 {len(good_matches)} 对特征点")

        # 空间一致性过滤（如果提供了图像尺寸）
        if img_shape and len(good_matches) > 10:
            good_matches = self._filter_spatial_consistency(kp1, kp2, good_matches, img_shape)

        return good_matches

    def _filter_spatial_consistency(self, kp1: List, kp2: List,
                                   matches: List, img_shape: Tuple[int, int],
                                   max_scale_ratio: float = 2.0) -> List:
        """空间一致性过滤 - 移除异常的比例和旋转"""
        if len(matches) < 8:
            return matches

        h, w = img_shape[:2]
        filtered_matches = []

        # 计算匹配点的尺度变化和方向变化
        scales = []
        angles = []

        for m in matches:
            pt1 = np.array(kp1[m.queryIdx].pt)
            pt2 = np.array(kp2[m.trainIdx].pt)
            diff = pt2 - pt1
            distance = np.linalg.norm(diff)
            scales.append(distance)

        scales = np.array(scales)
        median_scale = np.median(scales)

        # 只保留尺度变化在合理范围内的匹配
        for i, m in enumerate(matches):
            if 0.5 < scales[i] / (median_scale + 1e-6) < max_scale_ratio:
                filtered_matches.append(m)

        logger.info(f"空间过滤后保留 {len(filtered_matches)} / {len(matches)} 对特征点")
        return filtered_matches


def blend_with_borders(images: List[np.ndarray], masks: List[np.ndarray],
                       transforms: List[np.ndarray], show_borders: bool = True,
                       border_colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
    """带边界高亮的融合算法"""

    if not images:
        raise ValueError("图像列表为空")

    if len(images) == 1:
        return images[0]

    # 默认边界颜色（红、绿、蓝、黄、品红、青）
    if border_colors is None:
        border_colors = [
            (0, 0, 255),    # 红色
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 255, 255),  # 黄色
            (255, 0, 255),  # 品红
            (255, 255, 0),  # 青色
        ]

    # 计算画布大小
    min_x, min_y, max_x, max_y = _compute_canvas_bounds(images, transforms)
    canvas_width = max_x - min_x + 1
    canvas_height = max_y - min_y + 1

    # 创建画布和权重
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight_map = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    border_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 融合每张图像
    for idx, (img, transform) in enumerate(zip(images, transforms)):
        if transform is None:
            transform = np.eye(3)

        # 添加平移偏移
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        full_transform = translation @ transform

        # 投影变换图像
        warped = cv2.warpPerspective(
            img.astype(np.float32),
            full_transform,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # 创建权重（带边缘羽化）
        h, w = img.shape[:2]
        weight = np.ones((h, w), dtype=np.float32)

        # 边缘羽化
        edge_width = 40
        if edge_width > 0:
            gradient = np.linspace(0, 1, edge_width)
            weight[:, :edge_width] *= gradient
            weight[:, -edge_width:] *= gradient[::-1]
            weight[:edge_width, :] *= gradient[:, np.newaxis]
            weight[-edge_width:, :] *= gradient[::-1, np.newaxis]

        # 投影变换权重
        warped_weight = cv2.warpPerspective(
            weight,
            full_transform,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        warped_weight_3ch = np.dstack([warped_weight] * 3)

        # 累加图像和权重
        canvas += warped * warped_weight_3ch
        weight_map += warped_weight_3ch

        # 绘制边界
        if show_borders:
            # 计算变换后的四个角点
            corners = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype=np.float32)

            # 转换为透视变换需要的格式 (N, 1, 2)
            corners_reshaped = corners.reshape(-1, 1, 2)

            # 应用透视变换
            transformed_corners = cv2.perspectiveTransform(corners_reshaped, full_transform)

            # 转回 (4, 2) 格式
            transformed_corners = transformed_corners.reshape(4, 2)

            # 绘制边界框
            color = border_colors[idx % len(border_colors)]
            cv2.polylines(border_canvas, [transformed_corners.astype(np.int32)], True, color, 3)

            # 添加图像编号标签
            label_pos = tuple(transformed_corners[0].astype(np.int32))
            cv2.putText(border_canvas, f"Img{idx+1}", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 归一化
    weight_map_safe = np.maximum(weight_map, 1e-6)
    result = canvas / weight_map_safe
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 检查结果亮度
    mean_brightness = result.mean()
    if mean_brightness < 10:
        return images[len(images) // 2].copy()

    # 叠加边界
    if show_borders:
        # 使用alpha混合叠加边界
        result = cv2.addWeighted(result, 0.8, border_canvas, 0.3, 0)

    # 裁剪有效区域
    result = _crop_valid_region(result)

    return result


def blend_multi_band_viz(images: List[np.ndarray], masks: List[np.ndarray],
                         transforms: List[np.ndarray], feather_radius: int = 30) -> np.ndarray:
    """多频段融合，带重叠区域可视化"""

    if not images:
        raise ValueError("图像列表为空")

    if len(images) == 1:
        return images[0]

    # 计算画布大小
    min_x, min_y, max_x, max_y = _compute_canvas_bounds(images, transforms)
    canvas_width = max_x - min_x + 1
    canvas_height = max_y - min_y + 1

    # 创建画布和权重
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight_map = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)

    # 用于检测重叠区域
    overlap_count = np.zeros((canvas_height, canvas_width), dtype=np.int32)

    # 融合每张图像
    for idx, (img, transform) in enumerate(zip(images, transforms)):
        if transform is None:
            transform = np.eye(3)

        # 添加平移偏移
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        full_transform = translation @ transform

        # 投影变换图像
        warped = cv2.warpPerspective(
            img.astype(np.float32),
            full_transform,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # 创建权重（带边缘羽化）
        h, w = img.shape[:2]
        weight = np.ones((h, w), dtype=np.float32)

        # 边缘羽化
        edge_width = 50
        if edge_width > 0:
            gradient = np.linspace(0, 1, edge_width)
            weight[:, :edge_width] *= gradient
            weight[:, -edge_width:] *= gradient[::-1]
            weight[:edge_width, :] *= gradient[:, np.newaxis]
            weight[-edge_width:, :] *= gradient[::-1, np.newaxis]

        # 投影变换权重
        warped_weight = cv2.warpPerspective(
            weight,
            full_transform,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        warped_weight_3ch = np.dstack([warped_weight] * 3)

        # 记录哪些区域有图像
        mask = (warped_weight > 0.1).astype(np.uint8)
        overlap_count += mask

        # 累加图像和权重
        canvas += warped * warped_weight_3ch
        weight_map += warped_weight_3ch

    # 归一化
    weight_map_safe = np.maximum(weight_map, 1e-6)
    result = canvas / weight_map_safe
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 高亮重叠区域
    overlap_mask = (overlap_count > 1).astype(np.uint8)
    if overlap_mask.sum() > 0:
        # 创建重叠区域的高亮
        overlap_highlight = np.zeros_like(result)
        overlap_highlight[overlap_mask > 0] = [255, 255, 0]  # 黄色高亮

        # 只在重叠区域添加半透明高亮
        overlap_mask_3ch = np.dstack([overlap_mask] * 3)
        result = result.astype(np.float32) * 0.85 + overlap_highlight.astype(np.float32) * 0.15
        result = np.clip(result, 0, 255).astype(np.uint8)

    # 检查结果亮度
    mean_brightness = result.mean()
    if mean_brightness < 10:
        return images[len(images) // 2].copy()

    # 裁剪有效区域
    result = _crop_valid_region(result)

    return result


def blend_simple(images: List[np.ndarray], masks: List[np.ndarray],
                 transforms: List[np.ndarray], feather_radius: int = 30) -> np.ndarray:
    """简单的加权融合算法 - 保持原始色彩"""

    if not images:
        raise ValueError("图像列表为空")

    if len(images) == 1:
        return images[0]

    # 计算画布大小
    min_x, min_y, max_x, max_y = _compute_canvas_bounds(images, transforms)
    canvas_width = max_x - min_x + 1
    canvas_height = max_y - min_y + 1

    # 选择中间图像作为基准（保持其原始色彩）
    base_idx = len(images) // 2
    base_image = images[base_idx]
    base_transform = transforms[base_idx] if transforms[base_idx] is not None else np.eye(3)

    # 添加平移偏移
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    base_full_transform = translation @ base_transform

    # 先把基准图投影到画布上（保持原始色彩）
    canvas = cv2.warpPerspective(
        base_image,
        base_full_transform,
        (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # 创建已覆盖区域的掩膜
    coverage_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # 投影基准图的掩膜
    h, w = base_image.shape[:2]
    base_mask = np.ones((h, w), dtype=np.uint8) * 255
    warped_base_mask = cv2.warpPerspective(
        base_mask,
        base_full_transform,
        (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    coverage_mask = np.maximum(coverage_mask, warped_base_mask)

    # 融合其他图像
    for idx, (img, transform) in enumerate(zip(images, transforms)):
        if idx == base_idx:  # 跳过基准图
            continue

        if transform is None:
            transform = np.eye(3)

        full_transform = translation @ transform

        # 投影变换图像
        warped = cv2.warpPerspective(
            img,
            full_transform,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # 创建当前图像的掩膜
        img_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(
            img_mask,
            full_transform,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # 找出未覆盖的区域
        uncovered = (coverage_mask == 0).astype(np.uint8)

        # 计算重叠区域的边缘羽化
        overlap_edge = cv2.dilate(warped_mask, np.ones((feather_radius, feather_radius), np.uint8))
        overlap_edge = cv2.erode(overlap_edge, np.ones((feather_radius, feather_radius), np.uint8))
        overlap_edge = overlap_edge - coverage_mask

        # 在重叠区域做平滑融合
        overlap_mask = (warped_mask > 0) & (coverage_mask > 0)
        if overlap_mask.any():
            # 创建羽化权重
            overlap_weight = warped_mask.astype(np.float32) / 255.0
            kernel = cv2.getGaussianKernel(feather_radius * 2, feather_radius / 3)
            overlap_weight = cv2.filter2D(overlap_weight, -1, kernel.reshape(-1, 1))

            # 在重叠区域平滑融合
            overlap_region = (overlap_mask > 0)
            for c in range(3):
                canvas[overlap_region, c] = (
                    canvas[overlap_region, c].astype(np.float32) * 0.5 +
                    warped[overlap_region, c].astype(np.float32) * 0.5
                ).astype(np.uint8)

        # 在未覆盖区域直接复制
        if uncovered.any():
            canvas[uncovered > 0] = warped[uncovered > 0]

        # 更新覆盖掩膜
        coverage_mask = np.maximum(coverage_mask, warped_mask)

    # 检查结果亮度
    mean_brightness = canvas.mean()
    if mean_brightness < 10:
        return images[base_idx].copy()

    # 裁剪有效区域
    result = _crop_valid_region(canvas)

    return result


def blend_multi_band(images: List[np.ndarray], masks: List[np.ndarray],
                     transforms: List[np.ndarray], feather_radius: int = 30) -> np.ndarray:
    """多频段融合算法，提供更自然的融合效果"""

    if not images:
        raise ValueError("图像列表为空")

    if len(images) == 1:
        return images[0]

    try:
        # 计算画布大小
        min_x, min_y, max_x, max_y = _compute_canvas_bounds(images, transforms)

        canvas_width = max_x - min_x + 1
        canvas_height = max_y - min_y + 1

        # 创建拉普拉斯金字塔
        levels = 6  # 金字塔层数

        # 为每张图像创建拉普拉斯金字塔
        pyramids = []
        weight_pyramids = []

        for idx, (img, transform) in enumerate(zip(images, transforms)):
            if transform is None:
                transform = np.eye(3)

            # 添加平移偏移
            translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
            full_transform = translation @ transform

            # 投影变换
            warped = cv2.warpPerspective(
                img.astype(np.float32),
                full_transform,
                (canvas_width, canvas_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # 变换掩膜
            mask_array = masks[idx].astype(np.float32) if idx < len(masks) else np.ones_like(img[:, :, 0], dtype=np.float32)
            warped_mask = cv2.warpPerspective(
                mask_array,
                full_transform,
                (canvas_width, canvas_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # 创建高斯金字塔
            gaussian_pyramid = [warped]
            weight_pyramid = [warped_mask]

            for _ in range(levels - 1):
                warped = cv2.pyrDown(gaussian_pyramid[-1])
                warped_mask = cv2.pyrDown(weight_pyramid[-1])
                gaussian_pyramid.append(warped)
                weight_pyramid.append(warped_mask)

            # 创建拉普拉斯金字塔
            laplacian_pyramid = []
            for i in range(levels - 1):
                size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
                upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
                laplacian = gaussian_pyramid[i] - upsampled
                laplacian_pyramid.append(laplacian)
            laplacian_pyramid.append(gaussian_pyramid[-1])

            pyramids.append(laplacian_pyramid)
            weight_pyramids.append(weight_pyramid)

        # 融合金字塔
        fused_pyramid = []

        for level in range(levels):
            # 计算权重
            level_canvas = np.zeros_like(pyramids[0][level])

            for img_pyramid, weight_pyramid in zip(pyramids, weight_pyramids):
                if level < len(weight_pyramid):
                    weight = weight_pyramid[level].copy()

                    # 确保weight是二维数组
                    if len(weight.shape) == 3:
                        weight = weight[:, :, 0]  # 取第一个通道

                    # 羽化处理
                    if feather_radius > 0:
                        kernel_size = max(1, feather_radius // (2 ** level) * 2 + 1)
                        if kernel_size > 1:
                            weight = cv2.GaussianBlur(weight, (kernel_size, kernel_size), 0)

                    weight = np.clip(weight, 0, 1)

                    # 归一化权重 - 修复数组比较问题
                    weight_sum = np.sum(weight, axis=0, keepdims=True) + 1e-6
                    # 避免除零和数值不稳定
                    weight_sum = np.maximum(weight_sum, 1e-6)
                    weight = weight / weight_sum

                    # 融合
                    for c in range(3):
                        level_canvas[:, :, c] += img_pyramid[level][:, :, c] * weight

            fused_pyramid.append(level_canvas)

        # 重建图像
        result = fused_pyramid[-1]
        for level in range(levels - 2, -1, -1):
            size = (fused_pyramid[level].shape[1], fused_pyramid[level].shape[0])
            result = cv2.pyrUp(result, dstsize=size) + fused_pyramid[level]

        result = np.clip(result, 0, 255).astype(np.uint8)

        # 裁剪有效区域
        result = _crop_valid_region(result)

        return result

    except Exception as e:
        import traceback
        error_msg = f"多频段融合失败: {str(e)}\n{traceback.format_exc()}"
        raise RuntimeError(error_msg) from e


def blend_no_blend(images: List[np.ndarray], masks: List[np.ndarray],
                   transforms: List[np.ndarray], feather_radius: int = 30) -> np.ndarray:
    """不融合的拼接算法 - 完全保持原始色彩"""

    if not images:
        raise ValueError("图像列表为空")

    if len(images) == 1:
        return images[0]

    # 计算画布大小
    min_x, min_y, max_x, max_y = _compute_canvas_bounds(images, transforms)
    canvas_width = max_x - min_x + 1
    canvas_height = max_y - min_y + 1

    # 创建空画布
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 添加平移偏移
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    # 逐个放置图像，保持原始色彩（最后写入优先）
    for img, transform in zip(images, transforms):
        if transform is None:
            transform = np.eye(3)

        full_transform = translation @ transform

        # 投影变换图像到画布位置
        warped = cv2.warpPerspective(
            img,
            full_transform,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # 找出非黑色像素（有内容的区域）
        mask = (warped > 0).any(axis=2)

        # 直接复制像素（覆盖已有内容，不融合）
        canvas[mask] = warped[mask]

    # 裁剪有效区域
    result = _crop_valid_region(canvas)

    return result


def _compute_canvas_bounds(images: List[np.ndarray], transforms: List[np.ndarray]) -> Tuple[int, int, int, int]:
    """计算画布边界"""
    all_corners = []

    for img, transform in zip(images, transforms):
        h, w = img.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])

        if transform is not None:
            transformed = (transform @ corners.T).T
            transformed = transformed[:, :2] / transformed[:, 2, np.newaxis]
            all_corners.extend(transformed.tolist())
        else:
            all_corners.extend(corners[:, :2].tolist())

    all_corners = np.array(all_corners)

    min_x = int(np.floor(all_corners[:, 0].min()))
    min_y = int(np.floor(all_corners[:, 1].min()))
    max_x = int(np.ceil(all_corners[:, 0].max()))
    max_y = int(np.ceil(all_corners[:, 1].max()))

    return min_x, min_y, max_x, max_y


def _crop_valid_region(image: np.ndarray) -> np.ndarray:
    """裁剪有效区域"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 找到最大连通区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return image[y:y+h, x:x+w]

    return image


class CompatibleImprovedStitcher:
    """兼容原有接口的改进拼接器"""

    def __init__(self, viz_mode: str = "auto"):
        self.preprocessor = DentalImagePreprocessor()
        self.feature_matcher = DentalFeatureMatcher()
        self.viz_mode = viz_mode  # 可视化模式: auto, seamless, borders, overlap
        self.logs: List[str] = []

    def precheck(self, records: List[ImageRecord]) -> PrecheckReport:
        """预检查图像质量"""
        items: list[PrecheckItem] = []
        kept_indices: list[int] = []
        dropped_indices: list[int] = []

        sharpness_values = []
        for record in records:
            sharpness, exposure = compute_image_metrics(record.image)
            record.sharpness_score = sharpness
            record.exposure_score = exposure
            record.quality_score = combine_quality_score(sharpness, exposure)
            sharpness_values.append(sharpness)

        median_sharpness = float(np.median(sharpness_values)) if sharpness_values else 0.0
        sharpness_threshold = max(25.0, median_sharpness * 0.22)

        for idx, record in enumerate(records):
            keep = True
            reasons = []
            if record.sharpness_score < sharpness_threshold:
                keep = False
                reasons.append("模糊")
            if record.exposure_score < 0.25:
                reasons.append("曝光偏离较大")

            if keep:
                kept_indices.append(idx)
            else:
                dropped_indices.append(idx)

            items.append(
                PrecheckItem(
                    index=idx,
                    display_name=record.display_name,
                    sharpness_score=record.sharpness_score,
                    exposure_score=record.exposure_score,
                    quality_score=record.quality_score,
                    keep=keep,
                    reason="、".join(reasons) if reasons else "保留",
                )
            )

        return PrecheckReport(items=items, kept_indices=kept_indices, dropped_indices=dropped_indices)

    def score_candidates(self, records: List[ImageRecord]) -> Tuple[List[CandidateScore], List[str]]:
        """评估基准图候选"""
        logs: List[str] = []
        if len(records) < 2:
            return [], ["至少需要两张图像才能评估基准图。"]

        logs.append("开始评估基准图候选...")

        # 预处理所有图像
        processed_images = []
        for record in records:
            enhanced, mask = self.preprocessor.preprocess(record.image)
            processed_images.append((enhanced, mask))

        # 检测特征
        all_keypoints = []
        all_descriptors = []

        for enhanced, mask in processed_images:
            kp, desc = self.feature_matcher.detect_and_compute(enhanced, mask)
            all_keypoints.append(kp)
            all_descriptors.append(desc)
            logs.append(f"检测到 {len(kp)} 个特征点")

        # 计算两两匹配得分
        match_scores = {}
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if all_descriptors[i] is None or all_descriptors[j] is None:
                    continue

                matches = self.feature_matcher.match_features(
                    all_descriptors[i], all_descriptors[j],
                    method='orb' if all_descriptors[i].shape[1] == 32 else 'sift'
                )

                score = len(matches) if matches else 0
                match_scores[(i, j)] = score
                match_scores[(j, i)] = score
                logs.append(f"图像 {i} 和 {j} 匹配得分: {score}")

        # 评估每个图像作为基准的得分
        candidates = []
        for i, record in enumerate(records):
            connectivity_score = 0.0
            partner_count = 0

            for j in range(len(records)):
                if i == j:
                    continue
                if (i, j) in match_scores:
                    connectivity_score += match_scores[(i, j)]
                    partner_count += 1

            # 归一化得分
            if partner_count > 0:
                connectivity_score = connectivity_score / partner_count

            # 总分 = 连通性 * 0.7 + 质量 * 0.3
            total_score = connectivity_score * 0.7 + record.quality_score * 0.3

            candidates.append(CandidateScore(
                index=i,
                display_name=record.display_name,
                quality_score=record.quality_score,
                connectivity_score=connectivity_score,
                partner_count=partner_count,
                total_score=total_score,
                recommended=(i == 0)  # 默认第一个为推荐
            ))

        # 按总分排序
        candidates.sort(key=lambda x: x.total_score, reverse=True)

        # 标记推荐的基准图
        if candidates:
            candidates[0].recommended = True
            logs.append(f"推荐基准图: {candidates[0].display_name}")

        return candidates, logs

    def stitch(self, records: List[ImageRecord],
               anchor_index_override: Optional[int] = None) -> StitchResult:
        """执行拼接"""
        self.logs = []

        if len(records) < 2:
            return StitchResult(
                success=False,
                anchor_index=None,
                panorama=None,
                logs=self.logs + ["至少需要两张图像"],
                method_name="Improved Dental Stitcher v2.0"
            )

        try:
            # 直接使用原始图像，不做预处理
            self.logs.append("准备图像数据...")
            original_images = [record.image for record in records]

            # 选择基准图
            anchor_idx = self._select_anchor(records, anchor_index_override)
            self.logs.append(f"选择基准图: {records[anchor_idx].display_name}")

            # 检测特征（在原始图像上）
            self.logs.append("检测图像特征...")
            all_keypoints = []
            all_descriptors = []

            for img in original_images:
                # 简单的灰度转换
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                mask = np.ones(gray.shape, dtype=np.uint8) * 255

                kp, desc = self.feature_matcher.detect_and_compute(img, mask)
                all_keypoints.append(kp)
                all_descriptors.append(desc)
                self.logs.append(f"检测到 {len(kp)} 个特征点")

            # 计算变换矩阵并保存匹配结果
            self.logs.append("计算图像变换...")
            transforms, pairwise_matches = self._compute_transforms_with_matches(
                all_keypoints, all_descriptors, anchor_idx, original_images
            )

            # 创建简单的掩膜（全1）
            dummy_masks = [np.ones(img.shape[:2], dtype=np.uint8) for img in original_images]

            # 根据图像数量和用户选择选择融合策略
            num_images = len(original_images)
            blend_mode = ""

            if self.viz_mode == "边界高亮":
                self.logs.append("执行带边界高亮的融合...")
                result = blend_with_borders(original_images, dummy_masks, transforms, show_borders=True)
                blend_mode = "边界高亮融合"

            elif self.viz_mode == "重叠区域":
                self.logs.append("执行重叠区域可视化融合...")
                result = blend_multi_band_viz(original_images, dummy_masks, transforms)
                blend_mode = "重叠区域可视化"

            elif self.viz_mode == "无缝融合":
                self.logs.append("执行无缝融合...")
                result = blend_simple(original_images, dummy_masks, transforms)
                blend_mode = "无缝融合"

            elif self.viz_mode == "no_blend":
                self.logs.append("执行原始色彩拼接（不融合）...")
                result = blend_no_blend(original_images, dummy_masks, transforms)
                blend_mode = "原始色彩"

            else:  # auto
                if num_images <= 3:
                    self.logs.append("执行无缝融合（推荐用于少量图像）...")
                    result = blend_simple(original_images, dummy_masks, transforms)
                    blend_mode = "无缝融合"
                elif num_images <= 5:
                    self.logs.append("执行带边界高亮的融合...")
                    result = blend_with_borders(original_images, dummy_masks, transforms, show_borders=True)
                    blend_mode = "边界高亮融合"
                else:
                    self.logs.append("执行重叠区域可视化融合...")
                    result = blend_multi_band_viz(original_images, dummy_masks, transforms)
                    blend_mode = "重叠区域可视化"

            self.logs.append(f"拼接完成，结果尺寸: {result.shape}")

            return StitchResult(
                success=True,
                anchor_index=anchor_idx,
                panorama=result,
                logs=self.logs,
                method_name=f"Improved Dental Stitcher v2.0 ({blend_mode})",
                included_indices=list(range(len(records))),
                ordered_indices=list(range(len(records))),
                pairwise_matches=pairwise_matches
            )

        except Exception as e:
            self.logs.append(f"拼接失败: {str(e)}")
            import traceback
            self.logs.append(f"详细错误: {traceback.format_exc()}")
            logger.error(f"拼接错误: {e}", exc_info=True)
            return StitchResult(
                success=False,
                anchor_index=None,
                panorama=None,
                logs=self.logs,
                method_name="Improved Dental Stitcher v2.0"
            )

    def _select_anchor(self, records: List[ImageRecord],
                      override_idx: Optional[int] = None) -> int:
        """选择最佳基准图"""
        if override_idx is not None and 0 <= override_idx < len(records):
            return override_idx

        if len(records) == 0:
            return 0

        # 选择质量分数最高的图像作为基准
        best_idx = 0
        best_score = -1

        for i, record in enumerate(records):
            score = record.quality_score
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _compute_transforms(self, keypoints: List, descriptors: List,
                           anchor_idx: int) -> List[np.ndarray]:
        """计算所有图像相对于基准图的变换"""
        transforms = []
        anchor_kp = keypoints[anchor_idx]
        anchor_desc = descriptors[anchor_idx]

        for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            if i == anchor_idx:
                transforms.append(np.eye(3))
                continue

            # 匹配特征
            matches = self.feature_matcher.match_features(
                anchor_desc, desc,
                method='orb' if desc.shape[1] == 32 else 'sift'
            )

            self.logs.append(f"图像 {i} 与基准图匹配到 {len(matches)} 对特征点")

            if len(matches) >= 10:
                # 估计单应性矩阵
                src_pts = np.float32([anchor_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if homography is not None:
                    inliers = int(mask.sum())
                    self.logs.append(f"图像 {i} 单应性矩阵估计成功，内点数: {inliers}")
                    transforms.append(homography)
                else:
                    # 尝试仿射变换
                    affine, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
                    if affine is not None:
                        self.logs.append(f"图像 {i} 使用仿射变换，内点数: {int(mask.sum())}")
                        homography = np.vstack([affine, [0, 0, 1]])
                        transforms.append(homography)
                    else:
                        self.logs.append(f"图像 {i} 变换估计失败，使用单位矩阵")
                        transforms.append(np.eye(3))
            else:
                self.logs.append(f"图像 {i} 匹配点不足，使用单位矩阵")
                transforms.append(np.eye(3))

        return transforms

    def _compute_transforms_with_matches(self, keypoints: List, descriptors: List,
                                       anchor_idx: int, images: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        """计算所有图像相对于基准图的变换，并返回匹配结果"""
        transforms = []
        pairwise_matches = {}
        anchor_kp = keypoints[anchor_idx]
        anchor_desc = descriptors[anchor_idx]

        for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            if i == anchor_idx:
                transforms.append(np.eye(3))
                continue

            # 使用鲁棒匹配
            matches = self.feature_matcher.match_features_robust(
                anchor_desc, desc,
                anchor_kp, kp,
                method='orb' if desc.shape[1] == 32 else 'sift',
                img_shape=images[i].shape
            )

            self.logs.append(f"图像 {i} 与基准图匹配到 {len(matches)} 对特征点")

            # 创建MatchResult
            match_result = MatchResult(
                success=False,
                score=0.0,
                inliers=0,
                homography=None,
                inverse_homography=None,
                details={},
                sequence_distance=abs(i - anchor_idx),
                weighted_score=0.0,
                matched_points0=None,
                matched_points1=None
            )

            if len(matches) >= 4:
                # 使用鲁棒的变换估计
                homography, inliers = self._estimate_transform_robust(
                    anchor_kp, kp, matches, i
                )

                if homography is not None and inliers >= 4:
                    # 提取内点用于可视化
                    src_pts = np.float32([anchor_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    # 重新计算掩膜以获取内点
                    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        inlier_mask = mask.ravel().astype(bool)
                        if inlier_mask.sum() > 0:
                            matched_pts0 = src_pts[inlier_mask].reshape(-1, 2)
                            matched_pts1 = dst_pts[inlier_mask].reshape(-1, 2)

                            # 限制显示的点数
                            max_display_points = 50
                            if len(matched_pts0) > max_display_points:
                                indices = np.linspace(0, len(matched_pts0) - 1, max_display_points, dtype=int)
                                matched_pts0 = matched_pts0[indices]
                                matched_pts1 = matched_pts1[indices]

                            match_result = MatchResult(
                                success=True,
                                score=float(inliers),
                                inliers=inliers,
                                homography=homography,
                                inverse_homography=np.linalg.inv(homography),
                                details={"method": "robust_multi_stage"},
                                sequence_distance=abs(i - anchor_idx),
                                weighted_score=float(inliers * 0.8 + len(matches) * 0.2),
                                matched_points0=matched_pts0.astype(np.float32),
                                matched_points1=matched_pts1.astype(np.float32)
                            )
                        else:
                            match_result.inliers = inliers
                            match_result.score = float(inliers)
                    else:
                        match_result.inliers = inliers
                        match_result.score = float(inliers)

                    transforms.append(homography)
                else:
                    self.logs.append(f"图像 {i} 变换估计失败，使用单位矩阵")
                    transforms.append(np.eye(3))
            else:
                self.logs.append(f"图像 {i} 匹配点不足（{len(matches)} < 4），使用单位矩阵")
                transforms.append(np.eye(3))

            # 保存匹配结果
            pair_key = (min(anchor_idx, i), max(anchor_idx, i))
            pairwise_matches[pair_key] = match_result

        return transforms, pairwise_matches

    def _estimate_transform_robust(self, kp1: List, kp2: List,
                                   matches: List, img_idx: int) -> Tuple[Optional[np.ndarray], int]:
        """鲁棒的变换估计 - 多级回退策略"""

        if len(matches) < 4:
            self.logs.append(f"图像 {img_idx} 匹配点不足（{len(matches)} < 4），无法估计变换")
            return None, 0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 方法1: RANSAC单应性（严格）
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if H is not None:
            inliers = int(mask.sum())
            inlier_ratio = inliers / len(mask)
            if inlier_ratio > 0.2 and inliers >= 4:
                self.logs.append(f"图像 {img_idx} RANSAC单应性成功，内点: {inliers}/{len(mask)} ({inlier_ratio:.1%})")
                return H, inliers

        # 方法2: RHO单应性（对异常值更鲁棒）
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO, 5.0, maxIters=5000)
            if H is not None:
                inliers = int(mask.sum())
                inlier_ratio = inliers / len(mask)
                if inlier_ratio > 0.15 and inliers >= 4:
                    self.logs.append(f"图像 {img_idx} RHO单应性成功，内点: {inliers}/{len(mask)} ({inlier_ratio:.1%})")
                    return H, inliers
        except Exception as e:
            self.logs.append(f"图像 {img_idx} RHO单应性失败: {e}")

        # 方法3: LMEDS单应性（最小中位数）
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
        if H is not None:
            inliers = int(mask.sum())
            inlier_ratio = inliers / len(mask)
            if inlier_ratio > 0.1 and inliers >= 4:
                self.logs.append(f"图像 {img_idx} LMEDS单应性成功，内点: {inliers}/{len(mask)} ({inlier_ratio:.1%})")
                return H, inliers

        # 方法4: 部分仿射（旋转+平移+缩放）
        A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if A is not None:
            inliers = int(mask.sum())
            inlier_ratio = inliers / len(mask)
            if inlier_ratio > 0.1 and inliers >= 4:
                self.logs.append(f"图像 {img_idx} 部分仿射成功，内点: {inliers}/{len(mask)} ({inlier_ratio:.1%})")
                H = np.vstack([A, [0, 0, 1]])
                return H, inliers

        # 方法5: 完全仿射
        A, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
        if A is not None:
            inliers = int(mask.sum())
            inlier_ratio = inliers / len(mask)
            if inlier_ratio > 0.1 and inliers >= 4:
                self.logs.append(f"图像 {img_idx} 完全仿射成功，内点: {inliers}/{len(mask)} ({inlier_ratio:.1%})")
                H = np.vstack([A, [0, 0, 1]])
                return H, inliers

        # 方法6: 基于质心的相似变换（最后的回退）
        H = self._estimate_similarity_transform(src_pts, dst_pts)
        if H is not None:
            self.logs.append(f"图像 {img_idx} 使用相似变换作为回退")
            return H, len(matches)

        self.logs.append(f"图像 {img_idx} 所有变换估计方法都失败")
        return None, 0

    def _estimate_similarity_transform(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
        """估计相似变换（旋转+缩放+平移）"""
        try:
            # 计算质心
            src_centroid = np.mean(src_pts, axis=0)
            dst_centroid = np.mean(dst_pts, axis=0)

            # 中心化
            src_centered = src_pts - src_centroid
            dst_centered = dst_pts - dst_centroid

            # 估计旋转和缩放
            H = dst_centered.T @ src_centered
            U, S, Vt = np.linalg.svd(H)
            R = U @ Vt

            # 确保是旋转矩阵（det=1）
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = U @ Vt

            # 缩放因子
            scale = np.sum(S) / np.sum(np.linalg.norm(src_centered, axis=1))

            # 构建相似变换矩阵
            transform = np.eye(3, dtype=np.float64)
            transform[:2, :2] = scale * R
            transform[:2, 2] = dst_centroid.flatten() - scale * R @ src_centroid.flatten()

            return transform
        except Exception as e:
            logger.warning(f"相似变换估计失败: {e}")
            return None

    def _compute_transforms(self, keypoints: List, descriptors: List,
                           anchor_idx: int) -> List[np.ndarray]:
        """计算所有图像相对于基准图的变换"""
        transforms = []
        anchor_kp = keypoints[anchor_idx]
        anchor_desc = descriptors[anchor_idx]

        for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
            if i == anchor_idx:
                transforms.append(np.eye(3))
                continue

            # 匹配特征
            matches = self.feature_matcher.match_features(
                anchor_desc, desc,
                method='orb' if desc.shape[1] == 32 else 'sift'
            )

            self.logs.append(f"图像 {i} 与基准图匹配到 {len(matches)} 对特征点")

            if len(matches) >= 10:
                # 估计单应性矩阵
                src_pts = np.float32([anchor_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if homography is not None:
                    inliers = int(mask.sum())
                    self.logs.append(f"图像 {i} 单应性矩阵估计成功，内点数: {inliers}")
                    transforms.append(homography)
                else:
                    # 尝试仿射变换
                    affine, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
                    if affine is not None:
                        self.logs.append(f"图像 {i} 使用仿射变换，内点数: {int(mask.sum())}")
                        homography = np.vstack([affine, [0, 0, 1]])
                        transforms.append(homography)
                    else:
                        self.logs.append(f"图像 {i} 变换估计失败，使用单位矩阵")
                        transforms.append(np.eye(3))
            else:
                self.logs.append(f"图像 {i} 匹配点不足，使用单位矩阵")
                transforms.append(np.eye(3))

        return transforms