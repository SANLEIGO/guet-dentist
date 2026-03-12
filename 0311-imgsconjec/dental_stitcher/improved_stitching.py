from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import math

from dental_stitcher.models import ImageRecord, MatchResult, StitchResult
from dental_stitcher.utils import combine_quality_score, compute_image_metrics


class DentalImageEnhancer:
    """专门用于口腔图像的预处理和增强"""

    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """增强图像质量，改善视觉效果"""
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE增强亮度通道
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # 自适应增强牙齿区域
        l = DentalImageEnhancer._enhance_teeth_region(l, image)

        # 平滑处理
        l = cv2.bilateralFilter(l, 9, 75, 75)

        # 合并通道
        enhanced_lab = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _enhance_teeth_region(l_channel: np.ndarray, original: np.ndarray) -> np.ndarray:
        """增强牙齿区域"""
        # 创建牙齿掩膜
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

        # 牙齿的HSV范围
        tooth_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))

        # 形态学操作去除噪点
        kernel = np.ones((3, 3), np.uint8)
        tooth_mask = cv2.morphologyEx(tooth_mask, cv2.MORPH_OPEN, kernel)
        tooth_mask = cv2.morphologyEx(tooth_mask, cv2.MORPH_CLOSE, kernel)

        # 增强牙齿区域
        enhanced = l_channel.copy()
        enhanced[tooth_mask > 0] = np.clip(enhanced[tooth_mask > 0] * 1.1, 0, 255)

        return enhanced.astype(np.uint8)

    @staticmethod
    def color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """色彩迁移，使目标图像色彩与源图像一致"""
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

        # 计算均值和标准差
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)

        # 调整目标图像
        adjusted_lab = target_lab.copy()
        for i in range(3):
            if target_std[i, 0] > 0:
                adjusted_lab[:, :, i] = (adjusted_lab[:, :, i] - target_mean[i, 0]) * \
                                     (source_std[i, 0] / target_std[i, 0]) + source_mean[i, 0]

        adjusted_lab = np.clip(adjusted_lab, 0, 255)
        return cv2.cvtColor(adjusted_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


class DentalFeatureDetector:
    """改进的牙齿特征检测器"""

    def __init__(self):
        # 使用SIFT作为主要特征检测器
        self.sift = cv2.SIFT_create(
            nfeatures=2000,
            nOctaveLayers=8,
            contrastThreshold=0.02,
            edgeThreshold=10,
            sigma=1.6
        )
        # AKAZE作为备用
        self.akaze = cv2.AKAZE_create()

    def detect_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """检测图像特征点"""
        # 使用注意力掩膜
        mask = self._create_teeth_attention_mask(image)

        # 优先使用SIFT
        try:
            keypoints, descriptors = self.sift.detectAndCompute(image, mask)
            if descriptors is not None and len(keypoints) >= 30:
                return keypoints, descriptors
        except:
            pass

        # 退到AKAZE
        keypoints, descriptors = self.akaze.detectAndCompute(image, mask)
        return keypoints, descriptors

    def _create_teeth_attention_mask(self, image: np.ndarray) -> np.ndarray:
        """创建牙齿区域的注意力掩膜"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 排除黑色背景
        bg_mask = gray > 30

        # 牙齿区域
        tooth_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))

        # 排除软组织（粉色区域）
        soft_tissue = cv2.inRange(hsv, np.array([0, 30, 120]), np.array([25, 170, 255]))
        soft_tissue |= cv2.inRange(hsv, np.array([165, 30, 120]), np.array([180, 170, 255]))

        # 组合掩膜
        attention = bg_mask & tooth_mask & (~soft_tissue)

        # 平滑处理
        attention = cv2.GaussianBlur(attention.astype(np.float32), (5, 5), 2).astype(np.uint8)
        attention = (attention > 50).astype(np.uint8) * 255

        return attention


class DentalStitcher:
    """改进的口腔图像拼接器"""

    def __init__(self):
        self.enhancer = DentalImageEnhancer()
        self.feature_detector = DentalFeatureDetector()

    def stitch_images(self, images: List[np.ndarray],
                     homographies: List[np.ndarray]) -> np.ndarray:
        """执行多图像拼接"""
        if not images or len(images) < 2:
            raise ValueError("至少需要两张图像")

        # 图像增强
        enhanced_images = [self.enhancer.enhance_image(img) for img in images]

        # 计算画布大小
        H, W = enhanced_images[0].shape[:2]
        corners = [[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]]

        # 计算所有图像的边界
        all_corners = []
        for i, img in enumerate(enhanced_images):
            h, w = img.shape[:2]
            img_corners = [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]

            # 应用变换
            if i < len(homographies):
                transformed = []
                for corner in img_corners:
                    x = homographies[i] @ np.array(corner)
                    transformed.append([x[0]/x[2], x[1]/x[2]])
                all_corners.extend(transformed)
            else:
                all_corners.extend(img_corners)

        if all_corners:
            # 计算边界框
            all_corners = np.array(all_corners)
            min_x = int(np.min(all_corners[:, 0]))
            max_x = int(np.max(all_corners[:, 0]))
            min_y = int(np.min(all_corners[:, 1]))
            max_y = int(np.max(all_corners[:, 1]))

            # 调整画布大小
            canvas_width = max_x - min_x + 1
            canvas_height = max_y - min_y + 1

            # 创建画布
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            weight_canvas = np.zeros((canvas_height, canvas_width, 1), dtype=np.float32)

            # 混合图像
            for i, img in enumerate(enhanced_images):
                h, w = img.shape[:2]
                warped_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)

                # 创建变换矩阵（考虑画布偏移）
                translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

                if i < len(homographies):
                    M = homographies[i] @ translation
                else:
                    M = translation

                # 投影变换
                cv2.warpPerspective(img, M, (canvas_width, canvas_height),
                                 warped_img, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

                # 创建权重图（距离加权）
                weight_map = self._create_weight_map(img.shape[:2], M, (canvas_width, canvas_height))

                # 混合到画布
                canvas += warped_img * weight_map[..., np.newaxis]
                weight_canvas += weight_map[..., np.newaxis]

            # 最终混合
            mask = weight_canvas > 0
            canvas = canvas / np.clip(weight_canvas, 1e-6, None)
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)

            # 裁剪有效区域
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                canvas = canvas[y:y+h, x:x+w]

            return canvas

        return images[0]

    def _create_weight_map(self, img_shape: Tuple[int, int], M: np.ndarray,
                          canvas_size: Tuple[int, int]) -> np.ndarray:
        """创建渐变权重图"""
        h, w = img_shape
        canvas_w, canvas_h = canvas_size

        # 创建坐标网格
        x_coords = np.arange(w)
        y_coords = np.arange(h)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # 应用变换
        points = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx.ravel())], axis=1)
        transformed = (M @ points.T).T
        transformed = transformed[:, :2] / transformed[:, 2, np.newaxis]

        # 计算权重
        xx_trans = transformed[:, 0].reshape(h, w)
        yy_trans = transformed[:, 1].reshape(h, w)

        # 检查点是否在画布内
        in_canvas = (xx_trans >= 0) & (xx_trans < canvas_w) & (yy_trans >= 0) & (yy_trans < canvas_h)

        # 距离边缘的权重
        edge_weight = np.zeros((h, w))
        in_canvas_float = in_canvas.astype(np.float32)

        # 计算到画布边的距离
        dist_left = np.minimum(xx_trans, canvas_w - xx_trans)
        dist_top = np.minimum(yy_trans, canvas_h - yy_trans)
        min_dist = np.minimum(dist_left, dist_top)

        # 渐变权重
        edge_weight = np.clip(min_dist / 20.0, 0, 1) * in_canvas_float

        # 高斯模糊
        edge_weight = cv2.GaussianBlur(edge_weight, (5, 5), 2)

        return edge_weight


def multi_image_stitching(images: List[np.ndarray],
                         anchor_idx: int = 0) -> Tuple[np.ndarray, List[np.ndarray]]:
    """多图像拼接的主函数

    Args:
        images: 图像列表
        anchor_idx: 基准图索引

    Returns:
        tuple: (拼接结果, 单应性变换矩阵列表)
    """
    stitcher = DentalStitcher()

    if len(images) < 2:
        return images[0], []

    # 特征检测和匹配
    keypoints_list = []
    descriptors_list = []

    for img in images:
        kp, desc = stitcher.feature_detector.detect_features(img)
        keypoints_list.append(kp)
        descriptors_list.append(desc)

    # 估计单应性矩阵
    homographies = []
    for i in range(len(images)):
        if i == anchor_idx:
            homographies.append(np.eye(3))
        else:
            # 匹配基准图和当前图
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            matches = matcher.knnMatch(descriptors_list[anchor_idx], descriptors_list[i], k=2)

            # Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) >= 10:
                src_pts = np.float32([keypoints_list[anchor_idx][m.queryIdx].pt
                                    for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_list[i][m.trainIdx].pt
                                    for m in good_matches]).reshape(-1, 1, 2)

                # 估计单应性矩阵
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    homographies.append(H)
                else:
                    homographies.append(np.eye(3))
            else:
                homographies.append(np.eye(3))

    # 执行拼接
    result = stitcher.stitch_images(images, homographies)

    return result, homographies