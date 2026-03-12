from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

from dental_stitcher.models import ImageRecord, MatchResult, StitchResult, PrecheckItem, PrecheckReport
from dental_stitcher.utils import combine_quality_score, compute_image_metrics

logger = logging.getLogger(__name__)


@dataclass
class DentalImageData:
    """预处理后的牙齿图像数据"""
    image: np.ndarray
    enhanced_image: np.ndarray
    gray_image: np.ndarray
    teeth_mask: np.ndarray
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None
    histogram: Optional[np.ndarray] = None


class DentalImagePreprocessor:
    """专门的牙齿图像预处理类"""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def preprocess(self, image: np.ndarray) -> DentalImageData:
        """完整的图像预处理流程"""
        # 原始图像
        original = image.copy()

        # 创建牙齿掩膜
        teeth_mask = self._extract_teeth_mask(image)

        # 增强图像
        enhanced = self._enhance_image(image, teeth_mask)

        # 灰度图
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # 计算直方图用于色彩匹配
        histogram = self._compute_histogram(enhanced, teeth_mask)

        return DentalImageData(
            image=original,
            enhanced_image=enhanced,
            gray_image=gray,
            teeth_mask=teeth_mask,
            histogram=histogram
        )

    def _extract_teeth_mask(self, image: np.ndarray) -> np.ndarray:
        """提取牙齿区域掩膜"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 背景掩膜
        bg_mask = gray > 30

        # 牙齿区域（白色/浅色）
        tooth_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))

        # 排除软组织（粉色区域）
        soft_tissue = cv2.inRange(hsv, np.array([0, 30, 120]), np.array([25, 170, 255]))
        soft_tissue |= cv2.inRange(hsv, np.array([165, 30, 120]), np.array([180, 170, 255]))

        # 边缘增强
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        # 组合掩膜
        attention = bg_mask & tooth_mask & (~soft_tissue)
        attention = cv2.morphologyEx(attention, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        attention = cv2.morphologyEx(attention, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # 融合边缘信息
        attention = cv2.bitwise_or(attention, edges)

        return attention

    def _enhance_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """增强图像质量"""
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 只对牙齿区域增强
        teeth_l = l.copy()

        # CLAHE增强
        enhanced_l = self.clahe.apply(teeth_l)

        # 融合原始和增强的亮度
        mask_float = mask.astype(np.float32) / 255.0
        blended_l = (teeth_l.astype(np.float32) * (1 - mask_float * 0.7) +
                    enhanced_l.astype(np.float32) * mask_float * 0.7).astype(np.uint8)

        # 降噪
        blended_l = cv2.bilateralFilter(blended_l, 9, 75, 75)

        # 合并通道
        enhanced_lab = cv2.merge([blended_l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def _compute_histogram(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """计算直方图用于色彩匹配"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist


class DentalFeatureMatcher:
    """专门针对牙齿图像的特征匹配器"""

    def __init__(self):
        # 使用SIFT，对尺度和旋转变化更鲁棒
        self.sift = cv2.SIFT_create(
            nfeatures=3000,
            nOctaveLayers=8,
            contrastThreshold=0.02,
            edgeThreshold=10,
            sigma=1.6
        )
        # 使用ORB作为备用
        self.orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )

    def detect_and_compute(self, image_data: DentalImageData) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """检测特征点并计算描述子"""
        gray = image_data.gray_image
        mask = image_data.teeth_mask

        # 优先使用SIFT
        try:
            keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
            if descriptors is not None and len(keypoints) >= 50:
                return keypoints, descriptors
        except Exception as e:
            logger.warning(f"SIFT检测失败: {e}")

        # 退到ORB
        try:
            keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
            if descriptors is not None and len(keypoints) >= 30:
                return keypoints, descriptors
        except Exception as e:
            logger.warning(f"ORB检测失败: {e}")

        return [], None

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray,
                      method: str = 'sift') -> List[cv2.DMatch]:
        """匹配两组特征描述子"""
        if desc1 is None or desc2 is None:
            return []

        if method == 'orb':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # KNN匹配
        matches = matcher.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return good_matches


class DentalImageWarping:
    """专门针对牙齿图像的投影变换"""

    def __init__(self):
        pass

    def estimate_homography(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
                           matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """估计单应性矩阵"""
        if len(matches) < 10:
            return None, np.array([], dtype=np.uint8)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 使用RANSAC估计单应性矩阵
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return homography, mask

    def estimate_affine(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
                       matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """估计仿射变换矩阵（作为备选）"""
        if len(matches) < 6:
            return None, np.array([], dtype=np.uint8)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 估计仿射变换
        affine, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

        if affine is not None:
            # 转换为3x3矩阵
            homography = np.vstack([affine, [0, 0, 1]])
            return homography, mask

        return None, np.array([], dtype=np.uint8)


class DentalImageBlender:
    """专门针对牙齿图像的融合器"""

    def __init__(self):
        self.feather_width = 30  # 羽化宽度
        self.exposure_weight = 0.7  # 曝光补偿权重

    def blend_images(self, images: List[np.ndarray],
                    transforms: List[np.ndarray],
                    masks: List[np.ndarray]) -> np.ndarray:
        """融合多张图像"""
        if not images:
            raise ValueError("图像列表为空")

        if len(images) == 1:
            return images[0]

        # 计算画布大小
        min_x, min_y, max_x, max_y = self._compute_canvas_bounds(images, transforms)

        canvas_width = max_x - min_x + 1
        canvas_height = max_y - min_y + 1

        # 创建画布
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
        weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)

        # 融合每张图像
        for i, (img, transform) in enumerate(zip(images, transforms)):
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

            # 计算权重
            warped_mask = cv2.warpPerspective(
                masks[i].astype(np.float32),
                full_transform,
                (canvas_width, canvas_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # 羽化处理
            feather_radius = self.feather_width
            kernel_size = feather_radius * 2 + 1
            feathered_mask = cv2.GaussianBlur(warped_mask, (kernel_size, kernel_size), feather_radius / 3)

            # 融合到画布
            canvas += warped * feathered_mask[..., np.newaxis]
            weight_map += feathered_mask

        # 归一化
        valid_mask = weight_map > 1e-6
        canvas[valid_mask] /= weight_map[valid_mask][..., np.newaxis]
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

        # 裁剪有效区域
        result = self._crop_valid_region(canvas, valid_mask)

        # 后处理
        result = self._post_process(result)

        return result

    def _compute_canvas_bounds(self, images: List[np.ndarray],
                               transforms: List[np.ndarray]) -> Tuple[int, int, int, int]:
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

    def _crop_valid_region(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """裁剪有效区域"""
        mask_uint8 = mask.astype(np.uint8) * 255

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        # 找到最大连通区域
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return image[y:y+h, x:x+w]

        return image

    def _post_process(self, image: np.ndarray) -> np.ndarray:
        """后处理图像"""
        # 轻微去噪
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # 锐化
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)

        # CLAHE增强对比度
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        return enhanced


class ImprovedOralStitcher:
    """改进的口腔图像拼接器"""

    def __init__(self):
        self.preprocessor = DentalImagePreprocessor()
        self.feature_matcher = DentalFeatureMatcher()
        self.warper = DentalImageWarping()
        self.blender = DentalImageBlender()
        self.logs: List[str] = []

    def stitch(self, records: List[ImageRecord]) -> StitchResult:
        """执行拼接"""
        self.logs = []

        if len(records) < 2:
            return StitchResult(
                success=False,
                anchor_index=None,
                panorama=None,
                logs=self.logs + ["至少需要两张图像"],
                method_name="Improved Dental Stitcher"
            )

        try:
            # 预处理所有图像
            self.logs.append("开始预处理图像...")
            processed_images = []
            for record in records:
                processed = self.preprocessor.preprocess(record.image)
                processed_images.append(processed)
                self.logs.append(f"预处理完成: {record.display_name}")

            # 选择基准图
            anchor_idx = self._select_anchor(records)
            self.logs.append(f"选择基准图: {records[anchor_idx].display_name}")

            # 检测特征
            self.logs.append("检测图像特征...")
            for processed in processed_images:
                kp, desc = self.feature_matcher.detect_and_compute(processed)
                processed.keypoints = kp
                processed.descriptors = desc
                self.logs.append(f"检测到 {len(kp)} 个特征点")

            # 计算变换矩阵
            self.logs.append("计算图像变换...")
            transforms = self._compute_transforms(processed_images, anchor_idx)

            # 融合图像
            self.logs.append("融合图像...")
            images_to_blend = [p.enhanced_image for p in processed_images]
            masks = [p.teeth_mask for p in processed_images]

            result = self.blender.blend_images(images_to_blend, transforms, masks)

            self.logs.append(f"拼接完成，结果尺寸: {result.shape}")

            return StitchResult(
                success=True,
                anchor_index=anchor_idx,
                panorama=result,
                logs=self.logs,
                method_name="Improved Dental Stitcher v2.0",
                included_indices=list(range(len(records))),
                ordered_indices=list(range(len(records)))
            )

        except Exception as e:
            self.logs.append(f"拼接失败: {str(e)}")
            logger.error(f"拼接错误: {e}", exc_info=True)
            return StitchResult(
                success=False,
                anchor_index=None,
                panorama=None,
                logs=self.logs,
                method_name="Improved Dental Stitcher"
            )

    def _select_anchor(self, records: List[ImageRecord]) -> int:
        """选择最佳基准图"""
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

    def _compute_transforms(self, processed_images: List[DentalImageData],
                           anchor_idx: int) -> List[np.ndarray]:
        """计算所有图像相对于基准图的变换"""
        transforms = []
        anchor_data = processed_images[anchor_idx]

        for i, img_data in enumerate(processed_images):
            if i == anchor_idx:
                transforms.append(np.eye(3))
                continue

            # 匹配特征
            matches = self.feature_matcher.match_features(
                anchor_data.descriptors,
                img_data.descriptors,
                method='sift' if anchor_data.descriptors.shape[1] == 128 else 'orb'
            )

            self.logs.append(f"图像 {i} 与基准图匹配到 {len(matches)} 对特征点")

            if len(matches) >= 10:
                # 估计单应性矩阵
                homography, mask = self.warper.estimate_homography(
                    anchor_data.keypoints,
                    img_data.keypoints,
                    matches
                )

                if homography is not None:
                    inliers = int(mask.sum())
                    self.logs.append(f"图像 {i} 单应性矩阵估计成功，内点数: {inliers}")
                    transforms.append(homography)
                else:
                    # 尝试仿射变换
                    affine, mask = self.warper.estimate_affine(
                        anchor_data.keypoints,
                        img_data.keypoints,
                        matches
                    )
                    if affine is not None:
                        self.logs.append(f"图像 {i} 使用仿射变换，内点数: {int(mask.sum())}")
                        transforms.append(affine)
                    else:
                        self.logs.append(f"图像 {i} 变换估计失败，使用单位矩阵")
                        transforms.append(np.eye(3))
            else:
                self.logs.append(f"图像 {i} 匹配点不足，使用单位矩阵")
                transforms.append(np.eye(3))

        return transforms