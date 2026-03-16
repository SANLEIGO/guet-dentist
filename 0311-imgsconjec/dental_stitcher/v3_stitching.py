from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dental_stitcher.enhanced_stitching import (
    DentalFeatureMatcher,
    DentalImagePreprocessor,
    blend_multi_band_viz,
    blend_no_blend,
    blend_simple,
    blend_with_borders,
)
from dental_stitcher.models import CandidateScore, ImageRecord, MatchResult, PrecheckItem, PrecheckReport, StitchResult
from dental_stitcher.utils import combine_quality_score, compute_image_metrics


@dataclass
class GlobalModel:
    matrix: np.ndarray
    model_name: str
    inlier_mask: np.ndarray
    inlier_ratio: float
    reproj_error: float


class CompatibleV3Stitcher:
    """第三版拼接器：多策略匹配 + 全局模型选择 + 局部形变修正。"""

    def __init__(self, viz_mode: str = "auto") -> None:
        self.preprocessor = DentalImagePreprocessor()
        self.feature_matcher = DentalFeatureMatcher()
        self.viz_mode = viz_mode
        self.logs: List[str] = []

    def precheck(self, records: List[ImageRecord]) -> PrecheckReport:
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
        logs: List[str] = []
        if len(records) < 2:
            return [], ["至少需要两张图像才能评估基准图。"]

        logs.append("开始评估基准图候选...")

        processed_images = []
        for record in records:
            enhanced, mask = self.preprocessor.preprocess(record.image)
            processed_images.append((enhanced, mask))

        all_keypoints = []
        all_descriptors = []

        for enhanced, mask in processed_images:
            kp, desc = self.feature_matcher.detect_and_compute(enhanced, mask)
            all_keypoints.append(kp)
            all_descriptors.append(desc)
            logs.append(f"检测到 {len(kp)} 个特征点")

        match_scores: Dict[Tuple[int, int], float] = {}
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if all_descriptors[i] is None or all_descriptors[j] is None:
                    continue

                matches = self.feature_matcher.match_features(
                    all_descriptors[i], all_descriptors[j],
                    method="orb" if all_descriptors[i].shape[1] == 32 else "sift",
                )

                score = len(matches) if matches else 0
                match_scores[(i, j)] = score
                match_scores[(j, i)] = score
                logs.append(f"图像 {i} 和 {j} 匹配得分: {score}")

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

            if partner_count > 0:
                connectivity_score = connectivity_score / partner_count

            total_score = connectivity_score * 0.7 + record.quality_score * 0.3

            candidates.append(CandidateScore(
                index=i,
                display_name=record.display_name,
                quality_score=record.quality_score,
                connectivity_score=connectivity_score,
                partner_count=partner_count,
                total_score=total_score,
                recommended=(i == 0),
            ))

        candidates.sort(key=lambda x: x.total_score, reverse=True)
        if candidates:
            candidates[0].recommended = True
            logs.append(f"推荐基准图: {candidates[0].display_name}")

        return candidates, logs

    def stitch(self, records: List[ImageRecord], anchor_index_override: Optional[int] = None) -> StitchResult:
        self.logs = []
        if len(records) < 2:
            return StitchResult(
                success=False,
                anchor_index=None,
                panorama=None,
                logs=self.logs + ["至少需要两张图像"],
                method_name="Dental Stitcher v3",
            )

        try:
            original_images = [record.image for record in records]
            anchor_idx = self._select_anchor(records, anchor_index_override)
            self.logs.append(f"选择基准图: {records[anchor_idx].display_name}")

            enhanced_images: List[np.ndarray] = []
            masks: List[np.ndarray] = []
            for img in original_images:
                enhanced, mask = self.preprocessor.preprocess(img)
                enhanced_images.append(enhanced)
                masks.append(mask)

            self.logs.append("检测图像特征...")
            all_keypoints = []
            all_descriptors = []
            for enhanced, mask in zip(enhanced_images, masks):
                kp, desc = self.feature_matcher.detect_and_compute(enhanced, mask)
                all_keypoints.append(kp)
                all_descriptors.append(desc)
                self.logs.append(f"检测到 {len(kp)} 个特征点")

            transforms: List[np.ndarray] = []
            local_fields: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
            pairwise_matches: Dict[Tuple[int, int], MatchResult] = {}
            included_indices = []

            anchor_kp = all_keypoints[anchor_idx]
            anchor_desc = all_descriptors[anchor_idx]

            for i, (kp, desc) in enumerate(zip(all_keypoints, all_descriptors)):
                if i == anchor_idx:
                    transforms.append(np.eye(3))
                    local_fields.append(None)
                    included_indices.append(i)
                    continue

                if desc is None or anchor_desc is None:
                    match_result = self._build_failed_match_result(anchor_idx, i, "FEATURE_TOO_FEW")
                    pairwise_matches[(min(anchor_idx, i), max(anchor_idx, i))] = match_result
                    transforms.append(np.eye(3))
                    local_fields.append(None)
                    continue

                matches = self.feature_matcher.match_features_robust(
                    anchor_desc,
                    desc,
                    anchor_kp,
                    kp,
                    method="orb" if desc.shape[1] == 32 else "sift",
                    img_shape=original_images[i].shape,
                )
                self.logs.append(f"图像 {i} 与基准图匹配到 {len(matches)} 对特征点")

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
                    matched_points1=None,
                )

                if len(matches) < 6:
                    match_result.details["reason"] = "FEATURE_TOO_FEW"
                    pairwise_matches[(min(anchor_idx, i), max(anchor_idx, i))] = match_result
                    transforms.append(np.eye(3))
                    local_fields.append(None)
                    continue

                src_pts, dst_pts = self._extract_points(anchor_kp, kp, matches)
                global_model = self._estimate_global_model(src_pts, dst_pts)

                if global_model is None:
                    match_result.details["reason"] = "MODEL_UNSTABLE"
                    pairwise_matches[(min(anchor_idx, i), max(anchor_idx, i))] = match_result
                    transforms.append(np.eye(3))
                    local_fields.append(None)
                    continue

                inlier_mask = global_model.inlier_mask.ravel().astype(bool)
                inliers = int(inlier_mask.sum())
                if inliers < 6 or global_model.inlier_ratio < 0.15:
                    match_result.details["reason"] = "LOW_INLIER_RATIO"
                    match_result.inliers = inliers
                    match_result.score = float(inliers)
                    pairwise_matches[(min(anchor_idx, i), max(anchor_idx, i))] = match_result
                    transforms.append(np.eye(3))
                    local_fields.append(None)
                    continue

                inlier_src = src_pts[inlier_mask]
                inlier_dst = dst_pts[inlier_mask]
                residuals = self._compute_residuals(global_model.matrix, inlier_src, inlier_dst)
                coverage = self._coverage_ratio(inlier_src, original_images[anchor_idx].shape)

                use_local = (
                    inliers >= 10
                    and global_model.reproj_error > 2.0
                    and coverage > 0.12
                )

                field: Optional[Tuple[np.ndarray, np.ndarray]] = None
                reason = ""
                if use_local:
                    field = self._build_residual_field(
                        inlier_src,
                        residuals,
                        original_images[anchor_idx].shape,
                        grid_step=40,
                    )
                    if field is None:
                        reason = "LOCAL_WARP_UNDERCONSTRAINED"
                        use_local = False

                match_result = self._build_success_match_result(
                    global_model,
                    inliers,
                    src_pts,
                    dst_pts,
                    inlier_mask,
                    method="v3_global_local",
                    reason=reason,
                )
                match_result.details["model"] = global_model.model_name
                match_result.details["inlier_ratio"] = round(float(global_model.inlier_ratio), 4)
                match_result.details["reproj_error"] = round(float(global_model.reproj_error), 3)
                match_result.details["coverage_ratio"] = round(float(coverage), 4)
                match_result.details["used_local_warp"] = bool(use_local)

                transforms.append(global_model.matrix)
                local_fields.append(field)
                pairwise_matches[(min(anchor_idx, i), max(anchor_idx, i))] = match_result
                included_indices.append(i)

            if len(included_indices) < 2:
                return StitchResult(
                    success=False,
                    anchor_index=anchor_idx,
                    panorama=None,
                    logs=self.logs + ["有效匹配图像不足，无法拼接"],
                    method_name="Dental Stitcher v3",
                    included_indices=included_indices,
                    ordered_indices=included_indices,
                    pairwise_matches=pairwise_matches,
                )

            self.logs.append("开始融合图像...")
            panorama, blend_mode = self._blend_with_local_warp(
                original_images,
                transforms,
                local_fields,
                anchor_idx,
            )
            if blend_mode != "局部形变融合":
                self.logs.append(f"融合模式切换为: {blend_mode}")

            return StitchResult(
                success=True,
                anchor_index=anchor_idx,
                panorama=panorama,
                logs=self.logs,
                method_name=self._method_name(blend_mode),
                included_indices=included_indices,
                ordered_indices=sorted(included_indices),
                pairwise_matches=pairwise_matches,
            )

        except Exception as e:
            self.logs.append(f"拼接失败: {str(e)}")
            import traceback

            self.logs.append(f"详细错误: {traceback.format_exc()}")
            return StitchResult(
                success=False,
                anchor_index=None,
                panorama=None,
                logs=self.logs,
                method_name="Dental Stitcher v3",
            )

    def _select_anchor(self, records: List[ImageRecord], override_idx: Optional[int] = None) -> int:
        if override_idx is not None and 0 <= override_idx < len(records):
            return override_idx
        if len(records) == 0:
            return 0

        best_idx = 0
        best_score = -1.0
        for i, record in enumerate(records):
            score = record.quality_score
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _extract_points(self, kp1: List, kp2: List, matches: List) -> Tuple[np.ndarray, np.ndarray]:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        return src_pts, dst_pts

    def _estimate_global_model(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[GlobalModel]:
        best_model: Optional[GlobalModel] = None

        homography_candidates: List[Tuple[np.ndarray, np.ndarray, str]] = []
        for method, thresh, name in [
            (cv2.RANSAC, 3.0, "homography_ransac"),
            (cv2.RHO, 5.0, "homography_rho"),
            (cv2.LMEDS, 0.0, "homography_lmeds"),
        ]:
            try:
                H, mask = cv2.findHomography(src_pts, dst_pts, method, thresh)
            except Exception:
                H, mask = None, None
            if H is not None and mask is not None:
                homography_candidates.append((H, mask, name))

        for H, mask, name in homography_candidates:
            inlier_mask = mask.astype(bool)
            inlier_ratio = float(inlier_mask.sum()) / max(len(mask), 1)
            reproj_error = self._mean_reprojection_error(H, src_pts, dst_pts, inlier_mask)
            model = GlobalModel(
                matrix=H,
                model_name=name,
                inlier_mask=mask,
                inlier_ratio=inlier_ratio,
                reproj_error=reproj_error,
            )
            best_model = self._select_better_model(best_model, model)

        A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if A is not None and mask is not None:
            H_affine = np.vstack([A, [0, 0, 1]])
            inlier_mask = mask.astype(bool)
            inlier_ratio = float(inlier_mask.sum()) / max(len(mask), 1)
            reproj_error = self._mean_reprojection_error(H_affine, src_pts, dst_pts, inlier_mask)
            model = GlobalModel(
                matrix=H_affine,
                model_name="affine_partial",
                inlier_mask=mask,
                inlier_ratio=inlier_ratio,
                reproj_error=reproj_error,
            )
            best_model = self._select_better_model(best_model, model)

        return best_model

    def _select_better_model(self, current: Optional[GlobalModel], candidate: GlobalModel) -> GlobalModel:
        if current is None:
            return candidate
        score_current = current.inlier_ratio * 100.0 - current.reproj_error
        score_candidate = candidate.inlier_ratio * 100.0 - candidate.reproj_error
        if score_candidate > score_current:
            return candidate
        return current

    def _mean_reprojection_error(
        self,
        H: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        inlier_mask: np.ndarray,
    ) -> float:
        inlier_mask = inlier_mask.ravel().astype(bool)
        if inlier_mask.sum() == 0:
            return float("inf")
        src = src_pts[inlier_mask]
        dst = dst_pts[inlier_mask]
        ones = np.ones((src.shape[0], 1), dtype=np.float32)
        src_h = np.hstack([src, ones])
        proj = (H @ src_h.T).T
        proj_xy = proj[:, :2] / proj[:, 2:3]
        errors = np.linalg.norm(proj_xy - dst, axis=1)
        return float(np.mean(errors))

    def _compute_residuals(self, H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        ones = np.ones((src_pts.shape[0], 1), dtype=np.float32)
        src_h = np.hstack([src_pts, ones])
        proj = (H @ src_h.T).T
        proj_xy = proj[:, :2] / proj[:, 2:3]
        return dst_pts - proj_xy

    def _coverage_ratio(self, src_pts: np.ndarray, shape: Tuple[int, int, int]) -> float:
        if src_pts.shape[0] < 2:
            return 0.0
        h, w = shape[:2]
        min_xy = src_pts.min(axis=0)
        max_xy = src_pts.max(axis=0)
        area = max((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]), 0.0)
        return float(area / max(h * w, 1))

    def _build_residual_field(
        self,
        src_pts: np.ndarray,
        residuals: np.ndarray,
        anchor_shape: Tuple[int, int, int],
        grid_step: int = 40,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if src_pts.shape[0] < 8:
            return None

        h, w = anchor_shape[:2]
        grid_x = np.arange(0, w, grid_step, dtype=np.float32)
        grid_y = np.arange(0, h, grid_step, dtype=np.float32)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_points = np.stack([gx.ravel(), gy.ravel()], axis=1)

        # 限制残差点数量
        if src_pts.shape[0] > 200:
            indices = np.linspace(0, src_pts.shape[0] - 1, 200, dtype=int)
            src_pts = src_pts[indices]
            residuals = residuals[indices]

        diff = grid_points[:, None, :] - src_pts[None, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        sigma2 = float((grid_step * 1.5) ** 2)
        weights = np.exp(-dist2 / (2 * sigma2)) + 1e-6
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights = weights / weights_sum

        delta = weights @ residuals
        delta_x = delta[:, 0].reshape(gx.shape)
        delta_y = delta[:, 1].reshape(gy.shape)

        residual_norm = np.linalg.norm(residuals, axis=1)
        max_delta = float(max(np.percentile(residual_norm, 90) * 2, 15.0))
        delta_x = np.clip(delta_x, -max_delta, max_delta)
        delta_y = np.clip(delta_y, -max_delta, max_delta)

        return delta_x.astype(np.float32), delta_y.astype(np.float32)

    def _blend_with_local_warp(
        self,
        images: List[np.ndarray],
        transforms: List[np.ndarray],
        local_fields: List[Optional[Tuple[np.ndarray, np.ndarray]]],
        anchor_idx: int,
    ) -> Tuple[np.ndarray, str]:
        min_x, min_y, max_x, max_y = self._compute_canvas_bounds(images, transforms)
        canvas_w = max_x - min_x + 1
        canvas_h = max_y - min_y + 1

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        xs = (np.arange(canvas_w, dtype=np.float32) + min_x)
        ys = (np.arange(canvas_h, dtype=np.float32) + min_y)
        grid_x, grid_y = np.meshgrid(xs, ys)
        ones = np.ones_like(grid_x)
        anchor_coords = np.stack([grid_x, grid_y, ones], axis=2)

        for idx, (img, H, field) in enumerate(zip(images, transforms, local_fields)):
            if H is None:
                continue

            proj = anchor_coords @ H.T
            proj_x = proj[..., 0] / np.clip(proj[..., 2], 1e-6, None)
            proj_y = proj[..., 1] / np.clip(proj[..., 2], 1e-6, None)

            if field is not None:
                delta_x, delta_y = field
                dense_dx = cv2.resize(delta_x, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
                dense_dy = cv2.resize(delta_y, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
                proj_x = proj_x + dense_dx
                proj_y = proj_y + dense_dy

            map_x = proj_x.astype(np.float32)
            map_y = proj_y.astype(np.float32)

            warped = cv2.remap(
                img,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            valid = (
                (map_x >= 0)
                & (map_x < img.shape[1])
                & (map_y >= 0)
                & (map_y < img.shape[0])
            ).astype(np.float32)

            if idx == anchor_idx:
                valid = np.clip(valid * 1.2, 0, 1)

            valid = cv2.GaussianBlur(valid, (0, 0), 5)
            canvas += warped.astype(np.float32) * valid[..., None]
            weight_map += valid

        weight_map = np.clip(weight_map, 1e-6, None)
        result = canvas / weight_map[..., None]
        result = np.clip(result, 0, 255).astype(np.uint8)

        blend_mode = "局部形变融合"
        result = self._crop_valid_region(result)
        if self.viz_mode == "边界高亮":
            result = blend_with_borders(images, [], transforms, show_borders=True)
            blend_mode = "边界高亮融合"
        elif self.viz_mode == "重叠区域":
            result = blend_multi_band_viz(images, [], transforms)
            blend_mode = "重叠区域可视化"
        elif self.viz_mode == "无缝融合":
            result = blend_simple(images, [], transforms)
            blend_mode = "无缝融合"
        elif self.viz_mode == "no_blend":
            result = blend_no_blend(images, [], transforms)
            blend_mode = "原始色彩"
        elif self.viz_mode == "auto":
            if len(images) <= 3:
                result = blend_simple(images, [], transforms)
                blend_mode = "无缝融合"
            elif len(images) <= 5:
                result = blend_with_borders(images, [], transforms, show_borders=True)
                blend_mode = "边界高亮融合"
            else:
                result = blend_multi_band_viz(images, [], transforms)
                blend_mode = "重叠区域可视化"

        return result, blend_mode

    def _method_name(self, blend_mode: str) -> str:
        return f"Dental Stitcher v3 ({blend_mode})"

    def _compute_canvas_bounds(
        self, images: List[np.ndarray], transforms: List[np.ndarray]
    ) -> Tuple[int, int, int, int]:
        all_corners = []
        for img, transform in zip(images, transforms):
            h, w = img.shape[:2]
            corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32)
            if transform is not None:
                transformed = (transform @ corners.T).T
                transformed = transformed[:, :2] / transformed[:, 2:3]
                all_corners.extend(transformed.tolist())
            else:
                all_corners.extend(corners[:, :2].tolist())

        all_corners = np.array(all_corners)
        min_x = int(np.floor(all_corners[:, 0].min()))
        min_y = int(np.floor(all_corners[:, 1].min()))
        max_x = int(np.ceil(all_corners[:, 0].max()))
        max_y = int(np.ceil(all_corners[:, 1].max()))
        return min_x, min_y, max_x, max_y

    def _crop_valid_region(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return image[y:y + h, x:x + w]
        return image

    def _build_success_match_result(
        self,
        global_model: GlobalModel,
        inliers: int,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        inlier_mask: np.ndarray,
        method: str,
        reason: str,
    ) -> MatchResult:
        matched_pts0 = src_pts[inlier_mask]
        matched_pts1 = dst_pts[inlier_mask]
        if matched_pts0.shape[0] > 50:
            indices = np.linspace(0, matched_pts0.shape[0] - 1, 50, dtype=int)
            matched_pts0 = matched_pts0[indices]
            matched_pts1 = matched_pts1[indices]

        return MatchResult(
            success=True,
            score=float(inliers),
            inliers=inliers,
            homography=global_model.matrix,
            inverse_homography=np.linalg.inv(global_model.matrix),
            details={"method": method, "reason": reason},
            sequence_distance=0,
            weighted_score=float(inliers * 0.8 + len(src_pts) * 0.2),
            matched_points0=matched_pts0.astype(np.float32),
            matched_points1=matched_pts1.astype(np.float32),
        )

    def _build_failed_match_result(self, anchor_idx: int, img_idx: int, reason: str) -> MatchResult:
        return MatchResult(
            success=False,
            score=0.0,
            inliers=0,
            homography=None,
            inverse_homography=None,
            details={"reason": reason},
            sequence_distance=abs(img_idx - anchor_idx),
            weighted_score=0.0,
            matched_points0=None,
            matched_points1=None,
        )
