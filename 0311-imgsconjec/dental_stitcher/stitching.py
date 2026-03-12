from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dental_stitcher.models import CandidateScore, ImageRecord, MatchResult, PrecheckItem, PrecheckReport, StitchResult
from dental_stitcher.utils import combine_quality_score, compute_image_metrics

try:
    import torch
    import kornia as K
    from kornia.feature import LoFTR

    HAS_LOFTR = True
except Exception:
    torch = None
    K = None
    LoFTR = None
    HAS_LOFTR = False


@dataclass
class GraphEdge:
    source: int
    target: int
    score: float
    sequence_distance: int


class FeatureMatcher:
    def __init__(self) -> None:
        self.method_name = "AKAZE + RANSAC"
        self.device = None
        self.loftr = None
        self.akaze = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        if HAS_LOFTR:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.loftr = LoFTR(pretrained="outdoor").to(self.device).eval()
            self.method_name = f"LoFTR + RANSAC ({self.device})"

    def match(self, src: np.ndarray, dst: np.ndarray, relaxed: bool = False) -> MatchResult:
        if self.loftr is not None:
            result = self._match_with_loftr(src, dst, relaxed=relaxed)
            if result.success:
                return result
        return self._match_with_akaze(src, dst, relaxed=relaxed)

    def _match_with_loftr(self, src: np.ndarray, dst: np.ndarray, relaxed: bool = False) -> MatchResult:
        try:
            src_gray = self._prepare_gray(src)
            dst_gray = self._prepare_gray(dst)

            src_tensor = K.image_to_tensor(src_gray, False).float()[None] / 255.0
            dst_tensor = K.image_to_tensor(dst_gray, False).float()[None] / 255.0
            src_tensor = src_tensor.to(self.device)
            dst_tensor = dst_tensor.to(self.device)

            with torch.inference_mode():
                correspondences = self.loftr({"image0": src_tensor, "image1": dst_tensor})

            mkpts0 = correspondences["keypoints0"].detach().cpu().numpy()
            mkpts1 = correspondences["keypoints1"].detach().cpu().numpy()
            confidence = correspondences["confidence"].detach().cpu().numpy()

            min_matches = 12 if relaxed else 16
            min_inliers = 8 if relaxed else 12
            ransac_thresh = 4.5 if relaxed else 3.0

            if len(mkpts0) < min_matches:
                return MatchResult(False, 0.0, 0, None, None, {"reason": "too_few_matches"})

            homography, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransac_thresh)
            if homography is None or mask is None:
                return MatchResult(False, 0.0, 0, None, None, {"reason": "homography_failed"})

            inliers = int(mask.sum())
            if inliers < min_inliers:
                return MatchResult(False, 0.0, inliers, None, None, {"reason": "too_few_inliers"})
            if not self._has_sufficient_spatial_support(mkpts0, mkpts1, mask.ravel().astype(bool), src.shape, dst.shape, relaxed):
                return MatchResult(False, 0.0, inliers, None, None, {"reason": "poor_spatial_support"})

            inverse_h = np.linalg.inv(homography)
            conf_score = float(confidence[mask.ravel().astype(bool)].mean()) if inliers else 0.0
            score = inliers * (0.5 + conf_score)
            pts0, pts1 = self._sample_points(mkpts0, mkpts1, mask.ravel().astype(bool))
            return MatchResult(
                True,
                score,
                inliers,
                homography,
                inverse_h,
                {"avg_confidence": conf_score},
                matched_points0=pts0,
                matched_points1=pts1,
            )
        except Exception as exc:
            return MatchResult(False, 0.0, 0, None, None, {"reason": f"loftr_error:{exc}"})

    def _match_with_akaze(self, src: np.ndarray, dst: np.ndarray, relaxed: bool = False) -> MatchResult:
        src_gray = self._prepare_gray(src)
        dst_gray = self._prepare_gray(dst)
        src_mask = self._feature_focus_mask(src)
        dst_mask = self._feature_focus_mask(dst)

        kps0, des0 = self.akaze.detectAndCompute(src_gray, src_mask)
        kps1, des1 = self.akaze.detectAndCompute(dst_gray, dst_mask)
        min_keypoints = 8 if relaxed else 10
        if des0 is None or des1 is None or len(kps0) < min_keypoints or len(kps1) < min_keypoints:
            return MatchResult(False, 0.0, 0, None, None, {"reason": "descriptors_missing"})

        raw_matches = self.bf.knnMatch(des0, des1, k=2)
        good = [m for m, n in raw_matches if m.distance < 0.8 * n.distance]
        min_matches = 8 if relaxed else 10
        if len(good) < min_matches:
            return MatchResult(False, 0.0, 0, None, None, {"reason": "too_few_matches"})

        pts0 = np.float32([kps0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts1 = np.float32([kps1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0 if relaxed else 4.0)
        if (homography is None or mask is None) and relaxed:
            affine, affine_mask = cv2.estimateAffinePartial2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if affine is not None and affine_mask is not None:
                homography = np.vstack([affine, [0.0, 0.0, 1.0]])
                mask = affine_mask
        if homography is None or mask is None:
            return MatchResult(False, 0.0, 0, None, None, {"reason": "homography_failed"})

        inliers = int(mask.sum())
        min_inliers = 6 if relaxed else 8
        if inliers < min_inliers:
            return MatchResult(False, 0.0, inliers, None, None, {"reason": "too_few_inliers"})
        if not self._has_sufficient_spatial_support(pts0.reshape(-1, 2), pts1.reshape(-1, 2), mask.ravel().astype(bool), src.shape, dst.shape, relaxed):
            return MatchResult(False, 0.0, inliers, None, None, {"reason": "poor_spatial_support"})

        inverse_h = np.linalg.inv(homography)
        score = float(inliers)
        pts0_sample, pts1_sample = self._sample_points(pts0.reshape(-1, 2), pts1.reshape(-1, 2), mask.ravel().astype(bool))
        return MatchResult(
            True,
            score,
            inliers,
            homography,
            inverse_h,
            {"matches": len(good)},
            matched_points0=pts0_sample,
            matched_points1=pts1_sample,
        )

    def _sample_points(
        self,
        points0: np.ndarray,
        points1: np.ndarray,
        inlier_mask: np.ndarray,
        limit: int = 80,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        valid0 = points0[inlier_mask]
        valid1 = points1[inlier_mask]
        if len(valid0) == 0 or len(valid1) == 0:
            return None, None
        if len(valid0) > limit:
            sample_idx = np.linspace(0, len(valid0) - 1, limit, dtype=int)
            valid0 = valid0[sample_idx]
            valid1 = valid1[sample_idx]
        return valid0.astype(np.float32), valid1.astype(np.float32)

    def _prepare_gray(self, image: np.ndarray) -> np.ndarray:
        # Suppress highlights and de-emphasize soft tissue/background so teeth dominate the matcher.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        highlight_mask = (value > 245).astype(np.uint8) * 255
        softened = image.copy()
        if int(highlight_mask.sum()) > 0:
            softened = cv2.inpaint(image, highlight_mask, 3, cv2.INPAINT_TELEA)
        gray = cv2.cvtColor(softened, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        weight = self._feature_focus_weight(softened)
        weighted = np.clip(gray.astype(np.float32) * weight, 0, 255).astype(np.uint8)
        return weighted

    def _feature_focus_mask(self, image: np.ndarray) -> np.ndarray:
        weight = self._feature_focus_weight(image)
        return (weight > 0.28).astype(np.uint8) * 255

    def _feature_focus_weight(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        oral_mask = (gray > 8).astype(np.float32)
        soft_tissue = (
            (((hsv[:, :, 0] <= 18) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 45))
        ).astype(np.float32)
        tooth_like = (
            (lab[:, :, 0] > 125) &
            (hsv[:, :, 1] < 135) &
            (hsv[:, :, 2] > 70)
        ).astype(np.float32)

        edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

        weight = oral_mask * (0.18 + 0.75 * tooth_like + 0.28 * edges)
        weight *= (1.0 - 0.55 * soft_tissue)
        weight = cv2.GaussianBlur(weight, (0, 0), 3.0)
        return np.clip(weight, 0.0, 1.0)

    def _has_sufficient_spatial_support(
        self,
        points0: np.ndarray,
        points1: np.ndarray,
        inlier_mask: np.ndarray,
        src_shape: tuple[int, ...],
        dst_shape: tuple[int, ...],
        relaxed: bool,
    ) -> bool:
        inlier_points0 = points0[inlier_mask]
        inlier_points1 = points1[inlier_mask]
        if len(inlier_points0) < (6 if relaxed else 8):
            return False

        src_h, src_w = src_shape[:2]
        dst_h, dst_w = dst_shape[:2]
        span0 = inlier_points0.max(axis=0) - inlier_points0.min(axis=0)
        span1 = inlier_points1.max(axis=0) - inlier_points1.min(axis=0)

        min_w_ratio = 0.10 if relaxed else 0.14
        min_h_ratio = 0.08 if relaxed else 0.10
        spread0_ok = span0[0] >= src_w * min_w_ratio or span0[1] >= src_h * min_h_ratio
        spread1_ok = span1[0] >= dst_w * min_w_ratio or span1[1] >= dst_h * min_h_ratio
        if not (spread0_ok and spread1_ok):
            return False

        src_center = np.array([src_w * 0.5, src_h * 0.5], dtype=np.float32)
        dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
        center_dev0 = np.linalg.norm(inlier_points0.mean(axis=0) - src_center) / max(src_w, src_h)
        center_dev1 = np.linalg.norm(inlier_points1.mean(axis=0) - dst_center) / max(dst_w, dst_h)
        return center_dev0 < 0.75 and center_dev1 < 0.75


class OralStitcher:
    def __init__(self) -> None:
        self.matcher = FeatureMatcher()

    def precheck(self, records: list[ImageRecord]) -> PrecheckReport:
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
            if self._looks_like_large_span(records, idx):
                reasons.append("疑似跨段跨度过大")
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

    def _looks_like_large_span(self, records: list[ImageRecord], idx: int) -> bool:
        if records and records[0].segment == "full":
            return False
        if len(records) < 5:
            return False
        if idx in (0, len(records) - 1):
            return False
        current = records[idx]
        height, width = current.image.shape[:2]
        aspect = width / max(height, 1)
        return aspect > 1.8

    def score_candidates(self, records: list[ImageRecord]) -> tuple[list[CandidateScore], list[str]]:
        logs: list[str] = []
        if len(records) < 2:
            return [], ["至少需要两张图像才能评估基准图。"]

        working_records = self._prepare_records_for_matching(records, logs)
        pairwise = self._build_pairwise_matches(working_records, logs)
        candidates = self._rank_candidates(working_records, pairwise, logs)
        return candidates, logs

    def stitch(self, records: list[ImageRecord], anchor_index_override: int | None = None) -> StitchResult:
        logs: list[str] = []
        if len(records) < 2:
            return StitchResult(False, None, None, ["至少需要两张图像进行拼接。"], self.matcher.method_name)

        working_records = self._prepare_records_for_matching(records, logs)
        pairwise = self._build_pairwise_matches(working_records, logs)
        candidates = self._rank_candidates(working_records, pairwise, logs)

        if working_records and working_records[0].segment == "full":
            panorama, included_indices, ordered_indices = self._build_full_arch_strip(working_records, pairwise, logs)
            anchor_index = len(working_records) // 2
            return StitchResult(
                panorama is not None,
                anchor_index,
                panorama,
                logs,
                self.matcher.method_name + " + Strip",
                included_indices=included_indices,
                ordered_indices=ordered_indices,
                pairwise_matches=pairwise,
            )

        if anchor_index_override is not None:
            anchor_index = anchor_index_override
            if 0 <= anchor_index < len(working_records):
                logs.append(f"使用手动指定基准图: {working_records[anchor_index].display_name}")
            else:
                logs.append("手动指定基准图索引无效，回退为自动选择。")
                anchor_index = candidates[0].index if candidates else None
        else:
            anchor_index = self._select_anchor(working_records, pairwise, candidates, logs)
        if anchor_index is None:
            return StitchResult(False, None, None, logs + ["未能找到有效的拼接基准图。"], self.matcher.method_name)

        transforms = self._build_global_transforms(working_records, pairwise, anchor_index, logs)
        if len(transforms) < 2:
            return StitchResult(False, anchor_index, None, logs + ["有效连通图不足，无法完成拼接。"], self.matcher.method_name)

        panorama = self._warp_and_blend(working_records, transforms, logs)
        ordered_indices = self.ordered_sequence(pairwise, anchor_index, len(working_records))
        return StitchResult(
            panorama is not None,
            anchor_index,
            panorama,
            logs,
            self.matcher.method_name,
            included_indices=sorted(transforms.keys()),
            ordered_indices=[idx for idx in ordered_indices if idx in transforms],
            pairwise_matches=pairwise,
        )

    def _build_full_arch_strip(
        self,
        records: list[ImageRecord],
        pairwise: dict[tuple[int, int], MatchResult],
        logs: list[str],
    ) -> tuple[np.ndarray | None, list[int], list[int]]:
        if len(records) < 2:
            return None, [], []

        logs.append("完整牙弓模式: 切换为序列条带展开图输出。")
        ordered_indices = list(range(len(records)))

        strips: list[np.ndarray] = []
        centers_y: list[float] = []
        for idx in ordered_indices:
            strip, center_y = self._extract_dental_strip(records[idx].image)
            strips.append(strip)
            centers_y.append(center_y)

        heights = [strip.shape[0] for strip in strips]
        widths = [strip.shape[1] for strip in strips]
        target_height = max(heights)
        positions_x = [0]
        included_indices = [0]

        for idx in range(1, len(records)):
            key = (idx - 1, idx)
            result = pairwise.get(key)
            if result is None or not result.success:
                logs.append(f"条带跳过: {records[idx].display_name}, 原因=与前一张未建立稳定相邻匹配")
                continue
            dx = self._estimate_strip_dx(result, strips[idx - 1], strips[idx])
            if dx is None:
                logs.append(f"条带跳过: {records[idx].display_name}, 原因=横向位移估计失败")
                continue
            dx = max(int(widths[idx] * 0.18), min(dx, int(widths[idx - 1] * 0.92)))
            positions_x.append(positions_x[-1] + dx)
            included_indices.append(idx)
            logs.append(f"条带接入: {records[idx].display_name}, 与前一张横向步长={dx}")

        if len(included_indices) < 2:
            return None, included_indices, included_indices

        active_strips = [strips[idx] for idx in included_indices]
        active_positions = [positions_x[included_indices.index(idx)] for idx in included_indices]
        total_width = max(pos + strip.shape[1] for pos, strip in zip(active_positions, active_strips))
        canvas = np.zeros((target_height, total_width, 3), dtype=np.float32)
        weight_sum = np.zeros((target_height, total_width, 1), dtype=np.float32)

        baseline_y = int(np.median([centers_y[idx] for idx in included_indices]))
        for pos_x, idx, strip in zip(active_positions, included_indices, active_strips):
            strip_h, strip_w = strip.shape[:2]
            offset_y = max(0, min(target_height - strip_h, baseline_y - int(centers_y[idx])))
            y0 = offset_y
            y1 = offset_y + strip_h
            x0 = pos_x
            x1 = pos_x + strip_w
            weight = self._feather_weight(strip_h, strip_w)[..., None]
            canvas[y0:y1, x0:x1] += strip.astype(np.float32) * weight
            weight_sum[y0:y1, x0:x1] += weight

        panorama = canvas / np.clip(weight_sum, 1e-6, None)
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)
        valid_mask = (weight_sum[..., 0] > 0.01).astype(np.uint8) * 255
        valid_mask = self._largest_component(valid_mask)
        x, y, w, h = cv2.boundingRect(valid_mask)
        logs.append(f"条带展开图输出尺寸: {w}x{h}")
        return panorama[y : y + h, x : x + w], included_indices, included_indices

    def _extract_dental_strip(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        tooth_y_ratio = self._tooth_centroid_ratio(image)
        h, w = image.shape[:2]
        center_y = int(np.clip(round(tooth_y_ratio * h), 0, h - 1))
        band_half = max(70, int(h * 0.22))
        y0 = max(0, center_y - band_half)
        y1 = min(h, center_y + band_half)
        strip = image[y0:y1, :]
        return strip, float(center_y - y0)

    def _estimate_strip_dx(
        self,
        result: MatchResult,
        left_strip: np.ndarray,
        right_strip: np.ndarray,
    ) -> int | None:
        pts0 = result.matched_points0
        pts1 = result.matched_points1
        if pts0 is None or pts1 is None or len(pts0) < 4:
            return None
        delta_x = pts1[:, 0] - pts0[:, 0]
        median_dx = float(np.median(delta_x))
        if not np.isfinite(median_dx):
            return None
        step = int(round(max(20.0, left_strip.shape[1] + median_dx)))
        if step <= 0:
            step = int(left_strip.shape[1] * 0.35)
        return step

    def _prepare_records_for_matching(self, records: list[ImageRecord], logs: list[str]) -> list[ImageRecord]:
        if not records or records[0].segment != "full":
            return records
        logs.append("完整牙弓模式: 启用整组方向一致性校正与刚体链式拼接。")
        raw_angles = [self._estimate_arch_angle(record.image) for record in records]
        rotation_degrees = self._resolve_consistent_rotations(records, raw_angles)
        prepared: list[ImageRecord] = []
        for record, rotate_deg in zip(records, rotation_degrees):
            normalized = self._normalize_full_arch_image(record.image, rotate_deg)
            new_record = ImageRecord(path=record.path, arch=record.arch, segment=record.segment, image=normalized)
            new_record.quality_score = record.quality_score
            new_record.sharpness_score = record.sharpness_score
            new_record.exposure_score = record.exposure_score
            prepared.append(new_record)
            logs.append(f"标准化图像: {record.display_name}, 主方向旋正={rotate_deg:.1f}度")
        return prepared

    def _normalize_full_arch_image(self, image: np.ndarray, rotate_deg: float) -> np.ndarray:
        return self._rotate_image(image, rotate_deg)

    def _estimate_arch_angle(self, image: np.ndarray) -> float:
        mask = self.matcher._feature_focus_mask(image)
        points = np.column_stack(np.nonzero(mask > 0))
        if len(points) < 50:
            return 0.0
        points_xy = points[:, ::-1].astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(points_xy, mean=None, maxComponents=2)
        _ = mean
        direction = eigenvectors[0]
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
        return float(angle)

    def _resolve_consistent_rotations(self, records: list[ImageRecord], raw_angles: list[float]) -> list[float]:
        if not records:
            return []
        rotations = [0.0] * len(records)
        center_idx = len(records) // 2
        rotations[center_idx] = self._choose_rotation(records[center_idx], raw_angles[center_idx], None)

        for idx in range(center_idx - 1, -1, -1):
            rotations[idx] = self._choose_rotation(records[idx], raw_angles[idx], rotations[idx + 1])

        for idx in range(center_idx + 1, len(records)):
            rotations[idx] = self._choose_rotation(records[idx], raw_angles[idx], rotations[idx - 1])

        return rotations

    def _choose_rotation(self, record: ImageRecord, raw_angle: float, reference_rotation: float | None) -> float:
        candidates = [
            self._normalize_angle(-raw_angle),
            self._normalize_angle(-raw_angle + 180.0),
            self._normalize_angle(-raw_angle - 180.0),
        ]
        best_rotation = candidates[0]
        best_score = float("inf")
        for candidate in candidates:
            rotated = self._rotate_image(record.image, candidate)
            tooth_y_ratio = self._tooth_centroid_ratio(rotated)
            if record.arch == "lower":
                arch_penalty = abs(tooth_y_ratio - 0.72)
            else:
                arch_penalty = abs(tooth_y_ratio - 0.28)
            continuity_penalty = 0.0
            if reference_rotation is not None:
                continuity_penalty = self._rotation_distance(candidate, reference_rotation) / 180.0
            score = arch_penalty * 3.0 + continuity_penalty
            if score < best_score:
                best_score = score
                best_rotation = candidate
        return self._normalize_angle(best_rotation)

    def _rotation_distance(self, angle_a: float, angle_b: float) -> float:
        diff = self._normalize_angle(angle_a - angle_b)
        return abs(diff)

    def _normalize_angle(self, angle: float) -> float:
        normalized = (angle + 180.0) % 360.0 - 180.0
        if normalized == -180.0:
            return 180.0
        return normalized

    def _tooth_centroid_ratio(self, image: np.ndarray) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tooth_like = (
            (lab[:, :, 0] > 125) &
            (hsv[:, :, 1] < 135) &
            (hsv[:, :, 2] > 70) &
            (gray > 40)
        ).astype(np.uint8)
        points = np.column_stack(np.nonzero(tooth_like > 0))
        if len(points) == 0:
            return 0.5
        return float(points[:, 0].mean() / max(image.shape[0] - 1, 1))

    def _rotate_image(self, image: np.ndarray, rotate_deg: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w * 0.5, h * 0.5)
        matrix = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _cylindrical_warp(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w < 10 or h < 10:
            return image
        focal = max(w, h) * 1.1
        cx = w / 2.0
        cy = h / 2.0
        y_i, x_i = np.indices((h, w), dtype=np.float32)
        theta = (x_i - cx) / focal
        src_x = focal * np.tan(theta) + cx
        src_y = (y_i - cy) / np.cos(theta) + cy
        warped = cv2.remap(
            image,
            src_x.astype(np.float32),
            src_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = cv2.remap(
            np.ones((h, w), dtype=np.uint8) * 255,
            src_x.astype(np.float32),
            src_y.astype(np.float32),
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        if valid.max() == 0:
            return warped
        x, y, ww, hh = cv2.boundingRect(valid)
        return warped[y : y + hh, x : x + ww]

    def _curve_unwrap(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w < 32 or h < 32:
            return image

        curve = self._estimate_arch_curve(image)
        if curve is None:
            return image

        baseline = np.linspace(float(curve[0]), float(curve[-1]), w, dtype=np.float32)
        y_i, x_i = np.indices((h, w), dtype=np.float32)
        residual = (curve - baseline).astype(np.float32)
        residual = cv2.GaussianBlur(residual[None, :], (0, 0), sigmaX=max(5.0, w / 120.0)).reshape(-1)
        shift = residual[np.clip(x_i.astype(np.int32), 0, w - 1)]
        src_y = y_i + shift
        src_x = x_i

        unwrapped = cv2.remap(
            image,
            src_x,
            src_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = cv2.remap(
            np.ones((h, w), dtype=np.uint8) * 255,
            src_x,
            src_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        if valid.max() == 0:
            return unwrapped
        x, y, ww, hh = cv2.boundingRect(valid)
        return unwrapped[y : y + hh, x : x + ww]

    def _estimate_arch_curve(self, image: np.ndarray) -> np.ndarray | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        tooth_like = (
            (lab[:, :, 0] > 125) &
            (hsv[:, :, 1] < 140) &
            (hsv[:, :, 2] > 65) &
            (gray > 40)
        ).astype(np.uint8)
        tooth_like = cv2.morphologyEx(tooth_like, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        tooth_like = cv2.morphologyEx(tooth_like, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))

        h, w = tooth_like.shape
        samples_x = []
        samples_y = []
        step = max(4, w // 160)
        for x in range(0, w, step):
            ys = np.nonzero(tooth_like[:, x])[0]
            if len(ys) < 3:
                continue
            y_value = float(np.median(ys))
            samples_x.append(float(x))
            samples_y.append(y_value)

        expected_samples = max(1, len(range(0, w, step)))
        coverage = len(samples_x) / expected_samples
        if len(samples_x) < 8 or coverage < 0.28:
            return None

        degree = 2 if len(samples_x) < 20 else 3
        samples_x_np = np.array(samples_x, dtype=np.float32)
        samples_y_np = np.array(samples_y, dtype=np.float32)
        coeffs = np.polyfit(samples_x_np, samples_y_np, deg=degree)
        curve_x = np.arange(w, dtype=np.float32)
        curve_y = np.polyval(coeffs, curve_x).astype(np.float32)
        curve_y = cv2.GaussianBlur(curve_y[None, :], (0, 0), sigmaX=max(6.0, w / 80.0)).reshape(-1)
        curve_y = np.clip(curve_y, 0, h - 1)

        baseline_coeffs = np.polyfit(samples_x_np, samples_y_np, deg=1)
        baseline = np.polyval(baseline_coeffs, curve_x).astype(np.float32)
        residual = curve_y - baseline
        sample_fit = np.polyval(coeffs, samples_x_np).astype(np.float32)
        rmse = float(np.sqrt(np.mean((sample_fit - samples_y_np) ** 2)))
        amplitude = float(np.percentile(residual, 95) - np.percentile(residual, 5))

        if rmse > h * 0.10:
            return None
        if amplitude > h * 0.18:
            return None
        return curve_y

    def _build_pairwise_matches(self, records: list[ImageRecord], logs: list[str]) -> dict[tuple[int, int], MatchResult]:
        pairwise: dict[tuple[int, int], MatchResult] = {}
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                relaxed = records[i].segment == "full" and abs(i - j) <= 1
                result = self.matcher.match(records[i].image, records[j].image, relaxed=relaxed)
                seq_distance = abs(i - j)
                weighted_score = self._sequence_weight(seq_distance) * result.score if result.success else 0.0
                result.sequence_distance = seq_distance
                result.weighted_score = weighted_score
                pairwise[(i, j)] = result
                if result.success:
                    logs.append(
                        f"匹配成功: {records[i].display_name} <-> {records[j].display_name}, "
                        f"内点={result.inliers}, 原始得分={result.score:.1f}, "
                        f"顺序距离={seq_distance}, 加权得分={weighted_score:.1f}"
                    )
                else:
                    logs.append(
                        f"匹配失败: {records[i].display_name} <-> {records[j].display_name}, "
                        f"原因={result.details.get('reason', 'unknown')}"
                    )
        return pairwise

    def _select_anchor(
        self,
        records: list[ImageRecord],
        pairwise: dict[tuple[int, int], MatchResult],
        candidates: list[CandidateScore],
        logs: list[str],
    ) -> int | None:
        if not candidates:
            return None
        if records and records[0].segment == "full":
            center_index = (len(records) - 1) / 2.0
            best_idx = None
            best_score = -1e9
            for candidate in candidates:
                idx = candidate.index
                neighbor_support = 0.0
                for neighbor in (idx - 1, idx + 1):
                    if neighbor < 0 or neighbor >= len(records):
                        continue
                    key = (min(idx, neighbor), max(idx, neighbor))
                    result = pairwise.get(key)
                    if result and result.success:
                        neighbor_support += result.weighted_score
                center_penalty = abs(idx - center_index) * 40.0
                full_score = candidate.total_score + neighbor_support * 0.8 - center_penalty
                logs.append(
                    f"完整牙弓基准图重评分: {records[idx].display_name}, 邻接支持={neighbor_support:.1f}, "
                    f"中心惩罚={center_penalty:.1f}, 重评分={full_score:.2f}"
                )
                if full_score > best_score:
                    best_score = full_score
                    best_idx = idx
            if best_idx is not None:
                logs.append(f"完整牙弓模式最终基准图: {records[best_idx].display_name}")
                return best_idx
        return candidates[0].index

    def _rank_candidates(
        self,
        records: list[ImageRecord],
        pairwise: dict[tuple[int, int], MatchResult],
        logs: list[str],
    ) -> list[CandidateScore]:
        candidates: list[CandidateScore] = []
        center_index = (len(records) - 1) / 2.0
        for idx, record in enumerate(records):
            if record.quality_score == 0.0:
                sharpness, exposure = compute_image_metrics(record.image)
                record.sharpness_score = sharpness
                record.exposure_score = exposure
                record.quality_score = combine_quality_score(sharpness, exposure)
            connectivity = 0.0
            partners = 0
            for (i, j), result in pairwise.items():
                if not result.success:
                    continue
                if idx in (i, j):
                    connectivity += result.weighted_score
                    partners += 1
            center_bonus = 0.0
            if record.segment == "full":
                center_bonus = max(0.0, 18.0 - abs(idx - center_index) * 8.0)
            total_score = connectivity + record.quality_score * 15.0 + partners * 8.0 + center_bonus
            logs.append(
                f"候选基准图: {record.display_name}, 图像质量={record.quality_score:.2f}, "
                f"连通伙伴={partners}, 总分={total_score:.2f}"
            )
            candidates.append(
                CandidateScore(
                    index=idx,
                    display_name=record.display_name,
                    quality_score=record.quality_score,
                    connectivity_score=connectivity,
                    partner_count=partners,
                    total_score=total_score,
                )
            )

        candidates.sort(key=lambda item: item.total_score, reverse=True)
        if candidates:
            candidates[0].recommended = True
            logs.append(f"自动选定基准图: {candidates[0].display_name}")
        return candidates

    def _build_global_transforms(
        self,
        records: list[ImageRecord],
        pairwise: dict[tuple[int, int], MatchResult],
        anchor_index: int,
        logs: list[str],
    ) -> dict[int, np.ndarray]:
        if records and records[0].segment == "full":
            return self._build_full_arch_transforms(records, pairwise, anchor_index, logs)

        adjacency: dict[int, list[GraphEdge]] = {idx: [] for idx in range(len(records))}
        pair_lookup: dict[tuple[int, int], np.ndarray] = {}
        for (i, j), result in pairwise.items():
            if not result.success:
                continue
            forward = self._resolve_pair_transform(result, i, j, records)
            inverse = self._resolve_pair_transform(result, j, i, records)
            if forward is None or inverse is None:
                continue
            adjacency[i].append(GraphEdge(i, j, result.weighted_score, result.sequence_distance))
            adjacency[j].append(GraphEdge(j, i, result.weighted_score, result.sequence_distance))
            pair_lookup[(i, j)] = forward
            pair_lookup[(j, i)] = inverse

        transforms: dict[int, np.ndarray] = {anchor_index: np.eye(3, dtype=np.float64)}
        visited = {anchor_index}
        frontier: list[tuple[float, int, int, int]] = []
        if len(records) == 2:
            logs.append("检测到两图拼接，启用仿射优先与畸变过滤。")
        for edge in adjacency[anchor_index]:
            frontier.append((edge.score, -self._neighbor_priority(edge.sequence_distance), anchor_index, edge.target))

        while frontier:
            frontier.sort(key=lambda item: (item[0], item[1]), reverse=True)
            score, _, parent, node = frontier.pop(0)
            if node in visited:
                continue
            if (node, parent) not in pair_lookup:
                continue
            transforms[node] = transforms[parent] @ pair_lookup[(node, parent)]
            visited.add(node)
            logs.append(f"加入拼接图: {records[node].display_name}, 连接得分={score:.1f}")
            for edge in adjacency[node]:
                if edge.target not in visited:
                    frontier.append((edge.score, -self._neighbor_priority(edge.sequence_distance), node, edge.target))
        return transforms

    def _resolve_pair_transform(
        self,
        result: MatchResult,
        node: int,
        parent: int,
        records: list[ImageRecord],
    ) -> np.ndarray | None:
        transform = self._derive_chain_transform(result, node, parent)
        if transform is None:
            return None
        transform = self._refine_chain_transform(records[node].image, records[parent].image, transform)
        if transform is None:
            return None
        if not self._is_reasonable_chain_transform(transform, records[node].image.shape, records[parent].image.shape):
            return None
        return transform

    def _build_full_arch_transforms(
        self,
        records: list[ImageRecord],
        pairwise: dict[tuple[int, int], MatchResult],
        anchor_index: int,
        logs: list[str],
    ) -> dict[int, np.ndarray]:
        transforms: dict[int, np.ndarray] = {anchor_index: np.eye(3, dtype=np.float64)}
        logs.append("检测到完整牙弓模式，启用严格相邻的刚体链式拼接策略。")

        for idx in range(anchor_index - 1, -1, -1):
            parent, homography, score = self._best_chain_parent(
                idx,
                [idx + 1] if idx + 1 < len(records) else [],
                transforms,
                pairwise,
                records,
            )
            if homography is None or parent is None:
                logs.append(f"跳过图像: {records[idx].display_name}, 原因=左侧链路未建立")
                continue
            transforms[idx] = transforms[parent] @ homography
            logs.append(
                f"加入左侧链: {records[idx].display_name}, 父节点={records[parent].display_name}, "
                f"连接得分={score:.1f}, 几何模型=affine_preferred"
            )

        for idx in range(anchor_index + 1, len(records)):
            parent, homography, score = self._best_chain_parent(
                idx,
                [idx - 1] if idx - 1 >= 0 else [],
                transforms,
                pairwise,
                records,
            )
            if homography is None or parent is None:
                logs.append(f"跳过图像: {records[idx].display_name}, 原因=右侧链路未建立")
                continue
            transforms[idx] = transforms[parent] @ homography
            logs.append(
                f"加入右侧链: {records[idx].display_name}, 父节点={records[parent].display_name}, "
                f"连接得分={score:.1f}, 几何模型=affine_preferred"
            )

        return transforms

    def _ordered_chain_parents(self, idx: int, direction: str, total: int) -> list[int]:
        if direction == "left":
            candidates = [idx + 1, idx + 2]
        else:
            candidates = [idx - 1, idx - 2]
        return [candidate for candidate in candidates if 0 <= candidate < total]

    def _best_chain_parent(
        self,
        node: int,
        candidate_parents,
        transforms: dict[int, np.ndarray],
        pairwise: dict[tuple[int, int], MatchResult],
        records: list[ImageRecord],
    ) -> tuple[int | None, np.ndarray | None, float]:
        best_parent = None
        best_h = None
        best_score = -1.0
        for parent in candidate_parents:
            if parent not in transforms:
                continue
            key = (min(node, parent), max(node, parent))
            result = pairwise.get(key)
            if result is None or not result.success:
                continue
            if records and records[0].segment == "full":
                transform = self._resolve_full_arch_pair_transform(result, node, parent, records)
            else:
                transform = self._resolve_pair_transform(result, node, parent, records)
            if transform is None:
                continue
            distance_penalty = 12.0 * max(0, abs(node - parent) - 1)
            score = result.weighted_score - distance_penalty
            if score > best_score:
                best_parent = parent
                best_h = transform
                best_score = score
        return best_parent, best_h, best_score

    def _resolve_full_arch_pair_transform(
        self,
        result: MatchResult,
        node: int,
        parent: int,
        records: list[ImageRecord],
    ) -> np.ndarray | None:
        transform = self._derive_rigid_transform(result, node, parent)
        if transform is None:
            return None
        transform = self._refine_rigid_transform(records[node].image, records[parent].image, transform)
        if transform is None:
            return None
        if not self._is_reasonable_rigid_transform(transform, records[node].image.shape, records[parent].image.shape, node, parent):
            return None
        return transform

    def _derive_rigid_transform(self, result: MatchResult, node: int, parent: int) -> np.ndarray | None:
        pts0 = result.matched_points0
        pts1 = result.matched_points1
        if pts0 is None or pts1 is None or len(pts0) < 4:
            return None
        src_pts, dst_pts = (pts0, pts1) if node < parent else (pts1, pts0)
        affine, _ = cv2.estimateAffinePartial2D(
            src_pts.reshape(-1, 1, 2),
            dst_pts.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.5,
        )
        if affine is None:
            return None
        linear = affine[:, :2].astype(np.float64)
        u, _, vt = np.linalg.svd(linear)
        rigid = u @ vt
        if np.linalg.det(rigid) < 0:
            u[:, -1] *= -1
            rigid = u @ vt
        out = np.eye(3, dtype=np.float64)
        out[:2, :2] = rigid
        out[:2, 2] = affine[:, 2]
        return out

    def _refine_rigid_transform(
        self,
        src_image: np.ndarray,
        dst_image: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray | None:
        try:
            src_gray = self.matcher._prepare_gray(src_image)
            dst_gray = self.matcher._prepare_gray(dst_image)
            max_dim = max(src_gray.shape[0], src_gray.shape[1], dst_gray.shape[0], dst_gray.shape[1])
            scale = 1.0
            if max_dim > 1200:
                scale = 1200.0 / max_dim
                src_gray = cv2.resize(src_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                dst_gray = cv2.resize(dst_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            angle = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
            tx = transform[0, 2] * scale
            ty = transform[1, 2] * scale
            warp = np.array(
                [
                    [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), tx],
                    [np.sin(np.radians(angle)),  np.cos(np.radians(angle)), ty],
                ],
                dtype=np.float32,
            )
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
            _, refined = cv2.findTransformECC(
                dst_gray,
                src_gray,
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                None,
                5,
            )
            refined = refined.astype(np.float64)
            refined[:, 2] /= scale
            out = np.eye(3, dtype=np.float64)
            out[:2, :] = refined
            return out
        except Exception:
            return transform

    def _is_reasonable_rigid_transform(
        self,
        transform: np.ndarray,
        src_shape: tuple[int, ...],
        dst_shape: tuple[int, ...],
        node: int,
        parent: int,
    ) -> bool:
        linear = transform[:2, :2]
        should_be_identity = linear.T @ linear
        if np.max(np.abs(should_be_identity - np.eye(2))) > 0.08:
            return False
        rotation = np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))
        if abs(rotation) > 22.0:
            return False
        tx, ty = float(transform[0, 2]), float(transform[1, 2])
        src_h, src_w = src_shape[:2]
        dst_h, dst_w = dst_shape[:2]
        if abs(ty) > max(src_h, dst_h) * 0.35:
            return False
        if abs(tx) > max(src_w, dst_w) * 0.95:
            return False
        if node < parent and tx < -src_w * 0.15:
            return False
        if node > parent and tx > src_w * 0.15:
            return False
        return True

    def _derive_chain_transform(self, result: MatchResult, node: int, parent: int) -> np.ndarray | None:
        pts0 = result.matched_points0
        pts1 = result.matched_points1
        if pts0 is not None and pts1 is not None and len(pts0) >= 6:
            src_pts, dst_pts = (pts0, pts1) if node < parent else (pts1, pts0)
            affine, _ = cv2.estimateAffinePartial2D(
                src_pts.reshape(-1, 1, 2),
                dst_pts.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=4.0,
            )
            if affine is not None:
                return np.vstack([affine, [0.0, 0.0, 1.0]])
        fallback = result.homography if node < parent else result.inverse_homography
        if fallback is None:
            return None
        return fallback

    def _refine_chain_transform(
        self,
        src_image: np.ndarray,
        dst_image: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray | None:
        try:
            src_gray = self.matcher._prepare_gray(src_image)
            dst_gray = self.matcher._prepare_gray(dst_image)

            max_dim = max(src_gray.shape[0], src_gray.shape[1], dst_gray.shape[0], dst_gray.shape[1])
            scale = 1.0
            if max_dim > 1200:
                scale = 1200.0 / max_dim
                src_gray = cv2.resize(src_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                dst_gray = cv2.resize(dst_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            warp = transform[:2, :].astype(np.float32).copy()
            warp[:, 2] *= scale
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                80,
                1e-5,
            )
            _, refined = cv2.findTransformECC(
                dst_gray,
                src_gray,
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                None,
                5,
            )
            refined = refined.astype(np.float64)
            refined[:, 2] /= scale
            return np.vstack([refined, [0.0, 0.0, 1.0]])
        except Exception:
            return transform

    def _is_reasonable_chain_transform(
        self,
        transform: np.ndarray,
        src_shape: tuple[int, ...],
        dst_shape: tuple[int, ...],
    ) -> bool:
        if transform.shape != (3, 3):
            return False
        linear = transform[:2, :2]
        det = float(np.linalg.det(linear))
        if not np.isfinite(det) or abs(det) < 0.15 or abs(det) > 6.0:
            return False
        rotation = np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))
        if abs(rotation) > 35.0:
            return False
        perspective_mag = float(np.linalg.norm(transform[2, :2]))
        if perspective_mag > 0.001:
            return False

        src_h, src_w = src_shape[:2]
        dst_h, dst_w = dst_shape[:2]
        src_corners = np.float32([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(src_corners, transform).reshape(-1, 2)
        widths = np.linalg.norm(warped[1] - warped[0]) + np.linalg.norm(warped[2] - warped[3])
        heights = np.linalg.norm(warped[3] - warped[0]) + np.linalg.norm(warped[2] - warped[1])
        mean_w = widths / 2.0
        mean_h = heights / 2.0
        if mean_w <= 1 or mean_h <= 1:
            return False
        aspect_ratio = mean_w / mean_h
        dst_aspect = dst_w / max(dst_h, 1)
        if aspect_ratio > max(2.6, dst_aspect * 1.8) or aspect_ratio < min(0.45, dst_aspect * 0.6):
            return False

        translation = np.linalg.norm(transform[:2, 2])
        if translation > max(src_w, src_h) * 1.8:
            return False
        return True

    def ordered_sequence(self, pairwise: dict[tuple[int, int], MatchResult], anchor_index: int, count: int) -> list[int]:
        scores = []
        for idx in range(count):
            if idx == anchor_index:
                continue
            key = (min(idx, anchor_index), max(idx, anchor_index))
            result = pairwise.get(key)
            weighted = result.weighted_score if result and result.success else -1.0
            scores.append((weighted, -abs(idx - anchor_index), idx))
        scores.sort(reverse=True)
        ordered = [anchor_index]
        for weighted, _, idx in scores:
            if weighted >= 0:
                ordered.append(idx)
        return ordered

    def _sequence_weight(self, distance: int) -> float:
        # 完整牙弓模式需要跨过中心向两侧延展，不能像半侧那样惩罚过重。
        if distance <= 1:
            return 1.0
        if distance == 2:
            return 0.78
        if distance == 3:
            return 0.58
        return 0.32

    def _neighbor_priority(self, distance: int) -> int:
        return max(0, 10 - distance)

    def _warp_and_blend(
        self,
        records: list[ImageRecord],
        transforms: dict[int, np.ndarray],
        logs: list[str],
    ) -> np.ndarray | None:
        corners = []
        for idx, homography in transforms.items():
            image = records[idx].image
            h, w = image.shape[:2]
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(pts, homography)
            corners.append(warped)

        if not corners:
            return None

        all_corners = np.concatenate(corners, axis=0)
        min_x, min_y = np.floor(all_corners.min(axis=0).ravel()).astype(int)
        max_x, max_y = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

        tx = -min_x if min_x < 0 else 0
        ty = -min_y if min_y < 0 else 0
        translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            return None

        accum = np.zeros((height, width, 3), dtype=np.float32)
        weight_sum = np.zeros((height, width, 1), dtype=np.float32)

        for idx, homography in transforms.items():
            image = records[idx].image.astype(np.float32)
            h, w = image.shape[:2]
            weight = self._feather_weight(h, w)
            full_h = translation @ homography
            warped_image = cv2.warpPerspective(image, full_h, (width, height))
            warped_weight = cv2.warpPerspective(weight, full_h, (width, height))[..., None]

            accum += warped_image * warped_weight
            weight_sum += warped_weight

        panorama = accum / np.clip(weight_sum, 1e-6, None)
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)
        valid_mask = (weight_sum[..., 0] > 0).astype(np.uint8) * 255
        valid_mask = self._largest_component(valid_mask)
        x, y, w, h = cv2.boundingRect(valid_mask)
        logs.append(f"输出全景尺寸: {w}x{h}")
        return panorama[y : y + h, x : x + w]

    def _feather_weight(self, height: int, width: int) -> np.ndarray:
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        dist_x = 1.0 - np.abs(xx - 0.5) * 2.0
        dist_y = 1.0 - np.abs(yy - 0.5) * 2.0
        weight = np.clip(np.minimum(dist_x, dist_y), 0.05, 1.0)
        return weight


    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest = np.zeros_like(mask)
        largest[labels == largest_label] = 255
        largest = cv2.morphologyEx(largest, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8))
        return largest
