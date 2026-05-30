from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class FeaturesResult:
    keypoints: list[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    method: str
    fallback_reason: Optional[str] = None


@dataclass
class MatchResult:
    matches: list[cv2.DMatch]
    method: str


MIN_FEATURE_KEYPOINTS = 15


def extract_features(image: np.ndarray, mask: np.ndarray, method: str = "sift") -> FeaturesResult:
    if method == "loftr":
        return FeaturesResult(keypoints=[], descriptors=None, method="loftr", fallback_reason="deep_model_unavailable")

    prepared_image = _prepare_feature_image(image)
    prepared_mask = _expand_feature_mask(mask, prepared_image.shape[:2])

    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01, edgeThreshold=12, sigma=1.2)
    elif method == "akaze":
        detector = cv2.AKAZE_create()
    else:
        detector = cv2.ORB_create(nfeatures=3500, scaleFactor=1.2, nlevels=8, fastThreshold=12)

    keypoints, descriptors = detector.detectAndCompute(prepared_image, prepared_mask)
    if method == "sift" and descriptors is not None:
        descriptors = _to_root_sift(descriptors)

    if descriptors is None or len(keypoints) < MIN_FEATURE_KEYPOINTS:
        return FeaturesResult(keypoints=keypoints or [], descriptors=descriptors, method=method, fallback_reason="low_feature_count")
    return FeaturesResult(keypoints=keypoints, descriptors=descriptors, method=method)


def match_features(desc1: Optional[np.ndarray], desc2: Optional[np.ndarray], method: str) -> MatchResult:
    if desc1 is None or desc2 is None:
        return MatchResult(matches=[], method=method)

    if method == "orb" or method == "akaze":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        ratio_threshold = 0.78
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        ratio_threshold = 0.82

    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    reverse_raw_matches = matcher.knnMatch(desc2, desc1, k=2) if method == "sift" else None
    good: list[cv2.DMatch] = []

    reverse_pairs: set[tuple[int, int]] = set()
    if reverse_raw_matches is not None:
        for pair in reverse_raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                reverse_pairs.add((m.queryIdx, m.trainIdx))

    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance >= ratio_threshold * n.distance:
            continue
        if reverse_pairs and (m.trainIdx, m.queryIdx) not in reverse_pairs:
            continue
        good.append(m)

    if method == "sift" and len(good) < 8:
        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good.append(m)

    good.sort(key=lambda match: match.distance)
    return MatchResult(matches=good, method=method)


def _prepare_feature_image(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced


def _expand_feature_mask(mask: Optional[np.ndarray], target_shape: tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None

    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    if mask_np.shape[:2] != target_shape:
        mask_np = cv2.resize(mask_np, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_bin = (mask_np > 0).astype(np.uint8) * 255
    kernel_size = max(9, int(round(min(target_shape) * 0.035)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask_bin, kernel, iterations=1)


def _to_root_sift(descriptors: np.ndarray) -> np.ndarray:
    descriptors = descriptors.astype(np.float32, copy=True)
    row_sums = descriptors.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    descriptors /= row_sums
    np.sqrt(descriptors, out=descriptors)
    return descriptors
