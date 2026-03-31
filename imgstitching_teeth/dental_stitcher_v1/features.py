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


def extract_features(image: np.ndarray, mask: np.ndarray, method: str = "orb") -> FeaturesResult:
    if method == "loftr":
        return FeaturesResult(keypoints=[], descriptors=None, method="loftr", fallback_reason="deep_model_unavailable")
    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.02, edgeThreshold=10)
    elif method == "akaze":
        detector = cv2.AKAZE_create()
    else:
        detector = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)

    keypoints, descriptors = detector.detectAndCompute(image, mask)
    if descriptors is None or len(keypoints) < 15:
        return FeaturesResult(keypoints=keypoints or [], descriptors=descriptors, method=method, fallback_reason="low_feature_count")
    return FeaturesResult(keypoints=keypoints, descriptors=descriptors, method=method)


def match_features(desc1: Optional[np.ndarray], desc2: Optional[np.ndarray], method: str) -> MatchResult:
    if desc1 is None or desc2 is None:
        return MatchResult(matches=[], method=method)

    if method == "orb" or method == "akaze":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    good: list[cv2.DMatch] = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return MatchResult(matches=good, method=method)
