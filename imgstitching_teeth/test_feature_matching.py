from __future__ import annotations

import cv2
import numpy as np

from dental_stitcher_v1.features import extract_features, match_features
from dental_stitcher_v1.pipeline import _select_feature_pair_for_matching


def _make_feature_rich_image() -> np.ndarray:
    image = np.zeros((480, 640), dtype=np.uint8)
    image[:] = 28

    cv2.rectangle(image, (60, 60), (580, 420), 110, thickness=3)
    for idx in range(7):
        x = 100 + idx * 65
        cv2.circle(image, (x, 180), 24 + (idx % 3) * 4, 185, thickness=3)
        cv2.line(image, (x - 30, 300), (x + 30, 360), 220, thickness=2)
        cv2.line(image, (x + 30, 300), (x - 30, 360), 145, thickness=2)

    cv2.putText(image, "DENTAL", (180, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 240, 2, cv2.LINE_AA)
    cv2.putText(image, "ARCH", (240, 405), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 190, 2, cv2.LINE_AA)
    cv2.ellipse(image, (320, 250), (170, 95), 0, 0, 360, 150, thickness=2)
    return image


def _make_shifted_variant(image: np.ndarray) -> np.ndarray:
    transform = np.float32([[1.0, 0.0, 14.0], [0.0, 1.0, -10.0]])
    shifted = cv2.warpAffine(image, transform, (image.shape[1], image.shape[0]), borderValue=18)
    shifted = cv2.convertScaleAbs(shifted, alpha=1.08, beta=18)
    return shifted


def test_sift_extracts_features_from_masked_realistic_pattern() -> None:
    image = _make_feature_rich_image()
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.rectangle(mask, (90, 90), (550, 390), 255, thickness=-1)

    features = extract_features(image, mask, method="sift")

    assert features.descriptors is not None
    assert len(features.keypoints) >= 30
    assert features.method == "sift"


def test_sift_matches_brightness_shifted_images() -> None:
    image_a = _make_feature_rich_image()
    image_b = _make_shifted_variant(image_a)
    mask = np.ones_like(image_a, dtype=np.uint8) * 255

    features_a = extract_features(image_a, mask, method="sift")
    features_b = extract_features(image_b, mask, method="sift")
    matches = match_features(features_a.descriptors, features_b.descriptors, "sift")

    assert len(matches.matches) >= 20


def test_pipeline_feature_selection_falls_back_to_sift_when_needed() -> None:
    image_a = _make_feature_rich_image()
    image_b = _make_shifted_variant(image_a)
    mask = np.ones_like(image_a, dtype=np.uint8) * 255

    features_a, features_b, method, fallback_reason = _select_feature_pair_for_matching(
        image_a,
        mask,
        image_b,
        mask,
        preferred_method="loftr",
    )

    assert method == "sift"
    assert features_a.descriptors is not None
    assert features_b.descriptors is not None
    assert fallback_reason == "feature_fallback_loftr_to_sift"
