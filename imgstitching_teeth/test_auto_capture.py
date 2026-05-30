from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from dental_stitcher_v1.capture_region import infer_simple_capture_region
from dental_stitcher_v1.auto_capture import evaluate_auto_capture_frame, summarize_arch_progress
from dental_stitcher_v1.photo_quality import assess_photo_quality


def _make_test_image(*, shift: int = 0, blur: bool = False) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (6, 6, 8)

    x0 = 110 + shift
    x1 = 520 + shift
    cv2.rectangle(image, (x0, 140), (x1, 340), (180, 205, 235), thickness=-1)
    cv2.rectangle(image, (x0 + 20, 160), (x1 - 20, 320), (150, 185, 220), thickness=8)
    for idx in range(8):
        x = x0 + 35 + idx * 45
        cv2.line(image, (x, 150), (x, 330), (95, 125, 170), thickness=3)
    cv2.circle(image, (x0 + 120, 235), 28, (210, 235, 250), thickness=6)
    cv2.circle(image, (x1 - 120, 235), 28, (210, 235, 250), thickness=6)

    if blur:
        image = cv2.GaussianBlur(image, (11, 11), 0)
    return image


def _make_region_image(region: str) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (32, 22, 34)

    if region == "center":
        centers = [150, 210, 270, 330, 390, 450]
        widths = [54, 56, 58, 58, 56, 54]
    elif region == "left":
        centers = [120, 170, 220, 270, 320]
        widths = [58, 60, 58, 52, 48]
    elif region == "right":
        centers = [320, 370, 420, 470, 520]
        widths = [48, 52, 58, 60, 58]
    else:
        raise ValueError(region)

    for idx, (cx, width) in enumerate(zip(centers, widths)):
        x1 = int(cx - width // 2)
        x2 = int(cx + width // 2)
        y1 = 250 - max(0, 2 - abs(idx - len(centers) // 2)) * 14
        y2 = 390
        cv2.rectangle(image, (x1, y1), (x2, y2), (210, 225, 240), thickness=-1)
        cv2.rectangle(image, (x1 + 8, y1 + 8), (x2 - 8, y2 - 10), (184, 205, 228), thickness=4)
    cv2.ellipse(image, (320, 320), (235, 110), 0, 10, 170, (92, 52, 74), thickness=24)
    return image


def test_assess_photo_quality_scores_sharp_frame() -> None:
    report = assess_photo_quality(_make_test_image())

    assert report.passed is True
    assert report.acceptance_score > 0.62
    assert report.framing_score > 0.45
    assert report.acceptability_label in {"acceptable", "excellent"}


def test_assess_photo_quality_penalizes_blur() -> None:
    sharp_report = assess_photo_quality(_make_test_image())
    blur_report = assess_photo_quality(_make_test_image(blur=True))

    assert blur_report.acceptance_score < sharp_report.acceptance_score
    assert blur_report.passed is False or blur_report.acceptability_label == "retry"


def test_auto_capture_accepts_stable_novel_frame() -> None:
    current = _make_region_image("left")
    previous = current.copy()

    assessment = evaluate_auto_capture_frame(
        image=current,
        phase="upper_arch",
        accepted_count=0,
        previous_preview=previous,
        last_saved_image=None,
        seconds_since_last_capture=3.0,
    )

    assert assessment.should_capture is True
    assert assessment.capture_ready is True
    assert assessment.status_level == "success"


def test_auto_capture_rejects_duplicate_frame() -> None:
    current = _make_test_image()

    assessment = evaluate_auto_capture_frame(
        image=current,
        phase="upper_arch",
        accepted_count=4,
        previous_preview=current.copy(),
        last_saved_image=current.copy(),
        seconds_since_last_capture=3.0,
    )

    assert assessment.should_capture is False
    assert any("太像" in reason for reason in assessment.blocking_reasons)


def test_summarize_arch_progress_tracks_thresholds() -> None:
    packets = [SimpleNamespace(acceptance_score=0.8) for _ in range(12)]
    progress = summarize_arch_progress(packets, "lower_arch")

    assert progress.minimum_met is True
    assert progress.recommended_met is False
    assert progress.remaining_to_target == 4
    assert progress.completion_label == "已达最低可用量"

    progress_done = summarize_arch_progress(
        [SimpleNamespace(acceptance_score=0.82) for _ in range(16)],
        "lower_arch",
    )
    assert progress_done.recommended_met is True
    assert progress_done.completion_label == "已达推荐采集量"


def test_simple_region_recognition_detects_left_center_right() -> None:
    assert infer_simple_capture_region(_make_region_image("left")).predicted_region == "left"
    assert infer_simple_capture_region(_make_region_image("center")).predicted_region == "center"
    assert infer_simple_capture_region(_make_region_image("right")).predicted_region == "right"


def test_auto_capture_wrong_side_only_changes_guidance() -> None:
    current = _make_region_image("center")
    assessment = evaluate_auto_capture_frame(
        image=current,
        phase="upper_arch",
        accepted_count=0,  # 目标是左侧段
        previous_preview=current.copy(),
        last_saved_image=None,
        seconds_since_last_capture=3.0,
    )

    assert assessment.region_assessment.predicted_region == "center"
    assert assessment.region_matched is False
    assert assessment.should_capture is True
    assert any("不会因为左右位置不同而拦截保存" in reason for reason in assessment.blocking_reasons)
