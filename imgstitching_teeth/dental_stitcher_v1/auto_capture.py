"""自动拍照与引导逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from dental_stitcher_v1.capture_region import RegionAssessment, infer_simple_capture_region
from dental_stitcher_v1.photo_quality import QualityReport, assess_photo_quality

MIN_ACCEPTED_IMAGES = 12
TARGET_ACCEPTED_IMAGES = 16
AUTO_CAPTURE_MIN_INTERVAL_S = 1.25
MIN_QUALITY_ACCEPTANCE = 0.48
MIN_STABILITY_SCORE = 0.40
MIN_NOVELTY_SCORE = 0.22
MIN_BLOCKING_SHARPNESS = 10.0
MIN_BLOCKING_CONTENT_RATIO = 0.06
MIN_BLOCKING_BRIGHTNESS = 22.0
MAX_BLOCKING_BRIGHTNESS = 250.0


@dataclass(frozen=True)
class GuidedCaptureStep:
    key: str
    label: str
    instruction: str


@dataclass
class AutoCaptureAssessment:
    quality_report: QualityReport
    region_assessment: RegionAssessment
    acceptability_score: float
    capture_readiness: float
    stability_score: float
    novelty_score: float
    cooldown_remaining_s: float
    should_capture: bool
    capture_ready: bool
    current_step_index: int
    current_step_label: str
    current_instruction: str
    expected_region: str
    region_matched: bool
    status_level: str
    status_text: str
    blocking_reasons: list[str] = field(default_factory=list)
    highlight_messages: list[str] = field(default_factory=list)


@dataclass
class ArchProgress:
    arch_key: str
    arch_label: str
    accepted_count: int
    minimum_required: int
    recommended_target: int
    remaining_to_minimum: int
    remaining_to_target: int
    average_acceptance: float
    completion_score: float
    completion_label: str
    current_step_index: int
    current_step_label: str
    next_instruction: str
    minimum_met: bool
    recommended_met: bool


_ARCH_LABELS = {
    "upper": "上牙弓",
    "lower": "下牙弓",
}

_GUIDED_STEPS = {
    "upper": [
        GuidedCaptureStep("left_posterior", "左后牙起点", "从左后牙开始，镜头对准咬合面和颊侧交界。"),
        GuidedCaptureStep("left_mid", "左侧中段", "沿牙弓缓慢向前移动，保持上一张约 60% 重叠。"),
        GuidedCaptureStep("left_anterior", "左前牙段", "继续向前牙过渡，略微变换角度但不要跳拍。"),
        GuidedCaptureStep("incisors", "前牙正中", "把前牙弧度放在画面中央，先稳住再继续。"),
        GuidedCaptureStep("right_anterior", "右前牙段", "越过中线后继续缓慢移动，保持镜头高度一致。"),
        GuidedCaptureStep("right_mid", "右侧中段", "继续扫向右侧中段，避免只拍切缘。"),
        GuidedCaptureStep("right_posterior", "右后牙收尾", "让右后牙清楚出现，后牙区域至少保留两张清晰图。"),
        GuidedCaptureStep("right_posterior_extra", "右后牙补强", "轻微调整角度补一张右后牙，给重建留下冗余。"),
    ],
    "lower": [
        GuidedCaptureStep("left_posterior", "左后牙起点", "从下牙左后牙开始，镜头略偏下方看清牙弓连续形态。"),
        GuidedCaptureStep("left_mid", "左侧中段", "沿下牙弓缓慢向前移动，保持上一张约 60% 重叠。"),
        GuidedCaptureStep("left_anterior", "左前牙段", "继续向前牙过渡，保持镜头与牙面距离基本一致。"),
        GuidedCaptureStep("incisors", "前牙正中", "把前牙与牙弓中线放到画面中央，先停稳半秒。"),
        GuidedCaptureStep("right_anterior", "右前牙段", "越过中线后继续扫向右侧，保持连续不要跳拍。"),
        GuidedCaptureStep("right_mid", "右侧中段", "继续拍右侧中段，注意看到完整牙列轮廓。"),
        GuidedCaptureStep("right_posterior", "右后牙收尾", "把右后牙拍清楚，尤其注意最后几颗牙不要漏掉。"),
        GuidedCaptureStep("right_posterior_extra", "右后牙补强", "轻微换角度补一张右后牙，增加后牙区重叠。"),
    ],
}


def get_capture_steps(phase: str) -> list[GuidedCaptureStep]:
    return list(_GUIDED_STEPS[_normalize_arch_key(phase)])


def evaluate_auto_capture_frame(
    *,
    image: np.ndarray,
    phase: str,
    accepted_count: int,
    previous_preview: Optional[np.ndarray] = None,
    last_saved_image: Optional[np.ndarray] = None,
    seconds_since_last_capture: float = AUTO_CAPTURE_MIN_INTERVAL_S,
) -> AutoCaptureAssessment:
    """评估当前帧是否适合自动保存，并给出引导信息。"""
    arch_key = _normalize_arch_key(phase)
    steps = _GUIDED_STEPS[arch_key]
    step_index = min(accepted_count // 2, len(steps) - 1)
    step = steps[step_index]

    quality = assess_photo_quality(image)
    region = infer_simple_capture_region(image)
    stability_score = _estimate_stability_score(image, previous_preview)
    novelty_score = _estimate_novelty_score(image, last_saved_image)
    cooldown_remaining = max(0.0, AUTO_CAPTURE_MIN_INTERVAL_S - max(seconds_since_last_capture, 0.0))
    expected_region = _expected_region_for_step(step)

    quality_ready = _is_quality_ready_for_capture(quality)
    stability_ready = previous_preview is not None and stability_score >= MIN_STABILITY_SCORE
    novelty_ready = last_saved_image is None or novelty_score >= MIN_NOVELTY_SCORE
    cooldown_ready = cooldown_remaining <= 1e-6
    region_matched = region.predicted_region == expected_region

    capture_ready = quality_ready and stability_ready and cooldown_ready
    should_capture = capture_ready and novelty_ready

    blocking_reasons = _collect_blocking_reasons(
        quality=quality,
        region=region,
        expected_region=expected_region,
        accepted_count=accepted_count,
        previous_preview=previous_preview,
        stability_score=stability_score,
        last_saved_image=last_saved_image,
        novelty_score=novelty_score,
        cooldown_remaining=cooldown_remaining,
    )
    highlight_messages = [
        _describe_current_target(step, accepted_count),
        step.instruction,
        _format_region_guidance(region, expected_region),
    ]

    if should_capture:
        status_level = "success"
        status_text = f"当前画面合格，系统会自动保存这张 {step.label} 照片。"
    elif quality_ready and stability_ready and not novelty_ready:
        status_level = "info"
        status_text = "画面本身合格，但和上一张太像了，请沿牙弓继续缓慢移动。"
    elif quality_ready and not stability_ready:
        status_level = "info"
        status_text = "画面已基本合格，先停稳半秒，系统就会自动抓拍。"
    elif quality_ready and stability_ready and not cooldown_ready:
        status_level = "info"
        status_text = f"系统刚保存过，约 {cooldown_remaining:.1f} 秒后会继续自动抓拍。"
    else:
        status_level = "warning"
        status_text = "当前画面还没达到自动保存条件，请按提示微调。"

    capture_readiness = float(np.clip(
        0.60 * quality.acceptance_score
        + 0.20 * stability_score
        + 0.12 * novelty_score
        + 0.08 * (1.0 if cooldown_ready else 0.0),
        0.0,
        1.0,
    ))

    return AutoCaptureAssessment(
        quality_report=quality,
        region_assessment=region,
        acceptability_score=quality.acceptance_score,
        capture_readiness=capture_readiness,
        stability_score=stability_score,
        novelty_score=novelty_score,
        cooldown_remaining_s=cooldown_remaining,
        should_capture=should_capture,
        capture_ready=capture_ready,
        current_step_index=step_index,
        current_step_label=step.label,
        current_instruction=step.instruction,
        expected_region=expected_region,
        region_matched=region_matched,
        status_level=status_level,
        status_text=status_text,
        blocking_reasons=blocking_reasons,
        highlight_messages=highlight_messages,
    )


def summarize_arch_progress(
    packets: Sequence,
    phase: str,
    minimum_required: int = MIN_ACCEPTED_IMAGES,
    recommended_target: int = TARGET_ACCEPTED_IMAGES,
) -> ArchProgress:
    """汇总当前牙弓采集进度。"""
    arch_key = _normalize_arch_key(phase)
    steps = _GUIDED_STEPS[arch_key]
    count = len(packets)
    scores = [
        float(getattr(packet, "acceptance_score"))
        for packet in packets
        if getattr(packet, "acceptance_score", None) is not None
    ]
    average_acceptance = float(np.mean(scores)) if scores else 0.0

    count_score = min(count / max(recommended_target, 1), 1.0)
    quality_score = average_acceptance if scores else 0.0
    completion_score = float(np.clip(0.78 * count_score + 0.22 * quality_score, 0.0, 1.0))

    minimum_met = count >= minimum_required
    recommended_met = count >= recommended_target

    if recommended_met:
        completion_label = "已达推荐采集量"
        current_step_label = "当前牙弓已完成"
        next_instruction = "当前牙弓照片数量和质量都已足够，可以自动进入下一阶段。"
        step_index = len(steps) - 1
    else:
        step_index = min(count // 2, len(steps) - 1)
        step = steps[step_index]
        current_step_label = step.label
        next_instruction = _describe_current_target(step, count)
        if minimum_met:
            completion_label = "已达最低可用量"
        elif count > 0:
            completion_label = "继续采集"
        else:
            completion_label = "待开始"

    return ArchProgress(
        arch_key=arch_key,
        arch_label=_ARCH_LABELS[arch_key],
        accepted_count=count,
        minimum_required=minimum_required,
        recommended_target=recommended_target,
        remaining_to_minimum=max(0, minimum_required - count),
        remaining_to_target=max(0, recommended_target - count),
        average_acceptance=average_acceptance,
        completion_score=completion_score,
        completion_label=completion_label,
        current_step_index=step_index,
        current_step_label=current_step_label,
        next_instruction=next_instruction,
        minimum_met=minimum_met,
        recommended_met=recommended_met,
    )


def _normalize_arch_key(phase: str) -> str:
    mapping = {
        "upper": "upper",
        "upper_arch": "upper",
        "lower": "lower",
        "lower_arch": "lower",
    }
    arch_key = mapping.get(phase)
    if arch_key is None:
        raise ValueError(f"Unsupported capture phase: {phase}")
    return arch_key


def _describe_current_target(step: GuidedCaptureStep, accepted_count: int) -> str:
    local_index = (accepted_count % 2) + 1
    return f"当前目标：{step.label} 第 {local_index} 张，保持上一张约 60% 重叠。"


def _collect_blocking_reasons(
    *,
    quality: QualityReport,
    region: RegionAssessment,
    expected_region: str,
    accepted_count: int,
    previous_preview: Optional[np.ndarray],
    stability_score: float,
    last_saved_image: Optional[np.ndarray],
    novelty_score: float,
    cooldown_remaining: float,
) -> list[str]:
    reasons: list[str] = []

    if previous_preview is None:
        reasons.append("先稳定住镜头约半秒，让系统建立当前区域的参考画面。")
    if quality.fail_reasons:
        reasons.extend(quality.fail_reasons[:2])
    if quality.subject_fill_ratio < 0.12:
        reasons.append("镜头再靠近一点，让牙齿更多地占据画面。")
    elif quality.subject_fill_ratio > 0.88:
        reasons.append("镜头稍微拉远一点，避免局部过近。")
    if quality.subject_center_offset > 0.28:
        reasons.append("把牙列尽量移到画面中央。")
    if region.predicted_region == "unknown":
        reasons.append("左/中/右区域判断还不稳定，先把牙列更清楚地露出来。")
    elif region.predicted_region != expected_region and region.confidence >= 0.45:
        reasons.append(
            f"当前建议继续扫描{_region_label(expected_region)}，但系统不会因为左右位置不同而拦截保存。"
        )
    if quality.acceptance_score < MIN_QUALITY_ACCEPTANCE:
        reasons.append("清晰度或取景还差一点，请继续微调。")
    if previous_preview is not None and stability_score < MIN_STABILITY_SCORE:
        reasons.append("保持当前姿势不要晃动，停稳后系统会自动抓拍。")
    if last_saved_image is not None and novelty_score < MIN_NOVELTY_SCORE:
        reasons.append("这一帧和上一张太像了，沿牙弓继续移动一点再拍。")
    if cooldown_remaining > 1e-6:
        reasons.append(f"系统刚保存过，约 {cooldown_remaining:.1f} 秒后继续判断。")

    if not reasons and accepted_count == 0:
        reasons.append("继续从左后牙起步，系统会在画面合格时自动保存。")
    return _dedupe_preserve_order(reasons)


def _estimate_stability_score(image: np.ndarray, previous_image: Optional[np.ndarray]) -> float:
    if previous_image is None:
        return 0.0
    delta = _frame_delta(image, previous_image)
    return float(np.clip(1.0 - min(delta / 0.11, 1.0), 0.0, 1.0))


def _estimate_novelty_score(image: np.ndarray, last_saved_image: Optional[np.ndarray]) -> float:
    if last_saved_image is None:
        return 1.0
    delta = _frame_delta(image, last_saved_image)
    return float(np.clip(min(delta / 0.12, 1.0), 0.0, 1.0))


def _frame_delta(image_a: np.ndarray, image_b: np.ndarray) -> float:
    sample_a = _sample_frame(image_a)
    sample_b = _sample_frame(image_b)
    return float(np.mean(np.abs(sample_a.astype(np.float32) - sample_b.astype(np.float32))) / 255.0)


def _sample_frame(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (96, 72), interpolation=cv2.INTER_AREA)


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _expected_region_for_step(step: GuidedCaptureStep) -> str:
    if step.key.startswith("left"):
        return "left"
    if step.key == "incisors":
        return "center"
    return "right"


def _format_region_guidance(region: RegionAssessment, expected_region: str) -> str:
    if region.predicted_region == "unknown":
        return f"扫描引导：当前建议拍{_region_label(expected_region)}，区域识别还不稳定。"
    if region.predicted_region == expected_region:
        return (
            f"扫描引导：当前疑似{_region_label(region.predicted_region)}，"
            f"与建议扫描区域一致（{region.confidence * 100:.0f}%）。"
        )
    return (
        f"扫描引导：当前疑似{_region_label(region.predicted_region)}，"
        f"当前建议继续拍{_region_label(expected_region)}。"
    )


def _region_label(region: str) -> str:
    labels = {
        "left": "左侧段",
        "center": "前牙段",
        "right": "右侧段",
        "unknown": "未知区域",
    }
    return labels.get(region, region)


def _is_quality_ready_for_capture(quality: QualityReport) -> bool:
    if quality.acceptance_score < MIN_QUALITY_ACCEPTANCE:
        return False
    if not quality.color_validity:
        return False
    if quality.sharpness < MIN_BLOCKING_SHARPNESS:
        return False
    if quality.effective_content_ratio < MIN_BLOCKING_CONTENT_RATIO:
        return False
    if quality.brightness < MIN_BLOCKING_BRIGHTNESS or quality.brightness > MAX_BLOCKING_BRIGHTNESS:
        return False
    return True
