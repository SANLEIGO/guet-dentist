"""照片质量检测模块 — 用于拍摄采集时实时评估照片是否适合 COLMAP 3D 重建。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class QualityReport:
    """单张照片的质量评估结果。"""

    passed: bool
    sharpness: float = 0.0
    sharpness_grade: str = "blurry"          # "good" / "acceptable" / "blurry"
    exposure: float = 0.0                    # 0-1 曝光评分
    brightness: float = 0.0                  # 0-255 平均亮度
    brightness_grade: str = "good"           # "good" / "dark" / "overexposed"
    overexposed_ratio: float = 0.0           # 亮度>250 像素占比
    underexposed_ratio: float = 0.0          # 亮度<20 像素占比
    effective_content_ratio: float = 0.0     # 有效内容占比
    color_validity: bool = True              # 颜色是否正常
    subject_fill_ratio: float = 0.0          # 主体框占画面比例
    subject_center_offset: float = 1.0       # 主体中心偏离画面中心程度（0-1）
    framing_score: float = 0.0               # 取景评分（0-1）
    acceptance_score: float = 0.0            # 综合可接受度（0-1）
    acceptability_label: str = "retry"       # "excellent" / "acceptable" / "retry"
    fail_reasons: list[str] = field(default_factory=list)
    warn_reasons: list[str] = field(default_factory=list)


# ── 阈值常量 ──────────────────────────────────────────────

SHARPNESS_GOOD = 60.0
SHARPNESS_ACCEPTABLE = 18.0

BRIGHTNESS_MIN_GOOD = 70
BRIGHTNESS_MAX_GOOD = 210
BRIGHTNESS_MIN_ACCEPTABLE = 35
BRIGHTNESS_MAX_ACCEPTABLE = 245

OVEREXPOSED_GOOD = 0.08          # 8%
OVEREXPOSED_ACCEPTABLE = 0.28    # 28%
UNDEREXPOSED_GOOD = 0.18         # 18%
UNDEREXPOSED_ACCEPTABLE = 0.50   # 50%

EFFECTIVE_CONTENT_GOOD = 0.25    # 25%
EFFECTIVE_CONTENT_ACCEPTABLE = 0.10  # 10%

BLACK_BORDER_THRESHOLD = 15       # 亮度低于此值视为黑边
SUBJECT_CENTER_WARN = 0.28
SUBJECT_FILL_WARN_MIN = 0.12
SUBJECT_FILL_WARN_MAX = 0.88


def assess_photo_quality(image: np.ndarray) -> QualityReport:
    """评估单张照片的质量，判断是否适合用于 3D 重建。

    Args:
        image: BGR 格式 numpy 数组

    Returns:
        QualityReport 包含各项指标和是否通过的判定
    """
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []

    if image is None or image.size == 0:
        return QualityReport(
            passed=False,
            fail_reasons=["image_empty"],
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── 1. 清晰度（Laplacian 方差）──
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if sharpness >= SHARPNESS_GOOD:
        sharpness_grade = "good"
    elif sharpness >= SHARPNESS_ACCEPTABLE:
        sharpness_grade = "acceptable"
        warn_reasons.append(f"清晰度偏低 ({sharpness:.1f} < {SHARPNESS_GOOD})")
    else:
        sharpness_grade = "blurry"
        fail_reasons.append(f"图像模糊 ({sharpness:.1f} < {SHARPNESS_ACCEPTABLE})")

    effective_mask = gray >= BLACK_BORDER_THRESHOLD
    effective_pixels = gray[effective_mask] if np.any(effective_mask) else gray.reshape(-1)

    # ── 2. 亮度 / 曝光（只统计有效内容区域，避免黑边误伤）──
    brightness = float(effective_pixels.mean())
    if BRIGHTNESS_MIN_GOOD <= brightness <= BRIGHTNESS_MAX_GOOD:
        brightness_grade = "good"
    elif BRIGHTNESS_MIN_ACCEPTABLE <= brightness <= BRIGHTNESS_MAX_ACCEPTABLE:
        brightness_grade = "dark" if brightness < BRIGHTNESS_MIN_GOOD else "overexposed"
        warn_reasons.append(f"亮度偏差 ({brightness:.1f})")
    else:
        brightness_grade = "dark" if brightness < BRIGHTNESS_MIN_ACCEPTABLE else "overexposed"
        fail_reasons.append(f"亮度异常 ({brightness:.1f})")

    exposure = 1.0 - min(abs(brightness - 127.5) / 127.5, 1.0)

    # ── 3. 过曝 / 欠曝区域 ──
    total_pixels = gray.size
    exposure_pixels = max(int(effective_pixels.size), 1)
    overexposed_count = int(np.sum(effective_pixels > 250))
    underexposed_count = int(np.sum(effective_pixels < 20))
    overexposed_ratio = overexposed_count / exposure_pixels
    underexposed_ratio = underexposed_count / exposure_pixels

    if overexposed_ratio > OVEREXPOSED_ACCEPTABLE:
        fail_reasons.append(f"过曝区域过多 ({overexposed_ratio:.1%})")
    elif overexposed_ratio > OVEREXPOSED_GOOD:
        warn_reasons.append(f"轻微过曝 ({overexposed_ratio:.1%})")

    if underexposed_ratio > UNDEREXPOSED_ACCEPTABLE:
        fail_reasons.append(f"欠曝区域过多 ({underexposed_ratio:.1%})")
    elif underexposed_ratio > UNDEREXPOSED_GOOD:
        warn_reasons.append(f"轻微欠曝 ({underexposed_ratio:.1%})")

    # ── 4. 有效内容占比（非黑边）──
    effective_content_ratio = float(np.count_nonzero(effective_mask) / total_pixels)

    if effective_content_ratio < EFFECTIVE_CONTENT_ACCEPTABLE:
        fail_reasons.append(f"有效内容不足 ({effective_content_ratio:.1%})")
    elif effective_content_ratio < EFFECTIVE_CONTENT_GOOD:
        warn_reasons.append(f"有效内容偏少 ({effective_content_ratio:.1%})")

    # ── 5. 取景（居中 / 距离）──
    subject_fill_ratio, subject_center_offset, framing_score = _estimate_framing_metrics(effective_mask)
    if subject_fill_ratio < SUBJECT_FILL_WARN_MIN:
        warn_reasons.append(f"牙弓占画面偏少 ({subject_fill_ratio:.1%})，可再靠近一点")
    elif subject_fill_ratio > SUBJECT_FILL_WARN_MAX:
        warn_reasons.append(f"镜头可能过近 ({subject_fill_ratio:.1%})，可稍微拉远")

    if subject_center_offset > SUBJECT_CENTER_WARN:
        warn_reasons.append("牙弓偏离画面中心，可稍微回到中央")

    # ── 5. 颜色有效性（非全灰 / 全黑 / 全白）──
    color_validity = _check_color_validity(image)
    if not color_validity:
        fail_reasons.append("颜色异常（可能为全灰/全黑/全白）")

    # ── 6. 综合可接受度 ──
    acceptance_score = _compute_acceptance_score(
        sharpness=sharpness,
        brightness=brightness,
        overexposed_ratio=overexposed_ratio,
        underexposed_ratio=underexposed_ratio,
        effective_content_ratio=effective_content_ratio,
        framing_score=framing_score,
        color_validity=color_validity,
    )
    acceptability_label = _label_acceptance_score(acceptance_score)

    # ── 汇总判定 ──
    passed = len(fail_reasons) == 0

    return QualityReport(
        passed=passed,
        sharpness=sharpness,
        sharpness_grade=sharpness_grade,
        exposure=exposure,
        brightness=brightness,
        brightness_grade=brightness_grade,
        overexposed_ratio=overexposed_ratio,
        underexposed_ratio=underexposed_ratio,
        effective_content_ratio=effective_content_ratio,
        color_validity=color_validity,
        subject_fill_ratio=subject_fill_ratio,
        subject_center_offset=subject_center_offset,
        framing_score=framing_score,
        acceptance_score=acceptance_score,
        acceptability_label=acceptability_label,
        fail_reasons=fail_reasons,
        warn_reasons=warn_reasons,
    )


def _check_color_validity(image: np.ndarray) -> bool:
    """检查图像是否有有效的颜色信息（非全灰、全黑、全白）。"""
    if image.ndim != 3 or image.shape[2] != 3:
        return False

    b, g, r = cv2.split(image)
    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()

    # 全黑或全白
    if max(b_mean, g_mean, r_mean) < 10:
        return False
    if min(b_mean, g_mean, r_mean) > 245:
        return False

    # 三通道均值过于接近 = 可能是灰度图
    channel_range = max(b_mean, g_mean, r_mean) - min(b_mean, g_mean, r_mean)
    if channel_range < 3:
        return False

    return True


def _estimate_framing_metrics(effective_mask: np.ndarray) -> tuple[float, float, float]:
    """基于有效内容区域估计取景情况。"""
    if effective_mask.size == 0 or not np.any(effective_mask):
        return 0.0, 1.0, 0.0

    ys, xs = np.nonzero(effective_mask)
    height, width = effective_mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    bbox_area = float((x1 - x0 + 1) * (y1 - y0 + 1))
    total_area = float(height * width)
    subject_fill_ratio = bbox_area / total_area if total_area > 0 else 0.0

    center_x = ((x0 + x1) / 2.0) / max(width - 1, 1)
    center_y = ((y0 + y1) / 2.0) / max(height - 1, 1)
    center_distance = float(np.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2))
    subject_center_offset = min(center_distance / 0.70710678, 1.0)

    center_score = 1.0 - min(subject_center_offset / 0.45, 1.0)
    fill_score = 1.0 - min(abs(subject_fill_ratio - 0.42) / 0.42, 1.0)
    framing_score = float(np.clip(0.55 * center_score + 0.45 * fill_score, 0.0, 1.0))
    return subject_fill_ratio, subject_center_offset, framing_score


def _compute_acceptance_score(
    *,
    sharpness: float,
    brightness: float,
    overexposed_ratio: float,
    underexposed_ratio: float,
    effective_content_ratio: float,
    framing_score: float,
    color_validity: bool,
) -> float:
    """综合多项指标给出 0-1 可接受度。"""
    sharpness_score = _linear_score(sharpness, 12.0, SHARPNESS_GOOD)
    brightness_score = 1.0 - min(abs(brightness - 127.5) / 90.0, 1.0)
    exposure_penalty = 0.5 * min(overexposed_ratio / OVEREXPOSED_ACCEPTABLE, 1.0)
    exposure_penalty += 0.5 * min(underexposed_ratio / UNDEREXPOSED_ACCEPTABLE, 1.0)
    exposure_score = 1.0 - exposure_penalty
    content_score = _linear_score(effective_content_ratio, 0.10, EFFECTIVE_CONTENT_GOOD)
    color_score = 1.0 if color_validity else 0.0

    score = (
        0.30 * sharpness_score
        + 0.18 * brightness_score
        + 0.17 * exposure_score
        + 0.15 * content_score
        + 0.15 * framing_score
        + 0.05 * color_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _linear_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return float((value - low) / (high - low))


def _label_acceptance_score(score: float) -> str:
    if score >= 0.76:
        return "excellent"
    if score >= 0.48:
        return "acceptable"
    return "retry"


def format_quality_report(report: QualityReport) -> str:
    """将质量报告格式化为人类可读的文本。"""
    lines = []

    lines.append(f"清晰度: {report.sharpness:.1f} ({report.sharpness_grade})")
    lines.append(f"亮度: {report.brightness:.1f} ({report.brightness_grade})")
    lines.append(f"曝光评分: {report.exposure:.2f}")
    lines.append(f"过曝区域: {report.overexposed_ratio:.1%}")
    lines.append(f"欠曝区域: {report.underexposed_ratio:.1%}")
    lines.append(f"有效内容: {report.effective_content_ratio:.1%}")
    lines.append(f"主体占画面: {report.subject_fill_ratio:.1%}")
    lines.append(f"主体偏中心: {report.subject_center_offset:.2f}")
    lines.append(f"取景评分: {report.framing_score:.2f}")
    lines.append(f"颜色有效: {'是' if report.color_validity else '否'}")
    lines.append(f"综合可接受度: {report.acceptance_score:.2f} ({report.acceptability_label})")

    if report.passed:
        lines.append("\n结果: 通过")
    else:
        lines.append("\n结果: 未通过")
        for reason in report.fail_reasons:
            lines.append(f"  x {reason}")

    if report.warn_reasons:
        lines.append("\n警告:")
        for reason in report.warn_reasons:
            lines.append(f"  ! {reason}")

    return "\n".join(lines)
