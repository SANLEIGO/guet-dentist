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
    fail_reasons: list[str] = field(default_factory=list)
    warn_reasons: list[str] = field(default_factory=list)


# ── 阈值常量 ──────────────────────────────────────────────

SHARPNESS_GOOD = 80.0
SHARPNESS_ACCEPTABLE = 30.0

BRIGHTNESS_MIN_GOOD = 80
BRIGHTNESS_MAX_GOOD = 200
BRIGHTNESS_MIN_ACCEPTABLE = 50
BRIGHTNESS_MAX_ACCEPTABLE = 230

OVEREXPOSED_GOOD = 0.05          # 5%
OVEREXPOSED_ACCEPTABLE = 0.15    # 15%
UNDEREXPOSED_GOOD = 0.10         # 10%
UNDEREXPOSED_ACCEPTABLE = 0.30   # 30%

EFFECTIVE_CONTENT_GOOD = 0.70    # 70%
EFFECTIVE_CONTENT_ACCEPTABLE = 0.50  # 50%

BLACK_BORDER_THRESHOLD = 15       # 亮度低于此值视为黑边


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

    # ── 2. 亮度 / 曝光 ──
    brightness = float(gray.mean())
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
    overexposed_count = int(np.sum(gray > 250))
    underexposed_count = int(np.sum(gray < 20))
    overexposed_ratio = overexposed_count / total_pixels
    underexposed_ratio = underexposed_count / total_pixels

    if overexposed_ratio > OVEREXPOSED_ACCEPTABLE:
        fail_reasons.append(f"过曝区域过多 ({overexposed_ratio:.1%})")
    elif overexposed_ratio > OVEREXPOSED_GOOD:
        warn_reasons.append(f"轻微过曝 ({overexposed_ratio:.1%})")

    if underexposed_ratio > UNDEREXPOSED_ACCEPTABLE:
        fail_reasons.append(f"欠曝区域过多 ({underexposed_ratio:.1%})")
    elif underexposed_ratio > UNDEREXPOSED_GOOD:
        warn_reasons.append(f"轻微欠曝 ({underexposed_ratio:.1%})")

    # ── 4. 有效内容占比（非黑边）──
    effective_mask = gray >= BLACK_BORDER_THRESHOLD
    effective_content_ratio = float(np.count_nonzero(effective_mask) / total_pixels)

    if effective_content_ratio < EFFECTIVE_CONTENT_ACCEPTABLE:
        fail_reasons.append(f"有效内容不足 ({effective_content_ratio:.1%})")
    elif effective_content_ratio < EFFECTIVE_CONTENT_GOOD:
        warn_reasons.append(f"有效内容偏少 ({effective_content_ratio:.1%})")

    # ── 5. 颜色有效性（非全灰 / 全黑 / 全白）──
    color_validity = _check_color_validity(image)
    if not color_validity:
        fail_reasons.append("颜色异常（可能为全灰/全黑/全白）")

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


def format_quality_report(report: QualityReport) -> str:
    """将质量报告格式化为人类可读的文本。"""
    lines = []

    lines.append(f"清晰度: {report.sharpness:.1f} ({report.sharpness_grade})")
    lines.append(f"亮度: {report.brightness:.1f} ({report.brightness_grade})")
    lines.append(f"曝光评分: {report.exposure:.2f}")
    lines.append(f"过曝区域: {report.overexposed_ratio:.1%}")
    lines.append(f"欠曝区域: {report.underexposed_ratio:.1%}")
    lines.append(f"有效内容: {report.effective_content_ratio:.1%}")
    lines.append(f"颜色有效: {'是' if report.color_validity else '否'}")

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
