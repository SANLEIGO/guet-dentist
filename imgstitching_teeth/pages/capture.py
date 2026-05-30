"""拍照采集页面 — 全自动引导上下牙弓采集。"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime
from typing import Optional

import cv2
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dental_stitcher_v1.auto_capture import (
    AUTO_CAPTURE_MIN_INTERVAL_S,
    MIN_ACCEPTED_IMAGES,
    TARGET_ACCEPTED_IMAGES,
    ArchProgress,
    AutoCaptureAssessment,
    evaluate_auto_capture_frame,
    summarize_arch_progress,
)
from dental_stitcher_v1.camera_manager import CameraManager
from dental_stitcher_v1.io_utils import ImagePacket, bgr_to_rgb, resize_for_display
from dental_stitcher_v1.photo_quality import format_quality_report

st.set_page_config(
    page_title="拍照采集",
    page_icon="📷",
    layout="wide",
)


def _init_session_state() -> None:
    defaults = {
        "camera_manager": None,
        "camera_device_index": 0,
        "camera_opened": False,
        "capture_phase": "idle",  # "idle" | "upper_arch" | "lower_arch" | "review"
        "lower_arch_images": [],
        "upper_arch_images": [],
        "current_quality_report": None,
        "last_captured_image": None,
        "supported_resolutions": [],
        "live_preview_frame": None,
        "previous_preview_frame": None,
        "live_auto_assessment": None,
        "last_saved_packet": None,
        "last_saved_assessment": None,
        "last_capture_status": "等待开始自动采集",
        "last_auto_capture_at": 0.0,
        "upper_last_saved_frame": None,
        "lower_last_saved_frame": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


_init_session_state()


def _get_camera_manager() -> CameraManager:
    if st.session_state.camera_manager is None:
        st.session_state.camera_manager = CameraManager()
    return st.session_state.camera_manager


def _open_camera(device_index: int, width: int = 1920, height: int = 1080) -> bool:
    cam = _get_camera_manager()
    if cam.is_opened() and cam.device_index == device_index:
        return True
    ok = cam.open(device_index, width, height)
    st.session_state.camera_opened = ok
    st.session_state.camera_device_index = device_index
    if ok:
        _clear_live_cycle_state()
    return ok


def _close_camera() -> None:
    cam = _get_camera_manager()
    cam.close()
    st.session_state.camera_opened = False
    _clear_live_cycle_state()


def _clear_live_cycle_state() -> None:
    st.session_state.live_preview_frame = None
    st.session_state.previous_preview_frame = None
    st.session_state.live_auto_assessment = None
    st.session_state.current_quality_report = None
    st.session_state.last_auto_capture_at = 0.0


def _get_arch_key_for_phase(phase: Optional[str] = None) -> Optional[str]:
    phase = phase or st.session_state.capture_phase
    if phase == "upper_arch":
        return "upper"
    if phase == "lower_arch":
        return "lower"
    return None


def _get_arch_label_for_phase(phase: Optional[str] = None) -> str:
    phase = phase or st.session_state.capture_phase
    if phase == "upper_arch":
        return "上牙弓"
    if phase == "lower_arch":
        return "下牙弓"
    return ""


def _get_images_for_phase(phase: Optional[str] = None) -> list[ImagePacket]:
    phase = phase or st.session_state.capture_phase
    if phase == "upper_arch":
        return st.session_state.upper_arch_images
    if phase == "lower_arch":
        return st.session_state.lower_arch_images
    return []


def _get_arch_progress(phase: Optional[str] = None) -> Optional[ArchProgress]:
    phase = phase or st.session_state.capture_phase
    if phase not in ("upper_arch", "lower_arch"):
        return None
    return summarize_arch_progress(_get_images_for_phase(phase), phase)


def _set_last_saved_frame(arch_key: str, frame: Optional[object]) -> None:
    st.session_state[f"{arch_key}_last_saved_frame"] = None if frame is None else frame.copy()


def _get_last_saved_frame(arch_key: str) -> Optional[object]:
    return st.session_state.get(f"{arch_key}_last_saved_frame")


def _has_captured_images() -> bool:
    return bool(st.session_state.upper_arch_images or st.session_state.lower_arch_images)


def _build_capture_zip_bytes() -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for packet in st.session_state.upper_arch_images:
            success, encoded = cv2.imencode(".jpg", packet.image)
            if success:
                zf.writestr(f"upper/{packet.name}", encoded.tobytes())
        for packet in st.session_state.lower_arch_images:
            success, encoded = cv2.imencode(".jpg", packet.image)
            if success:
                zf.writestr(f"lower/{packet.name}", encoded.tobytes())
    return zip_buffer.getvalue()


def _render_capture_download_panel(*, key_prefix: str, title: str = "下载当前已拍照片") -> None:
    st.markdown(f"#### {title}")
    upper_count = len(st.session_state.upper_arch_images)
    lower_count = len(st.session_state.lower_arch_images)
    st.caption(f"ZIP 内将按 `upper/`、`lower/` 目录分别打包。当前：上牙弓 {upper_count} 张，下牙弓 {lower_count} 张。")

    if not _has_captured_images():
        st.info("还没有可下载的照片，开始采集后这里会直接提供 ZIP 下载。")
        return

    st.download_button(
        label="📦 下载已拍照片 (ZIP)",
        data=_build_capture_zip_bytes(),
        file_name=f"dental_capture_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        key=f"{key_prefix}_capture_zip_download",
        width="stretch",
    )


def _build_packet_from_frame(frame, assessment: AutoCaptureAssessment) -> ImagePacket:
    phase = st.session_state.capture_phase
    arch = _get_arch_key_for_phase(phase)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{arch}_{timestamp}.jpg" if arch else f"photo_{timestamp}.jpg"
    return ImagePacket(
        image=frame.copy(),
        name=name,
        timestamp=timestamp,
        arch=arch,
        acceptance_score=assessment.acceptability_score,
        quality_passed=assessment.quality_report.passed,
        meta={
            "acceptability_label": assessment.quality_report.acceptability_label,
            "framing_score": assessment.quality_report.framing_score,
            "step_label": assessment.current_step_label,
            "predicted_region": assessment.region_assessment.predicted_region,
            "expected_region": assessment.expected_region,
            "region_confidence": assessment.region_assessment.confidence,
        },
    )


def _save_photo(packet: ImagePacket) -> None:
    if packet.arch == "upper":
        st.session_state.upper_arch_images.append(packet)
    elif packet.arch == "lower":
        st.session_state.lower_arch_images.append(packet)


def _sync_last_saved_frame_from_images(phase: Optional[str] = None) -> None:
    phase = phase or st.session_state.capture_phase
    arch_key = _get_arch_key_for_phase(phase)
    if arch_key is None:
        return
    current_images = _get_images_for_phase(phase)
    if current_images:
        _set_last_saved_frame(arch_key, current_images[-1].image)
    else:
        _set_last_saved_frame(arch_key, None)


def _clear_current_arch() -> None:
    phase = st.session_state.capture_phase
    if phase == "upper_arch":
        st.session_state.upper_arch_images = []
        _set_last_saved_frame("upper", None)
    elif phase == "lower_arch":
        st.session_state.lower_arch_images = []
        _set_last_saved_frame("lower", None)
    _clear_live_cycle_state()


def _switch_capture_phase(new_phase: str, status_text: str) -> None:
    st.session_state.capture_phase = new_phase
    st.session_state.last_capture_status = status_text
    _clear_live_cycle_state()


def _restart_all_capture() -> None:
    st.session_state.lower_arch_images = []
    st.session_state.upper_arch_images = []
    st.session_state.last_saved_packet = None
    st.session_state.last_saved_assessment = None
    _set_last_saved_frame("upper", None)
    _set_last_saved_frame("lower", None)
    _switch_capture_phase("idle", "已重新开始自动采集流程")


def _advance_phase_if_ready() -> bool:
    phase = st.session_state.capture_phase
    progress = _get_arch_progress(phase)
    if progress is None or not progress.recommended_met:
        return False

    if phase == "upper_arch":
        next_phase = "review" if len(st.session_state.lower_arch_images) >= TARGET_ACCEPTED_IMAGES else "lower_arch"
        if next_phase == "lower_arch":
            _switch_capture_phase(
                "lower_arch",
                f"上牙弓已自动达标（{progress.accepted_count}/{TARGET_ACCEPTED_IMAGES}），请把镜头移到下牙弓。"
            )
            st.toast("上牙弓已完成，开始下牙弓自动采集", icon="✅")
        else:
            _switch_capture_phase(
                "review",
                f"上牙弓已自动达标（{progress.accepted_count}/{TARGET_ACCEPTED_IMAGES}），全部采集完成。"
            )
            st.toast("上下牙弓都已完成采集", icon="✅")
        return True

    if phase == "lower_arch":
        _switch_capture_phase(
            "review",
            f"下牙弓已自动达标（{progress.accepted_count}/{TARGET_ACCEPTED_IMAGES}），可进入后续处理。"
        )
        st.toast("下牙弓已完成，采集流程结束", icon="✅")
        return True

    return False


def _process_live_capture_cycle() -> bool:
    phase = st.session_state.capture_phase
    if not st.session_state.camera_opened or phase not in ("upper_arch", "lower_arch"):
        return False

    cam = _get_camera_manager()
    frame = cam.capture()
    if frame is None:
        st.session_state.live_preview_frame = None
        st.session_state.live_auto_assessment = None
        st.session_state.current_quality_report = None
        st.session_state.last_capture_status = "无法读取摄像头画面，请检查设备连接。"
        return False

    arch_key = _get_arch_key_for_phase(phase)
    current_images = _get_images_for_phase(phase)
    previous_preview = st.session_state.previous_preview_frame
    last_saved_frame = _get_last_saved_frame(arch_key) if arch_key else None

    last_capture_at = st.session_state.last_auto_capture_at
    if last_capture_at > 0:
        seconds_since_last_capture = time.monotonic() - last_capture_at
    else:
        seconds_since_last_capture = AUTO_CAPTURE_MIN_INTERVAL_S

    assessment = evaluate_auto_capture_frame(
        image=frame,
        phase=phase,
        accepted_count=len(current_images),
        previous_preview=previous_preview,
        last_saved_image=last_saved_frame,
        seconds_since_last_capture=seconds_since_last_capture,
    )

    st.session_state.live_preview_frame = frame
    st.session_state.previous_preview_frame = frame.copy()
    st.session_state.live_auto_assessment = assessment
    st.session_state.current_quality_report = assessment.quality_report

    if not assessment.should_capture:
        st.session_state.last_capture_status = assessment.status_text
        return False

    packet = _build_packet_from_frame(frame, assessment)
    _save_photo(packet)
    st.session_state.last_captured_image = packet
    st.session_state.last_saved_packet = packet
    st.session_state.last_saved_assessment = assessment
    st.session_state.last_auto_capture_at = time.monotonic()
    if arch_key is not None:
        _set_last_saved_frame(arch_key, frame)

    saved_count = len(_get_images_for_phase(phase))
    st.session_state.last_capture_status = (
        f"已自动保存{_get_arch_label_for_phase(phase)}第 {saved_count} 张，继续按提示缓慢移动。"
    )
    return _advance_phase_if_ready()


def _render_header_and_progress() -> None:
    st.title("3D 重建照片采集")
    st.markdown("全自动采集模式：系统会自动筛选并保存合格照片，医生只需要按提示移动镜头。")

    phases = ["准备开始", "上牙弓自动采集", "下牙弓自动采集", "检查并进入后续"]
    current_idx_map = {"idle": 0, "upper_arch": 1, "lower_arch": 2, "review": 3}
    current_idx = current_idx_map.get(st.session_state.capture_phase, 0)
    st.progress(current_idx / 3)

    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            if i == current_idx:
                st.markdown(f"**{phases[i]}**")
            else:
                st.markdown(f"~~{phases[i]}~~")


def _render_camera_sidebar() -> None:
    st.subheader("摄像头")
    cam = _get_camera_manager()
    devices = CameraManager.list_available_devices(max_check=5)

    selected = st.selectbox(
        "选择设备",
        devices if devices else [0],
        index=0 if not devices else devices.index(st.session_state.camera_device_index) if st.session_state.camera_device_index in devices else 0,
        key="camera_device_selector",
        help="刷新后选择你的 USB 内窥镜设备",
    )

    if selected != st.session_state.camera_device_index:
        st.session_state.camera_device_index = selected
        st.rerun()

    if st.session_state.camera_opened:
        st.success(f"已连接设备 {st.session_state.camera_device_index}")
        current_w, current_h = cam.frame_size
        st.info(f"当前分辨率: {current_w}x{current_h}")

        if st.button("检测支持的分辨率", key="check_res_btn"):
            supported = cam.get_supported_resolutions()
            if supported:
                st.session_state.supported_resolutions = supported
            else:
                st.warning("无法枚举支持的分辨率")

        if st.session_state.supported_resolutions:
            st.markdown("**支持的分辨率:**")
            for w, h in st.session_state.supported_resolutions:
                st.markdown(f"- {w}x{h}")

        if st.button("断开摄像头", key="disconnect_btn"):
            _close_camera()
            st.session_state.supported_resolutions = []
            st.rerun()
    else:
        resolution_options = {
            "1920x1080 (1080p)": (1920, 1080),
            "1280x720 (720p)": (1280, 720),
            "640x480 (VGA)": (640, 480),
        }
        resolution_label = st.selectbox(
            "目标分辨率",
            list(resolution_options.keys()),
            index=0,
            key="resolution_selector",
            help="优先使用高分辨率，自动筛选更稳定。",
        )
        target_width, target_height = resolution_options[resolution_label]

        if st.button("打开摄像头", key="connect_btn", type="primary"):
            if _open_camera(st.session_state.camera_device_index, target_width, target_height):
                st.toast("摄像头已连接", icon="📷")
                st.rerun()
            else:
                st.error("无法打开摄像头，请检查设备连接")
                st.markdown("""
                **故障排查：**
                - 确认 USB 设备已插入（如接了 USB 集线器请排除）
                - 在 macOS System Preferences → Privacy & Security → Camera 中授权终端
                - 在 `streamlit run` 前用系统相机确认摄像头可以工作
                """)


def _render_guide_text(phase: str) -> None:
    guides = {
        "idle": f"""
- 系统会先采集 **上牙弓**，达标后自动切换到 **下牙弓**
- 推荐每个牙弓自动保留 **{TARGET_ACCEPTED_IMAGES} 张**，最低可用 **{MIN_ACCEPTED_IMAGES} 张**
- 医生只要按牙弓方向缓慢移动镜头，不需要点击“拍照/保留”
- 相邻画面请保持 **60% 以上重叠**
- 尽量让牙列位于画面中央，镜头与牙面距离约 **1~2 cm**
        """,
        "upper_arch": """
- 从**左后牙**开始，沿上牙弓连续扫向右后牙
- 画面稳定、清晰、曝光合适且和上一张有明显位移时，系统会自动保存
- 如果提示“太像了”，说明当前位置重复，请继续沿牙弓缓慢移动
- 如果提示“先停稳”，说明当前速度偏快，请短暂停住半秒
        """,
        "lower_arch": """
- 现在切换到**下牙弓**，仍然从左后牙连续扫向右后牙
- 镜头可略偏下方，尽量看到连续牙弓轮廓
- 后牙区至少要留下两张清晰照片，系统会自动补足推荐数量
        """,
        "review": """
- 自动采集已结束，可检查照片数量并进入后续处理
- 如果某个牙弓还想补拍，可在下方重新进入对应牙弓
- 只要任意牙弓的照片足够合格，就可以先进入后续拼接/重建流程
        """,
    }
    st.markdown(guides.get(phase, ""))


def _render_live_preview_and_status() -> None:
    phase = st.session_state.capture_phase
    assessment: Optional[AutoCaptureAssessment] = st.session_state.live_auto_assessment
    progress = _get_arch_progress(phase)
    region_labels = {"left": "左侧段", "center": "前牙段", "right": "右侧段", "unknown": "未知区域"}

    preview_col, status_col = st.columns([3, 2])

    with preview_col:
        st.subheader("实时预览")
        frame = st.session_state.live_preview_frame
        if phase in ("upper_arch", "lower_arch"):
            if frame is not None:
                st.image(
                    bgr_to_rgb(resize_for_display(frame, max_width=900)),
                    caption="系统正在自动分析当前画面",
                    use_container_width=True,
                )
            elif st.session_state.camera_opened:
                st.warning("正在等待摄像头画面…")
            else:
                st.info("请先在侧边栏打开摄像头")
        else:
            if st.session_state.camera_opened:
                preview = _get_camera_manager().read_frame()
                if preview is not None:
                    st.image(
                        bgr_to_rgb(resize_for_display(preview, max_width=900)),
                        caption="当前摄像头预览",
                        use_container_width=True,
                    )
                else:
                    st.warning("无法读取摄像头画面")
            else:
                st.info("请先在侧边栏打开摄像头")

    with status_col:
        st.subheader("自动采集状态")
        st.info(st.session_state.last_capture_status)

        if phase in ("upper_arch", "lower_arch") and assessment is not None and progress is not None:
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("当前画面可接受度", f"{assessment.acceptability_score * 100:.0f}%")
                st.metric("稳定度", f"{assessment.stability_score * 100:.0f}%")
            with metric_col2:
                st.metric("自动保存准备度", f"{assessment.capture_readiness * 100:.0f}%")
                st.metric("新鲜度", f"{assessment.novelty_score * 100:.0f}%")
                st.metric("区域识别置信度", f"{assessment.region_assessment.confidence * 100:.0f}%")

            st.markdown(f"**{progress.arch_label}完成度：{progress.completion_score * 100:.0f}%**")
            st.progress(progress.completion_score)
            st.caption(
                f"已保存 {progress.accepted_count}/{progress.recommended_target} 张，"
                f"最低可用 {progress.minimum_required} 张，平均可接受度 {progress.average_acceptance * 100:.0f}%"
            )

            if assessment.status_level == "success":
                st.success(assessment.status_text)
            elif assessment.status_level == "warning":
                st.warning(assessment.status_text)
            else:
                st.info(assessment.status_text)

            st.markdown(f"**当前引导：{assessment.current_step_label}**")
            for message in assessment.highlight_messages:
                st.markdown(f"- {message}")

            region_match_text = "一致" if assessment.region_matched else "不一致"
            st.caption(
                f"区域识别：当前 {assessment.region_assessment.status_text}；"
                f"当前建议扫描 {region_labels.get(assessment.expected_region, assessment.expected_region)}；"
                f"参考匹配状态 {region_match_text}"
            )

            if assessment.blocking_reasons:
                st.markdown("**当前需调整：**")
                for reason in assessment.blocking_reasons[:4]:
                    st.markdown(f"- {reason}")

            with st.expander("查看当前帧质量详细报告"):
                st.text(format_quality_report(assessment.quality_report))
        elif phase == "idle":
            st.markdown("**流程说明**")
            st.markdown("- 点击“开始自动采集流程”后，系统会先引导上牙弓，再自动切到下牙弓。")
            st.markdown("- 采集期间不需要再点击拍照或确认保留。")
        elif phase == "review":
            upper_progress = summarize_arch_progress(st.session_state.upper_arch_images, "upper_arch")
            lower_progress = summarize_arch_progress(st.session_state.lower_arch_images, "lower_arch")
            st.metric("上牙弓完成度", f"{upper_progress.completion_score * 100:.0f}%")
            st.metric("下牙弓完成度", f"{lower_progress.completion_score * 100:.0f}%")
            st.caption("如需补拍，可重新进入任意牙弓自动采集。")


def _render_recent_capture_feedback() -> None:
    packet = st.session_state.last_saved_packet
    assessment: Optional[AutoCaptureAssessment] = st.session_state.last_saved_assessment
    if packet is None or assessment is None:
        return
    packet_arch_label = "上牙弓" if getattr(packet, "arch", None) == "upper" else "下牙弓"
    region_labels = {"left": "左侧段", "center": "前牙段", "right": "右侧段", "unknown": "未知区域"}

    st.divider()
    st.subheader("最近自动保存")

    cols = st.columns([1, 2])
    with cols[0]:
        st.image(
            bgr_to_rgb(resize_for_display(packet.image)),
            caption=f"{packet_arch_label} 最近一张",
        )

    with cols[1]:
        st.success(
            f"已自动保存 {packet.name} | 可接受度 {assessment.acceptability_score * 100:.0f}% | "
            f"步骤 {assessment.current_step_label} | 区域 {region_labels.get(assessment.region_assessment.predicted_region, assessment.region_assessment.predicted_region)}"
        )
        if assessment.quality_report.warn_reasons:
            for reason in assessment.quality_report.warn_reasons[:3]:
                st.markdown(f"- {reason}")
        st.caption("这张照片已经直接进入当前牙弓数据集，无需再手动确认。")
        with st.expander("查看已保存照片的质量报告"):
            st.text(format_quality_report(assessment.quality_report))


def _render_thumbnails_and_controls() -> None:
    phase = st.session_state.capture_phase
    if phase not in ("upper_arch", "lower_arch"):
        return

    current_images = _get_images_for_phase(phase)
    current_label = _get_arch_label_for_phase(phase)
    progress = _get_arch_progress(phase)

    if current_images:
        st.divider()
        st.subheader(f"{current_label}已自动保存照片 ({len(current_images)} 张)")

        cols_per_row = 5
        for i in range(0, len(current_images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(current_images):
                    continue
                packet = current_images[idx]
                acceptance_score = float(getattr(packet, "acceptance_score", 0.0) or 0.0)
                with col:
                    st.image(
                        bgr_to_rgb(resize_for_display(packet.image, max_width=150, max_height=150)),
                        caption=f"#{idx + 1} · {acceptance_score * 100:.0f}%",
                    )
                    if st.button("删除", key=f"delete_{phase}_{idx}"):
                        current_images.pop(idx)
                        _sync_last_saved_frame_from_images(phase)
                        _clear_live_cycle_state()
                        st.rerun()

    st.divider()
    control_col1, control_col2, control_col3 = st.columns([1, 1, 2])
    with control_col1:
        if st.button("🗑️ 清空当前牙弓", key="clear_current_btn"):
            _clear_current_arch()
            st.toast("已清空当前牙弓照片", icon="🗑️")
            st.rerun()

    with control_col2:
        next_label = "➡️ 提前切到下牙弓（兜底）" if phase == "upper_arch" else "➡️ 提前进入后续（兜底）"
        if st.button(next_label, key=f"fallback_next_{phase}", disabled=not progress.minimum_met):
            if phase == "upper_arch":
                _switch_capture_phase(
                    "lower_arch",
                    f"上牙弓已达到最低可用量（{progress.accepted_count}/{MIN_ACCEPTED_IMAGES}），已切换到下牙弓。"
                )
            else:
                _switch_capture_phase(
                    "review",
                    f"下牙弓已达到最低可用量（{progress.accepted_count}/{MIN_ACCEPTED_IMAGES}），可进入后续处理。"
                )
            st.rerun()

    with control_col3:
        if progress.recommended_met:
            st.success("当前牙弓已达到推荐采集量，系统会自动切换到下一阶段。")
        elif progress.minimum_met:
            st.info(
                f"当前已达到最低可用量（{progress.accepted_count}/{progress.minimum_required}），"
                f"系统会继续补足到推荐的 {progress.recommended_target} 张。"
            )
        else:
            st.info(
                f"当前已保存 {progress.accepted_count} 张，至少需要 {progress.minimum_required} 张，"
                f"推荐保存 {progress.recommended_target} 张。"
            )


def _render_phase_controls() -> None:
    phase = st.session_state.capture_phase

    if phase == "idle":
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🦷 开始自动采集流程", key="start_upper_btn", type="primary", width="stretch"):
                _switch_capture_phase("upper_arch", "上牙弓自动采集已开始，请从左后牙开始缓慢移动镜头。")
                st.rerun()
        with col2:
            st.info("系统会先完成上牙弓，再自动切换到下牙弓。整个采集过程不再需要点击拍照按钮。")
        return

    if phase != "review":
        return

    st.divider()
    st.subheader("当前可进入后续步骤")

    upper_progress = summarize_arch_progress(st.session_state.upper_arch_images, "upper_arch")
    lower_progress = summarize_arch_progress(st.session_state.lower_arch_images, "lower_arch")

    if upper_progress.minimum_met or lower_progress.minimum_met:
        done_text = []
        if upper_progress.minimum_met:
            done_text.append(f"上牙弓 {upper_progress.accepted_count} 张")
        if lower_progress.minimum_met:
            done_text.append(f"下牙弓 {lower_progress.accepted_count} 张")
        st.success("已完成：" + "，".join(done_text))
    else:
        st.warning("当前尚未达到最低可用张数，请继续补拍。")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🦷 补拍上牙弓", key="resume_upper_btn", width="stretch"):
            _switch_capture_phase("upper_arch", "已返回上牙弓自动采集，请继续从未覆盖区域补拍。")
            st.rerun()
    with col2:
        if st.button("🦷 补拍下牙弓", key="resume_lower_btn", width="stretch"):
            _switch_capture_phase("lower_arch", "已返回下牙弓自动采集，请继续从未覆盖区域补拍。")
            st.rerun()
    with col3:
        st.page_link("app.py", label="➡️ 进入后续处理", icon="➡️")

    _render_capture_download_panel(key_prefix="review", title="导出当前已拍照片")
    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        st.info("可以直接下载当前全部已拍照片，ZIP 内按上下牙弓分目录保存。")
    with export_col2:
        if st.button("🔄 重新开始全部采集", key="restart_btn", width="stretch"):
            _restart_all_capture()
            st.rerun()
    with export_col3:
        st.info("如果只想先处理一个牙弓，也可以现在直接进入后续步骤。")


def main() -> None:
    _init_session_state()

    should_rerun = False
    if (
        st.session_state.camera_opened
        and st.session_state.capture_phase in ("upper_arch", "lower_arch")
    ):
        st_autorefresh(interval=450, key="preview_autorefresh")
        should_rerun = _process_live_capture_cycle()

    _render_header_and_progress()

    with st.sidebar:
        st.title("📷 拍照设置")
        _render_camera_sidebar()
        st.divider()
        _render_guide_text(st.session_state.capture_phase)
        st.divider()
        _render_capture_download_panel(key_prefix="sidebar", title="导出已拍照片")

    _render_live_preview_and_status()
    _render_recent_capture_feedback()
    _render_thumbnails_and_controls()
    _render_phase_controls()

    if should_rerun:
        st.rerun()


if __name__ == "__main__":
    main()
