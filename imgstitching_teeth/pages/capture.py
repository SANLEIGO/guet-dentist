"""拍照采集页面 — 引导用户拍摄上下牙弓照片用于 COLMAP 3D 重建。"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dental_stitcher_v1.camera_manager import CameraManager
from dental_stitcher_v1.io_utils import ImagePacket, bgr_to_rgb, resize_for_display
from dental_stitcher_v1.photo_quality import assess_photo_quality, format_quality_report

# 页面设置（独立配置）
st.set_page_config(
    page_title="拍照采集",
    page_icon="📷",
    layout="wide",
)

# ── Session State 初始化 ──────────────────────────────────────────

def _init_session_state() -> None:
    defaults = {
        "camera_manager": None,
        "camera_device_index": 0,
        "camera_opened": False,
        # 采集阶段
        "capture_phase": "idle",  # "idle" | "lower_arch" | "upper_arch" | "done"
        # 已拍照片缓存
        "lower_arch_images": [],
        "upper_arch_images": [],
        "current_quality_report": None,
        "last_captured_image": None,
        "supported_resolutions": [],  # 缓存支持的分辨率列表
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


_init_session_state()


# ── 摄像头管理 ──────────────────────────────────────────

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
    return ok


def _close_camera() -> None:
    cam = _get_camera_manager()
    cam.close()
    st.session_state.camera_opened = False


# ── 照片管理 ──────────────────────────────────────────

def _capture_photo() -> Optional[ImagePacket]:
    cam = _get_camera_manager()
    if not cam.is_opened():
        return None
    frame = cam.capture()
    if frame is None:
        return None
    phase = st.session_state.capture_phase
    arch = "lower" if phase == "lower_arch" else "upper" if phase == "upper_arch" else None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{arch}_{timestamp}.jpg" if arch else f"photo_{timestamp}.jpg"
    return ImagePacket(image=frame, name=name, timestamp=timestamp, arch=arch)


def _save_photo(packet: ImagePacket) -> None:
    arch = packet.arch
    if arch == "lower":
        st.session_state.lower_arch_images.append(packet)
    elif arch == "upper":
        st.session_state.upper_arch_images.append(packet)


def _clear_current_arch() -> None:
    phase = st.session_state.capture_phase
    if phase == "lower_arch":
        st.session_state.lower_arch_images = []
    elif phase == "upper_arch":
        st.session_state.upper_arch_images = []


def _complete_current_arch() -> None:
    if st.session_state.capture_phase == "lower_arch":
        st.session_state.capture_phase = "upper_arch"
    elif st.session_state.capture_phase == "upper_arch":
        st.session_state.capture_phase = "done"
    st.session_state.current_quality_report = None
    st.session_state.last_captured_image = None


# ── UI ──────────────────────────────────────────

def _render_header_and_progress():
    st.title("3D 重建照片采集")
    st.markdown("引导拍摄上下牙弓照片，用于 COLMAP 三维重建")

    # 阶段指示器
    phases = ["准备开始", "下牙弓采集", "上牙弓采集", "完成"]
    current_idx_map = {"idle": 0, "lower_arch": 1, "upper_arch": 2, "done": 3}
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
        help="刷新后选择你的 USB 内窥镜设备"
    )

    if st.session_state.camera_device_selector != st.session_state.camera_device_index:
        st.session_state.camera_device_index = st.session_state.camera_device_selector
        st.rerun()

    if st.session_state.camera_opened:
        st.success(f"已连接设备 {st.session_state.camera_device_index}")

        # 显示当前分辨率
        current_w, current_h = cam.frame_size
        st.info(f"当前分辨率: {current_w}x{current_h}")

        # 检测支持分辨率的按钮（避免自动调用干扰）
        if st.button("检测支持的分辨率", key="check_res_btn"):
            supported = cam.get_supported_resolutions()
            if supported:
                st.session_state.supported_resolutions = supported
            else:
                st.warning("无法枚举支持的分辨率")

        # 显示缓存的支持分辨率
        if "supported_resolutions" in st.session_state and st.session_state.supported_resolutions:
            st.markdown("**支持的分辨率:**")
            for w, h in st.session_state.supported_resolutions:
                st.markdown(f"- {w}x{h}")

        if st.button("断开摄像头", key="disconnect_btn"):
            _close_camera()
            if "supported_resolutions" in st.session_state:
                del st.session_state.supported_resolutions
            st.rerun()
    else:
        # 分辨率选择（打开前）
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
            help="选择摄像头分辨率，高分辨率更适合 3D 重建。注意：摄像头可能不支持指定分辨率。"
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
                - 在macOS System Preferences → Privacy & Security → Camera 中授权终端
                - 在 `streamlit run` 前用系统相机确认摄像头可以工作
                """)


def _render_guide_text(phase: str) -> None:
    guides = {
        "idle": """
- **开始前准备：**\n
  - 确保拍摄环境光线充足但避免强反光\n
  - 患者保持口腔张开姿势\n
  - 内窥镜镜头保持清洁\n
- **拍摄建议：**\n
  - 建议先拍下牙弓（15~25 张），再拍上牙弓\n
  - 每张照片保持 60% 以上重叠\n
  - 从左到右或从右到左缓慢移动\n
  - 保持镜头与牙齿表面 1~2 cm 距离\n
        """,
        "lower_arch": """
- **下牙弓采集：**\n
  - 镜头从正前方略偏下角度拍摄下排牙齿\n
  - 从左侧磨牙开始，向右缓慢移动\n
  - 每拍一张稍作停顿，观察预览画面质量\n
  - 保持相邻照片有 60% 以上重叠区域\n
  - 建议拍摄 15~25 张覆盖完整下牙弓弧度
        """,
        "upper_arch": """
- **上牙弓采集：**\n
  - 镜头从正前方略偏上角度拍摄上排牙齿\n
  - 从右侧磨牙开始，向左缓慢移动\n
  - 同样保持重叠和拍摄速度\n
  - 建议拍摄 15~25 张覆盖完整上牙弓弧度
        """,
        "done": """
**采集完成！**\n\n
- 下牙弓：{len_lower} 张\n
- 上牙弓：{len_upper} 张\n
- 请检查已拍照片质量，必要时返回重拍\n
- 点击「导出到拼接页面」进行后续处理
        """.format(
            len_lower=len(st.session_state.lower_arch_images),
            len_upper=len(st.session_state.upper_arch_images),
        ),
    }
    st.markdown(guides.get(phase, ""))


def _render_preview_and_capture():
    cam = _get_camera_manager()
    phase = st.session_state.capture_phase

    # 拍照按钮在主区域
    if phase in ("lower_arch", "upper_arch"):
        message = {
            "lower_arch": "拍摄下牙弓",
            "upper_arch": "拍摄上牙弓",
        }.get(phase, "")

        # 实时视频预览区域
        if st.session_state.camera_opened:
            # 读取并显示当前帧
            frame = cam.read_frame()
            if frame is not None:
                st.image(
                    bgr_to_rgb(resize_for_display(frame, max_width=800)),
                    caption="实时预览",
                    use_container_width=True
                )
            else:
                st.warning("无法读取摄像头画面")
        else:
            st.info("👆 请先在侧边栏打开摄像头")

        st.divider()

        # 拍照按钮
        cols_btn = st.columns([3, 1])
        with cols_btn[0]:
            if st.session_state.camera_opened:
                if st.button(f"📸 {message}", key="capture_btn", type="primary", width="stretch"):
                    packet = _capture_photo()
                    if packet is not None:
                        quality = assess_photo_quality(packet.image)
                        st.session_state.current_quality_report = quality
                        st.session_state.last_captured_image = packet
                        st.toast("拍照完成，请检查质量", icon="📸")
                    else:
                        st.error("拍照失败，请检查摄像头连接")
            else:
                st.button(f"📸 {message}", key="capture_btn_disabled", disabled=True, width="stretch")

        with cols_btn[1]:
            if st.session_state.camera_opened:
                # 手动刷新按钮
                if st.button("🔄 刷新", key="refresh_btn"):
                    st.rerun()


def _render_quality_feedback():
    quality = st.session_state.current_quality_report
    packet = st.session_state.last_captured_image

    if quality is None or packet is None:
        return

    st.divider()
    st.subheader("最近一张质量检测")

    cols = st.columns([1, 2])
    with cols[0]:
        st.image(bgr_to_rgb(resize_for_display(packet.image)), caption="刚拍的照片")

    with cols[1]:
        if quality.passed:
            st.success("质量检测通过")
        else:
            st.warning("质量检测未通过")
            for reason in quality.fail_reasons:
                st.markdown(f"❌ {reason}")

        if quality.warn_reasons:
            for reason in quality.warn_reasons:
                st.markdown(f"⚠️ {reason}")

        with st.expander("详细报告"):
            st.text(format_quality_report(quality))

        # 决定保留或丢弃
        cols_action = st.columns(2)
        with cols_action[0]:
            if st.button("✅ 保留这张", key="keep_photo_btn", type="primary", disabled=quality.passed == False):
                _save_photo(packet)
                st.session_state.current_quality_report = None
                st.session_state.last_captured_image = None
                st.toast("已保存", icon="✅")
                st.rerun()

        with cols_action[1]:
            if st.button("🗑️ 丢弃重拍", key="discard_photo_btn"):
                st.session_state.current_quality_report = None
                st.session_state.last_captured_image = None
                st.toast("已丢弃", icon="🗑️")
                st.rerun()


def _render_thumbnails_and_controls():
    phase = st.session_state.capture_phase
    lower_images = st.session_state.lower_arch_images
    upper_images = st.session_state.upper_arch_images

    current_images = lower_images if phase == "lower_arch" else upper_images if phase == "upper_arch" else []
    current_label = "下牙弓" if phase == "lower_arch" else "上牙弓" if phase == "upper_arch" else ""

    if current_images:
        st.divider()
        st.subheader(f"{current_label}已拍照片 ({len(current_images)} 张)")

        cols_per_row = 5
        for i in range(0, len(current_images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(current_images):
                    with col:
                        st.image(
                            bgr_to_rgb(resize_for_display(current_images[idx].image, max_width=150, max_height=150)),
                            caption=f"#{idx + 1}",
                        )
                        if st.button("删除", key=f"delete_{phase}_{idx}"):
                            if phase == "lower_arch":
                                st.session_state.lower_arch_images.pop(idx)
                            elif phase == "upper_arch":
                                st.session_state.upper_arch_images.pop(idx)
                            st.rerun()

    # 阶段控制按钮
    if phase != "idle" and phase != "done":
        st.divider()
        cols_ctrl = st.columns([1, 1, 2])
        with cols_ctrl[0]:
            if st.button("🗑️ 清空重拍", key="clear_current_btn"):
                _clear_current_arch()
                st.toast("已清空当前牙弓照片", icon="🗑️")
                st.rerun()

        with cols_ctrl[1]:
            min_required = 10
            if st.button(f"✅ 完成当前牙弓 ({len(current_images)}/{min_required})", key="complete_arch_btn", disabled=len(current_images) < min_required):
                _complete_current_arch()
                st.rerun()
        with cols_ctrl[2]:
            if len(current_images) < min_required:
                st.info(f"至少需要 {min_required} 张照片才能完成当前牙弓")


def _render_phase_controls():
    phase = st.session_state.capture_phase

    if phase == "idle":
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🦷 开始下牙弓采集", key="start_lower_btn", type="primary", width="stretch"):
                st.session_state.capture_phase = "lower_arch"
                st.rerun()
        with col2:
            if st.button("🦷 开始上牙弓采集", key="start_upper_btn", width="stretch"):
                st.session_state.capture_phase = "upper_arch"
                st.rerun()

    elif phase == "done":
        st.divider()
        st.subheader("采集完成")
        st.success(f"""
        - 下牙弓：{len(st.session_state.lower_arch_images)} 张
        - 上牙弓：{len(st.session_state.upper_arch_images)} 张
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            # 导出 ZIP
            if st.button("📦 下载所有照片 (ZIP)", key="download_zip_btn", width="stretch"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for packet in st.session_state.lower_arch_images:
                        success, encoded = cv2.imencode(".jpg", packet.image)
                        if success:
                            zf.writestr(f"lower/{packet.name}", encoded.tobytes())
                    for packet in st.session_state.upper_arch_images:
                        success, encoded = cv2.imencode(".jpg", packet.image)
                        if success:
                            zf.writestr(f"upper/{packet.name}", encoded.tobytes())
                zip_buffer.seek(0)
                st.download_button(
                    label="⬇️ 点击下载 ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"dental_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                )

        with col2:
            if st.button("🔄 重新开始", key="restart_btn", width="stretch"):
                st.session_state.lower_arch_images = []
                st.session_state.upper_arch_images = []
                st.session_state.capture_phase = "idle"
                st.rerun()

        with col3:
            st.info("前往首页「图像拼接」使用采集的照片")


# ── 主函数 ──────────────────────────────────────────

def main() -> None:
    _init_session_state()
    _render_header_and_progress()

    # 自动刷新逻辑：只在摄像头打开且没有待处理照片时刷新
    if (st.session_state.camera_opened and
        st.session_state.capture_phase in ("lower_arch", "upper_arch") and
        st.session_state.last_captured_image is None):
        st_autorefresh(interval=300, key="preview_autorefresh")  # 300ms ≈ 3fps

    # 侧边栏
    with st.sidebar:
        st.title("📷 拍照设置")
        _render_camera_sidebar()
        st.divider()
        _render_guide_text(st.session_state.capture_phase)

    # 主区域
    _render_preview_and_capture()
    _render_quality_feedback()
    _render_thumbnails_and_controls()
    _render_phase_controls()


if __name__ == "__main__":
    main()
