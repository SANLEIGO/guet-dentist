from __future__ import annotations

import base64
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

from dental_stitcher_v1.io_utils import ImagePacket, bgr_to_rgb, load_uploaded_images, resize_for_display
from dental_stitcher_v1.pipeline import run_pipeline
from dental_stitcher_v1.segmentation import segment_teeth, fallback_full_mask
from dental_stitcher_v1.noise_removal import (
    remove_noise_from_stitched_result,
    get_noise_removal_method_config,
    format_noise_removal_stats
)
from dental_stitcher_v1.threed import (
    HunyuanServiceClient,
    HunyuanServiceConfig,
    HunyuanRuntimeManager,
    Prepared3DAsset,
    derive_mask_from_image,
    prepare_image_for_hunyuan3d,
)


st.set_page_config(page_title="口腔牙齿提取拼接 v1", page_icon="🦷", layout="wide")

HUNYUAN_RESULTS_DIR = Path(__file__).resolve().parent / ".runtime" / "hunyuan3d" / "results"


def _slugify_arch_label(label: str) -> str:
    mapping = {
        "上牙弓": "upper",
        "下牙弓": "lower",
        "左上": "upper_left",
        "右上": "upper_right",
        "左下": "lower_left",
        "右下": "lower_right",
        "未知牙弓": "unknown_arch",
    }
    return mapping.get(label, label.replace(" ", "_").lower())


def _encode_png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("无法将图像编码为 PNG。")
    return encoded.tobytes()


def _persist_hunyuan_result(job_uid: str, model_bytes: bytes, arch_slug: str = "unknown_arch") -> Path:
    HUNYUAN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = HUNYUAN_RESULTS_DIR / f"{arch_slug}_{job_uid}.glb"
    output_path.write_bytes(model_bytes)
    return output_path


def _default_hunyuan_steps(subfolder: str) -> int:
    return 5 if "turbo" in subfolder else 30


def _clean_status_text(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    cleaned = cleaned.replace("\r", " ").replace("\uFFFD", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


def _render_glb_preview(glb_bytes: bytes, height: int = 560) -> None:
    glb_b64 = base64.b64encode(glb_bytes).decode("utf-8")
    html = f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
    <script nomodule src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer-legacy.js"></script>
    <style>
      .viewer-shell {{
        width: 100%;
        height: {height}px;
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 18px;
        overflow: hidden;
        background:
          radial-gradient(circle at 20% 20%, rgba(214, 236, 245, 0.95), rgba(243, 246, 248, 0.96) 46%, rgba(234, 240, 244, 0.98) 100%);
      }}
      model-viewer {{
        width: 100%;
        height: 100%;
        background: transparent;
        --progress-bar-color: #2a768b;
        --poster-color: transparent;
      }}
    </style>
    <div class="viewer-shell">
      <model-viewer
        src="data:model/gltf-binary;base64,{glb_b64}"
        camera-controls
        auto-rotate
        shadow-intensity="1"
        exposure="1"
        environment-image="neutral"
        interaction-prompt="auto"
        ar-status="not-presenting">
      </model-viewer>
    </div>
    """
    components.html(html, height=height + 12)


def _clear_hunyuan_job_state() -> None:
    for key in (
        "hunyuan_job_uid",
        "hunyuan_job_status",
        "hunyuan_job_message",
        "hunyuan_job_result_bytes",
        "hunyuan_job_result_path",
    ):
        st.session_state.pop(key, None)


def _reset_hunyuan_generation_state(clear_prepared_asset: bool = False) -> None:
    if clear_prepared_asset:
        st.session_state.pop("prepared_3d_asset", None)
    st.session_state.pop("hunyuan_input_asset", None)
    st.session_state.pop("hunyuan_input_source_image", None)
    st.session_state.pop("hunyuan_input_source_name", None)
    st.session_state.pop("hunyuan_input_source_label", None)
    _clear_hunyuan_job_state()


def _uploaded_image_to_cv2(uploaded) -> np.ndarray:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图像文件：{uploaded.name}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError(f"不支持的图像通道格式：{uploaded.name}")
    return image


def _image_for_display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _build_manual_hunyuan_input_asset(uploaded, target_size: int = 512) -> tuple[Prepared3DAsset, np.ndarray]:
    image = _uploaded_image_to_cv2(uploaded)
    if image.ndim == 3 and image.shape[2] == 4:
        source_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        source_mask = image[:, :, 3]
    else:
        source_bgr = image
        source_mask = None
    prepared_asset = prepare_image_for_hunyuan3d(
        source_bgr,
        mask=source_mask,
        target_size=target_size,
        padding_ratio=0.12,
    )
    return prepared_asset, source_bgr


def _render_hunyuan_submission_panel(
    service_config: HunyuanServiceConfig,
    prepared_asset: Prepared3DAsset,
    region_label: str,
    source_label: str,
    source_name: str,
    source_image: np.ndarray,
) -> None:
    arch_slug = _slugify_arch_label(region_label)
    current_job_uid = st.session_state.get("hunyuan_job_uid")
    current_job_status = st.session_state.get("hunyuan_job_status")
    current_job_message = st.session_state.get("hunyuan_job_message")
    result_bytes = st.session_state.get("hunyuan_job_result_bytes")

    st.markdown("#### 当前单图输入")
    st.caption(f"当前来源: {source_label} | 文件: {source_name}")
    col_src, col_ready = st.columns(2)
    with col_src:
        st.markdown("#### 原始来源图")
        st.image(_image_for_display(resize_for_display(source_image)), width="stretch")
    with col_ready:
        st.markdown("#### 3D-ready 输入图")
        st.image(bgr_to_rgb(prepared_asset.bgr_image), width="stretch")

    st.markdown("#### 3D 输入元数据")
    st.dataframe(
        [{
            "原图尺寸": f"{prepared_asset.metadata['source_shape'][1]} x {prepared_asset.metadata['source_shape'][0]}",
            "输出尺寸": prepared_asset.metadata["target_size"],
            "原图覆盖率": f"{prepared_asset.metadata['source_mask_coverage']:.1%}",
            "输出覆盖率": f"{prepared_asset.metadata['prepared_mask_coverage']:.1%}",
            "裁剪框": ", ".join(map(str, prepared_asset.metadata["source_bbox_xyxy"])),
            "放置框": ", ".join(map(str, prepared_asset.metadata["placed_bbox_xyxy"])),
        }],
        width="stretch",
    )

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button(
            "⬇️ 下载 3D-ready PNG",
            data=prepared_asset.png_bytes(transparent=False),
            file_name=f"hunyuan3d_input_{arch_slug}.png",
            mime="image/png",
            width="stretch",
        )
    with dl_col2:
        st.download_button(
            "⬇️ 下载透明 PNG",
            data=prepared_asset.png_bytes(transparent=True),
            file_name=f"hunyuan3d_input_rgba_{arch_slug}.png",
            mime="image/png",
            width="stretch",
        )
    with dl_col3:
        st.download_button(
            "⬇️ 下载预处理元数据",
            data=prepared_asset.metadata_bytes(),
            file_name=f"hunyuan3d_input_metadata_{arch_slug}.json",
            mime="application/json",
            width="stretch",
        )

    st.markdown("#### Hunyuan3D-2.1 提交")
    should_autopoll = bool(current_job_uid) and not result_bytes and current_job_status not in {"completed", "error"}
    if should_autopoll:
        st_autorefresh(interval=5000, key="hunyuan_single_job_autorefresh")
        try:
            client = HunyuanServiceClient(service_config)
            job_status = client.get_job_status(current_job_uid)
            st.session_state.hunyuan_job_status = job_status.status
            if job_status.model_bytes:
                st.session_state.hunyuan_job_result_bytes = job_status.model_bytes
                st.session_state.hunyuan_job_result_path = str(
                    _persist_hunyuan_result(current_job_uid, job_status.model_bytes, arch_slug=arch_slug)
                )
            if job_status.message:
                st.session_state.hunyuan_job_message = job_status.message
            current_job_status = job_status.status
            current_job_message = job_status.message
            result_bytes = job_status.model_bytes or result_bytes
        except Exception as exc:
            st.warning(f"自动轮询 3D 任务状态失败：{exc}")

    submit_col, refresh_col = st.columns(2)
    submit_disabled = bool(current_job_uid) and current_job_status not in {None, "completed", "error"}
    with submit_col:
        if st.button("提交到 Hunyuan3D-2.1", key="submit_hunyuan_single", type="primary", width="stretch", disabled=submit_disabled):
            try:
                client = HunyuanServiceClient(service_config)
                job_uid = client.submit_image_async(
                    prepared_asset.bgra_image,
                    seed=1234,
                    num_inference_steps=_default_hunyuan_steps(service_config.subfolder),
                    guidance_scale=5.0,
                    octree_resolution=256,
                    texture=False,
                )
                st.session_state.hunyuan_job_uid = job_uid
                st.session_state.hunyuan_job_status = "submitted"
                st.session_state.hunyuan_job_message = None
                st.session_state.hunyuan_job_result_bytes = None
                st.session_state.hunyuan_job_result_path = None
            except Exception as exc:
                st.error(f"❌ 提交 Hunyuan3D-2.1 任务失败：{exc}")
    with refresh_col:
        if st.button("刷新 3D 任务状态", key="refresh_hunyuan_single_status", width="stretch"):
            job_uid = st.session_state.get("hunyuan_job_uid")
            if not job_uid:
                st.warning("⚠️ 当前没有待查询的 3D 任务。")
            else:
                try:
                    client = HunyuanServiceClient(service_config)
                    job_status = client.get_job_status(job_uid)
                    st.session_state.hunyuan_job_status = job_status.status
                    if job_status.model_bytes:
                        st.session_state.hunyuan_job_result_bytes = job_status.model_bytes
                        st.session_state.hunyuan_job_result_path = str(
                            _persist_hunyuan_result(job_uid, job_status.model_bytes, arch_slug=arch_slug)
                        )
                    if job_status.message:
                        st.session_state.hunyuan_job_message = job_status.message
                except Exception as exc:
                    st.error(f"❌ 查询 Hunyuan3D-2.1 状态失败：{exc}")

    current_job_uid = st.session_state.get("hunyuan_job_uid")
    current_job_status = st.session_state.get("hunyuan_job_status")
    current_job_message = st.session_state.get("hunyuan_job_message")
    current_job_result_path = st.session_state.get("hunyuan_job_result_path")
    if current_job_uid:
        st.caption(f"当前任务 UID: {current_job_uid} | 状态: {current_job_status or 'submitted'}")
    if submit_disabled:
        st.caption("当前 3D 服务一次只处理一个任务，请等待本次任务完成或失败后再提交新的任务。")
    if should_autopoll:
        st.info("任务进行中，页面会每 5 秒自动轮询一次。")
    if current_job_status == "error" and current_job_message:
        st.error(f"❌ Hunyuan3D-2.1 任务失败：{current_job_message}")

    result_bytes = st.session_state.get("hunyuan_job_result_bytes")
    if not result_bytes and current_job_result_path:
        result_path_obj = Path(current_job_result_path)
        if result_path_obj.exists():
            result_bytes = result_path_obj.read_bytes()
            st.session_state.hunyuan_job_result_bytes = result_bytes
    if result_bytes:
        st.success("✅ 已收到 Hunyuan3D-2.1 输出 mesh。")
        if current_job_result_path:
            st.caption(f"结果已保存到: {current_job_result_path}")
        st.markdown("#### GLB 在线预览")
        _render_glb_preview(result_bytes)
        st.download_button(
            "⬇️ 下载 3D Mesh (GLB)",
            data=result_bytes,
            file_name=f"hunyuan3d_demo_{arch_slug}.glb",
            mime="model/gltf-binary",
            width="stretch",
        )


def _render_hunyuan_generation_panel(
    *,
    region_label: str,
    prepared_source: Optional[np.ndarray],
    prepared_mask: Optional[np.ndarray],
) -> None:
    st.divider()
    st.markdown("### 🧊 第 3 步: Hunyuan3D-2.1 单图生成")
    if prepared_source is not None and prepared_mask is not None:
        st.info(
            """
            当前 3D Demo 支持两条单图输入链路：
            1. 从当前拼接/全景牙弓结果自动准备 3D-ready PNG，再直接提交到 Hunyuan3D-2.1。
            2. 手动上传一张牙齿全景图，做同样的单图预处理后直接提交。
            """
        )
    else:
        st.info("当前还没有自动生成用的牙弓结果，你仍然可以直接手动上传一张牙齿全景图。")

    service_config = HunyuanServiceConfig.from_env()
    probe_button_col, probe_info_col = st.columns([1, 2])
    with probe_button_col:
        if st.button("检测 Hunyuan3D 服务", key="probe_hunyuan3d_service", width="stretch"):
            probe_result = HunyuanServiceClient(service_config).probe()
            st.session_state.hunyuan_probe_result = probe_result
    with probe_info_col:
        st.caption(
            f"服务地址: {service_config.service_url} | 模型: {service_config.model_path} | 子目录: {service_config.subfolder}"
        )

    probe_result = st.session_state.get("hunyuan_probe_result")
    if probe_result is not None:
        if probe_result.reachable:
            st.success(f"✅ Hunyuan3D 服务可访问：{probe_result.url}（状态：{probe_result.status}）")
        else:
            st.warning(f"⚠️ Hunyuan3D 服务暂不可访问：{probe_result.message}")

    tab_labels: list[str] = []
    has_auto_source = prepared_source is not None and prepared_mask is not None
    if has_auto_source:
        tab_labels.append("自动准备单图")
    tab_labels.append("手动上传单图")
    tabs = st.tabs(tab_labels)

    tab_index = 0
    if has_auto_source:
        with tabs[tab_index]:
            prep_btn_col, prep_info_col = st.columns([1, 2])
            with prep_btn_col:
                if st.button("准备 3D 输入", key="prepare_hunyuan_input", type="secondary", width="stretch"):
                    try:
                        prepared_asset = prepare_image_for_hunyuan3d(
                            prepared_source,
                            mask=prepared_mask,
                            target_size=512,
                            padding_ratio=0.12,
                        )
                        _reset_hunyuan_generation_state(clear_prepared_asset=True)
                        st.session_state.prepared_3d_asset = prepared_asset
                    except Exception as exc:
                        st.error(f"❌ 3D 输入准备失败：{exc}")
            with prep_info_col:
                meta = st.session_state.get("stitched_for_3d_meta", {})
                st.caption(
                    "输入来源: "
                    f"{meta.get('region_label', 'unknown')} | "
                    f"可信度: {meta.get('confidence_level', 'unknown')} | "
                    f"严格牙齿输出: {'是' if meta.get('strict_teeth_only', True) else '否'}"
                )

            prepared_asset = st.session_state.get("prepared_3d_asset")
            if prepared_asset is not None:
                if st.button("使用当前 3D-ready 单图", key="use_auto_single_image", type="secondary", width="stretch"):
                    _reset_hunyuan_generation_state()
                    st.session_state.hunyuan_input_asset = prepared_asset
                    st.session_state.hunyuan_input_source_image = prepared_source.copy()
                    st.session_state.hunyuan_input_source_name = f"auto_prepared_{_slugify_arch_label(region_label)}.png"
                    st.session_state.hunyuan_input_source_label = "自动预处理单图"
        tab_index += 1

    with tabs[tab_index]:
        st.caption("手动入口接收 1 张牙齿全景图。支持 PNG/JPG/BMP/TIFF；如果上传透明 PNG，会直接保留 alpha。")
        manual_upload = st.file_uploader(
            "上传牙齿全景图",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=False,
            key="hunyuan_manual_single_image",
        )

        manual_image = None
        manual_error = None
        if manual_upload is not None:
            try:
                manual_image = _uploaded_image_to_cv2(manual_upload)
            except Exception as exc:
                manual_error = str(exc)
                st.error(f"❌ 手动单图读取失败：{manual_error}")

        if manual_image is not None:
            st.image(
                _image_for_display(resize_for_display(manual_image, max_width=420, max_height=260)),
                caption=manual_upload.name,
                width="stretch",
            )

        if st.button(
            "导入手动单图",
            key="build_manual_single_image",
            type="secondary",
            width="stretch",
            disabled=(manual_upload is None or manual_error is not None),
        ):
            try:
                manual_asset, source_bgr = _build_manual_hunyuan_input_asset(manual_upload)
                _reset_hunyuan_generation_state()
                st.session_state.hunyuan_input_asset = manual_asset
                st.session_state.hunyuan_input_source_image = source_bgr
                st.session_state.hunyuan_input_source_name = manual_upload.name
                st.session_state.hunyuan_input_source_label = "手动上传单图"
                st.success("✅ 已导入手动单图，下面可以直接提交到 Hunyuan3D-2.1。")
            except Exception as exc:
                st.error(f"❌ 手动单图导入失败：{exc}")

    prepared_asset = st.session_state.get("hunyuan_input_asset")
    source_image = st.session_state.get("hunyuan_input_source_image")
    source_name = st.session_state.get("hunyuan_input_source_name")
    source_label = st.session_state.get("hunyuan_input_source_label")
    if prepared_asset is not None and source_image is not None and source_name and source_label:
        _render_hunyuan_submission_panel(
            service_config,
            prepared_asset,
            region_label,
            source_label,
            source_name,
            source_image,
        )


def _render_hunyuan_runtime_panel() -> None:
    manager = HunyuanRuntimeManager()
    status = manager.read_status()

    with st.expander("🛠️ Hunyuan3D 本地服务", expanded=not status.service_healthy):
        st.caption("按提示点下一步即可。第一次使用通常只需要点一次。")

        clean_user_hint = _clean_status_text(status.user_hint)
        clean_model_message = _clean_status_text(status.model_message)
        clean_task_message = _clean_status_text(status.task_message)
        clean_service_message = _clean_status_text(status.service_message)

        if status.service_healthy:
            st.success(clean_user_hint)
        elif status.model_status == "invalid":
            st.warning(clean_user_hint)
        elif status.task_running:
            st.info(clean_user_hint)
        else:
            st.info(clean_user_hint)

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            env_state = "已就绪" if status.repo_exists and status.install_ready else "未准备"
            st.metric("运行环境", env_state)
        with summary_col2:
            model_state_label = {
                "ready": "已就绪",
                "missing": "未下载",
                "downloading": "下载中",
                "invalid": "需修复",
            }.get(status.model_status, "未知")
            st.metric("模型文件", model_state_label)
        with summary_col3:
            service_state = "已就绪" if status.service_healthy else ("运行中" if status.service_running else "未启动")
            st.metric("服务状态", service_state)

        st.caption(f"模型状态：{clean_model_message}")
        if status.task_running and clean_task_message:
            st.caption(f"当前进度：{clean_task_message}")
        elif clean_service_message and not status.service_healthy:
            st.caption(f"服务探测：{clean_service_message}")

        action_col1, action_col2, action_col3 = st.columns([2, 1, 1])
        with action_col1:
            primary_disabled = status.recommended_action is None
            if st.button(
                status.recommended_label,
                key="hunyuan_primary_action",
                type="primary",
                width="stretch",
                disabled=primary_disabled,
            ):
                ok, message = manager.launch_action(status.recommended_action) if status.recommended_action else (False, "当前无需操作。")
                if ok:
                    st.success(message)
                    st.rerun()
                st.warning(message)
        with action_col2:
            if st.button("刷新状态", key="hunyuan_runtime_refresh", width="stretch"):
                st.rerun()
        with action_col3:
            if st.button(
                "停止服务",
                key="hunyuan_stop_service",
                width="stretch",
                disabled=not status.service_running,
            ):
                ok, message = manager.stop_service()
                if ok:
                    st.success(message)
                    st.rerun()
                st.warning(message)

        if status.service_healthy:
            st.caption(f"服务地址：{status.service_url}")

        with st.expander("高级操作", expanded=False):
            st.caption("遇到问题时再使用这些按钮和日志。")

            advanced_col1, advanced_col2, advanced_col3 = st.columns(3)
            with advanced_col1:
                if st.button("仅初始化环境", key="hunyuan_setup_only", width="stretch"):
                    ok, message = manager.launch_action("setup")
                    if ok:
                        st.success(message)
                        st.rerun()
                    st.warning(message)
            with advanced_col2:
                if st.button("下载/修复模型", key="hunyuan_download_model_only", width="stretch"):
                    ok, message = manager.launch_action("download-model")
                    if ok:
                        st.success(message)
                        st.rerun()
                    st.warning(message)
            with advanced_col3:
                if st.button("仅启动服务", key="hunyuan_start_service_only", width="stretch"):
                    ok, message = manager.launch_action("start-service")
                    if ok:
                        st.success(message)
                        st.rerun()
                    st.warning(message)

            st.caption(f"仓库目录：{status.repo_dir}")
            st.caption(f"模型目录：{status.model_dir}")

            log_col1, log_col2 = st.columns(2)
            with log_col1:
                st.markdown("#### 后台任务日志")
                if status.task_log_tail:
                    st.code(status.task_log_tail, language="text")
                else:
                    st.caption("暂无后台任务日志。")
            with log_col2:
                st.markdown("#### 服务日志")
                if status.service_log_tail:
                    st.code(status.service_log_tail, language="text")
                else:
                    st.caption("暂无服务日志。")


def main() -> None:
    # 导航链接到拍照页面
    st.sidebar.page_link("pages/capture.py", label="📷 拍照采集", icon="📷")
    st.sidebar.divider()

    st.title("口腔牙齿提取拼接 v1")
    st.caption("标准化流水线：牙齿分割 → 牙齿区域特征提取 → 配准 → 仅牙齿区域融合。")
    _render_hunyuan_runtime_panel()

    # ============ 侧边栏配置 ============
    capture_upper_images = st.session_state.get("upper_arch_images", [])
    capture_lower_images = st.session_state.get("lower_arch_images", [])
    capture_arch_options: list[str] = []
    if len(capture_upper_images) > 0:
        capture_arch_options.append("上牙弓")
    if len(capture_lower_images) > 0:
        capture_arch_options.append("下牙弓")

    source_mode = "上传图像"
    selected_capture_arch: Optional[str] = None
    with st.sidebar:
        st.subheader("📁 图像来源")
        source_options = ["上传图像"]
        if capture_arch_options:
            source_options.append("使用拍照页已采集图像")
        source_mode = st.radio(
            "选择来源",
            source_options,
            index=1 if len(source_options) > 1 else 0,
            help="如果你已经在拍照页完成了一个牙弓的采集，可以直接在这里继续处理。"
        )

        uploads = None
        if source_mode == "上传图像":
            uploads = st.file_uploader(
                "上传多张口腔内窥镜图像",
                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                accept_multiple_files=True,
            )
        else:
            selected_capture_arch = st.selectbox(
                "选择已采集牙弓",
                capture_arch_options,
                index=0,
                help="直接复用拍照页中已经采集完成的牙弓照片。"
            )
            st.caption(
                "当前可用："
                + "，".join(
                    [
                        f"上牙弓 {len(capture_upper_images)} 张" if capture_upper_images else "",
                        f"下牙弓 {len(capture_lower_images)} 张" if capture_lower_images else "",
                    ]
                ).strip("，")
            )

        st.divider()

        has_input_source = bool(uploads) if source_mode == "上传图像" else bool(selected_capture_arch)
        if has_input_source:
            st.subheader("⚙️ 分割设置")
            seg_method = st.selectbox(
                "分割模型",
                ["AlphaDent (YOLOv8)", "U-Net"],
                index=0,
                help="AlphaDent: 快速准确 | U-Net: 深度学习分割"
            )
            seg_method_internal = "alphadent" if "AlphaDent" in seg_method else "unet"

            seg_conf = st.slider(
                "AlphaDent 置信度阈值",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="越低越敏感，越高越严格",
                disabled=(seg_method_internal == "unet")
            )
            use_grabcut = st.checkbox("启用 GrabCut 精细化", value=True, help="使用 GrabCut 优化分割边界")
            use_enhancement = st.checkbox("启用 CLAHE 图像增强", value=True, help="增强暗图像和低对比度图像，提高检测率")
            enhancement_level = st.slider(
                "增强强度",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.5,
                help="CLAHE 对比度限制，越高增强效果越明显"
            )

            st.subheader("🔧 自动标定设置")
            enable_calibration = st.checkbox(
                "启用自动标定",
                value=False,
                help="利用牙齿几何特性自动校正畸变。\n\n⚠️ 要求：\n1. 必须使用AlphaDent (YOLOv8) 分割\n2. 每张图像至少检测到4个牙齿实例\n3. 图像畸变不能过严重（否则YOLO会失败）"
            )

            if enable_calibration:
                if seg_method_internal == "unet":
                    st.error("❌ 自动标定不支持U-Net分割！")
                    st.caption("原因：U-Net是语义分割，无法提取单个牙齿实例信息（位置、类别、边界框）")
                    st.caption("解决：改用AlphaDent (YOLOv8) 分割，或禁用自动标定")
                    enable_calibration = False  # 强制禁用
                else:
                    st.warning("⚠️ 自动标定需要YOLO检测到足够的牙齿实例（≥4个）")
                    st.caption("ℹ️ 标定参数将自动保存，后续可复用")
                    st.caption("⚠️ 标定失败时会详细提示原因，请调整图像质量后重试")

            st.subheader("🔗 拼接设置")
            feature_method = st.selectbox(
                "特征方法",
                ["sift", "akaze", "orb", "loftr"],
                index=0,
                help="SIFT: 对真实拍摄更稳，当前推荐 | AKAZE: 中等速度与质量 | ORB: 最快但对角度和光照更敏感 | LoFTR: 预留深度匹配入口"
            )
            if source_mode == "上传图像":
                region_label = st.selectbox(
                    "采集区域",
                    ["左上", "左下", "右上", "右下"],
                    index=0,
                    help="一次只上传同一区域的相关图片，方便连续拼接"
                )
            else:
                region_label = selected_capture_arch or "未知牙弓"
                st.info(f"当前牙弓：{region_label}")

            st.subheader("✨ 去噪设置")
            enable_noise_removal = st.checkbox(
                "启用拼接结果去噪",
                value=True,  # 默认选中
                help="去除拼接后图像中的小块软组织噪点，保留牙齿主体区域"
            )

            if enable_noise_removal:
                noise_method = st.selectbox(
                    "去噪方法",
                    ["method4 (推荐)", "method1", "method2", "method3"],
                    index=0,
                    help="method4: 形态学核=7 + 连通域面积≥2000px (效果最好) | method1-3: 不同参数组合"
                )
                noise_method_internal = noise_method.split()[0]  # 提取 "method4"
                st.caption(f"ℹ️ 当前使用 {noise_method_internal}: 形态学开运算去除小噪点 + 连通域分析保留大区域")
            else:
                noise_method_internal = None

    # ============ 步骤 1: 上传和预览 ============
    if source_mode == "上传图像":
        if not uploads:
            st.info("👈 请先在左侧上传图像")
            _render_hunyuan_generation_panel(
                region_label="manual_import",
                prepared_source=st.session_state.get("stitched_for_3d_image"),
                prepared_mask=st.session_state.get("stitched_for_3d_mask"),
            )
            return
        packets = load_uploaded_images(uploads)
    else:
        if selected_capture_arch == "上牙弓":
            packets = list(capture_upper_images)
        elif selected_capture_arch == "下牙弓":
            packets = list(capture_lower_images)
        else:
            packets = []

    if len(packets) == 0:
        st.error("❌ 没有读取到有效图像文件")
        _render_hunyuan_generation_panel(
            region_label="manual_import",
            prepared_source=st.session_state.get("stitched_for_3d_image"),
            prepared_mask=st.session_state.get("stitched_for_3d_mask"),
        )
        return

    st.markdown(f"### 📸 已上传 {len(packets)} 张图像")
    if source_mode == "上传图像":
        st.info(f"当前区域：{region_label}。请确保本次上传的图片都来自同一区域，并按你希望的拼接顺序整理。")
    else:
        st.info(f"当前牙弓：{region_label}。这些照片来自拍照页的已采集结果，可直接用于后续拼接和三维处理。")
    cols = st.columns(min(len(packets), 4))
    for idx, (col, packet) in enumerate(zip(cols, packets)):
        with col:
            st.image(
                bgr_to_rgb(resize_for_display(packet.image)),
                caption=f"{packet.name}",
                width="stretch"
            )

    # ============ 步骤 2: 分割 ==========
    st.divider()
    st.markdown("### 🔪 第 1 步: 牙齿分割")

    col_seg_btn, col_seg_info = st.columns([1, 3])
    with col_seg_btn:
        run_segmentation = st.button("执行分割", type="secondary", width="stretch")

        with col_seg_info:
            method_name = "AlphaDent (YOLOv8)" if seg_method_internal == "alphadent" else "U-Net"
            grabcut_status = "启用" if use_grabcut else "不启用"
            enhancement_status = "启用" if use_enhancement else "不启用"
            st.info(
                f"""
                **分割说明**：
                - 使用 {method_name} 进行牙齿区域检测
                - GrabCut 精细化：{grabcut_status}
                - CLAHE 图像增强：{enhancement_status}
                - 绿色覆盖区域表示识别为牙齿的部分
                - 后续拼接只会保留分割出的牙齿主体区域
                """
            )

    if run_segmentation:
        with st.spinner("🔄 正在分割图像..."):
            st.session_state.pop("stitch_outputs", None)
            st.session_state.pop("stitched_for_3d_image", None)
            st.session_state.pop("stitched_for_3d_mask", None)
            st.session_state.pop("stitched_for_3d_meta", None)
            _reset_hunyuan_generation_state(clear_prepared_asset=True)
            seg_results = []
            seg_metrics = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            has_error = False
            error_messages = []

            for idx, packet in enumerate(packets):
                status_text.text(f"正在处理第 {idx+1}/{len(packets)} 张图像: {packet.name}")
                progress_bar.progress((idx + 1) / len(packets))

                try:
                    seg_result = segment_teeth(
                        packet.image,
                        method=seg_method_internal,
                        use_grabcut=use_grabcut,
                        use_enhancement=use_enhancement,
                        enhancement_level=enhancement_level
                    )

                    if cv2.countNonZero(seg_result.mask) == 0:
                        st.warning(f"⚠️ 图像 {idx} ({packet.name}) 分割结果为空，使用全白 mask")
                        seg_result = fallback_full_mask(packet.image)

                    mask_ratio = float(cv2.countNonZero(seg_result.mask) / seg_result.mask.size)

                    seg_results.append(seg_result)
                    seg_metrics.append({
                        "index": idx,
                        "name": packet.name,
                        "method": seg_result.method,
                        "fallback": seg_result.fallback_reason,
                        "coverage": mask_ratio,
                        "pixels": int(cv2.countNonZero(seg_result.mask)),
                        "total": seg_result.mask.size
                    })

                except RuntimeError as e:
                    has_error = True
                    error_msg = f"图像 {idx+1} ({packet.name}) 分割失败: {str(e)}"
                    error_messages.append(error_msg)
                    st.error(f"❌ {error_msg}")
                except Exception as e:
                    has_error = True
                    error_msg = f"图像 {idx+1} ({packet.name}) 发生意外错误: {str(e)}"
                    error_messages.append(error_msg)
                    st.error(f"❌ {error_msg}")

            if has_error:
                st.error("❌ 分割过程中出现错误，请检查错误信息并调整配置后重试。")
                if error_messages:
                    with st.expander("查看详细错误信息"):
                        for msg in error_messages:
                            st.text(msg)
                _render_hunyuan_generation_panel(
                    region_label=region_label,
                    prepared_source=st.session_state.get("stitched_for_3d_image"),
                    prepared_mask=st.session_state.get("stitched_for_3d_mask"),
                )
                st.stop()

            if len(seg_results) == 0:
                st.error("❌ 没有成功分割任何图像，请检查模型配置和图像质量。")
                _render_hunyuan_generation_panel(
                    region_label=region_label,
                    prepared_source=st.session_state.get("stitched_for_3d_image"),
                    prepared_mask=st.session_state.get("stitched_for_3d_mask"),
                )
                st.stop()

            st.session_state.seg_results = seg_results
            st.session_state.seg_metrics = seg_metrics

    # ============ 步骤 3: 分割结果展示 ============
    if "seg_results" in st.session_state:
        st.success(f"✅ 分割完成！共处理 {len(st.session_state.seg_results)} 张图像")

        with st.expander("📊 分割统计", expanded=False):
            metrics_df = []
            for m in st.session_state.seg_metrics:
                metrics_df.append({
                    "图像": m["name"],
                    "方法": m["method"],
                    "覆盖率": f"{m['coverage']:.1%}",
                    "像素数": f"{m['pixels']:,}",
                    "回退原因": m["fallback"] or "无"
                })
            st.dataframe(metrics_df, width="stretch")

        st.markdown("#### 🔍 分割结果对比（原图 vs 掩膜）")

        view_mode = st.radio(
            "查看模式",
            ["并排对比", "网格展示", "单独查看"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if view_mode == "单独查看":
            selected_idx = st.selectbox(
                "选择图像",
                range(len(packets)),
                format_func=lambda i: f"{i+1}. {packets[i].name}"
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原图**")
                st.image(
                    bgr_to_rgb(resize_for_display(packets[selected_idx].image)),
                    width="stretch"
                )

            with col2:
                st.markdown("**分割掩膜**")
                st.image(
                    bgr_to_rgb(resize_for_display(st.session_state.seg_results[selected_idx].overlay)),
                    width="stretch"
                )

            m = st.session_state.seg_metrics[selected_idx]
            st.markdown(f"""
            **统计信息**：
            - 方法: {m['method']}
            - 覆盖率: {m['coverage']:.1%} ({m['pixels']:,} / {m['total']:,} 像素)
            - 回退原因: {m['fallback'] or '无'}
            """)

        elif view_mode == "并排对比":
            for idx, (packet, seg_result) in enumerate(zip(packets, st.session_state.seg_results)):
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(
                            bgr_to_rgb(resize_for_display(packet.image)),
                            caption=f"原图: {packet.name}",
                            width="stretch"
                        )
                    with col2:
                        st.image(
                            bgr_to_rgb(resize_for_display(seg_result.overlay)),
                            caption=f"分割掩膜: {packet.name}",
                            width="stretch"
                        )
                    st.divider()

        else:
            for idx, (packet, seg_result) in enumerate(zip(packets, st.session_state.seg_results)):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        bgr_to_rgb(resize_for_display(packet.image)),
                        caption=f"原图 {idx+1}",
                        width="stretch"
                    )
                with col2:
                    st.image(
                        bgr_to_rgb(resize_for_display(seg_result.overlay)),
                        caption=f"掩膜 {idx+1}",
                        width="stretch"
                    )

        st.markdown("#### 💾 下载分割结果")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            if st.button("下载所有掩膜 (ZIP)", width="stretch"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, (packet, seg_result) in enumerate(zip(packets, st.session_state.seg_results)):
                        success, encoded = cv2.imencode(".png", seg_result.overlay)
                        if success:
                            zip_file.writestr(f"mask_overlay_{idx}_{packet.name}", encoded.tobytes())

                        success, encoded = cv2.imencode(".png", seg_result.mask)
                        if success:
                            zip_file.writestr(f"mask_binary_{idx}_{packet.name}", encoded.tobytes())

                zip_buffer.seek(0)
                st.download_button(
                    "⬇️ 点击下载 ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"segmentation_masks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    width="stretch"
                )

        with col_dl2:
            seg_report = {
                "timestamp": datetime.now().isoformat(),
                "total_images": len(packets),
                "arch_label": region_label,
                "metrics": st.session_state.seg_metrics
            }
            st.download_button(
                "⬇️ 下载分割报告 JSON",
                data=json.dumps(seg_report, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"segmentation_report_{_slugify_arch_label(region_label)}.json",
                mime="application/json",
                width="stretch"
            )

        # ============ 步骤 4: 拼接 ==========
        st.divider()
        st.markdown("### 🦷 第 2 步: 牙齿主体拼接")

        single_panorama_mode = len(packets) == 1
        if single_panorama_mode:
            st.info("检测到仅 1 张图像：将按单张全景图处理，跳过二维拼接，直接进入后续去噪、下载和三维准备流程。")
            selected_indices = [0]
        else:
            st.markdown("#### 🧭 拼接顺序")
            order_options = [f"{i+1}. {packet.name}" for i, packet in enumerate(packets)]
            ordered_labels = st.multiselect(
                "按拼接顺序选择图像（默认按上传顺序）",
                options=order_options,
                default=order_options,
                help="请只保留当前区域内、希望参与牙齿主体拼接的图像，并按顺序排列。"
            )
            selected_indices = [order_options.index(label) for label in ordered_labels]

            if len(selected_indices) < 2:
                st.warning("⚠️ 请选择至少 2 张图像参与拼接；如果只想测试单张全景图，请只上传 1 张。")
                _render_hunyuan_generation_panel(
                    region_label=region_label,
                    prepared_source=st.session_state.get("stitched_for_3d_image"),
                    prepared_mask=st.session_state.get("stitched_for_3d_mask"),
                )
                return

        st.info(
            f"""
            **拼接说明**：
            - 当前区域：**{region_label}**
            - {"单张全景图模式：跳过特征匹配和拼接" if single_panorama_mode else f"当前使用 **{feature_method.upper()}** 特征匹配方法"}
            - {"直接使用当前图像作为完整牙弓输入" if single_panorama_mode else "按你选择的顺序进行逐步拼接"}
            - 最终输出默认只保留牙齿主体区域
            - 即使低质量图像也会尽量输出结果，但会标记可信度、fallback 和降级原因
            """
        )

        col_stitch_btn, col_stitch_info = st.columns([1, 3])
        with col_stitch_btn:
            run_stitching = st.button("使用单张全景图" if single_panorama_mode else "开始拼接", type="primary", width="stretch")

        if run_stitching:
            spinner_text = "🔄 正在处理单张全景图..." if single_panorama_mode else "🔄 正在拼接牙齿主体..."
            with st.spinner(spinner_text):
                ordered_packets = [packets[i] for i in selected_indices]
                images = [p.image for p in ordered_packets]
                pipeline_seg_results = st.session_state.get("seg_results")
                if pipeline_seg_results and len(pipeline_seg_results) == len(packets):
                    pipeline_seg_results = [pipeline_seg_results[i] for i in selected_indices]
                else:
                    pipeline_seg_results = None
                outputs = run_pipeline(
                    images,
                    feature_method=feature_method,
                    seg_results=pipeline_seg_results,
                    enable_auto_calibration=enable_calibration  # 新增参数
                )
                st.session_state.stitch_outputs = outputs
                _reset_hunyuan_generation_state(clear_prepared_asset=True)

        outputs = st.session_state.get("stitch_outputs")
        if outputs is not None:
            # 显示标定诊断（如果启用标定）
            if enable_calibration and outputs.diagnostics.quality_gate and "calibration" in outputs.diagnostics.quality_gate:
                calibration_diag = outputs.diagnostics.quality_gate["calibration"]
                st.divider()
                st.markdown("#### 📊 自动标定诊断")

                # 判断标定是否成功
                calib_success = calibration_diag.get("quality_validation", {}).get("success", False)

                if calib_success:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rmse_improvement = calibration_diag.get("distortion_correction", {}).get("rmse_improvement", 0.0)
                        st.metric("RMSE改善", f"{rmse_improvement:.2f}px")
                    with col2:
                        spacing_improvement = calibration_diag.get("distortion_correction", {}).get("spacing_improvement", 0.0)
                        st.metric("间距一致性提升", f"{spacing_improvement:.2%}")
                    with col3:
                        confidence = "high" if calibration_diag.get("distortion_correction", {}).get("distortion_reasonable", False) else "medium"
                        st.metric("置信度", confidence)

                    st.success("✅ 标定成功，畸变已校正")
                else:
                    st.error(f"❌ 标定失败：{calibration_diag.get('failure_reason', 'unknown')}")
                    st.caption("请检查图像质量（清晰度、重叠度）后重试")

                # 详细诊断
                with st.expander("查看完整标定诊断"):
                    st.json(calibration_diag)

            result_col, diag_col = st.columns([1.3, 1.0], gap="large")

            with result_col:
                st.markdown("#### 🖼️ 牙齿提取拼接结果")
                quality_gate = outputs.diagnostics.quality_gate or {}
                confidence_level = quality_gate.get("confidence_level", "unknown")
                strict_teeth_only = quality_gate.get("strict_teeth_only", True)

                if outputs.stitched is None:
                    st.error("❌ 拼接失败，请检查图像重叠和清晰度")
                else:
                    # 应用去噪处理（如果启用）
                    stitched_to_display = outputs.stitched
                    stitched_mask_for_3d = outputs.stitched_mask.copy() if outputs.stitched_mask is not None else derive_mask_from_image(outputs.stitched)
                    noise_removal_stats = None

                    if enable_noise_removal and noise_method_internal:
                        st.info(f"🔄 正在应用去噪处理 ({noise_method_internal})...")

                        # 获取去噪配置
                        noise_config = get_noise_removal_method_config(noise_method_internal)

                        # 执行去噪
                        stitched_cleaned, noise_removal_stats = remove_noise_from_stitched_result(
                            outputs.stitched,
                            noise_config
                        )

                        stitched_to_display = stitched_cleaned
                        stitched_mask_for_3d = derive_mask_from_image(stitched_cleaned)

                        # 显示去噪统计
                        if noise_removal_stats:
                            st.success(f"✅ 去噪完成：去除 {noise_removal_stats['removed_noise_count']} 个小噪点")

                            with st.expander("📊 查看去噪统计详情"):
                                st.text(format_noise_removal_stats(noise_removal_stats))

                    # 显示拼接结果（去噪后或原始）
                    if confidence_level == "high":
                        st.success("✅ 单张全景图处理完成，当前结果可信度较高" if quality_gate.get("single_image_panorama_mode") else "✅ 拼接完成，当前结果可信度较高")
                    elif confidence_level == "medium":
                        st.warning("⚠️ 单张全景图已输出，但当前结果为中等可信度，请重点检查分割区域" if quality_gate.get("single_image_panorama_mode") else "⚠️ 拼接完成，但当前结果为中等可信度，请重点检查边缘和重叠区域")
                    else:
                        st.warning("⚠️ 单张全景图已输出，但当前结果低可信，仅建议作为参考" if quality_gate.get("single_image_panorama_mode") else "⚠️ 拼接已输出，但当前结果低可信，仅建议作为参考")

                    if not strict_teeth_only:
                        st.caption("当前结果包含 fallback full mask 输入，因此属于 degraded teeth-only 输出。")

                    # 显示标签
                    if quality_gate.get("single_image_panorama_mode"):
                        result_label = "去噪后的单张全景牙齿图" if enable_noise_removal else "单张全景牙齿图（未去噪）"
                    else:
                        result_label = "去噪后的牙齿拼接图" if enable_noise_removal else "牙齿拼接图（未去噪）"
                    st.image(bgr_to_rgb(stitched_to_display), caption=result_label, width="stretch")
                    st.session_state.stitched_for_3d_image = stitched_to_display.copy()
                    st.session_state.stitched_for_3d_mask = stitched_mask_for_3d.copy()
                    st.session_state.stitched_for_3d_meta = {
                        "region_label": region_label,
                        "confidence_level": confidence_level,
                        "strict_teeth_only": strict_teeth_only,
                        "noise_removal_enabled": enable_noise_removal,
                        "noise_method": noise_method_internal,
                    }

                    # 提供下载选项
                    col_dl1, col_dl2 = st.columns(2)

                    with col_dl1:
                        # 下载最终结果（去噪后或原始）
                        if quality_gate.get("single_image_panorama_mode"):
                            download_label = "⬇️ 下载去噪后全景图" if enable_noise_removal else "⬇️ 下载单张全景图"
                        else:
                            download_label = "⬇️ 下载去噪后拼接图" if enable_noise_removal else "⬇️ 下载牙齿拼接图"
                        arch_slug = _slugify_arch_label(region_label)
                        filename = (
                            f"panorama_teeth_cleaned_{arch_slug}.png"
                            if quality_gate.get("single_image_panorama_mode") and enable_noise_removal else
                            f"panorama_teeth_only_{arch_slug}.png"
                            if quality_gate.get("single_image_panorama_mode") else
                            f"stitched_teeth_cleaned_{arch_slug}.png"
                            if enable_noise_removal else
                            f"stitched_teeth_only_{arch_slug}.png"
                        )
                        st.download_button(
                            download_label,
                            data=_encode_png_bytes(stitched_to_display),
                            file_name=filename,
                            mime="image/png",
                            width="stretch",
                        )

                    with col_dl2:
                        # 如果启用了去噪，额外提供原始拼接图下载
                        if enable_noise_removal:
                            original_label = "⬇️ 下载原始全景图（未去噪）" if quality_gate.get("single_image_panorama_mode") else "⬇️ 下载原始拼接图（未去噪）"
                            original_prefix = "panorama_teeth_original" if quality_gate.get("single_image_panorama_mode") else "stitched_teeth_original"
                            st.download_button(
                                original_label,
                                data=_encode_png_bytes(outputs.stitched),
                                file_name=f"{original_prefix}_{_slugify_arch_label(region_label)}.png",
                                mime="image/png",
                                width="stretch",
                            )

                if quality_gate.get("single_image_panorama_mode"):
                    st.markdown("#### 🔬 单图输入预览")
                    st.image(bgr_to_rgb(outputs.match_visualization), width="stretch")
                else:
                    st.markdown("#### 🔬 匹配可视化")
                    st.image(outputs.match_visualization, width="stretch")

                st.markdown("#### 📈 质量指标")
                if outputs.diagnostics.per_image:
                    st.dataframe(
                        [{
                            "图像索引": item["index"],
                            "清晰度": round(item["sharpness"], 2),
                            "曝光": round(item["exposure"], 2),
                            "掩膜覆盖率": f"{item['mask_coverage']:.1%}",
                            "分割方法": item["segmentation_method"],
                            "Fallback": "是" if item.get("used_fallback_mask") else "否",
                            "严格牙齿输出": "是" if item.get("strict_teeth_only") else "否",
                        } for item in outputs.diagnostics.per_image],
                        width="stretch",
                    )

                if quality_gate:
                    reasons = quality_gate.get("fail_reasons") or quality_gate.get("degrade_reasons") or ["无"]
                    st.markdown("#### 🛡️ 拼接质量门控")
                    st.dataframe(
                        [{
                            "Gate通过": "是" if quality_gate.get("gate_passed") else "否",
                            "可信度": quality_gate.get("confidence_level", "unknown"),
                            "输出模式": outputs.diagnostics.output_mode,
                            "严格牙齿输出": "是" if quality_gate.get("strict_teeth_only", True) else "否",
                            "已纳入图像": ", ".join(map(str, quality_gate.get("accepted_indices", []))) if quality_gate.get("accepted_indices") else "当前两图模式",
                            "跳过图像": ", ".join(map(str, quality_gate.get("skipped_indices", []))) if quality_gate.get("skipped_indices") else "无",
                            "主要原因": ", ".join(reasons),
                        }],
                        width="stretch",
                    )

                    if quality_gate.get("steps"):
                        st.markdown("#### 🪜 逐步拼接明细")
                        st.dataframe(
                            [{
                                "参考图": step.get("reference_index"),
                                "候选图": step.get("candidate_index"),
                                "已纳入": "是" if step.get("accepted") else "否",
                                "Gate通过": "是" if step.get("gate_passed") else "否",
                                "可信度": step.get("confidence_level", "unknown"),
                                "累计牙齿覆盖率": f"{step.get('stitched_mask_coverage', 0.0):.1%}",
                                "失败原因": ", ".join(step.get("fail_reasons", [])) or "无",
                                "降级原因": ", ".join(step.get("degrade_reasons", [])) or "无",
                            } for step in quality_gate.get("steps", [])],
                            width="stretch",
                        )

            with diag_col:
                st.markdown("#### 📋 诊断信息")
                diagnostic_payload = outputs.diagnostics.to_dict()
                st.json(diagnostic_payload)
                st.download_button(
                    "⬇️ 下载诊断 JSON",
                    data=json.dumps(diagnostic_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"diagnostics_teeth_only_{_slugify_arch_label(region_label)}.json",
                    mime="application/json",
                    width="stretch",
                )
                st.caption(
                    f"分割来源: {diagnostic_payload.get('segmentation_source', 'pipeline_runtime')} | 输出模式: {diagnostic_payload.get('output_mode', 'unknown')}"
                )
    else:
        st.info("👆 点击上方「执行分割」按钮开始处理")

    _render_hunyuan_generation_panel(
        region_label=region_label,
        prepared_source=st.session_state.get("stitched_for_3d_image"),
        prepared_mask=st.session_state.get("stitched_for_3d_mask"),
    )

    st.divider()
    st.caption(
        """
        **使用提示**：
        1. 上传同一牙弓、同一侧段的口腔内窥镜图像；如果只上传 1 张，则默认它是完整全景图
        2. 点击「执行分割」查看分割效果
        3. 确认分割效果满意后，多图点击「开始拼接」，单图点击「使用单张全景图」
        4. 在第 3 步里可以直接把当前全景图送进 Hunyuan3D-2.1，或手动上传一张牙齿全景图
        5. 下载牙齿主体结果、3D-ready 输入图和生成后的 GLB
        """
    )


if __name__ == "__main__":
    main()
