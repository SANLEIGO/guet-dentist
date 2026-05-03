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

from dental_stitcher_v1.io_utils import bgr_to_rgb, load_uploaded_images, resize_for_display
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
    build_pseudo_multiview_pack,
    derive_mask_from_image,
    prepare_image_for_hunyuan3d,
)


st.set_page_config(page_title="口腔牙齿提取拼接 v1", page_icon="🦷", layout="wide")

HUNYUAN_RESULTS_DIR = Path(__file__).resolve().parent / ".runtime" / "hunyuan3d" / "results"


def _encode_png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("无法将图像编码为 PNG。")
    return encoded.tobytes()


def _persist_hunyuan_result(job_uid: str, model_bytes: bytes) -> Path:
    HUNYUAN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = HUNYUAN_RESULTS_DIR / f"{job_uid}.glb"
    output_path.write_bytes(model_bytes)
    return output_path


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


def _render_hunyuan_runtime_panel() -> None:
    manager = HunyuanRuntimeManager()
    status = manager.read_status()

    with st.expander("🛠️ Hunyuan3D 本地启动器", expanded=False):
        st.caption(
            "这里可以直接在当前机器上一键拉取 Hunyuan3D-2 代码、安装依赖、下载 2mv 模型并启动本地 bridge 服务。"
        )

        st.dataframe(
            [{
                "代码仓库": "已就绪" if status.repo_exists else "未下载",
                "依赖环境": "已安装" if status.install_ready else "未安装",
                "模型文件": "已下载" if status.model_ready else "未下载",
                "后台任务": f"运行中 (PID {status.task_pid})" if status.task_running else "空闲",
                "服务进程": f"运行中 (PID {status.service_pid})" if status.service_running else "未启动",
                "服务健康": "正常" if status.service_healthy else "未就绪",
            }],
            width="stretch",
        )

        st.caption(f"仓库目录: {status.repo_dir}")
        st.caption(f"模型目录: {status.model_dir}")
        st.caption(f"服务地址: {status.service_url}")

        action_col1, action_col2, action_col3 = st.columns(3)
        with action_col1:
            if st.button("一键下载并启动", key="hunyuan_bootstrap_and_start", type="primary", width="stretch"):
                ok, message = manager.launch_action("bootstrap-and-start")
                if ok:
                    st.success(message)
                    st.rerun()
                st.warning(message) if not ok else None
        with action_col2:
            if st.button("仅初始化环境", key="hunyuan_setup_only", width="stretch"):
                ok, message = manager.launch_action("setup")
                if ok:
                    st.success(message)
                    st.rerun()
                st.warning(message) if not ok else None
        with action_col3:
            if st.button("仅下载模型", key="hunyuan_download_model_only", width="stretch"):
                ok, message = manager.launch_action("download-model")
                if ok:
                    st.success(message)
                    st.rerun()
                st.warning(message) if not ok else None

        action_col4, action_col5, action_col6 = st.columns(3)
        with action_col4:
            if st.button("仅启动服务", key="hunyuan_start_service_only", width="stretch"):
                ok, message = manager.launch_action("start-service")
                if ok:
                    st.success(message)
                    st.rerun()
                st.warning(message) if not ok else None
        with action_col5:
            if st.button("停止服务", key="hunyuan_stop_service", width="stretch"):
                ok, message = manager.stop_service()
                if ok:
                    st.success(message)
                    st.rerun()
                st.warning(message) if not ok else None
        with action_col6:
            if st.button("刷新启动器状态", key="hunyuan_runtime_refresh", width="stretch"):
                st.rerun()

        if status.service_message:
            if status.service_healthy:
                st.success(f"服务探测结果: {status.service_message}")
            else:
                st.info(f"服务探测结果: {status.service_message}")

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
    with st.sidebar:
        st.subheader("📁 图像上传")
        uploads = st.file_uploader(
            "上传多张口腔内窥镜图像",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        )

        st.divider()

        if uploads:
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
                ["orb", "akaze", "sift", "loftr"],
                index=0,
                help="ORB: 快速稳定 | AKAZE: 高质量 | SIFT: 最精确 | LoFTR: 深度匹配"
            )
            region_label = st.selectbox(
                "采集区域",
                ["左上", "左下", "右上", "右下"],
                index=0,
                help="一次只上传同一区域的相关图片，方便连续拼接"
            )

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
    if not uploads:
        st.info("👈 请先在左侧上传图像")
        return

    packets = load_uploaded_images(uploads)
    if len(packets) == 0:
        st.error("❌ 没有读取到有效图像文件")
        return

    st.markdown(f"### 📸 已上传 {len(packets)} 张图像")
    st.info(f"当前区域：{region_label}。请确保本次上传的图片都来自同一区域，并按你希望的拼接顺序整理。")
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
            st.session_state.pop("prepared_3d_asset", None)
            st.session_state.pop("pseudo_multiview_pack", None)
            st.session_state.pop("hunyuan_job_uid", None)
            st.session_state.pop("hunyuan_job_status", None)
            st.session_state.pop("hunyuan_job_message", None)
            st.session_state.pop("hunyuan_job_result_bytes", None)
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
                st.stop()

            if len(seg_results) == 0:
                st.error("❌ 没有成功分割任何图像，请检查模型配置和图像质量。")
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
                "metrics": st.session_state.seg_metrics
            }
            st.download_button(
                "⬇️ 下载分割报告 JSON",
                data=json.dumps(seg_report, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="segmentation_report.json",
                mime="application/json",
                width="stretch"
            )

        # ============ 步骤 4: 拼接 ==========
        st.divider()
        st.markdown("### 🦷 第 2 步: 牙齿主体拼接")

        if len(packets) < 2:
            st.warning("⚠️ 至少需要 2 张图像才能进行拼接")
            st.info("💡 请上传更多图像后重试")
            return

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
            st.warning("⚠️ 请选择至少 2 张图像参与拼接")
            return

        st.info(
            f"""
            **拼接说明**：
            - 当前区域：**{region_label}**
            - 当前使用 **{feature_method.upper()}** 特征匹配方法
            - 按你选择的顺序进行逐步拼接
            - 最终输出默认只保留牙齿主体区域
            - 即使低质量图像也会尽量尝试拼接，但会标记可信度、fallback 和降级原因
            """
        )

        col_stitch_btn, col_stitch_info = st.columns([1, 3])
        with col_stitch_btn:
            run_stitching = st.button("开始拼接", type="primary", width="stretch")

        if run_stitching:
            with st.spinner("🔄 正在拼接牙齿主体..."):
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
                st.session_state.pop("prepared_3d_asset", None)
                st.session_state.pop("pseudo_multiview_pack", None)
                st.session_state.pop("hunyuan_job_uid", None)
                st.session_state.pop("hunyuan_job_status", None)
                st.session_state.pop("hunyuan_job_message", None)
                st.session_state.pop("hunyuan_job_result_bytes", None)

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
                        st.success("✅ 拼接完成，当前结果可信度较高")
                    elif confidence_level == "medium":
                        st.warning("⚠️ 拼接完成，但当前结果为中等可信度，请重点检查边缘和重叠区域")
                    else:
                        st.warning("⚠️ 拼接已输出，但当前结果低可信，仅建议作为参考")

                    if not strict_teeth_only:
                        st.caption("当前结果包含 fallback full mask 输入，因此属于 degraded teeth-only 输出。")

                    # 显示标签
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
                        download_label = "⬇️ 下载去噪后拼接图" if enable_noise_removal else "⬇️ 下载牙齿拼接图"
                        filename = "stitched_teeth_cleaned.png" if enable_noise_removal else "stitched_teeth_only_v1.png"
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
                            st.download_button(
                                "⬇️ 下载原始拼接图（未去噪）",
                                data=_encode_png_bytes(outputs.stitched),
                                file_name="stitched_teeth_original.png",
                                mime="image/png",
                                width="stretch",
                            )

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
                    file_name="diagnostics_teeth_only_v1.json",
                    mime="application/json",
                    width="stretch",
                )
                st.caption(
                    f"分割来源: {diagnostic_payload.get('segmentation_source', 'pipeline_runtime')} | 输出模式: {diagnostic_payload.get('output_mode', 'unknown')}"
                )

            prepared_source = st.session_state.get("stitched_for_3d_image")
            prepared_mask = st.session_state.get("stitched_for_3d_mask")
            if prepared_source is not None and prepared_mask is not None:
                st.divider()
                st.markdown("### 🧊 第 3 步: Hunyuan3D Demo 输入准备")
                st.info(
                    """
                    当前 3D Demo 先接入单张完整牙弓图的预处理链路：
                    拼接结果 → 目标区域裁剪 → 方形画布归一化 → 3D-ready PNG。
                    下一步我们会在这之上补 pseudo-multiview 和 Hunyuan3D-2mv 推理。
                    """
                )

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
                            st.session_state.prepared_3d_asset = prepared_asset
                            st.session_state.pop("pseudo_multiview_pack", None)
                            st.session_state.pop("hunyuan_job_uid", None)
                            st.session_state.pop("hunyuan_job_status", None)
                            st.session_state.pop("hunyuan_job_message", None)
                            st.session_state.pop("hunyuan_job_result_bytes", None)
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
                    col_src, col_ready = st.columns(2)
                    with col_src:
                        st.markdown("#### 原始 3D 来源图")
                        st.image(bgr_to_rgb(resize_for_display(prepared_source)), width="stretch")
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
                            file_name="hunyuan3d_input.png",
                            mime="image/png",
                            width="stretch",
                        )
                    with dl_col2:
                        st.download_button(
                            "⬇️ 下载透明 PNG",
                            data=prepared_asset.png_bytes(transparent=True),
                            file_name="hunyuan3d_input_rgba.png",
                            mime="image/png",
                            width="stretch",
                        )
                    with dl_col3:
                        st.download_button(
                            "⬇️ 下载预处理元数据",
                            data=prepared_asset.metadata_bytes(),
                            file_name="hunyuan3d_input_metadata.json",
                            mime="application/json",
                            width="stretch",
                        )

                    st.markdown("#### Pseudo Multiview")
                    mv_btn_col, mv_info_col = st.columns([1, 2])
                    with mv_btn_col:
                        if st.button("生成伪多视图", key="build_pseudo_multiview", type="secondary", width="stretch"):
                            pseudo_pack = build_pseudo_multiview_pack(
                                prepared_asset,
                                include_back=False,
                                side_strength=0.10,
                            )
                            st.session_state.pseudo_multiview_pack = pseudo_pack
                            st.session_state.pop("hunyuan_job_uid", None)
                            st.session_state.pop("hunyuan_job_status", None)
                            st.session_state.pop("hunyuan_job_message", None)
                            st.session_state.pop("hunyuan_job_result_bytes", None)
                    with mv_info_col:
                        st.caption(
                            "当前默认生成 3 个视角：front / left / right。"
                            "它们来自单张牙弓图的透视扰动，只用于 demo 条件输入。"
                        )

                    pseudo_pack = st.session_state.get("pseudo_multiview_pack")
                    if pseudo_pack is not None:
                        st.image(bgr_to_rgb(pseudo_pack.preview_grid()), caption="伪多视图预览", width="stretch")
                        mv_dl1, mv_dl2 = st.columns(2)
                        with mv_dl1:
                            st.download_button(
                                "⬇️ 下载伪多视图 ZIP",
                                data=pseudo_pack.archive_bytes(transparent=False),
                                file_name="pseudo_multiview_views.zip",
                                mime="application/zip",
                                width="stretch",
                            )
                        with mv_dl2:
                            st.download_button(
                                "⬇️ 下载透明伪多视图 ZIP",
                                data=pseudo_pack.archive_bytes(transparent=True),
                                file_name="pseudo_multiview_views_rgba.zip",
                                mime="application/zip",
                                width="stretch",
                            )

                        st.markdown("#### Hunyuan3D-2mv 提交")
                        current_job_uid = st.session_state.get("hunyuan_job_uid")
                        current_job_status = st.session_state.get("hunyuan_job_status")
                        current_job_message = st.session_state.get("hunyuan_job_message")
                        result_bytes = st.session_state.get("hunyuan_job_result_bytes")

                        should_autopoll = bool(current_job_uid) and not result_bytes and current_job_status not in {"completed", "error"}
                        if should_autopoll:
                            st_autorefresh(interval=5000, key="hunyuan_mv_job_autorefresh")
                            try:
                                client = HunyuanServiceClient(service_config)
                                job_status = client.get_job_status(current_job_uid)
                                st.session_state.hunyuan_job_status = job_status.status
                                if job_status.model_bytes:
                                    st.session_state.hunyuan_job_result_bytes = job_status.model_bytes
                                    st.session_state.hunyuan_job_result_path = str(
                                        _persist_hunyuan_result(current_job_uid, job_status.model_bytes)
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
                            if st.button("提交到 Hunyuan3D-2mv", key="submit_hunyuan_mv", type="primary", width="stretch", disabled=submit_disabled):
                                try:
                                    client = HunyuanServiceClient(service_config)
                                    job_uid = client.submit_multiview_async(
                                        pseudo_pack.payload_images(transparent=True),
                                        seed=1234,
                                        num_inference_steps=24,
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
                                    st.error(f"❌ 提交 Hunyuan3D-2mv 任务失败：{exc}")
                        with refresh_col:
                            if st.button("刷新 3D 任务状态", key="refresh_hunyuan_mv_status", width="stretch"):
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
                                                _persist_hunyuan_result(job_uid, job_status.model_bytes)
                                            )
                                        if job_status.message:
                                            st.session_state.hunyuan_job_message = job_status.message
                                    except Exception as exc:
                                        st.error(f"❌ 查询 Hunyuan3D-2mv 状态失败：{exc}")

                        current_job_uid = st.session_state.get("hunyuan_job_uid")
                        current_job_status = st.session_state.get("hunyuan_job_status")
                        current_job_message = st.session_state.get("hunyuan_job_message")
                        current_job_result_path = st.session_state.get("hunyuan_job_result_path")
                        if current_job_uid:
                            st.caption(
                                f"当前任务 UID: {current_job_uid} | 状态: {current_job_status or 'submitted'}"
                            )
                        if submit_disabled:
                            st.caption("当前 3D 服务一次只处理一个任务，请等待本次任务完成或失败后再提交新的任务。")
                        if should_autopoll:
                            st.info("任务进行中，页面会每 5 秒自动轮询一次。")
                        if current_job_status == "error" and current_job_message:
                            st.error(f"❌ Hunyuan3D-2mv 任务失败：{current_job_message}")

                        result_bytes = st.session_state.get("hunyuan_job_result_bytes")
                        if not result_bytes and current_job_result_path:
                            result_path_obj = Path(current_job_result_path)
                            if result_path_obj.exists():
                                result_bytes = result_path_obj.read_bytes()
                                st.session_state.hunyuan_job_result_bytes = result_bytes
                        if result_bytes:
                            st.success("✅ 已收到 Hunyuan3D-2mv 输出 mesh。")
                            if current_job_result_path:
                                st.caption(f"结果已保存到: {current_job_result_path}")
                            st.markdown("#### GLB 在线预览")
                            _render_glb_preview(result_bytes)
                            st.download_button(
                                "⬇️ 下载 3D Mesh (GLB)",
                                data=result_bytes,
                                file_name="hunyuan3d_demo.glb",
                                mime="model/gltf-binary",
                                width="stretch",
                            )
    else:
        st.info("👆 点击上方「执行分割」按钮开始处理")

    st.divider()
    st.caption(
        """
        **使用提示**：
        1. 上传同一牙弓、同一侧段的口腔内窥镜图像
        2. 点击「执行分割」查看分割效果
        3. 确认分割效果满意后，点击「开始拼接」
        4. 下载牙齿主体拼接结果和诊断报告
        """
    )


if __name__ == "__main__":
    main()
