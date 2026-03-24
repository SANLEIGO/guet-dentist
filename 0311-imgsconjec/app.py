from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from dental_stitcher_v1.io_utils import bgr_to_rgb, load_uploaded_images, resize_for_display
from dental_stitcher_v1.pipeline import run_pipeline
from dental_stitcher_v1.segmentation import segment_teeth, fallback_full_mask


st.set_page_config(page_title="口腔牙齿提取拼接 v1", layout="wide")


def main() -> None:
    st.title("口腔牙齿提取拼接 v1")
    st.caption("标准化流水线：牙齿分割 → 牙齿区域特征提取 → 配准 → 仅牙齿区域融合。")

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
            seg_conf = st.slider(
                "AlphaDent 置信度阈值",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="越低越敏感，越高越严格"
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

            st.subheader("🔧 拼接设置")
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
        st.info(
            """
            **分割说明**：
            - 使用 AlphaDent (YOLOv8) 进行牙齿区域检测
            - 可选 GrabCut 精细化分割边界
            - 绿色覆盖区域表示识别为牙齿的部分
            - 后续拼接只会保留分割出的牙齿主体区域
            """
        )

    if run_segmentation:
        with st.spinner("🔄 正在分割图像..."):
            seg_results = []
            seg_metrics = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, packet in enumerate(packets):
                status_text.text(f"正在处理第 {idx+1}/{len(packets)} 张图像: {packet.name}")
                progress_bar.progress((idx + 1) / len(packets))

                seg_result = segment_teeth(packet.image)

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
                )

            result_col, diag_col = st.columns([1.3, 1.0], gap="large")

            with result_col:
                st.markdown("#### 🖼️ 牙齿提取拼接结果")
                quality_gate = outputs.diagnostics.quality_gate or {}
                confidence_level = quality_gate.get("confidence_level", "unknown")
                strict_teeth_only = quality_gate.get("strict_teeth_only", True)
                if outputs.stitched is None:
                    st.error("❌ 拼接失败，请检查图像重叠和清晰度")
                else:
                    if confidence_level == "high":
                        st.success("✅ 拼接完成，当前结果可信度较高")
                    elif confidence_level == "medium":
                        st.warning("⚠️ 拼接完成，但当前结果为中等可信度，请重点检查边缘和重叠区域")
                    else:
                        st.warning("⚠️ 拼接已输出，但当前结果低可信，仅建议作为参考")

                    if not strict_teeth_only:
                        st.caption("当前结果包含 fallback full mask 输入，因此属于 degraded teeth-only 输出。")

                    st.image(bgr_to_rgb(outputs.stitched), width="stretch")
                    success, encoded = cv2.imencode(".png", outputs.stitched)
                    if success:
                        st.download_button(
                            "⬇️ 下载牙齿拼接图",
                            data=encoded.tobytes(),
                            file_name="stitched_teeth_only_v1.png",
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
