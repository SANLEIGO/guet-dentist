from __future__ import annotations

import cv2
import json
import streamlit as st

from dental_stitcher.stitching import OralStitcher
from dental_stitcher.enhanced_stitching import CompatibleImprovedStitcher
from dental_stitcher.utils import (
    arch_display_name,
    bgr_to_rgb,
    load_uploaded_records,
    render_match_visualization,
    segment_display_name,
    segment_guidance,
)


st.set_page_config(
    page_title="口腔牙齿图像拼接",
    layout="wide",
)


def main() -> None:
    st.title("口腔牙齿图像拼接")
    st.caption("针对普通口腔内窥镜二维图像，按上牙/下牙分别拼接，自动选择最合适的基准图。")

    with st.sidebar:
        st.subheader("输入设置")
        arch_label = st.radio("选择拼接区域", ["上牙", "下牙"], horizontal=True)
        arch = "upper" if arch_label == "上牙" else "lower"
        segment_label = st.radio("选择拼接范围", ["左侧段", "右侧段", "完整牙弓"], horizontal=True)
        segment_mapping = {"左侧段": "left", "右侧段": "right", "完整牙弓": "full"}
        segment = segment_mapping[segment_label]

        # 添加算法选择选项
        algorithm = st.selectbox(
            "拼接算法",
            ["改进算法（推荐）", "原始算法"],
            help="改进算法提供更好的光照融合和几何校正效果"
        )

        # 添加可视化模式选项
        viz_mode = st.selectbox(
            "可视化模式",
            ["原始色彩（推荐）", "自动选择", "无缝融合", "边界高亮", "重叠区域"],
            help="原始色彩模式完全保持图像原有颜色，适合医学诊断"
        )

        uploads = st.file_uploader(
            "导入同一区域的多张口腔图像",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        )
        run_clicked = st.button("开始拼接", type="primary", use_container_width=True)

        st.markdown(
            """
            **建议采集方式**

            - 同一牙弓、同一侧段单独上传
            - 一次只拼接半侧牙弓，避免左右两侧混合
            - 相邻视角之间保留充分重叠
            - 尽量覆盖同一段牙列，减少跨区跨度过大
            """
        )

    if not uploads:
        st.info("先在左侧上传口腔图像，再选择上牙或下牙。")
        return

    # 根据选择创建不同的拼接器
    use_improved = algorithm == "改进算法（推荐）"

    # 可视化模式映射
    viz_mode_mapping = {
        "原始色彩（推荐）": "no_blend",
        "自动选择": "auto",
        "无缝融合": "无缝融合",
        "边界高亮": "边界高亮",
        "重叠区域": "重叠区域"
    }

    if use_improved:
        stitcher = CompatibleImprovedStitcher(viz_mode=viz_mode_mapping[viz_mode])
    else:
        stitcher = OralStitcher()

    guidance = segment_guidance(arch, segment)
    records = load_uploaded_records(uploads, arch, segment)
    if not records:
        st.error("没有读到有效图像文件。")
        return

    if len(records) >= 5 and segment != "full":
        st.warning("你当前上传的图像数量较多，若覆盖的是整排牙弓而不只是单侧，建议把\"拼接范围\"改为\"完整牙弓\"。")

    precheck = stitcher.precheck(records)
    kept_records = [records[idx] for idx in precheck.kept_indices]

    st.markdown("### 第 1 步: 导入与区域确认")
    info_left, info_right = st.columns([1.15, 1.45], gap="large")
    with info_left:
        st.write(f"当前区域: `{arch_display_name(arch)}`")
        st.write(f"当前范围: `{segment_display_name(segment)}`")
        st.write(f"目标分区: `{guidance['region']}`")
        st.write(f"上传数量: `{len(records)}` 张")
        for idx, record in enumerate(records, start=1):
            st.write(f"{idx}. {record.display_name}")
    with info_right:
        st.info(guidance["anchor_hint"])
        st.info(guidance["order_hint"])
        st.image(
            [bgr_to_rgb(record.image) for record in records[:4]],
            caption=[record.display_name for record in records[:4]],
            use_container_width=True,
        )

    st.markdown("### 第 2 步: 预检与筛图")
    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("保留图像", len(precheck.kept_indices))
    stat2.metric("建议剔除", len(precheck.dropped_indices))
    stat3.metric("当前范围", segment_label)

    for item in precheck.items:
        tag = "保留" if item.keep else "建议剔除"
        st.write(
            f"- `{item.display_name}` | 清晰度 `{item.sharpness_score:.1f}` | "
            f"曝光 `{item.exposure_score:.2f}` | 综合质量 `{item.quality_score:.2f}` | {tag} | {item.reason}"
        )

    use_filtered = st.checkbox("默认仅使用预检保留的图像", value=True)
    base_indices = precheck.kept_indices if use_filtered and len(kept_records) >= 2 else list(range(len(records)))
    if use_filtered and len(kept_records) < 2:
        st.warning("预检后不足两张图，已自动回退为使用全部上传图像。")

    st.markdown("### 第 2.5 步: 手动确认参与拼接图像")
    selected_indices: list[int] = []
    for idx, record in enumerate(records):
        default_value = idx in base_indices
        checked = st.checkbox(
            f"保留 `{record.display_name}`",
            value=default_value,
            key=f"keep_{record.display_name}_{idx}",
        )
        if checked:
            selected_indices.append(idx)

    active_records = [records[idx] for idx in selected_indices]
    if len(active_records) < 2:
        st.warning("当前手动保留的图像不足两张。")

    st.markdown("### 第 3 步: 基准图候选评分")
    if len(active_records) >= 2:
        with st.spinner("正在评估最合适的基准图..."):
            candidates, score_logs = stitcher.score_candidates(active_records)
        if candidates:
            top = candidates[0]
            st.success(f"推荐基准图: {top.display_name}")
            st.caption("优先选择位于当前半侧牙弓中央、与相邻图像重叠充分的视角作为基准。")
            st.caption("评分时已对上传顺序更近的图像对进行加权，减少远距离误匹配。")
            for candidate in candidates:
                prefix = "推荐" if candidate.recommended else "候选"
                st.write(
                    f"- {prefix} `{candidate.display_name}` | 连通得分 `{candidate.connectivity_score:.1f}` | "
                    f"连通图数 `{candidate.partner_count}` | 图像质量 `{candidate.quality_score:.2f}` | "
                    f"总分 `{candidate.total_score:.2f}`"
                )
        else:
            score_logs = ["没有生成有效候选评分。"]
            st.warning("当前图像集没有得到可靠的基准图候选。")
    else:
        candidates, score_logs = [], ["可用图像不足，无法评估基准图。"]
        st.warning("至少需要两张有效图像才能评估基准图。")

    st.markdown("### 第 3.5 步: 基准图确认")
    anchor_mode = st.radio("基准图模式", ["自动选择", "手动指定"], horizontal=True)
    anchor_index_override = None
    if anchor_mode == "手动指定":
        if active_records:
            anchor_options = {record.display_name: idx for idx, record in enumerate(active_records)}
            chosen_name = st.selectbox("选择基准图", list(anchor_options.keys()))
            anchor_index_override = anchor_options[chosen_name]
        else:
            st.warning("当前没有可用于手动指定的图像。")

    st.markdown("### 第 4 步: 执行拼接")
    if not run_clicked:
        st.info("确认筛图结果后，点击左侧开始拼接。")
        return

    if len(active_records) < 2:
        st.error("可用于拼接的图像不足两张。")
        return

    with st.spinner("正在进行特征匹配与拼接..."):
        # 改进算法和原始算法都使用相同的接口
        result = stitcher.stitch(active_records, anchor_index_override=anchor_index_override)

    result_col, log_col = st.columns([1.45, 1.0], gap="large")
    with result_col:
        st.write(f"匹配方法: `{result.method_name}`")
        if not result.success or result.panorama is None:
            st.error("拼接失败。请检查图像是否属于同一牙弓局部、是否有足够重叠、是否过度模糊。")
        else:
            anchor_name = active_records[result.anchor_index].display_name if result.anchor_index is not None else "未知"
            included_names = [active_records[idx].display_name for idx in result.included_indices]
            ordered_names = [active_records[idx].display_name for idx in result.ordered_indices]
            st.success(f"拼接完成，自动基准图: {anchor_name}")
            st.write("参与拼接图像: " + ", ".join(included_names))
            st.write("链式优先顺序: " + " -> ".join(ordered_names))
            st.image(bgr_to_rgb(result.panorama), caption="拼接结果", use_container_width=True)
            success, encoded = cv2.imencode(".png", result.panorama)
            if success:
                st.download_button(
                    "下载拼接结果 PNG",
                    data=encoded.tobytes(),
                    file_name=f"{arch}_{segment}_stitched.png",
                    mime="image/png",
                    use_container_width=True,
                )

            report_rows = []
            for (i, j), match in result.pairwise_matches.items():
                report_rows.append(
                    {
                        "image_a": active_records[i].display_name,
                        "image_b": active_records[j].display_name,
                        "success": match.success,
                        "inliers": match.inliers,
                        "raw_score": round(float(match.score), 3),
                        "weighted_score": round(float(match.weighted_score), 3),
                        "sequence_distance": int(match.sequence_distance),
                        "reason": match.details.get("reason", ""),
                    }
                )
            report_payload = {
                "arch": arch,
                "segment": segment,
                "anchor_image": anchor_name,
                "method": result.method_name,
                "included_images": included_names,
                "ordered_images": ordered_names,
                "pairwise_scores": report_rows,
                "logs": result.logs,
            }
            st.download_button(
                "下载诊断报告 JSON",
                data=json.dumps(report_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{arch}_{segment}_report.json",
                mime="application/json",
                use_container_width=True,
            )

            st.markdown("#### 诊断视图")
            if len(result.ordered_indices) > 1:
                diagnostic_targets = {
                    active_records[idx].display_name: idx
                    for idx in result.ordered_indices
                    if idx != result.anchor_index
                }
                selected_target_name = st.selectbox(
                    "查看与基准图的匹配可视化",
                    list(diagnostic_targets.keys()),
                )
                target_idx = diagnostic_targets[selected_target_name]
                pair_key = (min(result.anchor_index, target_idx), max(result.anchor_index, target_idx))
                pair_match = result.pairwise_matches.get(pair_key)
                if pair_match and pair_match.success:
                    vis = render_match_visualization(
                        active_records[result.anchor_index].image,
                        active_records[target_idx].image,
                        pair_match,
                    )
                    st.image(vis, caption=f"{anchor_name} vs {selected_target_name}", use_container_width=True)
                    st.caption(
                        f"内点: {pair_match.inliers} | 原始得分: {pair_match.score:.1f} | "
                        f"加权得分: {pair_match.weighted_score:.1f} | 顺序距离: {pair_match.sequence_distance}"
                    )
                else:
                    st.warning("该图像与基准图之间没有成功匹配，无法显示连线诊断。")

            with st.expander("两两评分明细", expanded=False):
                st.dataframe(report_rows, use_container_width=True)

    with log_col:
        with st.expander("基准图评分日志", expanded=False):
            for line in score_logs:
                st.text(line)
        with st.expander("拼接处理日志", expanded=True):
            for line in result.logs:
                st.text(line)


if __name__ == "__main__":
    main()
