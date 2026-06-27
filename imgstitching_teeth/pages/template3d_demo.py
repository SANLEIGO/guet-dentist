from __future__ import annotations

import html
import json

import streamlit as st
import streamlit.components.v1 as components

from dental_stitcher_v1.io_utils import bgr_to_rgb, load_uploaded_images, resize_for_display
from dental_stitcher_v1.template3d import (
    ArchLabel,
    analyze_panorama_image,
    build_template_render_state,
    detect_caries,
    detections_to_feature_observations,
    draw_panorama_analysis_overlay,
    get_default_dental_arch_asset,
    load_model_data_uri,
    observations_to_resolved_features,
    overlay_caries_detections,
    read_asset_license,
    remap_panorama_analysis,
)
from dental_stitcher_v1.template3d.render_state import FEATURE_COLORS, FEATURE_LABELS
from dental_stitcher_v1.template3d.sample_cases import build_demo_features
from dental_stitcher_v1.template3d.schema import (
    FeatureType,
    ToothRenderState,
)
from dental_stitcher_v1.template3d.tooth_map import build_adult_template_teeth


st.set_page_config(page_title="三维牙弓模板演示", page_icon="D", layout="wide")


ARCH_OPTIONS = {
    "上牙弓": ArchLabel.UPPER,
    "下牙弓": ArchLabel.LOWER,
}

SCENARIO_OPTIONS = {
    "示例异常特征": "demo",
    "空白标准模板": "blank",
}

WORKFLOW_OPTIONS = {
    "单张全景自动识别": "panorama",
    "手动演示 / 局部绑定": "manual",
}


def main() -> None:
    st.title("牙齿特征识别与三维模板可视化原型")
    st.caption("这个页面演示新版方向：照片负责提供牙齿异常证据，三维标准牙弓模板负责展示结果。")

    controls, preview = st.columns([0.92, 1.75], gap="large")
    with controls:
        st.subheader("演示设置")
        workflow = WORKFLOW_OPTIONS[
            st.radio("工作流", list(WORKFLOW_OPTIONS.keys()), horizontal=True)
        ]
        if workflow == "panorama":
            arch_label, features, selected_tooth_id = _render_panorama_workflow_controls()
        else:
            arch_label, features, selected_tooth_id = _render_manual_workflow_controls()
        render_state = build_template_render_state(arch_label, features)

        _render_summary(render_state)
        st.download_button(
            "下载模板状态 JSON",
            data=json.dumps(render_state.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{render_state.template_id}_state.json",
            mime="application/json",
            width="stretch",
        )

    with preview:
        st.subheader("三维模板效果预览")
        _render_template_preview(render_state.teeth, arch_label, selected_tooth_id)

    detail, evidence = st.columns([1, 1], gap="large")
    selected_tooth = next(tooth for tooth in render_state.teeth if tooth.tooth_id == selected_tooth_id)
    with detail:
        st.subheader(f"FDI {selected_tooth_id} 牙位状态")
        _render_tooth_detail(selected_tooth)
    with evidence:
        st.subheader("证据来源")
        _render_evidence_panel(selected_tooth, arch_label)

    with st.expander("查看完整结构化结果", expanded=False):
        st.json(render_state.to_dict())


def _render_panorama_workflow_controls() -> tuple[ArchLabel, list, int]:
    record = st.session_state.get("template3d_panorama_record")
    default_arch = record["analysis"].arch_label if record else ArchLabel.UPPER
    arch_label = ARCH_OPTIONS[
        st.radio(
            "人工指定牙弓",
            list(ARCH_OPTIONS.keys()),
            index=_arch_option_index(default_arch),
            horizontal=True,
            key="panorama_manual_arch",
        )
    ]
    st.caption("上传一张全景/整段牙弓照片后，系统会按你指定的牙弓识别龋齿候选和缺牙候选。")
    mode_label = st.selectbox(
        "龋齿检测模式",
        ["高敏感复核", "平衡检测", "极高敏感复核"],
        index=0,
        help="自动流程默认走高敏感候选，结果必须人工确认。",
    )
    caries_mode, default_threshold = _caries_mode_config(mode_label)
    caries_threshold = st.slider(
        "龋齿检测阈值",
        0.03,
        0.80,
        default_threshold,
        0.01,
        key="panorama_caries_threshold",
    )
    missing_label = st.selectbox(
        "缺牙候选敏感度",
        ["标准", "保守", "敏感"],
        index=0,
        help="缺牙当前是基于牙齿亮区槽位的启发式估计，建议先用标准或保守。",
    )
    missing_sensitivity = _missing_sensitivity_value(missing_label)
    upload = st.file_uploader(
        "上传单张全景图",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=False,
        key="single_panorama_upload",
    )

    if upload is not None:
        packets = load_uploaded_images([upload])
        if packets:
            packet = packets[0]
            st.image(
                bgr_to_rgb(resize_for_display(packet.image, max_width=430, max_height=240)),
                caption=f"待分析：{packet.name}",
                width="stretch",
            )
            if st.button("识别龋齿和缺牙候选", type="primary", width="stretch"):
                with st.spinner("正在尝试分析单张全景图..."):
                    image_id = _image_id_from_name(packet.name)
                    analysis = analyze_panorama_image(
                        packet.image,
                        image_id=image_id,
                        arch_override=arch_label,
                        caries_conf_threshold=caries_threshold,
                        caries_mode=caries_mode,
                        missing_sensitivity=missing_sensitivity,
                    )
                st.session_state.template3d_panorama_record = {
                    "name": packet.name,
                    "image": packet.image,
                    "analysis": analysis,
                    "mode": caries_mode,
                    "caries_threshold": caries_threshold,
                    "missing_label": missing_label,
                    "missing_sensitivity": missing_sensitivity,
                }
                st.rerun()
        else:
            st.warning("没有读取到有效图片。")

    record = st.session_state.get("template3d_panorama_record")
    if not record:
        tooth_ids = [tooth.tooth_id for tooth in build_adult_template_teeth(arch_label)]
        selected_tooth_id = st.selectbox(
            "人工复核牙位",
            tooth_ids,
            format_func=lambda tooth_id: f"FDI {tooth_id}",
            key=f"panorama_placeholder_tooth_{arch_label.value}",
        )
        st.info("上传并运行分析后，这里会显示龋齿、缺牙候选结果。")
        return arch_label, [], selected_tooth_id

    analysis = record["analysis"]
    if arch_label != analysis.arch_label:
        analysis = remap_panorama_analysis(analysis, arch_label)
        record["analysis"] = analysis
        st.session_state.template3d_panorama_record = record
    st.success(f"已按人工指定的{_arch_display_name(analysis.arch_label)}进行牙位映射。")

    if analysis.notes:
        for note in analysis.notes:
            st.caption(note)

    overlay = draw_panorama_analysis_overlay(record["image"], analysis)
    st.image(
        bgr_to_rgb(resize_for_display(overlay, max_width=430, max_height=280)),
        caption=(
            f"{record.get('name', 'panorama')} | "
            f"龋齿候选 {len(analysis.caries_result.detections)} | "
            f"缺牙候选 {len(analysis.missing_slot_indices)}"
        ),
        width="stretch",
    )

    if analysis.error:
        st.warning(analysis.error)

    with st.expander("查看牙位槽位评分", expanded=False):
        st.dataframe(
            [
                {
                    "FDI": slot.tooth_id,
                    "槽位": slot.slot_index + 1,
                    "可见度": round(slot.occupancy_score, 3),
                    "缺牙候选": "是" if slot.is_missing_candidate else "否",
                }
                for slot in analysis.tooth_slots
            ],
            use_container_width=True,
            hide_index=True,
        )

    clear_col, export_col = st.columns([1, 1])
    with clear_col:
        if st.button("清除全景分析", width="stretch"):
            st.session_state.pop("template3d_panorama_record", None)
            st.rerun()
    with export_col:
        st.download_button(
            "下载全景分析 JSON",
            data=json.dumps(_panorama_analysis_to_dict(analysis), ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{analysis.image_id}_panorama_analysis.json",
            mime="application/json",
            width="stretch",
        )

    tooth_ids = [tooth.tooth_id for tooth in build_adult_template_teeth(analysis.arch_label)]
    default_tooth = _first_feature_tooth_id(analysis.features, tooth_ids[0])
    selected_tooth_id = st.selectbox(
        "人工复核牙位",
        tooth_ids,
        index=tooth_ids.index(default_tooth),
        format_func=lambda tooth_id: f"FDI {tooth_id}",
        key=f"panorama_review_tooth_{analysis.arch_label.value}",
    )
    return analysis.arch_label, list(analysis.features), selected_tooth_id


def _render_manual_workflow_controls() -> tuple[ArchLabel, list, int]:
    arch_label = ARCH_OPTIONS[
        st.radio("牙弓", list(ARCH_OPTIONS.keys()), horizontal=True, label_visibility="collapsed")
    ]
    scenario = SCENARIO_OPTIONS[
        st.selectbox("场景", list(SCENARIO_OPTIONS.keys()), index=0)
    ]

    base_features = build_demo_features(arch_label, scenario)
    tooth_ids = [tooth.tooth_id for tooth in build_adult_template_teeth(arch_label)]
    default_tooth = base_features[0].tooth_id if base_features else tooth_ids[0]
    selected_tooth_id = st.selectbox(
        "选中牙位",
        tooth_ids,
        index=tooth_ids.index(default_tooth),
        format_func=lambda tooth_id: f"FDI {tooth_id}",
    )

    detected_features = _render_caries_detection_panel(arch_label, selected_tooth_id)
    return arch_label, base_features + detected_features, selected_tooth_id


def _render_caries_detection_panel(arch_label: ArchLabel, selected_tooth_id: int) -> list:
    st.subheader("龋病模型检测")
    st.caption("当前先把检测框绑定到选中的 FDI 牙位；后续可接扫描顺序追踪自动推牙号。")
    mode_label = st.selectbox(
        "检测模式",
        ["高敏感复核", "平衡检测", "极高敏感复核"],
        index=0,
        help="不明显龋齿建议先用高敏感复核；极高敏感会增加误报，需要人工确认。",
    )
    mode, default_threshold = _caries_mode_config(mode_label)
    upload = st.file_uploader(
        "上传局部口腔照片",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=False,
    )
    conf_threshold = st.slider("检测阈值", 0.03, 0.80, default_threshold, 0.01)

    if upload is not None:
        packets = load_uploaded_images([upload])
        if packets:
            packet = packets[0]
            st.image(
                bgr_to_rgb(resize_for_display(packet.image, max_width=430, max_height=240)),
                caption=f"待检测：{packet.name}",
                width="stretch",
            )
            if st.button("运行龋病检测", type="primary", width="stretch"):
                with st.spinner("正在加载并运行 YOLOv8 龋病模型..."):
                    result = detect_caries(packet.image, conf_threshold=conf_threshold, mode=mode)
                image_id = _image_id_from_name(packet.name)
                if not result.success:
                    st.session_state.template3d_caries_record = {
                        "arch_label": arch_label.value,
                        "tooth_id": selected_tooth_id,
                        "features": [],
                        "detections": [],
                        "error": result.error,
                    }
                    st.error(result.error)
                else:
                    observations = detections_to_feature_observations(
                        result,
                        arch_label=arch_label,
                        image_id=image_id,
                        candidate_tooth_ids=[selected_tooth_id],
                    )
                    resolved_features = observations_to_resolved_features(observations, selected_tooth_id)
                    st.session_state.template3d_caries_record = {
                        "arch_label": arch_label.value,
                        "tooth_id": selected_tooth_id,
                        "features": resolved_features,
                        "detections": result.detections,
                        "overlay": overlay_caries_detections(packet.image, result),
                        "image_id": image_id,
                        "model_path": result.model_path,
                        "mode": result.mode,
                        "threshold": conf_threshold,
                        "error": None,
                    }
                    st.rerun()
        else:
            st.warning("没有读取到有效图片。")

    record = st.session_state.get("template3d_caries_record")
    if not record or record.get("arch_label") != arch_label.value:
        return []

    if record.get("error"):
        st.warning(record["error"])
        return []

    detections = record.get("detections", [])
    if "overlay" in record:
        st.image(
            bgr_to_rgb(resize_for_display(record["overlay"], max_width=430, max_height=260)),
            caption=f"检测结果：{len(detections)} 个疑似区域，绑定 FDI {record.get('tooth_id')}",
            width="stretch",
        )
    if detections:
        st.caption(f"模型：{record.get('model_path')} | 模式：{record.get('mode')} | 阈值：{record.get('threshold')}")
        if record.get("mode") in {"sensitive", "review"}:
            st.warning("高敏感结果更适合做候选提示，请结合原图人工复核后再确认。")
    else:
        st.info("当前图片没有检测到高于阈值的龋病疑似区域。")

    clear_col, bind_col = st.columns([1, 1])
    with clear_col:
        if st.button("清除检测结果", width="stretch"):
            st.session_state.pop("template3d_caries_record", None)
            st.rerun()
    with bind_col:
        if record.get("tooth_id") != selected_tooth_id and st.button("绑定到当前牙位", width="stretch"):
            features = record.get("features", [])
            for feature in features:
                feature.tooth_id = selected_tooth_id
                feature.notes = "模型已检测到疑似龋病区域；当前原型需人工确认牙位绑定。"
            record["tooth_id"] = selected_tooth_id
            st.session_state.template3d_caries_record = record
            st.rerun()

    return list(record.get("features", []))


def _caries_mode_config(label: str) -> tuple[str, float]:
    if label == "平衡检测":
        return "balanced", 0.25
    if label == "极高敏感复核":
        return "review", 0.05
    return "sensitive", 0.10


def _missing_sensitivity_value(label: str) -> float:
    return {
        "保守": 0.08,
        "标准": 0.14,
        "敏感": 0.24,
    }.get(label, 0.14)


def _arch_display_name(arch_label: ArchLabel) -> str:
    return {
        ArchLabel.UPPER: "上牙弓",
        ArchLabel.LOWER: "下牙弓",
        ArchLabel.UNKNOWN: "未知",
    }.get(arch_label, "未知")


def _arch_option_index(arch_label: ArchLabel) -> int:
    return 1 if arch_label == ArchLabel.LOWER else 0


def _first_feature_tooth_id(features: list, fallback: int) -> int:
    for feature in features:
        return feature.tooth_id
    return fallback


def _panorama_analysis_to_dict(analysis) -> dict:
    return {
        "image_id": analysis.image_id,
        "predicted_arch": {
            "arch_label": analysis.predicted_arch.arch_label.value,
            "confidence": analysis.predicted_arch.confidence,
            "status_text": analysis.predicted_arch.status_text,
            "metrics": analysis.predicted_arch.metrics,
        },
        "confirmed_arch": analysis.arch_label.value,
        "notes": analysis.notes,
        "tooth_bbox_xyxy": analysis.tooth_bbox_xyxy,
        "slot_scores": analysis.slot_scores,
        "missing_slot_indices": analysis.missing_slot_indices,
        "tooth_slots": [
            {
                "slot_index": slot.slot_index,
                "tooth_id": slot.tooth_id,
                "x0": slot.x0,
                "x1": slot.x1,
                "occupancy_score": slot.occupancy_score,
                "missing_confidence": slot.missing_confidence,
                "is_missing_candidate": slot.is_missing_candidate,
            }
            for slot in analysis.tooth_slots
        ],
        "caries_detections": [
            {
                "bbox_xyxy": detection.bbox_xyxy,
                "confidence": detection.confidence,
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "source": detection.source,
            }
            for detection in analysis.caries_result.detections
        ],
        "features": [feature.to_dict() for feature in analysis.features],
        "error": analysis.error,
    }


def _render_summary(render_state) -> None:
    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        st.metric("异常特征", render_state.evidence_summary["feature_count"])
    with metric_b:
        st.metric("涉及牙位", render_state.evidence_summary["affected_tooth_count"])
    with metric_c:
        st.metric("需复核", render_state.evidence_summary["review_required_count"])

    legend_rows = []
    for feature_type, label in FEATURE_LABELS.items():
        color = FEATURE_COLORS.get(feature_type, "#777777")
        legend_rows.append(
            f'<span class="legend-item"><span style="background:{color}"></span>{html.escape(label)}</span>'
        )
    st.markdown(
        """
        <style>
        .legend-wrap {display:flex; flex-wrap:wrap; gap:7px 10px; margin-top: 6px;}
        .legend-item {font-size:12px; color:#384047; white-space:nowrap;}
        .legend-item span {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:-1px;}
        </style>
        <div class="legend-wrap">""" + "".join(legend_rows) + "</div>",
        unsafe_allow_html=True,
    )


def _render_template_preview(teeth: list[ToothRenderState], arch_label: ArchLabel, selected_tooth_id: int) -> None:
    asset = get_default_dental_arch_asset()
    if not asset.exists:
        st.error(f"没有找到三维牙弓模型：{asset.model_path}")
        return

    model_src_attr = html.escape(load_model_data_uri(str(asset.model_path)), quote=True)
    asset_title = html.escape(asset.title)
    source_label = html.escape(f"{asset.title} by {asset.author} / {asset.license_name}")
    asset_note = html.escape("当前只加载原始 28 牙牙弓模型；龋齿高亮、缺牙隐藏、智齿逻辑已暂时停用。")

    components.html(
        f"""
        <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        <style>
          .real-model-shell {{
            height: 662px;
            position: relative;
            overflow: hidden;
            border-radius: 18px;
            border: 1px solid rgba(34, 41, 46, 0.14);
            background:
              radial-gradient(circle at 20% 15%, rgba(255, 244, 220, 0.88), transparent 28%),
              radial-gradient(circle at 83% 22%, rgba(184, 211, 206, 0.62), transparent 34%),
              linear-gradient(135deg, #f2eadb 0%, #e9eee8 48%, #d8e3e1 100%);
            box-shadow: 0 24px 70px rgba(37, 49, 56, 0.14);
            font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          model-viewer {{
            width: 100%;
            height: 100%;
            --poster-color: transparent;
          }}
          .model-note {{
            position: absolute;
            right: 20px;
            top: 20px;
            z-index: 4;
            width: min(340px, 42%);
            padding: 14px 16px;
            border-radius: 15px;
            color: #263238;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(39, 50, 56, 0.12);
            backdrop-filter: blur(14px);
            box-shadow: 0 18px 42px rgba(36, 48, 55, 0.14);
            font-size: 11px;
            line-height: 1.42;
          }}
          .model-credit {{
            position: absolute;
            left: 24px;
            bottom: 20px;
            z-index: 4;
            color: rgba(43, 52, 58, 0.74);
            font-size: 11px;
            padding: 7px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.64);
            border: 1px solid rgba(35, 45, 51, 0.10);
          }}
        </style>
        <div class="real-model-shell">
          <div class="model-note">{asset_note}</div>
          <model-viewer
            src="{model_src_attr}"
            alt="{asset_title}"
            camera-controls
            interaction-prompt="none"
            shadow-intensity="0.72"
            exposure="1.02"
            environment-image="neutral"
            camera-target="0m 3.62m 0.95m"
            camera-orbit="0deg 88deg 120%"
            field-of-view="24deg">
          </model-viewer>
          <div class="model-credit">{source_label}</div>
        </div>
        """,
        height=684,
    )

    with st.expander("模型资产与授权", expanded=False):
        st.json(asset.to_dict())
        license_text = read_asset_license(asset)
        if license_text:
            st.code(license_text, language="text")


def _render_tooth_detail(tooth: ToothRenderState) -> None:
    if not tooth.features:
        st.info("当前牙位没有异常特征。")
        st.write({"tooth_id": tooth.tooth_id, "visible": tooth.visible, "status": "normal"})
        return

    for feature in tooth.features:
        st.markdown(f"**{FEATURE_LABELS.get(feature.feature_type, '异常')}**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("置信度", f"{feature.confidence:.0%}")
        with col_b:
            st.metric("严重度", feature.severity.value)
        with col_c:
            st.metric("牙面", feature.surface.value)
        if feature.review_required:
            st.warning("该特征需要人工复核。")
        if feature.notes:
            st.caption(feature.notes)


def _render_evidence_panel(tooth: ToothRenderState, arch_label: ArchLabel) -> None:
    evidence_ids = []
    for feature in tooth.features:
        evidence_ids.extend(feature.evidence_image_ids)
    if evidence_ids:
        for evidence_id in dict.fromkeys(evidence_ids):
            st.code(evidence_id, language="text")
    else:
        st.info("没有绑定异常证据。")

    packets = _captured_packets_for_arch(arch_label)
    if packets:
        st.caption("当前拍照页已有采集图，可作为后续真实证据来源。")
        cols = st.columns(min(len(packets), 3))
        for col, packet in zip(cols, packets[:3]):
            with col:
                st.image(
                    bgr_to_rgb(resize_for_display(packet.image, max_width=320, max_height=220)),
                    caption=packet.name,
                    width="stretch",
                )


def _captured_packets_for_arch(arch_label: ArchLabel) -> list:
    if arch_label == ArchLabel.UPPER:
        return list(st.session_state.get("upper_arch_images", []))
    if arch_label == ArchLabel.LOWER:
        return list(st.session_state.get("lower_arch_images", []))
    return []


def _image_id_from_name(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    return safe or "uploaded_image"


def _first_affected_tooth(teeth: list[ToothRenderState]) -> int | None:
    for tooth in teeth:
        if tooth.features:
            return tooth.tooth_id
    return None


if __name__ == "__main__":
    main()
