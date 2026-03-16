from __future__ import annotations

import json

import cv2
import streamlit as st
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

from dental_stitcher_v1.io_utils import bgr_to_rgb, load_uploaded_images, resize_for_display
from dental_stitcher_v1.pipeline import run_pipeline


st.set_page_config(page_title="口腔内窥镜拼接 v1", layout="wide")


def main() -> None:
    st.title("口腔内窥镜图像拼接 v1")
    st.caption("标准化流水线：牙齿分割 → 牙齿区域特征提取 → 配准 → 融合。")

    with st.sidebar:
        st.subheader("输入")
        feature_method = st.selectbox("特征方法", ["orb", "akaze", "sift", "loftr"], index=0)
        use_deep = st.checkbox("启用深度分割（若可用）", value=False)
        uploads = st.file_uploader(
            "上传多张口腔内窥镜图像",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        )
        run_clicked = st.button("开始拼接", type="primary", use_container_width=True)

        st.subheader("龋齿 ROI 采样")
        roi_idx = st.number_input("ROI 图像索引", min_value=0, value=0, step=1)
        roi_enabled = st.checkbox("启用 ROI 取点", value=False)

    if not uploads:
        st.info("请先上传图像。")
        return

    packets = load_uploaded_images(uploads)
    if len(packets) == 0:
        st.error("没有读取到有效图像。")
        return

    if roi_enabled and st_canvas is None:
        st.warning("缺少 streamlit-drawable-canvas，请先安装后再使用 ROI 取点。")

    if not run_clicked:
        st.image([bgr_to_rgb(resize_for_display(p.image)) for p in packets], caption=[p.name for p in packets])
        if roi_enabled and st_canvas is not None and 0 <= roi_idx < len(packets):
            st.markdown("### 点击两下选择龋齿 ROI（左上、右下）")
            base_image = resize_for_display(packets[int(roi_idx)].image)
            base_pil = Image.fromarray(bgr_to_rgb(base_image))
            try:
                canvas = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#ff0000",
                    background_image=base_pil,
                    height=base_pil.height,
                    width=base_pil.width,
                    drawing_mode="rect",
                    key="roi_canvas",
                )
                if canvas.json_data and canvas.json_data.get("objects"):
                    obj = canvas.json_data["objects"][-1]
                    x, y = float(obj.get("left", 0)), float(obj.get("top", 0))
                    w, h = float(obj.get("width", 0)), float(obj.get("height", 0))
                    st.info(f"ROI 坐标：x={int(x)}, y={int(y)}, w={int(w)}, h={int(h)}")
            except AttributeError:
                st.error("当前 streamlit 版本缺少 image_to_url，无法在图上取点。")
                st.markdown("请升级依赖后再试：")
                st.code("pip install -U streamlit streamlit-drawable-canvas", language="bash")
                st.markdown("临时替代方案：手动输入 ROI 坐标")
                x = st.number_input("ROI x", min_value=0, value=0, step=1)
                y = st.number_input("ROI y", min_value=0, value=0, step=1)
                w = st.number_input("ROI w", min_value=1, value=10, step=1)
                h = st.number_input("ROI h", min_value=1, value=10, step=1)
                st.info(f"ROI 坐标：x={int(x)}, y={int(y)}, w={int(w)}, h={int(h)}")
        return

    if len(packets) < 2:
        st.warning("建议至少上传两张图像以进行配准。")

    images = [p.image for p in packets]

    with st.spinner("处理中..."):
        outputs = run_pipeline(images, feature_method=feature_method, use_deep_segmentation=use_deep)

    result_col, diag_col = st.columns([1.3, 1.0], gap="large")

    with result_col:
        st.markdown("### 拼接结果")
        if outputs.stitched is None:
            st.error("拼接失败，请检查图像重叠和清晰度。")
        else:
            st.image(bgr_to_rgb(outputs.stitched), use_container_width=True)
            success, encoded = cv2.imencode(".png", outputs.stitched)
            if success:
                st.download_button(
                    "下载拼接图",
                    data=encoded.tobytes(),
                    file_name="stitched_v1.png",
                    mime="image/png",
                    use_container_width=True,
                )

        st.markdown("### 分割掩膜可视化")
        mask_items = outputs.diagnostics.per_image
        if mask_items:
            cols = st.columns(len(mask_items))
            for col, item in zip(cols, mask_items):
                idx = int(item["index"])
                if idx < len(outputs.mask_overlay):
                    overlay = outputs.mask_overlay[idx]
                else:
                    overlay = outputs.mask_overlay[0]
                with col:
                    st.image(bgr_to_rgb(resize_for_display(overlay)), caption=f"图像 {idx}", use_container_width=True)
        else:
            st.image(bgr_to_rgb(outputs.mask_overlay[0]), use_container_width=True)

        st.markdown("### 匹配可视化")
        st.image(outputs.match_visualization, use_container_width=True)

        st.markdown("### 多图质量指标")
        if outputs.diagnostics.per_image:
            st.dataframe(
                [{
                    "index": item["index"],
                    "sharpness": round(item["sharpness"], 2),
                    "exposure": round(item["exposure"], 2),
                    "mask_coverage": round(item["mask_coverage"], 3),
                    "segmentation_method": item["segmentation_method"],
                    "segmentation_fallback": item["segmentation_fallback"],
                } for item in outputs.diagnostics.per_image],
                use_container_width=True,
            )

    with diag_col:
        st.markdown("### 诊断 JSON")
        diagnostic_payload = outputs.diagnostics.to_dict()
        st.json(diagnostic_payload)
        st.download_button(
            "下载诊断 JSON",
            data=json.dumps(diagnostic_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="diagnostics_v1.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
