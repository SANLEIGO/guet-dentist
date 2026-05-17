import os
from PIL import Image

# ===================== 只需要改这里 =====================
FOLDER = "VOCdevkit-4/VOC2012/JPEGImages"
# ======================================================

# 遍历文件夹里所有文件
for filename in os.listdir(FOLDER):
    # 只处理 PNG 文件
    if filename.lower().endswith(".png"):
        png_path = os.path.join(FOLDER, filename)

        # 生成同名 jpg
        jpg_filename = os.path.splitext(filename)[0] + ".jpg"
        jpg_path = os.path(FOLDER, jpg_filename)

        try:
            with Image.open(png_path) as img:
                # 处理透明通道（必须）
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(jpg_path, "JPEG", quality=95)
            print(f"✅ 已转换: {filename} -> {jpg_filename}")
        except Exception as e:
            print(f"❌ 失败: {filename} | 错误: {e}")

print("\n🎉 全部转换完成！")
