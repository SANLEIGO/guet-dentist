import os
import shutil

# ===================== 你只需要改这 3 个路径 =====================
# 1. 存放文件名的 txt 文件路径
TXT_PATH = "../VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"

# 2. 源文件夹（要从这里提取图片）
SOURCE_DIR = "../VOCdevkit/VOC2012/JPEGImages/"

# 3. 目标文件夹（提取出来的图片放这里）
TARGET_DIR = "../val_images"
# =================================================================

# 自动创建目标文件夹
os.makedirs(TARGET_DIR, exist_ok=True)

# 读取所有需要提取的文件名（不带后缀）
with open(TXT_PATH, "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f if line.strip()]

print(f"✅ 从 txt 中读取到 {len(filenames)} 个文件名")

# 开始提取
count = 0
not_found = []

for name in filenames:
    # 拼接成完整的 .jpg 路径
    src_file = os.path.join(SOURCE_DIR, f"{name}.jpg")

    if os.path.exists(src_file):
        # 复制文件（推荐，安全）
        shutil.copy2(src_file, TARGET_DIR)

        # 如果你想【移动】而不是复制，用下面这行（会从原文件夹删掉）
        # shutil.move(src_file, TARGET_DIR)

        count += 1
    else:
        not_found.append(name)

# 输出结果
print(f"\n🎉 提取完成！成功提取 {count} 张图片")

if not_found:
    print(f"\n⚠️  未找到 {len(not_found)} 个文件：")
    for n in not_found[:10]:  # 只显示前10个避免刷屏
        print(f"  {n}.jpg")
