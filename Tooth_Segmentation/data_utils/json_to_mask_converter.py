import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

def create_palette(num_cls=256):
    """创建调色板（你提供的原版）"""
    palette = [0] * (num_cls * 3)
    for j in range(num_cls):
        lab = j
        for i in range(8):
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            lab >>= 3
    return palette

def json_to_mask(json_path, save_path):
    """
    单个JSON转二分类mask
    0 = 背景
    1 = 牙齿
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图片宽高
    w = data['size']['width']
    h = data['size']['height']

    # 创建全0背景（单通道）
    mask = np.zeros((h, w), dtype=np.uint8)

    # 遍历所有牙齿多边形
    for obj in data['objects']:
        if obj['classTitle'] != 'Tooth':
            continue

        # 获取轮廓点
        points = obj['points']['exterior']
        if not points:
            continue

        # 转成PIL需要的格式：[(x1,y1), (x2,y2)...]
        polygon = [(int(p[0]), int(p[1])) for p in points]

        # 快速绘制多边形（内部自动填充为1）
        from PIL import ImageDraw
        temp = Image.fromarray(mask)
        draw = ImageDraw.Draw(temp)
        draw.polygon(polygon, outline=1, fill=1)
        mask = np.array(temp)

    # 保存为调色板PNG
    out_img = Image.fromarray(mask)
    out_img.putpalette(create_palette())
    out_img.save(save_path, format='PNG')

    print(f"✅ 生成掩码：{save_path}")

def batch_process_folder(json_folder, save_folder):
    """批量处理整个文件夹"""
    os.makedirs(save_folder, exist_ok=True)
    json_files = list(Path(json_folder).glob("*.json"))

    for json_file in json_files:
        save_name = json_file.stem + ".png"
        save_path = os.path.join(save_folder, save_name)
        json_to_mask(str(json_file), save_path)

    print(f"\n🎉 全部处理完成！共 {len(json_files)} 张")

# ====================== 使用方法 ======================
if __name__ == "__main__":
    # 你的JSON文件夹
    JSON_FOLDER = "dentalai-DatasetNinja/valid/ann"  # 改成你的路径
    # 掩码保存文件夹
    SAVE_FOLDER = "dentalai-DatasetNinja/valid/label"

    batch_process_folder(JSON_FOLDER, SAVE_FOLDER)
