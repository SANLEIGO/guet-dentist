import os
import json
import glob
import numpy as np
from PIL import Image

# 数据集路径
SPLIT_IMAGE_DIR = 'd:\\Python-Scripts\\U-Net\\splitimage'
SPLIT_LABEL_DIR = 'd:\\Python-Scripts\\U-Net\\splitlabel'
OUTPUT_MASK_DIR = 'd:\\Python-Scripts\\U-Net\\binary_splitmask'

# 二分类映射
BACKGROUND_CLASS = 0
TOOTH_CLASS = 1


def create_palette(num_cls=256):
    """创建调色板"""
    palette = [0] * (num_cls * 3)
    for j in range(num_cls):
        lab = j
        for i in range(8):
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            lab >>= 3
    return palette


def polygon_to_mask(polygon, width, height):
    """将多边形坐标转换为掩码"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 使用PIL的ImageDraw填充多边形
    from PIL import ImageDraw
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # 确保坐标是整数
    polygon = [(int(x), int(y)) for x, y in polygon]
    
    # 填充多边形
    if len(polygon) >= 3:
        draw.polygon(polygon, fill=1)
    
    return np.array(img)


def convert_json_to_binary_mask(json_file, output_dir):
    """将JSON标签转换为二分类掩码文件"""
    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像尺寸
        tile_size = data.get('tile_image_size', [1024, 1024])
        width, height = tile_size
        
        # 创建空白掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 处理每个牙齿
        if 'tooth' in data:
            for tooth in data['tooth']:
                segmentation = tooth.get('segmentation', [])
                
                if segmentation:
                    # 转换多边形为掩码
                    tooth_mask = polygon_to_mask(segmentation, width, height)
                    
                    # 将牙齿掩码添加到总掩码中（所有牙齿都标记为1）
                    mask[tooth_mask == 1] = TOOTH_CLASS
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_path = os.path.join(output_dir, f'{base_name}.png')
        
        # 创建调色板
        palette = create_palette(256)
        
        # 保存掩码
        mask_image = Image.fromarray(mask, mode='P')
        mask_image.putpalette(palette)
        mask_image.save(output_path)
        
        return True, f"转换成功: {base_name}"
        
    except Exception as e:
        return False, f"转换失败: {os.path.basename(json_file)}, 错误: {str(e)}"


def batch_convert():
    """批量转换所有JSON文件"""
    # 创建输出目录
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(SPLIT_LABEL_DIR, '*.json'))
    total_files = len(json_files)
    
    print(f"开始转换 {total_files} 个JSON文件为二分类掩码...")
    print("=" * 80)
    
    success_count = 0
    failure_count = 0
    
    for i, json_file in enumerate(json_files, 1):
        success, message = convert_json_to_binary_mask(json_file, OUTPUT_MASK_DIR)
        
        if success:
            success_count += 1
            status = "✓"
        else:
            failure_count += 1
            status = "✗"
        
        print(f"[{status}] {i}/{total_files} {message}")
    
    print("=" * 80)
    print(f"转换完成: 成功 {success_count}, 失败 {failure_count}")
    print(f"输出目录: {OUTPUT_MASK_DIR}")


if __name__ == "__main__":
    batch_convert()
