import os
import json
import glob
import numpy as np
from PIL import Image

# 数据集路径
SPLIT_IMAGE_DIR = 'd:\\Python-Scripts\\U-Net\\splitimage'
SPLIT_LABEL_DIR = 'd:\\Python-Scripts\\U-Net\\splitlabel'
OUTPUT_MASK_DIR = 'd:\\Python-Scripts\\U-Net\\splitmask'

# 牙齿编号到类别ID的映射（根据FDI系统）
TEETH_NUM_TO_CLASS = {
    11: 1,  # 右上颌中切牙
    12: 2,  # 右上颌侧切牙
    13: 3,  # 右上颌尖牙
    14: 4,  # 右上颌第一前磨牙
    15: 5,  # 右上颌第二前磨牙
    16: 6,  # 右上颌第一磨牙
    21: 7,  # 左上颌中切牙
    22: 8,  # 左上颌侧切牙
    23: 9,  # 左上颌尖牙
    24: 10, # 左上颌第一前磨牙
    25: 11, # 左上颌第二前磨牙
    26: 12, # 左上颌第一磨牙
    # 可以根据需要添加下颌牙齿的映射
}

# 背景类别
BACKGROUND_CLASS = 0


# def create_palette(num_classes):
#     """创建调色板"""
#     palette = []
#     for i in range(num_classes):
#         r = (i * 17) % 256
#         g = (i * 31) % 256
#         b = (i * 47) % 256
#         palette.extend([r, g, b])
#     return palette


def create_palette(num_cls=256):
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


def convert_json_to_mask(json_file, output_dir):
    """将JSON标签转换为掩码文件"""
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
                teeth_num = tooth.get('teeth_num')
                segmentation = tooth.get('segmentation', [])
                
                if teeth_num and segmentation:
                    # 获取类别ID
                    class_id = TEETH_NUM_TO_CLASS.get(teeth_num, BACKGROUND_CLASS)
                    
                    # 转换多边形为掩码
                    tooth_mask = polygon_to_mask(segmentation, width, height)
                    
                    # 将牙齿掩码添加到总掩码中
                    mask[tooth_mask == 1] = class_id
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_path = os.path.join(output_dir, f'{base_name}.png')
        
        # 创建调色板
        num_classes = len(TEETH_NUM_TO_CLASS) + 1  # 加上背景
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
    
    print(f"开始转换 {total_files} 个JSON文件...")
    print("=" * 80)
    
    success_count = 0
    failure_count = 0
    
    for i, json_file in enumerate(json_files, 1):
        success, message = convert_json_to_mask(json_file, OUTPUT_MASK_DIR)
        
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
