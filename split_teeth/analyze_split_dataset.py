import os
import json
import glob
from collections import Counter

# 数据集路径
SPLIT_IMAGE_DIR = 'd:\\Python-Scripts\\U-Net\\splitimage'
SPLIT_LABEL_DIR = 'd:\\Python-Scripts\\U-Net\\splitlabel'


def analyze_dataset():
    print("=" * 80)
    print("Split Dataset Analysis")
    print("=" * 80)
    
    # 1. 统计文件数量
    image_files = glob.glob(os.path.join(SPLIT_IMAGE_DIR, '*.png'))
    label_files = glob.glob(os.path.join(SPLIT_LABEL_DIR, '*.json'))
    
    print(f"[+] 图像文件数量: {len(image_files)}")
    print(f"[+] 标签文件数量: {len(label_files)}")
    print(f"[+] 文件数量匹配: {len(image_files) == len(label_files)}")
    print()
    
    # 2. 分析文件命名模式
    print("[+] 文件命名模式分析:")
    if image_files:
        sample_image = os.path.basename(image_files[0])
        sample_label = os.path.basename(label_files[0])
        print(f"  图像文件示例: {sample_image}")
        print(f"  标签文件示例: {sample_label}")
    print()
    
    # 3. 分析标签内容
    print("[+] 标签内容分析:")
    teeth_numbers = []
    tile_sizes = []
    original_sizes = []
    
    for label_file in label_files[:100]:  # 分析前100个文件
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 收集牙齿编号
            if 'tooth' in data:
                for tooth in data['tooth']:
                    if 'teeth_num' in tooth:
                        teeth_numbers.append(tooth['teeth_num'])
            
            # 收集瓦片尺寸
            if 'tile_image_size' in data:
                tile_sizes.append(tuple(data['tile_image_size']))
            
            # 收集原始图像尺寸
            if 'original_image_size' in data:
                original_sizes.append(tuple(data['original_image_size']))
                
        except Exception as e:
            print(f"  [✘] 解析文件失败: {os.path.basename(label_file)}, 错误: {e}")
    
    # 统计牙齿编号
    if teeth_numbers:
        tooth_counter = Counter(teeth_numbers)
        print(f"  发现牙齿编号: {sorted(tooth_counter.keys())}")
        print(f"  牙齿数量分布: {dict(tooth_counter)}")
    
    # 统计瓦片尺寸
    if tile_sizes:
        tile_counter = Counter(tile_sizes)
        print(f"  瓦片尺寸: {sorted(tile_counter.keys())}")
        print(f"  瓦片尺寸分布: {dict(tile_counter)}")
    
    # 统计原始图像尺寸
    if original_sizes:
        original_counter = Counter(original_sizes)
        print(f"  原始图像尺寸: {sorted(original_counter.keys())}")
        print(f"  原始图像尺寸分布: {dict(original_counter)}")
    print()
    
    # 4. 分析数据完整性
    print("[+] 数据完整性分析:")
    missing_labels = []
    missing_images = []
    
    # 检查每个图像是否有对应标签
    for image_file in image_files:
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        expected_label = os.path.join(SPLIT_LABEL_DIR, f'{base_name}.json')
        if not os.path.exists(expected_label):
            missing_labels.append(base_name)
    
    # 检查每个标签是否有对应图像
    for label_file in label_files:
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        expected_image = os.path.join(SPLIT_IMAGE_DIR, f'{base_name}.png')
        if not os.path.exists(expected_image):
            missing_images.append(base_name)
    
    print(f"  缺失标签的图像: {len(missing_labels)}")
    print(f"  缺失图像的标签: {len(missing_images)}")
    
    if missing_labels:
        print(f"  示例缺失标签的图像: {missing_labels[:5]}")
    if missing_images:
        print(f"  示例缺失图像的标签: {missing_images[:5]}")
    print()
    
    # 5. 分析分割质量
    print("[+] 分割质量分析:")
    segmentation_counts = []
    
    for label_file in label_files[:100]:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'tooth' in data:
                segmentation_counts.append(len(data['tooth']))
                
        except Exception as e:
            pass
    
    if segmentation_counts:
        print(f"  每张图像平均分割牙齿数: {sum(segmentation_counts)/len(segmentation_counts):.2f}")
        print(f"  分割牙齿数范围: {min(segmentation_counts)} - {max(segmentation_counts)}")
    print()
    
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    analyze_dataset()
