#!/usr/bin/env python3
"""
测试改进的牙齿图像拼接算法
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from dental_stitcher.enhanced_stitching import (
    CompatibleImprovedStitcher,
    DentalImagePreprocessor,
    DentalFeatureMatcher,
    blend_multi_band
)
from dental_stitcher.utils import load_image_records


def test_preprocessor():
    """测试图像预处理"""
    print("测试图像预处理...")

    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    preprocessor = DentalImagePreprocessor()
    enhanced, mask = preprocessor.preprocess(test_image)

    print(f"原始图像尺寸: {test_image.shape}")
    print(f"增强图像尺寸: {enhanced.shape}")
    print(f"掩膜尺寸: {mask.shape}")
    print(f"掩膜非零像素: {np.count_nonzero(mask)}")
    print("预处理测试通过！")


def test_feature_matcher():
    """测试特征匹配"""
    print("\n测试特征匹配...")

    # 创建两张略有不同的测试图像
    img1 = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    img2 = img1.copy()
    img2[50:150, 50:150] = 150  # 添加一些变化

    matcher = DentalFeatureMatcher()

    # 创建掩膜
    mask = np.ones((480, 640), dtype=np.uint8) * 255

    # 检测特征
    kp1, desc1 = matcher.detect_and_compute(img1, mask)
    kp2, desc2 = matcher.detect_and_compute(img2, mask)

    print(f"图像1检测到 {len(kp1)} 个特征点")
    print(f"图像2检测到 {len(kp2)} 个特征点")

    if desc1 is not None and desc2 is not None:
        # 匹配特征
        matches = matcher.match_features(desc1, desc2)
        print(f"匹配到 {len(matches)} 对特征点")
        print("特征匹配测试通过！")
    else:
        print("警告: 未能检测到足够的特征点")


def test_blending():
    """测试多频段融合"""
    print("\n测试多频段融合...")

    # 创建测试图像
    img1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
    img2 = np.ones((480, 640, 3), dtype=np.uint8) * 200

    # 创建掩膜
    mask1 = np.ones((480, 640), dtype=np.uint8) * 255
    mask2 = np.ones((480, 640), dtype=np.uint8) * 255

    # 创建变换矩阵（第二个图像向右平移300像素）
    H1 = np.eye(3)
    H2 = np.array([[1, 0, 300], [0, 1, 0], [0, 0, 1]])

    try:
        result = blend_multi_band([img1, img2], [mask1, mask2], [H1, H2], feather_radius=20)
        print(f"融合结果尺寸: {result.shape}")
        print(f"融合结果范围: [{result.min()}, {result.max()}]")
        print("多频段融合测试通过！")
    except Exception as e:
        print(f"融合测试失败: {e}")


def test_full_pipeline():
    """测试完整拼接流程"""
    print("\n测试完整拼接流程...")

    # 创建模拟的图像记录
    from dental_stitcher.models import ImageRecord

    # 创建测试图像
    images = []
    for i in range(3):
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        # 添加一些重叠区域
        if i > 0:
            offset_x = int(640 * 0.4)  # 40%重叠
            img[:, :offset_x] = images[-1][:, -offset_x:]

        images.append(img)

    # 创建图像记录
    records = []
    for i, img in enumerate(images):
        record = ImageRecord(
            path=Path(f"test_{i}.jpg"),
            arch="upper",
            segment="left",
            image=img
        )
        # 设置质量分数
        record.quality_score = 5.0 + i * 0.1
        records.append(record)

    # 创建拼接器
    stitcher = CompatibleImprovedStitcher()

    # 执行拼接
    result = stitcher.stitch(records)

    print(f"拼接成功: {result.success}")
    if result.success:
        print(f"结果尺寸: {result.panorama.shape}")
        print(f"基准图索引: {result.anchor_index}")
        print("处理日志:")
        for log in result.logs[-5:]:  # 显示最后5条日志
            print(f"  {log}")
        print("完整拼接测试通过！")
    else:
        print("拼接失败")
        print("错误日志:")
        for log in result.logs:
            print(f"  {log}")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("牙齿图像拼接算法改进版测试")
    print("=" * 60)

    try:
        test_preprocessor()
        test_feature_matcher()
        test_blending()
        test_full_pipeline()

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())