#!/usr/bin/env python3
"""
测试鲁棒几何变换的脚本

用于测试改进后的特征匹配和几何变换算法
"""

import cv2
import numpy as np
from dental_stitcher.enhanced_stitching import DentalFeatureMatcher

def test_geometric_distortion():
    """测试几何形变情况下的拼接"""

    print("=" * 60)
    print("测试鲁棒几何变换")
    print("=" * 60)

    # 创建测试图像 - 模拟牙齿纹理
    img1 = create_test_image((400, 400), pattern="teeth1")
    img2 = create_test_image((400, 400), pattern="teeth2")

    # 添加透视变换
    h, w = img2.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = np.float32([[50, 50], [w-30, 20], [w-20, h-40], [20, h-30]])

    # 应用透视变换
    M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
    img2_warped = cv2.warpPerspective(img2, M_perspective, (w, h))

    print("\n测试场景:")
    print(f"- 图像1尺寸: {img1.shape}")
    print(f"- 图像2尺寸: {img2_warped.shape}")
    print(f"- 已对图像2应用透视变换")

    # 初始化特征匹配器
    matcher = DentalFeatureMatcher()

    # 检测特征
    print("\n检测特征点...")
    mask1 = np.ones(img1.shape[:2], dtype=np.uint8) * 255
    mask2 = np.ones(img2_warped.shape[:2], dtype=np.uint8) * 255

    kp1, desc1 = matcher.detect_and_compute(img1, mask1)
    kp2, desc2 = matcher.detect_and_compute(img2_warped, mask2)

    print(f"✓ 图像1检测到 {len(kp1)} 个特征点")
    print(f"✓ 图像2检测到 {len(kp2)} 个特征点")

    # 使用鲁棒匹配
    print("\n使用鲁棒匹配算法...")
    method = 'sift' if desc1.shape[1] != 32 else 'orb'
    matches = matcher.match_features_robust(
        desc1, desc2, kp1, kp2,
        method=method,
        img_shape=img1.shape
    )

    print(f"✓ 鲁棒匹配找到 {len(matches)} 对匹配点")

    if len(matches) >= 4:
        # 估计变换
        print("\n估计几何变换...")
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 测试多种方法
        methods_to_test = [
            ("RANSAC (3.0)", cv2.RANSAC, 3.0),
            ("RHO", cv2.RHO, 5.0),
            ("LMEDS", cv2.LMEDS, 0),
        ]

        for name, method, thresh in methods_to_test:
            if method == cv2.LMEDS:
                H, mask = cv2.findHomography(src_pts, dst_pts, method)
            else:
                H, mask = cv2.findHomography(src_pts, dst_pts, method, thresh)

            if H is not None:
                inliers = int(mask.sum())
                inlier_ratio = inliers / len(mask)
                print(f"✓ {name:15s}: 内点 {inliers:3d}/{len(mask):3d} ({inlier_ratio:5.1%})")
            else:
                print(f"✗ {name:15s}: 失败")

        print("\n✓ 测试完成！改进后的算法应该能更好地处理透视变形")
    else:
        print(f"\n✗ 匹配点不足 ({len(matches)} < 4)，无法估计变换")

    print("=" * 60)

def create_test_image(size, pattern="teeth1"):
    """创建模拟牙齿纹理的测试图像"""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    if pattern == "teeth1":
        # 模拟牙齿1 - 白色背景 + 纹理
        img[:, :] = [240, 240, 230]
        # 添加一些"牙齿"形状
        cv2.rectangle(img, (50, 100), (150, 300), (220, 220, 210), -1)
        cv2.rectangle(img, (160, 100), (260, 300), (230, 230, 220), -1)
        cv2.rectangle(img, (270, 100), (370, 300), (220, 220, 210), -1)
        # 添加纹理
        noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

    elif pattern == "teeth2":
        # 模拟牙齿2 - 稍微不同的视角
        img[:, :] = [235, 235, 225]
        # 添加"牙齿"形状（偏移）
        cv2.rectangle(img, (100, 100), (200, 300), (225, 225, 215), -1)
        cv2.rectangle(img, (210, 100), (310, 300), (220, 220, 210), -1)
        cv2.rectangle(img, (320, 100), (420, 300), (230, 230, 220), -1)
        # 添加纹理
        noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

    return img

if __name__ == "__main__":
    test_geometric_distortion()
