"""
去除拼接图像中的噪点

使用形态学操作和连通域分析去除小块软组织噪点
"""

import cv2
import numpy as np
from pathlib import Path


def remove_noise_from_stitched_image(
    input_path: str,
    output_path: str,
    min_area_threshold: int = 1000,
    morph_kernel_size: int = 5,
    use_morphology: bool = True,
    use_connected_components: bool = True
):
    """
    去除拼接图像中的小块噪点

    Args:
        input_path: 输入图像路径
        output_path: 输出图像路径
        min_area_threshold: 最小面积阈值（像素），小于此面积的连通域将被去除
        morph_kernel_size: 形态学操作的核大小
        use_morphology: 是否使用形态学开运算
        use_connected_components: 是否使用连通域分析
    """
    print(f"处理图像: {input_path}")

    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图像: {input_path}")

    print(f"原始图像尺寸: {img.shape}")

    # 步骤1: 提取非黑色区域（拼接结果的有效区域）
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建二值图像（非零区域为前景）
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    print(f"二值图像非零像素数: {np.count_nonzero(binary)}")

    # 步骤2: 形态学开运算（先侵蚀后膨胀）去除小噪点
    if use_morphology:
        print(f"\n应用形态学开运算（核大小: {morph_kernel_size}x{morph_kernel_size}）")
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        binary_morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        print(f"形态学处理后非零像素数: {np.count_nonzero(binary_morphed)}")
    else:
        binary_morphed = binary

    # 步骤3: 连通域分析，保留大区域
    if use_connected_components:
        print(f"\n连通域分析（最小面积阈值: {min_area_threshold}像素）")
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_morphed, connectivity=8
        )

        print(f"检测到 {num_labels-1} 个连通域（不包括背景）")

        # 显示各个连通域的面积
        areas = stats[:, cv2.CC_STAT_AREA]
        print("\n连通域面积统计:")
        for i in range(1, num_labels):  # 0是背景
            print(f"  区域 {i}: 面积={areas[i]}px, 中心=({centroids[i][0]:.1f}, {centroids[i][1]:.1f})")

        # 创建掩膜，只保留大区域
        final_mask = np.zeros(binary_morphed.shape, dtype=np.uint8)

        # 找出大面积区域
        large_areas = []
        for i in range(1, num_labels):
            if areas[i] >= min_area_threshold:
                large_areas.append(i)
                final_mask[labels == i] = 255

        print(f"\n保留 {len(large_areas)} 个大区域: {large_areas}")
        print(f"最终掩膜非零像素数: {np.count_nonzero(final_mask)}")
    else:
        final_mask = binary_morphed

    # 步骤4: 应用掩膜到原始图像
    result = cv2.bitwise_and(img, img, mask=final_mask)

    # 步骤5: 可选的闭运算（填充牙齿内部的小孔洞）
    if use_morphology:
        print("\n应用闭运算填充内部孔洞")
        kernel_close = np.ones((3, 3), np.uint8)
        final_mask_refined = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        result = cv2.bitwise_and(img, img, mask=final_mask_refined)
        print(f"闭运算后非零像素数: {np.count_nonzero(final_mask_refined)}")

    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"\n✅ 结果已保存到: {output_path}")

    # 保存中间结果用于调试
    debug_dir = Path(output_path).parent / "debug"
    debug_dir.mkdir(exist_ok=True)

    base_name = Path(input_path).stem
    cv2.imwrite(str(debug_dir / f"{base_name}_1_binary.png"), binary)
    cv2.imwrite(str(debug_dir / f"{base_name}_2_morphed.png"), binary_morphed)
    cv2.imwrite(str(debug_dir / f"{base_name}_3_final_mask.png"), final_mask)
    print(f"中间结果已保存到: {debug_dir}")

    return result


def test_different_parameters(input_path: str):
    """
    测试不同的参数组合
    """
    base_name = Path(input_path).stem
    output_dir = Path("cleaned_results")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("测试不同的去噪参数")
    print("=" * 60)

    # 测试1: 只用连通域分析
    print("\n【方案1】只使用连通域分析（面积阈值=1000）")
    remove_noise_from_stitched_image(
        input_path,
        str(output_dir / f"{base_name}_method1.png"),
        min_area_threshold=1000,
        use_morphology=False,
        use_connected_components=True
    )

    # 测试2: 形态学 + 连通域分析（小核）
    print("\n【方案2】形态学（核=3）+ 连通域分析（面积=1000）")
    remove_noise_from_stitched_image(
        input_path,
        str(output_dir / f"{base_name}_method2.png"),
        min_area_threshold=1000,
        morph_kernel_size=3,
        use_morphology=True,
        use_connected_components=True
    )

    # 测试3: 形态学 + 连通域分析（中等核）
    print("\n【方案3】形态学（核=5）+ 连通域分析（面积=1000）")
    remove_noise_from_stitched_image(
        input_path,
        str(output_dir / f"{base_name}_method3.png"),
        min_area_threshold=1000,
        morph_kernel_size=5,
        use_morphology=True,
        use_connected_components=True
    )

    # 测试4: 形态学 + 连通域分析（大核，大阈值）
    print("\n【方案4】形态学（核=7）+ 连通域分析（面积=2000）")
    remove_noise_from_stitched_image(
        input_path,
        str(output_dir / f"{base_name}_method4.png"),
        min_area_threshold=2000,
        morph_kernel_size=7,
        use_morphology=True,
        use_connected_components=True
    )

    print("\n" + "=" * 60)
    print("所有测试完成！请查看 cleaned_results/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python remove_noise.py <输入图像> [输出图像]")
        print("  python remove_noise.py test <输入图像>  # 测试不同参数")
        print("\n示例:")
        print("  python remove_noise.py stitched_teeth_only_v1.png cleaned.png")
        print("  python remove_noise.py test stitched_teeth_only_v1.png")
        sys.exit(1)

    input_image = sys.argv[1]

    if sys.argv[1] == "test" and len(sys.argv) >= 3:
        # 测试模式
        test_different_parameters(sys.argv[2])
    elif len(sys.argv) >= 3:
        # 单次处理
        output_image = sys.argv[2]
        remove_noise_from_stitched_image(input_image, output_image)
    else:
        # 默认输出
        output_image = Path(input_image).stem + "_cleaned.png"
        remove_noise_from_stitched_image(input_image, output_image)