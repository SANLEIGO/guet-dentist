"""
批量去噪工具

一键处理所有拼接结果图片
"""

import cv2
import numpy as np
from pathlib import Path
import glob


def clean_stitched_image(
    input_path: str,
    output_path: str,
    method: str = "method3"
):
    """
    快速去噪处理

    Args:
        input_path: 输入图像路径
        output_path: 输出图像路径
        method: 去噪方法 (method1-4)
    """
    # 参数配置
    configs = {
        "method1": {"min_area": 1000, "kernel": 0, "use_morph": False},
        "method2": {"min_area": 1000, "kernel": 3, "use_morph": True},
        "method3": {"min_area": 1000, "kernel": 5, "use_morph": True},  # 推荐
        "method4": {"min_area": 2000, "kernel": 7, "use_morph": True},
    }

    config = configs[method]

    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        print(f"⚠️ 无法读取: {input_path}")
        return False

    # 提取有效区域
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 形态学开运算
    if config["use_morph"]:
        kernel = np.ones((config["kernel"], config["kernel"]), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]

    # 创建掩膜
    final_mask = np.zeros(binary.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        if areas[i] >= config["min_area"]:
            final_mask[labels == i] = 255

    # 闭运算填充孔洞
    if config["use_morph"]:
        kernel_close = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 应用掩膜
    result = cv2.bitwise_and(img, img, mask=final_mask)

    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"✅ 已处理: {Path(input_path).name} → {Path(output_path).name}")
    return True


def batch_clean(
    input_pattern: str = "stitched_teeth_only_v1*.png",
    output_dir: str = "cleaned_final",
    method: str = "method3"
):
    """
    批量处理所有拼接结果

    Args:
        input_pattern: 输入文件模式（glob格式）
        output_dir: 输出目录
        method: 去噪方法
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 查找所有输入文件
    input_files = glob.glob(input_pattern)

    if not input_files:
        print(f"⚠️ 没有找到匹配的文件: {input_pattern}")
        return

    print(f"\n{'='*60}")
    print(f"批量去噪处理")
    print(f"{'='*60}")
    print(f"输入文件模式: {input_pattern}")
    print(f"找到文件数量: {len(input_files)}")
    print(f"输出目录: {output_dir}")
    print(f"去噪方法: {method} (推荐)")
    print(f"{'='*60}\n")

    # 处理每个文件
    success_count = 0
    for input_file in sorted(input_files):
        input_name = Path(input_file).name
        output_file = str(output_path / input_name)

        if clean_stitched_image(input_file, output_file, method):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"成功: {success_count}/{len(input_files)}")
    print(f"输出位置: {output_dir}/")
    print(f"{'='*60}\n")


def interactive_mode():
    """
    交互式模式
    """
    print("\n" + "="*60)
    print("拼接图像去噪工具 - 交互模式")
    print("="*60)

    # 选择输入
    print("\n可用的拼接结果:")
    files = glob.glob("stitched_teeth_only_v1*.png")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {Path(f).name}")

    if not files:
        print("⚠️ 没有找到拼接结果文件")
        return

    # 选择文件
    choice = input("\n选择文件编号（或输入'all'处理全部）: ").strip()

    if choice == "all":
        # 选择方法
        print("\n去噪方法:")
        print("  1. method1 - 只用连通域分析")
        print("  2. method2 - 形态学(核=3) + 连通域")
        print("  3. method3 - 形态学(核=5) + 连通域 (推荐)")
        print("  4. method4 - 形态学(核=7) + 连通域 (激进)")

        method_choice = input("选择方法编号 [默认: 3]: ").strip() or "3"
        method = f"method{method_choice}"

        batch_clean(method=method)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                input_file = files[idx]
                output_file = f"cleaned_{Path(input_file).name}"

                print(f"\n处理: {Path(input_file).name}")
                print(f"输出: {output_file}")

                clean_stitched_image(input_file, output_file, method="method3")
            else:
                print("⚠️ 无效的选择")
        except ValueError:
            print("⚠️ 请输入数字或'all'")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # 交互模式
        interactive_mode()
    elif sys.argv[1] == "batch":
        # 批量处理
        method = sys.argv[2] if len(sys.argv) > 2 else "method3"
        batch_clean(method=method)
    else:
        # 单文件处理
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else f"cleaned_{Path(input_file).name}"
        method = sys.argv[3] if len(sys.argv) > 3 else "method3"

        print(f"\n处理: {input_file}")
        print(f"输出: {output_file}")
        print(f"方法: {method}\n")

        clean_stitched_image(input_file, output_file, method)

        print("\n使用说明:")
        print("  交互模式: python3 clean_batch.py")
        print("  批量处理: python3 clean_batch.py batch [method]")
        print("  单文件:   python3 clean_batch.py <input> <output> [method]")