"""
测试 U-Net 分割功能

用法:
    python test_unet.py --model_path pts/unet_model.pth --image_path test_image.jpg
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from dental_stitcher_v1.segmentation import segment_teeth, _get_unet_model


def test_unet_loading():
    """测试 U-Net 模型加载"""
    print("=" * 50)
    print("测试 U-Net 模型加载")
    print("=" * 50)

    model = _get_unet_model()

    if model is None:
        print("❌ U-Net 模型加载失败")
        print("   请检查:")
        print("   1. 模型文件是否存在于 pts/unet_model.pth")
        print("   2. .env 文件中 UNET_WEIGHTS 路径是否正确")
        print("   3. PyTorch 是否正确安装")
        return False

    print("✅ U-Net 模型加载成功")
    print(f"   模型类型: {type(model)}")
    return True


def test_unet_segmentation(image_path: str):
    """测试 U-Net 分割功能"""
    print("\n" + "=" * 50)
    print("测试 U-Net 分割功能")
    print("=" * 50)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return False

    print(f"✅ 成功读取图像: {image_path}")
    print(f"   图像尺寸: {image.shape}")

    # 测试 U-Net 分割
    print("\n执行 U-Net 分割...")
    result = segment_teeth(
        image,
        method="unet",
        use_grabcut=True,
        use_enhancement=True,
        enhancement_level=3.0
    )

    if result is None:
        print("❌ 分割失败")
        return False

    print("✅ 分割成功")
    print(f"   分割方法: {result.method}")
    print(f"   掩膜尺寸: {result.mask.shape}")
    print(f"   掩膜非零像素: {cv2.countNonZero(result.mask)}")
    print(f"   掩膜覆盖率: {cv2.countNonZero(result.mask) / result.mask.size:.2%}")

    if result.fallback_reason:
        print(f"   ⚠️  使用了回退方案: {result.fallback_reason}")

    # 保存结果
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    output_overlay = output_dir / "unet_overlay.png"
    output_mask = output_dir / "unet_mask.png"

    cv2.imwrite(str(output_overlay), result.overlay)
    cv2.imwrite(str(output_mask), result.mask)

    print(f"\n✅ 结果已保存:")
    print(f"   - {output_overlay}")
    print(f"   - {output_mask}")

    return True


def test_alphadent_comparison(image_path: str):
    """对比 AlphaDent 和 U-Net 分割结果"""
    print("\n" + "=" * 50)
    print("对比 AlphaDent 和 U-Net 分割")
    print("=" * 50)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return

    # AlphaDent 分割
    print("\n1. AlphaDent 分割...")
    alphadent_result = segment_teeth(
        image,
        method="alphadent",
        use_grabcut=True,
        use_enhancement=True,
        enhancement_level=3.0
    )

    if alphadent_result:
        print(f"   ✅ AlphaDent 分割成功")
        print(f"   方法: {alphadent_result.method}")
        print(f"   掩膜覆盖率: {cv2.countNonZero(alphadent_result.mask) / alphadent_result.mask.size:.2%}")

    # U-Net 分割
    print("\n2. U-Net 分割...")
    unet_result = segment_teeth(
        image,
        method="unet",
        use_grabcut=True,
        use_enhancement=True,
        enhancement_level=3.0
    )

    if unet_result:
        print(f"   ✅ U-Net 分割成功")
        print(f"   方法: {unet_result.method}")
        print(f"   掩膜覆盖率: {cv2.countNonZero(unet_result.mask) / unet_result.mask.size:.2%}")

    # 保存对比结果
    if alphadent_result and unet_result:
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)

        # 创建对比图
        h, w = image.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = alphadent_result.overlay
        comparison[:, w:] = unet_result.overlay

        # 添加标签
        cv2.putText(comparison, "AlphaDent", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "U-Net", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        output_comparison = output_dir / "comparison.png"
        cv2.imwrite(str(output_comparison), comparison)
        print(f"\n✅ 对比结果已保存: {output_comparison}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试 U-Net 分割功能")
    parser.add_argument("--model_path", type=str, default="pts/unet_model.pth",
                        help="U-Net 模型路径")
    parser.add_argument("--image_path", type=str, required=True,
                        help="测试图像路径")
    parser.add_argument("--compare", action="store_true",
                        help="是否对比 AlphaDent 和 U-Net 结果")

    args = parser.parse_args()

    # 测试模型加载
    if not test_unet_loading():
        print("\n⚠️  U-Net 模型不可用，将使用 AlphaDent 作为回退方案")
        if not args.compare:
            return

    # 测试分割
    if not test_unet_segmentation(args.image_path):
        return

    # 对比测试
    if args.compare:
        test_alphadent_comparison(args.image_path)

    print("\n" + "=" * 50)
    print("✅ 所有测试完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
