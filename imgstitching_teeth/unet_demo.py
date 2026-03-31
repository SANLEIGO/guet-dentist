"""
U-Net 分割示例

这个脚本演示了如何使用 U-Net 模型进行牙齿分割。
"""

import cv2
import numpy as np
from dental_stitcher_v1.segmentation import segment_teeth


def demo_unet_segmentation(image_path: str, output_prefix: str = "output"):
    """演示 U-Net 分割

    Args:
        image_path: 输入图像路径
        output_prefix: 输出文件前缀
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    print(f"原始图像尺寸: {image.shape}")

    # 测试不同的配置
    configs = [
        {
            "name": "U-Net + GrabCut + CLAHE",
            "method": "unet",
            "use_grabcut": True,
            "use_enhancement": True,
            "enhancement_level": 3.0
        },
        {
            "name": "U-Net + GrabCut (无增强)",
            "method": "unet",
            "use_grabcut": True,
            "use_enhancement": False,
        },
        {
            "name": "U-Net (无 GrabCut)",
            "method": "unet",
            "use_grabcut": False,
            "use_enhancement": False,
        },
        {
            "name": "AlphaDent + GrabCut + CLAHE (对比)",
            "method": "alphadent",
            "use_grabcut": True,
            "use_enhancement": True,
            "enhancement_level": 3.0
        },
    ]

    results = []

    for config in configs:
        print(f"\n{'='*50}")
        print(f"测试配置: {config['name']}")
        print(f"{'='*50}")

        result = segment_teeth(
            image,
            method=config["method"],
            use_grabcut=config["use_grabcut"],
            use_enhancement=config.get("use_enhancement", False),
            enhancement_level=config.get("enhancement_level", 3.0)
        )

        if result:
            coverage = cv2.countNonZero(result.mask) / result.mask.size
            print(f"✅ 成功")
            print(f"   方法: {result.method}")
            print(f"   覆盖率: {coverage:.2%}")
            if result.fallback_reason:
                print(f"   回退原因: {result.fallback_reason}")

            # 保存结果
            overlay_path = f"{output_prefix}_{config['name'].replace(' ', '_').replace('+', '_')}_overlay.png"
            mask_path = f"{output_prefix}_{config['name'].replace(' ', '_').replace('+', '_')}_mask.png"

            cv2.imwrite(overlay_path, result.overlay)
            cv2.imwrite(mask_path, result.mask)

            print(f"   已保存: {overlay_path}")
            print(f"   已保存: {mask_path}")

            results.append({
                "name": config["name"],
                "result": result,
                "overlay_path": overlay_path,
                "mask_path": mask_path
            })
        else:
            print(f"❌ 失败")

    # 创建对比图
    if len(results) >= 2:
        print(f"\n{'='*50}")
        print("创建对比图...")
        print(f"{'='*50}")

        n_results = len(results)
        cols = min(n_results, 3)
        rows = (n_results + cols - 1) // cols

        cell_h, cell_w = 300, 400
        comparison = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for idx, r in enumerate(results):
            row = idx // cols
            col = idx % cols

            # 缩放 overlay 到单元格大小
            resized = cv2.resize(r["result"].overlay, (cell_w - 20, cell_h - 60))

            # 放置到对比图中
            y_start = row * cell_h + 10
            y_end = y_start + resized.shape[0]
            x_start = col * cell_w + 10
            x_end = x_start + resized.shape[1]

            comparison[y_start:y_end, x_start:x_end] = resized

            # 添加标题
            title = r["name"]
            cv2.putText(comparison, title, (x_start, y_start + cell_h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        comparison_path = f"{output_prefix}_comparison.png"
        cv2.imwrite(comparison_path, comparison)
        print(f"✅ 对比图已保存: {comparison_path}")

    print(f"\n{'='*50}")
    print("演示完成！")
    print(f"{'='*50}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python unet_demo.py <image_path> [output_prefix]")
        print("示例: python unet_demo.py test_image.jpg my_test")
        sys.exit(1)

    image_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "unet_demo"

    demo_unet_segmentation(image_path, output_prefix)
