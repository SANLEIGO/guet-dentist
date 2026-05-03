"""
修复 U-Net 模型文件

如果你的 U-Net 模型文件是 state_dict 格式，这个脚本可以帮助你
加载它并重新保存为完整的模型对象。

用法:
    python fix_unet_model.py --state_dict pts/best_model_1.pth --output pts/unet_model.pth --arch resnet34_unet
"""

import argparse
import sys
from pathlib import Path

import torch


def create_resnet34_unet():
    """创建一个 ResNet34 + U-Net 架构的模型。

    根据你提供的 state_dict keys，这是一个 ResNet34 encoder + U-Net decoder 的架构。
    """
    try:
        from segmentation_models_pytorch import Unet
        model = Unet(
            encoder_name="resnet34",
            encoder_weights=None,  # 不使用预训练权重，我们会加载你的权重
            in_channels=3,
            classes=1
        )
        return model
    except ImportError:
        print("❌ segmentation_modelss-pytorch 未安装")
        print("请运行: pip install segmentation-models-pytorch")
        return None


def load_and_fix_state_dict(state_dict_path: str, output_path: str, architecture: str):
    """加载 state_dict 并重新保存为完整模型。"""
    print(f"正在加载 state_dict: {state_dict_path}")

    # 加载 state_dict
    state_dict = torch.load(state_dict_path, map_location='cpu')

    print(f"✅ State_dict 已加载")
    print(f"   包含 {len(state_dict)} 个参数")

    # 根据架构创建模型
    if architecture == "resnet34_unet":
        model = create_resnet34_unet()
        if model is None:
            return False
    else:
        print(f"❌ 不支持的架构: {architecture}")
        print("   支持的架构: resnet34_unet")
        return False

    # 加载权重
    print("正在加载权重到模型...")
    try:
        # 尝试直接加载
        model.load_state_dict(state_dict)
        print("✅ 权重加载成功")
    except Exception as e:
        print(f"⚠️ 直接加载失败: {e}")
        print("尝试使用 strict=False...")
        try:
            model.load_state_dict(state_dict, strict=False)
            print("✅ 权重加载成功（非严格模式）")
        except Exception as e2:
            print(f"❌ 加载权重失败: {e2}")
            return False

    # 设置为评估模式
    model.eval()

    # 保存完整模型
    print(f"正在保存完整模型到: {output_path}")
    torch.save(model, output_path)
    print("✅ 模型已保存")

    return True


def main():
    parser = argparse.ArgumentParser(description="修复 U-Net 模型文件")
    parser.add_argument(
        "--state_dict",
        type=str,
        required=True,
        help="State_dict 文件路径（例如 pts/best_model_1.pth）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pts/unet_model.pth",
        help="输出模型文件路径（默认: pts/unet_model.pth）"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet34_unet",
        choices=["resnet34_unet"],
        help="模型架构（默认: resnet34_unet）"
    )

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not Path(args.state_dict).exists():
        print(f"❌ 文件不存在: {args.state_dict}")
        sys.exit(1)

    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 修复模型
    success = load_and_fix_state_dict(
        args.state_dict,
        args.output,
        args.arch
    )

    if success:
        print("\n" + "=" * 50)
        print("✅ 修复完成！")
        print("=" * 50)
        print(f"新模型文件: {args.output}")
        print("\n请在 .env 文件中更新 UNET_WEIGHTS 路径:")
        print(f"UNET_WEIGHTS={Path(args.output).absolute()}")
        print("\n然后重新启动应用即可使用 U-Net 分割。")
    else:
        print("\n" + "=" * 50)
        print("❌ 修复失败")
        print("=" * 50)
        print("\n如果问题持续，请检查:")
        print("1. 模型架构是否正确")
        print("2. State_dict 是否完整")
        print("3. 是否安装了所有必需的依赖")

        sys.exit(1)


if __name__ == "__main__":
    main()
