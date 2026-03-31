# U-Net 模型使用说明

## 模型文件准备

本项目支持使用 U-Net 模型进行牙齿分割。如果您想使用 U-Net 分割方法，需要准备相应的模型文件。

### 模型文件位置

将 U-Net 模型文件放置在以下位置：

```
pts/unet_model.pth
```

或者在 `.env` 文件中自定义路径：

```bash
UNET_WEIGHTS=/path/to/your/unet_model.pth
```

## 模型格式要求

### 输入
- **尺寸**: 任意尺寸（代码会自动调整到 256x256）
- **格式**: RGB 图像
- **归一化**: ImageNet 标准 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 输出
模型输出应该是以下格式之一：

1. **单通道输出** (shape: `[batch, 1, H, W]`)
   - 经过 sigmoid 激活函数
   - 值范围 [0, 1]，表示前景概率

2. **双通道输出** (shape: `[batch, 2, H, W]`)
   - 经过 softmax 激活函数
   - 通道 0: 背景概率
   - 通道 1: 前景概率

### PyTorch 模型示例

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # 您的 U-Net 架构
        self.encoder = ...
        self.decoder = ...

    def forward(self, x):
        # 返回 logits 或概率
        return self.decoder(x)

# 保存模型
model = UNet()
torch.save(model.state_dict(), 'unet_model.pth')
# 或者保存整个模型
torch.save(model, 'unet_model.pth')
```

## 推荐的 U-Net 架构

如果需要从头训练 U-Net 模型，可以考虑以下实现：

### 选项 1: 使用 segmentation-models-pytorch

```bash
pip install segmentation-models-pytorch
```

```python
import torch
from segmentation_models_pytorch import Unet

model = Unet(
    encoder_name="resnet34",        # 选择 encoder
    encoder_weights="imagenet",     # 使用预训练权重
    in_channels=3,                  # 输入通道数 (RGB)
    classes=1                       # 输出类别数 (二分类)
)

# 保存模型
torch.save(model, 'pts/unet_model.pth')
```

### 选项 2: 使用 kornia（项目已依赖）

项目已经包含 `kornia` 库，可以使用其提供的 U-Net 实现。

### 选项 3: 自定义实现

参考经典的 U-Net 论文架构：
- 编码器：4 次下采样
- 解码器：4 次上采样 + 跳跃连接
- 输出：sigmoid 激活的单通道 mask

## 使用方法

1. **准备模型文件**：将训练好的 U-Net 模型放置在 `pts/unet_model.pth`

2. **在前端选择**：启动应用后，在分割设置中选择 "U-Net" 方法

3. **调整参数**：
   - GrabCut 精细化：建议启用
   - CLAHE 图像增强：建议启用，可以提高低质量图像的分割效果

4. **执行分割**：点击"执行分割"按钮

## 故障排查

### U-Net 不可用
如果 U-Net 模型未加载，系统会自动回退到 AlphaDent 方法。检查：
- 模型文件是否存在
- PyTorch 是否正确安装
- `.env` 文件中的路径是否正确

### 内存不足
U-Net 模型推理可能需要较多内存，如果遇到内存问题：
- 减小输入图像尺寸
- 使用较小的 batch size
- 关闭其他占用内存的应用

## 训练数据建议

如果要训练自己的 U-Net 模型，建议：
- 使用标注好的牙齿分割数据集
- 数据增强：旋转、翻转、亮度调整
- 损失函数：Dice Loss + Binary Cross Entropy
- 评估指标：Dice Coefficient, IoU

## 参考资源

- [U-Net 论文](https://arxiv.org/abs/1505.04597)
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [kornia 文档](https://kornia.readthedocs.io/)
