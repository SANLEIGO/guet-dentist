# U-Net 模型快速开始指南

## 方案 1: 使用现成的分割库（推荐）

如果你没有现成的 U-Net 模型，可以使用 `segmentation-models-pytorch` 库快速获得一个预训练模型。

### 步骤 1: 安装依赖

```bash
source .venv310/bin/activate
pip install segmentation-models-pytorch
```

### 步骤 2: 创建并保存模型

创建一个脚本 `create_unet_model.py`:

```python
import torch
from segmentation_models_pytorch import Unet

# 创建 U-Net 模型
# 使用预训练的 ResNet34 作为 encoder
model = Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

# 切换到评估模式
model.eval()

# 保存整个模型
torch.save(model, 'pts/unet_model.pth')

print("✅ U-Net 模型已保存到 pts/unet_model.pth")
```

运行脚本：

```bash
python create_unet_model.py
```

### 步骤 3: 测试

```bash
# 启动应用
./run.sh

# 或运行测试脚本
python test_unet.py --image_path /path/to/test/image.jpg --compare
```

---

## 方案 2: 使用 kornia（项目已依赖）

项目已经包含 `kornia` 库，可以使用其提供的 U-Net 实现。

### 步骤 1: 创建模型

创建 `create_kornia_unet.py`:

```python
import torch
import kornia

# kornia 的 U-Net 实现
# 注意：kornia 的 API 可能会有变化，请参考官方文档
from kornia.contrib import SegmentationModels

# 创建模型
model = SegmentationModels.unet(
    in_channels=3,
    num_classes=1,
    encoder_name="resnet34",
    pretrained=True
)

model.eval()

# 保存模型
torch.save(model, 'pts/unet_model.pth')

print("✅ Kornia U-Net 模型已保存")
```

---

## 方案 3: 简化的 U-Net（最小实现）

如果不想依赖额外的库，可以使用这个简化的 U-Net 实现。

### 步骤 1: 创建简化 U-Net

创建 `simple_unet.py`:

```python
import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.up4 = self.up_conv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = self.up_conv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.up_conv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.up_conv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # Final
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return self.final(dec1)

# 创建并保存模型
model = SimpleUNet(in_channels=3, out_channels=1)
model.eval()

# 保存模型
torch.save(model, 'pts/unet_model.pth')

print("✅ 简化 U-Net 模型已保存")
```

运行：

```bash
python simple_unet.py
```

---

## 方案 4: 训练自己的 U-Net 模型

如果你有标注好的牙齿分割数据集，可以训练自己的模型。

### 数据准备

准备以下结构的数据集：

```
dataset/
├── train/
│   ├── images/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── masks/
│       ├── 001.png
│       ├── 002.png
│       └── ...
└── val/
    ├── images/
    └── masks/
```

### 训练脚本

创建 `train_unet.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path

class TeethDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_files = list(self.images_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(str(self.image_files[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.masks_dir / (self.image_files[idx].stem + ".png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        # Transform
        if self.transform:
            # 这里应该添加适当的变换
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

# 定义模型（使用上面的 SimpleUNet 或其他实现）
from simple_unet import SimpleUNet

# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 数据加载
train_dataset = TeethDataset("dataset/train/images", "dataset/train/masks")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 训练循环
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# 保存模型
torch.save(model, 'pts/unet_model.pth')
print("✅ 训练完成，模型已保存")
```

运行训练：

```bash
python train_unet.py
```

---

## 测试 U-Net 集成

模型准备好后，测试集成是否正常工作：

```bash
# 方法 1: 使用测试脚本
python test_unet.py --image_path /path/to/test/image.jpg --compare

# 方法 2: 使用演示脚本
python unet_demo.py /path/to/test/image.jpg unet_test

# 方法 3: 启动应用并在界面中选择 U-Net
./run.sh
```

---

## 常见问题

### Q: U-Net 分割效果不好怎么办？
A: 可以尝试：
1. 启用 CLAHE 图像增强
2. 启用 GrabCut 精细化
3. 调整 CLAHE 增强强度
4. 使用更好的预训练模型
5. 在牙齿数据集上微调模型

### Q: 如何选择合适的 U-Net 架构？
A:
- **快速测试**: 使用简化的 U-Net（方案 3）
- **实际应用**: 使用 segmentation-models-pytorch（方案 1）
- **最佳性能**: 训练自己的模型（方案 4）

### Q: U-Net 和 AlphaDent 哪个更好？
A:
- **AlphaDent (YOLOv8)**: 专为牙齿检测设计，速度快，准确率高
- **U-Net**: 通用分割架构，需要训练数据，可自定义

建议优先使用 AlphaDent，除非你有特殊的分割需求或已经训练好的 U-Net 模型。

---

## 需要帮助？

如果遇到问题，请检查：
1. `pts/unet_model.pth` 文件是否存在
2. `.env` 文件中的 `UNET_WEIGHTS` 路径是否正确
3. PyTorch 和 torchvision 是否正确安装
4. 模型格式是否符合要求（参见 `pts/UNET_README.md`）

参考文档：
- `pts/UNET_README.md` - 详细的模型格式说明
- `CHANGELOG.md` - 更新日志
- `README.md` - 项目总体说明
