# 问题修复说明

## 已修复的问题

### ✅ 问题 1: 分割掩膜显示增强后的图像

**问题描述**：
之前启用了 CLAHE 图像增强后，显示的分割掩膜使用了增强后的图像，导致颜色失真。

**修复方案**：
- 分割时仍然使用增强后的图像（提高分割质量）
- 但显示掩膜时使用原始图像（保持真实色彩）
- 修改了 `_segment_alphadent()` 和 `_segment_unet()` 函数，添加 `original_image` 参数

**验证**：
```bash
# 上传一张图像
# 启用 CLAHE 图像增强
# 执行分割
# 查看掩膜 - 应该显示原始图像的色彩，而不是增强后的色彩
```

---

### ✅ 问题 2: U-Net 失败时自动回退到 AlphaDent

**问题描述**：
之前 U-Net 不可用时，系统会自动回退到 AlphaDent，导致用户不知道 U-Net 有问题。

**修复方案**：
- 移除了自动回退逻辑
- U-Net 或 AlphaDent 不可用时直接抛出 `RuntimeError`
- 错误信息清晰说明失败原因
- 前端捕获异常并显示详细错误信息

**验证**：
```bash
# 选择 U-Net 分割方法
# 如果 U-Net 不可用，会看到清晰的错误信息
# 不会自动切换到 AlphaDent
```

---

## 当前 U-Net 模型状态

### 问题诊断

你的 U-Net 模型文件 (`pts/best_model_1.pth`) 是一个 **state_dict** 格式，而不是完整的模型对象。

**这是什么意思？**
- State_dict 只包含模型的权重参数
- 不包含模型的结构定义
- 需要知道模型的架构才能加载

**如何判断？**
```python
import torch
checkpoint = torch.load('pts/best_model_1.pth')
type(checkpoint)  # <class 'collections.OrderedDict'>
```

从模型的 keys 可以看出，这是一个 **ResNet34 + U-Net** 架构：
- `resnet.*` - ResNet34 encoder
- `up_concat.*`, `up_conv.*`, `final.*` - U-Net decoder

---

## 解决方案

### 方案 1: 使用提供的修复脚本（推荐）

```bash
# 1. 安装 segmentation-models-pytorch
source .venv310/bin/activate
pip install segmentation-models-pytorch

# 2. 运行修复脚本
python fix_unet_model.py --state_dict pts/best_model_1.pth --output pts/unet_model.pth --arch resnet34_unet

# 3. 更新 .env 文件
# 将 UNET_WEIGHTS 设置为新的模型文件路径
```

### 方案 2: 重新保存模型

如果你有训练模型的代码，修改保存方式：

```python
# ❌ 错误的保存方式（只保存 state_dict）
torch.save(model.state_dict(), 'pts/best_model_1.pth')

# ✅ 正确的保存方式（保存完整模型）
torch.save(model, 'pts/unet_model.pth')
```

### 方案 3: 定义模型架构并加载权重

如果你知道模型的详细架构，可以定义模型类并加载权重：

```python
import torch
import torch.nn as nn

class YourUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义你的模型架构
        self.resnet = ...
        self.decoder = ...

    def forward(self, x):
        # 实现前向传播
        return output

# 创建模型并加载权重
model = YourUNet()
state_dict = torch.load('pts/best_model_1.pth')
model.load_state_dict(state_dict)
model.eval()

# 保存完整模型
torch.save(model, 'pts/unet_model.pth')
```

---

## 测试修复

### 1. 测试模型加载

```bash
python -c "
from dental_stitcher_v1.segmentation import _get_unet_model
model = _get_unet_model()
if model:
    print('✅ U-Net 模型加载成功')
else:
    from dental_stitcher_v1.segmentation import _UNET_MODEL_ERROR
    print(f'❌ U-Net 模型加载失败: {_UNET_MODEL_ERROR}')
"
```

### 2. 测试分割功能

```bash
# 使用测试图像
python test_unet.py --image_path /path/to/test/image.jpg

# 或在应用界面中选择 U-Net 并测试
./run.sh
```

### 3. 验证掩膜显示

1. 上传一张图像
2. 启用 CLAHE 图像增强（增强强度设为 5.0）
3. 执行分割
4. 查看分割掩膜
5. **验证**：掩膜应该显示原始图像的色彩，而不是增强后的色彩

---

## 错误信息说明

### U-Net 不可用的错误信息

如果 U-Net 不可用，你会看到以下错误信息之一：

1. **模型文件不存在**
   ```
   U-Net segmentation failed: unet_weights_not_found: /path/to/model.pth
   ```
   解决：检查模型文件路径，确保文件存在

2. **模型格式是 state_dict**
   ```
   U-Net segmentation failed: unet_state_dict_requires_model_definition:
   The model file appears to be a state_dict without architecture.
   Please save the complete model using torch.save(model, path).
   ```
   解决：使用 `fix_unet_model.py` 脚本修复模型文件

3. **模型推理失败**
   ```
   U-Net segmentation failed: unet_inference_failed: [具体错误]
   ```
   解决：检查模型架构和输入格式是否匹配

4. **PyTorch 不可用**
   ```
   U-Net segmentation failed: unet_torch_not_available: [错误信息]
   ```
   解决：安装 PyTorch 和 torchvision

---

## 文件更新清单

修改的文件：
- `dental_stitcher_v1/segmentation.py`
  - `segment_teeth()`: 添加 original_image 参数，抛出异常而不是回退
  - `_segment_alphadent()`: 使用原始图像显示掩膜
  - `_segment_unet()`: 使用原始图像显示掩膜
  - `_get_unet_model()`: 改进错误检测和消息

- `app.py`
  - 添加异常处理，显示详细错误信息
  - 分割失败时停止执行，显示错误

新增的文件：
- `fix_unet_model.py`: 模型修复脚本

---

## 需要帮助？

如果按照以上步骤仍然无法解决问题，请提供以下信息：

1. 完整的错误信息
2. 模型文件的大小和格式
3. 是否安装了 `segmentation-models-pytorch`
4. Python 和 PyTorch 版本

参考文档：
- `UNET_QUICKSTART.md` - U-Net 快速开始指南
- `pts/UNET_README.md` - U-Net 模型详细说明
- `fix_unet_model.py` - 模型修复脚本
