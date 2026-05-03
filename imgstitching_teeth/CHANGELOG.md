# 更新日志

## [v1.1] - 2026-03-31

### 新增功能 🎉

#### U-Net 模型支持
- ✅ 新增 U-Net 深度学习模型作为可选项的分割方法
- ✅ 用户可以在前端界面选择使用 AlphaDent (YOLOv8) 或 U-Net 进行分割
- ✅ U-Net 模型支持 GrabCut 精细化
- ✅ U-Net 模型支持 CLAHE 图像增强
- ✅ 自动回退机制：U-Net 不可用时自动使用 AlphaDent

#### 前端界面改进
- ✅ 新增分割模型选择器（AlphaDent vs U-Net）
- ✅ 根据选择的模型动态启用/禁用相关参数
- ✅ 更新分割说明，显示当前使用的模型和配置

#### 新增文档
- ✅ `pts/UNET_README.md` - U-Net 模型使用说明
  - 模型格式要求
  - 输入输出规范
  - 训练建议
  - 故障排查
- ✅ `test_unet.py` - U-Net 功能测试脚本
- ✅ `unet_demo.py` - U-Net 分割演示脚本

### 技术改进 🔧

#### segmentation.py
- 重构 `segment_teeth()` 函数，支持多种分割方法
- 新增参数：
  - `method`: 分割方法选择 ("alphadent" 或 "unet")
  - `use_grabcut`: 是否使用 GrabCut 精细化
  - `use_enhancement`: 是否使用 CLAHE 图像增强
  - `enhancement_level`: CLAHE 增强强度
- 新增 `_segment_unet()` 函数：U-Net 模型推理
- 新增 `_get_unet_model()` 函数：U-Net 模型加载和缓存
- 新增 `_apply_clahe()` 函数：CLAHE 图像增强
- 更新 `_segment_alphadent()` 函数：支持可选的 GrabCut

#### app.py
- 新增分割模型选择下拉框
- 更新分割说明，动态显示当前配置
- 更新分割调用，传递所有新参数

#### 配置文件
- ✅ `.env`: 新增 `UNET_WEIGHTS` 配置项
- ✅ `requirements.txt`: 新增 `torchvision>=0.17` 依赖

### 文档更新 📝

#### README.md
- 更新功能特性，说明两种分割方法
- 更新使用流程，说明模型选择
- 更新项目结构，包含 U-Net 模型文件
- 更新技术参数，添加 U-Net 相关配置
- 更新限制说明，提醒用户需要自备 U-Net 模型
- 更新故障排查，添加 U-Net 相关问题

### 使用说明 📖

#### 准备 U-Net 模型
1. 训练或下载 U-Net 模型
2. 将模型文件放置在 `pts/unet_model.pth`
3. 或在 `.env` 中自定义路径：`UNET_WEIGHTS=/path/to/model.pth`

#### 使用 U-Net 分割
1. 启动应用：`./run.sh`
2. 上传图像
3. 在"分割设置"中选择"U-Net"
4. 调整参数（GrabCut、CLAHE 等）
5. 点击"执行分割"

#### 测试 U-Net 功能
```bash
# 测试模型加载和分割
python test_unet.py --image_path test_image.jpg

# 对比 AlphaDent 和 U-Net
python test_unet.py --image_path test_image.jpg --compare

# 运行演示脚本
python unet_demo.py test_image.jpg my_test
```

### 兼容性 ✅

- ✅ 完全向后兼容，不影响现有 AlphaDent 功能
- ✅ U-Net 不可用时自动回退到 AlphaDent
- ✅ 支持所有现有特性（GrabCut、CLAHE、拼接等）

### 已知问题 ⚠️

1. **U-Net 模型需要用户自备**
   - 项目不包含预训练的 U-Net 模型
   - 用户需要自己训练或从其他来源获取
   - 参考 `pts/UNET_README.md` 了解模型要求

2. **U-Net 模型格式**
   - 当前实现假设标准 U-Net 输出格式
   - 如果您的模型格式不同，可能需要调整 `_segment_unet()` 函数

3. **内存占用**
   - U-Net 推理可能需要较多内存
   - 建议在 GPU 环境下使用，或使用较小的输入图像

### 未来改进 💡

- [ ] 支持更多分割模型（如 DeepLab, Mask R-CNN）
- [ ] 添加模型性能对比工具
- [ ] 支持自定义 U-Net 架构配置
- [ ] 添加模型训练脚本
- [ ] 支持模型集成（ensemble）

### 贡献者 👥

- Claude Code Assistant

---

## [v1.0] - 2026-03-21

### 初始版本
- ✅ AlphaDent (YOLOv8) 分割
- ✅ GrabCut 精细化
- ✅ CLAHE 图像增强
- ✅ ORB/AKAZE/SIFT 特征提取
- ✅ 图像配准和拼接
- ✅ Streamlit 前端界面
- ✅ 质量评估和诊断
