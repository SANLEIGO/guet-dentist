# 口腔内窥镜图像拼接 v1

基于深度学习的口腔内窥镜图像分割与拼接系统。

## ✨ 功能特性

### 🔪 牙齿分割
- **深度学习方法**：使用 AlphaDent (YOLOv8) 进行牙齿区域检测
- **精细化处理**：可选 GrabCut 边界优化
- **可视化反馈**：绿色覆盖层直观显示分割结果
- **多种查看模式**：并排对比、网格展示、单独查看

### 🔗 图像拼接
- **标准化流水线**：分割 → 特征提取 → 配准 → 融合
- **多特征方法**：支持 ORB、AKAZE、SIFT、LoFTR
- **质量评估**：清晰度、曝光度、匹配质量指标
- **详细诊断**：完整的 JSON 诊断数据

## 🚀 快速开始

### 方法 1: 使用启动脚本（推荐）

```bash
./run.sh
```

启动脚本会自动：
- ✅ 设置环境变量 `DENTAL_SEG_WEIGHTS`
- ✅ 检查模型文件是否存在
- ✅ 激活虚拟环境（.venv310 或 .venv）
- ✅ 检查依赖是否安装
- ✅ 启动 Streamlit 应用

### 方法 2: 手动启动

```bash
# 1. 设置环境变量（必需）
export DENTAL_SEG_WEIGHTS=$(pwd)/pts/alphadent_9cls_960.pt

# 2. 激活虚拟环境
source .venv310/bin/activate

# 3. 安装依赖（首次运行）
pip install -r requirements.txt

# 4. 启动应用
streamlit run app.py
```

## 📖 使用流程

### 1. 上传图像
- 在左侧侧边栏点击「上传多张口腔内窥镜图像」
- 支持 JPG、PNG、BMP、TIF 等格式
- 建议上传 2-4 张同一牙弓的图像

### 2. 调整参数（可选）
- **分割设置**：
  - 置信度阈值：0.01-0.5，默认 0.1
  - GrabCut 精细化：默认启用
- **拼接设置**：
  - 特征方法：ORB（推荐）、AKAZE、SIFT、LoFTR

### 3. 执行分割
- 点击「执行分割」按钮
- 查看分割结果对比（原图 vs 掩膜）
- 选择查看模式：并排对比、网格展示、单独查看

### 4. 下载分割结果（可选）
- 下载所有掩膜（ZIP 格式）
- 下载分割报告 JSON

### 5. 开始拼接
- 确认分割效果满意后
- 点击「开始拼接」按钮
- 查看拼接结果和诊断信息

### 6. 下载结果
- 下载拼接图 PNG
- 下载诊断 JSON

## 📁 项目结构

```
0311-imgsconjec/
├── app.py                          # Streamlit 应用入口
├── run.sh                          # 启动脚本
├── requirements.txt                # Python 依赖
├── README.md                       # 项目说明
├── USAGE.md                        # 详细使用指南
├── dental_stitcher_v1/             # 核心拼接模块
│   ├── segmentation.py             # AlphaDent + GrabCut 分割
│   ├── features.py                 # 特征提取（ORB/AKAZE/SIFT）
│   ├── registration.py             # 图像配准（单应性/仿射）
│   ├── blending.py                 # 羽化融合
│   ├── pipeline.py                 # 完整流水线
│   ├── visualization.py            # 可视化工具
│   ├── diagnostics.py              # 诊断数据结构
│   └── io_utils.py                 # I/O 工具
└── pts/
    └── alphadent_9cls_960.pt      # AlphaDent 模型权重（137MB）
```

## ⚙️ 技术参数

### 分割
- **模型**：AlphaDent (YOLOv8)
- **输入尺寸**：960x960
- **默认置信度**：0.1
- **后处理**：GrabCut (3次迭代)

### 特征提取
- **ORB**：2000 特征点，scaleFactor=1.2
- **AKAZE**：默认参数
- **SIFT**：2500 特征点，对比度 0.02

### 配准
- **方法**：单应性矩阵 (RANSAC)
- **回退**：仿射变换
- **阈值**：4.0 像素
- **最少点数**：8 对

### 融合
- **方法**：高斯羽化
- **羽化半径**：30 像素

## ⚠️ 注意事项

### 环境要求
- **Python**：3.9 或 3.10
- **依赖**：见 `requirements.txt`
- **环境变量**：必须设置 `DENTAL_SEG_WEIGHTS`

### 图像质量
- ✅ 确保图像清晰、曝光适中
- ✅ 相邻图像需要有 30-50% 的重叠
- ✅ 按从左到右或从右到左的顺序采集

### 当前限制
- ⚠️ 仅支持 **2 张图像**的拼接
- ⚠️ 不支持 LoFTR（未实现）
- ⚠️ 需要下载 137MB 的模型文件

## 🐛 故障排查

### 分割失败（全绿掩膜）
**原因**：AlphaDent 模型未加载

**解决方案**：
1. 检查环境变量：`echo $DENTAL_SEG_WEIGHTS`
2. 确认模型文件存在：`ls -lh pts/alphadent_9cls_960.pt`
3. 使用启动脚本：`./run.sh`

### 拼接失败
**原因**：图像重叠不足或质量不佳

**解决方案**：
1. 确保两张图像有充分重叠（30-50%）
2. 检查图像清晰度
3. 尝试更换特征方法（从 ORB 改为 SIFT）

### ultralytics 未安装
**解决方案**：
```bash
source .venv310/bin/activate
pip install ultralytics>=8.1.0
```

## 📝 更新日志

### v1.0 (2026-03-21)
- ✅ 完善前端分割交互逻辑
- ✅ 删除旧版拼接器和龋齿 ROI 逻辑
- ✅ 统一使用 AlphaDent + GrabCut 分割
- ✅ 新增 run.sh 启动脚本
- ✅ 新增 USAGE.md 使用指南

## 📄 许可证

本项目仅供学习和研究使用。
