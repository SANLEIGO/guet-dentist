# Session Context (2026-03-16)

## ✅ 项目状态概览
- 项目路径：`/Users/administrator/Downloads/teeth/0311-imgsconjec`
- 深度分割后端已替换为 **AlphaDent (YOLOv8)** + **GrabCut**。
- 权重路径环境变量：
  - `DENTAL_SEG_WEIGHTS=/Users/administrator/Downloads/teeth/0311-imgsconjec/pts/alphadent_9cls_960.pt`

## ✅ 已提交版本
- Commit：`46dc4f0`
- 内容：
  - `dental_stitcher_v1/segmentation.py`：AlphaDent + GrabCut 方案
  - `requirements.txt`：新增 `ultralytics>=8.1.0`

## ✅ 当前代码状态（最新回退）
- `segmentation.py` 已回退到稳定版本：
  - `conf=0.1`
  - 无失败重试（无 conf=0.01 / 提亮重试）
  - 无 PR_FGD 扩张
  - GrabCut 背景种子为原版（红软组织 + 暗软组织 + 边缘）
- `app_v1.py` 已修改为**显示全部 mask**（不再限制 4 张）

## ✅ 关键问题与现状
- AlphaDent+GrabCut 整体效果很好
- 但**暗牙可能只识别半截**
- 用户不想再动当前算法

## ✅ 用户希望的下一步方向
- 可能只在 **AlphaDent 失败时** 做轻度**亮度增强**（CV 图像增强）
- 暂时先停

## ✅ 测试样例文件（位于 `dental_stitcher_v1/`）
- `02right.png`
- `03middle.png`
- `54f78a9b38dbbd8899b87f3ba5ad58e672536313501a331aabe6458d.jpg`
- `4b23d1f...jpg`, `74bb29...jpg`, `02f57f...jpg` 等
