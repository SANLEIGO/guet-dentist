# 项目结构说明

## 核心文件

### 主应用程序
- `app.py` - Streamlit Web应用主程序（已更新支持改进算法）

### 改进的拼接算法
- `dental_stitcher/enhanced_stitching.py` - **主要改进算法**（推荐使用）
  - 多频段融合
  - 兼容原有接口
  - 完整的预处理和融合流程

### 备用算法实现
- `dental_stitcher/advanced_stitching.py` - 高级拼接器（备用版本）
- `dental_stitcher/improved_stitching.py` - 早期改进版本

### 原始算法
- `dental_stitcher/stitching.py` - 原始拼接算法实现

### 支持模块
- `dental_stitcher/models.py` - 数据模型定义
- `dental_stitcher/utils.py` - 工具函数
- `dental_stitcher/__init__.py` - 包初始化

## 文档文件

- `README.md` - 项目说明（已更新）
- `IMPROVEMENTS.md` - 详细技术改进说明
- `COMPLETION_SUMMARY.md` - 完成工作总结
- `PROJECT_STRUCTURE.md` - 本文件

## 测试和验证

- `test_improved.py` - 改进算法测试脚本
- `syntax_check.py` - Python语法检查脚本

## 配置文件

- `requirements.txt` - Python依赖列表

## 项目统计

- Python源文件: 6个
- 改进算法实现: 3个版本
- 文档文件: 4个
- 测试脚本: 2个

## 推荐使用

### 生产环境
使用 `dental_stitcher/enhanced_stitching.py` 中的 `CompatibleImprovedStitcher` 类

### 开发测试
使用 `test_improved.py` 进行功能测试
使用 `syntax_check.py` 进行语法验证

### 文档参考
- 快速开始: README.md
- 技术细节: IMPROVEMENTS.md
- 工作总结: COMPLETION_SUMMARY.md

## 版本信息

- 原始版本: v1.0 (dental_stitcher/stitching.py)
- 改进版本: v2.0 (dental_stitcher/enhanced_stitching.py)
- 更新日期: 2026-03-11
