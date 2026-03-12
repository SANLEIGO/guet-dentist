#!/usr/bin/env python3
"""
简单的语法检查
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def check_syntax():
    """检查语法"""
    files_to_check = [
        "dental_stitcher/enhanced_stitching.py",
        "dental_stitcher/advanced_stitching.py",
        "dental_stitcher/improved_stitching.py",
        "app.py"
    ]

    print("检查Python语法...")

    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # 编译代码检查语法
            compile(code, file_path, 'exec')
            print(f"✓ {file_path} 语法正确")
        except SyntaxError as e:
            print(f"✗ {file_path} 语法错误: {e}")
        except Exception as e:
            print(f"✗ {file_path} 错误: {e}")

    print("\n检查完成！")

if __name__ == "__main__":
    check_syntax()