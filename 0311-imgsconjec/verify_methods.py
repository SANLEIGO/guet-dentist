#!/usr/bin/env python3
"""
验证 CompatibleImprovedStitcher 类是否具有所有必需的方法
"""

def check_methods():
    """检查 CompatibleImprovedStitcher 类的方法"""

    # 读取文件内容
    with open('dental_stitcher/enhanced_stitching.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找 CompatibleImprovedStitcher 类定义
    class_start = content.find('class CompatibleImprovedStitcher:')
    if class_start == -1:
        print("错误: 找不到 CompatibleImprovedStitcher 类")
        return False

    # 提取类内容（到文件末尾或下一个类定义）
    class_content = content[class_start:]

    # 检查必需的方法
    required_methods = [
        '__init__',
        'precheck',
        'score_candidates',
        'stitch'
    ]

    all_found = True
    for method in required_methods:
        if f'def {method}(' in class_content:
            print(f"✓ 找到方法: {method}")
        else:
            print(f"✗ 缺少方法: {method}")
            all_found = False

    if all_found:
        print("\n所有必需的方法都已实现！")
        return True
    else:
        print("\n部分方法缺失，请检查实现。")
        return False

if __name__ == "__main__":
    check_methods()