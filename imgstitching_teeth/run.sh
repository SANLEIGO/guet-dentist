#!/bin/bash

# 口腔内窥镜拼接应用启动脚本

# 检查虚拟环境
if [ -d ".venv310" ]; then
    VENV_DIR=".venv310"
elif [ -d ".venv" ]; then
    VENV_DIR=".venv"
else
    echo "❌ 错误: 找不到虚拟环境 (.venv310 或 .venv)"
    exit 1
fi

echo "✓ 虚拟环境: $VENV_DIR"

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 检查依赖
echo "检查依赖..."
python -c "import ultralytics; print('✓ ultralytics 版本:', ultralytics.__version__)" 2>&1 || {
    echo "❌ 错误: ultralytics 未安装"
    echo "请运行: pip install -r requirements.txt"
    exit 1
}

python -c "import streamlit; print('✓ streamlit 版本:', streamlit.__version__)" 2>&1 || {
    echo "❌ 错误: streamlit 未安装"
    echo "请运行: pip install -r requirements.txt"
    exit 1
}

echo ""
echo "========================================"
echo "🚀 启动口腔内窥镜拼接应用..."
echo "========================================"
echo ""

# 启动 Streamlit
streamlit run app.py "$@"
