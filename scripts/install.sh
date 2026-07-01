#!/usr/bin/env bash
# ============================================================
# Miaowa Code — 一键安装脚本 (Linux / macOS)
# ============================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🐱 Miaowa Code 安装脚本${NC}"
echo "========================================"

# 检查 Python 版本
PYTHON_CMD=""
for cmd in python3.10 python3.11 python3.12 python3.13 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$("$cmd" -c "import sys; print(sys.version_info.major)")
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}错误: 未找到 Python >= 3.10${NC}"
    echo "请先安装 Python 3.10+: https://www.python.org/downloads/"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python: $($PYTHON_CMD --version)"

# 检查/安装 Poetry
if ! command -v poetry &>/dev/null; then
    echo -e "${YELLOW}正在安装 Poetry...${NC}"
    curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
fi

echo -e "${GREEN}✓${NC} Poetry: $(poetry --version)"

# 安装依赖
echo "正在安装项目依赖..."
poetry install --with dev

# 配置环境变量
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠${NC}  已创建 .env 文件，请编辑填入 DeepSeek API Key:"
    echo "   vi .env"
fi

echo ""
echo -e "${GREEN}✅ 安装完成！${NC}"
echo "运行 'poetry run miaowa' 启动 Miaowa Code"
