#!/usr/bin/env bash
# ============================================================
# Miaowa Code — Pre-commit Hook
# 在每次提交前自动运行 lint 和测试
# ============================================================
set -euo pipefail

echo "🔍 运行 Pre-commit 检查..."

# Lint 检查
echo "→ Ruff lint..."
poetry run ruff check src/ tests/

# 格式检查
echo "→ Ruff format check..."
poetry run ruff format --check src/ tests/

# 类型检查
echo "→ Mypy 类型检查..."
poetry run mypy src/

# 运行测试
echo "→ 运行测试..."
poetry run pytest

echo "✅ 所有检查通过！"
