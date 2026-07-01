# Miaowa Code — 开发指南

## 环境准备

```bash
# 进入项目目录
cd miaowa-code

# 安装运行时依赖
pip install -e .

# 安装开发依赖
pip install pytest pytest-asyncio pytest-cov ruff mypy
```

## 开发命令

```bash
# 运行测试
pytest

# 代码检查
ruff check src/

# 类型检查
mypy src/

# 运行 Miaowa（开发模式）
miaowa
```

## 项目结构

参见 [ARCHITECTURE.md](ARCHITECTURE.md) 了解完整的模块分层与数据流设计。

## 编码规范

- Python ≥ 3.10，使用现代语法（`str | None`、`match/case` 等）
- 遵循 [EditorConfig](../.editorconfig) 配置（4 空格缩进、100 字符行长）
- 所有公开 API 需提供类型注解
- 使用 `ruff` 进行 lint，`mypy` 进行类型检查

## 分支策略

- `main` — 稳定发布分支
- `develop` — 开发集成分支
- `feature/*` — 功能分支
- `fix/*` — Bug 修复分支

## 发布流程

1. 更新 `src/miaowa/__init__.py` 中的 `__version__`
2. 更新 `pyproject.toml` 中的 `version`
3. 运行完整测试套件
4. 创建 Git Tag
5. 发布到 PyPI

## MVP 开发阶段

| 阶段 | 内容 | 状态 |
|------|------|:--:|
| Phase 1 | 核心 MVP（CLI 入口、多轮对话、文件读写、搜索、项目分析） | ✅ 完成 |
| Phase 2 | 打磨（测试 + 文档 + 日志） | 🚧 进行中 |
| Phase 3 | 扩展（Shell 执行、多模型切换等高级功能） | ⏳ 待开始 |
| Phase 4 | 生态（插件 + 社区） | ⏳ 待开始 |
