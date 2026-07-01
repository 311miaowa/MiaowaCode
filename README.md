# Miaowa Code 🐱

轻量级终端 AI Agent 工具，基于 DeepSeek 大模型。

Miaowa Code 是一个运行在终端中的 AI 编程助手，灵感来源于 Claude Code。它能够理解你的代码库，执行文件操作和代码搜索，帮助你完成各种编程任务。

## ✨ 特性

- 🖥️ **终端原生** — 基于 `prompt-toolkit` 和 `rich` 构建的 TUI 体验
- 🤖 **DeepSeek 驱动** — 使用 DeepSeek 大模型，高性价比
- 🔧 **工具集成** — 内置文件读写、目录浏览、代码搜索、项目分析等工具
- 📁 **项目感知** — 自动理解项目结构和上下文
- ⚡ **异步架构** — 基于 `httpx` 的全异步设计

## 📦 安装

### 前置要求

- Python 3.10+

### 本地安装

```bash
# 进入项目目录
cd miaowa-code

# 安装依赖并注册 miaowa 命令
pip install -e .
```

## 🚀 使用

### 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 填入你的 API Key
# MIAOWA_API_KEY=sk-your-deepseek-api-key
```

### 启动

```bash
miaowa
```

进入 Miaowa 交互式终端后，你可以直接输入自然语言指令，例如：

- "分析当前项目结构"
- "解释 src/miaowa/main.py 的代码逻辑"
- "搜索项目中所有使用 async 函数的地方"

## 🔧 配置

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `MIAOWA_API_KEY` | DeepSeek API Key（必填） | — |
| `MIAOWA_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `MIAOWA_MODEL` | 默认模型 | `deepseek-chat` |

## 📁 项目结构

```
miaowa-code/
├── README.md                         # 项目说明
├── LICENSE                           # MIT 许可证
├── pyproject.toml                    # 项目配置
├── .gitignore                        # Git 忽略规则
├── .env.example                      # 环境变量模板
├── .editorconfig                     # 编辑器配置
│
├── docs/                             # 文档
│   ├── PRD.md                        # 产品需求文档
│   ├── ARCHITECTURE.md               # 架构设计文档
│   └── DEVELOPMENT.md                # 开发指南
│
├── src/miaowa/                       # 主包
│   ├── main.py                      # 入口点
│   ├── cli/                          # CLI 层 — 命令行接口
│   │   ├── repl.py                   # REPL 交互循环
│   │   ├── renderer.py               # 终端渲染器
│   │   ├── parser.py                 # 命令解析器
│   │   └── history.py                # 命令历史管理
│   ├── agent/                        # Agent 层 — 核心逻辑
│   │   ├── planner.py                # 任务规划器
│   │   ├── executor.py               # Tool 执行器
│   │   ├── context.py                # 上下文构建器
│   │   └── session.py                # 会话管理
│   ├── tools/                        # Tool 层 — 工具集
│   │   ├── base.py                   # Tool 基类
│   │   ├── registry.py               # Tool 注册中心
│   │   ├── validator.py              # 参数校验器
│   │   ├── filesystem.py             # 文件系统工具
│   │   ├── search.py                 # 搜索工具
│   │   └── analyzer.py               # 项目分析器
│   ├── llm/                          # LLM 适配层
│   │   ├── base.py                   # LLM 抽象基类
│   │   ├── deepseek.py               # DeepSeek 适配器
│   │   ├── types.py                  # 类型定义
│   │   └── tokenizer.py              # Token 计数
│   ├── core/                         # 核心基础设施
│   │   ├── config.py                 # 配置管理
│   │   ├── logger.py                 # 日志系统
│   │   └── exceptions.py             # 自定义异常
│   └── prompts/                      # 提示词模板
│       ├── system.py                 # 系统提示词
│       └── manager.py                # Prompt 管理器
│
├── tests/                            # 测试
│   ├── conftest.py                   # Pytest fixtures
│   ├── test_cli/                     # CLI 层测试
│   ├── test_agent/                   # Agent 层测试
│   ├── test_tools/                   # Tool 层测试
│   ├── test_llm/                     # LLM 层测试
│   └── test_core/                    # Core 层测试
│
├── fixtures/                         # 测试夹具
│   └── sample_python_project/        # 示例 Python 项目
│
└── scripts/                          # 辅助脚本
    ├── install.sh                    # 安装脚本 (Linux/macOS)
    └── install.ps1                   # 安装脚本 (Windows)
```

## 🧪 开发

```bash
# 安装开发依赖
pip install pytest pytest-asyncio pytest-cov ruff mypy

# 运行测试
pytest

# 代码检查
ruff check src/
mypy src/
```

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件。
