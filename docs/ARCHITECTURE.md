# Miaowa Code — 架构设计文档

## 分层架构

```
┌─────────────────────────────────────────┐
│              CLI 层 (cli/)               │  ← 命令行接口、REPL、渲染
├─────────────────────────────────────────┤
│             Agent 层 (agent/)            │  ← 任务规划、执行循环、会话管理
├─────────────────────────────────────────┤
│             Tool 层 (tools/)             │  ← 工具定义、注册、执行、校验
├─────────────────────────────────────────┤
│             LLM 层 (llm/)                │  ← LLM 适配器、Token 计数
├─────────────────────────────────────────┤
│             Core 层 (core/)              │  ← 配置、日志、异常、类型
├─────────────────────────────────────────┤
│           Prompts 层 (prompts/)          │  ← 提示词模板管理
└─────────────────────────────────────────┘
```

## 模块职责

### CLI 层 (`src/miaowa/cli/`)
- `main.py` — 命令行入口，参数解析
- `repl.py` — REPL 交互循环
- `renderer.py` — Rich Markdown 渲染
- `parser.py` — 命令解析器
- `history.py` — 命令历史管理

### Agent 层 (`src/miaowa/agent/`)
- `planner.py` — 任务规划器（ReAct 模式）
- `executor.py` — Tool 调用执行器
- `context.py` — 上下文构建器
- `session.py` — 会话管理

### Tool 层 (`src/miaowa/tools/`)
- `base.py` — Tool 基类与参数定义
- `registry.py` — Tool 注册中心
- `validator.py` — 参数校验器
- `filesystem.py` — 文件系统工具
- `search.py` — 搜索工具
- `analyzer.py` — 项目分析器

### LLM 层 (`src/miaowa/llm/`)
- `base.py` — LLM 抽象基类
- `deepseek.py` — DeepSeek 适配器
- `types.py` — LLM 相关类型定义
- `tokenizer.py` — Token 近似计数

### Core 层 (`src/miaowa/core/`)
- `config.py` — 多层配置管理
- `logger.py` — loguru 日志系统
- `exceptions.py` — 自定义异常体系

### Prompts 层 (`src/miaowa/prompts/`)
- `system.py` — 系统提示词
- `manager.py` — Prompt 管理器

## 数据流

```
用户输入 → CLI Parser → Agent Planner → Context Builder
                                            ↓
                                      LLM Adapter → DeepSeek API
                                            ↓
                                      Agent Executor → Tool Call? → Tool Executor
                                            ↓
                                      Response → CLI Renderer → 终端输出
```
