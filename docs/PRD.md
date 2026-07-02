# Miaowa Code — 产品需求文档 (PRD)

> 📄 完整版 PRD 见 `docs/Miaowa_Code_PRD.docx`

## 概述

Miaowa Code 是一个轻量级终端 AI Agent 工具，灵感来源于 Claude Code，基于 DeepSeek 大模型。

## MVP 目标

- 终端原生的 AI 编程助手
- 基于 DeepSeek 的高性价比 LLM 集成
- 文件读写、Shell 执行、代码搜索等核心工具
- 项目感知与上下文理解

## 目标用户

1. **独立全栈开发者** — 需要快速理解/修改项目
2. **开源项目维护者** — 需要高效的代码审查与项目管理
3. **编程学习者** — 需要交互式代码解释与指导

## MVP 功能清单

| 编号 | 功能 | 优先级 |
|------|------|--------|
| F-001 | CLI 入口（`miaowa` 命令） | P0 |
| F-002 | AI 多轮对话 | P0 |
| F-003 | 目录扫描 | P0 |
| F-004 | 文件读取 | P0 |
| F-005 | 代码搜索 | P0 |
| F-006 | 项目分析 | P0 |

## 技术架构

- **语言**: Python ≥ 3.10
- **包管理**: Poetry
- **CLI 框架**: Rich + Prompt Toolkit
- **LLM**: DeepSeek Chat API（通过 openai SDK）
- **异步**: httpx + asyncio

## 详细文档

- 架构设计: [ARCHITECTURE.md](ARCHITECTURE.md)
- 开发指南: [DEVELOPMENT.md](DEVELOPMENT.md)
