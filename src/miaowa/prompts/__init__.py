"""Prompts 模块 — 提示词模板管理。

提供系统提示词模板和 PromptManager 模板管理器。

Usage::

    from miaowa.prompts import PromptManager, SYSTEM_PROMPT_TEMPLATE

    # 获取系统提示词
    prompt = PromptManager.get_system_prompt("/path/to/project")

    # 注册自定义模板
    PromptManager.register_template("my_prompt", "请检查 {file_path}")
    msg = PromptManager.format_prompt("my_prompt", file_path="src/main.py")
"""

from miaowa.prompts.manager import PromptManager
from miaowa.prompts.system import SYSTEM_PROMPT_TEMPLATE, get_system_prompt

__all__ = [
    "SYSTEM_PROMPT_TEMPLATE",
    "PromptManager",
    "get_system_prompt",
]
