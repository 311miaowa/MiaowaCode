"""Prompt 管理器 — 模板注册、格式化与缓存。

提供 PromptManager 类，集中管理所有提示词模板，支持：
- 动态注册自定义模板
- 通过名称索引和关键字参数格式化模板
- 内置系统提示词的便捷获取方法

Usage::

    # 获取系统提示词
    prompt = PromptManager.get_system_prompt("/home/user/project")

    # 注册自定义模板
    PromptManager.register_template("greeting", "你好，{user_name}！")

    # 格式化模板
    msg = PromptManager.format_prompt("greeting", user_name="小明")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, ClassVar

from miaowa.prompts.system import SYSTEM_PROMPT_TEMPLATE


class PromptManager:
    """Prompt 管理器。

    以类级别的模板注册表为核心，提供模板的注册、格式化与缓存功能。
    所有方法均为类方法，调用方无需实例化即可使用。

    设计原则：
        - 内置模板在类初始化时自动注册（system_prompt 等）
        - 支持运行时动态添加新模板
        - 格式化失败时给出明确诊断信息，不静默吞错

    Attributes (class-level):
        _templates: 模板注册表，名称 → 模板字符串的映射。
        _builtin_names: 内置模板名称集合，用于区分内置与自定义模板。
    """

    _templates: ClassVar[dict[str, str]] = {}
    _builtin_names: ClassVar[set[str]] = set()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_initialized(cls) -> None:
        """延迟初始化内置模板（首次调用时触发）。

        延迟初始化的好处：
            - 避免模块导入时的副作用
            - 确保 system.py 中的 TEMPLATE 在注册前已完全加载
        """
        if "system_prompt" not in cls._templates:
            cls._register_builtin("system_prompt", SYSTEM_PROMPT_TEMPLATE)

    @classmethod
    def _register_builtin(cls, name: str, template: str) -> None:
        """注册内置模板（内部方法）。

        与 register_template 的区别：
            - 不发出「覆盖内置模板」的警告
            - 将名称加入 _builtin_names 集合
        """
        cls._templates[name] = template
        cls._builtin_names.add(name)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @classmethod
    def get_system_prompt(
        cls,
        current_dir: str | Path,
        provider: str = "DeepSeek",
        model: str = "deepseek-chat",
    ) -> str:
        """获取格式化后的系统提示词。

        内置模板 ``system_prompt`` 需要 ``current_dir``、``provider``、``model``
        三个参数。该方法是对 ``format_prompt("system_prompt", ...)`` 的语义化封装。

        Args:
            current_dir: 当前工作目录路径。str 或 Path 对象均可。
            provider: LLM 提供商名称。
            model: 模型名称。

        Returns:
            格式化后的系统提示词字符串。

        Raises:
            KeyError: 内置模板意外缺失时（理论上不应发生）。
        """
        cls._ensure_initialized()
        return cls.format_prompt(
            "system_prompt",
            current_dir=str(current_dir),
            provider=provider,
            model=model,
        )

    @classmethod
    def register_template(cls, name: str, template: str) -> None:
        """注册（或覆盖）一个提示词模板。

        模板字符串使用 Python ``str.format()`` 语法，
        通过 ``{param_name}`` 占位符定义可变部分。

        覆盖规则：
            - 覆盖已有自定义模板 → 静默覆盖
            - 覆盖内置模板 → 发出 UserWarning 警告，但仍执行覆盖

        Args:
            name: 模板名称（唯一标识符）。建议使用 snake_case 命名。
            template: 模板字符串，支持 ``{param}`` 格式占位符。

        Raises:
            ValueError: name 为空字符串时。
            TypeError: template 不是字符串时。

        Example:
            >>> PromptManager.register_template(
            ...     "code_review",
            ...     "请审查以下 {language} 代码：\\n```\\n{code}\\n```"
            ... )
        """
        if not name:
            raise ValueError("模板名称不能为空字符串")

        if not isinstance(template, str):
            raise TypeError(
                f"模板必须为 str 类型，实际类型: {type(template).__name__}"
            )

        # 覆盖内置模板时发出警告
        if name in cls._builtin_names:
            warnings.warn(
                f"正在覆盖内置模板 '{name}'。"
                f"这将改变系统默认行为，请确认这是预期操作。",
                UserWarning,
                stacklevel=2,
            )

        cls._templates[name] = template

    @classmethod
    def format_prompt(cls, template_name: str, **kwargs: Any) -> str:
        """按名称查找模板并格式化。

        查找已注册的模板，使用传入的关键字参数填充占位符。

        Args:
            template_name: 模板名称。
            **kwargs: 模板格式化参数。多余的参数会被忽略，
                缺失的必填参数会触发 KeyError。

        Returns:
            格式化后的提示词字符串。

        Raises:
            KeyError: 模板名称未注册时。
            KeyError: 模板所需的参数未提供时（由 str.format() 抛出）。
            ValueError: 模板格式化失败时（如存在格式字符串语法错误）。

        Example:
            >>> PromptManager.register_template("ask", "请解释 {topic}")
            >>> PromptManager.format_prompt("ask", topic="闭包")
            '请解释 闭包'
        """
        cls._ensure_initialized()

        if template_name not in cls._templates:
            available = ", ".join(sorted(cls._templates.keys()))
            raise KeyError(
                f"模板 '{template_name}' 未注册。"
                f"当前可用模板: [{available}]"
            )

        template = cls._templates[template_name]

        try:
            return template.format(**kwargs)
        except KeyError as e:
            import re

            expected = re.findall(r"\{(\w+)\}", template)
            raise KeyError(
                f"格式化模板 '{template_name}' 时缺少必需参数: {e}。"
                f"模板需要的参数: {expected}，"
                f"实际传入: {list(kwargs.keys())}"
            ) from e
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"格式化模板 '{template_name}' 失败: {e}。"
                f"请检查模板字符串中的占位符语法是否正确。"
            ) from e

    @classmethod
    def list_templates(cls) -> dict[str, str]:
        """列出所有已注册的模板名称（内置 / 自定义分别标注）。

        Returns:
            字典，键为模板名称，值为 ``"builtin"`` 或 ``"custom"``。
        """
        cls._ensure_initialized()
        result: dict[str, str] = {}
        for name in cls._templates:
            result[name] = "builtin" if name in cls._builtin_names else "custom"
        return result

    @classmethod
    def remove_template(cls, name: str) -> None:
        """移除一个已注册的模板。

        内置模板不可移除以保证系统行为稳定；尝试移除内置模板会抛出 ValueError。

        Args:
            name: 要移除的模板名称。

        Raises:
            KeyError: 模板名称未注册时。
            ValueError: 尝试移除内置模板时。

        Example:
            >>> PromptManager.register_template("temp", "临时: {msg}")
            >>> PromptManager.remove_template("temp")  # OK
            >>> PromptManager.remove_template("system_prompt")  # 抛出 ValueError
        """
        cls._ensure_initialized()

        if name not in cls._templates:
            available = ", ".join(sorted(cls._templates.keys()))
            raise KeyError(
                f"模板 '{name}' 未注册。当前可用模板: [{available}]"
            )

        if name in cls._builtin_names:
            raise ValueError(
                f"不允许移除内置模板 '{name}'。"
                f"如需修改内置模板，请使用 register_template() 覆盖。"
            )

        del cls._templates[name]
