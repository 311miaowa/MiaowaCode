"""Tool 参数校验器 — 验证 LLM 传入的工具参数是否符合定义。

在工具执行前对参数进行校验，确保：
- 必填参数完整
- 参数类型正确（含 bool/int 区分）
- 枚举值合法
- 无未知参数

Typical usage::

    from miaowa.tools.base import BaseTool
    from miaowa.tools.validator import ToolValidator

    validator = ToolValidator()
    validator.validate(my_tool, {"file_path": "/src/main.py"})
"""

from __future__ import annotations

from typing import Any

from miaowa.core.exceptions import ToolValidationError
from miaowa.tools.base import BaseTool, ToolParameter

# ============================================================================
# 类型映射
# ============================================================================

# Python 类型 → JSON Schema 类型名映射。
# **注意顺序**：bool 必须在 int 之前，因为 bool 是 int 的子类，
# isinstance(True, int) 返回 True，错误顺序会导致 boolean 被误判为 integer。
_PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
    list: "array",
    dict: "object",
}


def _get_display_type(value: Any) -> str:
    """获取值的可读 JSON Schema 类型名。

    遍历 _PYTHON_TO_JSON_TYPE 映射表，返回第一个匹配的类型名。
    未匹配时回退到 Python 类型的 __name__。

    Args:
        value: 任意 Python 对象。

    Returns:
        如 "string"、"integer"、"boolean"、"number"、"array"、"object"
        等 JSON Schema 类型名。
    """
    for py_type, json_name in _PYTHON_TO_JSON_TYPE.items():
        if isinstance(value, py_type):
            return json_name
    return type(value).__name__


# ============================================================================
# ToolValidator
# ============================================================================


class ToolValidator:
    """工具参数校验器。

    在执行工具前校验 LLM 传入的参数是否合法。
    校验失败时抛出 ToolValidationError，由上层 Agent Executor 捕获并反馈给 LLM。

    Usage::

        validator = ToolValidator()
        try:
            validator.validate(tool, arguments)
        except ToolValidationError as e:
            # 将错误信息反馈给 LLM，让其修正参数后重试
            ...
    """

    # -- 支持的类型及其 Python 运行时检查谓词 --------------------------------
    # 注意：bool 是 int 的子类，必须优先检查 boolean 类型，
    # 否则 isinstance(True, int) 返回 True 会导致整数校验误判。

    _TYPE_CHECKERS: dict[str, Any] = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }

    def validate(self, tool: BaseTool, arguments: dict[str, Any]) -> None:
        """校验工具参数，并填充非必填参数的默认值。

        按以下顺序处理：
        1. 未知参数检查 — arguments 中存在但未在 tool.parameters 中定义的键。
        2. 默认值填充 — 非必填参数未传入时，将 ToolParameter.default 写入 arguments。
        3. 必填参数检查 — tool.parameters 中 required=True 但未传入或为 None。
        4. 参数类型检查 — 传入值的 Python 类型与声明的 JSON Schema 类型不匹配。
        5. 枚举值检查 — 参数声明了 enum 限制但传入值不在允许范围内。
           **枚举值比较区分大小写**。

        Args:
            tool: 工具实例，提供 parameters 定义。
            arguments: LLM 传入的参数字典（键为参数名，值为参数值）。
                此字典会被原地修改：非必填参数的默认值会被写入。

        Raises:
            ToolValidationError: 参数校验失败时抛出。
                异常包含 param_name、expected、actual 字段，
                便于 Agent Executor 构造清晰的错误反馈。

        Note:
            - 值为 None 时跳过类型检查（None 视为未提供值）。
            - 值为 None 但参数 required=True 时，由必填检查捕获
              （None 表示参数未有效传入）。
            - 枚举值比较使用 Python 默认的相等比较（区分大小写），
              例如 enum=["create", "modify"] 会拒绝 "Create"。
        """
        # 构建参数名 → ToolParameter 的快速查找表
        param_map: dict[str, ToolParameter] = {p.name: p for p in tool.parameters}
        defined_names: set[str] = set(param_map.keys())

        # ------------------------------------------------------------------
        # 1. 检查未知参数
        # ------------------------------------------------------------------
        for arg_name in arguments:
            if arg_name not in defined_names:
                raise ToolValidationError(
                    f"工具 {tool.name!r} 不接受参数 {arg_name!r}。"
                    f" 可用参数: {', '.join(sorted(defined_names))}",
                    param_name=arg_name,
                    actual=arg_name,
                )

        # ------------------------------------------------------------------
        # 2. 填充默认值（仅对非必填、未传入且有默认值的参数）
        # ------------------------------------------------------------------
        for param in tool.parameters:
            if (
                not param.required
                and param.name not in arguments
                and param.default is not None
            ):
                arguments[param.name] = param.default

        # ------------------------------------------------------------------
        # 3. 检查必填参数
        # ------------------------------------------------------------------
        for param in tool.parameters:
            if param.required and (
                param.name not in arguments or arguments[param.name] is None
            ):
                raise ToolValidationError(
                    f"工具 {tool.name!r} 缺少必填参数 {param.name!r} "
                    f"({param.type}): {param.description}",
                    param_name=param.name,
                    expected=param.type,
                )

        # ------------------------------------------------------------------
        # 4 & 5. 检查参数类型 & 枚举值
        # ------------------------------------------------------------------
        for arg_name, arg_value in arguments.items():
            param = param_map[arg_name]

            # None 值跳过类型检查（仅 non-required 参数可能走到这里，
            # required=None 已在步骤 3 被拦截）
            if arg_value is None:
                continue

            # -- 类型检查 ----------------------------------------------------
            checker = self._TYPE_CHECKERS.get(param.type)
            if checker is not None and not checker(arg_value):
                actual_display = _get_display_type(arg_value)
                raise ToolValidationError(
                    f"工具 {tool.name!r} 参数 {param.name!r} 类型错误: "
                    f"期望 {param.type}，实际为 {actual_display} "
                    f"(值: {arg_value!r})",
                    param_name=param.name,
                    expected=param.type,
                    actual=actual_display,
                )

            # -- 枚举值检查（区分大小写）--------------------------------------
            if param.enum is not None:
                if arg_value not in param.enum:
                    raise ToolValidationError(
                        f"工具 {tool.name!r} 参数 {param.name!r} 的值 "
                        f"{arg_value!r} 不在允许范围内。"
                        f" 可选值: {param.enum}"
                        f"（注意：枚举值比较区分大小写）",
                        param_name=param.name,
                        expected=f"one of {param.enum}",
                        actual=arg_value,
                    )
