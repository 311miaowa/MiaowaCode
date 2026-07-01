"""Tool 注册中心 — 全局工具注册、查询与管理。

典型用法::

    from miaowa.tools.registry import ToolRegistry
    from miaowa.tools.base import BaseTool

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(SearchCodeTool())

    # 查询单个工具
    tool = registry.get("read_file")

    # 生成 LLM API 所需的工具定义列表
    tools_schema = registry.get_definitions()

    # 生成系统提示词
    prompt_text = registry.get_system_prompt_text()
"""

from __future__ import annotations

from miaowa.core.exceptions import ToolNotFoundError
from miaowa.tools.base import BaseTool

# ------------------------------------------------------------------
# 类型 → Python isinstance 谓词（用于注册期 enum 一致性校验）
# ------------------------------------------------------------------
# bool 是 int 的子类，必须优先检查 boolean 类型，
# 否则 isinstance(True, int) 返回 True 会导致整数校验误判。

_TYPE_CHECKERS: dict[str, callable] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


class ToolRegistry:
    """工具注册中心。

    维护工具名称到 BaseTool 实例的映射，提供注册、查询、
    OpenAI schema 批量导出与系统提示词生成功能。

    Attributes:
        _tools: 内部有序字典，键为工具名称，值为 BaseTool 实例。
            使用 dict 保证 Python 3.7+ 的插入顺序保留。
    """

    def __init__(self) -> None:
        """初始化空的工具注册表。"""
        self._tools: dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """注册一个工具实例。

        若工具名已存在，覆盖旧实例（并记录警告日志）。

        注册前执行定义期校验：
        - 工具类型检查（必须是 BaseTool 子类实例）
        - enum 值与声明类型的一致性检查

        Args:
            tool: BaseTool 子类实例。

        Raises:
            TypeError: 传入对象不是 BaseTool 子类实例。
            ToolValidationError: enum 值与声明的参数类型不一致。
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"register() 仅接受 BaseTool 子类实例，收到: {type(tool).__name__!r}"
            )

        # 定义期校验：enum 值必须与声明的参数类型一致
        self._validate_enum_types(tool)

        name = tool.name
        if name in self._tools:
            # 不引入硬依赖 logger，直接使用预留的日志调用模式
            import sys

            print(
                f"[ToolRegistry] 工具 {name!r} 已注册，将被覆盖。",
                file=sys.stderr,
            )
        self._tools[name] = tool

    @staticmethod
    def _validate_enum_types(tool: BaseTool) -> None:
        """校验工具的 enum 值与声明的参数类型一致。

        若参数的 enum 列表中包含与 type 不兼容的值，
        这些值永远无法通过运行时的类型检查，属于定义错误。
        """
        from miaowa.core.exceptions import ToolValidationError

        for param in tool.parameters:
            if param.enum is None:
                continue
            checker = _TYPE_CHECKERS.get(param.type)
            if checker is None:
                continue  # 未知类型跳过（运行时校验会捕获）
            for value in param.enum:
                # enum 值必须是字符串（JSON Schema 规范），但 Python
                # 中可能混入其他类型。这里仅做类型一致性检查。
                if not checker(value):
                    raise ToolValidationError(
                        f"工具 {tool.name!r} 参数 {param.name!r} 的 "
                        f"枚举值 {value!r} 与声明类型 {param.type!r} 不一致。"
                        f" 请修正 enum 列表或参数 type。",
                        param_name=param.name,
                        expected=param.type,
                        actual=value,
                    )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseTool:
        """按名称获取工具实例。

        Args:
            name: 工具名称（如 "read_file"）。

        Returns:
            对应的 BaseTool 子类实例。

        Raises:
            ToolNotFoundError: 指定名称的工具未注册。
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def list_all(self) -> list[BaseTool]:
        """列出所有已注册的工具。

        Returns:
            按注册顺序排列的 BaseTool 实例列表。
        """
        return list(self._tools.values())

    # ------------------------------------------------------------------
    # 批量导出
    # ------------------------------------------------------------------

    def get_definitions(self) -> list[dict]:
        """获取所有工具的 OpenAI Function Calling schema 列表。

        遍历所有已注册工具，依次调用 BaseTool.to_openai_schema()，
        返回可直接传给 LLM API 的 tools 参数列表。

        Returns:
            dict 列表，每个元素为一个工具的 OpenAI schema。

        Example::

            registry = ToolRegistry()
            registry.register(ReadFileTool())
            schema = registry.get_definitions()
            # → [{"type": "function", "function": {...}}, ...]
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def get_system_prompt_text(self) -> str:
        """生成所有工具的系统提示词文本。

        将所有已注册工具的 BaseTool.to_system_prompt() 输出拼接，
        工具之间以双换行分隔。

        Returns:
            格式化的多工具描述字符串，可直接嵌入系统提示词。
        """
        return "\n\n".join(
            tool.to_system_prompt() for tool in self._tools.values()
        )

    # ------------------------------------------------------------------
    # 魔术方法
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """返回已注册工具数量。"""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """支持 name in registry 语法。"""
        return name in self._tools

    def __getitem__(self, name: str) -> BaseTool:
        """支持 registry["tool_name"] 语法。

        与 get() 等价，未找到时抛出 ToolNotFoundError。
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]
