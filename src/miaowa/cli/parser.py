"""CLI 命令解析器 — 将用户输入解析为 ParsedCommand 结构化对象。

解析规则：
- 以 "/" 开头的输入为元命令（meta），用于控制 REPL 行为。
- 其余输入为自然语言（natural），将作为对话内容传递给 Agent。
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# ParsedCommand — 解析结果数据结构
# ---------------------------------------------------------------------------


@dataclass
class ParsedCommand:
    """CLI 输入解析结果。

    Attributes:
        type: 命令类型。取值：
            - "meta": 元命令（以 "/" 开头），用于控制 REPL 本身。
            - "natural": 自然语言输入，将作为对话内容传递给 Agent。
        command: 元命令名称（含前导 "/"），如 "/model"。
            type="natural" 时为 None。
        args: 元命令的参数列表，按空格分词。
            type="natural" 时为 None。
        content: 自然语言对话内容（即原始输入去除首尾空白）。
            type="meta" 时为 None。
        is_known: 该元命令是否为已知（已注册）命令。
            type="natural" 时始终为 True。
        parse_error: 解析过程中的警告/错误信息。
            仅在 shlex 解析失败、输入过长等场景下非 None。
    """

    type: str
    command: str | None = None
    args: list[str] | None = None
    content: str | None = None
    is_known: bool = True
    parse_error: str | None = None

    def __post_init__(self) -> None:
        """校验 type 取值与相关字段的互斥性。"""
        if self.type not in ("meta", "natural"):
            raise ValueError(
                f"无效的 type 值: {self.type!r}，必须是 'meta' 或 'natural'"
            )
        if self.type == "meta":
            if self.command is None or not self.command.startswith("/"):
                raise ValueError(
                    f"type='meta' 时 command 必须是以 '/' 开头的元命令名称，"
                    f"实际: {self.command!r}"
                )
        if self.type == "natural" and self.content is None:
            raise ValueError("type='natural' 时 content 不能为 None")


# ---------------------------------------------------------------------------
# CommandParser
# ---------------------------------------------------------------------------


class CommandParser:
    """CLI 命令解析器。

    将用户输入文本解析为 ParsedCommand 对象，
    区分元命令（/ 开头）和自然语言对话输入。

    支持通过 register_command() 注册自定义元命令。

    Usage::

        parser = CommandParser()
        result = parser.parse("/model deepseek-chat")
        # ParsedCommand(type="meta", command="/model",
        #               args=["deepseek-chat"], content=None, is_known=True)

        result = parser.parse("帮我写一个排序函数")
        # ParsedCommand(type="natural", command=None,
        #               args=None, content="帮我写一个排序函数")
    """

    # ------------------------------------------------------------------
    # 类级别默认值（实例化时复制到实例属性，避免跨实例共享）
    # ------------------------------------------------------------------

    # 默认元命令表：命令名 → 中文描述
    META_COMMANDS: dict[str, str] = {
        "/quit": "退出 Miaowa Code",
        "/exit": "退出 Miaowa Code（同 /quit）",
        "/clear": "清空对话历史",
        "/help": "显示帮助信息",
        "/model": "显示或切换当前模型",
        "/tokens": "显示 Token 用量统计",
        "/cost": "显示 API 费用估算",
        "/cache": "显示缓存状态信息",
        "/debug": "切换调试模式",
    }

    # 默认可接受参数的元命令集合
    _COMMANDS_WITH_ARGS: frozenset[str] = frozenset({"/model", "/help"})

    # 最大输入长度（字符数），超出将截断并附加 parse_error
    MAX_INPUT_LENGTH: int = 10_000

    def __init__(self) -> None:
        """初始化解析器，将类级别默认值复制为实例属性。"""
        self._meta_commands: dict[str, str] = dict(self.META_COMMANDS)
        self._commands_with_args: set[str] = set(self._COMMANDS_WITH_ARGS)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def parse(self, user_input: str) -> ParsedCommand:
        """解析用户输入，返回结构化的 ParsedCommand。

        解析规则：
        - 以 "/" 开头 → meta 命令：按空格分词，第一个 token 为命令名，
          其余为参数。is_known 指示该命令是否在注册表中。
        - 否则 → natural 输入：整个文本作为对话内容。

        Args:
            user_input: 用户输入的原始文本。空字符串或全空白输入将返回
                type="natural"、content="" 的 ParsedCommand。

        Returns:
            ParsedCommand 实例。当输入超长时，content 被截断并设置
            parse_error。

        Examples::

            >>> p = CommandParser()
            >>> p.parse("/quit")
            ParsedCommand(type='meta', command='/quit', args=[],
                          content=None, is_known=True, parse_error=None)
            >>> p.parse("/model deepseek-chat")
            ParsedCommand(type='meta', command='/model',
                          args=['deepseek-chat'], content=None,
                          is_known=True, parse_error=None)
            >>> p.parse("你好")
            ParsedCommand(type='natural', command=None, args=None,
                          content='你好')
        """
        # 输入长度检查
        if len(user_input) > self.MAX_INPUT_LENGTH:
            truncated = user_input[: self.MAX_INPUT_LENGTH]
            result = self._parse_impl(truncated)
            result.parse_error = (
                f"输入过长（{len(user_input)} 字符），"
                f"已截断至 {self.MAX_INPUT_LENGTH} 字符"
            )
            return result

        return self._parse_impl(user_input)

    def _parse_impl(self, user_input: str) -> ParsedCommand:
        """内部解析实现（不检查长度）。"""
        stripped = user_input.strip()

        if not stripped:
            return ParsedCommand(type="natural", content=stripped)

        if self.is_meta_command(stripped):
            return self._parse_meta(stripped)

        return ParsedCommand(type="natural", content=stripped)

    def register_command(
        self, command: str, description: str, accepts_args: bool = False
    ) -> None:
        """注册自定义元命令，用于扩展内置命令集。

        Args:
            command: 命令名称（含前导 "/"），如 "/review"。
            description: 命令的中文描述，用于帮助文本。
            accepts_args: 该命令是否接受参数。

        Raises:
            ValueError: command 不以 "/" 开头。

        Example::

            parser = CommandParser()
            parser.register_command("/review", "代码审查", accepts_args=True)
        """
        if not command.startswith("/"):
            raise ValueError(f"元命令必须以 '/' 开头: {command!r}")
        self._meta_commands[command] = description
        if accepts_args:
            self._commands_with_args.add(command)

    def get_command_completions(self) -> dict[str, str]:
        """获取所有已注册命令及其描述（供 prompt_toolkit 补全器使用）。

        Returns:
            命令名 → 描述的映射字典副本。
        """
        return dict(self._meta_commands)

    def accepts_args(self, command: str) -> bool:
        """判断给定的元命令是否接受参数。

        Args:
            command: 元命令名称（含前导 "/"），如 "/model"。

        Returns:
            True 如果该命令在可接受参数的命令集合中。
        """
        return command in self._commands_with_args

    def get_command_description(self, command: str) -> str | None:
        """获取元命令的描述文本。

        Args:
            command: 元命令名称（含前导 "/"），如 "/model"。

        Returns:
            命令的中文描述；若命令未注册则返回 None。
        """
        return self._meta_commands.get(command)

    # ------------------------------------------------------------------
    # 判断方法
    # ------------------------------------------------------------------

    @staticmethod
    def is_meta_command(text: str) -> bool:
        """判断给定文本是否为元命令（以 "/" 开头）。

        Args:
            text: 待判断的文本。

        Returns:
            True 如果文本以 "/" 开头，否则 False。

        Examples::

            >>> CommandParser.is_meta_command("/quit")
            True
            >>> CommandParser.is_meta_command("你好")
            False
            >>> CommandParser.is_meta_command("")
            False
        """
        return len(text) > 0 and text[0] == "/"

    def is_known_command(self, command: str) -> bool:
        """判断给定的元命令是否为已知（已注册）命令。

        Args:
            command: 元命令名称（含前导 "/"），如 "/model"。

        Returns:
            True 如果命令在已注册的命令表中。
        """
        return command in self._meta_commands

    # ------------------------------------------------------------------
    # 帮助文本
    # ------------------------------------------------------------------

    def get_help_text(self) -> str:
        """生成元命令帮助文本。

        Returns:
            格式化的帮助字符串，按命令名排序，包含中英文描述。
            接受参数的命令会在描述后标注 [参数]。

        Example output::

            ──────────── 元命令帮助 ────────────

            /cache      显示缓存状态信息
            /clear      清空对话历史
            /debug      切换调试模式
            /exit       退出 Miaowa Code（同 /quit）
            /help       显示帮助信息 [参数]
            /model      显示或切换当前模型 [参数]
            /quit       退出 Miaowa Code
            /tokens     显示 Token 用量统计
            /cost       显示 API 费用估算

            输入以 / 开头的命令可执行对应操作，
            其他输入将作为自然语言传递给 AI Agent。
        """
        lines: list[str] = []
        lines.append("")
        lines.append("──────────── 元命令帮助 ────────────")
        lines.append("")

        # 按命令名排序
        for cmd in sorted(self._meta_commands.keys()):
            desc = self._meta_commands[cmd]
            if cmd in self._commands_with_args:
                desc += " [参数]"
            lines.append(f"  {cmd:<12}{desc}")

        lines.append("")
        lines.append("  输入以 / 开头的命令可执行对应操作，")
        lines.append("  其他输入将作为自然语言传递给 AI Agent。")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _parse_meta(self, text: str) -> ParsedCommand:
        """解析元命令输入。

        使用 shlex.split 进行安全的 shell 风格分词，
        支持带引号的参数（如 /model "gpt-4 turbo"）。
        shlex 解析失败时回退至 str.split 并记录 parse_error。

        Args:
            text: 已去除首尾空白的元命令文本（以 "/" 开头）。

        Returns:
            ParsedCommand(type="meta", ...)，附带 is_known 标记。
        """
        parse_error: str | None = None

        try:
            tokens = shlex.split(text)
        except ValueError:
            parse_error = (
                f"参数中包含未闭合的引号或非法转义，"
                f"已按空格分词处理，结果可能不符合预期"
            )
            tokens = text.split()

        # 此处 tokens 不可能为空：text 非空（由 parse() 中的 stripped
        # 检查保证）且 is_meta_command() 确保首字符为 "/"。
        command = tokens[0]
        args = tokens[1:] if len(tokens) > 1 else []

        is_known = command in self._meta_commands

        return ParsedCommand(
            type="meta",
            command=command,
            args=args,
            is_known=is_known,
            parse_error=parse_error,
        )
