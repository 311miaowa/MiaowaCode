"""CommandParser 单元测试。"""

from __future__ import annotations

import pytest

from miaowa.cli.parser import CommandParser, ParsedCommand


# ---------------------------------------------------------------------------
# ParsedCommand 构造与校验
# ---------------------------------------------------------------------------


class TestParsedCommand:
    """ParsedCommand 数据类的构造与校验测试。"""

    def test_meta_command_valid(self) -> None:
        """有效的 meta 命令应正常构造。"""
        cmd = ParsedCommand(type="meta", command="/quit", args=[])
        assert cmd.type == "meta"
        assert cmd.command == "/quit"
        assert cmd.args == []
        assert cmd.content is None
        assert cmd.is_known is True
        assert cmd.parse_error is None

    def test_natural_command_valid(self) -> None:
        """有效的 natural 命令应正常构造。"""
        cmd = ParsedCommand(type="natural", content="你好")
        assert cmd.type == "natural"
        assert cmd.command is None
        assert cmd.args is None
        assert cmd.content == "你好"
        assert cmd.is_known is True

    def test_invalid_type_raises(self) -> None:
        """无效的 type 值应抛出 ValueError。"""
        with pytest.raises(ValueError, match="无效的 type 值"):
            ParsedCommand(type="invalid", command=None, args=None, content=None)

    def test_meta_without_command_raises(self) -> None:
        """type='meta' 但 command 为 None 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="command 必须是以 '/' 开头"):
            ParsedCommand(type="meta", command=None, args=[])

    def test_meta_command_without_slash_raises(self) -> None:
        """type='meta' 但 command 不以 '/' 开头应抛出 ValueError。"""
        with pytest.raises(ValueError, match="command 必须是以 '/' 开头"):
            ParsedCommand(type="meta", command="quit", args=[])

    def test_natural_without_content_raises(self) -> None:
        """type='natural' 但 content 为 None 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="content 不能为 None"):
            ParsedCommand(type="natural", content=None)

    def test_parse_error_field_default(self) -> None:
        """parse_error 字段默认为 None。"""
        cmd = ParsedCommand(type="meta", command="/quit", args=[])
        assert cmd.parse_error is None

    def test_is_known_default_true(self) -> None:
        """is_known 字段默认为 True。"""
        cmd = ParsedCommand(type="meta", command="/quit", args=[])
        assert cmd.is_known is True


# ---------------------------------------------------------------------------
# CommandParser.is_meta_command
# ---------------------------------------------------------------------------


class TestIsMetaCommand:
    """is_meta_command 静态方法测试。"""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("/quit", True),
            ("/exit", True),
            ("/model", True),
            ("/help", True),
            ("/tokens", True),
            ("/cost", True),
            ("/cache", True),
            ("/debug", True),
            ("/unknown", True),  # 未知命令也是 meta
            ("/", True),          # 单斜杠也是 meta
            ("你好", False),
            ("hello", False),
            ("", False),
            (" /quit", False),    # 前导空格 → 不算 meta
            ("不是/命令", False),
        ],
    )
    def test_is_meta_command(self, text: str, expected: bool) -> None:
        """参数化测试：判断文本是否为元命令。"""
        assert CommandParser.is_meta_command(text) == expected


# ---------------------------------------------------------------------------
# CommandParser.parse — 自然语言输入
# ---------------------------------------------------------------------------


class TestParseNatural:
    """自然语言输入解析测试。"""

    def test_simple_text(self) -> None:
        parser = CommandParser()
        result = parser.parse("帮我写一个排序函数")
        assert result.type == "natural"
        assert result.command is None
        assert result.args is None
        assert result.content == "帮我写一个排序函数"

    def test_empty_string(self) -> None:
        parser = CommandParser()
        result = parser.parse("")
        assert result.type == "natural"
        assert result.content == ""

    def test_whitespace_only(self) -> None:
        parser = CommandParser()
        result = parser.parse("   ")
        assert result.type == "natural"
        assert result.content == ""

    def test_multiline_text(self) -> None:
        parser = CommandParser()
        result = parser.parse("第一行\n第二行\n第三行")
        assert result.type == "natural"
        assert result.content == "第一行\n第二行\n第三行"

    def test_text_with_leading_trailing_whitespace(self) -> None:
        parser = CommandParser()
        result = parser.parse("  你好世界  ")
        assert result.type == "natural"
        assert result.content == "你好世界"


# ---------------------------------------------------------------------------
# CommandParser.parse — 简单元命令（无参数）
# ---------------------------------------------------------------------------


class TestParseMetaSimple:
    """简单元命令（无参数）解析测试。"""

    @pytest.mark.parametrize(
        "user_input,expected_command",
        [
            ("/quit", "/quit"),
            ("/exit", "/exit"),
            ("/clear", "/clear"),
            ("/tokens", "/tokens"),
            ("/cost", "/cost"),
            ("/cache", "/cache"),
            ("/debug", "/debug"),
        ],
    )
    def test_no_arg_meta_commands(
        self, user_input: str, expected_command: str
    ) -> None:
        parser = CommandParser()
        result = parser.parse(user_input)
        assert result.type == "meta"
        assert result.command == expected_command
        assert result.args == []
        assert result.content is None
        assert result.is_known is True

    def test_unknown_meta_command(self) -> None:
        """未知的 / 命令也应被解析为 meta，is_known=False 供上层处理。"""
        parser = CommandParser()
        result = parser.parse("/unknown")
        assert result.type == "meta"
        assert result.command == "/unknown"
        assert result.args == []
        assert result.is_known is False

    def test_solo_slash(self) -> None:
        """单独的 '/' 符号 — 是 meta 但不是已知命令。"""
        parser = CommandParser()
        result = parser.parse("/")
        assert result.type == "meta"
        assert result.command == "/"
        assert result.args == []
        assert result.is_known is False


# ---------------------------------------------------------------------------
# CommandParser.parse — 带参数的元命令
# ---------------------------------------------------------------------------


class TestParseMetaWithArgs:
    """带参数的元命令解析测试。"""

    def test_model_with_single_arg(self) -> None:
        parser = CommandParser()
        result = parser.parse("/model deepseek-chat")
        assert result.type == "meta"
        assert result.command == "/model"
        assert result.args == ["deepseek-chat"]
        assert result.is_known is True

    def test_model_with_multiple_args(self) -> None:
        parser = CommandParser()
        result = parser.parse("/model deepseek-chat temperature=0.7")
        assert result.type == "meta"
        assert result.command == "/model"
        assert result.args == ["deepseek-chat", "temperature=0.7"]
        assert result.is_known is True

    def test_help_with_command_arg(self) -> None:
        parser = CommandParser()
        result = parser.parse("/help /model")
        assert result.type == "meta"
        assert result.command == "/help"
        assert result.args == ["/model"]
        assert result.is_known is True

    def test_quoted_arg(self) -> None:
        """带引号的参数应被 shlex 正确处理。"""
        parser = CommandParser()
        result = parser.parse('/model "deepseek chat v2"')
        assert result.type == "meta"
        assert result.command == "/model"
        assert result.args == ["deepseek chat v2"]
        assert result.parse_error is None

    def test_mixed_quoted_and_unquoted_args(self) -> None:
        """混合引号与无引号参数 — /cmd 非已知命令，is_known=False。"""
        parser = CommandParser()
        result = parser.parse('/cmd arg1 "arg 2" arg3')
        assert result.type == "meta"
        assert result.command == "/cmd"
        assert result.args == ["arg1", "arg 2", "arg3"]
        assert result.is_known is False

    def test_extra_spaces_between_args(self) -> None:
        """多余空格应被正确忽略。"""
        parser = CommandParser()
        result = parser.parse("/model   deepseek-chat    v3")
        assert result.type == "meta"
        assert result.command == "/model"
        assert result.args == ["deepseek-chat", "v3"]


# ---------------------------------------------------------------------------
# CommandParser.parse — shlex 解析失败/parse_error
# ---------------------------------------------------------------------------


class TestParseMetaShlexError:
    """shlex 解析异常时的处理测试。"""

    def test_unclosed_double_quote(self) -> None:
        """未闭合的双引号应触发 parse_error，并回退到空格分词。"""
        parser = CommandParser()
        result = parser.parse('/model "deepseek-chat')
        assert result.type == "meta"
        assert result.command == "/model"
        # 回退到 split()：引号被当作普通字符
        assert result.args == ['"deepseek-chat']
        assert result.parse_error is not None
        assert "未闭合的引号" in result.parse_error

    def test_unclosed_single_quote(self) -> None:
        """未闭合的单引号应触发 parse_error。"""
        parser = CommandParser()
        result = parser.parse("/model 'deepseek-chat")
        assert result.type == "meta"
        assert result.parse_error is not None
        assert "未闭合的引号" in result.parse_error

    def test_shlex_ok_no_error(self) -> None:
        """正确的引号使用不应产生 parse_error。"""
        parser = CommandParser()
        result = parser.parse('/model "deepseek-chat"')
        assert result.parse_error is None
        assert result.args == ["deepseek-chat"]


# ---------------------------------------------------------------------------
# CommandParser.parse — 输入长度限制
# ---------------------------------------------------------------------------


class TestParseInputLength:
    """输入长度限制测试。"""

    def test_input_at_limit_is_accepted(self) -> None:
        """恰好等于 MAX_INPUT_LENGTH 的输入应正常处理。"""
        parser = CommandParser()
        # 构造恰好在限制内的输入
        content = "x" * parser.MAX_INPUT_LENGTH
        result = parser.parse(content)
        assert result.parse_error is None

    def test_input_exceeds_limit_is_truncated(self) -> None:
        """超出 MAX_INPUT_LENGTH 的输入应截断并设 parse_error。"""
        parser = CommandParser()
        content = "x" * (parser.MAX_INPUT_LENGTH + 100)
        result = parser.parse(content)
        assert result.parse_error is not None
        assert "输入过长" in result.parse_error
        assert len(result.content) == parser.MAX_INPUT_LENGTH

    def test_short_input_no_truncation(self) -> None:
        """普通短输入不受影响。"""
        parser = CommandParser()
        result = parser.parse("hello")
        assert result.parse_error is None
        assert result.content == "hello"


# ---------------------------------------------------------------------------
# CommandParser.is_known_command
# ---------------------------------------------------------------------------


class TestIsKnownCommand:
    """is_known_command 方法测试。"""

    def test_known_commands(self) -> None:
        parser = CommandParser()
        for cmd in parser.META_COMMANDS:
            assert parser.is_known_command(cmd), f"{cmd} 应该是已知命令"

    def test_unknown_command(self) -> None:
        parser = CommandParser()
        assert not parser.is_known_command("/unknown")
        assert not parser.is_known_command("/foo")

    def test_without_slash(self) -> None:
        parser = CommandParser()
        assert not parser.is_known_command("quit")
        assert not parser.is_known_command("help")

    def test_after_register(self) -> None:
        """注册后应变为已知命令。"""
        parser = CommandParser()
        assert not parser.is_known_command("/review")
        parser.register_command("/review", "代码审查")
        assert parser.is_known_command("/review")


# ---------------------------------------------------------------------------
# CommandParser.accepts_args
# ---------------------------------------------------------------------------


class TestAcceptsArgs:
    """accepts_args 方法测试。"""

    def test_commands_with_args(self) -> None:
        parser = CommandParser()
        assert parser.accepts_args("/model") is True
        assert parser.accepts_args("/help") is True

    def test_commands_without_args(self) -> None:
        parser = CommandParser()
        assert parser.accepts_args("/quit") is False
        assert parser.accepts_args("/clear") is False
        assert parser.accepts_args("/tokens") is False

    def test_unknown_command(self) -> None:
        parser = CommandParser()
        assert parser.accepts_args("/unknown") is False

    def test_after_register_with_args(self) -> None:
        parser = CommandParser()
        parser.register_command("/review", "代码审查", accepts_args=True)
        assert parser.accepts_args("/review") is True

    def test_after_register_without_args(self) -> None:
        parser = CommandParser()
        parser.register_command("/review", "代码审查", accepts_args=False)
        assert parser.accepts_args("/review") is False


# ---------------------------------------------------------------------------
# CommandParser.register_command — 扩展机制
# ---------------------------------------------------------------------------


class TestRegisterCommand:
    """register_command 扩展机制测试。"""

    def test_register_new_command(self) -> None:
        parser = CommandParser()
        parser.register_command("/review", "代码审查")
        assert parser.is_known_command("/review")

    def test_register_command_with_args(self) -> None:
        parser = CommandParser()
        parser.register_command("/review", "代码审查", accepts_args=True)
        assert parser.is_known_command("/review")
        assert parser.accepts_args("/review")

    def test_register_updates_help_text(self) -> None:
        parser = CommandParser()
        parser.register_command("/review", "代码审查")
        text = parser.get_help_text()
        assert "/review" in text
        assert "代码审查" in text

    def test_register_without_slash_raises(self) -> None:
        parser = CommandParser()
        with pytest.raises(ValueError, match="元命令必须以 '/' 开头"):
            parser.register_command("review", "代码审查")

    def test_register_overwrite_existing(self) -> None:
        """注册同名命令应覆盖原有描述。"""
        parser = CommandParser()
        old_desc = parser._meta_commands["/quit"]
        parser.register_command("/quit", "自定义退出")
        assert parser._meta_commands["/quit"] == "自定义退出"
        assert parser._meta_commands["/quit"] != old_desc

    def test_register_does_not_affect_other_instances(self) -> None:
        """注册命令不应影响其他 CommandParser 实例。"""
        parser1 = CommandParser()
        parser2 = CommandParser()
        parser1.register_command("/review", "代码审查")
        assert parser1.is_known_command("/review")
        assert not parser2.is_known_command("/review")


# ---------------------------------------------------------------------------
# CommandParser.get_help_text
# ---------------------------------------------------------------------------


class TestGetHelpText:
    """get_help_text 方法测试。"""

    def test_contains_all_commands(self) -> None:
        parser = CommandParser()
        text = parser.get_help_text()
        for cmd in parser.META_COMMANDS:
            assert cmd in text, f"帮助文本应包含命令 {cmd}"

    def test_contains_header(self) -> None:
        parser = CommandParser()
        text = parser.get_help_text()
        assert "元命令帮助" in text

    def test_non_empty(self) -> None:
        parser = CommandParser()
        text = parser.get_help_text()
        assert len(text) > 0

    def test_sorted_output(self) -> None:
        """验证帮助文本中的命令按字母顺序排列。"""
        parser = CommandParser()
        text = parser.get_help_text()
        # 提取帮助文本中出现的命令（按行）
        commands_in_text = [
            line.strip().split()[0]
            for line in text.splitlines()
            if line.strip().startswith("/")
        ]
        assert commands_in_text == sorted(commands_in_text)

    def test_args_commands_have_hint(self) -> None:
        """接受参数的命令应在帮助中标示 [参数]。"""
        parser = CommandParser()
        text = parser.get_help_text()
        # /model 和 /help 应该标注 [参数]
        assert "/model" in text
        assert "/help" in text
        assert "[参数]" in text

    def test_non_args_commands_no_hint(self) -> None:
        """不接受参数的命令不应有 [参数] 标注。"""
        parser = CommandParser()
        text = parser.get_help_text()
        # 检查 /quit 行没有 [参数]
        for line in text.splitlines():
            if line.strip().startswith("/quit"):
                assert "[参数]" not in line


# ---------------------------------------------------------------------------
# CommandParser.META_COMMANDS 完整性
# ---------------------------------------------------------------------------


class TestMetaCommandsRegistry:
    """META_COMMANDS 注册表完整性测试。"""

    def test_all_commands_start_with_slash(self) -> None:
        parser = CommandParser()
        for cmd in parser.META_COMMANDS:
            assert cmd.startswith("/"), f"命令 {cmd} 必须以 '/' 开头"

    def test_all_descriptions_are_non_empty(self) -> None:
        parser = CommandParser()
        for cmd, desc in parser.META_COMMANDS.items():
            assert len(desc) > 0, f"命令 {cmd} 的描述不能为空"

    def test_minimum_expected_commands_present(self) -> None:
        """确保至少包含 PRD 中定义的核心元命令。"""
        parser = CommandParser()
        required = {"/quit", "/exit", "/clear", "/help", "/model",
                     "/tokens", "/cost", "/cache", "/debug"}
        missing = required - set(parser.META_COMMANDS.keys())
        assert not missing, f"缺少元命令: {missing}"
