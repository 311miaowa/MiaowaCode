"""Miaowa 异常体系单元测试。

验证异常继承树、RetriableError Mixin、cause chaining、上下文属性、
以及 MiaowaError.__str__ 格式化。
"""

from __future__ import annotations

import pytest

from miaowa.core.exceptions import (
    AgentError,
    BinaryFileError,
    ConfigError,
    ConfigFormatError,
    ConfigMissingError,
    ContextLengthExceededError,
    FileSystemError,
    FileTooLargeError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMTimeoutError,
    LLMToolCallParseError,
    MiaowaError,
    ProjectFileNotFoundError,
    RetriableError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)


# ============================================================================
# 1. test_exception_hierarchy — isinstance 链检查
# ============================================================================


class TestExceptionHierarchy:
    """验证全部自定义异常的继承链是否符合设计文档。"""

    # -- 配置错误链 -------------------------------------------------------

    def test_config_missing_error_chain(self):
        """ConfigMissingError → ConfigError → MiaowaError → Exception。"""
        err = ConfigMissingError("missing key", key_name="api_key")
        assert isinstance(err, ConfigMissingError)
        assert isinstance(err, ConfigError)
        assert isinstance(err, MiaowaError)
        assert isinstance(err, Exception)

    def test_config_format_error_chain(self):
        """ConfigFormatError → ConfigError → MiaowaError → Exception。"""
        err = ConfigFormatError("bad format", file_path="/tmp/x.yaml")
        assert isinstance(err, ConfigFormatError)
        assert isinstance(err, ConfigError)
        assert isinstance(err, MiaowaError)
        assert isinstance(err, Exception)

    # -- LLM 错误链 -------------------------------------------------------

    def test_llm_auth_error_chain(self):
        """LLMAuthenticationError → LLMError → MiaowaError。"""
        err = LLMAuthenticationError("unauthorized", status_code=401)
        assert isinstance(err, LLMAuthenticationError)
        assert isinstance(err, LLMError)
        assert isinstance(err, MiaowaError)

    def test_llm_rate_limit_error_chain(self):
        """LLMRateLimitError → RetriableError + LLMError → MiaowaError。"""
        err = LLMRateLimitError("rate limited", retry_after=30.0)
        assert isinstance(err, LLMRateLimitError)
        assert isinstance(err, RetriableError)
        assert isinstance(err, LLMError)
        assert isinstance(err, MiaowaError)

    def test_llm_timeout_error_chain(self):
        """LLMTimeoutError → RetriableError + LLMError → MiaowaError。"""
        err = LLMTimeoutError("timeout", timeout_value=120.0)
        assert isinstance(err, LLMTimeoutError)
        assert isinstance(err, RetriableError)
        assert isinstance(err, LLMError)
        assert isinstance(err, MiaowaError)

    def test_llm_connection_error_chain(self):
        """LLMConnectionError → RetriableError + LLMError → MiaowaError。"""
        err = LLMConnectionError("connection refused", host="api.example.com")
        assert isinstance(err, LLMConnectionError)
        assert isinstance(err, RetriableError)
        assert isinstance(err, LLMError)
        assert isinstance(err, MiaowaError)

    def test_llm_response_parse_error_chain(self):
        """LLMResponseParseError → LLMError → MiaowaError（不可重试）。"""
        err = LLMResponseParseError("bad JSON", raw_body="<html>...</html>")
        assert isinstance(err, LLMResponseParseError)
        assert isinstance(err, LLMError)
        assert isinstance(err, MiaowaError)
        assert not isinstance(err, RetriableError)

    def test_llm_tool_call_parse_error_chain(self):
        """LLMToolCallParseError → LLMError → MiaowaError（不可重试）。"""
        err = LLMToolCallParseError("bad args", tool_name="search", raw_arguments="{")
        assert isinstance(err, LLMToolCallParseError)
        assert isinstance(err, LLMError)
        assert isinstance(err, MiaowaError)
        assert not isinstance(err, RetriableError)

    # -- 工具错误链 -------------------------------------------------------

    def test_tool_not_found_error_chain(self):
        """ToolNotFoundError → ToolError → MiaowaError。"""
        err = ToolNotFoundError("unknown_tool")
        assert isinstance(err, ToolNotFoundError)
        assert isinstance(err, ToolError)
        assert isinstance(err, MiaowaError)

    def test_tool_validation_error_chain(self):
        """ToolValidationError → ToolError → MiaowaError。"""
        err = ToolValidationError("bad param", param_name="count", expected="int", actual="str")
        assert isinstance(err, ToolValidationError)
        assert isinstance(err, ToolError)
        assert isinstance(err, MiaowaError)

    def test_tool_execution_error_chain(self):
        """ToolExecutionError → ToolError → MiaowaError。"""
        err = ToolExecutionError("runtime crash", tool_name="search")
        assert isinstance(err, ToolExecutionError)
        assert isinstance(err, ToolError)
        assert isinstance(err, MiaowaError)

    # -- 文件系统错误链 ---------------------------------------------------

    def test_project_file_not_found_error_chain(self):
        """ProjectFileNotFoundError → FileSystemError → MiaowaError。"""
        err = ProjectFileNotFoundError("not found", file_path="/tmp/x.py")
        assert isinstance(err, ProjectFileNotFoundError)
        assert isinstance(err, FileSystemError)
        assert isinstance(err, MiaowaError)

    def test_binary_file_error_chain(self):
        """BinaryFileError → FileSystemError → MiaowaError。"""
        err = BinaryFileError("binary file", file_path="/tmp/x.png")
        assert isinstance(err, BinaryFileError)
        assert isinstance(err, FileSystemError)
        assert isinstance(err, MiaowaError)

    def test_file_too_large_error_chain(self):
        """FileTooLargeError → FileSystemError → MiaowaError。"""
        err = FileTooLargeError("too large", file_size=5_000_000, max_size=1_048_576)
        assert isinstance(err, FileTooLargeError)
        assert isinstance(err, FileSystemError)
        assert isinstance(err, MiaowaError)

    # -- Agent 错误链 ------------------------------------------------------

    def test_context_length_exceeded_error_chain(self):
        """ContextLengthExceededError → AgentError → MiaowaError。"""
        err = ContextLengthExceededError(
            "context too long", current_tokens=200_000, max_tokens=128_000
        )
        assert isinstance(err, ContextLengthExceededError)
        assert isinstance(err, AgentError)
        assert isinstance(err, MiaowaError)

    # -- 交叉检查：非重试异常 ----------------------------------------------

    def test_non_retriable_exceptions(self):
        """验证不应重试的异常确实不实现 RetriableError。"""
        non_retriable = [
            ConfigMissingError("x", key_name="k"),
            ConfigFormatError("x", file_path="f"),
            LLMAuthenticationError("x", status_code=401),
            LLMResponseParseError("x"),
            LLMToolCallParseError("x"),
            ToolNotFoundError("x"),
            ToolValidationError("x"),
            ToolExecutionError("x"),
            ProjectFileNotFoundError("x", file_path="f"),
            BinaryFileError("x"),
            FileTooLargeError("x"),
            ContextLengthExceededError("x"),
        ]
        for err in non_retriable:
            assert not isinstance(err, RetriableError), (
                f"{type(err).__name__} 不应实现 RetriableError"
            )

    def test_miaowa_error_is_base_of_all(self):
        """所有自定义异常都是 MiaowaError 的（直接或间接）子类。"""
        all_exceptions = [
            MiaowaError("x"),
            ConfigMissingError("x", key_name="k"),
            ConfigFormatError("x"),
            LLMAuthenticationError("x"),
            LLMRateLimitError("x"),
            LLMTimeoutError("x"),
            LLMConnectionError("x"),
            LLMResponseParseError("x"),
            LLMToolCallParseError("x"),
            ToolNotFoundError("x"),
            ToolValidationError("x"),
            ToolExecutionError("x"),
            ProjectFileNotFoundError("x", file_path="f"),
            BinaryFileError("x"),
            FileTooLargeError("x"),
            AgentError("x"),
            ContextLengthExceededError("x"),
        ]
        for err in all_exceptions:
            assert isinstance(err, MiaowaError), (
                f"{type(err).__name__} 应是 MiaowaError 的子类"
            )

    def test_retriable_is_not_miaowa_error(self):
        """RetriableError 是独立 Mixin，不应继承 MiaowaError。"""
        assert not issubclass(RetriableError, MiaowaError)
        assert not issubclass(RetriableError, Exception)


# ============================================================================
# 2. test_retriable_mixin — 可重试异常的属性
# ============================================================================


class TestRetriableMixin:
    """RetriableError Mixin 的行为验证。"""

    def test_rate_limit_is_retriable(self):
        """LLMRateLimitError 可被 isinstance 识别为 RetriableError。"""
        err = LLMRateLimitError("rate limited", retry_after=5.0, quota_type="rpm")
        assert isinstance(err, RetriableError)
        assert err.retry_after == 5.0
        assert err.quota_type == "rpm"

    def test_timeout_is_retriable(self):
        """LLMTimeoutError 可被识别为 RetriableError。"""
        err = LLMTimeoutError("timeout", timeout_value=60.0)
        assert isinstance(err, RetriableError)
        assert err.timeout_value == 60.0

    def test_connection_is_retriable(self):
        """LLMConnectionError 可被识别为 RetriableError。"""
        err = LLMConnectionError("connection failed", host="api.deepseek.com")
        assert isinstance(err, RetriableError)
        assert err.host == "api.deepseek.com"

    def test_simulated_executor_retry_logic(self):
        """模拟 Agent Executor 根据 RetriableError 判断是否重试。"""
        def should_retry(error: BaseException) -> bool:
            return isinstance(error, RetriableError)

        assert should_retry(LLMRateLimitError("x")) is True
        assert should_retry(LLMTimeoutError("x")) is True
        assert should_retry(LLMConnectionError("x")) is True
        assert should_retry(LLMAuthenticationError("x")) is False
        assert should_retry(ConfigMissingError("x", key_name="k")) is False
        assert should_retry(ToolExecutionError("x")) is False

    def test_retriable_mixin_can_be_used_standalone(self):
        """RetriableError 可作为独立 Mixin 使用。"""

        class CustomRetriableError(RetriableError, MiaowaError):
            pass

        err = CustomRetriableError("test")
        assert isinstance(err, RetriableError)
        assert isinstance(err, MiaowaError)


# ============================================================================
# 3. test_exception_cause_chain — raise X from Y
# ============================================================================


class TestExceptionCauseChain:
    """验证异常链（__cause__ / __context__）。"""

    def test_raise_config_error_from_oserror(self):
        """ConfigFormatError 可携带 OS 级异常的 cause chain。"""
        original = OSError("Permission denied")
        try:
            raise ConfigFormatError("cannot read config", file_path="/tmp/c.yaml") from original
        except ConfigFormatError as exc:
            assert exc.__cause__ is original
            assert isinstance(exc.__cause__, OSError)

    def test_raise_tool_error_from_value_error(self):
        """ToolExecutionError 可携带 ValueError 的 cause chain。"""
        original = ValueError("invalid value")
        try:
            raise ToolExecutionError("tool failed", tool_name="search") from original
        except ToolExecutionError as exc:
            assert exc.__cause__ is original
            assert exc.tool_name == "search"

    def test_raise_without_explicit_cause(self):
        """不使用 from 时 __cause__ 为 None，但 __context__ 隐式设置。"""
        try:
            try:
                raise ValueError("inner")
            except ValueError:
                raise ConfigFormatError("outer", file_path="/tmp/c.yaml")
        except ConfigFormatError as exc:
            assert exc.__cause__ is None
            assert exc.__context__ is not None
            assert isinstance(exc.__context__, ValueError)

    def test_nested_cause_chain(self):
        """多层 cause chain 可追溯。"""
        root = OSError("disk full")
        mid = ConfigFormatError("yaml parse failed")  # 注意：ConfigFormatError 有必填参数 message
        # Manually set cause chain
        try:
            raise ConfigFormatError("yaml parse failed", file_path="/tmp/c.yaml") from root
        except ConfigFormatError as mid:
            try:
                raise ToolExecutionError("search crashed", tool_name="search") from mid
            except ToolExecutionError as top:
                assert top.__cause__ is mid
                assert top.__cause__.__cause__ is root

    def test_miaowa_error_supports_cause_in_message(self):
        """异常消息中可引用 cause。"""
        original = ValueError("bad data")
        try:
            raise ToolExecutionError(
                "search tool failed", tool_name="search", original_error=original
            ) from original
        except ToolExecutionError as exc:
            assert exc.original_error is original
            assert "bad data" in str(exc.original_error)


# ============================================================================
# 4. test_exception_context — 上下文属性
# ============================================================================


class TestExceptionContextAttributes:
    """验证各异常类携带的结构化上下文字段。"""

    def test_config_missing_error_has_key_name(self):
        """ConfigMissingError.key_name 存储缺失的配置键名。"""
        err = ConfigMissingError("missing", key_name="llm.api_key")
        assert err.key_name == "llm.api_key"

    def test_config_missing_error_key_name_none(self):
        """key_name 默认为 None。"""
        err = ConfigMissingError("missing")
        assert err.key_name is None

    def test_config_format_error_has_file_path(self):
        """ConfigFormatError.file_path 存储出错的配置文件路径。"""
        err = ConfigFormatError("bad yaml", file_path="/home/user/config.yaml")
        assert err.file_path == "/home/user/config.yaml"

    def test_llm_auth_error_has_status_code(self):
        """LLMAuthenticationError.status_code 存储 HTTP 状态码。"""
        err = LLMAuthenticationError("auth failed", status_code=401)
        assert err.status_code == 401

    def test_llm_auth_error_status_code_none(self):
        """status_code 默认为 None。"""
        err = LLMAuthenticationError("auth failed")
        assert err.status_code is None

    def test_llm_rate_limit_error_has_retry_after_and_quota(self):
        """LLMRateLimitError 携带 retry_after 与 quota_type。"""
        err = LLMRateLimitError("rate limited", retry_after=15.5, quota_type="tpm")
        assert err.retry_after == 15.5
        assert err.quota_type == "tpm"

    def test_llm_timeout_error_has_timeout_value(self):
        """LLMTimeoutError 携带 timeout_value。"""
        err = LLMTimeoutError("timeout", timeout_value=120.0)
        assert err.timeout_value == 120.0

    def test_llm_connection_error_has_host(self):
        """LLMConnectionError 携带 host。"""
        err = LLMConnectionError("connection failed", host="api.deepseek.com")
        assert err.host == "api.deepseek.com"

    def test_llm_response_parse_error_has_raw_body(self):
        """LLMResponseParseError 携带 raw_body。"""
        err = LLMResponseParseError("bad json", raw_body="<html>error</html>")
        assert err.raw_body == "<html>error</html>"

    def test_llm_tool_call_parse_error_has_tool_and_args(self):
        """LLMToolCallParseError 携带 tool_name 与 raw_arguments。"""
        err = LLMToolCallParseError(
            "bad args", tool_name="read_file", raw_arguments='{"path": 123}'
        )
        assert err.tool_name == "read_file"
        assert err.raw_arguments == '{"path": 123}'

    def test_tool_not_found_error_has_tool_name(self):
        """ToolNotFoundError 携带 tool_name。"""
        err = ToolNotFoundError("unknown_tool")
        assert err.tool_name == "unknown_tool"
        assert "unknown_tool" in str(err)

    def test_tool_validation_error_has_param_details(self):
        """ToolValidationError 携带 param_name、expected、actual。"""
        err = ToolValidationError(
            "bad type", param_name="count", expected="int", actual="str"
        )
        assert err.param_name == "count"
        assert err.expected == "int"
        assert err.actual == "str"

    def test_tool_execution_error_has_tool_name_and_original(self):
        """ToolExecutionError 携带 tool_name 与 original_error。"""
        orig = ValueError("something went wrong")
        err = ToolExecutionError("crashed", tool_name="search", original_error=orig)
        assert err.tool_name == "search"
        assert err.original_error is orig

    def test_project_file_not_found_has_file_path(self):
        """ProjectFileNotFoundError 携带 file_path。"""
        err = ProjectFileNotFoundError("not found", file_path="src/main.py")
        assert err.file_path == "src/main.py"

    def test_binary_file_error_has_file_path(self):
        """BinaryFileError 携带 file_path。"""
        err = BinaryFileError("binary", file_path="image.png")
        assert err.file_path == "image.png"

    def test_file_too_large_error_has_size_info(self):
        """FileTooLargeError 携带 file_size 与 max_size。"""
        err = FileTooLargeError("too big", file_size=5_000_000, max_size=1_048_576)
        assert err.file_size == 5_000_000
        assert err.max_size == 1_048_576

    def test_context_length_exceeded_error_has_token_info(self):
        """ContextLengthExceededError 携带 current_tokens 与 max_tokens。"""
        err = ContextLengthExceededError(
            "context overflow", current_tokens=200_000, max_tokens=128_000
        )
        assert err.current_tokens == 200_000
        assert err.max_tokens == 128_000


# ============================================================================
# 5. MiaowaError.__str__ 格式化
# ============================================================================


class TestMiaowaErrorStrFormatting:
    """验证 MiaowaError.__str__ 自动附加上下文字段。"""

    def test_str_with_context_attributes(self):
        """带有上下文字段的异常在 str() 中显示字段信息。"""
        err = ConfigMissingError("API Key 缺失", key_name="llm.api_key")
        s = str(err)
        assert "API Key 缺失" in s
        assert "key_name" in s
        assert "llm.api_key" in s

    def test_str_without_context_attributes(self):
        """无额外上下文字段的异常只显示消息本身。"""
        err = ConfigFormatError("格式错误")
        s = str(err)
        assert "格式错误" in s
        # 有 file_path=None 的会被 __str__ 中的条件跳过（value is not None）
        # 所以这里只包含消息文本

    def test_str_with_multiple_fields(self):
        """多字段异常按字母序显示所有非 None 字段。"""
        err = FileTooLargeError("文件过大", file_size=100, max_size=50)
        s = str(err)
        assert "file_size=100" in s
        assert "max_size=50" in s

    def test_str_excludes_private_fields(self):
        """_ 开头的私有属性不出现在 str() 中。"""
        err = MiaowaError("test")
        err._internal = "secret"  # type: ignore[attr-defined]
        s = str(err)
        assert "secret" not in s

    def test_str_excludes_none_fields(self):
        """值为 None 的字段不出现在 str() 中。"""
        err = ConfigMissingError("missing key", key_name=None)  # type: ignore[arg-type]
        s = str(err)
        assert "key_name=None" not in s
        assert "key_name=" not in s  # None 字段完全不出现在格式化输出中

    def test_str_plain_miaowa_error(self):
        """纯 MiaowaError 仅显示消息。"""
        err = MiaowaError("something went wrong")
        assert str(err) == "something went wrong"


# ============================================================================
# 附加：异常的可 pickle 性（用于跨进程传递）
# ============================================================================


class TestExceptionPickle:
    """异常对象应可 pickle（便于跨进程或多线程场景）。"""

    def test_all_exceptions_are_picklable(self):
        """核心异常均可 pickle / unpickle，上下文属性正确恢复。"""
        import pickle

        exceptions = [
            ConfigMissingError("missing", key_name="k"),
            ConfigFormatError("format", file_path="f"),
            LLMAuthenticationError("auth", status_code=401),
            LLMRateLimitError("rate", retry_after=5.0, quota_type="rpm"),
            LLMTimeoutError("timeout", timeout_value=60.0),
            LLMConnectionError("conn", host="h"),
            LLMResponseParseError("parse", raw_body="b"),
            LLMToolCallParseError("tool parse", tool_name="t", raw_arguments="a"),
            ToolNotFoundError("t"),
            ToolValidationError("v", param_name="p", expected="e", actual="a"),
            ToolExecutionError("e", tool_name="t", original_error=ValueError("orig")),
            ProjectFileNotFoundError("f", file_path="p"),
            BinaryFileError("b", file_path="p"),
            FileTooLargeError("l", file_size=1, max_size=2),
            ContextLengthExceededError("c", current_tokens=1, max_tokens=2),
        ]

        for err in exceptions:
            restored = pickle.loads(pickle.dumps(err))
            assert type(restored) is type(err)
            # 验证上下文字段被正确恢复
            for attr in vars(err):
                if not attr.startswith("_"):
                    orig_val = getattr(err, attr)
                    rest_val = getattr(restored, attr)
                    # Exception 子类的 __eq__ 基于 identity，需逐个比较
                    if isinstance(orig_val, BaseException):
                        assert type(rest_val) is type(orig_val), (
                            f"{type(err).__name__}.{attr} 类型不匹配"
                        )
                        assert str(rest_val) == str(orig_val), (
                            f"{type(err).__name__}.{attr} 消息不匹配"
                        )
                    else:
                        assert rest_val == orig_val, (
                            f"{type(err).__name__}.{attr} 未正确恢复: "
                            f"{rest_val!r} != {orig_val!r}"
                        )
