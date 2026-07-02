"""Miaowa 分层异常体系。

异常继承树::

    MiaowaError
    ├── ConfigError
    │   ├── ConfigMissingError                   [CRITICAL]
    │   └── ConfigFormatError
    ├── LLMError
    │   ├── LLMAuthenticationError
    │   ├── LLMRateLimitError                    [Retriable]
    │   ├── LLMTimeoutError                      [Retriable]
    │   ├── LLMConnectionError                   [Retriable]  ← NEW
    │   ├── LLMResponseParseError                ← 拆分自 LLMInvalidResponseError
    │   └── LLMToolCallParseError                ← 拆分自 LLMInvalidResponseError
    ├── ToolError
    │   ├── ToolNotFoundError
    │   ├── ToolValidationError
    │   └── ToolExecutionError
    ├── FileSystemError
    │   ├── ProjectFileNotFoundError             ← 原 FileNotFoundError_Miaowa
    │   ├── BinaryFileError
    │   └── FileTooLargeError
    └── AgentError
        └── ContextLengthExceededError

[Retriable] 标记表示异常实现了 RetriableError Mixin，
Agent Executor 可通过 isinstance(err, RetriableError) 判断是否自动重试。
"""

from __future__ import annotations

# ============================================================================
# 基类
# ============================================================================


class MiaowaError(Exception):
    """Miaowa 所有自定义异常的基类。

    触发场景：本项目内部任意模块在遇到可预期的错误条件时，
    均应抛出本类或其子类，而非直接使用通用 Exception。

    所有子类均支持 cause chaining（raise X from Y），
    通过 __cause__ 属性携带原始异常供上层日志与诊断。
    """

    def __str__(self) -> str:
        """格式化错误信息，自动附加结构化上下文字段。"""
        base = super().__str__()
        # 收集所有非 None、非私有、非 __cause__ 的实例属性
        extras: dict[str, object] = {}
        for key in sorted(self.__dict__):
            if key.startswith("_"):
                continue
            value = self.__dict__[key]
            if value is not None:
                extras[key] = value
        if extras:
            context = ", ".join(f"{k}={v!r}" for k, v in extras.items())
            return f"{base} [{context}]"
        return base


# ============================================================================
# 可重试标记 Mixin
# ============================================================================


class RetriableError:
    """Mixin：标记异常属于可自动重试的类型。

    使用方法:
        class LLMRateLimitError(RetriableError, LLMError):
            ...

    Agent Executor 可通过 ``isinstance(err, RetriableError)`` 判断是否重试，
    而无需维护硬编码的异常类型白名单。
    """


# ============================================================================
# 配置错误
# ============================================================================


class ConfigError(MiaowaError):
    """配置相关错误基类。"""


class ConfigMissingError(ConfigError):
    """必需配置项缺失。

    触发场景：
        - MIAOWA_API_KEY 环境变量未设置。
        - 必需的 config.yaml 文件不存在。
        - 关键配置键值对缺失。

    严重程度：CRITICAL — 通常导致应用无法启动。
    """

    def __init__(self, message: str, *, key_name: str | None = None) -> None:
        super().__init__(message)
        self.key_name = key_name


class ConfigFormatError(ConfigError):
    """配置文件格式错误。

    触发场景：
        - .env / YAML / TOML 文件语法错误或解析失败。
        - 配置值类型与预期不符（如 timeout 应为 int 但给了 str）。

    严重程度：ERROR — 可降级使用默认值继续运行。
    """

    def __init__(self, message: str, *, file_path: str | None = None) -> None:
        super().__init__(message)
        self.file_path = file_path


# ============================================================================
# LLM 调用错误
# ============================================================================


class LLMError(MiaowaError):
    """LLM 调用过程中的错误基类。"""


class LLMAuthenticationError(LLMError):
    """API Key 无效或认证失败。

    触发场景：
        - HTTP 401 Unauthorized。
        - API Key 被撤销、过期或格式不正确。
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMRateLimitError(RetriableError, LLMError):
    """API 速率限制。

    触发场景：
        - HTTP 429 Too Many Requests。
        - 超出账户的 RPM / TPM 配额。
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        quota_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.quota_type = quota_type  # "rpm" / "tpm"


class LLMTimeoutError(RetriableError, LLMError):
    """LLM 请求超时。

    触发场景：
        - httpx.TimeoutException。
        - 模型推理耗时超过配置的超时阈值。
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_value: float | None = None,
    ) -> None:
        super().__init__(message)
        self.timeout_value = timeout_value


class LLMConnectionError(RetriableError, LLMError):
    """LLM API 网络连接失败。

    触发场景：
        - DNS 解析失败。
        - TCP 连接被拒或超时（httpx.ConnectError）。
        - TLS / SSL 握手失败。
        - 代理配置错误。

    与 LLMTimeoutError 的区别：
        Timeout 是请求已发出但响应超时（应用层）；
        ConnectionError 是连接根本无法建立（传输层）。
    """

    def __init__(self, message: str, *, host: str | None = None) -> None:
        super().__init__(message)
        self.host = host


class LLMResponseParseError(LLMError):
    """LLM 响应体格式异常 — JSON 解析失败或缺少必要字段。

    触发场景：
        - 响应体不是合法 JSON（如 API 返回了 HTML 错误页）。
        - 响应 JSON 缺少顶层必要字段（如 choices）。
        - 响应结构变更（API 版本不兼容）。

    与 LLMToolCallParseError 的区别：
        本类关注 HTTP → JSON 的解析失败；
        LLMToolCallParseError 关注 JSON 内部 tool_calls 参数的格式问题。
    """

    def __init__(self, message: str, *, raw_body: str | None = None) -> None:
        super().__init__(message)
        self.raw_body = raw_body


class LLMToolCallParseError(LLMError):
    """工具调用参数解析失败 — arguments 不是合法 JSON 或不符合预期 schema。

    触发场景：
        - tool_calls[*].function.arguments 不是有效 JSON 字符串。
        - arguments 解析后的 dict 缺少必填键。
        - arguments 值类型与 ToolParameter.type 不匹配。

    与 LLMResponseParseError 的区别：
        本类关注从已解析的 JSON 响应中提取 tool call 参数时的错误；
        此时 HTTP 通信和 JSON 解析通常已成功。
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        raw_arguments: str | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.raw_arguments = raw_arguments


# ============================================================================
# 工具执行错误
# ============================================================================


class ToolError(MiaowaError):
    """工具执行过程中的错误基类。

    触发场景：Tool 注册、参数校验、执行等环节的异常。
    """


class ToolNotFoundError(ToolError):
    """请求的工具未在注册表中找到。

    触发场景：
        - LLM 请求调用一个未注册的 tool name。
        - 工具名称拼写错误或对应工具未加载。
    """

    def __init__(self, message: str, *, tool_name: str | None = None) -> None:
        super().__init__(message)
        self.tool_name = tool_name if tool_name is not None else message


class ToolValidationError(ToolError):
    """工具参数校验失败。

    触发场景：
        - 必填参数缺失。
        - 参数类型与 ToolParameter.type 不匹配。
        - 参数值不在 enum 允许范围内。
    """

    def __init__(
        self,
        message: str,
        *,
        param_name: str | None = None,
        expected: str | None = None,
        actual: object = None,
    ) -> None:
        super().__init__(message)
        self.param_name = param_name
        self.expected = expected
        self.actual = actual


class ToolExecutionError(ToolError):
    """工具运行时执行失败。

    触发场景：
        - 工具内部逻辑抛出未捕获的异常。
        - 外部依赖不可用（如网络请求失败、子系统崩溃）。
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        original_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.original_error = original_error


# ============================================================================
# 文件系统错误
# ============================================================================


class FileSystemError(MiaowaError):
    """文件系统操作错误基类。

    触发场景：文件读写、路径操作等环节的异常。

    Note:
        本类与 ToolError 平级（而非其子类），因为文件系统错误
        也可能在 Tool 层之外触发（如 Config 加载时的文件读取）。
    """


class ProjectFileNotFoundError(FileSystemError):
    """项目内目标文件不存在。

    触发场景：
        - 尝试读取一个不存在的项目文件路径。
        - 尝试写入到不存在的目录（且未启用自动创建）。

    与 Python 内置 FileNotFoundError (OSError 子类) 的区别：
        内置版表示任意 OS 级文件不存在；本类专指项目上下文中的文件定位失败，
        语义上更具体，便于 UI 层给出更有针对性的错误提示。
    """

    def __init__(self, message: str, *, file_path: str | None = None) -> None:
        super().__init__(message)
        self.file_path = file_path


class BinaryFileError(FileSystemError):
    """目标文件为二进制格式，无法按文本处理。

    触发场景：
        - 尝试以文本模式读取图片、压缩包等二进制文件。
        - 文件编码检测失败。
    """

    def __init__(self, message: str, *, file_path: str | None = None) -> None:
        super().__init__(message)
        self.file_path = file_path


class FileTooLargeError(FileSystemError):
    """文件大小超出允许上限。

    触发场景：
        - 文件大小超过配置的 max_file_size。
        - 上下文窗口不足以容纳文件内容。
    """

    def __init__(
        self,
        message: str,
        *,
        file_size: int | None = None,
        max_size: int | None = None,
    ) -> None:
        super().__init__(message)
        self.file_size = file_size
        self.max_size = max_size


# ============================================================================
# Agent 层错误
# ============================================================================


class AgentError(MiaowaError):
    """Agent 层（Planner / Executor / ContextBuilder）错误基类。

    触发场景：Agent 推理循环、任务规划、上下文管理等环节的异常。
    PRD 第 5 章各组件（Planner、ContextBuilder、Executor、MemoryManager）
    在 MVP 阶段统一使用本类及其子类。
    """


class ContextLengthExceededError(AgentError):
    """上下文超出模型最大窗口限制。

    触发场景：
        - 对话历史 + 项目上下文 + 系统提示词的 token 数超过模型上下文窗口。
        - 单次加载的文件内容过大，无法纳入当前对话。
    """

    def __init__(
        self,
        message: str,
        *,
        current_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens
