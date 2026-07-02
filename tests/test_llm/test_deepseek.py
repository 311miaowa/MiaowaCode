"""DeepSeekAdapter 单元测试。

使用 pytest-mock mock AsyncOpenAI 客户端，覆盖：
非流式/流式请求、工具调用、异常映射、消息/工具格式化。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIConnectionError, APIError, APITimeoutError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import RateLimitError as OpenAIRateLimitError

from miaowa.core.config import LLMConfig
from miaowa.core.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMToolCallParseError,
)
from miaowa.llm.deepseek import DeepSeekAdapter, _extract_retry_after, _map_openai_error, _normalize_finish_reason
from miaowa.llm.types import (
    ChatResponse,
    Message,
    ToolCall,
)


# ---------------------------------------------------------------------------
# 辅助：构造 mock SDK 响应
# ---------------------------------------------------------------------------


def _mock_chat_completion(
    content: str = "Hello!",
    finish_reason: str = "stop",
    tool_calls: list | None = None,
    usage: dict | None = None,
) -> MagicMock:
    """构造一个符合 openai SDK ChatCompletion 结构的 mock 对象。"""
    completion = MagicMock()
    choice = MagicMock()
    message = MagicMock()

    message.content = content
    message.tool_calls = tool_calls
    choice.message = message
    choice.finish_reason = finish_reason

    completion.choices = [choice]

    if usage:
        usage_mock = MagicMock()
        usage_mock.prompt_tokens = usage.get("prompt_tokens", 0)
        usage_mock.completion_tokens = usage.get("completion_tokens", 0)
        usage_mock.total_tokens = usage.get("total_tokens", 0)
        completion.usage = usage_mock
    else:
        completion.usage = None

    return completion


def _mock_tool_call_sdk(id: str, name: str, arguments: str) -> MagicMock:
    """构造一个符合 openai SDK tool_call 结构的 mock 对象。"""
    tc = MagicMock()
    tc.id = id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _mock_stream_chunk(
    delta_content: str | None = None,
    finish_reason: str | None = None,
    tool_call_delta: MagicMock | None = None,
) -> MagicMock:
    """构造一个符合 openai SDK ChatCompletionChunk 的 mock 对象。"""
    chunk = MagicMock()
    choice = MagicMock()
    delta = MagicMock()

    delta.content = delta_content
    delta.tool_calls = [tool_call_delta] if tool_call_delta else None
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk.choices = [choice]
    return chunk


def _make_adapter(mocker, model="deepseek-v4-flash", **overrides):
    """创建 DeepSeekAdapter，其 AsyncOpenAI 客户端已被 mock。"""
    mock_client = MagicMock()
    create_mock = AsyncMock()
    mock_client.chat.completions.create = create_mock

    mocker.patch("miaowa.llm.deepseek.AsyncOpenAI", return_value=mock_client)

    config_kwargs = dict(
        api_key="sk-test-key",
        base_url="https://api.deepseek.com/v1",
        model=model,
        temperature=0.3,
        max_tokens=4096,
        timeout=120,
    )
    config_kwargs.update(overrides)
    config = LLMConfig(**config_kwargs)

    adapter = DeepSeekAdapter(config)
    return adapter, mock_client, create_mock


# ============================================================================
# 1. test_chat_success — mock 正常 ChatCompletion 响应
# ============================================================================


class TestChatSuccess:

    @pytest.mark.asyncio
    async def test_chat_returns_content(self, mocker):
        """正常响应包含 content 文本。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)
        create_mock.return_value = _mock_chat_completion(
            content="你好，有什么可以帮助你的？",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        )

        response = await adapter.chat([Message(role="user", content="你好")])

        assert isinstance(response, ChatResponse)
        assert response.content == "你好，有什么可以帮助你的？"
        assert response.finish_reason == "stop"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        assert response.tool_calls is None

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_calls_api_with_correct_params(self, mocker):
        """验证 API 调用参数正确传递。"""
        adapter, mock_client, create_mock = _make_adapter(
            mocker, model="deepseek-v4-pro", temperature=0.7, max_tokens=2048,
        )
        create_mock.return_value = _mock_chat_completion()

        await adapter.chat([Message(role="user", content="test")])

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["model"] == "deepseek-v4-pro"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["stream"] is False
        assert "messages" in call_kwargs

        await adapter.close()


# ============================================================================
# 2. test_chat_with_tools — mock 含 tool_calls 的响应
# ============================================================================


class TestChatWithTools:

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self, mocker):
        """响应包含 tool_calls 时正确解析为 ToolCall 列表。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)
        mock_tc = _mock_tool_call_sdk("call_001", "read_file", '{"path": "main.py"}')
        create_mock.return_value = _mock_chat_completion(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[mock_tc],
            usage={"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
        )

        response = await adapter.chat([Message(role="user", content="读取 main.py")])

        assert response.finish_reason == "tool_calls"
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_001"
        assert response.tool_calls[0].name == "read_file"
        assert response.tool_calls[0].arguments == {"path": "main.py"}
        assert response.content is None

        await adapter.close()

    @pytest.mark.asyncio
    async def test_chat_with_multiple_tool_calls(self, mocker):
        """多个 tool_calls 全部被解析。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)
        tc1 = _mock_tool_call_sdk("c1", "search", '{"query": "error"}')
        tc2 = _mock_tool_call_sdk("c2", "read_file", '{"path": "log.txt"}')
        create_mock.return_value = _mock_chat_completion(
            finish_reason="tool_calls",
            tool_calls=[tc1, tc2],
        )

        response = await adapter.chat([Message(role="user", content="debug")])

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[1].name == "read_file"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_malformed_tool_call_json_raises_parse_error(self, mocker):
        """tool_calls arguments 不是合法 JSON 时抛出 LLMToolCallParseError。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)
        bad_tc = _mock_tool_call_sdk("c1", "bad_tool", "{invalid json")
        create_mock.return_value = _mock_chat_completion(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[bad_tc],
        )

        with pytest.raises(LLMToolCallParseError) as exc_info:
            await adapter.chat([Message(role="user", content="test")])
        assert exc_info.value.tool_name == "bad_tool"
        assert exc_info.value.raw_arguments == "{invalid json"

        await adapter.close()


# ============================================================================
# 3. test_stream — mock 流式响应 chunks
# ============================================================================


class TestStream:

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, mocker):
        """流式请求逐块 yield StreamChunk。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)

        async def _mock_stream():
            for c in [
                _mock_stream_chunk(delta_content="你好"),
                _mock_stream_chunk(delta_content="，世界"),
                _mock_stream_chunk(delta_content="！", finish_reason="stop"),
            ]:
                yield c

        create_mock.return_value = _mock_stream()

        chunks = []
        async for chunk in adapter.stream([Message(role="user", content="hi")]):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta_content == "你好"
        assert chunks[1].delta_content == "，世界"
        assert chunks[2].delta_content == "！"
        assert chunks[2].finish_reason == "stop"

        await adapter.close()

    @pytest.mark.asyncio
    async def test_stream_empty_chunk_handled(self, mocker):
        """空 delta（如首个 chunk 仅含 role）不崩溃。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)

        async def _mock_stream():
            yield _mock_stream_chunk(delta_content=None)

        create_mock.return_value = _mock_stream()

        chunks = [c async for c in adapter.stream([Message(role="user", content="hi")])]

        assert len(chunks) == 1
        assert chunks[0].delta_content is None
        assert chunks[0].finish_reason is None

        await adapter.close()


# ============================================================================
# 4. test_stream_tool_call_accumulation — 分片 tool call delta 聚合
# ============================================================================


class TestStreamToolCallAccumulation:

    @pytest.mark.asyncio
    async def test_stream_tool_call_delta(self, mocker):
        """工具调用的增量信息按 chunk 传递。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)

        tc_delta1 = MagicMock()
        tc_delta1.index = 0
        tc_delta1.type = "function"
        tc_delta1.id = "call_abc"
        tc_delta1.function = MagicMock()
        tc_delta1.function.name = "search_files"
        tc_delta1.function.arguments = '{"query"'

        tc_delta2 = MagicMock()
        tc_delta2.index = 0
        tc_delta2.type = "function"
        tc_delta2.id = None
        tc_delta2.function = MagicMock()
        tc_delta2.function.name = None
        tc_delta2.function.arguments = ': "error"'

        tc_delta3 = MagicMock()
        tc_delta3.index = 0
        tc_delta3.type = "function"
        tc_delta3.id = None
        tc_delta3.function = MagicMock()
        tc_delta3.function.name = None
        tc_delta3.function.arguments = "}"

        async def _mock_stream():
            for c, tc in [
                (_mock_stream_chunk(tool_call_delta=tc_delta1), None),
                (_mock_stream_chunk(tool_call_delta=tc_delta2), None),
                (_mock_stream_chunk(tool_call_delta=tc_delta3, finish_reason="tool_calls"), None),
            ]:
                yield c

        create_mock.return_value = _mock_stream()

        chunks = []
        async for chunk in adapter.stream([Message(role="user", content="search")]):
            chunks.append(chunk)

        assert chunks[0].tool_call_delta is not None
        assert chunks[0].tool_call_delta["id"] == "call_abc"
        assert chunks[0].tool_call_delta["function"]["name"] == "search_files"
        assert chunks[0].tool_call_delta["function"]["arguments"] == '{"query"'

        assert chunks[1].tool_call_delta["function"]["arguments"] == ': "error"'
        assert chunks[2].tool_call_delta["function"]["arguments"] == "}"
        assert chunks[2].finish_reason == "tool_calls"

        await adapter.close()


# ============================================================================
# 5. test_auth_error_mapping — AuthenticationError → LLMAuthenticationError
# ============================================================================


class TestAuthErrorMapping:

    @pytest.mark.asyncio
    async def test_auth_error_mapped(self, mocker):
        """openai.AuthenticationError → LLMAuthenticationError。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)
        auth_err = OpenAIAuthenticationError(
            message="Invalid API Key",
            response=MagicMock(status_code=401),
            body=None,
        )
        create_mock.side_effect = auth_err

        with pytest.raises(LLMAuthenticationError) as exc_info:
            await adapter.chat([Message(role="user", content="hi")])

        assert "认证失败" in str(exc_info.value)
        assert exc_info.value.status_code == 401

        await adapter.close()

    def test_map_function_returns_auth_error(self):
        """_map_openai_error 直接调用也返回正确类型。"""
        err = OpenAIAuthenticationError(
            message="Bad key",
            response=MagicMock(status_code=401),
            body=None,
        )
        mapped = _map_openai_error(err)
        assert isinstance(mapped, LLMAuthenticationError)

    def test_api_error_401_maps_to_auth_error(self):
        """HTTP 401 的通用 APIError 也映射为认证错误。"""
        err = APIError(message="Unauthorized", request=MagicMock(), body=None)
        err.code = 401  # set on instance, not class

        mapped = _map_openai_error(err)
        assert isinstance(mapped, LLMAuthenticationError)


# ============================================================================
# 6. test_rate_limit_mapping — RateLimitError → LLMRateLimitError
# ============================================================================


class TestRateLimitMapping:

    @pytest.mark.asyncio
    async def test_rate_limit_error_mapped(self, mocker):
        """openai.RateLimitError → LLMRateLimitError。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)
        create_mock.side_effect = OpenAIRateLimitError(
            message="Too many requests",
            response=MagicMock(status_code=429),
            body=None,
        )

        with pytest.raises(LLMRateLimitError) as exc_info:
            await adapter.chat([Message(role="user", content="hi")])

        assert "速率限制" in str(exc_info.value)
        await adapter.close()

    def test_map_function_returns_rate_limit_error(self):
        """_map_openai_error 对 RateLimitError 返回 LLMRateLimitError。"""
        err = OpenAIRateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        mapped = _map_openai_error(err)
        assert isinstance(mapped, LLMRateLimitError)


# ============================================================================
# 7. test_timeout_mapping — APITimeoutError → LLMTimeoutError
# ============================================================================


class TestTimeoutMapping:

    @pytest.mark.asyncio
    async def test_timeout_error_mapped(self, mocker):
        """openai.APITimeoutError → LLMTimeoutError。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)
        create_mock.side_effect = APITimeoutError(request=MagicMock())

        with pytest.raises(LLMTimeoutError) as exc_info:
            await adapter.chat([Message(role="user", content="hi")])

        assert "超时" in str(exc_info.value)
        await adapter.close()

    def test_map_function_returns_timeout_error(self):
        """_map_openai_error 对 APITimeoutError 返回 LLMTimeoutError。"""
        mapped = _map_openai_error(APITimeoutError(request=MagicMock()))
        assert isinstance(mapped, LLMTimeoutError)


# ============================================================================
# 8. test_connection_error_mapping — APIConnectionError → LLMConnectionError
# ============================================================================


class TestConnectionErrorMapping:

    def test_connection_error_mapped(self):
        """openai.APIConnectionError → LLMConnectionError。"""
        mapped = _map_openai_error(
            APIConnectionError(request=MagicMock(), message="Connection refused")
        )
        assert isinstance(mapped, LLMConnectionError)
        assert "连接失败" in str(mapped)
        assert mapped.host == "api.deepseek.com"

    @pytest.mark.asyncio
    async def test_connection_error_during_chat(self, mocker):
        """chat() 期间发生连接错误映射为 LLMConnectionError。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)
        create_mock.side_effect = APIConnectionError(
            request=MagicMock(), message="Connection refused"
        )

        with pytest.raises(LLMConnectionError) as exc_info:
            await adapter.chat([Message(role="user", content="hi")])

        assert "连接失败" in str(exc_info.value)
        assert exc_info.value.host == "api.deepseek.com"
        await adapter.close()

    def test_api_error_5xx_maps_to_connection_error(self):
        """HTTP 5xx 的通用 APIError 映射为连接错误（可重试）。"""
        err = APIError(message="Server error", request=MagicMock(), body=None)
        err.code = 503

        mapped = _map_openai_error(err)
        assert isinstance(mapped, LLMConnectionError)


# ============================================================================
# 9. test_unknown_error_mapping — 未知异常 → LLMError
# ============================================================================


class TestUnknownErrorMapping:

    def test_unknown_exception_mapped_to_llm_error(self):
        """非 openai 异常统一包装为 LLMError。"""
        mapped = _map_openai_error(ValueError("unexpected"))
        assert isinstance(mapped, LLMError)
        assert "未知错误" in str(mapped)

    @pytest.mark.asyncio
    async def test_generic_exception_during_chat(self, mocker):
        """chat 过程中的通用异常通过 _map_openai_error 转换。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)
        create_mock.side_effect = RuntimeError("boom")

        with pytest.raises(LLMError):
            await adapter.chat([Message(role="user", content="hi")])

        await adapter.close()


# ============================================================================
# 10. test_message_formatting — _messages_to_dicts 输出验证
# ============================================================================


class TestMessageFormatting:

    def test_simple_user_message(self):
        result = DeepSeekAdapter._messages_to_dicts([
            Message(role="user", content="hello")
        ])
        assert result == [{"role": "user", "content": "hello"}]

    def test_system_message(self):
        result = DeepSeekAdapter._messages_to_dicts([
            Message(role="system", content="You are helpful.")
        ])
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_assistant_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="c1", name="search", arguments={"query": "bug"})],
        )
        result = DeepSeekAdapter._messages_to_dicts([msg])
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"][0]["id"] == "c1"
        assert result[0]["tool_calls"][0]["type"] == "function"
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"
        assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"query": "bug"}'

    def test_tool_message_with_tool_call_id(self):
        msg = Message(role="tool", content="result data", tool_call_id="call_001")
        result = DeepSeekAdapter._messages_to_dicts([msg])
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "result data"
        assert result[0]["tool_call_id"] == "call_001"

    def test_multiple_messages(self):
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
        ]
        result = DeepSeekAdapter._messages_to_dicts(messages)
        assert len(result) == 3
        assert [m["role"] for m in result] == ["system", "user", "assistant"]


# ============================================================================
# 11. test_tool_formatting — _format_tools / tools 在请求中
# ============================================================================


class TestToolFormatting:

    @pytest.mark.asyncio
    async def test_tools_passed_in_request(self, mocker):
        """tools 参数被正确传递到 API 请求。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)
        create_mock.return_value = _mock_chat_completion(content="ok")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        await adapter.chat([Message(role="user", content="read")], tools=tools)

        call_kwargs = create_mock.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools

        await adapter.close()

    @pytest.mark.asyncio
    async def test_no_tools_when_none(self, mocker):
        """tools=None 时请求参数中不含 tools 键。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)
        create_mock.return_value = _mock_chat_completion(content="ok")

        await adapter.chat([Message(role="user", content="hi")], tools=None)

        call_kwargs = create_mock.call_args.kwargs
        assert "tools" not in call_kwargs

        await adapter.close()


# ============================================================================
# 附加：count_tokens / get_model_info / _normalize_finish_reason
# ============================================================================


class TestCountTokens:

    def test_count_tokens_empty(self, mocker):
        adapter, _, _ = _make_adapter(mocker)
        assert adapter.count_tokens("") == 0

    def test_count_tokens_english(self, mocker):
        adapter, _, _ = _make_adapter(mocker)
        tokens = adapter.count_tokens("hello world")
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_tokens_chinese(self, mocker):
        adapter, _, _ = _make_adapter(mocker)
        tokens = adapter.count_tokens("你好世界")
        assert tokens > 0
        assert isinstance(tokens, int)


class TestGetModelInfo:

    def test_model_info_fields(self, mocker):
        adapter, _, _ = _make_adapter(mocker, model="deepseek-v4-flash")
        info = adapter.get_model_info()
        assert info["provider"] == "deepseek"
        assert info["model"] == "deepseek-v4-flash"
        assert info["max_tokens"] == 1_000_000
        assert info["supports_streaming"] is True
        assert info["supports_tools"] is True


class TestNormalizeFinishReason:

    def test_known_reasons(self):
        assert _normalize_finish_reason("stop") == "stop"
        assert _normalize_finish_reason("tool_calls") == "tool_calls"
        assert _normalize_finish_reason("length") == "length"
        assert _normalize_finish_reason("content_filter") == "content_filter"

    def test_none_defaults_to_stop(self):
        assert _normalize_finish_reason(None) == "stop"

    def test_unknown_defaults_to_stop(self):
        assert _normalize_finish_reason("unknown_reason") == "stop"

    def test_function_call_maps_to_tool_calls(self):
        assert _normalize_finish_reason("function_call") == "tool_calls"


# ============================================================================
# 附加：_extract_retry_after
# ============================================================================


class TestExtractRetryAfter:
    """_extract_retry_after 从 openai RateLimitError 提取 Retry-After 值。"""

    def test_extracts_retry_after_from_headers(self):
        """标准 Retry-After 头被正确提取。"""
        response = MagicMock()
        response.headers = {"Retry-After": "30.5"}
        err = OpenAIRateLimitError("429", response=response, body=None)
        assert _extract_retry_after(err) == 30.5

    def test_lowercase_header_key(self):
        """小写 retry-after 头也被识别。"""
        response = MagicMock()
        response.headers = {"retry-after": "15"}
        err = OpenAIRateLimitError("429", response=response, body=None)
        assert _extract_retry_after(err) == 15.0

    def test_returns_none_when_no_response(self):
        """无 response 属性返回 None（防御：SDK 版本差异或 mock 异常）。"""
        # 直接构造一个没有 response 属性的假异常来测试防御分支
        fake_err = MagicMock(spec=[])
        del fake_err.response  # 确保 getattr(fake_err, "response", None) → None
        assert _extract_retry_after(fake_err) is None

    def test_returns_none_when_no_headers(self):
        """response 无 headers 属性时返回 None。"""
        fake_err = MagicMock()
        fake_err.response = MagicMock(spec=[])
        assert _extract_retry_after(fake_err) is None

    def test_returns_none_when_header_missing(self):
        """response 有 headers 但无 Retry-After 键时返回 None。"""
        response = MagicMock()
        response.headers = {"Content-Type": "application/json"}
        err = OpenAIRateLimitError("429", response=response, body=None)
        assert _extract_retry_after(err) is None

    def test_invalid_float_value_returns_none(self):
        """Retry-After 值无法转为 float 时返回 None。"""
        response = MagicMock()
        response.headers = {"Retry-After": "not-a-number"}
        err = OpenAIRateLimitError("429", response=response, body=None)
        assert _extract_retry_after(err) is None


# ============================================================================
# 附加：响应解析边界
# ============================================================================


class TestResponseParsingEdgeCases:

    @pytest.mark.asyncio
    async def test_empty_choices_raises_parse_error(self, mocker):
        """API 返回 choices=[] 时抛出 LLMResponseParseError。"""
        from miaowa.core.exceptions import LLMResponseParseError

        adapter, mock_client, create_mock = _make_adapter(mocker)
        bad_completion = MagicMock()
        bad_completion.choices = []
        create_mock.return_value = bad_completion

        with pytest.raises(LLMResponseParseError, match="不包含 choices"):
            await adapter.chat([Message(role="user", content="hi")])

        await adapter.close()

    @pytest.mark.asyncio
    async def test_stream_chunk_no_choices_returns_empty(self, mocker):
        """流式 chunk 无 choices 时返回空 StreamChunk（不崩溃）。"""
        adapter, mock_client, create_mock = _make_adapter(mocker)

        empty_chunk = MagicMock()
        empty_chunk.choices = []

        async def _mock_stream():
            yield empty_chunk

        create_mock.return_value = _mock_stream()

        chunks = [c async for c in adapter.stream([Message(role="user", content="hi")])]

        assert len(chunks) == 1
        assert chunks[0].delta_content is None
        assert chunks[0].finish_reason is None

        await adapter.close()
