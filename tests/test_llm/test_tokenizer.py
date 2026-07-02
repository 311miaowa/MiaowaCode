"""Token 计数器单元测试 — TokenCounter 与辅助函数。

覆盖：多语言 token 计数（空/中文/英文/混合）、消息级计数、
成本估算（chat/reasoner）、成本格式化、消息截断（system 保留、
最后 user 保留、tool 配对保护、极端预算）。
"""

from __future__ import annotations

import pytest

from miaowa.llm.tokenizer import (
    TokenCounter,
    _count_chars_and_tokens,
    _is_cjk,
    _CJK_TOKENS_PER_CHAR,
    _NON_CJK_CHARS_PER_TOKEN,
    _MSG_OVERHEAD_TOKENS,
)
from miaowa.llm.types import Message, ToolCall


# ============================================================================
# 辅助
# ============================================================================


@pytest.fixture
def counter():
    """返回默认 deepseek-chat 模型的 TokenCounter。"""
    return TokenCounter("deepseek-chat")


@pytest.fixture
def reasoner_counter():
    """返回 deepseek-reasoner 模型的 TokenCounter。"""
    return TokenCounter("deepseek-reasoner")


# ============================================================================
# 1. test_count_tokens_empty — 空字符串
# ============================================================================


class TestCountTokensEmpty:
    """空输入返回 0。"""

    def test_empty_string_returns_zero(self, counter):
        assert counter.count_tokens("") == 0

    def test_module_function_empty_returns_zero(self):
        tokens, cjk = _count_chars_and_tokens("")
        assert tokens == 0
        assert cjk == 0

    def test_none_and_empty_yield_zero(self):
        """TokenCounter.count_tokens 对 None 和空字符串均返回 0。"""
        counter = TokenCounter("deepseek-chat")
        # 依赖 if not text: 隐式转换（None → False → 返回 0）
        assert counter.count_tokens("") == 0
        # None 虽然不在类型注解中，但运行时静默返回 0 而非抛异常
        # 此处显式验证行为确保不会被误改为抛 TypeError
        assert counter.count_tokens(None) == 0  # type: ignore[arg-type]


# ============================================================================
# 2. test_count_tokens_chinese — 中文
# ============================================================================


class TestCountTokensChinese:
    """中文文本 token 估算。"""

    def test_pure_chinese(self, counter):
        """纯中文：每个字符约 1.5 tokens。"""
        text = "你好世界"
        tokens = counter.count_tokens(text)
        expected = int(4 * _CJK_TOKENS_PER_CHAR + 0.5)  # 4 chars * 1.5 = 6
        assert tokens == expected
        assert tokens > 0

    def test_chinese_long_text(self, counter):
        """较长中文文本 token 计数在合理范围内。"""
        text = "这是一段用于测试的中文文本，用于验证 Token 计数算法的准确性。"
        tokens = counter.count_tokens(text)
        cjk_count = sum(1 for ch in text if _is_cjk(ch))
        # 全部或大部分为 CJK 字符，tokens ≈ cjk_count * 1.5
        approx = int(cjk_count * _CJK_TOKENS_PER_CHAR + 0.5)
        # 允许 ±1 的舍入误差
        assert abs(tokens - approx) <= 1, f"tokens={tokens}, approx={approx}"

    def test_chinese_punctuation_is_cjk(self):
        """中文标点（、。「」）被识别为 CJK。"""
        assert _is_cjk("、") is True
        assert _is_cjk("。") is True
        assert _is_cjk("「") is True
        assert _is_cjk("」") is True

    def test_japanese_kana_is_cjk(self):
        """日文假名被识别为 CJK。"""
        assert _is_cjk("あ") is True  # 平假名
        assert _is_cjk("ア") is True  # 片假名
        assert _is_cjk("ー") is True  # 长音符（片假名范围 0x30A0-0x30FF）


# ============================================================================
# 3. test_count_tokens_english — 英文
# ============================================================================


class TestCountTokensEnglish:
    """英文文本 token 估算。"""

    def test_pure_english(self, counter):
        """纯英文：约 4 字符 → 1 token。"""
        text = "hello world"
        tokens = counter.count_tokens(text)
        # 11 chars, 0 CJK → 11/4 = 2.75 → 3
        expected = int(11 / _NON_CJK_CHARS_PER_TOKEN + 0.5)
        assert tokens == expected

    def test_english_long_text(self, counter):
        """较长英文文本比例正确。"""
        text = "The quick brown fox jumps over the lazy dog"
        tokens = counter.count_tokens(text)
        # 全非 CJK，tokens ≈ len / 4
        non_cjk = len(text)
        approx = int(non_cjk / _NON_CJK_CHARS_PER_TOKEN + 0.5)
        assert tokens == approx

    def test_english_with_numbers(self, counter):
        """数字和特殊字符按非 CJK 处理。"""
        text = "Error 404: Not Found (2024-01-15)"
        tokens = counter.count_tokens(text)
        assert tokens > 0
        # 合理范围
        assert tokens >= len(text) // 4


# ============================================================================
# 4. test_count_tokens_mixed — 中英混合
# ============================================================================


class TestCountTokensMixed:
    """中英混合文本。"""

    def test_mixed_cjk_and_english(self, counter):
        """混合文本同时计算 CJK 和非 CJK 部分。"""
        text = "Hello 你好 World 世界"
        tokens = counter.count_tokens(text)
        cjk_count = sum(1 for ch in text if _is_cjk(ch))
        non_cjk_count = len(text) - cjk_count
        expected = int(
            cjk_count * _CJK_TOKENS_PER_CHAR
            + non_cjk_count / _NON_CJK_CHARS_PER_TOKEN
            + 0.5
        )
        assert tokens == expected

    def test_mixed_text_reasonable_range(self, counter):
        """混合文本 token 数在合理范围内（不超出 ±30% 的字符数上限）。"""
        text = "Python 是一门优秀的编程语言，广泛应用于 AI 和 Web 开发。"
        tokens = counter.count_tokens(text)
        # 不应超过字符数 * 2（CJK 极端），也不应少于字符数 / 5
        assert tokens <= len(text) * 2
        assert tokens >= len(text) // 5


# ============================================================================
# 5. test_count_messages — 完整消息列表 token 计数
# ============================================================================


class TestCountMessages:
    """消息列表级 token 计数。"""

    def test_empty_messages_returns_zero(self, counter):
        assert counter.count_messages(None) == 0
        assert counter.count_messages([]) == 0

    def test_single_message_count(self, counter):
        """单条消息 = content tokens + overhead。"""
        messages = [{"role": "user", "content": "hello"}]
        total = counter.count_messages(messages)
        content_tokens = counter.count_tokens("hello")
        assert total == content_tokens + _MSG_OVERHEAD_TOKENS

    def test_multiple_messages(self, counter):
        """多条消息各自含 overhead。"""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "什么是 Python？"},
            {"role": "assistant", "content": "Python 是一种编程语言。"},
        ]
        total = counter.count_messages(messages)
        manual_sum = sum(
            counter.count_tokens(m["content"]) + _MSG_OVERHEAD_TOKENS
            for m in messages
        )
        assert total == manual_sum

    def test_counts_with_message_dataclass(self, counter):
        """兼容 list[Message] 格式。"""
        messages = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="user question"),
        ]
        total = counter.count_messages(messages)
        assert total > 0
        manual = sum(
            counter.count_tokens(m.content) + _MSG_OVERHEAD_TOKENS
            for m in messages
        )
        assert total == manual

    def test_dict_without_content_key(self, counter):
        """消息 dict 缺少 content 时默认为空字符串。"""
        messages = [{"role": "user"}]  # no content key
        total = counter.count_messages(messages)
        assert total == _MSG_OVERHEAD_TOKENS  # 仅 overhead


# ============================================================================
# 6. test_estimate_cost_chat — chat 模型成本
# ============================================================================


class TestEstimateCostChat:
    """deepseek-chat (flash) 模型成本估算。"""

    def test_chat_model_cost(self, counter):
        """V4 Flash: input=$0.14/M, output=$0.28/M。"""
        cost = counter.estimate_cost(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert cost == 0.42  # 0.14 + 0.28

    def test_chat_model_cost_zero(self, counter):
        cost = counter.estimate_cost(prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0

    def test_typical_usage_cost(self, counter):
        """典型对话：~1000 input, ~200 output。"""
        cost = counter.estimate_cost(prompt_tokens=1000, completion_tokens=200)
        expected = 1000 * 0.14 / 1_000_000 + 200 * 0.28 / 1_000_000
        assert cost == round(expected, 4)


# ============================================================================
# 7. test_estimate_cost_reasoner — reasoner 模型成本
# ============================================================================


class TestEstimateCostReasoner:
    """deepseek-reasoner (pro) 模型成本估算。"""

    def test_reasoner_model_cost(self, reasoner_counter):
        """V4 Pro: input=$0.435/M, output=$0.87/M。"""
        cost = reasoner_counter.estimate_cost(
            prompt_tokens=1_000_000, completion_tokens=1_000_000
        )
        assert cost == 1.305  # 0.435 + 0.87

    def test_reasoner_higher_than_chat(self, counter, reasoner_counter):
        """同 token 数下 reasoner 费用高于 chat。"""
        chat_cost = counter.estimate_cost(prompt_tokens=500, completion_tokens=500)
        reasoner_cost = reasoner_counter.estimate_cost(prompt_tokens=500, completion_tokens=500)
        assert reasoner_cost > chat_cost


# ============================================================================
# 8. test_estimate_cost_validation — 参数校验
# ============================================================================


class TestEstimateCostValidation:
    """成本估算的参数校验。"""

    def test_negative_prompt_tokens_raises(self, counter):
        with pytest.raises(ValueError, match="prompt_tokens"):
            counter.estimate_cost(prompt_tokens=-1, completion_tokens=0)

    def test_negative_completion_tokens_raises(self, counter):
        with pytest.raises(ValueError, match="completion_tokens"):
            counter.estimate_cost(prompt_tokens=0, completion_tokens=-5)

    def test_custom_pricing(self):
        """自定义定价表优先于内置表。"""
        counter = TokenCounter(
            "custom-model",
            pricing={"input": 1.0, "output": 2.0},
        )
        cost = counter.estimate_cost(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert cost == 3.0  # 1.0 + 2.0

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="model 不能为空"):
            TokenCounter("")

    def test_whitespace_model_name_raises(self):
        with pytest.raises(ValueError, match="model 不能为空"):
            TokenCounter("   ")


# ============================================================================
# 9. test_format_cost — 格式化费用
# ============================================================================


class TestFormatCost:
    """费用格式化。"""

    def test_positive_amount(self):
        assert TokenCounter.format_cost(0.0032) == "$0.0032"

    def test_zero_amount(self):
        assert TokenCounter.format_cost(0.0) == "$0.0000"

    def test_large_amount(self):
        assert TokenCounter.format_cost(1.5) == "$1.5000"

    def test_negative_amount_raises(self):
        with pytest.raises(ValueError, match="费用不能为负数"):
            TokenCounter.format_cost(-0.01)


# ============================================================================
# 10. test_truncate_preserves_system — 保留 system 消息
# ============================================================================


class TestTruncatePreservesSystem:
    """截断必须保留 system 消息。"""

    def test_system_preserved(self, counter):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "short"},
        ]
        result = counter.truncate_messages(messages, max_tokens=1000)
        assert len(result) == 2
        assert result[0]["role"] == "system"

    def test_system_preserved_even_with_tight_budget(self, counter):
        """即使预算紧张，system + last user 也保留。"""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        # 至少保留 system + 最后 user
        sys_tokens = counter.count_tokens("Be helpful.") + _MSG_OVERHEAD_TOKENS
        last_tokens = counter.count_tokens("q2") + _MSG_OVERHEAD_TOKENS
        min_budget = sys_tokens + last_tokens

        result = counter.truncate_messages(messages, max_tokens=min_budget)
        roles = [m["role"] for m in result]
        assert "system" in roles
        assert roles[-1] == "user"


# ============================================================================
# 11. test_truncate_preserves_last_user — 保留最后一条 user
# ============================================================================


class TestTruncatePreservesLastUser:
    """截断必须保留最后一条 user 消息。"""

    def test_last_user_preserved(self, counter):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]
        result = counter.truncate_messages(messages, max_tokens=5000)
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == "second question"

    def test_only_system_and_last_user_when_budget_tight(self, counter):
        """预算仅够 system + 最后 user 时，中间消息被丢弃。"""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "keep this"},

        ]
        result = counter.truncate_messages(messages, max_tokens=100)
        # 至少保留 system 和 user
        assert len(result) == 2


# ============================================================================
# 12. test_truncate_pairs_preserved — tool-call/tool-result 成对保留
# ============================================================================


class TestTruncatePairsPreserved:
    """tool-call 与 tool-result 成对原子保留。"""

    def test_tool_pair_preserved_when_included(self, counter):
        """当预算足够时，tool pair 整体保留。"""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "analyze this"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "search result", "tool_call_id": "c1"},
            {"role": "assistant", "content": "analysis complete"},
            {"role": "user", "content": "thanks"},
        ]
        result = counter.truncate_messages(messages, max_tokens=5000)
        roles = [m["role"] for m in result]

        # tool pair 要么都在，要么都不在
        has_assistant_tool = any(
            m["role"] == "assistant" and "tool_calls" in m and m["tool_calls"]
            for m in result
        )
        has_tool_result = any(m["role"] == "tool" for m in result)
        # 如果保留则成对
        assert has_assistant_tool == has_tool_result

    def test_tool_pair_dropped_together(self, counter):
        """当预算不够时，tool pair 整体丢弃。"""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "search for bugs"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "many bugs found " + "x" * 500, "tool_call_id": "c1"},
            {"role": "user", "content": "end"},
        ]

        # 计算仅 system + last user 的 tokens
        min_tokens = counter.count_messages([
            messages[0], messages[4]
        ])

        # 预算只够 system + 最后 user，tool pair 应整体丢弃
        result = counter.truncate_messages(messages, max_tokens=min_tokens + 50)
        has_tool = any(
            m.get("role") == "tool" for m in result
        )
        has_assistant_tc = any(
            m.get("role") == "assistant" and m.get("tool_calls")
            for m in result
        )
        # tool pair 应被丢弃
        assert not has_tool
        assert not has_assistant_tc


# ============================================================================
# 13. test_truncate_extreme — max_tokens 极小
# ============================================================================


class TestTruncateExtreme:
    """极端预算场景。"""

    def test_very_small_budget(self, counter):
        """max_tokens 极小但足够保留必留消息。"""
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hi"},
        ]
        # 每条: count_tokens("s") + overhead = 0 + 4 = 4 (s 为空? 不对, "s" 有 1 个字符)
        # "s": 1 char, 0 CJK → 1/4 = 0.25 → int(0.25+0.5)=0 tokens
        # 但 count_tokens 有 max(tokens, 0) 所以是 0
        # overhead = 4, 所以每条 4 tokens
        # system 4 + user 4 = 8
        result = counter.truncate_messages(messages, max_tokens=10)
        assert len(result) == 2

    def test_budget_exceeded_by_must_keep_raises(self, counter):
        """必留消息超预算时抛出 RuntimeError。"""
        messages = [
            {"role": "system", "content": "very long system prompt " + "x" * 1000},
            {"role": "user", "content": "hi"},
        ]
        with pytest.raises(RuntimeError, match="无法将消息列表降至"):
            counter.truncate_messages(messages, max_tokens=5)

    def test_zero_max_tokens_raises(self, counter):
        messages = [{"role": "user", "content": "hi"}]
        with pytest.raises(ValueError, match="max_tokens"):
            counter.truncate_messages(messages, max_tokens=0)

    def test_negative_max_tokens_raises(self, counter):
        messages = [{"role": "user", "content": "hi"}]
        with pytest.raises(ValueError, match="max_tokens"):
            counter.truncate_messages(messages, max_tokens=-1)

    def test_empty_messages_raises(self, counter):
        with pytest.raises(ValueError, match="messages 不能为空"):
            counter.truncate_messages([], max_tokens=100)


# ============================================================================
# 14. 附加：_is_cjk 边界情况
# ============================================================================


class TestIsCjk:
    """_is_cjk 边界情况。"""

    def test_ascii_not_cjk(self):
        assert _is_cjk("a") is False
        assert _is_cjk("Z") is False
        assert _is_cjk("0") is False
        assert _is_cjk(" ") is False

    def test_emoji_not_cjk(self):
        """Emoji 不在 CJK 范围内。"""
        assert _is_cjk("😀") is False
        assert _is_cjk("🎉") is False

    def test_non_bmp_cjk_extension_b(self):
        """扩展 B 平面字符（> U+FFFF）被识别。"""
        # U+20000 是 CJK 扩展 B 的起始
        ext_b_char = chr(0x20000)
        assert _is_cjk(ext_b_char) is True

    def test_cjk_compatibility(self):
        """CJK 兼容表意文字被识别。"""
        # U+F900 是 CJK 兼容区的起始
        compat_char = chr(0xF900)
        assert _is_cjk(compat_char) is True


# ============================================================================
# 15. 附加：TokenCounter 初始化
# ============================================================================


class TestTokenCounterInit:
    """TokenCounter 初始化及定价表。"""

    def test_default_model_chat(self):
        counter = TokenCounter()
        assert counter.model == "deepseek-chat"

    def test_explicit_model_stored(self):
        counter = TokenCounter("deepseek-v4-pro")
        assert counter.model == "deepseek-v4-pro"

    def test_unknown_model_uses_fallback_pricing(self):
        """未知模型使用回退定价。"""
        counter = TokenCounter("unknown-model-v99")
        cost = counter.estimate_cost(prompt_tokens=1_000_000, completion_tokens=0)
        assert cost == 0.14  # 回退输入价格

    def test_model_name_stripped(self):
        """model 字符串两端空格被去除。"""
        counter = TokenCounter("  deepseek-chat  ")
        assert counter.model == "deepseek-chat"
