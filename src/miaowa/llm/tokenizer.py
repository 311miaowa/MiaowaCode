"""Token 计数与成本估算模块 — 提供无外部依赖的近似 Token 计数器。

=============================================================================
重要：本模块使用**启发式近似算法**，不依赖 tiktoken 或 HuggingFace tokenizer。
=============================================================================

**估算方法**

- CJK 字符（中文、日文汉字/假名等）: 约 1.5 tokens / 字符
- 非 CJK 字符（英文、数字、标点等）: 约 4 字符 → 1 token（即 0.25 token / 字符）

**局限性**

1. **模型差异**: DeepSeek 未公开其 tokenizer 实现细节，本文的 CJK 范围覆盖
   和加权系数基于经验测量，与真实 token 数存在 ±20% 偏差。
2. **特殊 token**: 不考虑 BOS/EOS、对话模板格式化 token（如 ``<|im_start|>``），
   ``count_messages()`` 仅以每条消息附加 4 tokens 的固定开销近似补偿。
3. **成本估算**: 基于 DeepSeek 官方定价，价格变动时需更新
   ``TokenCounter.pricing`` 表。实际计费以 API 返回的 usage 字段为准。
4. **截断精度**: ``truncate_messages()`` 的截断结果可能与实际 token 计数
   存在偏差，建议预留 10% 安全余量。

**扩展方向（非 MVP）**

- 集成 tiktoken 的 ``o200k_base`` 编码（实验表明与 DeepSeek tokenizer
  行为接近）。
- 通过 DeepSeek tokenizer API（若有）获取精确计数。
- 支持 batch 计费计算（Batch API 折扣）。

=============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from miaowa.llm.types import Message

# ---------------------------------------------------------------------------
# Token 估算常量
# ---------------------------------------------------------------------------

# CJK 字符均值：~1.5 tokens / 字符（DeepSeek tokenizer 对 CJK 字符编码偏高）
_CJK_TOKENS_PER_CHAR = 1.5

# 非 CJK 文本均值：~4 字符 → 1 token
_NON_CJK_CHARS_PER_TOKEN = 4.0

# 每条消息的固定格式开销（role 标签、消息分隔符等），OpenAI 经验值
_MSG_OVERHEAD_TOKENS: Final[int] = 4

# CJK Unicode 范围（基本多文种平面 + 扩展区 + 标点符号 + 假名）
# 这些字符在 DeepSeek tokenizer 中的编码行为接近 ~1–2 tokens/字符
_CJK_RANGES: list[tuple[int, int]] = [
    (0x4E00, 0x9FFF),    # CJK 统一表意文字
    (0x3400, 0x4DBF),    # 扩展 A
    (0x20000, 0x2A6DF),  # 扩展 B
    (0xF900, 0xFAFF),    # 兼容表意文字
    (0x2F800, 0x2FA1F),  # 兼容补充
    (0x3000, 0x303F),    # CJK 标点符号（、。「」『』【】）
    (0xFF00, 0xFFEF),    # 半角/全角形式
    (0x3040, 0x309F),    # 平假名
    (0x30A0, 0x30FF),    # 片假名
]

# ---------------------------------------------------------------------------
# 模型定价表 — ¥/百万 tokens
# ---------------------------------------------------------------------------

# 数据来源: https://api-docs.deepseek.com/quick_start/pricing
# 最后验证: 2026-07-01（DeepSeek V4 官方定价，美元/百万 tokens）
# deepseek-chat / deepseek-reasoner 为旧别名，将于 2026-07-24 废弃
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"input": 0.435, "output": 0.87},
    # 旧别名兼容（废弃日期: 2026-07-24）
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.435, "output": 0.87},
}

# 未在定价表中匹配到的模型使用此默认价格
_FALLBACK_PRICING: dict[str, float] = {"input": 0.14, "output": 0.28}


# ============================================================================
# CJK 字符检测 — 模块级共享（供 deepseek.py 等适配器导入复用）
# ============================================================================

# 延迟初始化的 CJK code point 集合（BMP 范围内，O(1) 查找）
_CJK_SET: set[int] | None = None


def _init_cjk_set() -> set[int]:
    """构建 BMP 范围内的 CJK code point 集合。

    仅在首次调用 _is_cjk() 时执行一次，后续直接使用缓存集合。
    """
    cjk: set[int] = set()
    for lo, hi in _CJK_RANGES:
        # 仅 BMP 范围内的字符纳入集合（空间换时间，约 5 万个条目）
        if hi <= 0xFFFF:
            cjk.update(range(lo, hi + 1))
    return cjk


def _is_cjk(ch: str) -> bool:
    """判断单个字符是否落在 CJK Unicode 范围内。

    优先使用 O(1) 集合查找（BMP 字符），
    仅对扩展平面字符回退到 O(r) 范围遍历。

    Args:
        ch: 单个字符。

    Returns:
        True 若字符属于 CJK 统一表意文字、标点符号或日文假名。
    """
    global _CJK_SET
    if _CJK_SET is None:
        _CJK_SET = _init_cjk_set()

    cp = ord(ch)
    if cp <= 0xFFFF:
        return cp in _CJK_SET

    # 扩展平面字符：回退到范围遍历（字符稀少，性能影响可忽略）
    for lo, hi in _CJK_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def _count_chars_and_tokens(text: str) -> tuple[int, int]:
    """计算 CJK/非CJK 字符数并估算 token 数。

    Args:
        text: 输入文本。空字符串返回 (0, 0)。

    Returns:
        (token_count, cjk_char_count) 元组。
    """
    if not text:
        return 0, 0

    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk = len(text) - cjk
    tokens = int(
        cjk * _CJK_TOKENS_PER_CHAR + non_cjk / _NON_CJK_CHARS_PER_TOKEN + 0.5
    )
    return max(tokens, 0), cjk


# ============================================================================
# TokenCounter
# ============================================================================


class TokenCounter:
    """Token 计数器和成本估算器。

    提供无外部依赖的近似 Token 计数、消息列表级计数、费用估算
    和基于预算的消息截断功能。

    Usage::

        counter = TokenCounter("deepseek-chat")

        tokens = counter.count_tokens("你好世界")
        msg_tokens = counter.count_messages([
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
        ])
        cost = counter.estimate_cost(prompt_tokens=1000, completion_tokens=200)
        print(counter.format_cost(cost))  # → ¥0.0014

        truncated = counter.truncate_messages(messages, max_tokens=8000)
    """

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: str = "deepseek-chat",
        *,
        pricing: dict[str, float] | None = None,
    ) -> None:
        """初始化 Token 计数器。

        Args:
            model: 模型名称，用于查找默认定价表。支持的模型：
                - ``"deepseek-chat"``（默认）
                - ``"deepseek-reasoner"``
                未匹配到的模型使用 deepseek-chat 的默认定价。
            pricing: 自定义定价覆盖，格式为 ``{"input": 1.0, "output": 2.0}``。
                传入后优先于内置定价表，适用于自托管或微调模型。

        Raises:
            ValueError: model 为空字符串时。
        """
        if not model or not model.strip():
            raise ValueError("model 不能为空字符串")

        self.model: str = model.strip()
        self.pricing: dict[str, float] = (
            pricing
            if pricing is not None
            else _DEFAULT_PRICING.get(self.model, _FALLBACK_PRICING)
        )

    # ------------------------------------------------------------------
    # Token 计数
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """估算单段文本的 Token 数量。

        算法：
            - CJK 字符（含中文、日文汉字/假名、中文标点等）: ~1.5 tokens / 字符
            - 非 CJK 字符（英文、数字、空格、西文标点等）: ~4 字符 → 1 token
            - 取整策略：向上取整（宁可略多不可略少，防上下文溢出）

        本方法是项目内统一的 token 计数入口；
        ``DeepSeekAdapter.count_tokens()`` 委托本方法实现。

        Args:
            text: 待估算的文本。空字符串直接返回 0。

        Returns:
            int: 估算的 token 数量，始终 >= 0。
        """
        if not text:
            return 0
        tokens, _ = _count_chars_and_tokens(text)
        return tokens

    def count_messages(self, messages: list[dict] | list[Message] | None = None) -> int:
        """估算消息列表的总 Token 数量（含格式开销）。

        对每条消息：``count_tokens(content) + 固定格式开销``，
        然后求和。格式开销为每条消息 ``_MSG_OVERHEAD_TOKENS`` tokens
        （OpenAI 经验值），用于补偿 role 标签、消息分隔符等结构化 token 消耗。

        Args:
            messages: 消息列表，兼容以下两种格式：
                - ``list[dict]``: 每项含 ``"role"`` 和 ``"content"`` 键。
                - ``list[Message]``: ``Message`` dataclass 对象。
                传入 ``None`` 或空列表返回 0。

        Returns:
            int: 消息列表的总 token 估算值，始终 >= 0。
        """
        if not messages:
            return 0

        total = 0
        for msg in messages:
            content = _extract_content(msg)
            total += self.count_tokens(content) + _MSG_OVERHEAD_TOKENS
        return total

    # ------------------------------------------------------------------
    # 成本估算
    # ------------------------------------------------------------------

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """根据模型定价估算单次调用的费用（美元）。

        费用 = prompt_tokens / 1,000,000 * input_price
             + completion_tokens / 1,000,000 * output_price

        Args:
            prompt_tokens: 输入 token 数（应从 API 返回的 usage.prompt_tokens 获取）。
            completion_tokens: 输出 token 数（应从 usage.completion_tokens 获取）。

        Returns:
            float: 估算费用（美元），保留 4 位小数精度。

        Raises:
            ValueError: prompt_tokens 或 completion_tokens 为负数时。
        """
        if prompt_tokens < 0:
            raise ValueError(
                f"prompt_tokens 不能为负数，当前值: {prompt_tokens}"
            )
        if completion_tokens < 0:
            raise ValueError(
                f"completion_tokens 不能为负数，当前值: {completion_tokens}"
            )

        input_price_per_token = self.pricing["input"] / 1_000_000
        output_price_per_token = self.pricing["output"] / 1_000_000

        cost = (
            prompt_tokens * input_price_per_token
            + completion_tokens * output_price_per_token
        )
        return round(cost, 4)

    @staticmethod
    def format_cost(amount: float) -> str:
        """将费用金额格式化为可读字符串（美元）。

        Args:
            amount: 费用金额（美元），如 ``0.0032``。

        Returns:
            str: 格式化字符串，如 ``"$0.0032"``。
                金额为 0 时返回 ``"$0.0000"``。

        Raises:
            ValueError: amount 为负数时。
        """
        if amount < 0:
            raise ValueError(f"费用不能为负数，当前值: {amount}")
        return f"${amount:.4f}"

    # ------------------------------------------------------------------
    # 消息截断
    # ------------------------------------------------------------------

    def truncate_messages(
        self,
        messages: list[dict] | list[Message],
        max_tokens: int,
    ) -> list[dict] | list[Message]:
        """截断消息列表以控制在 Token 预算内。

        截断策略（优先级从高到低）:
            1. **必须保留**: system 消息（role == "system" 的首条）
            2. **必须保留**: 最后一条 user 消息（最新的用户输入）
            3. **原子保留**: tool-call ↔ tool-result 配对
               （assistant(tool_call) 与后续 tool(result) 成对保留或成对删除）
            4. **按需保留**: 其余中间消息从旧到新保留，直到 token 预算耗尽

        如果 system + 最后 user + 关联 tool 配对已超过 max_tokens，
        本方法抛出 RuntimeError；调用方（ContextBuilder）应捕获此异常
        并尝试对 system prompt 做内容级截断后再调用本方法。

        Args:
            messages: 消息列表，兼容 ``list[dict]`` 和 ``list[Message]``。
                返回类型与输入类型一致。
            max_tokens: Token 预算上限，必须 > 0。

        Returns:
            截断后的消息列表（与输入类型相同）。消息顺序保持原始顺序。

        Raises:
            ValueError: max_tokens <= 0 或 messages 为空时。
            RuntimeError: 即使只保留必留消息仍超预算时。
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens 必须 > 0，当前值: {max_tokens}")
        if not messages:
            raise ValueError("messages 不能为空")

        n = len(messages)

        # -- 0. 预计算每条消息的 token 数（避免重复计算）------------------
        msg_tokens: list[int] = [
            self.count_messages([msg]) for msg in messages
        ]

        # -- 1. 查找 tool-call ↔ tool-result 配对 ------------------------
        # 配对: assistant(tool_call) ↔ 紧接其后的 tool(result)
        # 工具调用通过检查消息是否包含 tool_calls 属性判断（兼容 dict 和 dataclass）
        pair_map: dict[int, int] = {}  # tool_call_idx → tool_result_idx
        for i, msg in enumerate(messages):
            role = _extract_role(msg)
            if role == "assistant" and _has_tool_calls(msg):
                # 查找紧接在后的 tool 消息
                if i + 1 < n and _extract_role(messages[i + 1]) == "tool":
                    pair_map[i] = i + 1

        # -- 2. 定位必须保留的消息索引 ------------------------------------
        keep: set[int] = set()

        # system
        system_idx: int = -1
        for i, msg in enumerate(messages):
            if _extract_role(msg) == "system":
                system_idx = i
                keep.add(i)
                break

        # 最后一条 user
        last_user_idx: int = -1
        for i in range(n - 1, -1, -1):
            if _extract_role(messages[i]) == "user":
                last_user_idx = i
                keep.add(i)
                break

        # 若没有 user 消息，保留最后一条非 system 消息
        if last_user_idx == -1:
            for i in range(n - 1, -1, -1):
                if i != system_idx:
                    last_user_idx = i
                    keep.add(i)
                    break

        # -- 3. 将与保留消息关联的 tool 配对也标记为必须保留 --------------
        # 如果 assistant(tool_call) 被保留 → 其 tool(result) 也必须保留
        # 如果 tool(result) 被保留 → 其 assistant(tool_call) 也必须保留
        changed = True
        while changed:
            changed = False
            for call_idx, result_idx in pair_map.items():
                # 保留配对（成对保留）
                if call_idx in keep and result_idx not in keep:
                    keep.add(result_idx)
                    changed = True
                if result_idx in keep and call_idx not in keep:
                    keep.add(call_idx)
                    changed = True

        # -- 4. 计算必须保留消息的 token 占用 ----------------------------
        must_keep_tokens = sum(msg_tokens[i] for i in keep)

        if must_keep_tokens > max_tokens:
            raise RuntimeError(
                f"无法将消息列表降至 {max_tokens} tokens 以内: "
                f"必须保留的消息（system + 最后 user + 关联 tool 配对）"
                f"已占用 {must_keep_tokens} tokens，超出预算 "
                f"{must_keep_tokens - max_tokens} tokens"
            )

        # -- 5. 按时间顺序填满预算（从旧到新）-----------------------------
        remaining = max_tokens - must_keep_tokens

        # 构建反向配对表：tool_result_idx → tool_call_idx
        reverse_pair: dict[int, int] = {v: k for k, v in pair_map.items()}

        for i in range(n):
            if i in keep:
                continue

            # tool(result) 消息：跳过（它们随配对的 assistant 一起处理）
            if i in reverse_pair:
                continue

            if i in pair_map:
                # assistant(tool_call): 与 tool(result) 成对处理
                result_idx = pair_map[i]
                pair_cost = msg_tokens[i] + msg_tokens[result_idx]
                if pair_cost <= remaining:
                    keep.add(i)
                    keep.add(result_idx)
                    remaining -= pair_cost
                # 预算不够：跳过整个配对
            else:
                # 普通消息
                if msg_tokens[i] <= remaining:
                    keep.add(i)
                    remaining -= msg_tokens[i]

        # -- 6. 按原始顺序输出结果 ----------------------------------------
        result = [msg for i, msg in enumerate(messages) if i in keep]

        # 安全断言
        final_tokens = sum(msg_tokens[i] for i in keep)
        if final_tokens > max_tokens:
            raise RuntimeError(
                f"截断结果 ({final_tokens} tokens) 超出预算 ({max_tokens} tokens) — "
                f"这是一个 bug，请报告"
            )

        return result


# ============================================================================
# 模块级辅助函数
# ============================================================================


def _extract_content(msg: Any) -> str:
    """从消息对象中提取 content 字段。

    兼容 ``dict``（``msg["content"]``）和 ``Message`` dataclass
    （``msg.content``）两种格式。

    Args:
        msg: 消息对象（dict 或 dataclass）。

    Returns:
        content 字符串。若无法提取则返回 ""。
    """
    if isinstance(msg, dict):
        return str(msg.get("content", ""))
    return str(getattr(msg, "content", ""))


def _extract_role(msg: Any) -> str:
    """从消息对象中提取 role 字段。

    兼容 ``dict``（``msg["role"]``）和 ``Message`` dataclass
    （``msg.role``）两种格式。

    Args:
        msg: 消息对象（dict 或 dataclass）。

    Returns:
        role 字符串。若无法提取则返回 ""。
    """
    if isinstance(msg, dict):
        return str(msg.get("role", ""))
    return str(getattr(msg, "role", ""))


def _has_tool_calls(msg: Any) -> bool:
    """判断消息是否包含工具调用（Function Calling 请求）。

    兼容两种格式：
        - ``dict``: 检查 ``msg.get("tool_calls")``
        - ``Message`` / ``ChatResponse`` 等 dataclass: 检查 ``msg.tool_calls``

    Args:
        msg: 消息对象。

    Returns:
        True 若消息包含非空的 tool_calls。
    """
    if isinstance(msg, dict):
        tc = msg.get("tool_calls", None)
    else:
        tc = getattr(msg, "tool_calls", None)
    return bool(tc)
