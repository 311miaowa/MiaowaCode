"""对话记忆管理器 — 会话状态维护与对话历史管理。

PRD §5.2.5: MemoryManager 负责维护当前会话的对话历史，
SessionManager 管理多会话生命周期。

MVP 范围::

    - MemoryManager: 纯内存存储，单会话历史（≤50 条消息）
    - SessionManager: 持有当前 MemoryManager，支持新建/切换会话

V2 演进方向（预留）::
    - 持久化到 SQLite / JSON 文件
    - 多会话并行管理
    - 会话摘要与自动压缩
    - 跨会话上下文共享

Typical usage::

    from miaowa.agent.session import MemoryManager, SessionManager

    # 方式 1: 直接使用 MemoryManager（轻量场景）
    mem = MemoryManager()
    mem.add("user", "什么是闭包？")
    mem.add("assistant", "闭包是指...")
    history = mem.get_history(last_n=10)

    # 方式 2: 通过 SessionManager（CLI 场景）
    sessions = SessionManager()
    sessions.current.add("user", "你好")
    sessions.new_session()  # 开始新会话
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from miaowa.core.logger import get_logger
from miaowa.llm.types import Message

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 模块常量
# ---------------------------------------------------------------------------

_DEFAULT_MAX_HISTORY = 50
"""单会话最大消息条数。超出时裁剪为最近的一半。"""

_TRIM_KEEP_COUNT = 25
"""裁剪时保留的最近消息条数（= MAX_HISTORY / 2）。"""

_SESSION_ID_BYTES = 8
"""会话 ID 的随机字节数（生成 16 位 hex 字符串）。"""

_MAX_ARCHIVED_SESSIONS = 10
"""SessionManager 最多归档的旧会话数。"""

_VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})
"""LLM API 接受的消息角色集合。"""


# ============================================================================
# MemoryManager
# ============================================================================


class MemoryManager:
    """对话记忆管理器 — MVP 纯内存实现。

    维护当前会话的对话历史，提供消息添加、查询、裁剪和统计功能。
    每条消息自动记录时间戳，支持按角色检索最新消息。

    Attributes:
        MAX_HISTORY: 单会话最大消息条数（类常量，默认 50）。
    """

    MAX_HISTORY: int = _DEFAULT_MAX_HISTORY
    """单会话最大消息条数。超出时自动裁剪为最近 25 条。"""

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """初始化空的对话记忆。

        ``_history`` 内部存储格式::

            {
                "role": "user" | "assistant" | "tool",
                "content": "...",
                "timestamp": 1700000000.123,
                # 以下字段按需存在:
                "tool_call_id": "call_abc123",   # role="tool" 时
                "tool_calls": [...],              # role="assistant" 时
            }
        """
        self._history: list[dict[str, Any]] = []
        self._session_id: str = self._generate_session_id()
        self._created_at: float = time.monotonic()
        self._message_count: int = 0  # 累计消息数（含已裁剪的）

        logger.info(
            f"MemoryManager 创建: session_id={self._session_id}"
        )

    # ------------------------------------------------------------------
    # 消息操作
    # ------------------------------------------------------------------

    def add(self, role: str, content: str, **extra_fields: Any) -> None:
        """添加一条消息到对话历史。

        自动记录时间戳，超限时触发裁剪（保留最近 25 条）。

        .. note:: 并发安全性

            MVP 阶段所有操作在单协程 ReAct 循环中串行执行，无竞争风险。
            V2 引入并发工具调用时，需将本方法改为 async 并添加 asyncio.Lock。

        Args:
            role: 消息角色，必须为 ``"system"``、``"user"``、``"assistant"``
                或 ``"tool"`` 之一。
            content: 消息文本内容。空字符串也会被记录。
            **extra_fields: 额外字段（如 ``tool_call_id``、``tool_calls``），
                会原样存入历史条目。

        Raises:
            ValueError: role 为空字符串或非法角色时。
        """
        if not role:
            raise ValueError("消息角色不能为空字符串")
        if role not in _VALID_ROLES:
            raise ValueError(
                f"非法的消息角色: {role!r}。"
                f"允许的角色: {', '.join(sorted(_VALID_ROLES))}"
            )

        entry: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": time.monotonic(),
            **extra_fields,
        }
        self._history.append(entry)
        self._message_count += 1

        # 超限裁剪
        if len(self._history) > self.MAX_HISTORY:
            logger.info(
                f"对话历史超限 ({len(self._history)} > {self.MAX_HISTORY})，"
                f"触发裁剪"
            )
            self._trim_history()

    # ------------------------------------------------------------------
    # AgentExecutor 兼容接口
    # ------------------------------------------------------------------

    async def save(self, messages: list[dict[str, Any]]) -> None:
        """批量保存消息（供 AgentExecutor 调用的异步接口）。

        对每条消息调用 ``add()``，保留 ``tool_call_id`` /
        ``tool_calls`` 等扩展字段。

        Args:
            messages: 消息列表，每项含 ``"role"`` 和 ``"content"`` 键，
                以及可选的 ``tool_call_id`` / ``tool_calls`` 等字段。
        """
        for msg in messages:
            extra = {}
            if "tool_call_id" in msg:
                extra["tool_call_id"] = msg["tool_call_id"]
            if "tool_calls" in msg:
                extra["tool_calls"] = msg["tool_calls"]
            self.add(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                **extra,
            )
        logger.debug(f"save: {len(messages)} 条消息已保存")

    async def load(self) -> list[dict[str, Any]]:
        """加载对话历史（供 AgentExecutor 调用的异步接口）。

        返回内部存储的原始 dict 列表，由 Executor 自行转换为
        ``Message`` 对象。

        Returns:
            对话历史列表，每项为含 role / content / timestamp 的 dict。
            V2 可扩展为从文件/数据库加载。
        """
        logger.debug(f"load: 返回 {len(self._history)} 条消息")
        return list(self._history)

    # ------------------------------------------------------------------
    # 历史查询
    # ------------------------------------------------------------------

    def get_history(self, last_n: int | None = None) -> list[Message]:
        """获取对话历史，返回 ``Message`` 对象列表。

        保留扩展字段（``tool_call_id``、``tool_calls``），
        确保 LLM API 调用时可正确关联 tool-call / tool-result。

        Args:
            last_n: 仅返回最近 N 条消息。None 时返回全部。

        Returns:
            ``list[Message]``，按时间升序排列。
        """
        items = self._history[-last_n:] if last_n else self._history
        return [
            Message(
                role=item["role"],
                content=item["content"],
                tool_call_id=item.get("tool_call_id"),
                tool_calls=item.get("tool_calls"),
            )
            for item in items
        ]

    def get_last_user_message(self) -> Message | None:
        """获取最近一条 user 角色消息。

        Returns:
            最近一条 user 消息，无 user 消息时返回 None。
        """
        for item in reversed(self._history):
            if item["role"] == "user":
                return Message(
                    role="user",
                    content=item["content"],
                    tool_call_id=item.get("tool_call_id"),
                    tool_calls=item.get("tool_calls"),
                )
        return None

    def get_last_assistant_message(self) -> Message | None:
        """获取最近一条 assistant 角色消息。

        Returns:
            最近一条 assistant 消息，无 assistant 消息时返回 None。
        """
        for item in reversed(self._history):
            if item["role"] == "assistant":
                return Message(
                    role="assistant",
                    content=item["content"],
                    tool_call_id=item.get("tool_call_id"),
                    tool_calls=item.get("tool_calls"),
                )
        return None

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """清空对话历史，保留会话 ID 和创建时间。

        ``_message_count`` 归零，相当于重新开始同一会话。
        """
        count = len(self._history)
        self._history.clear()
        self._message_count = 0
        logger.info(
            f"对话历史已清空: 移除了 {count} 条消息, "
            f"session_id={self._session_id}"
        )

    # ------------------------------------------------------------------
    # V4 预留接口
    # ------------------------------------------------------------------

    def summarize(self) -> str | None:
        """[V4] 生成当前对话的摘要文本。

        V4 阶段将调用 LLM 对长对话进行压缩，返回精炼的摘要字符串。
        MVP 阶段始终返回 None。

        Returns:
            对话摘要文本，MVP 始终返回 None。
        """
        return None

    def get_stats(self) -> dict[str, Any]:
        """获取会话统计信息。

        Returns:
            dict 包含:
                - ``session_id`` (str): 会话唯一标识
                - ``created_at`` (float): 会话创建时间戳
                - ``message_count`` (int): 累计消息总数（含已裁剪）
                - ``current_size`` (int): 当前内存中的消息条数
                - ``age_seconds`` (float): 会话存活时间（秒）
                - ``role_distribution`` (dict): 各角色消息数量分布
                - ``last_activity`` (float | None): 最近一条消息的时间戳
        """
        role_dist: dict[str, int] = {}
        for item in self._history:
            role = item["role"]
            role_dist[role] = role_dist.get(role, 0) + 1

        return {
            "session_id": self._session_id,
            "created_at": self._created_at,
            "message_count": self._message_count,
            "current_size": len(self._history),
            "age_seconds": time.monotonic() - self._created_at,
            "role_distribution": role_dist,
            "last_activity": (
                self._history[-1]["timestamp"] if self._history else None
            ),
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _generate_session_id(self) -> str:
        """生成唯一的会话标识符。

        使用 uuid4 的 hex 字符串前 16 位作为短 ID，
        兼顾唯一性和终端展示友好性。

        Returns:
            16 字符的十六进制字符串，如 ``"a1b2c3d4e5f6g7h8"``。
        """
        return uuid.uuid4().hex[:_SESSION_ID_BYTES * 2]

    def _trim_history(self) -> None:
        """裁剪对话历史，保留最近的消息。

        策略：
            1. 保留所有 system 消息（索引无关紧要，都会被保留）
            2. 其余消息从旧到新丢弃，直至不超过 ``MAX_HISTORY``
            3. 确保对话上下文聚焦于最新交互

        Note:
            裁剪是静默的——仅记录日志，不改变 ``_message_count``
            （该字段反映累计总数）。
        """
        if len(self._history) <= self.MAX_HISTORY:
            return

        # 收集所有 system 消息（始终保留）
        system_msgs = [
            item for item in self._history if item["role"] == "system"
        ]
        non_system = [
            item for item in self._history if item["role"] != "system"
        ]

        # 对非 system 消息保留最近 _TRIM_KEEP_COUNT 条
        budget = max(_TRIM_KEEP_COUNT - len(system_msgs), 0)
        kept_non_system = non_system[-budget:] if budget > 0 else []

        removed = len(self._history) - len(system_msgs) - len(kept_non_system)
        self._history = system_msgs + kept_non_system

        logger.info(
            f"对话历史已裁剪: 移除 {removed} 条旧消息, "
            f"保留 {len(self._history)} 条 "
            f"(system={len(system_msgs)}, 其他={len(kept_non_system)}), "
            f"session_id={self._session_id}"
        )


# ============================================================================
# SessionManager
# ============================================================================


class SessionManager:
    """会话管理器 — MVP 持有单个 MemoryManager。

    管理 MemoryManager 实例的生命周期，支持创建新会话和
    查询会话状态。MVP 仅维护当前活跃会话。

    Attributes:
        current: 当前的 ``MemoryManager`` 实例。

    Typical usage::

        sm = SessionManager()
        sm.current.add("user", "你好")
        stats = sm.current.get_stats()

        # 开始新对话
        old = sm.current
        new = sm.new_session()
        print(f"已归档 {old.get_stats()['current_size']} 条消息，开始新会话")
    """

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """初始化会话管理器，创建默认 MemoryManager。"""
        self._current = MemoryManager()
        self._previous_sessions: list[MemoryManager] = []

        logger.info(
            "SessionManager 初始化完成: "
            f"current_session={self._current._session_id}"
        )

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def current(self) -> MemoryManager:
        """获取当前活跃的 MemoryManager。

        Returns:
            当前的 MemoryManager 实例。
        """
        return self._current

    # ------------------------------------------------------------------
    # 会话操作
    # ------------------------------------------------------------------

    def new_session(self) -> MemoryManager:
        """创建新会话，将当前会话移至归档列表。

        旧会话不会被销毁——调用方可在归档前保存其统计信息。

        Returns:
            新创建的 ``MemoryManager`` 实例（同时设置为本会话管理器的
            ``current`` 属性）。

        Example::

            sm = SessionManager()
            sm.current.add("user", "第一轮对话")

            old_stats = sm.current.get_stats()
            sm.new_session()  # 旧会话归档，current 指向新会话
            print(f"上一轮共 {old_stats['current_size']} 条消息")
        """
        old = self._current
        self._previous_sessions.append(old)

        # 超出上限时丢弃最旧的归档会话
        while len(self._previous_sessions) > _MAX_ARCHIVED_SESSIONS:
            discarded = self._previous_sessions.pop(0)
            logger.debug(
                f"归档会话超限 ({_MAX_ARCHIVED_SESSIONS})，"
                f"丢弃最旧会话: {discarded._session_id}"
            )

        self._current = MemoryManager()

        logger.info(
            f"新会话已创建: new_session={self._current._session_id}, "
            f"旧会话已归档 (session={old._session_id}, "
            f"消息数={len(old._history)}, "
            f"归档总数={len(self._previous_sessions)})"
        )
        return self._current

    def get_session_count(self) -> int:
        """获取已创建的会话总数（含当前）。

        Returns:
            会话总数（当前 1 个 + 已归档 N 个）。
        """
        return 1 + len(self._previous_sessions)

    def get_all_stats(self) -> list[dict[str, Any]]:
        """获取所有会话的统计信息汇总。

        Returns:
            统计信息列表，按创建时间降序排列（最新在前）。
        """
        all_sessions = [self._current] + list(self._previous_sessions)
        return [s.get_stats() for s in all_sessions]
