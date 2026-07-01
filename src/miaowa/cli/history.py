"""命令历史 — 基于 prompt_toolkit InMemoryHistory 的命令持久化。

MVP 提供内存历史（InMemoryHistory），V4 将扩展文件持久化。
"""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.history import InMemoryHistory


class CommandHistory:
    """基于 prompt_toolkit InMemoryHistory 的 REPL 命令历史。

    封装 ``InMemoryHistory``，提供文件持久化的预留接口（V4 实现）。
    通过 ``pt_history`` 属性可直接传给 ``PromptSession``。

    Attributes:
        pt_history: prompt_toolkit 的 ``InMemoryHistory`` 实例，
            供 ``PromptSession(history=...)`` 使用。

    Usage::

        hist = CommandHistory(max_size=1000)
        session = PromptSession(history=hist.pt_history)
    """

    def __init__(self, max_size: int = 1000) -> None:
        """初始化命令历史。

        Args:
            max_size: 内存中保留的最大历史条数。V4 持久化时
                文件中的历史不受此限制。
        """
        self.max_size = max_size
        self.pt_history = InMemoryHistory()

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def append(self, text: str) -> None:
        """追加一条命令到历史。

        Args:
            text: 用户输入的命令文本。
        """
        if text.strip():
            self.pt_history.append_string(text.strip())

    def clear(self) -> None:
        """清空内存中的全部历史。"""
        # InMemoryHistory 没有 clear() 方法，通过替换新实例实现清空
        max_size = self.max_size
        self.pt_history = InMemoryHistory()

    def get_all(self) -> list[str]:
        """获取所有历史命令。

        Returns:
            命令列表，按时间升序排列（最早在前）。
        """
        items = []
        for item in self.pt_history.get_strings():
            items.append(item)
        return items

    @property
    def count(self) -> int:
        """当前历史条数。"""
        return len(self.get_all())

    # ------------------------------------------------------------------
    # V4 预留接口（文件持久化）
    # ------------------------------------------------------------------

    def save_to_file(self, filepath: Path) -> None:
        """[V4] 将历史保存到文件。

        V4 将实现 JSON 或纯文本格式的持久化存储。

        Args:
            filepath: 目标文件路径。

        Note:
            MVP 阶段为预留桩方法，调用不产生任何效果。
        """
        # V4: 实现 JSON Lines 持久化
        # with open(filepath, "a", encoding="utf-8") as f:
        #     for cmd in self.get_all():
        #         f.write(json.dumps({"cmd": cmd}) + "\n")
        pass

    def load_from_file(self, filepath: Path) -> None:
        """[V4] 从文件加载历史。

        V4 将实现从 JSON/纯文本文件恢复历史记录。

        Args:
            filepath: 源文件路径。

        Note:
            MVP 阶段为预留桩方法，调用不产生任何效果。
        """
        # V4: 从 JSON Lines 文件加载
        # if filepath.exists():
        #     with open(filepath, "r", encoding="utf-8") as f:
        #         for line in f:
        #             data = json.loads(line)
        #             self.append(data["cmd"])
        pass
