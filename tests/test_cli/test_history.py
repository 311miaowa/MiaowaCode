"""CommandHistory 单元测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from miaowa.cli.history import CommandHistory


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def history() -> CommandHistory:
    """提供空的 CommandHistory。"""
    return CommandHistory(max_size=100)


# ---------------------------------------------------------------------------
# 构造与基本属性
# ---------------------------------------------------------------------------


class TestInit:
    """初始化测试。"""

    def test_default_max_size(self) -> None:
        h = CommandHistory()
        assert h.max_size == 1000

    def test_custom_max_size(self) -> None:
        h = CommandHistory(max_size=500)
        assert h.max_size == 500

    def test_pt_history_created(self) -> None:
        h = CommandHistory()
        assert h.pt_history is not None


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------


class TestAppend:
    """append 方法测试。"""

    def test_append_single(self, history: CommandHistory) -> None:
        history.append("hello")
        assert history.count == 1
        assert history.get_all() == ["hello"]

    def test_append_multiple(self, history: CommandHistory) -> None:
        history.append("first")
        history.append("second")
        history.append("third")
        assert history.count == 3
        assert history.get_all() == ["first", "second", "third"]

    def test_append_empty_string(self, history: CommandHistory) -> None:
        """空字符串不应被追加。"""
        history.append("")
        assert history.count == 0

    def test_append_whitespace_only(self, history: CommandHistory) -> None:
        """纯空白字符串不应被追加。"""
        history.append("   ")
        assert history.count == 0

    def test_append_strips_whitespace(self, history: CommandHistory) -> None:
        """追加时应去除首尾空白。"""
        history.append("  hello  ")
        assert history.get_all() == ["hello"]

    def test_append_meta_command(self, history: CommandHistory) -> None:
        """元命令也应被正确记录。"""
        history.append("/model deepseek-chat")
        assert history.count == 1
        assert history.get_all() == ["/model deepseek-chat"]


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    """clear 方法测试。"""

    def test_clear_empty(self, history: CommandHistory) -> None:
        """清空空历史不应崩溃。"""
        history.clear()
        assert history.count == 0

    def test_clear_with_items(self, history: CommandHistory) -> None:
        history.append("cmd1")
        history.append("cmd2")
        assert history.count == 2
        history.clear()
        assert history.count == 0
        assert history.get_all() == []

    def test_after_clear_can_append(self, history: CommandHistory) -> None:
        """清空后仍可追加新命令。"""
        history.append("old")
        history.clear()
        history.append("new")
        assert history.get_all() == ["new"]


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------


class TestGetAll:
    """get_all 方法测试。"""

    def test_empty_returns_empty_list(self, history: CommandHistory) -> None:
        assert history.get_all() == []

    def test_returns_copy_not_reference(self, history: CommandHistory) -> None:
        """返回的列表应为独立副本。"""
        history.append("cmd")
        items = history.get_all()
        items.append("injected")
        assert history.get_all() == ["cmd"]


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


class TestCount:
    """count 属性测试。"""

    def test_initial_zero(self, history: CommandHistory) -> None:
        assert history.count == 0

    def test_reflects_append(self, history: CommandHistory) -> None:
        history.append("a")
        assert history.count == 1
        history.append("b")
        assert history.count == 2

    def test_reflects_clear(self, history: CommandHistory) -> None:
        history.append("a")
        history.clear()
        assert history.count == 0


# ---------------------------------------------------------------------------
# V4 预留接口
# ---------------------------------------------------------------------------


class TestV4Stubs:
    """V4 文件持久化桩方法测试。"""

    def test_save_to_file_does_not_crash(self, history: CommandHistory) -> None:
        """save_to_file 桩方法不应抛出异常。"""
        history.append("test")
        history.save_to_file(Path("/tmp/nonexistent/test.json"))

    def test_load_from_file_does_not_crash(self, history: CommandHistory) -> None:
        """load_from_file 桩方法不应抛出异常。"""
        history.load_from_file(Path("/tmp/nonexistent/test.json"))

    def test_load_from_file_does_not_modify_history(
        self, history: CommandHistory
    ) -> None:
        """load_from_file 桩方法不应修改现有历史。"""
        history.append("existing")
        history.load_from_file(Path("/tmp/nonexistent/test.json"))
        assert history.get_all() == ["existing"]

    def test_save_and_load_roundtrip_stub(self, history: CommandHistory) -> None:
        """桩方法的保存/加载不产生实际效果，但不应崩溃。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            history.append("test")
            history.save_to_file(path)
            # 桩方法不写文件，load 也不读
            history.load_from_file(path)
            # 历史不变
            assert history.get_all() == ["test"]


# ---------------------------------------------------------------------------
# prompt_toolkit 集成
# ---------------------------------------------------------------------------


class TestPromptToolkitIntegration:
    """与 prompt_toolkit InMemoryHistory 的集成测试。"""

    def test_pt_history_reflects_append(self) -> None:
        h = CommandHistory()
        h.append("hello")
        items = list(h.pt_history.get_strings())
        assert items == ["hello"]

    def test_pt_history_reflects_clear(self) -> None:
        h = CommandHistory()
        h.append("hello")
        h.clear()
        items = list(h.pt_history.get_strings())
        assert items == []

    def test_pt_history_append_directly(self) -> None:
        """直接操作 pt_history 也会反映到 get_all。"""
        h = CommandHistory()
        h.pt_history.append_string("direct")
        assert "direct" in h.get_all()
