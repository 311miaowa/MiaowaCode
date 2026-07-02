"""搜索工具单元测试 — SearchFilesTool。

覆盖文件名搜索、内容搜索、大小写敏感性、结果上限、
二进制跳过、空结果、无效正则、空查询、ReDoS 防护。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from miaowa.core.config import Config
from miaowa.tools.search import (
    SearchFilesTool,
    _expand_brace_pattern,
    _has_regex_metachars,
)


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------


def _make_file(dir_path: Path, name: str, content: str) -> Path:
    """创建 UTF-8 文本文件。"""
    f = dir_path / name
    f.write_text(content, encoding="utf-8")
    return f


def _make_binary_file(dir_path: Path, name: str) -> Path:
    """创建被判定为二进制的文件（null 字节 > 10%）。"""
    f = dir_path / name
    f.write_bytes(b"\x00" * 200 + b"text" + b"\x00" * 100)
    return f


# ============================================================================
# 文件名搜索
# ============================================================================


class TestSearchByFilename:
    """按文件名搜索。"""

    @pytest.mark.asyncio
    async def test_search_filename_exact_match(self, tmp_path, mock_config):
        """按文件名搜索找到匹配的文件。"""
        _make_file(tmp_path, "hello_world.py", "code")
        _make_file(tmp_path, "goodbye.py", "code")
        _make_file(tmp_path, "README.md", "doc")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="hello", search_type="filename")

        assert result.success
        assert result.data["search_type"] == "filename"
        assert result.data["total_results"] == 1
        assert result.data["results"][0]["name"] == "hello_world.py"

    @pytest.mark.asyncio
    async def test_search_filename_substring_match(self, tmp_path, mock_config):
        """文件名搜索为子串匹配（*query*）。"""
        _make_file(tmp_path, "test_utils.py", "code")
        _make_file(tmp_path, "utils_test.py", "code")
        _make_file(tmp_path, "main.py", "code")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="utils", search_type="filename")

        assert result.success
        names = [r["name"] for r in result.data["results"]]
        assert "test_utils.py" in names
        assert "utils_test.py" in names
        assert "main.py" not in names

    @pytest.mark.asyncio
    async def test_search_filename_with_file_pattern(self, tmp_path, mock_config):
        """file_pattern 限制搜索的文件类型。"""
        _make_file(tmp_path, "test_app.py", "code")
        _make_file(tmp_path, "test_app.js", "code")
        _make_file(tmp_path, "test_app.md", "doc")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(
            query="test", search_type="filename", file_pattern="*.py"
        )

        assert result.success
        names = [r["name"] for r in result.data["results"]]
        assert "test_app.py" in names
        assert "test_app.js" not in names

    @pytest.mark.asyncio
    async def test_search_filename_recursive(self, tmp_path, mock_config):
        """文件名搜索递归进入子目录。"""
        sub = tmp_path / "subdir"
        sub.mkdir()
        _make_file(sub, "nested_file.py", "code")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="nested", search_type="filename")

        assert result.success
        assert result.data["total_results"] == 1
        assert result.data["results"][0]["name"] == "nested_file.py"


# ============================================================================
# 内容搜索
# ============================================================================


class TestSearchByContent:
    """按文件内容搜索。"""

    @pytest.mark.asyncio
    async def test_search_content_finds_match(self, tmp_path, mock_config):
        """内容搜索返回匹配的行。"""
        _make_file(tmp_path, "config.py", "DEBUG = True\nLOG_LEVEL = 'INFO'\n")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="DEBUG", search_type="content")

        assert result.success
        assert result.data["search_type"] == "content"
        assert result.data["total_results"] == 1
        assert result.data["results"][0]["line"] == 1
        assert "DEBUG" in result.data["results"][0]["content"]

    @pytest.mark.asyncio
    async def test_search_content_multiple_matches(self, tmp_path, mock_config):
        """同一文件中多行匹配全部返回。"""
        content = "import os\nimport sys\nimport re\n"
        _make_file(tmp_path, "imports.py", content)

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="import", search_type="content")

        assert result.success
        assert result.data["total_results"] == 3
        lines = [r["line"] for r in result.data["results"]]
        assert lines == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_search_content_with_file_pattern(self, tmp_path, mock_config):
        """file_pattern 在内容搜索中限制候选文件。"""
        _make_file(tmp_path, "notes.md", "Important TODO: fix bug")
        _make_file(tmp_path, "todo.py", "# TODO: implement feature")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(
            query="TODO", search_type="content", file_pattern="*.py"
        )

        assert result.success
        paths = [r["path"] for r in result.data["results"]]
        assert "todo.py" in paths
        assert "notes.md" not in paths


# ============================================================================
# 大小写敏感性
# ============================================================================


class TestSearchCaseInsensitive:
    """大小写不敏感搜索。"""

    @pytest.mark.asyncio
    async def test_filename_case_insensitive(self, tmp_path, mock_config):
        """默认不区分大小写匹配文件名。"""
        _make_file(tmp_path, "MyModule.py", "code")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="mymodule", search_type="filename")

        assert result.success
        assert result.data["total_results"] == 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="Windows fnmatch 跟随 OS 大小写不敏感语义，case_sensitive=True 不生效",
    )
    async def test_filename_case_sensitive(self, tmp_path, mock_config):
        """case_sensitive=True 时大小写敏感（非 Windows 平台）。"""
        _make_file(tmp_path, "MyModule.py", "code")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(
            query="mymodule", search_type="filename", case_sensitive=True
        )

        # 大小写敏感模式：mymodule ≠ MyModule，应返回 0 结果
        assert result.success
        assert result.data["total_results"] == 0

    @pytest.mark.asyncio
    async def test_content_case_insensitive(self, tmp_path, mock_config):
        """内容搜索默认不区分大小写。"""
        _make_file(tmp_path, "f.py", "Hello World\n")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="hello", search_type="content")

        assert result.success
        assert result.data["total_results"] == 1

    @pytest.mark.asyncio
    async def test_content_case_sensitive(self, tmp_path, mock_config):
        """内容搜索大小写敏感模式。"""
        _make_file(tmp_path, "f.py", "Hello World\n")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(
            query="hello", search_type="content", case_sensitive=True
        )

        assert result.success
        assert result.data["total_results"] == 0

    @pytest.mark.asyncio
    async def test_file_pattern_case_insensitive(self, tmp_path, mock_config):
        """file_pattern 也不区分大小写。"""
        _make_file(tmp_path, "Test.PY", "code")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(
            query="Test", search_type="filename", file_pattern="*.py"
        )

        assert result.success
        assert result.data["total_results"] == 1


# ============================================================================
# 结果上限
# ============================================================================


class TestSearchMaxResults:
    """max_results 限制。"""

    @pytest.mark.asyncio
    async def test_max_results_caps_output(self, tmp_path, mock_config):
        """max_results 正确限制返回数。"""
        for i in range(20):
            _make_file(tmp_path, f"item_{i:02d}.py", f"# item {i}")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="item", search_type="content", max_results=5)

        assert result.success
        assert result.data["total_results"] == 5
        assert result.data["truncated"] is True

    @pytest.mark.asyncio
    async def test_max_results_less_than_one_rejected(self, tmp_path, mock_config):
        """max_results < 1 返回 fail。"""
        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="test", max_results=0)

        assert not result.success
        assert "max_results" in result.error


# ============================================================================
# 二进制跳过
# ============================================================================


class TestSearchBinarySkip:
    """内容搜索自动跳过二进制文件。"""

    @pytest.mark.asyncio
    async def test_binary_file_skipped_in_content_search(self, tmp_path, mock_config):
        """内容搜索时二进制文件被静默跳过。"""
        _make_binary_file(tmp_path, "data.bin")
        _make_file(tmp_path, "README.md", "hello world\n")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="hello", search_type="content")

        assert result.success
        # 只有 README.md 的结果，data.bin 被跳过
        paths = [r["path"] for r in result.data["results"]]
        assert all(not p.endswith(".bin") for p in paths)

    @pytest.mark.asyncio
    async def test_utf16_file_not_skipped(self, tmp_path, mock_config):
        """UTF-16 BOM 文件在文件名搜索中不被误跳过。"""
        f = tmp_path / "utf16.txt"
        f.write_bytes(b"\xff\xfeh\x00e\x00l\x00l\x00o\x00")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="utf16", search_type="filename")

        assert result.success
        assert any(r["name"] == "utf16.txt" for r in result.data["results"])


# ============================================================================
# 无结果
# ============================================================================


class TestSearchNoResults:
    """无匹配结果场景。"""

    @pytest.mark.asyncio
    async def test_no_filename_match(self, tmp_path, mock_config):
        """文件名搜索无匹配返回空。"""
        _make_file(tmp_path, "a.py", "code")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="nonexistent", search_type="filename")

        assert result.success
        assert result.data["total_results"] == 0
        assert result.data["results"] == []

    @pytest.mark.asyncio
    async def test_no_content_match(self, tmp_path, mock_config):
        """内容搜索无匹配返回空。"""
        _make_file(tmp_path, "code.py", "print('hello')")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="NONEXISTENT_TEXT_XYZ", search_type="content")

        assert result.success
        assert result.data["total_results"] == 0

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path, mock_config):
        """空目录无结果。"""
        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="anything", search_type="filename")

        assert result.success
        assert result.data["total_results"] == 0


# ============================================================================
# 无效正则
# ============================================================================


class TestSearchInvalidRegex:
    """正则语法错误时的回退。"""

    @pytest.mark.asyncio
    async def test_invalid_regex_falls_back_to_literal(self, tmp_path, mock_config):
        """无效正则表达式自动回退为字面搜索。"""
        # "[unclosed" 是无效正则
        _make_file(tmp_path, "test.py", "pattern: [unclosed bracket\n")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="[unclosed", search_type="content")

        assert result.success
        # 应回退为字面搜索，找到包含 [unclosed 的行
        assert result.data["total_results"] >= 1

    @pytest.mark.asyncio
    async def test_valid_regex_used_directly(self, tmp_path, mock_config):
        """有效正则直接用于搜索。"""
        _make_file(tmp_path, "nums.py", "abc123def\nxyz456uvw\n")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query=r"\d+", search_type="content")

        assert result.success
        assert result.data["total_results"] == 2


# ============================================================================
# 空查询
# ============================================================================


class TestSearchEmptyQuery:
    """空查询或空白查询拒绝。"""

    @pytest.mark.asyncio
    async def test_empty_query_rejected(self, tmp_path, mock_config):
        """空字符串查询返回 fail。"""
        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="", search_type="filename")

        assert not result.success
        assert "不能为空" in result.error

    @pytest.mark.asyncio
    async def test_whitespace_only_query_rejected(self, tmp_path, mock_config):
        """仅空白字符的查询返回 fail。"""
        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="   \t  ", search_type="content")

        assert not result.success
        assert "不能为空" in result.error


# ============================================================================
# ReDoS 防护
# ============================================================================


class TestSearchReDoSProtection:
    """ReDoS 防护：超长行截断 + 恶意正则。"""

    @pytest.mark.asyncio
    async def test_long_line_truncated(self, tmp_path, mock_config):
        """超过 _MAX_LINE_LENGTH 的行被截断后再搜索（不会卡死）。"""
        # 创建一行极长的文本（超过 4096 字符），末尾放关键词
        long_line = "x" * 5000 + " FINDME_HERE"
        _make_file(tmp_path, "long.txt", long_line + "\n")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="FINDME_HERE", search_type="content")

        # 行被截断到 4096 字符，关键词在 5000 位置 → 可能找不到
        # 但关键是：不应卡死或超时
        assert result.success

    @pytest.mark.asyncio
    async def test_evil_regex_does_not_hang(self, tmp_path, mock_config):
        """真实 ReDoS 模式 (a+)+b 在受限输入下不挂起。

        (a+)+b 是 OWASP 认证的灾难性回溯正则。当输入 N 个 'a'
        且无结尾 'b' 时，Python re 模块会产生 O(2^N) 回溯。
        此处使用 N=20 确保测试在 <0.1s 完成；更长的行由
        _MAX_LINE_LENGTH=4096 截断保护。
        """
        # 20 个 'a' — 约 100 万次回溯，现代 CPU < 0.1s
        evil_line = "a" * 20 + " rest"
        _make_file(tmp_path, "evil.txt", evil_line + "\n")

        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query=r"(a+)+b", search_type="content")

        # 不应挂起，正常返回（无匹配或快速完成）
        assert result.success

        # 不应挂起，应正常返回
        assert result.success


# ============================================================================
# 路径安全
# ============================================================================


class TestSearchPathSecurity:
    """搜索工具的路径穿越防护。"""

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self, tmp_path, mock_config):
        """绝对路径查询被拒绝。"""
        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="test", search_type="filename", path="C:/Windows")

        assert not result.success
        assert "绝对路径" in result.error

    @pytest.mark.asyncio
    async def test_parent_directory_traversal_rejected(self, tmp_path, mock_config):
        """搜索路径 ../ 穿越被拦截。"""
        safe = tmp_path / "safe"
        safe.mkdir()
        tool = SearchFilesTool(project_root=safe, config=mock_config)
        result = await tool.execute(query="test", search_type="filename", path="../outside")

        assert not result.success
        assert "超出项目根目录" in result.error

    @pytest.mark.asyncio
    async def test_nonexistent_path(self, tmp_path, mock_config):
        """搜索路径不存在返回 fail。"""
        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(
            query="test", search_type="filename", path="no_such_dir"
        )

        assert not result.success
        assert "不存在" in result.error


# ============================================================================
# 真实二进制文件测试
# ============================================================================


class TestSearchRealBinaryFiles:
    """使用真实二进制文件（非合成 null 字节）验证检测逻辑。"""

    @pytest.fixture
    def real_png(self, tmp_path) -> Path:
        """生成最小化的真实 PNG 文件（1×1 红色像素）。"""
        import struct
        import zlib

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            crc = zlib.crc32(c) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        idat = zlib.compress(b"\x00\xff\x00\x00")
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", idat)
            + _chunk(b"IEND", b"")
        )
        f = tmp_path / "real.png"
        f.write_bytes(png_data)
        return f

    @pytest.mark.asyncio
    async def test_real_png_skipped_in_content_search(self, tmp_path, mock_config, real_png):
        """真实 PNG 文件在内容搜索中被跳过。"""
        _make_file(tmp_path, "notes.txt", "hello png\n")
        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="hello", search_type="content")

        assert result.success
        paths = [r["path"] for r in result.data["results"]]
        assert all(not p.endswith(".png") for p in paths)

    @pytest.mark.asyncio
    async def test_real_png_found_in_filename_search(self, tmp_path, mock_config, real_png):
        """真实 PNG 文件在文件名搜索中正常匹配。"""
        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="real", search_type="filename")

        assert result.success
        names = [r["name"] for r in result.data["results"]]
        assert "real.png" in names


# ============================================================================
# _BINARY_DETECT_HEADER_SIZE 边界测试
# ============================================================================


class TestBinaryHeaderBoundary:
    """_BINARY_DETECT_HEADER_SIZE=1024 边界附近的二进制检测。"""

    @pytest.fixture
    def _make_sized_file(self, tmp_path):
        """工厂：创建指定大小的文件。"""
        def _create(name: str, size: int, *, null_ratio: float = 0.0) -> Path:
            f = tmp_path / name
            if null_ratio > 0:
                null_bytes = int(size * null_ratio)
                data = b"\x00" * null_bytes + b"x" * (size - null_bytes)
            else:
                data = b"x" * size
            f.write_bytes(data)
            return f
        return _create

    @pytest.mark.asyncio
    async def test_exactly_header_size_no_nulls(self, tmp_path, mock_config, _make_sized_file):
        """恰好 1024 字节、无 null → 识别为文本。"""
        _make_sized_file("exact.txt", 1024)
        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="xxx", search_type="content")
        assert result.success

    @pytest.mark.asyncio
    async def test_smaller_than_header_size(self, tmp_path, mock_config, _make_sized_file):
        """小于 1024 字节（1023）、有少量 null → 仍低于阈值（~10%）→ 文本。"""
        f = tmp_path / "small.bin"
        f.write_bytes(b"\x00" * 100 + b"x" * 923)  # 1023 bytes, ~9.8% nulls
        tool = SearchFilesTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(query="xxx", search_type="content")
        assert result.success


# ============================================================================
# 辅助函数
# ============================================================================


class TestExpandBracePattern:
    """_expand_brace_pattern 大括号展开。"""

    def test_single_brace_expansion(self):
        """{a,b} 展开为多个模式。"""
        assert _expand_brace_pattern("*.{py,js}") == ["*.py", "*.js"]

    def test_no_brace_returns_single(self):
        """无大括号返回单元素列表。"""
        assert _expand_brace_pattern("*.py") == ["*.py"]

    def test_empty_brace_returns_original(self):
        """空大括号 {} 返回原模式。"""
        assert _expand_brace_pattern("*.{}") == ["*.{}"]

    def test_multiple_extensions(self):
        """多选项展开。"""
        result = _expand_brace_pattern("*.{py,js,ts,jsx}")
        assert result == ["*.py", "*.js", "*.ts", "*.jsx"]


class TestHasRegexMetachars:
    """_has_regex_metachars 检测正则元字符。"""

    def test_plain_text_no_metachars(self):
        """纯文本不含元字符。"""
        assert _has_regex_metachars("hello world") is False
        assert _has_regex_metachars("simple_text_123") is False

    def test_regex_with_metachars(self):
        """正则表达式含元字符。"""
        assert _has_regex_metachars(r"\d+") is True
        assert _has_regex_metachars(r"foo.*bar") is True
        assert _has_regex_metachars(r"[abc]") is True
        assert _has_regex_metachars(r"(group)") is True
        assert _has_regex_metachars(r"a{2,4}") is True
