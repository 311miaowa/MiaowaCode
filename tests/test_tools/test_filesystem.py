"""文件系统工具单元测试 — ReadFileTool 与 ListDirectoryTool。

覆盖正常读取、错误处理、二进制检测、编码回退、行号截取、
路径穿越防护、目录列表（递归/过滤/排除/排序）等场景。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from miaowa.core.types import ToolResult
from miaowa.tools.filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    _resolve_path_within_root,
    _PathValidationError,
)


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------


def _make_text_file(dir_path: Path, name: str, content: str) -> Path:
    """在指定目录中创建 UTF-8 文本文件并返回其 Path。"""
    f = dir_path / name
    f.write_text(content, encoding="utf-8")
    return f


def _make_binary_file(dir_path: Path, name: str) -> Path:
    """创建被判定为二进制的文件（null 字节占比 > 10%）。"""
    f = dir_path / name
    # 1024 字节的 header 检测窗口，需要 >103 个 null 字节
    data = b"\x00" * 200 + b"some text" + b"\x00" * 100
    f.write_bytes(data)
    return f


# ============================================================================
# ReadFileTool 测试
# ============================================================================


class TestReadFileSuccess:
    """正常读取场景。"""

    @pytest.mark.asyncio
    async def test_read_entire_file(self, tmp_path, mock_config):
        """读取完整文件内容并验证返回字段。"""
        _make_text_file(tmp_path, "hello.py", "print('hello')\nprint('world')\n")

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="hello.py")

        assert result.success
        assert result.data["path"] == "hello.py"
        assert result.data["content"] == "print('hello')\nprint('world')"
        assert result.data["total_lines"] == 2
        assert result.data["lines_returned"] == 2
        assert result.data["end_line_truncated"] is False

    @pytest.mark.asyncio
    async def test_read_with_crlf_line_endings(self, tmp_path, mock_config):
        """CRLF 行尾被 splitlines() 正确处理。"""
        f = tmp_path / "crlf.txt"
        f.write_bytes(b"line1\r\nline2\r\nline3\r\n")

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="crlf.txt")

        assert result.success
        assert result.data["total_lines"] == 3
        assert result.data["content"] == "line1\nline2\nline3"

    @pytest.mark.asyncio
    async def test_read_with_bom_utf8(self, tmp_path, mock_config):
        """UTF-8 BOM 文件被正确识别为文本并解码。"""
        f = tmp_path / "bom.txt"
        f.write_bytes(b"\xef\xbb\xbfhello world\n")

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="bom.txt")

        assert result.success
        assert "hello world" in result.data["content"]


class TestReadFileNotFound:
    """文件不存在或非普通文件场景。"""

    @pytest.mark.asyncio
    async def test_file_not_found(self, tmp_path, mock_config):
        """不存在的文件返回 fail。"""
        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="nonexistent.py")

        assert not result.success
        assert "文件不存在" in result.error

    @pytest.mark.asyncio
    async def test_path_is_directory(self, tmp_path, mock_config):
        """路径指向目录时返回 fail。"""
        subdir = tmp_path / "mydir"
        subdir.mkdir()

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="mydir")

        assert not result.success
        assert "不是普通文件" in result.error

    @pytest.mark.asyncio
    async def test_file_too_large(self, tmp_path, mock_config):
        """超过 max_file_size 的文件返回 fail。"""
        f = tmp_path / "huge.py"
        # 创建刚好超过 1MB 的文件
        mock_config.project.max_file_size = 1024
        f.write_bytes(b"x" * 2048)

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="huge.py")

        assert not result.success
        assert "文件过大" in result.error


class TestReadFileBinaryRejected:
    """二进制文件拒绝场景。"""

    @pytest.mark.asyncio
    async def test_binary_file_rejected(self, tmp_path, mock_config):
        """null 字节占比超阈值时返回 fail。"""
        _make_binary_file(tmp_path, "data.bin")

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="data.bin")

        assert not result.success
        assert "二进制" in result.error

    @pytest.mark.asyncio
    async def test_utf16_bom_accepted_as_text(self, tmp_path, mock_config):
        """UTF-16 LE BOM 文件通过 BOM 检测，不被误判为二进制。"""
        f = tmp_path / "utf16.txt"
        # UTF-16 LE 编码："hello" → h\0e\0l\0l\0o\0
        f.write_bytes(b"\xff\xfeh\x00e\x00l\x00l\x00o\x00")

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="utf16.txt")

        # BOM 检测先于 null 字节检测，应成功
        assert result.success


class TestReadFileLineRange:
    """行号范围截取。"""

    @pytest.mark.asyncio
    async def test_read_line_range(self, tmp_path, mock_config):
        """start_line 和 end_line 正确截取内容。"""
        content = "\n".join(f"line {i}" for i in range(1, 11)) + "\n"  # 10 lines
        _make_text_file(tmp_path, "lines.txt", content)

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="lines.txt", start_line=3, end_line=5)

        assert result.success
        assert result.data["lines_returned"] == 3
        assert result.data["content"] == "line 3\nline 4\nline 5"

    @pytest.mark.asyncio
    async def test_end_line_none_reads_to_end(self, tmp_path, mock_config):
        """end_line=None 时读到文件末尾。"""
        content = "\n".join(f"line {i}" for i in range(1, 6)) + "\n"
        _make_text_file(tmp_path, "lines.txt", content)

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="lines.txt", start_line=4)

        assert result.success
        assert "line 4" in result.data["content"]
        assert "line 5" in result.data["content"]

    @pytest.mark.asyncio
    async def test_end_line_truncated_when_exceeds_total(self, tmp_path, mock_config):
        """end_line 超出总行数时静默截断并标记。"""
        content = "a\nb\nc\n"
        _make_text_file(tmp_path, "short.txt", content)

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="short.txt", start_line=1, end_line=100)

        assert result.success
        assert result.data["end_line_truncated"] is True
        assert result.data["lines_returned"] == 3

    @pytest.mark.asyncio
    async def test_read_file_max_lines_cap(self, tmp_path, mock_config):
        """read_file_max_lines 上限生效。"""
        lines = [f"line {i}" for i in range(1, 101)]  # 100 lines
        _make_text_file(tmp_path, "big.txt", "\n".join(lines) + "\n")

        mock_config.tools.read_file_max_lines = 20
        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="big.txt")

        assert result.success
        assert result.data["lines_returned"] <= 20
        assert result.data["end_line_truncated"] is True


class TestReadFileStartExceeds:
    """start_line 超限场景。"""

    @pytest.mark.asyncio
    async def test_start_line_exceeds_total(self, tmp_path, mock_config):
        """start_line 超过文件总行数返回 fail。"""
        _make_text_file(tmp_path, "short.txt", "only one line\n")

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="short.txt", start_line=100)

        assert not result.success
        assert "超出文件总行数" in result.error

    @pytest.mark.asyncio
    async def test_start_line_less_than_one(self, tmp_path, mock_config):
        """start_line < 1 返回 fail。"""
        _make_text_file(tmp_path, "f.py", "content\n")

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="f.py", start_line=0)

        assert not result.success
        assert "start_line 必须 >= 1" in result.error


class TestReadFileStartGtEnd:
    """start_line > end_line 场景。"""

    @pytest.mark.asyncio
    async def test_start_greater_than_end(self, tmp_path, mock_config):
        """start_line > end_line 返回 fail。"""
        content = "\n".join(f"l{i}" for i in range(1, 11)) + "\n"
        _make_text_file(tmp_path, "lines.txt", content)

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="lines.txt", start_line=5, end_line=3)

        assert not result.success
        assert "end_line" in result.error

    @pytest.mark.asyncio
    async def test_start_equals_end(self, tmp_path, mock_config):
        """start_line == end_line 返回单行。"""
        content = "\n".join(f"line {i}" for i in range(1, 6)) + "\n"
        _make_text_file(tmp_path, "lines.txt", content)

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="lines.txt", start_line=2, end_line=2)

        assert result.success
        assert result.data["lines_returned"] == 1
        assert result.data["content"] == "line 2"


class TestReadFileEmpty:
    """空文件场景。"""

    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path, mock_config):
        """0 字节文件返回空内容。"""
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="empty.txt")

        assert result.success
        assert result.data["content"] == ""
        assert result.data["total_lines"] == 0
        assert result.data["lines_returned"] == 0
        assert result.data["end_line_truncated"] is False

    @pytest.mark.asyncio
    async def test_whitespace_only_file(self, tmp_path, mock_config):
        """仅含空行的文件（splitlines 返回空列表的特殊情况）。"""
        f = tmp_path / "blank.txt"
        f.write_text("\n\n\n", encoding="utf-8")

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="blank.txt")

        assert result.success
        assert "content" in result.data


class TestReadFileEncodingFallback:
    """编码回退链验证。"""

    @pytest.mark.asyncio
    async def test_gbk_encoding_fallback(self, tmp_path, mock_config):
        """GBK 编码文件通过 fallback 链正确解码。"""
        f = tmp_path / "gbk_file.txt"
        # GBK 编码的中文文本
        gbk_text = "你好世界，这是 GBK 编码的测试文件。\n第二行内容。\n"
        f.write_bytes(gbk_text.encode("gbk"))

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="gbk_file.txt")

        assert result.success
        assert "你好世界" in result.data["content"]
        # encoding 字段应被设置（非 utf-8）
        assert "encoding" in result.data

    @pytest.mark.asyncio
    async def test_latin1_fallback_always_succeeds(self, tmp_path, mock_config):
        """Latin-1 兜底编码始终成功（不会抛出 UnicodeDecodeError）。"""
        f = tmp_path / "random.bin"
        # 随机字节，无 BOM 且 null 占比低于阈值
        # 只用高字节（0x80-0xFF），不会有 null 误判
        # 用高字节（0x80-0xFF），避免 null 字节误判
        random_bytes = bytes(list(range(0x80, 0x100)) * 2)  # 256 字节，重复两次
        f.write_bytes(random_bytes)

        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="random.bin")

        # latin-1 兜底，应成功读取
        assert result.success
        assert "encoding" in result.data


class TestReadFilePathTraversal:
    """路径穿越防护。"""

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self, tmp_path, mock_config):
        """绝对路径被拒绝。"""
        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        # Windows: Path("C:/Windows").is_absolute() → True
        # Unix:    Path("/etc/passwd").is_absolute() → True
        result = await tool.execute(path="C:/Windows")

        assert not result.success
        assert "绝对路径" in result.error

    @pytest.mark.asyncio
    async def test_parent_directory_traversal(self, tmp_path, mock_config):
        """../ 穿越被拦截。"""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        # 在项目根外创建文件
        outside_file = tmp_path.parent / "secret.txt"
        outside_file.write_text("secret", encoding="utf-8")

        try:
            tool = ReadFileTool(project_root=subdir, config=mock_config)
            result = await tool.execute(path="../secret.txt")
            assert not result.success
            assert "超出项目根目录" in result.error
        finally:
            if outside_file.exists():
                outside_file.unlink()

    @pytest.mark.asyncio
    async def test_mid_path_traversal(self, tmp_path, mock_config):
        """路径中间包含 ../ 穿越（如 foo/../../secret.txt）被拦截。"""
        subdir = tmp_path / "safe"
        subdir.mkdir()
        # 创建合法子目录使前半段路径存在
        (subdir / "child").mkdir()
        outside_file = tmp_path.parent / "stolen.txt"
        outside_file.write_text("stolen", encoding="utf-8")

        try:
            tool = ReadFileTool(project_root=subdir, config=mock_config)
            result = await tool.execute(path="child/../../stolen.txt")
            assert not result.success
            assert "超出项目根目录" in result.error
        finally:
            if outside_file.exists():
                outside_file.unlink()

    @pytest.mark.asyncio
    async def test_consecutive_dotdot_at_end(self, tmp_path, mock_config):
        """路径以 .. 结尾（如 foo/..）被 resolve() 拦截。"""
        subdir = tmp_path / "safe"
        subdir.mkdir()
        (subdir / "child").mkdir()

        tool = ReadFileTool(project_root=subdir, config=mock_config)
        # foo/.. resolve 后等同于 . 仍在范围内 → 不应报错
        # foo/../.. 尝试逃逸
        result = await tool.execute(path="child/../..")

        assert not result.success
        assert "超出项目根目录" in result.error

    @pytest.mark.asyncio
    async def test_resolve_path_within_root_raises(self, tmp_path):
        """_resolve_path_within_root 对穿越路径抛出 _PathValidationError。"""
        subdir = tmp_path / "safe"
        subdir.mkdir()

        with pytest.raises(_PathValidationError, match="超出项目根目录"):
            _resolve_path_within_root("../outside", subdir, item_type="文件路径")


class TestReadFileToolInit:
    """初始化校验。"""

    def test_init_with_valid_directory(self, tmp_path, mock_config):
        """有效的目录初始化成功。"""
        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        assert tool.project_root == tmp_path.resolve()

    def test_init_with_file_raises(self, tmp_path, mock_config):
        """project_root 是文件而非目录时抛出 ValueError。"""
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello", encoding="utf-8")

        with pytest.raises(ValueError, match="project_root 必须是目录"):
            ReadFileTool(project_root=f, config=mock_config)

    def test_project_root_property_returns_copy(self, tmp_path, mock_config):
        """project_root 属性返回新实例，修改不影响内部状态。"""
        tool = ReadFileTool(project_root=tmp_path, config=mock_config)
        root1 = tool.project_root
        root2 = tool.project_root
        assert root1 == root2
        assert root1 is not root2  # 不同对象


# ============================================================================
# ListDirectoryTool 测试
# ============================================================================


class TestListDirectoryFlat:
    """非递归目录列表。"""

    @pytest.mark.asyncio
    async def test_list_flat_returns_files_and_dirs(self, tmp_path, mock_config):
        """非递归模式返回直接子项，目录在前。"""
        (tmp_path / "dir_a").mkdir()
        (tmp_path / "dir_b").mkdir()
        _make_text_file(tmp_path, "f1.py", "content")
        _make_text_file(tmp_path, "f2.md", "docs")

        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path=".")

        assert result.success
        assert result.data["is_recursive"] is False
        assert result.data["total_items"] == 4
        # 目录在前，文件在后
        items = result.data["items"]
        assert items[0]["type"] == "directory"
        assert items[1]["type"] == "directory"
        assert items[2]["type"] == "file"
        assert items[3]["type"] == "file"
        names = [i["name"] for i in items]
        assert names == sorted(names, key=str.lower)
        # 目录间有序，文件间有序
        dir_names = [i["name"] for i in items if i["type"] == "directory"]
        file_names = [i["name"] for i in items if i["type"] == "file"]
        assert dir_names == sorted(dir_names, key=str.lower)
        assert file_names == sorted(file_names, key=str.lower)

    @pytest.mark.asyncio
    async def test_list_default_path_is_dot(self, tmp_path, mock_config):
        """默认 path='.' 列出项目根目录内容。"""
        _make_text_file(tmp_path, "root_file.txt", "hi")

        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute()

        assert result.success
        assert any(i["name"] == "root_file.txt" for i in result.data["items"])


class TestListDirectoryRecursive:
    """递归模式。"""

    @pytest.mark.asyncio
    async def test_recursive_with_depth_control(self, tmp_path, mock_config):
        """max_depth 正确限制递归深度。"""
        # 创建嵌套结构: level1/level2/deep.txt
        level2 = tmp_path / "level1" / "level2"
        level2.mkdir(parents=True)
        _make_text_file(level2, "deep.txt", "deep")

        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)

        # depth=1: 只看得到 level1/
        r1 = await tool.execute(path=".", recursive=True, max_depth=1)
        items1 = r1.data["items"]
        assert any(i["name"] == "level1" and i["type"] == "directory" for i in items1)
        # 不应该看到 level2 下的文件
        assert not any(i["name"] == "deep.txt" for i in items1)

        # depth=3: 看得到 deep.txt
        r3 = await tool.execute(path=".", recursive=True, max_depth=3)
        items3 = r3.data["items"]
        assert any(i["name"] == "deep.txt" for i in items3)

    @pytest.mark.asyncio
    async def test_max_depth_less_than_one_rejected(self, tmp_path, mock_config):
        """max_depth < 1 返回 fail。"""
        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path=".", recursive=True, max_depth=0)

        assert not result.success
        assert "max_depth" in result.error

    @pytest.mark.asyncio
    async def test_non_recursive_max_depth_is_zero(self, tmp_path, mock_config):
        """非递归模式 data.max_depth 为 0。"""
        _make_text_file(tmp_path, "f.txt", "x")

        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path=".")

        assert result.success
        assert result.data["max_depth"] == 0


class TestListDirectoryPatternFilter:
    """通配符过滤。"""

    @pytest.mark.asyncio
    async def test_pattern_filter_fnmatch(self, tmp_path, mock_config):
        """pattern 参数使用 fnmatch 过滤文件名（不区分大小写）。"""
        _make_text_file(tmp_path, "test_a.py", "a")
        _make_text_file(tmp_path, "test_b.py", "b")
        _make_text_file(tmp_path, "other.md", "md")

        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path=".", pattern="*.py")

        assert result.success
        names = [i["name"] for i in result.data["items"]]
        assert "test_a.py" in names
        assert "test_b.py" in names
        assert "other.md" not in names

    @pytest.mark.asyncio
    async def test_pattern_case_insensitive(self, tmp_path, mock_config):
        """pattern 过滤不区分大小写。"""
        _make_text_file(tmp_path, "README.md", "readme")

        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path=".", pattern="*.MD")

        assert result.success
        names = [i["name"] for i in result.data["items"]]
        assert "README.md" in names


class TestListDirectoryExcludeDirs:
    """排除目录。"""

    @pytest.mark.asyncio
    async def test_excluded_dirs_skipped(self, tmp_path, mock_config):
        """config.project.exclude_dirs 中的目录被跳过。"""
        (tmp_path / "node_modules").mkdir()
        _make_text_file(tmp_path / "node_modules", "package.json", "{}")
        (tmp_path / "src").mkdir()
        _make_text_file(tmp_path / "src", "main.py", "code")

        mock_config.project.exclude_dirs = ["node_modules", "__pycache__"]
        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path=".", recursive=True)

        assert result.success
        names = [i["name"] for i in result.data["items"]]
        assert "src" in names
        assert "main.py" in names
        assert "node_modules" not in names
        assert "package.json" not in names


class TestListDirectoryEmpty:
    """空目录。"""

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path, mock_config):
        """空目录返回 items=[]、total_items=0。"""
        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path=".")

        assert result.success
        assert result.data["total_items"] == 0
        assert result.data["items"] == []


class TestListDirectoryNotFound:
    """目录不存在。"""

    @pytest.mark.asyncio
    async def test_directory_not_found(self, tmp_path, mock_config):
        """不存在的目录返回 fail。"""
        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="nonexistent_dir")

        assert not result.success
        assert "目录不存在" in result.error

    @pytest.mark.asyncio
    async def test_path_is_file_not_directory(self, tmp_path, mock_config):
        """路径指向文件时返回 fail。"""
        _make_text_file(tmp_path, "just_a_file.txt", "text")

        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="just_a_file.txt")

        assert not result.success
        assert "不是目录" in result.error


class TestListDirectoryTruncation:
    """_MAX_DIR_ITEMS 上限截断验证。"""

    @pytest.mark.asyncio
    async def test_truncation_when_exceeds_max_items(self, tmp_path, mock_config, mocker):
        """条目数超过 _MAX_DIR_ITEMS 时触发截断标记。"""
        # Mock _MAX_DIR_ITEMS 为小值以触发截断
        mocker.patch("miaowa.tools.filesystem._MAX_DIR_ITEMS", 3)

        for i in range(10):
            _make_text_file(tmp_path, f"file_{i:02d}.txt", "content")

        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path=".")

        assert result.success
        assert result.data["truncated"] is True
        assert result.data["total_items"] == 3


class TestListDirectoryPathTraversal:
    """ListDirectory 的路径穿越防护。"""

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self, tmp_path, mock_config):
        """绝对路径被拒绝。"""
        tool = ListDirectoryTool(project_root=tmp_path, config=mock_config)
        result = await tool.execute(path="C:/Windows")

        assert not result.success
        assert "绝对路径" in result.error

    @pytest.mark.asyncio
    async def test_parent_directory_traversal_rejected(self, tmp_path, mock_config):
        """../ 穿越被拦截。"""
        subdir = tmp_path / "safe_zone"
        subdir.mkdir()

        tool = ListDirectoryTool(project_root=subdir, config=mock_config)
        result = await tool.execute(path="..")

        assert not result.success
        assert "超出项目根目录" in result.error
