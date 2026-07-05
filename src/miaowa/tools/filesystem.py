"""文件系统工具 — 文件读取、目录列表等操作工具。

PRD §6.2.1: read_file 工具实现。
PRD §6.2.2: list_directory 工具实现。
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles

from miaowa.core.config import Config
from miaowa.core.types import ToolParameter, ToolResult
from miaowa.tools.base import BaseTool

if TYPE_CHECKING:
    from miaowa.tools.gitignore_filter import GitignoreFilter

# ============================================================================
# 常量
# ============================================================================

# 二进制文件检测：null 字节占比阈值（PRD §6.2.1）
_BINARY_NULL_RATIO_THRESHOLD = 0.1

# 二进制检测时读取的文件头部字节数
_BINARY_DETECT_HEADER_SIZE = 1024

# BOM (Byte Order Mark) 前缀 → 对应的编码名
# 在 null 字节检测之前先检查 BOM，避免 UTF-16 等合法文本文件被误判为二进制。
_BOMS: dict[bytes, str] = {
    b"\xfe\xff": "utf-16-be",
    b"\xff\xfe": "utf-16-le",
    b"\xef\xbb\xbf": "utf-8-sig",
}

# 文件编码 fallback 链 — 按优先级依次尝试解码
_FALLBACK_ENCODINGS: tuple[str, ...] = (
    "utf-8",
    "gbk",
    "gb2312",
    "latin-1",  # 兜底编码，永远不会抛出 UnicodeDecodeError
)

# 目录列表最大返回条目数（防止 LLM 上下文溢出）
_MAX_DIR_ITEMS = 500


# ============================================================================
# 共享工具函数 — 路径解析与安全检查
# ============================================================================


def _resolve_path_within_root(
    path_str: str,
    project_root: Path,
    *,
    item_type: str = "路径",
) -> Path:
    """解析相对路径并验证其在 project_root 内。

    ReadFileTool 和 ListDirectoryTool 共用此函数，
    确保路径穿越防护逻辑在单一位置维护。

    Args:
        path_str: 用户传入的相对路径字符串。
        project_root: 项目根目录（已 resolve 的绝对路径）。
        item_type: 用于错误消息中的描述，如 "文件路径"、"目录路径"。

    Returns:
        解析并规范化后的绝对 Path。

    Raises:
        _PathValidationError: 路径为绝对路径或解析后超出 project_root 范围。
            调用方应将此异常转换为 ToolResult.fail()。
    """
    raw_path = Path(path_str)
    if raw_path.is_absolute():
        raise _PathValidationError(
            f"路径参数必须为相对路径（相对于项目根目录），"
            f"收到绝对路径: {path_str}"
        )
    full_path = (project_root / raw_path).resolve()

    try:
        full_path.relative_to(project_root)
    except ValueError:
        raise _PathValidationError(
            f"安全限制：{item_type}超出项目根目录范围 — "
            f"{path_str} (解析路径: {full_path})"
        )

    return full_path


class _PathValidationError(ValueError):
    """路径校验失败（内部使用，由调用方转换为 ToolResult.fail()）。"""


# ============================================================================
# ReadFileTool
# ============================================================================


class ReadFileTool(BaseTool):
    """读取项目文件内容的工具。

    支持文本文件读取，自动检测并拒绝二进制文件。
    可通过行号范围参数读取文件的部分内容。

    安全机制：
        - 路径穿越防护：resolve() + relative_to() 双重校验
        - 符号链接：resolve() 跟随符号链接后仍受前缀约束
        - 单次读取：raw bytes 只读取一次，消除 TOCTOU 窗口
        - BOM 检测：先于 null 字节检测，避免 UTF-16 误判

    Attributes:
        name: 工具名称 "read_file"。
        description: 工具功能描述（供 LLM 理解）。
        parameters: 工具参数定义列表。
    """

    name = "read_file"
    description = (
        "读取指定路径的文件内容。"
        "支持文本文件，自动跳过二进制文件。"
        "可以指定行号范围读取部分内容。"
    )
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="相对于项目根目录的文件路径",
            required=True,
        ),
        ToolParameter(
            name="start_line",
            type="integer",
            description="起始行号（从1开始），默认 1",
            required=False,
            default=1,
        ),
        ToolParameter(
            name="end_line",
            type="integer",
            description="结束行号（包含），不填则读到文件末尾",
            required=False,
        ),
    ]

    def __init__(
        self,
        project_root: Path,
        config: Config,
        *,
        gitignore_filter: GitignoreFilter | None = None,
    ) -> None:
        """初始化 ReadFileTool。

        Args:
            project_root: 项目根目录的绝对路径。所有文件路径均以此为基准解析。
            config: Miaowa 应用配置对象，用于读取文件大小上限等设置。
            gitignore_filter: 可选的 .gitignore 过滤器。传入后 ReadFileTool
                在校验文件路径时会检查目标文件是否被忽略。

        Raises:
            ValueError: project_root 不是目录时抛出。
        """
        resolved = project_root.resolve()
        if not resolved.is_dir():
            raise ValueError(
                f"project_root 必须是目录，实际路径: {resolved}"
            )
        self._project_root = resolved
        self._config = config
        self._gitignore_filter = gitignore_filter

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        """返回项目根目录路径的副本。

        返回新 Path 实例而非内部引用，对返回值本身的修改
        不会影响 ReadFileTool 的内部状态。
        """
        return Path(self._project_root)

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行文件读取。

        执行流程（PRD §6.2.1，含安全加固）:
            1. 解析路径：拒绝绝对路径，基于 project_root 解析。
            2. 安全检查：resolve() + relative_to() 防止目录穿越。
            3. .gitignore 过滤：检查文件是否被 .gitignore 规则排除。
            4. 存在检查：文件不存在或不是普通文件时返回错误。
            5. 大小检查：文件超过 config.project.max_file_size 时返回错误。
            6. 异步读取：使用 aiofiles 一次性读取 raw bytes。
            7. 二进制检测：BOM 检测 → null 字节占比检测。
            8. 编码解码：按 fallback 链依次尝试解码。
            9. 行号处理：slice、范围校验、read_file_max_lines 上限。
            10. 返回 ToolResult。

        Args:
            **kwargs: 经过 ToolValidator 校验后的参数字典。
                - path (str): 相对于项目根目录的文件路径。
                - start_line (int): 起始行号，默认 1。
                - end_line (int | None): 结束行号，None 表示到文件末尾。

        Returns:
            ToolResult:
                - success=True: data 为 dict，包含:
                    - path (str): 原始请求路径
                    - content (str): 截取后的文件内容
                    - total_lines (int): 文件总行数
                    - lines_returned (int): 实际返回的行数
                    - end_line_truncated (bool): end_line 是否被静默截断
                - success=False: error 为具体错误描述（含解析后路径）。
        """
        path_str: str = kwargs["path"]
        start_line: int = kwargs.get("start_line", 1)
        end_line: int | None = kwargs.get("end_line", None)

        # ------------------------------------------------------------------
        # 1. 解析路径 & 安全检查（复用共享函数）
        # ------------------------------------------------------------------
        try:
            full_path = _resolve_path_within_root(
                path_str, self._project_root, item_type="文件路径"
            )
        except _PathValidationError as exc:
            return ToolResult.fail(str(exc))

        # ------------------------------------------------------------------
        # 3. .gitignore 过滤
        # ------------------------------------------------------------------
        if self._gitignore_filter is not None:
            if self._gitignore_filter.is_ignored(full_path):
                return ToolResult.fail(
                    f"文件被 .gitignore 规则排除: {path_str} "
                    f"(解析路径: {full_path})"
                )

        # ------------------------------------------------------------------
        # 4. 存在检查
        # ------------------------------------------------------------------
        if not full_path.exists():
            return ToolResult.fail(
                f"文件不存在: {path_str} (解析路径: {full_path})"
            )
        if not full_path.is_file():
            return ToolResult.fail(
                f"路径不是普通文件: {path_str} (解析路径: {full_path})"
            )

        # ------------------------------------------------------------------
        # 4. 大小检查 — 在读取之前拦截超大文件
        #
        # Path.stat() 是轻量级 OS 元数据查询（不读取文件内容），
        # 在此处以同步方式调用不会显著阻塞事件循环。
        # ------------------------------------------------------------------
        file_size = full_path.stat().st_size
        max_size = self._config.project.max_file_size
        if file_size > max_size:
            return ToolResult.fail(
                f"文件过大: {path_str} ({file_size:,} bytes)，"
                f"超过允许上限 ({max_size:,} bytes) "
                f"(解析路径: {full_path})"
            )

        # ------------------------------------------------------------------
        # 5. 异步读取 raw bytes — 整个文件只读取这一次
        #
        # 单次读取消除 TOCTOU 窗口：二进制检测、编码解码均基于同一份
        # 字节数据，磁盘上的文件在读取期间被替换不影响一致性。
        # ------------------------------------------------------------------
        try:
            async with aiofiles.open(full_path, mode="rb") as f:
                raw_bytes: bytes = await f.read()
        except OSError as exc:
            return ToolResult.fail(
                f"无法读取文件（权限不足或 I/O 错误）: "
                f"{path_str} (解析路径: {full_path}) — {exc}"
            )

        # 空文件（0 字节）直接返回空内容
        if len(raw_bytes) == 0:
            return ToolResult.ok({
                "path": path_str,
                "content": "",
                "total_lines": 0,
                "lines_returned": 0,
                "end_line_truncated": False,
            })

        # ------------------------------------------------------------------
        # 6. 二进制检测
        #
        # 检测顺序：
        #   a) BOM 前缀匹配 → 记录对应编码，跳过 null 字节检测
        #   b) 无 BOM + null 字节占比 > 阈值 → 视为二进制
        #
        # BOM 检测必须在 null 字节检测之前，否则 UTF-16LE 文件
        # （每两个字节含一个 0x00）会被误判为二进制。
        # ------------------------------------------------------------------
        header = raw_bytes[:_BINARY_DETECT_HEADER_SIZE]

        # 检查 BOM 前缀，确定文件声明的编码
        bom_encoding: str | None = None
        for bom_prefix, enc in _BOMS.items():
            if header.startswith(bom_prefix):
                bom_encoding = enc
                break

        if bom_encoding is None:
            # 无 BOM — 执行 null 字节检测
            null_count = header.count(b"\x00")
            if len(header) > 0 and null_count / len(header) > _BINARY_NULL_RATIO_THRESHOLD:
                return ToolResult.fail(
                    f"文件似乎是二进制格式（null 字节占比 "
                    f"{null_count / len(header):.0%}），"
                    f"无法以文本方式读取: "
                    f"{path_str} (解析路径: {full_path})"
                )

        # ------------------------------------------------------------------
        # 7. 编码解码 — BOM 编码优先，随后按 fallback 链尝试
        #
        # 顺序：BOM 指示编码 → UTF-8 → GBK → GB2312 → Latin-1（兜底）
        # Latin-1 永远不会抛出 UnicodeDecodeError（256 个码点全覆盖），
        # 但解码结果可能是语义上的乱码。
        # ------------------------------------------------------------------
        text: str | None = None
        used_encoding: str | None = None

        # 构建编码尝试序列（BOM 编码优先）
        encodings_to_try: list[str] = []
        if bom_encoding is not None:
            encodings_to_try.append(bom_encoding)
        encodings_to_try.extend(_FALLBACK_ENCODINGS)

        for encoding in encodings_to_try:
            try:
                text = raw_bytes.decode(encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            # 理论上不可达（latin-1 总是成功），但保留防御性检查
            return ToolResult.fail(
                f"无法以任何已知编码读取文件: "
                f"{path_str} (解析路径: {full_path})"
            )

        # ------------------------------------------------------------------
        # 8. 行号处理
        # ------------------------------------------------------------------
        # 使用 splitlines() 替代 split("\n")，统一处理 \n、\r\n、\r
        lines: list[str] = text.splitlines()
        total_lines = len(lines)

        # 空内容（仅含空白字符等 splitlines 返回空列表的情况）
        if total_lines == 0:
            return ToolResult.ok({
                "path": path_str,
                "content": "",
                "total_lines": 0,
                "lines_returned": 0,
                "end_line_truncated": False,
            })

        # -- 校验 start_line -------------------------------------------------
        if start_line < 1:
            return ToolResult.fail(
                f"start_line 必须 >= 1，实际值: {start_line}"
            )
        if start_line > total_lines:
            return ToolResult.fail(
                f"起始行号 {start_line} 超出文件总行数 {total_lines}。"
                f"请将 start_line 设为 1 到 {total_lines} 之间的值: "
                f"{path_str} (解析路径: {full_path})"
            )

        # -- 校验 end_line ---------------------------------------------------
        if end_line is not None and end_line < start_line:
            return ToolResult.fail(
                f"end_line ({end_line}) 不能小于 start_line ({start_line})"
            )

        # -- 确定有效 end_line ------------------------------------------------
        end_line_raw = end_line  # 保存原始值用于检测截断
        if end_line is not None:
            end_line_truncated = end_line > total_lines
            end_line = min(end_line, total_lines)
        else:
            end_line = total_lines
            end_line_truncated = False

        # -- 应用 read_file_max_lines 上限 -----------------------------------
        max_lines = self._config.tools.read_file_max_lines
        if end_line - start_line + 1 > max_lines:
            end_line = start_line + max_lines - 1
            end_line_truncated = True

        # -- 切片（行号为 1-indexed → 0-indexed）----------------------------
        selected_lines = lines[start_line - 1 : end_line]
        sliced_content = "\n".join(selected_lines)
        lines_returned = len(selected_lines)

        # ------------------------------------------------------------------
        # 9. 返回结果
        # ------------------------------------------------------------------
        data: dict[str, Any] = {
            "path": path_str,
            "content": sliced_content,
            "total_lines": total_lines,
            "lines_returned": lines_returned,
            "end_line_truncated": end_line_truncated,
        }
        if used_encoding and used_encoding != "utf-8":
            data["encoding"] = used_encoding

        return ToolResult.ok(data)


# ============================================================================
# ListDirectoryTool
# ============================================================================


class ListDirectoryTool(BaseTool):
    """列出目录内容的工具。

    支持递归列出子目录、深度控制、通配符文件名过滤。
    自动排除配置中指定的目录（如 .git、node_modules 等）。

    Attributes:
        name: 工具名称 "list_directory"。
        description: 工具功能描述（供 LLM 理解）。
        parameters: 工具参数定义列表。
    """

    name = "list_directory"
    description = (
        "列出指定目录的文件和子目录。"
        "支持递归列出和深度控制。"
        "可以用通配符过滤文件名。"
    )
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="相对于项目根目录的目录路径，默认 '.' 表示项目根目录",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="recursive",
            type="boolean",
            description="是否递归列出子目录内容，默认 false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="max_depth",
            type="integer",
            description="递归最大深度（仅在 recursive=true 时生效），默认 3。"
            "depth=1 表示仅列出目标目录的直接子项",
            required=False,
            default=3,
        ),
        ToolParameter(
            name="pattern",
            type="string",
            description="文件名过滤通配符（如 \"*.py\"、\"test_*.py\"），"
            "使用 fnmatch 匹配规则。不填则返回所有条目",
            required=False,
        ),
    ]

    def __init__(
        self,
        project_root: Path,
        config: Config,
        *,
        gitignore_filter: GitignoreFilter | None = None,
    ) -> None:
        """初始化 ListDirectoryTool。

        Args:
            project_root: 项目根目录的绝对路径。所有路径均以此为基准解析。
            config: Miaowa 应用配置对象，用于获取排除目录列表。
            gitignore_filter: 可选的 .gitignore 过滤器。传入后在遍历目录时
                自动跳过被 .gitignore 规则忽略的文件和目录。

        Raises:
            ValueError: project_root 不是目录时抛出。
        """
        resolved = project_root.resolve()
        if not resolved.is_dir():
            raise ValueError(
                f"project_root 必须是目录，实际路径: {resolved}"
            )
        self._project_root = resolved
        self._config = config
        self._gitignore_filter = gitignore_filter
        # 一次性转换为 set，避免每次 execute() 重复转换
        self._exclude_dirs: set[str] = set(config.project.exclude_dirs)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        """返回项目根目录路径的副本。

        返回新 Path 实例而非内部引用，对返回值本身的修改
        不会影响 ListDirectoryTool 的内部状态。
        """
        return Path(self._project_root)

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行目录列表。

        执行流程（PRD §6.2.2）:
            1. 输入校验（max_depth）。
            2. 解析路径、安全检查（复用共享函数，防止目录穿越）。
            3. 目录存在检查（PermissionError 报错，不静默）。
            4. 遍历目录（使用 os.scandir，排除 exclude_dirs）。
            5. 应用 pattern 过滤（fnmatch，统一不区分大小写）。
            6. 截断检查（条目数超过上限时截断并标记）。
            7. 排序：目录在前，文件在后，各自按字母序。
            8. 返回格式化结果。

        Args:
            **kwargs: 经过 ToolValidator 校验后的参数字典。
                - path (str): 目录路径，默认 "."。
                - recursive (bool): 是否递归，默认 False。
                - max_depth (int): 递归深度，默认 3。
                - pattern (str | None): 通配符过滤，None 表示不过滤。

        Returns:
            ToolResult:
                - success=True: data 为 dict，包含:
                    - path (str): 原始请求路径
                    - is_recursive (bool): 是否递归模式
                    - max_depth (int): 递归深度（非递归模式下为 0）
                    - total_items (int): 返回的条目数
                    - truncated (bool): 是否因条目数上限被截断
                    - items (list[dict]): 条目列表，每项含:
                        - name (str): 条目名称
                        - type (str): "file" 或 "directory"
                        - path (str): 相对于项目根目录的路径
                        - size (int): 文件大小（字节），目录为 0
                - success=False: error 为具体错误描述。
        """
        path_str: str = kwargs.get("path", ".")
        recursive: bool = kwargs.get("recursive", False)
        max_depth: int = kwargs.get("max_depth", 3)
        pattern: str | None = kwargs.get("pattern", None)

        # ------------------------------------------------------------------
        # 1. 输入校验
        # ------------------------------------------------------------------
        if max_depth < 1:
            return ToolResult.fail(
                f"max_depth 必须 >= 1，实际值: {max_depth}"
            )

        # ------------------------------------------------------------------
        # 2. 解析路径 & 安全检查（复用共享函数）
        # ------------------------------------------------------------------
        try:
            full_path = _resolve_path_within_root(
                path_str, self._project_root, item_type="目录路径"
            )
        except _PathValidationError as exc:
            return ToolResult.fail(str(exc))

        # ------------------------------------------------------------------
        # 3. 目录存在检查
        # ------------------------------------------------------------------
        if not full_path.exists():
            return ToolResult.fail(
                f"目录不存在: {path_str} (解析路径: {full_path})"
            )
        if not full_path.is_dir():
            return ToolResult.fail(
                f"路径不是目录: {path_str} (解析路径: {full_path})"
            )

        # ------------------------------------------------------------------
        # 4. 遍历目录
        # ------------------------------------------------------------------
        try:
            if recursive:
                items = self._walk_recursive(
                    full_path, self._project_root,
                    max_depth, self._exclude_dirs,
                )
            else:
                items = self._list_flat(
                    full_path, self._project_root, self._exclude_dirs,
                )
        except PermissionError:
            return ToolResult.fail(
                f"无法访问目录（权限不足）: "
                f"{path_str} (解析路径: {full_path})"
            )

        # ------------------------------------------------------------------
        # 5. 应用 pattern 过滤（fnmatch，统一不区分大小写）
        # ------------------------------------------------------------------
        if pattern is not None:
            pattern_lower = pattern.lower()
            items = [
                item for item in items
                if fnmatch.fnmatch(item["name"].lower(), pattern_lower)
            ]

        # ------------------------------------------------------------------
        # 6. 截断检查 — 条目数超过上限时截断并标记
        # ------------------------------------------------------------------
        truncated = len(items) > _MAX_DIR_ITEMS
        if truncated:
            items = items[:_MAX_DIR_ITEMS]

        # ------------------------------------------------------------------
        # 7. 排序：目录在前、文件在后，各自按字母序（不区分大小写）
        # ------------------------------------------------------------------
        dirs = [item for item in items if item["type"] == "directory"]
        files = [item for item in items if item["type"] == "file"]
        dirs.sort(key=lambda x: x["name"].lower())
        files.sort(key=lambda x: x["name"].lower())
        items = dirs + files

        # ------------------------------------------------------------------
        # 8. 返回结果
        # ------------------------------------------------------------------
        return ToolResult.ok({
            "path": path_str,
            "is_recursive": recursive,
            "max_depth": max_depth if recursive else 0,
            "total_items": len(items),
            "truncated": truncated,
            "items": items,
        })

    # ------------------------------------------------------------------
    # 内部遍历方法
    # ------------------------------------------------------------------

    def _list_flat(
        self,
        dir_path: Path,
        base_path: Path,
        exclude_dirs: set[str],
    ) -> list[dict[str, Any]]:
        """单层目录列表（非递归模式）。

        使用 os.scandir() 而非 Path.iterdir()，利用 DirEntry 缓存
        的 stat 信息避免为每个条目触发额外的系统调用。

        Args:
            dir_path: 要列出的目录绝对路径。
            base_path: 项目根目录（用于计算相对路径）。
            exclude_dirs: 需要排除的目录名集合。

        Returns:
            条目字典列表（未排序）。

        Raises:
            PermissionError: 目录无法访问时抛出（由 execute() 处理）。
        """
        items: list[dict[str, Any]] = []

        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_dir():
                    if entry.name in exclude_dirs:
                        continue
                    # .gitignore 过滤
                    if self._gitignore_filter is not None:
                        if self._gitignore_filter.is_ignored(Path(entry.path)):
                            continue
                    rel_path = Path(entry.path).relative_to(base_path).as_posix()
                    items.append({
                        "name": entry.name,
                        "type": "directory",
                        "path": rel_path,
                        "size": 0,
                    })
                else:
                    # .gitignore 过滤
                    if self._gitignore_filter is not None:
                        if self._gitignore_filter.is_ignored(Path(entry.path)):
                            continue
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    rel_path = Path(entry.path).relative_to(base_path).as_posix()
                    items.append({
                        "name": entry.name,
                        "type": "file",
                        "path": rel_path,
                        "size": size,
                    })
        return items

    def _walk_recursive(
        self,
        dir_path: Path,
        base_path: Path,
        max_depth: int,
        exclude_dirs: set[str],
    ) -> list[dict[str, Any]]:
        """递归目录遍历（带深度控制、条目数上限）。

        深度计数从 1 开始（直接子项为 depth=1）。

        Args:
            dir_path: 当前要遍历的目录绝对路径。
            base_path: 项目根目录（用于计算相对路径）。
            max_depth: 最大递归深度。
            exclude_dirs: 需要排除的目录名集合。

        Returns:
            条目字典列表（未排序），顺序为深度优先、目录优先。

        Raises:
            PermissionError: 顶层目录无法访问时抛出（由 execute() 处理）。
        """
        items: list[dict[str, Any]] = []
        # max_items+1：多收一条用于检测是否触发截断
        self._walk_dir(
            dir_path, base_path, depth=1, max_depth=max_depth,
            exclude_dirs=exclude_dirs,
            gitignore_filter=self._gitignore_filter,
            results=items,
            max_items=_MAX_DIR_ITEMS + 1,
        )
        return items

    @staticmethod
    def _walk_dir(
        dir_path: Path,
        base_path: Path,
        depth: int,
        max_depth: int,
        exclude_dirs: set[str],
        results: list[dict[str, Any]],
        max_items: int,
        *,
        gitignore_filter: GitignoreFilter | None = None,
    ) -> None:
        """递归遍历的递归辅助方法。

        采用深度优先遍历（DFS），使用 os.scandir() 减少系统调用。
        目录项总是在其子项之前被记录。

        Args:
            dir_path: 当前目录路径。
            base_path: 项目根目录。
            depth: 当前深度（1-indexed）。
            max_depth: 最大深度。
            exclude_dirs: 排除目录名集合。
            results: 结果列表（原地追加）。
            max_items: 最大收集条目数（含余量），达到后停止收集但继续
                （仅统计，不存储）以确保截断标记准确。
            gitignore_filter: 可选的 .gitignore 过滤器。
        """
        if depth > max_depth:
            return
        if len(results) >= max_items:
            return  # 已收集足够条目，停止遍历

        try:
            with os.scandir(dir_path) as entries:
                for entry in entries:
                    if len(results) >= max_items:
                        return  # 提前终止

                    if entry.is_dir():
                        if entry.name in exclude_dirs:
                            continue
                        # .gitignore 过滤
                        if gitignore_filter is not None:
                            if gitignore_filter.is_ignored(Path(entry.path)):
                                continue
                        rel_path = Path(entry.path).relative_to(base_path).as_posix()
                        results.append({
                            "name": entry.name,
                            "type": "directory",
                            "path": rel_path,
                            "size": 0,
                        })
                        # 递归进入子目录
                        ListDirectoryTool._walk_dir(
                            Path(entry.path), base_path,
                            depth=depth + 1,
                            max_depth=max_depth,
                            exclude_dirs=exclude_dirs,
                            results=results,
                            max_items=max_items,
                            gitignore_filter=gitignore_filter,
                        )
                    else:
                        # .gitignore 过滤
                        if gitignore_filter is not None:
                            if gitignore_filter.is_ignored(Path(entry.path)):
                                continue
                        try:
                            size = entry.stat().st_size
                        except OSError:
                            size = 0
                        rel_path = Path(entry.path).relative_to(base_path).as_posix()
                        results.append({
                            "name": entry.name,
                            "type": "file",
                            "path": rel_path,
                            "size": size,
                        })
        except PermissionError:
            # 子目录权限不足：跳过该目录，不中断整体遍历
            pass
