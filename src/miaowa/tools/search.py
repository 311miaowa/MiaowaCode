"""搜索工具 — 文件名搜索与文件内容搜索。

PRD §6.2.3: search_files 工具实现。

支持两种搜索模式：
- filename: 按文件名匹配（fnmatch 通配符）
- content:  文件内容搜索（正则表达式 / 关键词）
"""

from __future__ import annotations

import asyncio
import fnmatch
import re as _re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles

from miaowa.core.config import Config
from miaowa.core.logger import get_logger
from miaowa.core.types import ToolParameter, ToolResult
from miaowa.tools.base import BaseTool

if TYPE_CHECKING:
    from miaowa.tools.gitignore_filter import GitignoreFilter

logger = get_logger(__name__)

# ============================================================================
# 常量
# ============================================================================

# 二进制文件检测：null 字节占比阈值
_BINARY_NULL_RATIO_THRESHOLD = 0.1

# 二进制检测时读取的文件头部字节数
_BINARY_DETECT_HEADER_SIZE = 1024

# BOM (Byte Order Mark) 前缀 → 编码名（UTF-16 文件不应误判为二进制）
_BOMS: dict[bytes, str] = {
    b"\xfe\xff": "utf-16-be",
    b"\xff\xfe": "utf-16-le",
    b"\xef\xbb\xbf": "utf-8-sig",
}

# 内容搜索：单行最大匹配长度（字符数），超长行截断以防止 ReDoS
_MAX_LINE_LENGTH = 4096

# 搜索结果 content 字段最大长度（字符数），超出部分省略并加 "..."
_CONTENT_MAX_LENGTH = 200

# 单次搜索最大候选文件数（防止超大项目耗尽内存）
_MAX_CANDIDATE_FILES = 10_000


# ============================================================================
# 共享工具函数
# ============================================================================


class _PathValidationError(ValueError):
    """路径校验失败（内部使用，由调用方转换为 ToolResult.fail()）。"""


def _resolve_path_within_root(
    path_str: str,
    project_root: Path,
    *,
    item_type: str = "路径",
) -> Path:
    """解析相对路径并验证其在 project_root 内。

    与 filesystem.py 中的同名函数保持同步。
    后续应将此函数提升至公共模块（如 miaowa.tools._utils）。
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


def _expand_brace_pattern(pattern: str) -> list[str]:
    """将 {a,b} 大括号展开为多个 fnmatch 模式。

    fnmatch 不支持 glob 的大括号展开语法。
    此函数提取 {a,b,c} 并生成多个独立模式，实现等价功能。

    Args:
        pattern: 可能包含 {a,b} 的文件模式，如 "*.{py,js,ts}"。

    Returns:
        展开后的模式列表。无大括号时返回单元素列表。
        例如 "*.{py,js}" → ["*.py", "*.js"]。
    """
    m = _re.match(r'^(.*)\{([^}]+)\}(.*)$', pattern)
    if m is None:
        return [pattern]
    prefix, options_str, suffix = m.groups()
    options = [opt.strip() for opt in options_str.split(",") if opt.strip()]
    if not options:
        return [pattern]
    return [f"{prefix}{opt}{suffix}" for opt in options]


def _has_regex_metachars(query: str) -> bool:
    """检测 query 是否包含正则表达式元字符。

    若 query 不含任何元字符，内容搜索时自动对字面字符串进行
    re.escape()，避免 "."、"(" 等被误解为正则语法。
    """
    _REGEX_META = set(r".^$*+?{}[]\|()")
    return bool(set(query) & _REGEX_META)


# ============================================================================
# SearchFilesTool
# ============================================================================


class SearchFilesTool(BaseTool):
    """项目文件搜索工具。

    支持两种搜索模式：
        - filename: 按文件名通配符匹配（内部使用 fnmatch）。
        - content:  按文件内容搜索，支持正则表达式和关键词。

    自动跳过二进制文件、超大文件以及配置中排除的目录（如 .git、node_modules）。

    Attributes:
        name: 工具名称 "search_files"。
        description: 工具功能描述（供 LLM 理解）。
        parameters: 工具参数定义列表。
    """

    name = "search_files"
    description = (
        "在项目中搜索文件（按文件名匹配）或文件内容（按文本/正则表达式搜索）。"
        "支持通配符限制搜索的文件类型。"
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="搜索关键词或正则表达式。"
            "filename 模式下支持通配符（如 \"test_*.py\"），"
            "content 模式下支持 Python re 正则语法",
            required=True,
        ),
        ToolParameter(
            name="search_type",
            type="string",
            description="搜索类型：\"filename\" 按文件名匹配，"
            "\"content\" 按文件内容搜索",
            required=False,
            default="content",
            enum=["filename", "content"],
        ),
        ToolParameter(
            name="path",
            type="string",
            description="搜索起始路径（相对于项目根目录），默认 '.' 表示整个项目",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="file_pattern",
            type="string",
            description="限定搜索的文件类型（如 \"*.py\"、\"*.{js,ts}\"），"
            "使用 fnmatch 匹配（支持 {a,b} 大括号展开）。"
            "content 模式下仅搜索匹配的文件。",
            required=False,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="最大返回结果数，默认 30。达到上限后停止搜索",
            required=False,
            default=30,
        ),
        ToolParameter(
            name="case_sensitive",
            type="boolean",
            description="是否区分大小写，默认 false（不区分）",
            required=False,
            default=False,
        ),
    ]

    def __init__(
        self,
        project_root: Path,
        config: Config,
        *,
        gitignore_filter: GitignoreFilter | None = None,
    ) -> None:
        """初始化 SearchFilesTool。

        Args:
            project_root: 项目根目录的绝对路径。
            config: Miaowa 应用配置对象。
            gitignore_filter: 可选的 .gitignore 过滤器。传入后在搜索时
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
        # 缓存排除目录集合，避免每次 execute() 重复转换
        self._exclude_dirs: set[str] = set(config.project.exclude_dirs)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        """返回项目根目录路径的副本。"""
        return Path(self._project_root)

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行文件搜索。

        根据 search_type 分派到 _search_filename 或 _search_content。

        Args:
            **kwargs: 经过 ToolValidator 校验后的参数字典。
                - query (str): 搜索关键词/正则表达式。
                - search_type (str): "filename" 或 "content"。
                - path (str): 搜索起始路径。
                - file_pattern (str | None): 文件类型过滤。
                - max_results (int): 最大结果数。
                - case_sensitive (bool): 是否区分大小写。

        Returns:
            ToolResult:
                - success=True: data 包含 query, search_type, search_path,
                  total_results, max_results, truncated, results。
                - success=False: error 为错误描述。
        """
        query: str = kwargs["query"]
        search_type: str = kwargs.get("search_type", "content")
        path_str: str = kwargs.get("path", ".")
        file_pattern: str | None = kwargs.get("file_pattern", None)
        max_results: int = kwargs.get("max_results", 30)
        case_sensitive: bool = kwargs.get("case_sensitive", False)

        # ------------------------------------------------------------------
        # 1. 输入校验
        # ------------------------------------------------------------------
        if not query.strip():
            return ToolResult.fail("搜索关键词不能为空")

        if max_results < 1:
            return ToolResult.fail(
                f"max_results 必须 >= 1，实际值: {max_results}"
            )

        # ------------------------------------------------------------------
        # 2. 解析路径 & 安全检查
        # ------------------------------------------------------------------
        try:
            search_dir = _resolve_path_within_root(
                path_str, self._project_root, item_type="搜索路径"
            )
        except _PathValidationError as exc:
            return ToolResult.fail(str(exc))

        if not search_dir.exists():
            return ToolResult.fail(
                f"路径不存在: {path_str} (解析路径: {search_dir})"
            )
        if not search_dir.is_dir():
            return ToolResult.fail(
                f"搜索路径不是目录: {path_str} (解析路径: {search_dir})"
            )

        # ------------------------------------------------------------------
        # 3. 分派搜索
        # ------------------------------------------------------------------
        if search_type == "filename":
            return await self._search_filename(
                query, path_str, search_dir, file_pattern,
                max_results, case_sensitive,
            )
        else:
            return await self._search_content(
                query, path_str, search_dir, file_pattern,
                max_results, case_sensitive,
            )

    # ------------------------------------------------------------------
    # 文件名搜索
    # ------------------------------------------------------------------

    async def _search_filename(
        self,
        query: str,
        path_str: str,
        search_dir: Path,
        file_pattern: str | None,
        max_results: int,
        case_sensitive: bool,
    ) -> ToolResult:
        """按文件名搜索（异步）。

        使用 asyncio.to_thread 在后台线程中执行 rglob 遍历，
        避免同步文件系统操作阻塞事件循环。

        Returns:
            ToolResult，data.results 中每项含 path, name, size。
        """
        logger.debug(
            f"[search_files] 文件名搜索开始: "
            f"query={query!r}, dir={search_dir}, "
            f"pattern={file_pattern!r}, max={max_results}"
        )

        # 构建 fnmatch 模式
        match_pattern = f"*{query}*"
        if not case_sensitive:
            match_pattern = match_pattern.lower()

        # 展开大括号模式
        patterns = (
            _expand_brace_pattern(file_pattern) if file_pattern else None
        )

        # 在后台线程中收集候选文件（rglob 是同步生成器）
        def collect_candidates() -> list[Path]:
            candidates: list[Path] = []
            for entry in search_dir.rglob("*"):
                if len(candidates) >= _MAX_CANDIDATE_FILES:
                    break
                # 跳过排除目录
                if self._exclude_dirs.intersection(
                    entry.relative_to(search_dir).parts
                ):
                    continue
                # .gitignore 过滤
                if self._gitignore_filter is not None:
                    if self._gitignore_filter.is_ignored(entry):
                        continue
                if not entry.is_file():
                    continue
                # file_pattern 过滤
                if patterns is not None:
                    target = entry.name if case_sensitive else entry.name.lower()
                    if not any(
                        fnmatch.fnmatch(target, p if case_sensitive else p.lower())
                        for p in patterns
                    ):
                        continue
                candidates.append(entry)
            return candidates

        candidates = await asyncio.to_thread(collect_candidates)

        # 在主线程中匹配（文件名匹配是纯 CPU，不需要异步 I/O）
        results: list[dict[str, Any]] = []
        files_scanned = 0

        for entry in candidates:
            if len(results) >= max_results:
                break

            target_name = entry.name if case_sensitive else entry.name.lower()
            if not fnmatch.fnmatch(target_name, match_pattern):
                continue

            files_scanned += 1

            # 在后台线程中获取文件大小
            try:
                size = await asyncio.to_thread(self._get_file_size, entry)
            except OSError:
                size = 0

            results.append({
                "path": entry.relative_to(self._project_root).as_posix(),
                "name": entry.name,
                "size": size,
            })

        logger.debug(
            f"[search_files] 文件名搜索完成: "
            f"{len(results)} 个结果, 扫描 {files_scanned} 个文件"
        )

        return ToolResult.ok({
            "query": query,
            "search_type": "filename",
            "search_path": path_str,
            "total_results": len(results),
            "max_results": max_results,
            "files_scanned": files_scanned,
            "truncated": len(results) >= max_results,
            "results": results,
        })

    # ------------------------------------------------------------------
    # 内容搜索
    # ------------------------------------------------------------------

    async def _search_content(
        self,
        query: str,
        path_str: str,
        search_dir: Path,
        file_pattern: str | None,
        max_results: int,
        case_sensitive: bool,
    ) -> ToolResult:
        """按文件内容搜索（异步）。

        流程：
        1. 在后台线程中收集候选文件路径（rglob + stat）。
        2. 使用 aiofiles 异步读取文件内容。
        3. 对每行执行正则匹配（超长行截断以防护 ReDoS）。

        Returns:
            ToolResult，data.results 中每项含 path, line, content, match, groups。
        """
        # -- 编译正则表达式 ---------------------------------------------------
        flags: int = 0 if case_sensitive else _re.IGNORECASE

        # 若 query 不含正则元字符，自动转义为字面搜索
        if not _has_regex_metachars(query):
            effective_pattern = _re.escape(query)
        else:
            effective_pattern = query

        try:
            regex = _re.compile(effective_pattern, flags)
        except _re.error as exc:
            # 语法错误 → 回退到转义后的字面搜索
            logger.debug(
                f"[search_files] 正则编译失败 ({exc})，回退到字面搜索: "
                f"{_re.escape(query)!r}"
            )
            regex = _re.compile(_re.escape(query), flags)

        logger.debug(
            f"[search_files] 内容搜索开始: "
            f"pattern={regex.pattern!r}, dir={search_dir}, "
            f"file_pattern={file_pattern!r}, max={max_results}"
        )

        max_file_size: int = self._config.project.max_file_size

        # 展开大括号模式
        patterns = (
            _expand_brace_pattern(file_pattern) if file_pattern else None
        )

        # 在后台线程中收集候选文件
        def collect_candidates() -> list[Path]:
            candidates: list[Path] = []
            for entry in search_dir.rglob("*"):
                if len(candidates) >= _MAX_CANDIDATE_FILES:
                    break
                if self._exclude_dirs.intersection(
                    entry.relative_to(search_dir).parts
                ):
                    continue
                # .gitignore 过滤
                if self._gitignore_filter is not None:
                    if self._gitignore_filter.is_ignored(entry):
                        continue
                if not entry.is_file():
                    continue
                # file_pattern 过滤
                if patterns is not None:
                    target = entry.name if case_sensitive else entry.name.lower()
                    if not any(
                        fnmatch.fnmatch(target, p if case_sensitive else p.lower())
                        for p in patterns
                    ):
                        continue
                # 文件大小检查（提前过滤超大文件）
                try:
                    if entry.stat().st_size > max_file_size:
                        continue
                except OSError:
                    continue
                candidates.append(entry)
            return candidates

        candidates = await asyncio.to_thread(collect_candidates)

        # 逐个处理文件（异步 I/O）
        results: list[dict[str, Any]] = []
        files_scanned = 0
        files_skipped_binary = 0

        for entry in candidates:
            if len(results) >= max_results:
                break

            # -- 二进制检测（异步读取头部）---------------------------------------
            if await self._is_binary_file_async(entry):
                files_skipped_binary += 1
                continue

            # -- 读取并搜索文件内容（异步）---------------------------------------
            matches_in_file = await self._search_in_file_async(
                entry, regex, max_results - len(results)
            )
            if matches_in_file:
                results.extend(matches_in_file)

            files_scanned += 1

        logger.debug(
            f"[search_files] 内容搜索完成: "
            f"{len(results)} 个结果, "
            f"扫描 {files_scanned} 个文件, "
            f"跳过二进制 {files_skipped_binary}"
        )

        return ToolResult.ok({
            "query": query,
            "search_type": "content",
            "search_path": path_str,
            "total_results": len(results),
            "max_results": max_results,
            "files_scanned": files_scanned,
            "truncated": len(results) >= max_results,
            "results": results,
        })

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _get_file_size(file_path: Path) -> int:
        """获取文件大小（用于 asyncio.to_thread 包装）。"""
        return file_path.stat().st_size

    @staticmethod
    async def _is_binary_file_async(file_path: Path) -> bool:
        """异步检测文件是否为二进制格式。

        读取文件头部字节，先检查 BOM 再检查 null 字节占比。
        UTF-16LE 文件（每两字节含一个 0x00）通过 BOM 检测
        正确识别为文本文件。

        Args:
            file_path: 文件路径。

        Returns:
            True 表示二进制文件（或无法读取，为安全起见静默跳过）。
        """
        try:
            async with aiofiles.open(file_path, mode="rb") as f:
                header = await f.read(_BINARY_DETECT_HEADER_SIZE)
        except OSError:
            return True  # 无法读取 → 视为二进制，静默跳过

        if len(header) == 0:
            return False

        # BOM 检测优先于 null 字节检测
        if any(header.startswith(bom) for bom in _BOMS):
            return False

        null_count = header.count(b"\x00")
        return null_count / len(header) > _BINARY_NULL_RATIO_THRESHOLD

    async def _search_in_file_async(
        self,
        file_path: Path,
        regex: _re.Pattern[str],
        remaining: int,
    ) -> list[dict[str, Any]]:
        """异步在单个文件中搜索匹配行。

        使用 aiofiles 异步读取，对每行执行正则匹配。
        超长行截断至 _MAX_LINE_LENGTH 以防止 ReDoS。

        Args:
            file_path: 要搜索的文件路径。
            regex: 编译好的正则表达式对象。
            remaining: 剩余可添加的结果配额。

        Returns:
            匹配结果列表，每项含 path, line, content, match, groups。
        """
        try:
            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                text = await f.read()
        except (UnicodeDecodeError, OSError):
            return []

        rel_path = file_path.relative_to(self._project_root).as_posix()
        matches: list[dict[str, Any]] = []

        for line_no, line in enumerate(text.splitlines(), start=1):
            # 超长行截断以防止 ReDoS
            if len(line) > _MAX_LINE_LENGTH:
                line = line[:_MAX_LINE_LENGTH]

            m = regex.search(line)
            if m is None:
                continue

            # 截断 content 以控制上下文大小
            content = line.strip()
            if len(content) > _CONTENT_MAX_LENGTH:
                match_pos = m.start()
                match_end = m.end()
                # 保留匹配位置周围的上下文
                ctx_start = max(0, match_pos - 40)
                ctx_end = min(len(content), match_end + 40)
                content = (
                    ("..." if ctx_start > 0 else "")
                    + content[ctx_start:ctx_end]
                    + ("..." if ctx_end < len(content) else "")
                )

            matches.append({
                "path": rel_path,
                "line": line_no,
                "content": content,
                "match": m.group(0),  # 完整匹配（非捕获组）
                "groups": list(m.groups()) if m.groups() else [],
            })

            if len(matches) >= remaining:
                break

        return matches
