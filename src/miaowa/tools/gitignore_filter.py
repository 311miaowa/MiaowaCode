""".gitignore 感知的文件过滤系统。

基于 .gitignore 文件规则，在文件遍历中自动排除被忽略的文件和目录。
与 git 的行为保持一致：支持嵌套 .gitignore、! 取反规则、mtime 缓存失效。

PRD §3.3: .gitignore 感知的文件过滤。

Typical usage::

    from pathlib import Path
    from miaowa.tools.gitignore_filter import GitignoreFilter

    f = GitignoreFilter(project_root=Path.cwd())
    f.is_ignored(Path("node_modules/foo.js"))  # True
    f.refresh_if_needed()  # 检查 .gitignore 文件是否已变化
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pathspec

from miaowa.core.config import Config
from miaowa.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# .gitignore 文件名
_GITIGNORE_FILENAME = ".gitignore"

# 单个 .gitignore 文件最大尺寸（字节），超过则跳过以防止 OOM（#1 安全加固）
_MAX_GITIGNORE_SIZE = 1_048_576  # 1 MB

# .gitignore 文件最大行数，超过则跳过逐行容错解析（#2 安全加固）
_MAX_GITIGNORE_LINES = 500

# refresh_if_needed 最小间隔（秒），避免每次调用都执行全树 walk（#6 性能优化）
_MIN_REFRESH_INTERVAL = 2.0

# is_ignored 结果缓存最大条目数（#5 性能优化 — LRU 风格缓存）
_MAX_CACHE_SIZE = 4096


# ============================================================================
# GitignoreFilter
# ============================================================================


class GitignoreFilter:
    """基于 .gitignore 的文件过滤器。

    扫描项目根目录及所有子目录中的 .gitignore 文件，
    使用 pathspec 库解析规则，提供 is_ignored() 查询接口。

    性能优化（#5/#6/#9）：
        - is_ignored() 结果使用 LRU 风格缓存，refresh 时清空
        - spec_dir 相对路径在加载时预计算，避免每次调用重复 relative_to
        - refresh_if_needed() 有时间节流（2 秒间隔）

    安全加固（#1/#2）：
        - .gitignore 文件尺寸上限（1 MB）
        - 逐行容错解析行数上限（500 行）

    Attributes:
        project_root: 项目根目录绝对路径。
        _specs: (spec_dir_rel, PathSpec, mtime) 三元组列表，
            按目录深度从浅到深排列（父目录优先于子目录）。
            spec_dir_rel 是相对于 project_root 的路径。
    """

    def __init__(self, project_root: Path) -> None:
        """初始化 GitignoreFilter。

        扫描 project_root 及其所有子目录中的 .gitignore 文件，
        使用 pathspec.PathSpec 解析每条规则，构建 (directory, spec, mtime) 映射。

        Args:
            project_root: 项目根目录的绝对路径。

        Raises:
            ValueError: project_root 不是目录时抛出。
        """
        resolved = project_root.resolve()
        if not resolved.is_dir():
            raise ValueError(
                f"project_root 必须是目录，实际路径: {resolved}"
            )
        self._project_root = resolved
        # _specs: (相对于 project_root 的 Path, PathSpec, mtime) 三元组列表
        # 按目录深度从浅到深排列（父目录优先于子目录）。
        # 例如 root .gitignore → (Path("."), spec, mtime)
        #      src/.gitignore  → (Path("src"), spec, mtime)
        self._specs: list[tuple[Path, pathspec.PathSpec, float]] = []

        # -- 性能：LRU 风格缓存 + refresh 节流 -------------------------------
        self._cache: dict[str, bool] = {}
        self._cache_hits: int = 0
        self._last_refresh: float = 0.0

        self._load_all()

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        """返回项目根目录路径的副本。"""
        return Path(self._project_root)

    @property
    def spec_count(self) -> int:
        """返回已加载的 .gitignore 文件数量（用于测试）。"""
        return len(self._specs)

    @property
    def cache_info(self) -> dict[str, int]:
        """返回缓存统计信息（用于性能监控）。"""
        return {
            "size": len(self._cache),
            "hits": self._cache_hits,
        }

    # ------------------------------------------------------------------
    # is_ignored
    # ------------------------------------------------------------------

    def is_ignored(self, path: Path) -> bool:
        """检查指定路径是否被 .gitignore 规则忽略。

        按 git 语义处理：
            - 将绝对路径转为相对于 project_root 的路径
            - 按 .gitignore 文件层级从浅到深检查
            - 最后一个匹配的规则决定最终结果
            - ! 取反规则可重新包含被父级忽略的文件
            - .git/ 目录及其内容总是被忽略

        Args:
            path: 要检查的绝对路径（文件或目录）。

        Returns:
            True 表示路径应被忽略，False 表示不应忽略。
        """
        # 始终 resolve，消除 .. 组件和符号链接，同时为缓存键提供一致格式。
        # 工具链传入路径已经过 _resolve_path_within_root 预解析，
        # 对已归一化路径 resolve() 几乎零开销，但提供防御纵深。
        abs_path = path.resolve()
        cache_key = str(abs_path)

        # -- 缓存检查（#5 性能优化）------------------------------------------
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        result = self._is_ignored_impl(abs_path)

        # 缓存结果（带简单的 LRU 驱逐）
        if len(self._cache) >= _MAX_CACHE_SIZE:
            # 驱逐最旧的一半条目
            remove_count = _MAX_CACHE_SIZE // 2
            for key in list(self._cache.keys())[:remove_count]:
                del self._cache[key]
        self._cache[cache_key] = result
        return result

    def _is_ignored_impl(self, abs_path: Path) -> bool:
        """is_ignored 的实际实现（abs_path 必须已经 resolve）。"""
        try:
            rel_path = abs_path.relative_to(self._project_root)
        except ValueError:
            # 路径不在项目根目录内 — 不忽略
            return False

        # 总是忽略 .git/ 目录及其内容
        if rel_path.parts and rel_path.parts[0] == ".git":
            return True

        # 按 .gitignore 层级从浅到深检查
        # 最后一个匹配的规则决定结果（git 语义：深层规则覆盖浅层规则）
        result: bool | None = None

        for spec_dir_rel, spec, _mtime in self._specs:
            # spec_dir_rel 已在 _load_all 中预计算（#9 性能优化）
            try:
                path_from_spec = rel_path.relative_to(spec_dir_rel)
            except ValueError:
                # 路径不在该 .gitignore 的范围内
                continue

            posix_path = path_from_spec.as_posix()

            # 检查该 spec 是否匹配
            match_result = self._match_spec(spec, posix_path)
            if match_result is not None:
                result = match_result

        return result is True

    # ------------------------------------------------------------------
    # refresh_if_needed
    # ------------------------------------------------------------------

    def refresh_if_needed(self) -> None:
        """检查所有已加载 .gitignore 文件的 mtime。

        如有任一文件变化（修改/删除/新增），重新加载全部 .gitignore。
        mtime 缓存机制避免每次查询都重复解析 .gitignore 文件。

        同时检测新增的 .gitignore 文件（当前 .gitignore 文件总数
        与已加载的数量不一致时触发重载）。

        性能优化（#6）：
            带时间节流 — 距离上次刷新不足 _MIN_REFRESH_INTERVAL 秒时跳过。
        """
        # -- 时间节流（#6 性能优化）------------------------------------------
        now = time.monotonic()
        if now - self._last_refresh < _MIN_REFRESH_INTERVAL:
            return
        self._last_refresh = now

        # 快速检查：统计当前磁盘上的 .gitignore 文件数量
        current_count = 0
        for dirpath_str, dirnames, _filenames in os.walk(
            str(self._project_root), topdown=True,
        ):
            if ".git" in dirnames:
                dirnames.remove(".git")
            dirpath = Path(dirpath_str)
            if (dirpath / _GITIGNORE_FILENAME).is_file():
                current_count += 1

        if current_count != len(self._specs):
            logger.debug(
                "[GitignoreFilter] .gitignore 文件数量变化 "
                f"({len(self._specs)} → {current_count})，重新加载"
            )
            self._reload()
            return

        for spec_dir_rel, _spec, cached_mtime in self._specs:
            gitignore_path = (
                self._project_root / spec_dir_rel / _GITIGNORE_FILENAME
            )
            try:
                current_mtime = gitignore_path.stat().st_mtime
            except OSError:
                # 文件被删除 — 需要重新加载
                if cached_mtime != 0:
                    logger.debug(
                        "[GitignoreFilter] .gitignore 文件已删除，重新加载"
                    )
                    self._reload()
                    return
                continue

            if current_mtime != cached_mtime:
                logger.debug(
                    "[GitignoreFilter] .gitignore 文件已变化，重新加载"
                )
                self._reload()
                return

    # ------------------------------------------------------------------
    # from_config (工厂方法)
    # ------------------------------------------------------------------

    @staticmethod
    def from_config(config: Config, project_root: Path) -> GitignoreFilter | None:
        """根据配置创建 GitignoreFilter 实例。

        决策逻辑：
            1. 如果 config.project.use_gitignore 为 False，返回 None
            2. 如果项目根目录无 .gitignore 文件，返回 None
            3. 否则创建并返回 GitignoreFilter 实例

        Args:
            config: Miaowa 应用配置。
            project_root: 项目根目录的绝对路径。

        Returns:
            GitignoreFilter 实例，或在不需要过滤时返回 None。
        """
        if not config.project.use_gitignore:
            logger.debug("[GitignoreFilter] use_gitignore=False，跳过初始化")
            return None

        root_gitignore = project_root.resolve() / _GITIGNORE_FILENAME
        if not root_gitignore.is_file():
            logger.debug(
                f"[GitignoreFilter] 项目根目录无 .gitignore 文件: {project_root}"
            )
            return None

        try:
            return GitignoreFilter(project_root)
        except Exception as exc:
            logger.warning(f"[GitignoreFilter] 初始化失败: {exc}")
            return None

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _reload(self) -> None:
        """清空规格列表和缓存，重新加载所有 .gitignore 文件。"""
        self._specs.clear()
        self._cache.clear()
        self._cache_hits = 0
        self._load_all()

    def _load_all(self) -> None:
        """扫描并加载所有 .gitignore 文件。

        使用 os.walk 遍历项目目录，跳过 .git/ 目录。
        按目录深度排序，确保父目录的规则先于子目录的规则被评估。

        对于无法解析的 .gitignore 文件行，记录 warning 并跳过该行
        （不阻塞整体加载流程）。

        安全加固（#1）：
            跳过超过 _MAX_GITIGNORE_SIZE 字节的 .gitignore 文件。
        """
        entries: list[tuple[Path, pathspec.PathSpec, float]] = []

        for dirpath_str, dirnames, _filenames in os.walk(
            str(self._project_root),
            topdown=True,
        ):
            dirpath = Path(dirpath_str)

            # 跳过 .git/ 目录（不遍历其内容）
            if ".git" in dirnames:
                dirnames.remove(".git")

            gitignore_path = dirpath / _GITIGNORE_FILENAME
            if not gitignore_path.is_file():
                continue

            # -- 尺寸检查（#1 安全加固）--------------------------------------
            try:
                file_size = gitignore_path.stat().st_size
            except OSError:
                continue
            if file_size > _MAX_GITIGNORE_SIZE:
                logger.warning(
                    f"[GitignoreFilter] 跳过超大 .gitignore 文件 "
                    f"({file_size:,} > {_MAX_GITIGNORE_SIZE:,} bytes): "
                    f"{gitignore_path}"
                )
                continue

            # 记录 mtime 用于后续缓存失效检测
            try:
                mtime = gitignore_path.stat().st_mtime
            except OSError:
                continue

            # 读取 .gitignore 文件内容
            try:
                raw_text = gitignore_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning(
                    f"[GitignoreFilter] 无法读取 {gitignore_path}: {exc}"
                )
                continue

            # 使用 pathspec 解析规则
            # from_lines 对无效行内部容错（跳过空行/注释），
            # 对语法异常行可能抛出异常，此处捕获并跳过整行
            lines = raw_text.splitlines()
            try:
                spec = pathspec.PathSpec.from_lines("gitignore", lines)
            except Exception as exc:
                logger.warning(
                    f"[GitignoreFilter] 解析 {gitignore_path} 失败: {exc}。"
                    f"尝试逐行解析以跳过异常行"
                )
                # 逐行解析，跳过有问题的行
                spec = self._parse_lines_resilient(lines, gitignore_path)
                if spec is None:
                    continue

            # 预计算 spec_dir 的相对路径（#9 性能优化）
            # 对根目录：Path(".").parts == ()，len=0，排最前 — 正确
            spec_dir_rel = dirpath.relative_to(self._project_root)
            entries.append((spec_dir_rel, spec, mtime))

        # 按目录深度排序（从浅到深），确保父目录规则先于子目录
        entries.sort(key=lambda e: len(e[0].parts))

        self._specs = entries
        logger.debug(
            f"[GitignoreFilter] 已加载 {len(self._specs)} 个 .gitignore 文件"
        )

    @staticmethod
    def _parse_lines_resilient(
        lines: list[str],
        file_path: Path,
    ) -> pathspec.PathSpec | None:
        """逐行解析 .gitignore 规则，跳过异常行。

        安全加固（#2）：
            超过 _MAX_GITIGNORE_LINES 行时放弃逐行容错解析。

        Args:
            lines: .gitignore 文件的每一行。
            file_path: .gitignore 文件路径（用于日志）。

        Returns:
            解析成功的 PathSpec，或 None（全部行均解析失败时）。
        """
        if len(lines) > _MAX_GITIGNORE_LINES:
            logger.warning(
                f"[GitignoreFilter] 跳过逐行容错解析 — "
                f"行数过多 ({len(lines)} > {_MAX_GITIGNORE_LINES}): {file_path}"
            )
            return None

        valid_lines: list[str] = []
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            # 跳过空行和注释
            if not stripped or stripped.startswith("#"):
                valid_lines.append(line)
                continue
            # 尝试用 pathspec 解析该行
            try:
                # 通过创建临时 spec 来验证该行是否合法
                pathspec.PathSpec.from_lines("gitignore", [line])
                valid_lines.append(line)
            except Exception as exc:
                logger.warning(
                    f"[GitignoreFilter] 跳过 {file_path}:{i} — "
                    f"无法解析规则 {line!r}: {exc}"
                )
                # 将该行转为注释以避免影响后续规则
                valid_lines.append(f"# [miaowa:skipped] {line}")

        if not valid_lines:
            return None

        try:
            return pathspec.PathSpec.from_lines("gitignore", valid_lines)
        except Exception:
            return None

    @staticmethod
    def _match_spec(
        spec: pathspec.PathSpec,
        posix_path: str,
    ) -> bool | None:
        """检查单个 PathSpec 是否匹配指定路径。

        使用 pathspec 内部 patterns 列表逐个检查匹配状态，
        以区分"匹配到忽略规则"、"匹配到取反规则"和"无匹配"三种情况。

        注意：git 的目录模式（如 build/）需要尾随 / 才会被 pathspec 匹配，
        因此对每个 pattern 同时检查 posix_path 和 posix_path + "/"。

        Args:
            spec: pathspec.PathSpec 实例。
            posix_path: POSIX 风格（/ 分隔）的相对路径字符串。

        Returns:
            - True: 被忽略规则匹配（路径应被忽略）
            - False: 被取反规则（!）匹配（路径不应被忽略）
            - None: 无任何规则匹配（该 spec 不影响此路径）
        """
        # 依次检查 spec 中的每条 pattern
        # 使用 gitignore 语义：最后一个匹配的 pattern 决定结果
        last_match: bool | None = None

        # 同时检查无尾随 / 和有尾随 / 的变体
        # 无尾随 / → 匹配文件模式和通用模式（如 *.log）
        # 有尾随 / → 匹配目录模式（如 build/）
        paths_to_check = (posix_path, posix_path + "/")

        for pattern in spec.patterns:
            # GitIgnorePattern 使用编译好的正则表达式
            # regex 使用 match() 语义（从字符串开头匹配，符合 gitignore 规范）
            regex = getattr(pattern, "regex", None)
            if regex is None:
                continue

            for check_path in paths_to_check:
                if regex.match(check_path):
                    # pattern.include:
                    #   True  → 普通忽略规则（如 *.log）
                    #   False → 取反规则（如 !important.log）
                    last_match = bool(getattr(pattern, "include", True))
                    break

        return last_match
