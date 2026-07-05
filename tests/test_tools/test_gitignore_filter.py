"""GitignoreFilter 单元测试。

覆盖基本忽略规则、取反规则、嵌套 .gitignore、mtime 缓存失效、
from_config 工厂方法、异常容错等场景。

PRD §3.3: .gitignore 感知的文件过滤。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from miaowa.core.config import Config
from miaowa.tools.gitignore_filter import GitignoreFilter


# ============================================================================
# 辅助工具
# ============================================================================


def _make_gitignore(dir_path: Path, content: str) -> Path:
    """在指定目录创建 .gitignore 文件并返回其 Path。"""
    f = dir_path / ".gitignore"
    f.write_text(content, encoding="utf-8")
    return f


def _make_file(dir_path: Path, name: str, content: str = "test") -> Path:
    """在指定目录创建文本文件。"""
    f = dir_path / name
    f.write_text(content, encoding="utf-8")
    return f


# ============================================================================
# is_ignored — 基本忽略规则
# ============================================================================


class TestBasicIgnorePatterns:
    """基本的 .gitignore 忽略规则。"""

    def test_glob_pattern_ignored(self, tmp_path):
        """*.log 模式应忽略匹配的日志文件。"""
        _make_gitignore(tmp_path, "*.log\n")
        _make_file(tmp_path, "test.log")
        _make_file(tmp_path, "test.py")

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / "test.log") is True
        assert gf.is_ignored(tmp_path / "test.py") is False

    def test_directory_pattern_ignored(self, tmp_path):
        """build/ 模式应忽略同名目录。"""
        _make_gitignore(tmp_path, "build/\n")
        (tmp_path / "build").mkdir()
        (tmp_path / "src").mkdir()

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / "build") is True
        assert gf.is_ignored(tmp_path / "src") is False

    def test_files_inside_ignored_dir_also_ignored(self, tmp_path):
        """被忽略目录中的文件也一并被忽略。"""
        _make_gitignore(tmp_path, "build/\n")
        (tmp_path / "build").mkdir()
        _make_file(tmp_path / "build", "output.js")

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / "build" / "output.js") is True

    def test_specific_file_ignored(self, tmp_path):
        """指定文件名精确忽略。"""
        _make_gitignore(tmp_path, "secrets.txt\n")
        _make_file(tmp_path, "secrets.txt")
        _make_file(tmp_path, "config.txt")

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / "secrets.txt") is True
        assert gf.is_ignored(tmp_path / "config.txt") is False

    def test_empty_gitignore_does_not_filter(self, tmp_path):
        """空 .gitignore 不应过滤任何文件。"""
        _make_gitignore(tmp_path, "# just a comment\n\n")
        _make_file(tmp_path, "app.py")

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / "app.py") is False


# ============================================================================
# is_ignored — 取反规则（!）
# ============================================================================


class TestNegationPatterns:
    """! 取反规则。"""

    def test_negation_reincludes_file(self, tmp_path):
        """! 规则应重新包含被父规则忽略的文件。"""
        _make_gitignore(tmp_path, "*.log\n!important.log\n")
        _make_file(tmp_path, "debug.log")
        _make_file(tmp_path, "important.log")

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / "debug.log") is True
        assert gf.is_ignored(tmp_path / "important.log") is False

    def test_negation_has_no_effect_without_prior_ignore(self, tmp_path):
        """仅有 ! 规则而无前置忽略规则时，文件不应被忽略。"""
        _make_gitignore(tmp_path, "!important.log\n")
        _make_file(tmp_path, "important.log")

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / "important.log") is False


# ============================================================================
# is_ignored — 嵌套 .gitignore
# ============================================================================


class TestNestedGitignore:
    """子目录 .gitignore 覆盖父目录规则。"""

    def test_child_overrides_parent(self, tmp_path):
        """子目录的 ! 规则应覆盖父目录的忽略规则。"""
        _make_gitignore(tmp_path, "*.log\n")
        (tmp_path / "src").mkdir()
        _make_gitignore(tmp_path / "src", "!debug.log\n")
        _make_file(tmp_path, "error.log")
        _make_file(tmp_path / "src", "debug.log")
        _make_file(tmp_path / "src", "app.py")

        gf = GitignoreFilter(tmp_path)
        # 根目录 .gitignore 忽略所有 *.log
        assert gf.is_ignored(tmp_path / "error.log") is True
        # src/.gitignore 取反 debug.log → 不忽略
        assert gf.is_ignored(tmp_path / "src" / "debug.log") is False
        # src/app.py 不匹配任何规则
        assert gf.is_ignored(tmp_path / "src" / "app.py") is False

    def test_child_adds_new_ignore(self, tmp_path):
        """子目录的 .gitignore 可添加新的忽略规则。"""
        (tmp_path / "src").mkdir()
        _make_gitignore(tmp_path / "src", "*.tmp\n")
        _make_file(tmp_path / "src", "data.tmp")
        _make_file(tmp_path / "src", "app.py")

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / "src" / "data.tmp") is True
        assert gf.is_ignored(tmp_path / "src" / "app.py") is False

    def test_deeply_nested_gitignore(self, tmp_path):
        """三层嵌套 .gitignore 的规则链。"""
        _make_gitignore(tmp_path, "*.log\n")  # 忽略所有 .log
        (tmp_path / "a").mkdir()
        _make_gitignore(tmp_path / "a", "!important.log\n")  # 重新包含 important.log
        (tmp_path / "a" / "b").mkdir()
        _make_gitignore(tmp_path / "a" / "b", "important.log\n")  # 再次忽略 important.log

        _make_file(tmp_path, "error.log")
        _make_file(tmp_path / "a", "important.log")
        _make_file(tmp_path / "a" / "b", "important.log")

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / "error.log") is True
        # a/.gitignore 取反 → 不忽略
        assert gf.is_ignored(tmp_path / "a" / "important.log") is False
        # a/b/.gitignore 再次忽略 → 忽略
        assert gf.is_ignored(tmp_path / "a" / "b" / "important.log") is True


# ============================================================================
# is_ignored — .git 目录
# ============================================================================


class TestGitDirectoryAlwaysIgnored:
    """.git/ 目录及其内容始终被忽略。"""

    def test_git_dir_always_ignored(self, tmp_path):
        """即使 .gitignore 为空，.git/ 也应被忽略。"""
        _make_gitignore(tmp_path, "# empty\n")
        (tmp_path / ".git").mkdir()

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / ".git") is True

    def test_git_inner_files_always_ignored(self, tmp_path):
        """.git/ 中的文件应被忽略。"""
        _make_gitignore(tmp_path, "*.log\n")  # .gitignore 不涉及 .git
        (tmp_path / ".git").mkdir()
        _make_file(tmp_path / ".git", "config")

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / ".git" / "config") is True


# ============================================================================
# is_ignored — 边界情况
# ============================================================================


class TestEdgeCases:
    """边界情况测试。"""

    def test_path_outside_project_root_not_ignored(self, tmp_path):
        """项目之外的路径不判断为忽略。"""
        _make_gitignore(tmp_path, "*.log\n")

        gf = GitignoreFilter(tmp_path)
        outside = tmp_path.parent / "outside.log"
        assert gf.is_ignored(outside) is False

    def test_non_existent_file_checked_by_path(self, tmp_path):
        """不存在文件的路径也应按规则匹配（路径匹配而非文件存在性）。"""
        _make_gitignore(tmp_path, "*.log\n")

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / "not_created_yet.log") is True
        assert gf.is_ignored(tmp_path / "not_created_yet.py") is False

    def test_no_gitignore_file(self, tmp_path):
        """无任何 .gitignore 文件时不过滤任何内容。"""
        _make_file(tmp_path, "app.py")
        (tmp_path / "src").mkdir()
        _make_file(tmp_path / "src", "utils.py")

        gf = GitignoreFilter(tmp_path)
        assert gf.is_ignored(tmp_path / "app.py") is False
        assert gf.is_ignored(tmp_path / "src" / "utils.py") is False


# ============================================================================
# from_config 工厂方法
# ============================================================================


class TestFromConfig:
    """GitignoreFilter.from_config 工厂方法。"""

    def test_returns_instance_when_gitignore_exists(self, tmp_path):
        """项目有 .gitignore 且 use_gitignore=True 时应返回实例。"""
        _make_gitignore(tmp_path, "*.log\n")
        config = Config()
        config.project.use_gitignore = True

        gf = GitignoreFilter.from_config(config, tmp_path)
        assert gf is not None
        assert isinstance(gf, GitignoreFilter)

    def test_returns_none_when_use_gitignore_false(self, tmp_path):
        """use_gitignore=False 时应返回 None。"""
        _make_gitignore(tmp_path, "*.log\n")
        config = Config()
        config.project.use_gitignore = False

        gf = GitignoreFilter.from_config(config, tmp_path)
        assert gf is None

    def test_returns_none_when_no_gitignore_file(self, tmp_path):
        """项目根目录无 .gitignore 时应返回 None。"""
        config = Config()
        config.project.use_gitignore = True

        gf = GitignoreFilter.from_config(config, tmp_path)
        assert gf is None

    def test_default_config_has_use_gitignore_true(self):
        """ProjectConfig 的 use_gitignore 默认值应为 True。"""
        config = Config()
        assert config.project.use_gitignore is True


# ============================================================================
# refresh_if_needed — mtime 缓存失效
# ============================================================================


class TestRefreshIfNeeded:
    """mtime 缓存失效检测。"""

    def test_no_reload_when_mtime_unchanged(self, tmp_path):
        """mtime 无变化时不应重新加载。"""
        _make_gitignore(tmp_path, "*.log\n")

        gf = GitignoreFilter(tmp_path)
        original_count = gf.spec_count
        gf.refresh_if_needed()
        assert gf.spec_count == original_count

    def test_reload_when_mtime_changed(self, tmp_path):
        """.gitignore 被修改后应重新加载。"""
        gi = _make_gitignore(tmp_path, "*.log\n")

        gf = GitignoreFilter(tmp_path)
        original_count = gf.spec_count

        # 等待文件系统时间戳更新（至少 1ms）
        time.sleep(0.01)
        gi.write_text("*.log\n*.tmp\n", encoding="utf-8")

        gf.refresh_if_needed()
        # 重新加载后的 spec_count 应该不变（还是同一个文件）
        assert gf.spec_count == original_count

    def test_reload_when_gitignore_deleted(self, tmp_path):
        """.gitignore 被删除后应重新加载（从列表中移除）。"""
        gi = _make_gitignore(tmp_path, "*.log\n")

        gf = GitignoreFilter(tmp_path)
        assert gf.spec_count == 1

        gi.unlink()

        gf.refresh_if_needed()
        assert gf.spec_count == 0

    def test_reload_when_new_gitignore_added(self, tmp_path):
        """新增 .gitignore 文件后应被检测到。"""
        gf = GitignoreFilter(tmp_path)
        assert gf.spec_count == 0

        # 创建 .gitignore 并手动触发重载
        _make_gitignore(tmp_path, "*.log\n")
        gf.refresh_if_needed()
        assert gf.spec_count == 1


# ============================================================================
# 异常容错
# ============================================================================


class TestErrorResilience:
    """异常行容错 — 解析错误时 logger.warning + 跳过异常行。"""

    def test_invalid_pattern_line_is_skipped(self, tmp_path, caplog):
        """无法解析的行应被跳过，不阻塞整体加载。"""
        _make_gitignore(tmp_path, "*.log\n[invalid regex\n*.py\n")

        gf = GitignoreFilter(tmp_path)
        # 仍应加载成功
        assert gf.spec_count == 1
        # *.log 仍应生效
        assert gf.is_ignored(tmp_path / "test.log") is True
        # *.py 也应生效（跳过了中间无效行）
        assert gf.is_ignored(tmp_path / "test.py") is True

    def test_binary_gitignore_skipped(self, tmp_path):
        """无法以 UTF-8 解码的 .gitignore 应被跳过。"""
        gi = tmp_path / ".gitignore"
        gi.write_bytes(b"\x00\x00\x00invalid utf-8\xff\xfe")

        gf = GitignoreFilter(tmp_path)
        # 应加载成功（无有效的 .gitignore）
        assert gf.spec_count == 0


# ============================================================================
# 属性
# ============================================================================


class TestProperties:
    """属性测试。"""

    def test_project_root_is_copy(self, tmp_path):
        """project_root 属性返回副本而非内部引用。"""
        gf = GitignoreFilter(tmp_path)
        pr = gf.project_root
        assert pr == tmp_path.resolve()
        assert pr is not gf._project_root  # pyright: ignore[reportPrivateUsage]

    def test_spec_count_with_multiple_files(self, tmp_path):
        """多个 .gitignore 文件应被正确计数。"""
        _make_gitignore(tmp_path, "*.log\n")
        (tmp_path / "src").mkdir()
        _make_gitignore(tmp_path / "src", "*.tmp\n")
        (tmp_path / "tests").mkdir()
        _make_gitignore(tmp_path / "tests", "*.pyc\n")

        gf = GitignoreFilter(tmp_path)
        assert gf.spec_count == 3

    def test_gitignore_in_git_dir_is_skipped(self, tmp_path):
        """.git/ 目录中的 .gitignore 不应被加载。"""
        (tmp_path / ".git").mkdir(parents=True)
        _make_gitignore(tmp_path / ".git", "*.log\n")

        gf = GitignoreFilter(tmp_path)
        assert gf.spec_count == 0


# ============================================================================
# 初始化错误
# ============================================================================


class TestInitErrors:
    """初始化错误处理。"""

    def test_non_directory_project_root_raises(self, tmp_path):
        """project_root 不是目录时应抛出 ValueError。"""
        fake = tmp_path / "not_a_dir"
        with pytest.raises(ValueError, match="project_root 必须是目录"):
            GitignoreFilter(fake)
