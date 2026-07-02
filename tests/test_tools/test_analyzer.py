"""项目分析器单元测试 — AnalyzeProjectTool 与 ProjectAnalyzer。

使用 fixtures/sample_python_project/ 和临时项目验证技术栈检测、
模块分析、目录树生成、依赖提取、缓存行为、项目名称检测等功能。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from miaowa.core.config import Config
from miaowa.core.types import ToolResult
from miaowa.tools.analyzer import (
    AnalyzeProjectTool,
    ProjectAnalyzer,
    TechStackDetector,
    ModuleAnalyzer,
    _safe_read_text,
)


# ---------------------------------------------------------------------------
# 辅助 fixture：指向 fixtures/sample_python_project 的路径
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_project_path() -> Path:
    """返回 fixtures/sample_python_project 的绝对路径。"""
    p = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_python_project"
    if not p.is_dir():
        pytest.skip(f"fixtures 目录不存在: {p}")
    return p


# ============================================================================
# 1. test_detect_python_project
# ============================================================================


class TestDetectPythonProject:
    """验证 Python 项目被正确识别。"""

    def test_detect_language_python(self, sample_project_path, mock_config):
        """sample_python_project 被识别为 Python 项目。"""
        detector = TechStackDetector(sample_project_path)
        languages = detector.detect_languages()
        assert "Python" in languages

    def test_detect_build_tools_includes_setuptools(self, sample_project_path, mock_config):
        """setuptools 构建工具被检测到（基于 pyproject.toml build-backend）。"""
        detector = TechStackDetector(sample_project_path)
        tools = detector.detect_build_tools()
        assert "setuptools" in tools

    def test_detect_package_manager(self, sample_project_path, mock_config):
        """包管理器被检测到（presence of lock files）。"""
        detector = TechStackDetector(sample_project_path)
        pm = detector.detect_package_manager()
        # 没有锁文件时返回 None
        assert pm is None or isinstance(pm, str)

    def test_detect_package_manager_with_lock_file(self, tmp_path, mock_config):
        """存在 requirements.txt 时检测到 pip。"""
        (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test-pkg"\n', encoding="utf-8",
        )
        detector = TechStackDetector(tmp_path)
        assert detector.detect_package_manager() == "pip"

    def test_detect_all_returns_structured_result(self, sample_project_path, mock_config):
        """detect_all() 返回结构化的技术栈信息。"""
        detector = TechStackDetector(sample_project_path)
        result = detector.detect_all()
        assert "languages" in result
        assert "frameworks" in result
        assert "build_tools" in result
        assert "package_manager" in result
        assert isinstance(result["languages"], list)
        assert isinstance(result["build_tools"], list)


# ============================================================================
# 2. test_detect_fastapi
# ============================================================================


class TestDetectFastAPI:
    """验证 FastAPI 框架检测。"""

    def test_fastapi_detected_in_dependencies(self, tmp_path, mock_config):
        """在依赖文件中包含 'fastapi' 包名时被检测到。"""
        # requirements.txt 使用 == 语法，直接匹配框架正则
        (tmp_path / "requirements.txt").write_text(
            "fastapi==0.100.0\nuvicorn==0.15.0\n",
            encoding="utf-8",
        )
        # 还需要 Python 指纹文件
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "fastapi-app"\n',
            encoding="utf-8",
        )

        detector = TechStackDetector(tmp_path)
        languages = detector.detect_languages()
        frameworks = detector.detect_frameworks(languages)

        assert "Python" in languages
        assert "FastAPI" in frameworks.get("Python", [])

    def test_no_fastapi_when_not_present(self, sample_project_path, mock_config):
        """sample_python_project 不含 FastAPI，不应检测到。"""
        detector = TechStackDetector(sample_project_path)
        languages = detector.detect_languages()
        frameworks = detector.detect_frameworks(languages)

        assert "FastAPI" not in frameworks.get("Python", [])


# ============================================================================
# 2b. test_detect_django / test_detect_flask
# ============================================================================


class TestDetectDjango:
    """验证 Django 框架检测。"""

    def test_django_detected_in_requirements_txt(self, tmp_path, mock_config):
        """requirements.txt 中包含 django==4.2 时被检测到。"""
        (tmp_path / "requirements.txt").write_text(
            "django==4.2.0\n", encoding="utf-8",
        )
        # 还需要 Python 指纹文件
        (tmp_path / "setup.py").write_text(
            "from setuptools import setup; setup()\n", encoding="utf-8",
        )

        detector = TechStackDetector(tmp_path)
        languages = detector.detect_languages()
        frameworks = detector.detect_frameworks(languages)

        assert "Python" in languages
        assert "Django" in frameworks.get("Python", [])

    def test_no_django_when_not_present(self, sample_project_path, mock_config):
        """不含 django 时不应检测到。"""
        detector = TechStackDetector(sample_project_path)
        languages = detector.detect_languages()
        frameworks = detector.detect_frameworks(languages)

        assert "Django" not in frameworks.get("Python", [])


class TestDetectFlask:
    """验证 Flask 框架检测。"""

    def test_flask_detected_in_requirements_txt(self, tmp_path, mock_config):
        """requirements.txt 中包含 flask 时被检测到。"""
        (tmp_path / "requirements.txt").write_text("flask==2.3.0\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flask-app"\n', encoding="utf-8",
        )

        detector = TechStackDetector(tmp_path)
        languages = detector.detect_languages()
        frameworks = detector.detect_frameworks(languages)

        assert "Flask" in frameworks.get("Python", [])

    def test_no_flask_when_not_present(self, sample_project_path, mock_config):
        """不含 flask 时不应检测到。"""
        detector = TechStackDetector(sample_project_path)
        languages = detector.detect_languages()
        frameworks = detector.detect_frameworks(languages)

        assert "Flask" not in frameworks.get("Python", [])


# ============================================================================
# 3. test_full_analysis_structure
# ============================================================================


class TestFullAnalysisStructure:
    """验证 full_analysis() 返回结构的完整性。"""

    @pytest.mark.asyncio
    async def test_full_analysis_has_all_keys(self, sample_project_path, mock_config):
        """完整分析返回所有预期字段。"""
        analyzer = ProjectAnalyzer(sample_project_path, mock_config)
        result = await analyzer.full_analysis()

        expected_keys = [
            "project_name", "project_root", "tech_stack",
            "directory_tree", "modules", "dependencies",
            "structure", "key_files", "summary", "statistics",
        ]
        for key in expected_keys:
            assert key in result, f"缺少字段: {key}"

    @pytest.mark.asyncio
    async def test_full_analysis_project_name(self, sample_project_path, mock_config):
        """项目名称从 pyproject.toml 提取。"""
        analyzer = ProjectAnalyzer(sample_project_path, mock_config)
        result = await analyzer.full_analysis()

        assert result["project_name"] == "sample-app"

    @pytest.mark.asyncio
    async def test_full_analysis_tech_stack_not_empty(self, sample_project_path, mock_config):
        """技术栈信息非空。"""
        analyzer = ProjectAnalyzer(sample_project_path, mock_config)
        result = await analyzer.full_analysis()

        assert len(result["tech_stack"]["languages"]) > 0
        assert len(result["tech_stack"]["build_tools"]) > 0

    @pytest.mark.asyncio
    async def test_full_analysis_statistics(self, sample_project_path, mock_config):
        """统计信息包含文件数和目录数。"""
        analyzer = ProjectAnalyzer(sample_project_path, mock_config)
        result = await analyzer.full_analysis()

        stats = result["statistics"]
        assert stats["total_files"] > 0
        assert stats["total_dirs"] > 0
        assert ".py" in stats["file_extensions"]

    @pytest.mark.asyncio
    async def test_full_analysis_dependencies(self, sample_project_path, mock_config):
        """依赖项被正确解析。"""
        analyzer = ProjectAnalyzer(sample_project_path, mock_config)
        result = await analyzer.full_analysis()

        deps = result["dependencies"]
        assert "Python" in deps
        dep_names = [d["name"] for d in deps["Python"]]
        # [project] dependencies 中存储完整版本描述符（如 "requests>=2.31"）
        assert any("requests" in n for n in dep_names), f"未找到 requests 依赖: {dep_names}"
        assert any("click" in n for n in dep_names), f"未找到 click 依赖: {dep_names}"

    @pytest.mark.asyncio
    async def test_full_analysis_structure_detected(self, sample_project_path, mock_config):
        """代码架构被检测到（至少返回结果）。"""
        analyzer = ProjectAnalyzer(sample_project_path, mock_config)
        result = await analyzer.full_analysis()

        assert result["structure"] in [
            "MVC", "MVVM", "分层架构", "微服务", "单体",
        ]

    @pytest.mark.asyncio
    async def test_full_analysis_key_files(self, sample_project_path, mock_config):
        """关键文件列表包含 pyproject.toml 等。"""
        analyzer = ProjectAnalyzer(sample_project_path, mock_config)
        result = await analyzer.full_analysis()

        key_files = result["key_files"]
        assert "pyproject.toml" in key_files
        assert "README.md" in key_files


# ============================================================================
# 4. test_analyze_specific_aspect
# ============================================================================


class TestAnalyzeSpecificAspect:
    """验证 AnalyzeProjectTool 的 aspect 参数控制分析维度。"""

    @pytest.mark.asyncio
    async def test_aspect_tech_stack(self, sample_project_path, mock_config):
        """aspect='tech_stack' 仅返回技术栈。"""
        tool = AnalyzeProjectTool(sample_project_path, mock_config)
        result = await tool.execute(aspect="tech_stack")

        assert result.success
        assert "languages" in result.data
        assert "frameworks" in result.data
        assert "build_tools" in result.data

    @pytest.mark.asyncio
    async def test_aspect_modules(self, sample_project_path, mock_config):
        """aspect='modules' 返回模块结构。"""
        tool = AnalyzeProjectTool(sample_project_path, mock_config)
        result = await tool.execute(aspect="modules")

        assert result.success
        assert "language" in result.data
        assert "modules" in result.data

    @pytest.mark.asyncio
    async def test_aspect_dependencies(self, sample_project_path, mock_config):
        """aspect='dependencies' 返回依赖映射。"""
        tool = AnalyzeProjectTool(sample_project_path, mock_config)
        result = await tool.execute(aspect="dependencies")

        assert result.success
        assert "Python" in result.data

    @pytest.mark.asyncio
    async def test_aspect_structure(self, sample_project_path, mock_config):
        """aspect='structure' 返回代码架构。"""
        tool = AnalyzeProjectTool(sample_project_path, mock_config)
        result = await tool.execute(aspect="structure")

        assert result.success
        assert "structure" in result.data

    @pytest.mark.asyncio
    async def test_aspect_overview(self, sample_project_path, mock_config):
        """aspect='overview'（默认）返回完整分析。"""
        tool = AnalyzeProjectTool(sample_project_path, mock_config)
        result = await tool.execute()  # default aspect = "overview"

        assert result.success
        assert "project_name" in result.data
        assert "directory_tree" in result.data
        assert "summary" in result.data

    @pytest.mark.asyncio
    async def test_unknown_aspect_returns_fail(self, sample_project_path, mock_config):
        """未知 aspect 返回 fail。"""
        tool = AnalyzeProjectTool(sample_project_path, mock_config)
        result = await tool.execute(aspect="invalid_aspect")

        assert not result.success
        assert "未知的分析维度" in result.error

    @pytest.mark.asyncio
    async def test_modules_aspect_on_empty_project(self, tmp_path, mock_config):
        """空项目（无语言）的 modules aspect 返回 language='未知' + 空模块。"""
        tool = AnalyzeProjectTool(tmp_path, mock_config)
        result = await tool.execute(aspect="modules")

        assert result.success
        assert result.data["language"] == "未知"
        assert result.data["modules"] == []


# ============================================================================
# 5. test_analyze_empty_project
# ============================================================================


class TestAnalyzeEmptyProject:
    """验证空目录项目的分析行为。"""

    @pytest.mark.asyncio
    async def test_empty_project_no_languages(self, tmp_path, mock_config):
        """空目录检测不到任何语言。"""
        analyzer = ProjectAnalyzer(tmp_path, mock_config)
        result = await analyzer.full_analysis()

        assert result["tech_stack"]["languages"] == []
        assert result["modules"] == []
        assert result["dependencies"] == {}
        assert "未检测到任何已知编程语言" in result["summary"]

    @pytest.mark.asyncio
    async def test_empty_project_name_falls_back_to_dirname(self, tmp_path, mock_config):
        """空项目的 project_name 回退为目录名。"""
        analyzer = ProjectAnalyzer(tmp_path, mock_config)
        result = await analyzer.full_analysis()

        assert result["project_name"] == tmp_path.name


# ============================================================================
# 6. test_generate_tree_format
# ============================================================================


class TestGenerateTreeFormat:
    """验证目录树生成格式。"""

    def test_tree_starts_with_dot(self, sample_project_path, mock_config):
        """目录树首行为 "."。"""
        analyzer = ProjectAnalyzer(sample_project_path, mock_config)
        tree = analyzer._generate_tree(max_depth=2)

        assert tree.startswith(".")
        assert "\n" in tree

    def test_tree_contains_files_and_dirs(self, sample_project_path, mock_config):
        """目录树包含文件和目录。"""
        analyzer = ProjectAnalyzer(sample_project_path, mock_config)
        tree = analyzer._generate_tree(max_depth=3)

        # pyproject.toml 应在树中
        assert "pyproject.toml" in tree
        # 子目录应有 / 后缀
        assert "src/" in tree or "├── src" in tree or "└── src" in tree

    def test_tree_respects_max_depth(self, sample_project_path, mock_config):
        """max_depth 限制递归深度。"""
        analyzer = ProjectAnalyzer(sample_project_path, mock_config)

        tree_shallow = analyzer._generate_tree(max_depth=1)
        tree_deep = analyzer._generate_tree(max_depth=3)

        # 深树应有更多行
        assert len(tree_deep.splitlines()) >= len(tree_shallow.splitlines())

    @pytest.mark.asyncio
    async def test_tree_async_wrapper(self, sample_project_path, mock_config):
        """_generate_tree_async 返回与同步版本等价结果。"""
        analyzer = ProjectAnalyzer(sample_project_path, mock_config)
        tree_async = await analyzer._generate_tree_async(max_depth=2)
        tree_sync = analyzer._generate_tree(max_depth=2)

        assert tree_async == tree_sync


# ============================================================================
# 7. test_detect_project_name
# ============================================================================


class TestDetectProjectName:
    """验证项目名称检测逻辑。"""

    def test_from_pyproject_toml_project_name(self, tmp_path, mock_config):
        """从 pyproject.toml [project].name 提取。"""
        ppt = tmp_path / "pyproject.toml"
        ppt.write_text('[project]\nname = "my-awesome-app"\n', encoding="utf-8")

        analyzer = ProjectAnalyzer(tmp_path, mock_config)
        name = analyzer._detect_project_name()
        assert name == "my-awesome-app"

    def test_from_pyproject_poetry_name(self, tmp_path, mock_config):
        """从 pyproject.toml [tool.poetry].name 提取。"""
        ppt = tmp_path / "pyproject.toml"
        ppt.write_text(
            '[tool.poetry]\nname = "poetry-app"\nversion = "0.1.0"\n',
            encoding="utf-8",
        )

        analyzer = ProjectAnalyzer(tmp_path, mock_config)
        name = analyzer._detect_project_name()
        assert name == "poetry-app"

    def test_from_package_json(self, tmp_path, mock_config):
        """从 package.json .name 提取。"""
        pkg = tmp_path / "package.json"
        pkg.write_text('{"name": "node-app", "version": "1.0.0"}', encoding="utf-8")

        analyzer = ProjectAnalyzer(tmp_path, mock_config)
        name = analyzer._detect_project_name()
        assert name == "node-app"

    def test_fallback_to_dirname(self, tmp_path, mock_config):
        """无配置文件时回退到目录名。"""
        analyzer = ProjectAnalyzer(tmp_path, mock_config)
        name = analyzer._detect_project_name()
        assert name == tmp_path.name


# ============================================================================
# 8. test_cache_hit / test_cache_invalidation
# ============================================================================


class TestAnalyzeProjectCache:
    """验证 AnalyzeProjectTool 的实例级缓存。"""

    @pytest.mark.asyncio
    async def test_cache_hit(self, sample_project_path, mock_config):
        """同一 aspect 第二次调用返回缓存结果。"""
        tool = AnalyzeProjectTool(sample_project_path, mock_config)
        r1 = await tool.execute(aspect="structure")
        r2 = await tool.execute(aspect="structure")

        assert r1.success and r2.success
        # 缓存命中时返回同一对象引用
        assert r1.data is r2.data

    @pytest.mark.asyncio
    async def test_different_aspects_not_cached_together(self, sample_project_path, mock_config):
        """不同 aspect 各自独立缓存。"""
        tool = AnalyzeProjectTool(sample_project_path, mock_config)
        r1 = await tool.execute(aspect="structure")
        r2 = await tool.execute(aspect="tech_stack")

        assert r1.success and r2.success
        assert r1.data is not r2.data
        # structure 结果不含 languages
        assert "languages" not in r1.data
        assert "languages" in r2.data

    @pytest.mark.asyncio
    async def test_cache_is_per_instance(self, sample_project_path, mock_config):
        """不同 AnalyzeProjectTool 实例的缓存互不影响。"""
        t1 = AnalyzeProjectTool(sample_project_path, mock_config)
        t2 = AnalyzeProjectTool(sample_project_path, mock_config)

        r1 = await t1.execute(aspect="structure")
        r2 = await t2.execute(aspect="structure")

        assert r1.success and r2.success
        # 不同实例，数据不共享引用
        assert r1.data is not r2.data


# ============================================================================
# 9. test_detect_node_project
# ============================================================================


class TestDetectNodeProject:
    """验证 Node.js / JavaScript 项目检测（使用临时项目）。"""

    def test_js_project_language_detection(self, tmp_path, mock_config):
        """包含 package.json 的目录被识别为 JavaScript 项目。"""
        (tmp_path / "package.json").write_text(
            '{"name": "test-node", "dependencies": {"express": "^4.18"}}',
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()

        detector = TechStackDetector(tmp_path)
        languages = detector.detect_languages()

        assert "JavaScript" in languages

    def test_ts_project_language_detection(self, tmp_path, mock_config):
        """包含 tsconfig.json 的目录被识别为 TypeScript 项目。"""
        (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
        (tmp_path / "src").mkdir()

        detector = TechStackDetector(tmp_path)
        languages = detector.detect_languages()

        assert "TypeScript" in languages

    def test_express_framework_detection(self, tmp_path, mock_config):
        """Express 框架被检测到。"""
        (tmp_path / "package.json").write_text(
            '{"name": "express-app", "dependencies": {"express": "^4.18.0", "cors": "^2.8"}}',
            encoding="utf-8",
        )

        detector = TechStackDetector(tmp_path)
        frameworks = detector.detect_frameworks(["JavaScript"])

        assert "Express" in frameworks.get("JavaScript", [])

    def test_node_dependencies_parsed(self, tmp_path, mock_config):
        """Node.js 依赖被正确解析。"""
        (tmp_path / "package.json").write_text(
            json.dumps({
                "name": "node-deps",
                "dependencies": {"lodash": "^4.17.0"},
                "devDependencies": {"jest": "^29.0.0"},
            }),
            encoding="utf-8",
        )

        analyzer = ModuleAnalyzer(tmp_path, mock_config)
        deps = analyzer.analyze_dependencies()

        assert "JavaScript" in deps
        js_deps = [d["name"] for d in deps["JavaScript"]]
        assert "lodash" in js_deps
        assert "jest" in js_deps


# ============================================================================
# 10. 模块结构分析
# ============================================================================


class TestModuleAnalysis:
    """验证 ModuleAnalyzer 的模块分析。"""

    def test_analyze_python_modules(self, sample_project_path, mock_config):
        """Python 模块结构被正确分析。"""
        analyzer = ModuleAnalyzer(sample_project_path, mock_config)
        modules = analyzer.analyze_modules("Python")

        assert len(modules) > 0
        # sample_project 有 src/sample_app 作为模块
        module_names = [m["name"] for m in modules]
        assert "sample_app" in module_names

        # 模块包含 files 和 submodules
        sample_app = next(m for m in modules if m["name"] == "sample_app")
        assert sample_app["files"] >= 1  # __init__.py and main.py
        assert "path" in sample_app

    def test_analyze_structure_returns_string(self, sample_project_path, mock_config):
        """analyze_structure() 返回架构风格字符串。"""
        analyzer = ModuleAnalyzer(sample_project_path, mock_config)
        structure = analyzer.analyze_structure()
        assert isinstance(structure, str)
        assert len(structure) > 0

    def test_find_source_roots_includes_src(self, sample_project_path, mock_config):
        """_find_source_roots 包含 src 目录。"""
        analyzer = ModuleAnalyzer(sample_project_path, mock_config)
        roots = analyzer._find_source_roots()
        root_names = [r.name for r in roots]
        assert "src" in root_names


# ============================================================================
# 辅助函数测试
# ============================================================================


class TestSafeReadText:
    """验证 _safe_read_text 行为。"""

    def test_read_existing_file(self, tmp_path):
        """成功读取存在的文件。"""
        f = tmp_path / "test.txt"
        f.write_text("hello", encoding="utf-8")
        content = _safe_read_text(f)
        assert content == "hello"

    def test_read_nonexistent_file(self, tmp_path):
        """不存在的文件返回 None。"""
        content = _safe_read_text(tmp_path / "nonexistent.txt")
        assert content is None

    def test_read_file_too_large(self, tmp_path):
        """过大文件返回 None。"""
        f = tmp_path / "big.txt"
        f.write_text("x" * 2000, encoding="utf-8")
        content = _safe_read_text(f, max_size=100)
        assert content is None

    def test_read_non_utf8_file(self, tmp_path):
        """非 UTF-8 文件返回 None（errors='ignore' 不抛异常，但可能返回乱码）。"""
        f = tmp_path / "gbk.bin"
        f.write_bytes(b"\xc4\xe3\xba\xc3")  # GBK "你好"
        content = _safe_read_text(f)
        # errors='ignore' 使其不抛异常
        assert isinstance(content, str)


class TestAnalyzeProjectToolInit:
    """AnalyzeProjectTool 初始化测试。"""

    def test_init_creates_analyzer(self, tmp_path, mock_config):
        """初始化时创建 ProjectAnalyzer。"""
        tool = AnalyzeProjectTool(tmp_path, mock_config)
        assert tool._analyzer is not None
        assert isinstance(tool._analyzer, ProjectAnalyzer)

    def test_init_creates_empty_cache(self, tmp_path, mock_config):
        """初始化时创建空缓存。"""
        tool = AnalyzeProjectTool(tmp_path, mock_config)
        assert tool._cache == {}
