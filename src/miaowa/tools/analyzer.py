"""项目分析器 — 技术栈检测、模块分析、结构识别、依赖提取。

PRD §6.3: 项目分析器实现。

组成部分:
    - TechStackDetector: 技术栈指纹检测（语言/框架/构建工具/包管理器）
    - ModuleAnalyzer:   模块结构分析与依赖提取
    - ProjectAnalyzer:  整合分析器（目录树生成/项目名检测/关键文件/完整分析）
    - AnalyzeProjectTool: BaseTool 对外接口

Typical usage::

    from pathlib import Path
    from miaowa.core.config import ConfigManager
    from miaowa.tools.analyzer import ProjectAnalyzer

    config = ConfigManager.load_default()
    analyzer = ProjectAnalyzer(project_root=Path.cwd(), config=config)
    result: dict = await analyzer.full_analysis()
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from miaowa.core.config import Config
from miaowa.core.logger import get_logger
from miaowa.core.types import ToolParameter, ToolResult
from miaowa.tools.base import BaseTool

logger = get_logger(__name__)

# ============================================================================
# 常量：技术栈指纹
# ============================================================================

# 15 种语言的指纹文件（PRD §6.3.1）。
# 注：PRD 使用 "Node.js" / ".NET" 命名；本实现使用更精确的
# "JavaScript" / "TypeScript" / "C#" 分别标识，输出更准确。
TECH_FINGERPRINTS: dict[str, list[str]] = {
    "Python": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
    "JavaScript": ["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb"],
    "TypeScript": ["tsconfig.json", "tsconfig.base.json", "tsconfig.build.json"],
    "Go": ["go.mod", "go.sum", "go.work"],
    "Rust": ["Cargo.toml", "Cargo.lock"],
    "Java": ["pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle"],
    "Kotlin": ["build.gradle.kts", "settings.gradle.kts"],
    "Ruby": ["Gemfile", "Rakefile", ".ruby-version"],
    "PHP": ["composer.json", "composer.lock"],
    "C/C++": ["CMakeLists.txt", "Makefile", "configure.ac", "meson.build"],
    "C#": ["*.csproj", "*.sln", "global.json", "NuGet.Config"],
    "Swift": ["Package.swift"],
    "Dart": ["pubspec.yaml", "pubspec.lock"],
    "Elixir": ["mix.exs"],
    "R": ["DESCRIPTION", "NAMESPACE"],
}

# 4 类语言的框架正则匹配（在依赖文件中搜索标识符）
FRAMEWORK_PATTERNS: dict[str, dict[str, list[str]]] = {
    "Python": {
        "Django": [r'"django"', r"'django'", r"django==[\d.]+"],
        "Flask": [r'"flask"', r"'flask'", r"flask==[\d.]+"],
        "FastAPI": [r'"fastapi"', r"'fastapi'", r"fastapi==[\d.]+"],
        "Pyramid": [r'"pyramid"', r"'pyramid'"],
    },
    "JavaScript": {
        "Express": [r'"express"'],
        "Next.js": [r'"next"'],
        "NestJS": [r'"@nestjs/'],
        "React": [r'"react"'],
        "Vue": [r'"vue"'],
        "Angular": [r'"@angular/'],
    },
    "Go": {
        "Gin": [r"github\.com/gin-gonic/gin"],
        "Echo": [r"github\.com/labstack/echo"],
        "Fiber": [r"github\.com/gofiber/fiber"],
        "Chi": [r"github\.com/go-chi/chi"],
    },
    "Java": {
        "Spring Boot": [r"spring-boot-starter"],
        "Quarkus": [r"quarkus"],
        "Micronaut": [r"micronaut"],
    },
}

# 构建工具指标文件
_BUILD_TOOL_INDICATORS: dict[str, list[str]] = {
    "setuptools": ["pyproject.toml", "setup.py", "setup.cfg"],
    "poetry": ["pyproject.toml"],  # checked via [tool.poetry]
    "pipenv": ["Pipfile"],
    "hatch": ["pyproject.toml"],  # checked via [tool.hatch]
    "meson": ["meson.build"],
    "cmake": ["CMakeLists.txt"],
    "make": ["Makefile"],
    "maven": ["pom.xml"],
    "gradle": ["build.gradle", "build.gradle.kts"],
    "cargo": ["Cargo.toml"],
    "npm": ["package.json"],
    "yarn": ["yarn.lock"],
    "pnpm": ["pnpm-lock.yaml"],
    "bun": ["bun.lockb"],
    "composer": ["composer.json"],
    "bundler": ["Gemfile"],
}

# 包管理器指标文件
_PACKAGE_MANAGER_INDICATORS: dict[str, list[str]] = {
    "pip": ["requirements.txt"],
    "poetry": ["poetry.lock"],
    "pipenv": ["Pipfile.lock"],
    "npm": ["package-lock.json"],
    "yarn": ["yarn.lock"],
    "pnpm": ["pnpm-lock.yaml"],
    "bundler": ["Gemfile.lock"],
    "composer": ["composer.lock"],
    "cargo": ["Cargo.lock"],
    "maven": ["pom.xml"],
    "gradle": ["build.gradle", "build.gradle.kts"],
}

# 已知代码组织结构的目录特征
_STRUCTURE_PATTERNS: dict[str, list[list[str]]] = {
    "MVC": [["models", "views", "controllers"]],
    "MVVM": [["models", "views", "viewmodels"], ["model", "view", "viewmodel"]],
    "分层架构": [["presentation", "application", "domain", "infrastructure"],
                 ["api", "service", "repository", "model"],
                 ["handler", "usecase", "repository", "entity"]],
    "微服务": [["services"], ["microservices"], ["apps"]],
}

# 依赖文件最大读取大小（字节），防止恶意超大配置文件
_MAX_DEP_FILE_SIZE = 1_048_576  # 1 MB

# 已知的源代码目录名
_SOURCE_DIR_NAMES: frozenset[str] = frozenset({
    "src", "lib", "app", "pkg", "cmd", "internal",
    "controllers", "models", "views", "routes", "services",
    "handlers", "middleware", "utils", "helpers", "config",
    "tests", "test", "spec",
})


# ============================================================================
# 共享工具函数
# ============================================================================


def _safe_read_text(file_path: Path, max_size: int = _MAX_DEP_FILE_SIZE) -> str | None:
    """安全读取文本文件，带大小限制和异常日志。

    Args:
        file_path: 文件路径。
        max_size: 最大允许字节数。

    Returns:
        文件文本内容，文件不存在/过大/编码错误时返回 None。
    """
    try:
        if not file_path.is_file():
            return None
        if file_path.stat().st_size > max_size:
            logger.warning(
                f"跳过过大依赖文件 ({file_path.stat().st_size:,} > {max_size:,} bytes): {file_path}"
            )
            return None
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning(f"无法读取文件: {file_path} — {exc}")
        return None


# ============================================================================
# Part A: TechStackDetector — 技术栈指纹检测
# ============================================================================


class TechStackDetector:
    """技术栈检测器。

    通过项目根目录中的指纹文件识别使用的编程语言、框架、
    构建工具和包管理器。

    Attributes:
        project_root: 项目根目录路径。
    """

    def __init__(self, project_root: Path) -> None:
        """初始化技术栈检测器。

        Args:
            project_root: 项目根目录的绝对路径。
        """
        self._project_root = project_root
        self._pyproject_cache: dict[str, list[str]] | None = None

    # ------------------------------------------------------------------
    # detect_languages
    # ------------------------------------------------------------------

    def detect_languages(self) -> list[str]:
        """检测项目中使用的编程语言。

        遍历 TECH_FINGERPRINTS，检查项目根目录中是否存在
        对应语言的指纹文件。每种语言只需匹配至少一个指纹文件。

        Returns:
            检测到的语言名称列表，按注册顺序排列。
            无匹配时返回空列表。
        """
        detected: list[str] = []
        for language, fingerprints in TECH_FINGERPRINTS.items():
            for fp in fingerprints:
                if "*" in fp:
                    # 通配符匹配（如 "*.csproj"）
                    if list(self._project_root.glob(fp)):
                        detected.append(language)
                        break
                elif (self._project_root / fp).exists():
                    detected.append(language)
                    break
        logger.info(f"[TechStack] 检测到语言: {detected}")
        return detected

    # ------------------------------------------------------------------
    # detect_frameworks
    # ------------------------------------------------------------------

    def detect_frameworks(self, languages: list[str] | None = None) -> dict[str, list[str]]:
        """检测项目中使用的框架。

        对于每种检测到的语言，读取其依赖配置文件，
        使用 FRAMEWORK_PATTERNS 中的正则表达式匹配框架标识符。

        Args:
            languages: 要检测的语言列表。None 时先调用 detect_languages()。

        Returns:
            {语言: [框架名, ...]} 的映射。
            未检测到框架的语言不会出现在结果中。
        """
        if languages is None:
            languages = self.detect_languages()

        result: dict[str, list[str]] = {}

        for lang in languages:
            lang_patterns = FRAMEWORK_PATTERNS.get(lang)
            if lang_patterns is None:
                continue

            dep_text = self._read_dependency_files(lang)
            if not dep_text:
                continue

            frameworks: list[str] = []
            for fw_name, patterns in lang_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, dep_text):
                        frameworks.append(fw_name)
                        break

            if frameworks:
                result[lang] = frameworks

        logger.info(f"[TechStack] 检测到框架: {result}")
        return result

    # ------------------------------------------------------------------
    # detect_build_tools
    # ------------------------------------------------------------------

    def detect_build_tools(self) -> list[str]:
        """检测项目中使用的构建工具。

        检查 _BUILD_TOOL_INDICATORS 中的指标文件是否存在。
        对于 pyproject.toml 会进一步检查具体工具配置（poetry / hatch）。

        Returns:
            检测到的构建工具名称列表。
            无匹配时返回空列表。
        """
        detected: list[str] = []

        for tool, indicators in _BUILD_TOOL_INDICATORS.items():
            for ind in indicators:
                indicator_path = self._project_root / ind
                if not indicator_path.exists():
                    continue

                # pyproject.toml 需进一步检查具体工具节
                if ind == "pyproject.toml":
                    pyproj_tools = self._detect_pyproject_tools()
                    if tool in pyproj_tools:
                        detected.append(tool)
                else:
                    detected.append(tool)
                break  # 一个工具只检测一次

        # 去重且保持顺序
        seen: set[str] = set()
        unique: list[str] = []
        for t in detected:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        logger.info(f"[TechStack] 检测到构建工具: {unique}")
        return unique

    # ------------------------------------------------------------------
    # detect_package_manager
    # ------------------------------------------------------------------

    def detect_package_manager(self) -> str | None:
        """检测项目中使用的包管理器。

        检查 _PACKAGE_MANAGER_INDICATORS 中的锁文件/配置文件，
        返回第一个匹配的包管理器。

        Returns:
            包管理器名称，未检测到时返回 None。
        """
        for pm, indicators in _PACKAGE_MANAGER_INDICATORS.items():
            for ind in indicators:
                if (self._project_root / ind).exists():
                    logger.info(f"[TechStack] 检测到包管理器: {pm}")
                    return pm
        logger.info("[TechStack] 未检测到包管理器")
        return None

    # ------------------------------------------------------------------
    # detect_all
    # ------------------------------------------------------------------

    def detect_all(self) -> dict[str, Any]:
        """运行全部技术栈检测。

        Returns:
            dict 包含:
                - languages (list[str]): 编程语言列表
                - frameworks (dict[str, list[str]]): 框架映射
                - build_tools (list[str]): 构建工具列表
                - package_manager (str | None): 包管理器
        """
        languages = self.detect_languages()
        frameworks = self.detect_frameworks(languages)
        build_tools = self.detect_build_tools()
        package_manager = self.detect_package_manager()

        return {
            "languages": languages,
            "frameworks": frameworks,
            "build_tools": build_tools,
            "package_manager": package_manager,
        }

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _read_dependency_files(self, language: str) -> str:
        """读取某语言对应的依赖配置文件内容。

        Args:
            language: 语言名称（如 "Python"）。

        Returns:
            所有依赖配置文件内容拼接后的文本。无文件时返回 ""。
        """
        # 每种语言的主要依赖文件
        lang_files: dict[str, list[str]] = {
            "Python": ["pyproject.toml", "setup.py", "setup.cfg",
                        "requirements.txt", "Pipfile"],
            "JavaScript": ["package.json"],
            "TypeScript": ["package.json"],
            "Go": ["go.mod"],
            "Rust": ["Cargo.toml"],
            "Java": ["pom.xml", "build.gradle", "build.gradle.kts"],
            "Kotlin": ["build.gradle.kts"],
            "Ruby": ["Gemfile"],
            "PHP": ["composer.json"],
            "Dart": ["pubspec.yaml"],
            "Elixir": ["mix.exs"],
        }

        candidates = lang_files.get(language, [])
        parts: list[str] = []
        for fname in candidates:
            fpath = self._project_root / fname
            text = _safe_read_text(fpath)
            if text:
                parts.append(text)
        return "\n".join(parts)

    def _detect_pyproject_tools(self) -> list[str]:
        """检查 pyproject.toml 中配置的构建工具。

        检测优先级：
            1. [tool.poetry] → poetry
            2. [tool.hatch]   → hatch
            3. [build-system].build-backend 包含 "setuptools" → setuptools

        结果会被缓存到 _pyproject_cache（同一实例多次调用只解析一次 TOML）。

        Returns:
            检测到的工具名列表。
        """
        if self._pyproject_cache is not None:
            return self._pyproject_cache

        fpath = self._project_root / "pyproject.toml"
        text = _safe_read_text(fpath)
        if text is None:
            self._pyproject_cache = []
            return []

        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            logger.warning(f"pyproject.toml TOML 解析失败: {exc}")
            self._pyproject_cache = []
            return []

        results: list[str] = []

        # [tool.poetry]
        if isinstance(data.get("tool", {}).get("poetry"), dict):
            results.append("poetry")

        # [tool.hatch]
        if isinstance(data.get("tool", {}).get("hatch"), dict):
            results.append("hatch")

        # [build-system] build-backend
        build_backend = data.get("build-system", {}).get("build-backend", "")
        if isinstance(build_backend, str) and "setuptools" in build_backend:
            results.append("setuptools")

        self._pyproject_cache = results
        return results


# ============================================================================
# Part B: ModuleAnalyzer — 模块结构分析
# ============================================================================


class ModuleAnalyzer:
    """模块结构分析器。

    分析源代码目录中的模块组织方式、依赖关系和代码结构模式。

    Attributes:
        project_root: 项目根目录路径。
        config: Miaowa 应用配置。
    """

    def __init__(self, project_root: Path, config: Config) -> None:
        """初始化模块分析器。

        Args:
            project_root: 项目根目录的绝对路径。
            config: Miaowa 应用配置对象。
        """
        self._project_root = project_root
        self._config = config
        self._exclude_dirs: set[str] = set(config.project.exclude_dirs)

    # ------------------------------------------------------------------
    # analyze_modules
    # ------------------------------------------------------------------

    def analyze_modules(self, language: str) -> list[dict[str, Any]]:
        """按语言类型分析模块结构。

        在项目源码目录中查找源文件，按目录分组为模块。
        每种语言使用不同的文件扩展名进行识别。

        Args:
            language: 编程语言名称（如 "Python"）。

        Returns:
            模块信息列表，每项含:
                - name (str): 模块/目录名
                - path (str): 相对路径
                - files (int): 源文件数量
                - submodules (list[dict]): 子模块列表（递归）
        """
        ext_map: dict[str, list[str]] = {
            "Python": [".py"],
            "JavaScript": [".js", ".jsx", ".mjs", ".cjs"],
            "TypeScript": [".ts", ".tsx", ".mts", ".cts"],
            "Go": [".go"],
            "Rust": [".rs"],
            "Java": [".java"],
            "Kotlin": [".kt", ".kts"],
            "Ruby": [".rb"],
            "PHP": [".php"],
            "C/C++": [".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"],
            "C#": [".cs"],
            "Swift": [".swift"],
            "Dart": [".dart"],
            "Elixir": [".ex", ".exs"],
        }

        extensions = ext_map.get(language, [])
        modules: list[dict[str, Any]] = []

        # 常见的源码根目录候选
        source_roots = self._find_source_roots()

        for src_root in source_roots:
            modules.extend(
                self._scan_modules(src_root, extensions)
            )

        logger.info(
            f"[ModuleAnalyzer] {language} 模块分析: "
            f"在 {len(source_roots)} 个源目录中发现 {len(modules)} 个顶层模块"
        )
        return modules

    # ------------------------------------------------------------------
    # analyze_dependencies
    # ------------------------------------------------------------------

    def analyze_dependencies(self) -> dict[str, list[dict[str, str]]]:
        """提取项目依赖列表。

        读取依赖配置文件（如 requirements.txt、package.json 等），
        解析并返回各语言的依赖名称和版本信息。

        Returns:
            {语言/生态: [{"name": ..., "version": ...}, ...]} 映射。
            无依赖文件时返回空 dict。
        """
        result: dict[str, list[dict[str, str]]] = {}

        # -- Python 依赖 ----------------------------------------------------
        py_deps = self._parse_python_deps()
        if py_deps:
            result["Python"] = py_deps

        # -- Node.js 依赖 ---------------------------------------------------
        js_deps = self._parse_node_deps()
        if js_deps:
            result["JavaScript"] = js_deps

        # -- Go 依赖 --------------------------------------------------------
        go_deps = self._parse_go_deps()
        if go_deps:
            result["Go"] = go_deps

        # -- Rust 依赖 ------------------------------------------------------
        rs_deps = self._parse_rust_deps()
        if rs_deps:
            result["Rust"] = rs_deps

        logger.info(
            f"[ModuleAnalyzer] 依赖分析: "
            f"{sum(len(v) for v in result.values())} 个依赖项, "
            f"{len(result)} 个生态"
        )
        return result

    # ------------------------------------------------------------------
    # analyze_structure
    # ------------------------------------------------------------------

    def analyze_structure(self) -> str:
        """分析代码组织方式。

        通过检查项目目录结构中的特征模式，识别代码架构风格：
        MVC、MVVM、分层架构、微服务、单体。

        Returns:
            架构风格名称。未识别时返回 "单体（默认）"。

        Note:
            检测顺序：微服务 → 分层 → MVC → MVVM → 单体（默认）。
            优先识别更具体的模式。
        """
        # 收集所有已知目录名
        all_dirs: set[str] = set()
        for entry in self._project_root.rglob("*"):
            if entry.is_dir():
                # 跳过排除目录
                parts = set(entry.relative_to(self._project_root).parts)
                if parts & self._exclude_dirs:
                    continue
                all_dirs.add(entry.name)

        # 按优先级检测
        # 1. 微服务：存在 services/ 或 apps/ 目录，且其下有独立子目录
        if self._check_structure("微服务", all_dirs):
            logger.info("[ModuleAnalyzer] 检测到架构: 微服务")
            return "微服务"

        # 2. 分层架构
        if self._check_structure("分层架构", all_dirs):
            logger.info("[ModuleAnalyzer] 检测到架构: 分层架构")
            return "分层架构"

        # 3. MVC
        if self._check_structure("MVC", all_dirs):
            logger.info("[ModuleAnalyzer] 检测到架构: MVC")
            return "MVC"

        # 4. MVVM
        if self._check_structure("MVVM", all_dirs):
            logger.info("[ModuleAnalyzer] 检测到架构: MVVM")
            return "MVVM"

        logger.info("[ModuleAnalyzer] 检测到架构: 单体（默认）")
        return "单体"

    # ------------------------------------------------------------------
    # 内部辅助：模块扫描
    # ------------------------------------------------------------------

    def _find_source_roots(self) -> list[Path]:
        """查找项目中的源码根目录。

        Returns:
            源码根目录 Path 列表，按优先级排序。
        """
        candidates: list[Path] = []

        for name in ("src", "lib", "app", "pkg", "cmd", "internal"):
            d = self._project_root / name
            if d.is_dir():
                candidates.append(d)

        # 若未找到标准源目录，回退到项目根目录
        if not candidates:
            candidates.append(self._project_root)

        return candidates

    def _scan_modules(
        self,
        root: Path,
        extensions: list[str],
    ) -> list[dict[str, Any]]:
        """递归扫描目录中的模块结构。

        Args:
            root: 扫描根目录。
            extensions: 视为源文件的扩展名列表。

        Returns:
            模块信息列表。
        """
        modules: list[dict[str, Any]] = []
        try:
            entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return modules

        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name in self._exclude_dirs:
                continue
            if entry.name.startswith("."):
                continue

            # 检查此目录是否有匹配扩展名的直接文件
            file_count = 0
            submodules: list[dict[str, Any]] = []
            try:
                for child in sorted(entry.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                    if child.is_file() and child.suffix in extensions:
                        file_count += 1
                    elif child.is_dir() and child.name not in self._exclude_dirs:
                        if not child.name.startswith("."):
                            sub = self._scan_modules(child, extensions)
                            if sub:
                                submodules.extend(sub)
            except PermissionError:
                pass

            # 如果有源文件或有子模块，记录此模块
            if file_count > 0 or submodules:
                modules.append({
                    "name": entry.name,
                    "path": entry.relative_to(self._project_root).as_posix(),
                    "files": file_count,
                    "submodules": submodules,
                })

        return modules

    # ------------------------------------------------------------------
    # 内部辅助：结构检测
    # ------------------------------------------------------------------

    def _check_structure(self, name: str, all_dirs: set[str]) -> bool:
        """检查目录集合是否匹配某种架构模式。

        Args:
            name: 架构模式名称（MVC / MVVM / 分层架构 / 微服务）。
            all_dirs: 项目中所有目录名的集合。

        Returns:
            True 表示匹配（任一子模式满足即可）。
        """
        patterns = _STRUCTURE_PATTERNS.get(name, [])
        for pattern in patterns:
            if all(d in all_dirs for d in pattern):
                return True
        return False

    # ------------------------------------------------------------------
    # 内部辅助：依赖解析
    # ------------------------------------------------------------------

    def _parse_python_deps(self) -> list[dict[str, str]]:
        """解析 Python 项目依赖。

        Returns:
            依赖列表。
        """
        deps: list[dict[str, str]] = []

        # pyproject.toml
        ppt_text = _safe_read_text(self._project_root / "pyproject.toml")
        if ppt_text:
            try:
                data = tomllib.loads(ppt_text)
                poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
                for name, spec in poetry_deps.items():
                    if name.lower() == "python":
                        continue
                    version = spec if isinstance(spec, str) else spec.get("version", "*") if isinstance(spec, dict) else "*"
                    deps.append({"name": name, "version": str(version)})

                project_deps = data.get("project", {}).get("dependencies", [])
                for d in project_deps:
                    deps.append({"name": str(d), "version": "*"})
            except tomllib.TOMLDecodeError as exc:
                logger.warning(f"pyproject.toml TOML 解析失败: {exc}")

        # requirements.txt
        req_text = _safe_read_text(self._project_root / "requirements.txt")
        if req_text:
            for line in req_text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parsed = re.split(r"[<>=!~]+", line, maxsplit=1)
                name = parsed[0].strip()
                version = parsed[1].strip() if len(parsed) > 1 else "*"
                if name:
                    deps.append({"name": name, "version": version})

        return deps

    def _parse_node_deps(self) -> list[dict[str, str]]:
        """解析 Node.js 项目依赖（package.json）。"""
        text = _safe_read_text(self._project_root / "package.json")
        if text is None:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(f"package.json JSON 解析失败: {exc}")
            return []

        deps: list[dict[str, str]] = []
        for section in ("dependencies", "devDependencies"):
            for name, version in data.get(section, {}).items():
                deps.append({"name": name, "version": str(version)})
        return deps

    def _parse_go_deps(self) -> list[dict[str, str]]:
        """解析 Go 项目依赖（go.mod）。"""
        text = _safe_read_text(self._project_root / "go.mod")
        if text is None:
            return []
        deps: list[dict[str, str]] = []
        in_require = False
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("require ("):
                in_require = True
                continue
            if in_require:
                if line == ")":
                    in_require = False
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    deps.append({"name": parts[0], "version": parts[1]})
            elif line.startswith("require "):
                parts = line.split()
                if len(parts) >= 3:
                    deps.append({"name": parts[1], "version": parts[2]})
        return deps

    def _parse_rust_deps(self) -> list[dict[str, str]]:
        """解析 Rust 项目依赖（Cargo.toml）。"""
        text = _safe_read_text(self._project_root / "Cargo.toml")
        if text is None:
            return []
        deps: list[dict[str, str]] = []
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            logger.warning(f"Cargo.toml TOML 解析失败: {exc}")
            return deps
        for name, spec in data.get("dependencies", {}).items():
            version = spec if isinstance(spec, str) else spec.get("version", "*") if isinstance(spec, dict) else "*"
            deps.append({"name": name, "version": str(version)})
        return deps


# ============================================================================
# Part C: ProjectAnalyzer — 整合分析器
# ============================================================================


class ProjectAnalyzer:
    """项目整合分析器。

    组合 TechStackDetector 和 ModuleAnalyzer，提供完整的项目分析功能。
    包括目录树生成、项目名称检测、关键文件查找和综合分析。

    Attributes:
        project_root: 项目根目录路径。
        config: Miaowa 应用配置。
        tech_stack: TechStackDetector 实例。
        modules: ModuleAnalyzer 实例。
    """

    def __init__(self, project_root: Path, config: Config) -> None:
        """初始化项目分析器。

        Args:
            project_root: 项目根目录的绝对路径。
            config: Miaowa 应用配置对象。
        """
        self._project_root = project_root
        self._config = config
        self.tech_stack = TechStackDetector(project_root)
        self.modules = ModuleAnalyzer(project_root, config)

    # ------------------------------------------------------------------
    # full_analysis
    # ------------------------------------------------------------------

    async def full_analysis(self) -> dict[str, Any]:
        """执行完整的项目分析。

        Returns:
            dict 包含:
                - project_name (str): 项目名称
                - project_root (str): 项目根目录路径
                - tech_stack (dict): 技术栈信息
                - directory_tree (str): tree 风格的目录文本
                - modules (list[dict]): 模块结构列表
                - dependencies (dict): 依赖映射
                - structure (str): 代码架构风格
                - key_files (list[str]): 关键文件路径
                - summary (str): 项目分析摘要文本
                - statistics (dict): 统计信息
        """
        logger.info(f"[ProjectAnalyzer] 开始完整分析: {self._project_root}")

        # 技术栈
        tech_stack = self.tech_stack.detect_all()

        # 无任何语言检测到时提前返回
        if not tech_stack["languages"]:
            project_name = self._detect_project_name()
            return {
                "project_name": project_name,
                "project_root": str(self._project_root),
                "tech_stack": tech_stack,
                "directory_tree": "",
                "modules": [],
                "dependencies": {},
                "structure": "未知",
                "key_files": [],
                "summary": (
                    f"项目「{project_name}」未检测到任何已知编程语言。"
                    f"请确认目录包含有效的项目文件。"
                ),
                "statistics": {"total_files": 0, "total_dirs": 0, "file_extensions": {}},
            }

        # 目录树（异步，避免阻塞事件循环）
        directory_tree = await self._generate_tree_async()
        logger.info("[ProjectAnalyzer] 目录树生成完成")

        # 以下密集 I/O 操作卸载到线程池，避免阻塞事件循环
        primary_lang = tech_stack["languages"][0]

        # 模块分析 + 依赖 + 结构（并行在线程池执行）
        modules_list, dependencies, structure = await asyncio.gather(
            asyncio.to_thread(self.modules.analyze_modules, primary_lang),
            asyncio.to_thread(self.modules.analyze_dependencies),
            asyncio.to_thread(self.modules.analyze_structure),
        )

        # 项目名称 + 关键文件（并行在线程池执行）
        project_name, key_files = await asyncio.gather(
            asyncio.to_thread(self._detect_project_name),
            asyncio.to_thread(self._find_key_files),
        )

        # 统计（rglob 遍历，大项目可能耗时）
        statistics = await asyncio.to_thread(self._compute_statistics)

        # 摘要（含依赖概览）
        summary = self._generate_summary(
            project_name, tech_stack, structure, statistics, dependencies,
        )

        logger.info(f"[ProjectAnalyzer] 完整分析完成: {project_name}")

        return {
            "project_name": project_name,
            "project_root": str(self._project_root),
            "tech_stack": tech_stack,
            "directory_tree": directory_tree,
            "modules": modules_list,
            "dependencies": dependencies,
            "structure": structure,
            "key_files": key_files,
            "summary": summary,
            "statistics": statistics,
        }

    # ------------------------------------------------------------------
    # _generate_tree
    # ------------------------------------------------------------------

    def _generate_tree(self, max_depth: int = 3) -> str:
        """生成 tree 风格的目录结构文本。

        使用 os.scandir() 减少系统调用，通过 asyncio.to_thread
        在线程中执行遍历以免阻塞事件循环。首行输出 "." 与 tree 命令一致。
        自动跳过 exclude_dirs 中的目录和隐藏文件。

        Args:
            max_depth: 最大递归深度，默认 3。

        Returns:
            格式化的树状文本字符串。
        """
        exclude_dirs: set[str] = set(self._config.project.exclude_dirs)
        lines: list[str] = ["."]

        def _walk(dir_path: str, prefix: str, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                with os.scandir(dir_path) as entries_raw:
                    # 排序：目录在前，文件在后，各自按字母
                    entries = sorted(
                        entries_raw,
                        key=lambda e: (not e.is_dir(), e.name.lower()),
                    )
            except PermissionError:
                return

            # 过滤隐藏文件和排除目录
            visible: list[os.DirEntry] = []
            for e in entries:
                if e.name.startswith("."):
                    continue
                if e.is_dir() and e.name in exclude_dirs:
                    continue
                visible.append(e)

            for i, entry in enumerate(visible):
                is_last = (i == len(visible) - 1)
                connector = "└── " if is_last else "├── "
                if entry.is_dir():
                    lines.append(f"{prefix}{connector}{entry.name}/")
                    extension = "    " if is_last else "│   "
                    _walk(entry.path, prefix + extension, depth + 1)
                else:
                    lines.append(f"{prefix}{connector}{entry.name}")

        _walk(str(self._project_root), "", 1)
        return "\n".join(lines)

    async def _generate_tree_async(self, max_depth: int = 3) -> str:
        """异步包装 _generate_tree，避免阻塞事件循环。"""
        return await asyncio.to_thread(self._generate_tree, max_depth)

    # ------------------------------------------------------------------
    # _detect_project_name
    # ------------------------------------------------------------------

    def _detect_project_name(self) -> str:
        """从配置文件中提取项目名称。

        检查优先级:
            1. pyproject.toml → [project].name 或 [tool.poetry].name
            2. package.json → .name
            3. Cargo.toml → [package].name
            4. go.mod → module 行第一个词的最后一段
            5. pom.xml → <artifactId>
            6. 回退到项目根目录名

        Returns:
            检测到的项目名称。
        """
        # pyproject.toml
        ppt_text = _safe_read_text(self._project_root / "pyproject.toml")
        if ppt_text:
            try:
                data = tomllib.loads(ppt_text)
                name = (
                    data.get("project", {}).get("name")
                    or data.get("tool", {}).get("poetry", {}).get("name")
                )
                if name:
                    return str(name)
            except tomllib.TOMLDecodeError as exc:
                logger.warning(f"pyproject.toml TOML 解析失败: {exc}")

        # package.json
        pkg_text = _safe_read_text(self._project_root / "package.json")
        if pkg_text:
            try:
                data = json.loads(pkg_text)
                name = data.get("name")
                if name:
                    return str(name)
            except json.JSONDecodeError as exc:
                logger.warning(f"package.json JSON 解析失败: {exc}")

        # Cargo.toml
        cargo_text = _safe_read_text(self._project_root / "Cargo.toml")
        if cargo_text:
            try:
                data = tomllib.loads(cargo_text)
                name = data.get("package", {}).get("name")
                if name:
                    return str(name)
            except tomllib.TOMLDecodeError as exc:
                logger.warning(f"Cargo.toml TOML 解析失败: {exc}")

        # go.mod — 取 module 路径最后一段
        go_text = _safe_read_text(self._project_root / "go.mod")
        if go_text:
            for line in go_text.splitlines():
                if line.startswith("module "):
                    module_path = line.split(maxsplit=1)[1].strip()
                    return module_path.rsplit("/", maxsplit=1)[-1]

        # pom.xml
        pom = self._project_root / "pom.xml"
        if pom.is_file():
            pom_text = _safe_read_text(pom)
            if pom_text:
                try:
                    tree = ElementTree.fromstring(pom_text)
                    ns = {"maven": "http://maven.apache.org/POM/4.0.0"}
                    artifact = tree.find("maven:artifactId", ns)
                    if artifact is None:
                        artifact = tree.find("artifactId")
                    if artifact is not None and artifact.text:
                        return artifact.text.strip()
                except ElementTree.ParseError as exc:
                    logger.warning(f"pom.xml XML 解析失败: {exc}")

        # 回退：目录名
        return self._project_root.name

    # ------------------------------------------------------------------
    # _find_key_files
    # ------------------------------------------------------------------

    def _find_key_files(self) -> list[str]:
        """查找项目中的关键文件。

        搜索以下类型的文件:
            - README 类文件
            - CHANGELOG 类文件
            - 主配置文件
            - 入口文件

        Returns:
            相对于项目根目录的关键文件路径列表。
        """
        key_names = {
            "readme.md", "readme.rst", "readme.txt", "README.md", "README.rst", "README",
            "CHANGELOG.md", "CHANGELOG.rst", "CHANGELOG", "HISTORY.md",
            "LICENSE", "LICENSE.md", "LICENSE.txt",
            "CONTRIBUTING.md", "CONTRIBUTING.rst",
            "pyproject.toml", "setup.py", "setup.cfg",
            "package.json", "tsconfig.json",
            "go.mod", "Cargo.toml", "Makefile", "Dockerfile",
            ".github", ".gitignore", ".env.example", ".editorconfig",
        }

        key_files: list[str] = []
        for name in key_names:
            candidate = self._project_root / name
            if candidate.exists():
                key_files.append(name)

        # 入口文件
        entry_candidates = [
            "src/main.py", "src/app.py", "src/index.js", "src/index.ts",
            "main.py", "app.py", "index.js", "index.ts",
            "cmd/main.go", "main.go",
            "src/main.rs", "main.rs",
        ]
        for ec in entry_candidates:
            if (self._project_root / ec).is_file():
                key_files.append(ec)
                break  # 只取第一个匹配的入口文件

        return sorted(key_files)

    # ------------------------------------------------------------------
    # _compute_statistics
    # ------------------------------------------------------------------

    def _compute_statistics(self) -> dict[str, Any]:
        """计算项目文件统计信息。

        Returns:
            dict 包含:
                - total_files (int): 文件总数
                - total_dirs (int): 目录总数
                - file_extensions (dict[str, int]): 文件扩展名分布
        """
        exclude_dirs: set[str] = set(self._config.project.exclude_dirs)
        ext_counter: dict[str, int] = {}
        total_files = 0
        total_dirs = 0

        for entry in self._project_root.rglob("*"):
            # 跳过排除目录
            parts = set(entry.relative_to(self._project_root).parts)
            if parts & exclude_dirs:
                continue

            if entry.is_dir():
                if not entry.name.startswith("."):
                    total_dirs += 1
            elif entry.is_file():
                if not entry.name.startswith("."):
                    total_files += 1
                    ext = entry.suffix or "(no extension)"
                    ext_counter[ext] = ext_counter.get(ext, 0) + 1

        # 扩展名按数量降序
        ext_sorted = dict(
            sorted(ext_counter.items(), key=lambda kv: -kv[1])
        )

        return {
            "total_files": total_files,
            "total_dirs": total_dirs,
            "file_extensions": ext_sorted,
        }

    # ------------------------------------------------------------------
    # _generate_summary
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_summary(
        project_name: str,
        tech_stack: dict[str, Any],
        structure: str,
        statistics: dict[str, Any],
        dependencies: dict[str, list[dict[str, str]]] | None = None,
    ) -> str:
        """生成人类可读的项目分析摘要。

        Args:
            project_name: 项目名称。
            tech_stack: 技术栈检测结果。
            structure: 代码架构风格。
            statistics: 文件统计信息。
            dependencies: 依赖映射（{生态: [{name, version}]}）。

        Returns:
            多行中文摘要文本。
        """
        parts: list[str] = []
        parts.append(f"项目「{project_name}」分析摘要")

        # 技术栈
        langs = tech_stack.get("languages", [])
        if langs:
            parts.append(f"主要语言: {', '.join(langs)}")

        frameworks = tech_stack.get("frameworks", {})
        if frameworks:
            fw_list: list[str] = []
            for lang, fws in frameworks.items():
                fw_list.append(f"{lang}: {', '.join(fws)}")
            parts.append(f"框架: {'; '.join(fw_list)}")

        build_tools = tech_stack.get("build_tools", [])
        if build_tools:
            parts.append(f"构建工具: {', '.join(build_tools)}")

        pm = tech_stack.get("package_manager")
        if pm:
            parts.append(f"包管理器: {pm}")

        # 架构
        parts.append(f"代码架构: {structure}")

        # 统计
        parts.append(
            f"项目规模: {statistics.get('total_files', 0)} 个文件, "
            f"{statistics.get('total_dirs', 0)} 个目录"
        )

        # 依赖概览
        if dependencies:
            dep_counts = [
                f"{eco}: {len(items)} 个"
                for eco, items in dependencies.items()
            ]
            parts.append(f"依赖项: {', '.join(dep_counts)}")

        return "\n".join(parts)


# ============================================================================
# Part D: AnalyzeProjectTool — BaseTool 对外接口
# ============================================================================


class AnalyzeProjectTool(BaseTool):
    """项目分析工具（BaseTool 接口）。

    提供按需分析项目的能力，通过 aspect 参数控制分析深度：
        - overview:  完整分析（含目录树和摘要）
        - tech_stack: 仅技术栈
        - modules:   仅模块结构
        - dependencies: 仅依赖
        - structure: 仅代码架构

    Attributes:
        name: 工具名称 "analyze_project"。
        description: 工具功能描述。
        parameters: 工具参数定义列表。
    """

    name = "analyze_project"
    description = (
        "分析项目结构，检测技术栈、框架、构建工具、项目架构风格、"
        "模块组织和依赖关系。可通过 aspect 参数按需选择分析深度。"
    )
    parameters = [
        ToolParameter(
            name="aspect",
            type="string",
            description="分析维度。overview=完整分析，tech_stack=技术栈，"
            "modules=模块结构，dependencies=依赖关系，structure=代码架构",
            required=False,
            default="overview",
            enum=["overview", "tech_stack", "modules", "dependencies", "structure"],
        ),
    ]

    def __init__(self, project_root: Path, config: Config) -> None:
        """初始化 AnalyzeProjectTool。

        ProjectAnalyzer 在初始化时创建一次，后续 execute() 调用复用。
        分析结果缓存在实例的 _cache 字典中，同一 aspect 的重复调用
        直接返回缓存结果，避免重复分析。

        Args:
            project_root: 项目根目录的绝对路径。
            config: Miaowa 应用配置对象。
        """
        self._project_root = project_root
        self._config = config
        self._analyzer = ProjectAnalyzer(project_root, config)
        self._cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行项目分析（带实例级缓存）。

        根据 aspect 参数选择分析维度，使用缓存的 ProjectAnalyzer 实例。
        同一 aspect 的后续调用直接返回缓存结果。

        Args:
            **kwargs:
                - aspect (str): 分析维度，默认 "overview"。

        Returns:
            ToolResult:
                - success=True: data 包含对应维度的分析结果。
                - success=False: error 为错误描述。
        """
        aspect: str = kwargs.get("aspect", "overview")

        # 缓存命中
        if aspect in self._cache:
            logger.info(f"[AnalyzeProjectTool] 缓存命中: aspect={aspect}")
            return ToolResult.ok(self._cache[aspect])

        logger.info(f"[AnalyzeProjectTool] 执行分析: aspect={aspect}")

        try:
            if aspect == "overview":
                result_data = await self._analyzer.full_analysis()
            elif aspect == "tech_stack":
                result_data = self._analyzer.tech_stack.detect_all()
            elif aspect == "modules":
                # 仅检测语言（不执行完整 detect_all）
                langs = self._analyzer.tech_stack.detect_languages()
                primary = langs[0] if langs else "未知"
                result_data = {
                    "language": primary,
                    "modules": self._analyzer.modules.analyze_modules(primary),
                }
            elif aspect == "dependencies":
                result_data = self._analyzer.modules.analyze_dependencies()
            elif aspect == "structure":
                result_data = {
                    "structure": self._analyzer.modules.analyze_structure(),
                }
            else:
                return ToolResult.fail(
                    f"未知的分析维度: {aspect!r}。"
                    f"可选值: overview, tech_stack, modules, dependencies, structure"
                )

            # 存入缓存
            self._cache[aspect] = result_data
            return ToolResult.ok(result_data)

        except Exception as exc:
            logger.error(f"[AnalyzeProjectTool] 分析失败: {exc}")
            return ToolResult.fail(
                f"项目分析失败: {type(exc).__name__} — {exc}"
            )
