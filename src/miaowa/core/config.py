"""配置管理模块 — 多层配置加载、校验与合并。

配置优先级（由低到高）::

    默认值  <  config.yaml  <  .env  <  环境变量  <  CLI 参数

与 PRD §3.4.1 5 级优先级完全一致：
    1. CLI 参数（最高）
    2. 真实环境变量
    3. .env 文件（项目根目录 > 当前目录，不影响已有环境变量）
    4. 用户配置文件 ~/.miaowa/config.yaml（支持 ${ENV_VAR} 引用）
    5. 系统默认值（代码内硬编码）

YAML 文件搜索顺序:
    项目级: ./config.yaml, ./miaowa.yaml, ./.miaowa.yaml（含 .yml 变体）
    用户级: ~/.miaowa/config.yaml, ~/.miaowa/miaowa.yaml

Typical usage::

    config = ConfigManager.load(cli_overrides={"log_level": "DEBUG"})
    ConfigManager.validate(config)
    # config.llm.api_key 已从 MIAOWA_API_KEY 环境变量填充
"""

from __future__ import annotations

import os
import re
import typing
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml
from dotenv import load_dotenv

from miaowa.core.exceptions import ConfigFormatError, ConfigMissingError

# ---------------------------------------------------------------------------
# 默认常量（不可变 tuple；dataclass 字段使用可变 list 副本）
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = (
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".tox",
    "egg-info",
    ".eggs",
)

DEFAULT_BINARY_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".pyc",
    ".pyo",
    ".class",
    ".o",
    ".a",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
)

# ---------------------------------------------------------------------------
# 子配置 dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """LLM 相关配置。

    Attributes:
        provider: LLM 提供商名称，如 "deepseek"、"openai"。
        api_key: API 密钥（优先从环境变量 MIAOWA_API_KEY 加载）。
        base_url: API 基础地址。
            OpenAI SDK 直接在此 URL 后追加端点路径（如 /chat/completions），
            因此本字段需要包含 /v1 前缀以匹配 DeepSeek API 的实际路径。
            如 DeepSeek: https://api.deepseek.com/v1
            → 实际请求: https://api.deepseek.com/v1/chat/completions
        model: 默认模型名称。
        temperature: 生成温度，范围 (0, 2]，默认 0.3。
        max_tokens: 单次响应最大 token 数，默认 4096。
        timeout: HTTP 请求超时时间（秒），默认 120。
    """

    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 120


@dataclass
class UIConfig:
    """终端 UI 相关配置。

    Attributes:
        theme: 终端界面主题，可选 "dark"、"light"、"auto"。
        syntax_theme: 代码块语法高亮主题（Rich / Pygments 主题名）。
        max_history: REPL 命令历史最大条数。
    """

    theme: str = "dark"
    syntax_theme: str = "monokai"
    max_history: int = 1000


@dataclass
class ProjectConfig:
    """项目分析和文件操作相关配置。

    Attributes:
        exclude_dirs: 扫描与搜索时排除的目录列表。
        max_file_size: 允许读取的文件最大字节数（默认 1 MB）。
        binary_extensions: 视为二进制文件的扩展名列表。
        use_gitignore: 是否使用 .gitignore 规则过滤文件。
            默认 True。当项目根目录无 .gitignore 文件时自动退化为无过滤。
    """

    exclude_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_DIRS))
    max_file_size: int = 1_048_576  # 1 MB
    binary_extensions: list[str] = field(
        default_factory=lambda: list(DEFAULT_BINARY_EXTENSIONS)
    )
    use_gitignore: bool = True


@dataclass
class ToolsConfig:
    """内置工具相关配置。

    Attributes:
        read_file_max_lines: read_file 工具每次读取的最大行数。
        search_max_results: search 工具单次返回的最大结果数。
        shell_timeout: shell 命令执行超时时间（秒），默认 300。
        shell_sandbox: 是否启用沙箱模式执行 shell 命令。
    """

    read_file_max_lines: int = 2000
    search_max_results: int = 50
    shell_timeout: int = 300
    shell_sandbox: bool = False


@dataclass
class LoggingConfig:
    """日志相关配置。

    Attributes:
        level: 控制台日志级别（DEBUG / INFO / WARNING / ERROR / CRITICAL）。
            默认 INFO，开发时可设为 DEBUG。
        file_level: 文件日志级别，默认 DEBUG（记录所有细节）。
        file: 日志文件路径，支持 ~ 展开。
        max_size: 单个日志文件大小上限（支持 "10MB"、"500KB" 格式）。
        backup_count: 日志文件保留轮转数量。
        format: 日志格式字符串（loguru 风格）。
    """

    level: str = "WARNING"
    file_level: str = "DEBUG"
    file: str = "~/.miaowa/logs/miaowa.log"
    max_size: str = "10MB"
    backup_count: int = 3
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )


# ---------------------------------------------------------------------------
# 总配置 dataclass
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Miaowa 应用总配置。

    聚合所有子配置模块，由 ConfigManager.load() 构造。

    Attributes:
        llm: LLM 相关配置。
        ui: 终端 UI 相关配置。
        project: 项目分析配置。
        tools: 内置工具配置。
        logging: 日志配置。
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# ConfigManager
# ---------------------------------------------------------------------------

# 匹配 ${VAR_NAME} 和 ${VAR_NAME:-default_value}
_ENV_VAR_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


class ConfigManager:
    """配置管理器 — 多层配置加载、合并与校验。

    与 PRD §3.4.1 的 5 级优先级完全对齐：

        CLI 参数  >  环境变量  >  .env 文件  >  config.yaml  >  默认值

    使用方式::

        config = ConfigManager.load(cli_overrides={"log_level": "DEBUG"})
        ConfigManager.validate(config)  # 检查 api_key 等必填项
    """

    # -- 配置文件搜索路径（PRD §3.4.1）--------------------------------------

    CONFIG_FILE_NAMES: ClassVar[tuple[str, ...]] = (
        "config.yaml",
        "config.yml",
        "miaowa.yaml",
        "miaowa.yml",
        ".miaowa.yaml",
        ".miaowa.yml",
    )

    SEARCH_DIRS: ClassVar[tuple[str, ...]] = (
        ".",            # 项目级
        "~/.miaowa",    # 用户级 — 与 PRD 路径一致
    )

    # -- 环境变量 → 配置字段映射 --------------------------------------------

    ENV_MAP: ClassVar[dict[str, tuple[str, str]]] = {
        # LLM
        "MIAOWA_API_KEY": ("llm", "api_key"),
        "MIAOWA_BASE_URL": ("llm", "base_url"),
        "MIAOWA_MODEL": ("llm", "model"),
        "MIAOWA_TEMPERATURE": ("llm", "temperature"),
        "MIAOWA_MAX_TOKENS": ("llm", "max_tokens"),
        "MIAOWA_TIMEOUT": ("llm", "timeout"),
        # UI
        "MIAOWA_THEME": ("ui", "theme"),
        "MIAOWA_SYNTAX_THEME": ("ui", "syntax_theme"),
        "MIAOWA_MAX_HISTORY": ("ui", "max_history"),
        # Project
        "MIAOWA_MAX_FILE_SIZE": ("project", "max_file_size"),
        "MIAOWA_USE_GITIGNORE": ("project", "use_gitignore"),
        # Tools
        "MIAOWA_READ_MAX_LINES": ("tools", "read_file_max_lines"),
        "MIAOWA_SEARCH_MAX_RESULTS": ("tools", "search_max_results"),
        "MIAOWA_SHELL_TIMEOUT": ("tools", "shell_timeout"),
        # Logging
        "MIAOWA_LOG_LEVEL": ("logging", "level"),
        "MIAOWA_LOG_FILE_LEVEL": ("logging", "file_level"),
        "MIAOWA_LOG_FILE": ("logging", "file"),
    }

    # -- 缓存 ----------------------------------------------------------------

    _cached_config: ClassVar[Config | None] = None

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        cli_overrides: dict[str, Any] | None = None,
        *,
        use_cache: bool = False,
    ) -> Config:
        """按优先级加载并合并所有配置源。

        加载顺序（对应 PRD §3.4.1 的 5 级优先级）:
            1. 构建默认 Config
            2. 查找并加载 config.yaml / miaowa.yaml
            3. 加载 .env 文件（项目根目录 > 当前目录）
            4. 应用环境变量覆盖（含 .env 中已加载的变量）
            5. 应用 CLI 参数覆盖

        Args:
            cli_overrides: CLI 传入的配置覆盖字典。
                键可以是顶层字段名（如 ``"log_level"``）或
                点号分隔的嵌套路径（如 ``"llm.model"``）。
            use_cache: 是否使用上一次 load() 的缓存结果。
                默认 False（每次重新加载）；设为 True 可避免重复 I/O。

        Returns:
            合并后的 Config 实例。

        Raises:
            ConfigFormatError: YAML / .env 解析失败时。
            ConfigMissingError: YAML 中引用了未定义的环境变量且无默认值时。
        """
        if use_cache and cls._cached_config is not None:
            return cls._cached_config

        config = cls._build_defaults()

        # -- 1. YAML 配置文件 ------------------------------------------
        yaml_path = cls._find_config_file()
        if yaml_path is not None:
            yaml_data = cls._load_yaml(yaml_path)
            config = cls._merge_dict(config, yaml_data)

        # -- 2. .env 文件 ----------------------------------------------
        cls._load_dotenv()

        # -- 3. 环境变量 ------------------------------------------------
        config = cls._apply_env_overrides(config)

        # -- 4. CLI 参数 ------------------------------------------------
        if cli_overrides:
            config = cls._apply_cli_overrides(config, cli_overrides)

        cls._cached_config = config
        return config

    @classmethod
    def load_default(cls) -> Config:
        """加载纯默认配置（跳过所有外部配置源）。

        用于测试环境或无须外部配置的简单场景。
        """
        return cls._build_defaults()

    @classmethod
    def load_from_dict(cls, data: dict[str, Any]) -> Config:
        """从字典构造 Config（跳过文件 / 环境变量加载）。

        用于测试或编程式配置。

        Args:
            data: 扁平或嵌套的配置字典，键名与 Config 字段对应。

        Returns:
            合并后的 Config 实例。
        """
        config = cls._build_defaults()
        return cls._merge_dict(config, data)

    @staticmethod
    def validate(config: Config) -> None:
        """校验配置完整性。

        当前校验规则:
            - llm.api_key 不能为空字符串或未展开的变量占位符。
            - llm.temperature 在 (0, 2] 范围内。
            - llm.max_tokens 必须 > 0。

        Args:
            config: 待校验的 Config 实例。

        Raises:
            ConfigMissingError: api_key 为空或为未展开的 ``${...}`` 占位符时。
            ConfigFormatError: temperature / max_tokens 值不合法时。
        """
        api_key = config.llm.api_key

        # api_key 必填校验
        if not api_key:
            raise ConfigMissingError(
                "LLM API Key 未配置。请设置环境变量 MIAOWA_API_KEY "
                "或在 YAML 配置文件 llm.api_key 字段中填写",
                key_name="llm.api_key",
            )

        # 检测未展开的 ${VAR} 占位符（环境变量未定义且无默认值时残留）
        if _ENV_VAR_RE.search(api_key):
            raise ConfigMissingError(
                f"llm.api_key 包含未展开的环境变量引用: {api_key!r}。"
                f"请设置对应的环境变量或使用 ${{VAR:-default}} 语法提供默认值",
                key_name="llm.api_key",
            )

        # temperature 范围校验
        if not (0 < config.llm.temperature <= 2):
            raise ConfigFormatError(
                f"llm.temperature 必须在 (0, 2] 范围内，当前值: {config.llm.temperature}",
            )

        # max_tokens 合法性校验
        if config.llm.max_tokens <= 0:
            raise ConfigFormatError(
                f"llm.max_tokens 必须大于 0，当前值: {config.llm.max_tokens}",
            )

    @classmethod
    def ensure_config_dir(cls) -> Path:
        """确保 ~/.miaowa/ 目录存在，返回其 Path。

        可用于日志文件、配置文件、会话数据的初始化。

        Returns:
            ~/.miaowa/ 的 Path 对象（目录已确保存在）。
        """
        p = Path("~/.miaowa").expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def expand_path(path: str) -> Path:
        """展开 ~ 并返回 Path 对象。

        Args:
            path: 可能包含 ~ 的路径字符串。

        Returns:
            展开后的绝对 Path。
        """
        return Path(path).expanduser().resolve()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @classmethod
    def _build_defaults(cls) -> Config:
        """构造包含所有默认值的 Config 实例。"""
        return Config()

    # -- 配置文件查找 ---------------------------------------------------

    @classmethod
    def _find_config_file(cls) -> Path | None:
        """在 SEARCH_DIRS 中查找 CONFIG_FILE_NAMES，返回首个匹配的路径。

        搜索顺序:
            1. ./config.yaml, ./miaowa.yaml, ./.miaowa.yaml ...（项目级）
            2. ~/.miaowa/config.yaml, ~/.miaowa/miaowa.yaml ...（用户级）
        """
        for search_dir in cls.SEARCH_DIRS:
            base = Path(search_dir).expanduser().resolve()
            if not base.is_dir():
                continue
            for fname in cls.CONFIG_FILE_NAMES:
                candidate = base / fname
                if candidate.is_file():
                    return candidate
        return None

    # -- YAML 加载 -----------------------------------------------------

    @classmethod
    def _load_yaml(cls, path: Path) -> dict[str, Any]:
        """加载 YAML 文件并解析 ${ENV_VAR} 引用。

        Args:
            path: YAML 文件路径。

        Returns:
            解析后的配置字典。

        Raises:
            ConfigFormatError: YAML 语法错误或文件不可读时。
            ConfigMissingError: 环境变量引用未定义且无默认值时。
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigFormatError(
                f"无法读取配置文件: {path} — {exc}",
                file_path=str(path),
            ) from exc

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigFormatError(
                f"配置文件 YAML 解析失败: {path} — {exc}",
                file_path=str(path),
            ) from exc

        if not isinstance(data, dict):
            raise ConfigFormatError(
                f"配置文件顶层必须是映射 (dict)，实际类型: {type(data).__name__}",
                file_path=str(path),
            )

        return typing.cast(dict[str, Any], cls._resolve_env_vars(data))

    @classmethod
    def _resolve_env_vars(cls, data: Any) -> Any:
        """递归解析数据中的 ${VAR} 和 ${VAR:-default} 占位符。

        Args:
            data: 任意嵌套的配置数据（dict / list / str / 其他）。

        Returns:
            替换后的数据。

        Raises:
            ConfigMissingError: 遇到未定义且无默认值的环境变量引用时。
        """
        if isinstance(data, dict):
            return {k: cls._resolve_env_vars(v) for k, v in data.items()}
        if isinstance(data, list):
            return [cls._resolve_env_vars(item) for item in data]
        if isinstance(data, str):
            return cls._expand_env_var(data)
        return data

    @classmethod
    def _expand_env_var(cls, value: str) -> str:
        """展开字符串中的 ${VAR} / ${VAR:-default} 模式。

        环境变量已定义 → 替换为变量值；
        环境变量未定义但有默认值 → 替换为默认值；
        环境变量未定义且无默认值 → 发出警告并保留原样，
        后续 ``validate()`` 阶段会拒绝 ``api_key`` 等关键字段中的未展开占位符。

        返回类型始终为 str；不执行类型推断（如 ``"${PORT:-8080}"`` 返回 ``"8080"``，
        由调用方按需转换）。
        """

        def _replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            # 未定义且无默认值：记录警告，保留原样供 validate 检测
            warnings.warn(
                f"配置文件中引用了未定义的环境变量: ${{{var_name}}}，"
                f"且未提供默认值（${var_name}:-default）。"
                f"该占位符将被保留，可能导致后续校验失败。",
                RuntimeWarning,
                stacklevel=3,
            )
            return match.group(0)

        # 若整个字符串就是一个变量引用，直接返回展开结果
        if _ENV_VAR_RE.fullmatch(value.strip()):
            return _ENV_VAR_RE.sub(_replacer, value)
        return _ENV_VAR_RE.sub(_replacer, value)

    # -- .env 加载 -----------------------------------------------------

    @classmethod
    def _find_project_root(cls) -> Path | None:
        """向上搜索项目根目录。

        判定标准：包含 .git 或 pyproject.toml 的目录。
        最多向上搜索 10 层；到达文件系统根目录时停止。

        Returns:
            项目根路径，未找到则返回 None。
        """
        current = Path.cwd().resolve()
        for _ in range(10):
            if (current / ".git").is_dir() or (current / "pyproject.toml").is_file():
                return current
            parent = current.parent
            if parent == current:  # 到达文件系统根
                break
            current = parent
        return None

    @classmethod
    def _load_dotenv(cls) -> None:
        """加载 .env 文件（静默：文件不存在时不报错）。

        搜索优先级:
            1. 项目根目录的 .env（通过 .git / pyproject.toml 定位）
            2. 当前工作目录的 .env
            3. python-dotenv 自动搜索
        """
        # 1. 项目根目录
        root = cls._find_project_root()
        if root is not None:
            env_file = root / ".env"
            if env_file.is_file():
                load_dotenv(dotenv_path=str(env_file), override=False)
                return

        # 2. 当前目录
        cwd_env = Path.cwd() / ".env"
        if cwd_env.is_file():
            load_dotenv(dotenv_path=str(cwd_env), override=False)
            return

        # 3. python-dotenv 自动搜索
        load_dotenv(override=False)

    # -- 环境变量覆盖 --------------------------------------------------

    @classmethod
    def _apply_env_overrides(cls, config: Config) -> Config:
        """将 ENV_MAP 中注册的环境变量覆盖到 Config 对应字段。

        ``load_dotenv(override=False)`` 确保真实环境变量优先级高于 .env 值
        （PRD 第 2 级 > 第 3 级）。
        """
        for env_name, (section, field_name) in cls.ENV_MAP.items():
            raw = os.environ.get(env_name)
            if raw is None:
                continue

            section_obj = getattr(config, section)
            field_type = cls._field_type(Config, section, field_name)
            converted = cls._coerce(raw, field_type, env_name=env_name)

            setattr(section_obj, field_name, converted)

        return config

    # -- CLI 参数覆盖 --------------------------------------------------

    @classmethod
    def _apply_cli_overrides(
        cls, config: Config, overrides: dict[str, Any]
    ) -> Config:
        """将 CLI 参数覆盖到 Config。

        支持两种键格式:
            - 顶层字段: ``{"log_level": "DEBUG"}``  → ``logging.level``
            - 嵌套路径: ``{"llm.model": "deepseek-v3"}`` → ``config.llm.model``
        """
        # 顶层字段名 → (section, field) 映射
        flat_to_nested: dict[str, tuple[str, str]] = {
            "log_level": ("logging", "level"),
            "log_file_level": ("logging", "file_level"),
            "log_file": ("logging", "file"),
            "api_key": ("llm", "api_key"),
            "base_url": ("llm", "base_url"),
            "model": ("llm", "model"),
            "temperature": ("llm", "temperature"),
            "max_tokens": ("llm", "max_tokens"),
            "timeout": ("llm", "timeout"),
            "theme": ("ui", "theme"),
            "syntax_theme": ("ui", "syntax_theme"),
            "max_history": ("ui", "max_history"),
            "max_file_size": ("project", "max_file_size"),
            "use_gitignore": ("project", "use_gitignore"),
            "read_max_lines": ("tools", "read_file_max_lines"),
            "search_max_results": ("tools", "search_max_results"),
            "shell_timeout": ("tools", "shell_timeout"),
        }

        for key, value in overrides.items():
            # 嵌套路径（如 "llm.model"）
            if "." in key:
                parts = key.split(".", maxsplit=1)
                if len(parts) == 2:
                    section_name, field_name = parts
                    if hasattr(config, section_name):
                        section = getattr(config, section_name)
                        if hasattr(section, field_name):
                            setattr(section, field_name, value)
                continue

            # 扁平顶层字段
            if key in flat_to_nested:
                section_name, field_name = flat_to_nested[key]
                section = getattr(config, section_name)
                setattr(section, field_name, value)

        return config

    # -- 字典合并 ------------------------------------------------------

    @classmethod
    def _merge_dict(cls, config: Config, data: dict[str, Any]) -> Config:
        """将嵌套字典递归合并到 Config 的对应子配置中。

        Args:
            config: 当前 Config 实例。
            data: 待合并的字典，顶层键对应 Config 的子配置名称
                （如 ``"llm"``、``"ui"``、``"project"``、``"tools"``、``"logging"``）。

        Returns:
            合并后的 Config（原地修改）。

        Note:
            list 类型字段（如 exclude_dirs、binary_extensions）使用
            **合并追加**（扩展默认值）而非完全替换。
            标量字段（str / int / float / bool）直接覆盖。
        """
        for section_name, section_data in data.items():
            if not hasattr(config, section_name):
                continue
            section = getattr(config, section_name)
            if isinstance(section_data, dict) and section is not None:
                for key, value in section_data.items():
                    if hasattr(section, key):
                        # 对 list 类型字段做合并而非替换
                        existing = getattr(section, key)
                        if isinstance(existing, list) and isinstance(value, list):
                            # 合并去重，保持顺序（用户值追加在默认值之后）
                            merged = list(dict.fromkeys(existing + value))
                            setattr(section, key, merged)
                        else:
                            setattr(section, key, value)
        return config

    # -- 辅助工具 ------------------------------------------------------

    @staticmethod
    def _field_type(
        root_cls: type, section_name: str, field_name: str
    ) -> type:
        """查询嵌套 dataclass 中某个字段的运行时类型。

        使用 ``typing.get_type_hints()`` 而非 ``__dataclass_fields__``，
        因为 ``from __future__ import annotations`` 环境下后者返回的是字符串。

        Args:
            root_cls: 根 dataclass 类型（Config）。
            section_name: 子配置名（如 ``"llm"``）。
            field_name: 字段名（如 ``"temperature"``）。

        Returns:
            字段类型，无法确定时返回 str。
        """
        try:
            root_hints = typing.get_type_hints(root_cls)
            section_cls = root_hints[section_name]
            section_hints = typing.get_type_hints(section_cls)
            return typing.cast(type, section_hints[field_name])
        except (KeyError, AttributeError, TypeError):
            return str

    @staticmethod
    def _coerce(
        raw: str, target_type: type, *, env_name: str = "<unknown>"
    ) -> Any:
        """将字符串环境变量值转换为目标类型。

        支持: str, int, float, bool（``"1"`` / ``"true"`` / ``"yes"`` / ``"on"`` 为 True）。

        Args:
            raw: 原始字符串值。
            target_type: 目标 Python 类型。
            env_name: 环境变量名（用于错误消息）。

        Returns:
            转换后的值。

        Raises:
            ConfigFormatError: 值无法转换为目标类型时。
        """
        if target_type is str:
            return raw
        if target_type is int:
            try:
                return int(raw)
            except ValueError as err:
                raise ConfigFormatError(
                    f"环境变量 {env_name} 的值应为整数，实际值: {raw!r}"
                ) from err
        if target_type is float:
            try:
                return float(raw)
            except ValueError as err:
                raise ConfigFormatError(
                    f"环境变量 {env_name} 的值应为浮点数，实际值: {raw!r}"
                ) from err
        if target_type is bool:
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return raw
