"""ConfigManager 单元测试。

覆盖配置加载的所有路径：默认值、环境变量、YAML 文件、
${ENV_VAR} 展开、CLI 覆盖、校验、异常处理、.env 加载。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from miaowa.core.config import (
    DEFAULT_BINARY_EXTENSIONS,
    DEFAULT_EXCLUDE_DIRS,
    Config,
    ConfigManager,
)
from miaowa.core.exceptions import ConfigFormatError, ConfigMissingError


# ---------------------------------------------------------------------------
# 辅助 fixture：阻断 load() 中的文件 / dotenv 探测
# ---------------------------------------------------------------------------


def _isolate_load(mocker, *, clear_env: bool = False):
    """让 ConfigManager.load() 不触碰真实文件系统和外部环境变量。

    - mock _find_config_file（跳过 YAML）
    - mock _load_dotenv（跳过 .env）
    - 当 clear_env=True 时，额外 mock _apply_env_overrides 为透传，
      完全隔离外部环境变量，防止开发者 Shell 中的 MIAOWA_* 变量污染测试。
    """
    mocker.patch.object(ConfigManager, "_find_config_file", return_value=None)
    mocker.patch.object(ConfigManager, "_load_dotenv")
    if clear_env:
        mocker.patch.object(
            ConfigManager,
            "_apply_env_overrides",
            side_effect=lambda config: config,
        )


# ============================================================================
# 1. test_load_default_config — 默认值正确
# ============================================================================


class TestLoadDefaultConfig:
    """验证 load_default() 返回的 Config 各字段均为代码内默认值。"""

    def test_load_default_returns_config_instance(self):
        """load_default() 返回 Config 实例。"""
        config = ConfigManager.load_default()
        assert isinstance(config, Config)

    def test_llm_section_defaults(self):
        """LLM 子配置默认值。"""
        llm = ConfigManager.load_default().llm
        assert llm.provider == "deepseek"
        assert llm.api_key == ""
        assert llm.base_url == "https://api.deepseek.com/v1"
        assert llm.model == "deepseek-v4-flash"
        assert llm.temperature == 0.3
        assert llm.max_tokens == 4096
        assert llm.timeout == 120

    def test_ui_section_defaults(self):
        """UI 子配置默认值。"""
        ui = ConfigManager.load_default().ui
        assert ui.theme == "dark"
        assert ui.syntax_theme == "monokai"
        assert ui.max_history == 1000

    def test_project_section_defaults(self):
        """Project 子配置默认值（含排除目录与二进制扩展名）。"""
        project = ConfigManager.load_default().project
        assert isinstance(project.exclude_dirs, list)
        assert "node_modules" in project.exclude_dirs
        assert ".git" in project.exclude_dirs
        assert project.max_file_size == 1_048_576
        assert isinstance(project.binary_extensions, list)
        assert ".png" in project.binary_extensions
        assert ".exe" in project.binary_extensions

    def test_tools_section_defaults(self):
        """Tools 子配置默认值。"""
        tools = ConfigManager.load_default().tools
        assert tools.read_file_max_lines == 2000
        assert tools.search_max_results == 50
        assert tools.shell_timeout == 300
        assert tools.shell_sandbox is False

    def test_logging_section_defaults(self):
        """Logging 子配置默认值。"""
        logging_cfg = ConfigManager.load_default().logging
        assert logging_cfg.level == "WARNING"
        assert logging_cfg.file_level == "DEBUG"
        assert logging_cfg.file == "~/.miaowa/logs/miaowa.log"
        assert logging_cfg.max_size == "10MB"
        assert logging_cfg.backup_count == 3

    def test_consecutive_calls_return_independent_objects(self):
        """连续调用 load_default() 返回不同 Config 实例。"""
        c1 = ConfigManager.load_default()
        c2 = ConfigManager.load_default()
        assert c1 is not c2
        # 修改一个不影响另一个
        c1.llm.model = "modified"
        assert c2.llm.model == "deepseek-v4-flash"


# ============================================================================
# 2. test_load_from_env_vars — 环境变量覆盖
# ============================================================================


class TestLoadFromEnvVars:
    """验证 ENV_MAP 中注册的环境变量能正确覆盖 Config 对应字段。"""

    def test_env_var_overrides_model(self, monkeypatch, mocker):
        """MIAOWA_MODEL 环境变量覆盖 llm.model。"""
        _isolate_load(mocker)
        monkeypatch.setenv("MIAOWA_MODEL", "env-model-v2")

        config = ConfigManager.load()
        assert config.llm.model == "env-model-v2"

    def test_env_var_overrides_temperature_float(self, monkeypatch, mocker):
        """MIAOWA_TEMPERATURE 字符串转 float。"""
        _isolate_load(mocker)
        monkeypatch.setenv("MIAOWA_TEMPERATURE", "0.7")

        config = ConfigManager.load()
        assert config.llm.temperature == 0.7
        assert isinstance(config.llm.temperature, float)

    def test_env_var_overrides_max_tokens_int(self, monkeypatch, mocker):
        """MIAOWA_MAX_TOKENS 字符串转 int。"""
        _isolate_load(mocker)
        monkeypatch.setenv("MIAOWA_MAX_TOKENS", "8192")

        config = ConfigManager.load()
        assert config.llm.max_tokens == 8192
        assert isinstance(config.llm.max_tokens, int)

    def test_env_var_overrides_theme(self, monkeypatch, mocker):
        """MIAOWA_THEME 环境变量覆盖 ui.theme。"""
        _isolate_load(mocker)
        monkeypatch.setenv("MIAOWA_THEME", "light")

        config = ConfigManager.load()
        assert config.ui.theme == "light"

    def test_env_var_overrides_log_level(self, monkeypatch, mocker):
        """MIAOWA_LOG_LEVEL 环境变量覆盖 logging.level。"""
        _isolate_load(mocker)
        monkeypatch.setenv("MIAOWA_LOG_LEVEL", "DEBUG")

        config = ConfigManager.load()
        assert config.logging.level == "DEBUG"

    def test_unset_env_var_does_not_override(self, mocker):
        """未设置环境变量时 load() 使用默认值（完全隔离外部 env）。"""
        _isolate_load(mocker, clear_env=True)

        config = ConfigManager.load()
        assert config.llm.model == "deepseek-v4-flash"
        assert config.llm.temperature == 0.3
        assert config.ui.theme == "dark"

    def test_env_var_invalid_int_raises(self, monkeypatch, mocker):
        """环境变量值无法转为 int 时抛出 ConfigFormatError。"""
        _isolate_load(mocker)
        monkeypatch.setenv("MIAOWA_MAX_TOKENS", "not-a-number")

        with pytest.raises(ConfigFormatError, match="环境变量 MIAOWA_MAX_TOKENS"):
            ConfigManager.load()

    def test_env_var_invalid_float_raises(self, monkeypatch, mocker):
        """环境变量值无法转为 float 时抛出 ConfigFormatError。"""
        _isolate_load(mocker)
        monkeypatch.setenv("MIAOWA_TEMPERATURE", "abc")

        with pytest.raises(ConfigFormatError, match="环境变量 MIAOWA_TEMPERATURE"):
            ConfigManager.load()


# ============================================================================
# 3. test_load_from_yaml — YAML 加载（使用 tmp_path fixture）
# ============================================================================


class TestLoadFromYaml:
    """验证从 YAML 配置文件加载并合并到 Config。"""

    def test_yaml_overrides_llm_fields(self, tmp_path, mocker):
        """YAML 中的 llm 字段覆盖默认值。"""
        yaml_file = tmp_path / "miaowa.yaml"
        yaml_file.write_text(
            yaml.dump({"llm": {"model": "yaml-model", "temperature": 0.5}}),
            encoding="utf-8",
        )

        mocker.patch.object(ConfigManager, "_find_config_file", return_value=yaml_file)
        mocker.patch.object(ConfigManager, "_load_dotenv")

        config = ConfigManager.load()
        assert config.llm.model == "yaml-model"
        assert config.llm.temperature == 0.5
        # 未被 YAML 覆盖的字段保持默认值
        assert config.llm.provider == "deepseek"

    def test_yaml_overrides_multiple_sections(self, tmp_path, mocker):
        """YAML 中多个顶层键同时覆盖对应子配置。"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "llm": {"timeout": 300},
                    "ui": {"theme": "light"},
                    "logging": {"level": "DEBUG"},
                }
            ),
            encoding="utf-8",
        )

        mocker.patch.object(ConfigManager, "_find_config_file", return_value=yaml_file)
        mocker.patch.object(ConfigManager, "_load_dotenv")

        config = ConfigManager.load()
        assert config.llm.timeout == 300
        assert config.ui.theme == "light"
        assert config.logging.level == "DEBUG"

    def test_yaml_list_fields_merge_not_replace(self, tmp_path, mocker):
        """YAML 中的 list 字段与默认值合并而非替换。"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            yaml.dump({"project": {"exclude_dirs": ["custom_cache", "logs"]}}),
            encoding="utf-8",
        )

        mocker.patch.object(ConfigManager, "_find_config_file", return_value=yaml_file)
        mocker.patch.object(ConfigManager, "_load_dotenv")

        config = ConfigManager.load()
        # 默认值保留
        assert "node_modules" in config.project.exclude_dirs
        assert ".git" in config.project.exclude_dirs
        # 新值追加
        assert "custom_cache" in config.project.exclude_dirs
        assert "logs" in config.project.exclude_dirs

    def test_yml_extension_variant(self, tmp_path, mocker):
        """_find_config_file 能匹配 .yml 变体文件名。"""
        yaml_file = tmp_path / ".miaowa.yml"
        yaml_file.write_text(
            yaml.dump({"llm": {"model": "dot-yml-model"}}),
            encoding="utf-8",
        )

        # 将 tmp_path 注入搜索路径 —— mock _find_config_file 直接返回该文件
        mocker.patch.object(ConfigManager, "_find_config_file", return_value=yaml_file)
        mocker.patch.object(ConfigManager, "_load_dotenv")

        config = ConfigManager.load()
        assert config.llm.model == "dot-yml-model"

    def test_yaml_top_level_not_dict_raises(self, tmp_path, mocker):
        """YAML 顶层不是 dict 时抛出 ConfigFormatError。"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("- just a list\n- not a dict\n", encoding="utf-8")

        mocker.patch.object(ConfigManager, "_find_config_file", return_value=yaml_file)
        mocker.patch.object(ConfigManager, "_load_dotenv")

        with pytest.raises(ConfigFormatError, match="必须是映射"):
            ConfigManager.load()


# ============================================================================
# 4. test_env_var_in_yaml — ${MIAOWA_API_KEY} 语法展开
# ============================================================================


class TestEnvVarExpansionInYaml:
    """验证 YAML 值中的 ${VAR} 占位符被正确替换。"""

    def test_full_string_replaced(self, monkeypatch):
        """整个字符串就是一个变量引用时直接返回环境变量值。"""
        monkeypatch.setenv("MY_API_KEY", "sk-12345")
        result = ConfigManager._expand_env_var("${MY_API_KEY}")
        assert result == "sk-12345"

    def test_partial_string_replaced(self, monkeypatch):
        """变量引用嵌入在较长字符串中。"""
        monkeypatch.setenv("HOST", "api.example.com")
        result = ConfigManager._expand_env_var("https://${HOST}/v1")
        assert result == "https://api.example.com/v1"

    def test_multiple_vars_in_string(self, monkeypatch):
        """同一字符串中包含多个变量引用。"""
        monkeypatch.setenv("HOST", "db.example.com")
        monkeypatch.setenv("PORT", "5432")
        result = ConfigManager._expand_env_var("postgres://${HOST}:${PORT}/mydb")
        assert result == "postgres://db.example.com:5432/mydb"

    def test_undefined_var_without_default_warns(self, monkeypatch):
        """未定义且无默认值的变量发出警告并保留原样。"""
        monkeypatch.delenv("UNDEFINED_VAR", raising=False)
        with pytest.warns(RuntimeWarning, match="未定义的环境变量"):
            result = ConfigManager._expand_env_var("prefix_${UNDEFINED_VAR}_suffix")
        assert result == "prefix_${UNDEFINED_VAR}_suffix"

    def test_resolve_env_vars_descends_into_nested_structures(self, monkeypatch):
        """_resolve_env_vars 递归处理嵌套 dict/list。"""
        monkeypatch.setenv("TOKEN", "abc123")
        data = {
            "auth": {"key": "${TOKEN}"},
            "endpoints": ["https://${TOKEN}.example.com", "https://fallback.com"],
        }
        resolved = ConfigManager._resolve_env_vars(data)
        assert resolved["auth"]["key"] == "abc123"
        assert resolved["endpoints"][0] == "https://abc123.example.com"
        assert resolved["endpoints"][1] == "https://fallback.com"


# ============================================================================
# 5. test_env_var_with_default — ${VAR:-default} 语法
# ============================================================================


class TestEnvVarWithDefault:
    """验证 ${VAR:-default} 语法：变量未定义时使用默认值。"""

    def test_default_used_when_var_unset(self, monkeypatch):
        """环境变量未设置时使用默认值。"""
        monkeypatch.delenv("OPTIONAL_VAR", raising=False)
        result = ConfigManager._expand_env_var("${OPTIONAL_VAR:-8080}")
        assert result == "8080"

    def test_default_ignored_when_var_set(self, monkeypatch):
        """环境变量已设置时忽略默认值。"""
        monkeypatch.setenv("OPTIONAL_VAR", "9090")
        result = ConfigManager._expand_env_var("${OPTIONAL_VAR:-8080}")
        assert result == "9090"

    def test_empty_default(self, monkeypatch):
        """默认值可以为空字符串。"""
        monkeypatch.delenv("EMPTY_VAR", raising=False)
        result = ConfigManager._expand_env_var("${EMPTY_VAR:-}")
        assert result == ""

    def test_default_with_special_chars(self, monkeypatch):
        """默认值包含特殊字符（冒号、斜杠等）。"""
        monkeypatch.delenv("DB_URL", raising=False)
        result = ConfigManager._expand_env_var(
            "${DB_URL:-postgres://localhost:5432/miaowa}"
        )
        assert result == "postgres://localhost:5432/miaowa"

    def test_fullmatch_with_default(self, monkeypatch):
        """整个字符串匹配 ${VAR:-default} 模式。"""
        monkeypatch.delenv("PORT", raising=False)
        result = ConfigManager._expand_env_var("${PORT:-3000}")
        assert result == "3000"


# ============================================================================
# 6. test_cli_override — cli_overrides 参数覆盖
# ============================================================================


class TestCliOverride:
    """验证 CLI 参数覆盖（最高优先级）。"""

    def test_flat_key_override_model(self, mocker):
        """扁平键 'model' 覆盖 llm.model。"""
        _isolate_load(mocker)

        config = ConfigManager.load(cli_overrides={"model": "cli-model"})
        assert config.llm.model == "cli-model"

    def test_flat_key_override_log_level(self, mocker):
        """扁平键 'log_level' 覆盖 logging.level。"""
        _isolate_load(mocker)

        config = ConfigManager.load(cli_overrides={"log_level": "ERROR"})
        assert config.logging.level == "ERROR"

    def test_dotted_key_override(self, mocker):
        """点号分隔的嵌套路径直接覆盖。"""
        _isolate_load(mocker)

        config = ConfigManager.load(cli_overrides={"llm.temperature": 1.5})
        assert config.llm.temperature == 1.5

    def test_cli_overrides_env_vars(self, monkeypatch, mocker):
        """CLI 参数优先级高于环境变量。"""
        _isolate_load(mocker)
        monkeypatch.setenv("MIAOWA_MODEL", "env-model")

        config = ConfigManager.load(cli_overrides={"model": "cli-wins"})
        assert config.llm.model == "cli-wins"

    def test_cli_overrides_yaml(self, tmp_path, mocker):
        """CLI 参数优先级高于 YAML 配置。"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            yaml.dump({"llm": {"model": "yaml-model"}}),
            encoding="utf-8",
        )
        mocker.patch.object(ConfigManager, "_find_config_file", return_value=yaml_file)
        mocker.patch.object(ConfigManager, "_load_dotenv")

        config = ConfigManager.load(cli_overrides={"model": "cli-wins-again"})
        assert config.llm.model == "cli-wins-again"

    def test_unknown_flat_key_ignored(self, mocker):
        """未知的扁平键静默忽略，不抛异常。"""
        _isolate_load(mocker)

        config = ConfigManager.load(cli_overrides={"nonexistent_key": "value"})
        assert isinstance(config, Config)

    def test_empty_cli_overrides(self, mocker):
        """空的 cli_overrides 不影响加载。"""
        _isolate_load(mocker, clear_env=True)

        config = ConfigManager.load(cli_overrides={})
        assert config.llm.model == "deepseek-v4-flash"


# ============================================================================
# 7. test_validate_missing_api_key — 缺失 API Key 抛异常
# ============================================================================


class TestValidateMissingApiKey:
    """验证 api_key 为空时 validate() 抛出 ConfigMissingError。"""

    def test_empty_api_key_raises(self):
        """api_key 为空字符串时抛出 ConfigMissingError。"""
        config = Config()
        config.llm.api_key = ""

        with pytest.raises(ConfigMissingError, match="API Key 未配置"):
            ConfigManager.validate(config)

    def test_api_key_none_equivalent_to_empty(self):
        """api_key 为 None 时空字符串判定同样失败（dataclass 默认 ""）。"""
        config = Config()
        config.llm.api_key = ""  # dataclass 默认值

        with pytest.raises(ConfigMissingError):
            ConfigManager.validate(config)

    def test_unexpanded_placeholder_raises(self):
        """api_key 包含未展开的 ${VAR} 占位符时抛出 ConfigMissingError。"""
        config = Config()
        config.llm.api_key = "${MIAOWA_API_KEY}"

        with pytest.raises(ConfigMissingError, match="未展开的环境变量引用"):
            ConfigManager.validate(config)

    def test_error_includes_key_name_context(self):
        """异常对象携带 key_name 上下文属性。"""
        config = Config()
        config.llm.api_key = ""

        with pytest.raises(ConfigMissingError) as exc_info:
            ConfigManager.validate(config)
        assert exc_info.value.key_name == "llm.api_key"


# ============================================================================
# 8. test_validate_with_api_key — 有 Key 验证通过
# ============================================================================


class TestValidateWithApiKey:
    """验证 api_key 有效时 validate() 不抛异常。"""

    def test_valid_api_key_passes(self, mock_config):
        """设置有效的 api_key 后 validate() 无异常。"""
        # 不抛异常即通过
        ConfigManager.validate(mock_config)

    def test_valid_api_key_with_all_defaults(self):
        """所有默认值 + 有效 api_key 通过校验。"""
        config = Config()
        config.llm.api_key = "sk-valid-key"
        ConfigManager.validate(config)


# ============================================================================
# 9. test_validate_temperature_and_tokens — temperature / max_tokens 合法性
# ============================================================================


class TestValidateTemperatureAndTokens:
    """验证 temperature 范围和 max_tokens 正数校验。"""

    def test_temperature_zero_raises(self):
        """temperature=0（不在 (0, 2] 范围）抛出 ConfigFormatError。"""
        config = Config()
        config.llm.api_key = "sk-ok"
        config.llm.temperature = 0.0

        with pytest.raises(ConfigFormatError, match="temperature"):
            ConfigManager.validate(config)

    def test_temperature_above_two_raises(self):
        """temperature > 2 抛出 ConfigFormatError。"""
        config = Config()
        config.llm.api_key = "sk-ok"
        config.llm.temperature = 2.1

        with pytest.raises(ConfigFormatError, match="temperature"):
            ConfigManager.validate(config)

    def test_max_tokens_zero_raises(self):
        """max_tokens=0 抛出 ConfigFormatError。"""
        config = Config()
        config.llm.api_key = "sk-ok"
        config.llm.max_tokens = 0

        with pytest.raises(ConfigFormatError, match="max_tokens"):
            ConfigManager.validate(config)

    def test_max_tokens_negative_raises(self):
        """max_tokens 为负抛出 ConfigFormatError。"""
        config = Config()
        config.llm.api_key = "sk-ok"
        config.llm.max_tokens = -1

        with pytest.raises(ConfigFormatError, match="max_tokens"):
            ConfigManager.validate(config)


# ============================================================================
# 10. test_config_file_not_found — 配置文件不存在静默回退
# ============================================================================


class TestConfigFileNotFound:
    """验证无配置文件时 load() 静默使用默认值。"""

    def test_no_config_file_returns_defaults(self, mocker):
        """_find_config_file 返回 None 时使用默认配置。"""
        mocker.patch.object(ConfigManager, "_find_config_file", return_value=None)
        mocker.patch.object(ConfigManager, "_load_dotenv")

        config = ConfigManager.load()
        assert config.llm.model == "deepseek-v4-flash"
        assert config.llm.temperature == 0.3

    def test_search_dir_not_exists_is_skipped(self, mocker, tmp_path):
        """搜索目录不存在时跳过（而非崩溃）。"""
        # 强制 SEARCH_DIRS 指向不存在的目录
        mocker.patch.object(
            ConfigManager,
            "SEARCH_DIRS",
            (str(tmp_path / "nonexistent_dir"),),
        )
        result = ConfigManager._find_config_file()
        assert result is None


# ============================================================================
# 11. test_config_file_malformed — 格式错误抛 ConfigFormatError
# ============================================================================


class TestConfigFileMalformed:
    """验证 YAML 格式错误、文件不可读等场景抛出 ConfigFormatError。"""

    def test_invalid_yaml_syntax_raises(self, tmp_path, mocker):
        """YAML 语法错误时抛出 ConfigFormatError。"""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("llm: {bad: [unclosed", encoding="utf-8")

        mocker.patch.object(ConfigManager, "_find_config_file", return_value=yaml_file)
        mocker.patch.object(ConfigManager, "_load_dotenv")

        with pytest.raises(ConfigFormatError, match="YAML 解析失败"):
            ConfigManager.load()

    def test_error_includes_file_path(self, tmp_path, mocker):
        """ConfigFormatError 携带 file_path 上下文。"""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("{invalid yaml", encoding="utf-8")

        with pytest.raises(ConfigFormatError) as exc_info:
            ConfigManager._load_yaml(yaml_file)
        assert exc_info.value.file_path == str(yaml_file)

    def test_unreadable_file_raises(self, tmp_path):
        """文件存在但不可读时抛出 ConfigFormatError。"""
        # 创建一个路径指向的文件不存在（模拟 OS 读取错误）
        nonexistent = tmp_path / "nonexistent.yaml"
        # _load_yaml 直接接收不存在的路径 → OSError 被包装
        with pytest.raises(ConfigFormatError, match="无法读取配置文件"):
            ConfigManager._load_yaml(nonexistent)


# ============================================================================
# 12. test_dotenv_loading — .env 文件加载
# ============================================================================


class TestDotenvLoading:
    """验证 .env 文件被正确发现和加载。

    Note: 所有测试 mock 掉 load_dotenv() 调用本身，仅验证路径推导逻辑，
    不实际修改 os.environ（避免测试间环境变量污染）。
    """

    def test_dotenv_in_project_root_is_loaded(self, tmp_path, mocker):
        """项目根目录的 .env 文件路径被正确传递给 load_dotenv。"""
        project_root = tmp_path / "fake_project"
        project_root.mkdir()
        (project_root / ".git").mkdir()  # 标记为项目根
        env_file = project_root / ".env"
        env_file.write_text("MIAOWA_CUSTOM_VAR=from_dotenv\n", encoding="utf-8")

        mocker.patch.object(ConfigManager, "_find_project_root", return_value=project_root)
        mock_load = mocker.patch("miaowa.core.config.load_dotenv")

        ConfigManager._load_dotenv()

        mock_load.assert_called_once_with(
            dotenv_path=str(env_file), override=False
        )

    def test_dotenv_fallback_to_cwd(self, tmp_path, mocker):
        """无项目根目录时回退到当前工作目录的 .env。"""
        mocker.patch.object(ConfigManager, "_find_project_root", return_value=None)

        env_file = tmp_path / ".env"
        env_file.write_text("MIAOWA_CWD_VAR=from_cwd\n", encoding="utf-8")
        mocker.patch("pathlib.Path.cwd", return_value=tmp_path)
        mock_load = mocker.patch("miaowa.core.config.load_dotenv")

        ConfigManager._load_dotenv()

        mock_load.assert_called_once_with(
            dotenv_path=str(env_file), override=False
        )

    def test_dotenv_integration_with_load(self, tmp_path, monkeypatch, mocker):
        """.env 加载集成到 load() 流程：模拟 dotenv 加载后 env 值覆盖 YAML 默认值。"""
        project_root = tmp_path / "proj"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        env_file = project_root / ".env"
        env_file.write_text("MIAOWA_MODEL=dotenv-model\n", encoding="utf-8")

        # Mock _load_dotenv 以避免实际修改 os.environ；
        # 通过 monkeypatch 手动设置环境变量来模拟 dotenv 的加载效果
        mocker.patch.object(ConfigManager, "_find_project_root", return_value=project_root)
        mocker.patch.object(ConfigManager, "_find_config_file", return_value=None)
        mocker.patch.object(ConfigManager, "_load_dotenv")  # 阻止真实 dotenv 调用

        monkeypatch.setenv("MIAOWA_MODEL", "dotenv-model")

        config = ConfigManager.load()
        assert config.llm.model == "dotenv-model"


# ============================================================================
# 附加：load_from_dict / ensure_config_dir / expand_path
# ============================================================================


class TestLoadFromDict:
    """验证 load_from_dict() 从字典构造 Config。"""

    def test_load_from_dict_overrides_defaults(self):
        """字典中的字段覆盖默认值。"""
        config = ConfigManager.load_from_dict(
            {"llm": {"model": "dict-model", "temperature": 0.9}}
        )
        assert config.llm.model == "dict-model"
        assert config.llm.temperature == 0.9
        # 未指定的保持默认
        assert config.llm.provider == "deepseek"

    def test_load_from_dict_empty_dict(self):
        """空字典返回纯默认 Config。"""
        config = ConfigManager.load_from_dict({})
        assert config.llm.model == "deepseek-v4-flash"


class TestEnsureConfigDir:
    """验证 ensure_config_dir() 创建目录。"""

    def test_creates_directory(self, tmp_path, monkeypatch):
        """ensure_config_dir 创建 ~/.miaowa/ 目录。"""
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "expanduser", lambda self: fake_home / ".miaowa")

        result = ConfigManager.ensure_config_dir()
        assert result.is_dir()
        assert result.name == ".miaowa"

    def test_idempotent_when_dir_already_exists(self, tmp_path, monkeypatch):
        """目录已存在时 ensure_config_dir 幂等不报错。"""
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        miaowa_dir = fake_home / ".miaowa"
        miaowa_dir.mkdir()  # 预先创建
        monkeypatch.setattr(Path, "expanduser", lambda self: miaowa_dir)

        result = ConfigManager.ensure_config_dir()
        assert result.is_dir()
        assert result == miaowa_dir


class TestExpandPath:
    """验证 expand_path() 展开 ~。"""

    def test_expand_tilde(self, tmp_path, monkeypatch):
        """~ 展开为用户主目录。"""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        result = ConfigManager.expand_path("~/logs/app.log")
        assert result.is_absolute()


class TestFieldType:
    """验证 _field_type 类型查询。"""

    def test_known_field_returns_correct_type(self):
        """已知字段返回正确的 Python 类型。"""
        assert ConfigManager._field_type(Config, "llm", "temperature") is float
        assert ConfigManager._field_type(Config, "llm", "max_tokens") is int
        assert ConfigManager._field_type(Config, "llm", "model") is str

    def test_unknown_section_falls_back_to_str(self):
        """未知 section 返回 str 兜底。"""
        assert ConfigManager._field_type(Config, "nonexistent", "field") is str

    def test_unknown_field_falls_back_to_str(self):
        """未知 field 返回 str 兜底。"""
        assert ConfigManager._field_type(Config, "llm", "nonexistent_field") is str


class TestCoerce:
    """验证 _coerce 类型转换。"""

    def test_bool_true_variants(self):
        """bool 类型识别多种 true 变体。"""
        for val in ("1", "true", "True", "TRUE", "yes", "YES", "on", "ON"):
            assert ConfigManager._coerce(val, bool) is True

    def test_bool_false_variants(self):
        """bool 类型：其他字符串转为 False。"""
        for val in ("0", "false", "no", "off", "anything"):
            assert ConfigManager._coerce(val, bool) is False

    def test_str_passthrough(self):
        """str 类型原样返回。"""
        assert ConfigManager._coerce("hello", str) == "hello"

    def test_int_conversion(self):
        """int 类型字符串正确转换为整数。"""
        assert ConfigManager._coerce("4096", int) == 4096
        assert isinstance(ConfigManager._coerce("4096", int), int)

    def test_float_conversion(self):
        """float 类型字符串正确转换为浮点数。"""
        assert ConfigManager._coerce("0.7", float) == 0.7
        assert isinstance(ConfigManager._coerce("0.7", float), float)

    def test_int_conversion_raises_on_invalid(self):
        """int 转换失败时抛出 ConfigFormatError。"""
        with pytest.raises(ConfigFormatError, match="应为整数"):
            ConfigManager._coerce("not-a-number", int, env_name="TEST_VAR")

    def test_float_conversion_raises_on_invalid(self):
        """float 转换失败时抛出 ConfigFormatError。"""
        with pytest.raises(ConfigFormatError, match="应为浮点数"):
            ConfigManager._coerce("abc", float, env_name="TEST_VAR")

    def test_unknown_type_falls_back_to_raw_string(self):
        """无法识别的目标类型原样返回字符串。"""
        assert ConfigManager._coerce("anything", list) == "anything"  # type: ignore[arg-type]


# ============================================================================
# 附加：_find_project_root
# ============================================================================


class TestFindProjectRoot:
    """验证 _find_project_root() 的项目根目录发现逻辑。

    使用 Path.cwd mock 而非 monkeypatch.chdir，避免影响其他测试。
    """

    def test_finds_root_with_git_dir(self, tmp_path, mocker):
        """当前目录包含 .git 时直接返回该目录。"""
        project_root = tmp_path / "my_project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        mocker.patch.object(Path, "cwd", return_value=project_root)

        result = ConfigManager._find_project_root()
        assert result == project_root

    def test_finds_root_with_pyproject_toml(self, tmp_path, mocker):
        """当前目录包含 pyproject.toml 时返回该目录。"""
        project_root = tmp_path / "py_project"
        project_root.mkdir()
        (project_root / "pyproject.toml").write_text("[project]\nname='test'\n")
        mocker.patch.object(Path, "cwd", return_value=project_root)

        result = ConfigManager._find_project_root()
        assert result == project_root

    def test_searches_upward_to_ancestor_with_git(self, tmp_path, mocker):
        """从深层子目录向上搜索，在祖先目录找到 .git。"""
        project_root = tmp_path / "ancestor_project"
        (project_root / ".git").mkdir(parents=True)
        deep_dir = project_root / "src" / "miaowa" / "core"
        deep_dir.mkdir(parents=True)

        mocker.patch.object(Path, "cwd", return_value=deep_dir)

        result = ConfigManager._find_project_root()
        assert result == project_root

    def test_returns_none_when_no_marker_found(self, tmp_path, mocker):
        """向上搜索 10 层均无 .git / pyproject.toml 时返回 None。"""
        start = tmp_path / "deep" / "nested" / "dir"
        start.mkdir(parents=True)
        mocker.patch.object(Path, "cwd", return_value=start)

        result = ConfigManager._find_project_root()
        assert result is None


# ============================================================================
# 缓存行为
# ============================================================================


class TestCacheBehavior:
    """验证 _cached_config 缓存行为。"""

    def test_cache_disabled_by_default(self, mocker):
        """默认 use_cache=False 时每次重新加载。"""
        _isolate_load(mocker)
        ConfigManager._cached_config = None

        c1 = ConfigManager.load()
        c2 = ConfigManager.load()
        assert c1 is not c2  # 不同实例

    def test_cache_enabled_returns_same_instance(self, mocker):
        """use_cache=True 时返回缓存实例。"""
        _isolate_load(mocker)
        ConfigManager._cached_config = None

        c1 = ConfigManager.load(use_cache=True)
        c2 = ConfigManager.load(use_cache=True)
        assert c1 is c2

    def test_cache_is_stale_after_new_load_without_cache(self, mocker):
        """不带缓存的新加载会更新缓存。"""
        _isolate_load(mocker)
        ConfigManager._cached_config = None

        c1 = ConfigManager.load(use_cache=True)
        c2 = ConfigManager.load(use_cache=False)
        assert c1 is not c2
