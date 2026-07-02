"""Pytest 配置与共享 fixtures。

为所有测试模块提供通用的测试夹具和配置。
"""

from __future__ import annotations

import pytest

from miaowa.core.config import Config, LLMConfig, LoggingConfig, UIConfig, ProjectConfig, ToolsConfig


@pytest.fixture
def sample_config_dict() -> dict:
    """提供测试用的最小有效配置字典。

    Note: 当前未被测试直接引用；保留供后续 test_llm / test_agent 模块使用，
    这些模块可能需要 AppConfig 风格的扁平配置字典。
    """
    return {
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "max_turns": 50,
        "timeout": 120,
    }


@pytest.fixture
def mock_config() -> Config:
    """提供带有有效 API Key 的完整 Config 实例。

    llm.api_key 预先设置为有效值，便于测试 validate() 等需要
    合法配置的场景。
    """
    config = Config()
    config.llm.api_key = "test-api-key-for-unit-tests"
    return config


@pytest.fixture
def mock_logging_config(tmp_path) -> LoggingConfig:
    """提供指向临时目录的 LoggingConfig 实例。

    日志文件和目录均指向 tmp_path，确保测试间隔离且不会
    污染真实的 ~/.miaowa/logs/ 目录。
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return LoggingConfig(
        level="DEBUG",
        file_level="DEBUG",
        file=str(log_dir / "miaowa.log"),
        max_size="1 MB",
        backup_count=2,
    )
