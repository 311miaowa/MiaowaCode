"""Pytest 配置与共享 fixtures。

为所有测试模块提供通用的测试夹具和配置。
"""

import pytest


@pytest.fixture
def sample_config_dict() -> dict:
    """提供测试用的最小有效配置字典。"""
    return {
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "max_turns": 50,
        "timeout": 120,
    }
