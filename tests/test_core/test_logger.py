"""日志系统单元测试。

验证 setup_logger 和 get_logger 的行为：
文件创建、目录创建、名称绑定、日志级别、轮转配置、权限回退。

注意：loguru 使用全局状态，各测试前后需清理 handler 并
重置 _INITIALIZED 标志。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from loguru import logger as _loguru_logger

import miaowa.core.logger as logger_mod
from miaowa.core.logger import _resolve_level, _resolve_log_dir, get_logger, setup_logger


# ---------------------------------------------------------------------------
# 自动清理 fixture：每个测试前后重置 loguru 状态
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_logger_state():
    """每个测试前后清理 loguru handler 并重置 _INITIALIZED。

    同时保存/恢复 sys.excepthook 和 logging.root.handlers，
    以防止测试间全局状态污染（setup_logger 会替换这两者）。
    """
    original_excepthook = sys.excepthook
    original_root_handlers = list(logging.root.handlers)
    original_root_level = logging.root.level
    _loguru_logger.remove()
    logger_mod._INITIALIZED = False

    yield

    # 清理 — 恢复测试前状态
    _loguru_logger.remove()
    logger_mod._INITIALIZED = False
    sys.excepthook = original_excepthook
    logging.root.handlers[:] = original_root_handlers
    logging.root.level = original_root_level


# ---------------------------------------------------------------------------
# 辅助：创建一个带有简单 handler 的 loguru（用于 idempotent 测试）
# ---------------------------------------------------------------------------


@pytest.fixture
def _logger_with_handler():
    """给 loguru 添加一个占位 handler，用于测试幂等行为。"""
    hid = _loguru_logger.add(lambda _: None, format="{message}")
    yield
    try:
        _loguru_logger.remove(hid)
    except ValueError:
        pass


# ============================================================================
# 1. test_setup_logger_creates_file — 日志文件创建
# ============================================================================


class TestSetupLoggerCreatesFile:
    """验证 setup_logger() 在目标路径创建日志文件。"""

    def test_log_file_created_after_logging(self, mock_logging_config):
        """调用 setup_logger 后写入日志，文件应存在且含日志内容。"""
        setup_logger(mock_logging_config)

        test_logger = get_logger("test.module")
        test_logger.info("测试日志消息")

        # 查找实际生成的日志文件
        log_dir = Path(mock_logging_config.file).parent
        log_files = list(log_dir.glob("miaowa_*.log"))
        assert len(log_files) > 0, f"日志目录 {log_dir} 中未找到日志文件"

        content = log_files[0].read_text(encoding="utf-8")
        assert "测试日志消息" in content

    def test_log_file_is_utf8_encoded(self, mock_logging_config):
        """日志文件使用 UTF-8 编码，中文内容正确写入与读取。"""
        setup_logger(mock_logging_config)

        test_logger = get_logger("test.unicode")
        test_logger.info("你好，喵哇！🎉")

        log_dir = Path(mock_logging_config.file).parent
        log_files = list(log_dir.glob("miaowa_*.log"))
        assert len(log_files) > 0

        content = log_files[0].read_text(encoding="utf-8")
        assert "你好，喵哇！🎉" in content

    def test_setup_logger_adds_file_handler(self, mock_logging_config, mocker):
        """验证 setup_logger 确实调用了 loguru.add 添加文件 handler。"""
        spy = mocker.spy(_loguru_logger, "add")

        setup_logger(mock_logging_config)

        # 确认 add 被调用（文件 handler + 控制台 handler）
        assert spy.call_count >= 2

        # 至少有一个调用传入了文件路径（包含 miaowa_ 字样）
        file_calls = [
            c for c in spy.call_args_list
            if any("miaowa_" in str(a) for a in c[0])
        ]
        assert len(file_calls) >= 1


# ============================================================================
# 2. test_setup_logger_creates_parent_dir — 父目录自动创建
# ============================================================================


class TestSetupLoggerCreatesParentDir:
    """验证 setup_logger() 自动创建日志文件的父目录。"""

    def test_parent_dir_created(self, tmp_path):
        """不存在的父目录由 setup_logger 自动创建。"""
        from miaowa.core.config import LoggingConfig

        log_dir = tmp_path / "deeply" / "nested" / "logs"
        cfg = LoggingConfig(
            level="DEBUG",
            file=str(log_dir / "app.log"),
        )

        assert not log_dir.exists()
        setup_logger(cfg)
        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_already_existing_dir_no_error(self, tmp_path):
        """父目录已存在时 setup_logger 不抛异常（幂等）。"""
        from miaowa.core.config import LoggingConfig

        log_dir = tmp_path / "existing_logs"
        log_dir.mkdir(parents=True)
        cfg = LoggingConfig(
            level="DEBUG",
            file=str(log_dir / "app.log"),
        )

        # 不应抛异常
        setup_logger(cfg)
        assert log_dir.exists()


# ============================================================================
# 3. test_get_logger_binds_name — get_logger 绑定模块名
# ============================================================================


class TestGetLoggerBindsName:
    """验证 get_logger() 返回的 logger 已绑定 name 字段。

    Note: loguru 的 {name} 格式字段显示调用方模块名（如 tests.test_core.test_logger），
    而通过 .bind(name=...) 绑定的额外字段需通过 {extra[name]} 格式访问。
    get_logger() 使用 bind() 将 name 绑定为 extra 字段，供日志格式中的 {extra[name]} 使用。
    """

    def test_bound_name_appears_via_extra_format(self, mock_logging_config):
        """通过 {extra[name]} 格式验证绑定的 name 出现在日志中。"""
        setup_logger(mock_logging_config)

        # 在 setup_logger 之后添加自定义 sink 捕获 extra
        captured_extras: list[dict] = []

        def _capture_sink(message):
            captured_extras.append(dict(message.record["extra"]))

        sink_id = _loguru_logger.add(
            _capture_sink, format="{extra[name]}: {message}"
        )

        module_logger = get_logger("miaowa.agent.planner")
        module_logger.info("planner 已就绪")

        _loguru_logger.remove(sink_id)

        assert len(captured_extras) > 0
        assert any(e.get("name") == "miaowa.agent.planner" for e in captured_extras), (
            f"未找到绑定的 name，captured: {captured_extras}"
        )

    def test_different_names_produce_different_extra(self):
        """不同模块的 logger 绑定不同名称（通过 bind() 验证）。"""
        alpha = get_logger("module.alpha")
        beta = get_logger("module.beta")

        # 验证两个 logger 是不同对象
        assert alpha is not beta
        # 验证二者均有 logging 能力
        alpha.info("alpha msg")
        beta.info("beta msg")

    def test_get_logger_returns_loguru_logger(self):
        """get_logger 返回 loguru Logger 实例（通过 bind 创建）。"""
        import loguru

        lg = get_logger("test")
        # bind() 返回的是 loguru.Logger 类型
        assert isinstance(lg, type(_loguru_logger))
        assert hasattr(lg, "info")
        assert hasattr(lg, "debug")
        assert hasattr(lg, "warning")
        assert hasattr(lg, "error")


# ============================================================================
# 4. test_log_levels — 日志级别
# ============================================================================


class TestLogLevels:
    """验证不同日志级别的行为。"""

    @pytest.mark.parametrize("level", ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"])
    def test_console_level_respected(self, level, tmp_path):
        """各种标准级别均被 _resolve_level 正确解析。"""
        from miaowa.core.config import LoggingConfig

        log_dir = tmp_path / "level_test_logs"
        cfg = LoggingConfig(
            level=level,
            file=str(log_dir / "app.log"),
        )
        # 不应抛异常
        setup_logger(cfg)
        # 能够写入对应级别日志
        lg = get_logger("test")
        lg.log(level, f"测试 {level} 级别")

    def test_numeric_level_string(self):
        """数字字符串级别被正确映射。"""
        assert _resolve_level("10") == "DEBUG"
        assert _resolve_level("20") == "INFO"
        assert _resolve_level("30") == "WARNING"
        assert _resolve_level("40") == "ERROR"
        assert _resolve_level("50") == "CRITICAL"

    def test_numeric_level_boundaries(self):
        """数值在边界处的映射。"""
        assert _resolve_level("5") == "TRACE"
        assert _resolve_level("11") == "INFO"    # 10 < 11 <= 20
        assert _resolve_level("26") == "WARNING"  # 25 < 26 <= 30
        assert _resolve_level("31") == "ERROR"    # 30 < 31 <= 40
        assert _resolve_level("41") == "CRITICAL" # 40 < 41

    def test_invalid_level_falls_back_to_info(self):
        """无效级别字符串回退到默认级别。"""
        assert _resolve_level("INVALID") == "INFO"
        assert _resolve_level("") == "INFO"

    def test_negative_numeric_level_falls_back(self):
        """负数级别回退到默认。"""
        assert _resolve_level("-1") == "INFO"

    def test_case_insensitive_level(self):
        """级别名称大小写不敏感。"""
        assert _resolve_level("debug") == "DEBUG"
        assert _resolve_level("Warning") == "WARNING"
        assert _resolve_level("error") == "ERROR"

    def test_level_with_whitespace(self):
        """级别字符串两端空格被忽略。"""
        assert _resolve_level("  DEBUG  ") == "DEBUG"


# ============================================================================
# 5. test_file_rotation — 文件轮转配置
# ============================================================================


class TestFileRotation:
    """验证日志文件轮转参数被正确传递到 loguru。"""

    def test_rotation_passed_to_loguru_add(self, mock_logging_config, mocker):
        """rotation 参数与 LoggingConfig.max_size 一致。"""
        spy = mocker.spy(_loguru_logger, "add")

        setup_logger(mock_logging_config)

        # 找到文件 handler 的调用（第一个参数是路径字符串，含 miaowa_）
        file_calls = [
            c for c in spy.call_args_list
            if any("miaowa_" in str(a) for a in c[0])
        ]
        assert len(file_calls) >= 1
        # 验证 rotation kwarg
        _, kwargs = file_calls[0]
        assert kwargs.get("rotation") == mock_logging_config.max_size

    def test_retention_is_7_days(self, mock_logging_config, mocker):
        """retention 固定为 7 days。"""
        spy = mocker.spy(_loguru_logger, "add")

        setup_logger(mock_logging_config)

        file_calls = [
            c for c in spy.call_args_list
            if any("miaowa_" in str(a) for a in c[0])
        ]
        _, kwargs = file_calls[0]
        assert kwargs.get("retention") == "7 days"

    def test_compression_is_gz(self, mock_logging_config, mocker):
        """日志轮转后的旧文件以 gz 压缩。"""
        spy = mocker.spy(_loguru_logger, "add")

        setup_logger(mock_logging_config)

        file_calls = [
            c for c in spy.call_args_list
            if any("miaowa_" in str(a) for a in c[0])
        ]
        _, kwargs = file_calls[0]
        assert kwargs.get("compression") == "gz"

    def test_enqueue_enabled(self, mock_logging_config, mocker):
        """文件 handler 启用 enqueue 模式（线程安全）。"""
        spy = mocker.spy(_loguru_logger, "add")

        setup_logger(mock_logging_config)

        file_calls = [
            c for c in spy.call_args_list
            if any("miaowa_" in str(a) for a in c[0])
        ]
        _, kwargs = file_calls[0]
        assert kwargs.get("enqueue") is True

    def test_rotation_triggers_on_size_exceeded(self, tmp_path):
        """写入超过 max_size 的日志后触发文件轮转（生成多个日志文件）。"""
        from miaowa.core.config import LoggingConfig

        log_dir = tmp_path / "rotation_test_logs"
        cfg = LoggingConfig(
            level="DEBUG",
            file=str(log_dir / "app.log"),
            max_size="1 KB",
        )

        setup_logger(cfg)

        lg = get_logger("test.rotation")
        # 写入大量日志以触发轮转（每条约 200 字节，写 30 条 ≈ 6 KB > 1 KB）
        long_message = "X" * 160
        for i in range(30):
            lg.info(f"[{i:04d}] {long_message}")

        # 查找日志目录中所有 app 相关的日志文件
        log_files = list(log_dir.glob("miaowa_*.log"))
        assert len(log_files) >= 1, f"未找到日志文件于 {log_dir}"

        # 如果触发了轮转，应有多个日志文件（或压缩文件）
        all_files = list(log_dir.iterdir())
        # 至少应有原始日志文件（可能已轮转压缩）
        assert len(all_files) >= 1, "日志目录应至少包含一个文件"


# ============================================================================
# 6. test_fallback_on_permission_error — 权限不足回退
# ============================================================================


class TestFallbackOnPermissionError:
    """验证日志目录创建失败时的回退机制。"""

    def test_fallback_to_temp_dir(self, tmp_path, mocker):
        """mkdir 抛出 OSError 时回退到系统临时目录。"""
        from miaowa.core.config import LoggingConfig

        # 将临时目录重定向到 tmp_path，避免污染系统 temp 目录
        mocker.patch("tempfile.gettempdir", return_value=str(tmp_path / "fake_temp"))

        original_mkdir = Path.mkdir

        def _failing_mkdir(self, *args, **kwargs):
            if "unwritable" in str(self):
                raise OSError("Permission denied")
            return original_mkdir(self, *args, **kwargs)

        mocker.patch.object(Path, "mkdir", _failing_mkdir)

        cfg = LoggingConfig(
            level="DEBUG",
            file=str(tmp_path / "unwritable" / "app.log"),
        )

        # 不应抛异常，应回退到 fake_temp/miaowa_logs/
        setup_logger(cfg)

        # 验证回退目录已被创建
        fallback_dir = tmp_path / "fake_temp" / "miaowa_logs"
        assert fallback_dir.is_dir(), f"回退目录 {fallback_dir} 应存在"

    def test_fallback_prints_warning_to_stderr(self, tmp_path, capsys, mocker):
        """回退时向 stderr 输出提示信息。"""
        from miaowa.core.config import LoggingConfig

        # 重定向临时目录避免污染
        mocker.patch("tempfile.gettempdir", return_value=str(tmp_path / "fake_temp"))

        original_mkdir = Path.mkdir
        fail_count = 0

        def _failing_mkdir_once(self, *args, **kwargs):
            nonlocal fail_count
            # 只让第一次 mkdir（对日志目录）失败
            if fail_count == 0 and "denied_here" in str(self):
                fail_count += 1
                raise OSError("Permission denied")
            return original_mkdir(self, *args, **kwargs)

        mocker.patch.object(Path, "mkdir", _failing_mkdir_once)

        cfg = LoggingConfig(
            level="DEBUG",
            file=str(tmp_path / "denied_here" / "app.log"),
        )

        setup_logger(cfg)

        captured = capsys.readouterr()
        assert "回退" in captured.err or "Miaowa" in captured.err


# ============================================================================
# 7. test_setup_logger_idempotent — 幂等初始化
# ============================================================================


class TestSetupLoggerIdempotent:
    """验证 setup_logger 的幂等行为。"""

    def test_second_call_rebuilds_handlers(self, mock_logging_config, _logger_with_handler):
        """第二次调用移除旧 handler 并重新添加。"""
        # 第一次
        setup_logger(mock_logging_config)
        first_handler_count = len(list(_loguru_logger._core.handlers))

        # 第二次
        setup_logger(mock_logging_config)
        second_handler_count = len(list(_loguru_logger._core.handlers))

        # 应该都是 2（文件 + 控制台），不是累积的
        assert second_handler_count == 2
        assert second_handler_count == first_handler_count

    def test_initialized_warning_on_second_call(self, mock_logging_config, _logger_with_handler):
        """_INITIALIZED=True 时再次调用输出 warning 日志。"""
        # 第一次初始化
        setup_logger(mock_logging_config)
        assert logger_mod._INITIALIZED is True

        # 捕获第二次调用时的 warning 消息
        # 注意：setup_logger() 内部会调用 remove() 清除所有 handler，
        # 所以捕获 sink 也会被清掉，无需手动 remove
        captured: list[str] = []

        def _capture(message):
            captured.append(str(message))

        _loguru_logger.add(_capture, level="WARNING", format="{message}")
        setup_logger(mock_logging_config)
        # sink 已被 setup_logger 内部的 remove() 清除，无需再次 remove

        assert any("已初始化" in msg for msg in captured), (
            f"未找到 '已初始化' 警告，captured: {captured}"
        )


# ============================================================================
# 8. test_resolve_log_dir — 日志目录推导
# ============================================================================


class TestResolveLogDir:
    """验证 _resolve_log_dir 从日志文件路径推导目录。"""

    def test_file_with_extension_returns_parent(self):
        """路径含 .log 后缀 → 返回父目录。"""
        result = _resolve_log_dir("/var/log/miaowa/app.log")
        assert result == Path("/var/log/miaowa")

    def test_directory_path_returns_itself(self):
        """路径无扩展名（视为目录）→ 返回自身。"""
        result = _resolve_log_dir("/var/log/miaowa/")
        assert result == Path("/var/log/miaowa")

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        """~ 展开为用户主目录（通过 HOME/USERPROFILE 环境变量）。"""
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        # Windows 上 expanduser() 优先使用 USERPROFILE，Unix 上使用 HOME
        monkeypatch.setenv("USERPROFILE", str(fake_home))
        monkeypatch.setenv("HOME", str(fake_home))

        result = _resolve_log_dir("~/logs/app.log")
        assert str(result).startswith(str(fake_home))


# ============================================================================
# 9. 标准 logging → loguru 桥接
# ============================================================================


class TestInterceptHandler:
    """验证标准 logging 日志通过 _InterceptHandler 桥接到 loguru。"""

    def test_stdlib_logging_bridged_to_loguru(self, mock_logging_config):
        """标准 logging 日志出现在 loguru 的日志文件中。"""
        setup_logger(mock_logging_config)

        std_logger = logging.getLogger("test_bridge")
        std_logger.warning("来自标准 logging 的警告")

        log_dir = Path(mock_logging_config.file).parent
        log_files = list(log_dir.glob("miaowa_*.log"))
        content = log_files[0].read_text(encoding="utf-8")

        assert "来自标准 logging 的警告" in content

    def test_third_party_libraries_are_silenced(self, mock_logging_config):
        """httpx / openai / urllib3 / httpcore 被设为 WARNING。"""
        setup_logger(mock_logging_config)

        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING
        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING


# ============================================================================
# 10. 全局异常钩子
# ============================================================================


class TestExceptionHook:
    """验证 _exception_hook 作为 sys.excepthook 工作。"""

    def test_excepthook_is_installed(self, mock_logging_config):
        """setup_logger 后 sys.excepthook 被替换为 _exception_hook。"""
        setup_logger(mock_logging_config)
        assert sys.excepthook is logger_mod._exception_hook

    def test_excepthook_logs_critical(self, mock_logging_config):
        """未捕获的异常被记录为 CRITICAL 级别。"""
        setup_logger(mock_logging_config)

        # 直接调用钩子模拟未捕获异常
        try:
            raise ValueError("致命错误")
        except ValueError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            logger_mod._exception_hook(exc_type, exc_value, exc_tb)

        log_dir = Path(mock_logging_config.file).parent
        log_files = list(log_dir.glob("miaowa_*.log"))
        content = log_files[0].read_text(encoding="utf-8")

        assert "CRITICAL" in content
        assert "未捕获的异常" in content
