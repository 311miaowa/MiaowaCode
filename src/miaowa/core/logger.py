"""日志系统 — 基于 loguru 的统一日志管理。

模块提供两个核心 API：

setup_logger(config)
    初始化全局日志处理器（文件 + 控制台 + logging 桥接），应用启动时调用一次。
    幂等 —— 重复调用会先移除已有 handler 再重新构建。

get_logger(name)
    获取绑定模块名称的 logger 实例，供各模块使用。

日志输出位置：
    - 文件：~/.miaowa/logs/miaowa_{date}.log（默认 DEBUG，按日 + 按大小轮转）
    - 控制台：sys.stderr（配置 level，默认 INFO，彩色简洁格式）
    - 标准 logging 桥：第三方库 logging 输出统一接入 loguru

Typical usage::

    from miaowa.core.config import ConfigManager
    from miaowa.core.logger import setup_logger, get_logger

    config = ConfigManager.load()
    setup_logger(config.logging)

    logger = get_logger(__name__)
    logger.info("Miaowa Code 启动成功")
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger as _loguru_logger

if TYPE_CHECKING:
    from miaowa.core.config import LoggingConfig

# ---------------------------------------------------------------------------
# 模块级常量
# ---------------------------------------------------------------------------

_LOG_DIR = Path("~/.miaowa/logs").expanduser()
"""日志文件根目录。"""

_INITIALIZED: bool = False
"""标记 setup_logger() 是否已被调用（防重复初始化）。"""

# -- 文件日志格式 -----------------------------------------------------------

_FILE_FORMAT: str = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# -- 控制台日志格式（简洁彩色）-----------------------------------------------


_CONSOLE_FORMAT: str = (
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

_DEFAULT_CONSOLE_LEVEL = "INFO"


# ============================================================================
# 标准 logging → loguru 桥接（PRD §3.5.1）
# ============================================================================


class _InterceptHandler(logging.Handler):
    """将标准 logging 日志重定向到 loguru。

    自动跳过 logging 内部栈帧，确保日志中 {name}:{function}:{line}
    指向原始调用位置而非桥接代码本身。
    """

    def emit(self, record: logging.LogRecord) -> None:
        # 获取原始调用栈帧（跳过 logging 内部帧）
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
        frame: Any | None = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# ============================================================================
# 全局未捕获异常钩子
# ============================================================================


def _exception_hook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: Any,  # types.TracebackType | None; 用 Any 避免版本兼容问题
) -> None:
    """将未捕获的异常通过 loguru 记录为 CRITICAL 级别日志。"""
    _loguru_logger.opt(exception=(exc_type, exc_value, exc_tb)).critical(
        "未捕获的异常导致程序退出"
    )


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def setup_logger(config: LoggingConfig) -> None:
    """初始化全局日志系统。

    执行过程：
        1. 移除 loguru 内置的默认 handler。
        2. 添加按日 + 按大小轮转的文件 handler。
        3. 添加彩色控制台 handler。
        4. 安装标准 logging → loguru 桥接。
        5. 安装全局未捕获异常钩子。

    Args:
        config: LoggingConfig 实例，来自 Config.logging。

    Note:
        幂等 —— 重复调用会先移除已有 handler 再重新添加。
    """
    global _INITIALIZED

    if _INITIALIZED:
        _loguru_logger.warning("setup_logger 已初始化，移除旧 handler 并重新构建")

    # -- 1. 移除所有已有 handler ---------------------------------------
    _loguru_logger.remove()

    # -- 2. 确保日志目录存在（含回退） ---------------------------------
    log_dir = _resolve_log_dir(config.file)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 权限不足 / 磁盘满 / 路径非法 → 回退到系统临时目录
        log_dir = Path(tempfile.gettempdir()) / "miaowa_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[Miaowa] 日志目录不可用，已回退至: {log_dir}",
            file=sys.stderr,
        )

    # -- 3. 文件 handler -----------------------------------------------
    file_path = log_dir / "miaowa_{time:YYYY-MM-DD}.log"

    _loguru_logger.add(
        str(file_path),
        level=_resolve_level(config.file_level),
        format=_FILE_FORMAT,
        rotation=config.max_size,
        retention="7 days",
        compression="gz",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    # -- 4. 控制台 handler ---------------------------------------------
    console_level = _resolve_level(config.level)

    _loguru_logger.add(
        sys.stderr,
        level=console_level,
        format=_CONSOLE_FORMAT,
        colorize=True,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    # -- 5. 标准 logging 桥接（PRD §3.5.1） ----------------------------
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    # 降低第三方库日志噪音
    for lib in ("httpx", "openai", "urllib3", "httpcore"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # -- 6. 全局未捕获异常钩子 ------------------------------------------
    sys.excepthook = _exception_hook

    _INITIALIZED = True


def get_logger(name: str):  # type: ignore[no-untyped-def]
    """获取绑定模块名的 logger 实例。

    Args:
        name: 模块名，通常传入 ``__name__``。

    Returns:
        绑定了 name 字段的 loguru.Logger。

    Note:
        通过本函数返回的 logger 输出日志时，{name} 会自动填充为传入的模块名。
        若直接使用 ``from loguru import logger`` 而未调用 ``bind()``，
        日志中的 {name} 将为空字符串。

    Example::

        logger = get_logger(__name__)
        logger.info("Agent 推理完成")
        # 输出 → … | miaowa.agent.planner:think:142 | Agent 推理完成
    """
    return _loguru_logger.bind(name=name)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _resolve_log_dir(file_path: str) -> Path:
    """从配置的日志文件路径推导日志目录。

    Args:
        file_path: 配置中的日志文件路径（可能含 ~）。

    Returns:
        展开 ~ 后的日志目录 Path。
    """
    p = Path(file_path).expanduser()
    # 若末尾路径段带有扩展名特征（如 ".log"），取其父目录
    name = p.name
    if "." in name and p.suffix:
        return p.parent
    return p


def _resolve_level(level: str) -> str:
    """将配置中的日志级别字符串转换为 loguru 可识别的级别。

    支持的输入：
        - 标准级别名：DEBUG / INFO / WARNING / ERROR / CRITICAL
        - 数字级别（字符串形式）：如 "10" → DEBUG
        - 大小写不敏感

    Args:
        level: 配置中的日志级别字符串。

    Returns:
        标准化后的日志级别字符串（大写）。
    """
    level_upper = level.strip().upper()
    valid_levels = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
    if level_upper in valid_levels:
        return level_upper
    # 尝试数字解析
    try:
        level_no = int(level)
    except ValueError:
        return _DEFAULT_CONSOLE_LEVEL
    # 非法负数 → 默认
    if level_no < 0:
        return _DEFAULT_CONSOLE_LEVEL
    # loguru 数字级别：TRACE=5, DEBUG=10, INFO=20, SUCCESS=25,
    # WARNING=30, ERROR=40, CRITICAL=50
    if level_no <= 5:
        return "TRACE"
    if level_no <= 10:
        return "DEBUG"
    if level_no <= 20:
        return "INFO"
    if level_no <= 25:
        return "SUCCESS"
    if level_no <= 30:
        return "WARNING"
    if level_no <= 40:
        return "ERROR"
    return "CRITICAL"
