#!/usr/bin/env python
#***********************************************
#      Filename: logging.py
#   Description: 日志打印工具
#***********************************************


"""
Deep Research 的集中式日志配置。
默认情况下提供 stdout/stderr 控制台处理程序，也可以重定向到文件。
日志打印等级可通过 `DEEP_RESEARCH_LOG_LEVEL` 环境变量来控制。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


# 直接使用标准库 logging：本项目的 logger 与 langchain / openai / httpx 等三方库
# 共用同一个 root，setup_logging 装的 handler 才能一并接住三方库日志。
getLogger = logging.getLogger
Logger = logging.Logger
StreamHandler = logging.StreamHandler
FileHandler = logging.FileHandler
Formatter = logging.Formatter

_LOG_CONFIGURED = False
_LOG_DIR = Path("/log/deepresearch")
_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_ENV_LEVEL_KEY = "DEEP_RESEARCH_LOG_LEVEL"


class _MaxLevelFilter(logging.Filter):
    """只允许不大于设定级别的日志打印"""

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003  (filter name)
        return record.levelno <= self.level


def _parse_level(level: Optional[str | int]) -> int:
    """解析日志的等级"""

    env_level = level if level is not None else os.getenv(_ENV_LEVEL_KEY, "INFO")

    if isinstance(env_level, int):
        return env_level

    return logging._nameToLevel.get(str(env_level).upper(), logging.INFO)  # type: ignore[attr-defined]


def setup_logging(level: Optional[str | int] = None) -> logging.Logger:
    """
    配置root logger，包括console和文件
    - Console：INFO 及以下级别输出到标准输出 (stdout)；WARNING 及以上级别输出到标准错误输出 (stderr)。
    - File logging：仅当 ``/log/deepresearch/`` 存在时才启用。
    """

    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return logging.getLogger()

    root = logging.getLogger()
    root.setLevel(_parse_level(level))

    # 设置打印格式
    formatter = logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    # INFO及以下写到stdout
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    # WARNING及以上写到stderr
    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    # 写到文件
    if _LOG_DIR.exists() and _LOG_DIR.is_dir():
        try:
            app_log_path = _LOG_DIR / "app.log"
            error_log_path = _LOG_DIR / "app.error.log"

            file_handler = logging.FileHandler(app_log_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

            error_file_handler = logging.FileHandler(error_log_path)
            error_file_handler.setLevel(logging.WARNING)
            error_file_handler.setFormatter(formatter)
            root.addHandler(error_file_handler)
        except OSError:
            # 如果无法写入文件，则继续仅使用Console
            pass

    _LOG_CONFIGURED = True
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """返回一个具有全局配置的logger."""

    setup_logging()
    return logging.getLogger(name)


__all__ = [
    "getLogger",
    "Logger",
    "StreamHandler",
    "FileHandler",
    "Formatter",
    "setup_logging",
    "get_logger",
]
