#***********************************************
#      Filename: utils.py
#   Description: 工具函数库
#***********************************************

import os
import re
import yaml
from pathlib import Path
from datetime import datetime


# ===== ENV EXPANSION =====

# 匹配 ${VAR} 与 ${VAR:默认值}，用于让配置文件从环境变量（含 .env）读取密钥。
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _expand_env(value):
    """递归展开配置中的 ${VAR} / ${VAR:默认值} 占位符。

    - 环境变量已设置且非空 → 用其值（这是「懒人模式」：密钥写在 .env 里即可）
    - 未设置或为空 → 有默认值则用默认值；无默认值则原样保留，便于排查
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda m: os.environ.get(m.group(1))
            or (m.group(2) if m.group(2) is not None else m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


# ===== CONFIG PATH =====

# 默认配置文件的唯一来源，各模块统一引用，避免默认值漂移。
DEFAULT_CONFIG_PATH = "config/deepseek.yml"


def resolve_config_path() -> str:
    """解析当前使用的配置文件路径：环境变量 CONFIG_PATH 优先，否则回退默认值。"""
    return os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)


# ===== DEPTH PRESET =====

# 调研深度档位：一条指令同时控制主管决策轮数与并行子代理数。
# 轮数上限不宜设高：实测报告质量通常在前几轮就收敛，之后再迭代只是重复检索。
DEPTH_PRESETS = {
    "quick":    {"max_iterations": 3,  "max_concurrent": 2},
    "standard": {"max_iterations": 6,  "max_concurrent": 3},
    "deep":     {"max_iterations": 10, "max_concurrent": 3},
}

DEFAULT_DEPTH = "standard"


def resolve_depth(name: str | None = None) -> dict:
    """解析深度档位：入参 > 环境变量 RESEARCH_DEPTH > 默认档。

    返回值形如 {"max_iterations": 3, "max_concurrent": 3}。
    """
    depth = (name or os.environ.get("RESEARCH_DEPTH") or DEFAULT_DEPTH).strip().lower()
    if depth not in DEPTH_PRESETS:
        raise ValueError(
            f"未知的调研深度档位 '{depth}'，可选：{', '.join(DEPTH_PRESETS)}"
        )
    return DEPTH_PRESETS[depth]


# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """获取今天的日期并返回格式化的字符串。

    注意：`%-d`（去前导零）是 Linux/macOS 的 GNU 扩展，
    Windows 的 strftime 不支持，会抛 ValueError: Invalid format string，
    因此这里手动拼接 day，保证跨平台行为一致。
    """
    now = datetime.now()
    return now.strftime("%a %b ") + str(now.day) + now.strftime(", %Y")

def get_current_dir() -> Path:
    """获取当前的目录"""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


# ===== CONFIG LOADER =====

def get_config_yml(path, section_name, subsection_name=None):
    """读取yaml文件"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    with open(path, encoding="utf8") as f:
        data = yaml.safe_load(f)
        # 展开 ${VAR} / ${VAR:默认值}，让密钥可以从环境变量或 .env 注入
        data = _expand_env(data)
        try:
            return (
                data[section_name]
                if subsection_name is None
                else data[section_name][subsection_name]
            )
        except KeyError as e:
            raise KeyError(
                f"No such section or subsection in config file: {section_name}, {subsection_name}. Config file: {path}"
            ) from e


def load_config(stage_name=None, config_path=None):
    """加载配置。

    config_path 省略时走 resolve_config_path()（CONFIG_PATH 环境变量 → 默认值），
    避免传 None 直接落到 os.path.isfile(None) 报 TypeError。
    """
    return get_config_yml(
        path=config_path or resolve_config_path(),
        section_name="stages",
        subsection_name=stage_name,
    )
