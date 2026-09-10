#***********************************************
#      Filename: utils.py
#   Description: 工具函数库
#***********************************************

import os
import yaml
from pathlib import Path
from datetime import datetime


# ===== CONFIG PATH =====

# 默认配置文件路径。
# 所有模块统一从这里引用（而非各自硬编码），避免出现「LLM 读 deepseek.yml、
# 搜索读 default.yml」这类不一致——default.yml 并不存在，会导致搜索初始化
# 直接抛 FileNotFoundError。
DEFAULT_CONFIG_PATH = "config/deepseek.yml"


def resolve_config_path() -> str:
    """解析当前使用的配置文件路径：环境变量 CONFIG_PATH 优先，否则回退默认值。"""
    return os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)


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
    """加载配置"""
    return get_config_yml(
        path=config_path, section_name="stages", subsection_name=stage_name
    )
