#***********************************************
#      Filename: __init__.py
#   Description: 包初始化 —— 自动加载项目根目录的 .env
#***********************************************

# 「懒人模式」入口：只要项目根目录存在 .env，就在导入本包时自动加载，
# 于是 config/*.yml 里的 ${QWEN_API_KEY} / ${TAVILY_API_KEY} 等占位符
# 会被自动替换成真实密钥，用户无需改任何 YAML。
#
# 使用 load_dotenv(override=False)：已存在的真实环境变量优先，.env 只做补充。

from pathlib import Path as _Path

_ENV_FILE = _Path(__file__).resolve().parent.parent / ".env"

if _ENV_FILE.is_file():
    try:
        from dotenv import load_dotenv as _load_dotenv

        _load_dotenv(_ENV_FILE, override=False)
    except ImportError:  # python-dotenv 未安装时静默跳过，不影响正常配置方式
        pass
