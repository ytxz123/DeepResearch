#***********************************************
#      Filename: llm.py
#   Description: 大模型客户端 
#***********************************************


from __future__ import annotations

import os
from typing import Any, Dict, Optional
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException

from deep_research.utils import load_config, resolve_config_path
from deep_research import logging as dr_logging


logger = dr_logging.get_logger(__name__)

# 缓存CONFIG，避免重复导入(config_path, stage, loader_id) 
_CONFIG_CACHE: Dict[tuple[str, str, int], Dict[str, Any]] = {}

# 默认的stage
DEFAULT_STAGE = "prod"


class LLMConfigError(ValueError):
    """当LLM配置错误或者不合法时抛出该异常"""


def _resolve_stage(stage: str | None) -> str:
    return stage or os.environ.get("STAGE") or DEFAULT_STAGE


def _load_stage_config(stage_name: str | None, config_path: str | None) -> Dict[str, Any]:
    """加载config文件"""

    # Key作为config loader的唯一标识
    cache_key = (resolve_config_path(), stage_name, id(load_config))

    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    cfg = load_config(stage_name=stage_name, config_path=config_path)
    if cfg is None:
        raise LLMConfigError(f"No config found for stage '{stage_name}'")

    _CONFIG_CACHE[cache_key] = cfg
    return cfg


def _build_openai_kwargs(
    handle: str,
    api_cfg: Dict[str, Any],
    max_tokens: int | None,
    timeout_seconds: Optional[int],
) -> Dict[str, Any]:
    """初始化llm client参数，例如api_key, base_url"""

    model = handle or api_cfg.get("default_model")
    if not model:
        raise LLMConfigError("OpenAI config requires a model name!")

    kwargs: Dict[str, Any] = {
        "model": model,
        # 显式指定 provider：deepseek-* 无内置前缀规则，交给 init_chat_model 推断会失败
        "model_provider": "openai",
    }

    # api_key, base_url
    for key in ("api_key", "base_url", "organization"):
        if api_cfg.get(key):
            kwargs[key] = api_cfg[key]

    # 温度系数
    if api_cfg.get("temperature") is not None:
        kwargs["temperature"] = api_cfg["temperature"]

    # 最大token数
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    # 请求超时
    if timeout_seconds is not None:
        # OpenAI兼容接口可用接收timeout/request_timeout
        kwargs["timeout"] = timeout_seconds
        kwargs["request_timeout"] = timeout_seconds

    # 厂商自定义参数（可选），会原样透传到请求体的 extra_body 中
    extra_body = {k: v for k, v in (api_cfg.get("extra_body") or {}).items() if v is not None}
    if extra_body:
        kwargs["extra_body"] = extra_body

    return kwargs


def _resolve_config_max_tokens(api_cfg: Dict[str, Any], handle: str) -> int | None:
    """解析最大token数"""
    models_cfg = api_cfg.get("models") or {}
    model_cfg = models_cfg.get(handle) or {}
    return model_cfg.get("max_tokens")


def _resolve_timeout_seconds(api_cfg: Dict[str, Any], role_cfg: Dict[str, Any]) -> Optional[int]:
    """解析timeout/request_timeout/timeout_seconds参数"""
    for cfg in (role_cfg, api_cfg):
        for key in ("timeout", "request_timeout", "timeout_seconds"):
            if cfg.get(key) is not None:
                return cfg.get(key)
    return None


def _build_kwargs(
    backend: str,
    handle: str,
    api_cfg: Dict[str, Any],
    role_cfg: Dict[str, Any],
    max_tokens: int | None,
    timeout_seconds: int | None,
) -> Dict[str, Any]:

    if backend == "openai":
        return _build_openai_kwargs(handle, api_cfg, max_tokens, timeout_seconds)
    else:
        raise LLMConfigError(f"Unsupported backend '{backend}'")


def get_chat_model(role: str, *, stage: str | None = None, max_tokens: int | None = None):
    """根据config和role返回LLM client.

    Args:
        role: 角色名，例如supervisor, writer
        stage: stage name 
        max_tokens: 最大tokens 
    """

    config_path = resolve_config_path()
    resolved_stage = _resolve_stage(stage)

    cfg = _load_stage_config(resolved_stage, config_path)

    roles_cfg = cfg.get("roles", {})
    if role not in roles_cfg:
        # 清除cache重新加载一次
        _CONFIG_CACHE.clear()
        cfg = _load_stage_config(resolved_stage, config_path)
        roles_cfg = cfg.get("roles", {})

    # 如果role配置错误
    if role not in roles_cfg:
        available = ", ".join(sorted(roles_cfg.keys())) or "<none>"
        raise LLMConfigError(
            f"Role '{role}' not found for stage '{resolved_stage}' using config '{config_path}'. Available: {available}"
        )

    role_cfg = roles_cfg[role]
    backend = role_cfg.get("backend")
    handle = role_cfg.get("handle")
    if not backend or not handle:
        raise LLMConfigError(f"Role '{role}' is missing backend or handle")

    # 解析llm api config
    api_cfg = cfg.get("cognition", {}).get(backend)
    if api_cfg is None:
        raise LLMConfigError(f"No cognition config for backend '{backend}'")

    resolved_timeout = _resolve_timeout_seconds(api_cfg, role_cfg)
    logger.info(
        "Selected cognition backend '%s' for role '%s' with handle '%s' (timeout=%s)",
        backend,
        role,
        handle,
        resolved_timeout,
    )

    # 输出上限优先级：调用方入参 > 角色级配置 > 模型级配置
    resolved_max_tokens = max_tokens
    if resolved_max_tokens is None:
        resolved_max_tokens = role_cfg.get("max_tokens")
    if resolved_max_tokens is None:
        resolved_max_tokens = _resolve_config_max_tokens(api_cfg, handle)

    kwargs = _build_kwargs(
        backend=backend,
        handle=handle,
        api_cfg=api_cfg,
        role_cfg=role_cfg,
        max_tokens=resolved_max_tokens,
        timeout_seconds=resolved_timeout
    )
    return init_chat_model(**kwargs)


def _resolve_structured_output_method(
    role_cfg: Dict[str, Any], api_cfg: Dict[str, Any]
) -> str | None:
    """解析结构化输出方法：优先 role 级配置，其次 backend 级配置。"""
    for cfg in (role_cfg, api_cfg):
        method = cfg.get("structured_output_method")
        if method:
            return method
    return None


def get_structured_output_method(role: str, *, stage: str | None = None) -> str | None:
    """读取 config 中 role 配置的 structured_output_method。

    不同厂商对结构化输出（response_format）的支持不同。DeepSeek 不支持
    json_schema（报错 "This response_format type is unavailable now"），
    需在 config 中配置 `structured_output_method: json_mode`。

    返回 None 表示使用 langchain 默认方法（json_schema）。
    """
    config_path = resolve_config_path()
    resolved_stage = _resolve_stage(stage)
    cfg = _load_stage_config(resolved_stage, config_path)
    roles_cfg = cfg.get("roles", {})
    role_cfg = roles_cfg.get(role, {})
    backend = role_cfg.get("backend")
    api_cfg = cfg.get("cognition", {}).get(backend, {}) if backend else {}
    return _resolve_structured_output_method(role_cfg, api_cfg)


def with_structured_output(model, schema, role, *, stage: str | None = None):
    """包装 model.with_structured_output，按 config 指定 method。

    未配置 method 时保持 langchain 默认行为（json_schema）；DeepSeek 等
    不支持 json_schema 的厂商通过 config 配置 `structured_output_method:
    json_mode` 兼容。

    注意：OpenAI/DeepSeek 的 json_object 模式要求 prompt 中必须包含
    "json" 字样，而 langchain 的 json_mode 不会自动注入格式指令。
    因此这里在输入消息前插入一条 SystemMessage（含字段 schema 与
    "json" 字样），否则 DeepSeek 会返回 400：
    "Prompt must contain the word 'json' in some form".
    """
    kwargs: Dict[str, Any] = {}
    method = get_structured_output_method(role, stage=stage)
    if method:
        kwargs["method"] = method
    structured = model.with_structured_output(schema, **kwargs)

    if method == "json_mode":
        try:
            format_instructions = PydanticOutputParser(
                pydantic_object=schema
            ).get_format_instructions()
        except Exception:
            format_instructions = (
                "Output ONLY valid JSON matching the requested schema. "
                "Do not include explanations or markdown code fences."
            )
        instruction = SystemMessage(content=format_instructions)

        def _inject(messages):
            if isinstance(messages, str):
                return [instruction, HumanMessage(content=messages)]
            return [instruction] + list(messages)

        # DeepSeek 这类 thinking 模型偶发输出无法解析的 JSON（如 "[1]"），
        # 对解析失败做一次重试，避免整个调研流程中途崩溃。
        return (RunnableLambda(_inject) | structured).with_retry(
            retry_if_exception_type=(OutputParserException,),
            wait_exponential_jitter=False,
            stop_after_attempt=2,
        )

    return structured
