"""openai_api 教学模块（OpenAI 概念 + DeepSeek OpenAI 兼容实调）。"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from collections import Counter
from typing import Callable, Literal, Required, TypedDict, cast, get_origin, get_type_hints

from openai import APIConnectionError, APITimeoutError, OpenAI
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from openai.types.embedding_create_params import EmbeddingCreateParams
from openai.types.moderation_create_params import ModerationCreateParams
from openai.types.responses.response_create_params import ResponseCreateParamsBase

DEFAULT_PROMPT = "请用简短中文说明 LLM API 常见能力与参数作用。"
ONLINE_ENDPOINTS = ("responses", "embeddings", "moderations")
DEFAULT_PROBE_LIMIT = 1
DEFAULT_PROBE_TIMEOUT_SECONDS = 30.0
TraceEvent = dict[str, object]

ProbeMode = Literal["safe_only", "requires_opt_in"]
CallMode = Literal["safe", "side_effect"]
FeatureStatus = Literal[
    "ok",
    "unsupported_by_provider",
    "auth_error",
    "rate_limited",
    "provider_server_error",
    "network_error",
    "skipped_by_policy",
    "unknown_error",
]
FeatureStability = Literal["stable", "preview", "compat_risk"]
SideEffectLevel = Literal["none", "low", "high"]


class EndpointCallResult(TypedDict):
    """核心三端点调用结果（兼容历史 API）。"""

    endpoint: str
    status: str
    http_status: int | None
    provider_error_type: str | None
    summary: str


class FeatureProbeResult(TypedDict):
    """全量 feature probe 结构化结果。"""

    feature_id: str
    status: FeatureStatus
    http_status: int | None
    provider_error_type: str | None
    summary: str
    call_mode: CallMode
    executed: bool


class FeatureLesson(TypedDict):
    """单个 API 功能教学卡片。"""

    feature_id: str
    display_name: str
    category: str
    stability: FeatureStability
    side_effect_level: SideEffectLevel
    api_surface: list[str]
    what_it_does: str
    when_to_use: str
    core_params: dict[str, str]
    all_param_names: list[str]
    required_params: list[str]
    sync_example: str
    async_example: str
    exercise_concept: str
    exercise_hands_on: str


class DemoResult(TypedDict):
    """教学模块统一返回结构。"""

    final_answer: str
    trace: list[TraceEvent]


class FeatureRegistryItem(TypedDict):
    """内部 feature 元数据。"""

    feature_id: str
    display_name: str
    category: str
    resource_path: str
    stability: FeatureStability
    side_effect_level: SideEffectLevel
    default_probe_mode: ProbeMode
    what_it_does: str
    when_to_use: str


def _feature(
    feature_id: str,
    display_name: str,
    category: str,
    resource_path: str,
    stability: FeatureStability,
    side_effect_level: SideEffectLevel,
    default_probe_mode: ProbeMode,
    what_it_does: str,
    when_to_use: str,
) -> FeatureRegistryItem:
    return {
        "feature_id": feature_id,
        "display_name": display_name,
        "category": category,
        "resource_path": resource_path,
        "stability": stability,
        "side_effect_level": side_effect_level,
        "default_probe_mode": default_probe_mode,
        "what_it_does": what_it_does,
        "when_to_use": when_to_use,
    }


FEATURE_REGISTRY: tuple[FeatureRegistryItem, ...] = (
    _feature(
        "responses",
        "Responses",
        "generation",
        "responses.create",
        "stable",
        "none",
        "safe_only",
        "统一生成接口，支持文本、多模态与工具调用。",
        "需要统一处理生成请求、工具调用和结构化输出时。",
    ),
    _feature(
        "chat",
        "Chat Completions",
        "generation",
        "chat.completions.create",
        "stable",
        "none",
        "safe_only",
        "经典对话生成接口，适合消息式输入。",
        "已有 chat 消息结构或迁移旧项目时。",
    ),
    _feature(
        "completions",
        "Completions",
        "generation",
        "completions.create",
        "compat_risk",
        "none",
        "safe_only",
        "旧版文本 completion 接口。",
        "维护历史 prompt-only 接口兼容时。",
    ),
    _feature(
        "embeddings",
        "Embeddings",
        "retrieval",
        "embeddings.create",
        "stable",
        "none",
        "safe_only",
        "将文本编码为向量。",
        "RAG 召回、语义搜索和相似度计算场景。",
    ),
    _feature(
        "moderations",
        "Moderations",
        "safety",
        "moderations.create",
        "stable",
        "none",
        "safe_only",
        "对输入内容进行风险审核。",
        "UGC 审核、合规前置检查。",
    ),
    _feature(
        "files",
        "Files",
        "data",
        "files.list",
        "stable",
        "low",
        "safe_only",
        "文件上传与管理。",
        "需要托管知识文件、训练数据或中间文件时。",
    ),
    _feature(
        "uploads",
        "Uploads",
        "data",
        "uploads.create",
        "preview",
        "high",
        "requires_opt_in",
        "分片上传能力，用于大文件。",
        "需要对大文件做断点/分片上传时。",
    ),
    _feature(
        "batches",
        "Batches",
        "orchestration",
        "batches.list",
        "stable",
        "low",
        "safe_only",
        "批量异步任务管理。",
        "离线高吞吐任务调度时。",
    ),
    _feature(
        "models",
        "Models",
        "management",
        "models.list",
        "stable",
        "none",
        "safe_only",
        "模型元信息查询。",
        "动态能力探测、模型白名单和版本检查。",
    ),
    _feature(
        "fine_tuning",
        "Fine Tuning",
        "training",
        "fine_tuning.jobs.list",
        "preview",
        "low",
        "safe_only",
        "微调任务生命周期管理。",
        "需要领域定制模型训练与追踪。",
    ),
    _feature(
        "vector_stores",
        "Vector Stores",
        "retrieval",
        "vector_stores.list",
        "stable",
        "low",
        "safe_only",
        "向量库管理和检索。",
        "文档检索增强、知识库问答。",
    ),
    _feature(
        "conversations",
        "Conversations",
        "conversation",
        "conversations.create",
        "preview",
        "high",
        "requires_opt_in",
        "会话容器管理。",
        "需要跨请求保留会话上下文时。",
    ),
    _feature(
        "realtime",
        "Realtime",
        "realtime",
        "realtime.client_secrets.create",
        "preview",
        "low",
        "safe_only",
        "实时会话和低延迟交互。",
        "语音助手、实时协作与 streaming 交互。",
    ),
    _feature(
        "webhooks",
        "Webhooks",
        "integration",
        "webhooks.verify_signature",
        "stable",
        "none",
        "safe_only",
        "Webhook 验签与回调事件校验。",
        "服务端接收异步回调并做安全校验。",
    ),
    _feature(
        "evals",
        "Evals",
        "evaluation",
        "evals.list",
        "preview",
        "low",
        "safe_only",
        "模型评测任务管理。",
        "构建自动化评测基线与质量回归。",
    ),
    _feature(
        "containers",
        "Containers",
        "runtime",
        "containers.list",
        "compat_risk",
        "low",
        "safe_only",
        "容器化运行时资源管理。",
        "需要隔离运行环境或沙箱资源时。",
    ),
    _feature(
        "skills",
        "Skills",
        "orchestration",
        "skills.list",
        "compat_risk",
        "low",
        "safe_only",
        "技能编排与版本管理。",
        "需要平台化路由能力与技能版本治理。",
    ),
    _feature(
        "videos",
        "Videos",
        "multimodal",
        "videos.list",
        "preview",
        "low",
        "safe_only",
        "视频生成任务与结果管理。",
        "视频生成、重混与轮询下载场景。",
    ),
    _feature(
        "images",
        "Images",
        "multimodal",
        "images.generate",
        "stable",
        "high",
        "requires_opt_in",
        "图像生成与编辑。",
        "营销素材、创意草图、视觉内容生成。",
    ),
    _feature(
        "audio",
        "Audio",
        "multimodal",
        "audio.transcriptions.create",
        "stable",
        "high",
        "requires_opt_in",
        "音频转写、翻译与语音合成。",
        "语音转文本、字幕生成与语音交互。",
    ),
    _feature(
        "beta",
        "Beta",
        "preview",
        "beta.assistants.list",
        "preview",
        "low",
        "safe_only",
        "实验性 API 集合。",
        "提前验证新能力并评估升级影响。",
    ),
    _feature(
        "client_with_options",
        "Client with_options/copy",
        "client_ops",
        "client.with_options(...).responses.create",
        "stable",
        "none",
        "safe_only",
        "基于同一 client 快速覆盖 timeout/retry/header 等选项。",
        "按请求级别调整客户端配置时。",
    ),
    _feature(
        "client_with_raw_response",
        "Client with_raw_response",
        "client_ops",
        "client.with_raw_response.responses.create",
        "stable",
        "none",
        "safe_only",
        "直接获取原始 HTTP 响应对象。",
        "需要调试状态码、headers、原始 body 时。",
    ),
    _feature(
        "client_with_streaming_response",
        "Client with_streaming_response",
        "client_ops",
        "client.with_streaming_response.responses.create",
        "stable",
        "none",
        "safe_only",
        "流式读取响应并按需解析。",
        "低延迟输出、逐段消费与长响应场景。",
    ),
)


FEATURE_IDS: tuple[str, ...] = tuple(item["feature_id"] for item in FEATURE_REGISTRY)


_FEATURE_CORE_PARAMS: dict[str, dict[str, str]] = {
    "responses": {
        "model": "模型名称，决定能力、延迟与成本。",
        "input": "输入内容，可为文本或结构化输入。",
        "instructions": "系统级指令，约束回答风格和任务边界。",
        "tools": "函数/工具定义，支持 tool calling。",
        "temperature": "采样随机度，越高越发散。",
        "max_output_tokens": "限制输出长度与成本。",
    },
    "chat": {
        "model": "模型名称。",
        "messages": "消息数组，chat 接口核心输入。",
        "tools": "可调用工具定义。",
        "tool_choice": "工具选择策略。",
        "temperature": "采样随机度。",
        "max_completion_tokens": "输出 token 上限。",
    },
    "completions": {
        "model": "completion 模型名称（多为历史接口）。",
        "prompt": "纯文本 prompt。",
        "max_tokens": "输出长度上限。",
        "temperature": "采样随机度。",
        "top_p": "核采样概率阈值。",
    },
    "embeddings": {
        "model": "向量模型名称。",
        "input": "待向量化文本。",
        "dimensions": "向量维度（模型支持时可配置）。",
        "encoding_format": "向量编码格式（float/base64）。",
    },
    "moderations": {
        "model": "审核模型名称。",
        "input": "待审核文本或多模态内容。",
    },
    "files": {
        "purpose": "文件用途标识。",
        "file": "上传文件内容。",
        "limit": "列表分页大小。",
    },
    "uploads": {
        "bytes": "总字节数。",
        "filename": "上传文件名。",
        "mime_type": "文件类型。",
        "purpose": "用途标识。",
    },
    "batches": {
        "input_file_id": "批处理输入文件 ID。",
        "endpoint": "批处理目标端点。",
        "completion_window": "批处理时间窗口。",
    },
    "models": {
        "model": "模型 ID。",
    },
    "fine_tuning": {
        "training_file": "训练文件 ID。",
        "model": "基座模型。",
        "hyperparameters": "训练超参数。",
    },
    "vector_stores": {
        "name": "向量库名称。",
        "file_ids": "关联文件 ID 列表。",
        "limit": "分页大小。",
    },
    "conversations": {
        "items": "初始会话输入项。",
        "metadata": "会话元数据。",
    },
    "realtime": {
        "session": "实时会话配置。",
        "expires_after": "临时令牌有效期。",
    },
    "webhooks": {
        "payload": "原始回调 payload。",
        "headers": "回调请求头。",
        "secret": "Webhook 签名密钥。",
    },
    "evals": {
        "name": "评测名称。",
        "data_source_config": "评测数据源配置。",
        "testing_criteria": "评测判定标准。",
    },
    "containers": {
        "name": "容器名称。",
        "limit": "分页大小。",
    },
    "skills": {
        "name": "技能名称。",
        "description": "技能说明。",
        "tools": "技能工具定义。",
    },
    "videos": {
        "prompt": "视频生成提示词。",
        "model": "视频模型名称。",
        "seconds": "视频时长。",
    },
    "images": {
        "prompt": "图像生成提示词。",
        "model": "图像模型名称。",
        "size": "图像尺寸。",
        "quality": "图像质量档位。",
    },
    "audio": {
        "file": "音频文件输入。",
        "model": "音频模型名称。",
        "language": "语种提示。",
        "response_format": "返回格式。",
    },
    "beta": {
        "limit": "分页大小。",
    },
    "client_with_options": {
        "timeout": "临时覆盖请求超时。",
        "max_retries": "临时覆盖重试次数。",
    },
    "client_with_raw_response": {
        "status_code": "原始 HTTP 状态码。",
        "headers": "原始响应头。",
        "parse()": "将原始响应解析为 SDK 对象。",
    },
    "client_with_streaming_response": {
        "stream": "流式读取响应体。",
        "parse()": "流结束后解析完整对象。",
    },
}


_FEATURE_SYNC_EXAMPLES: dict[str, str] = {
    "responses": "response = client.responses.create(model=model, input=user_text)",
    "chat": "resp = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': user_text}])",
    "completions": "resp = client.completions.create(model=model, prompt=user_text, max_tokens=64)",
    "embeddings": "vec = client.embeddings.create(model=embed_model, input=[user_text])",
    "moderations": "moderation = client.moderations.create(model=mod_model, input=user_text)",
    "files": "page = client.files.list(limit=1)",
    "uploads": "upload = client.uploads.create(bytes=1024, filename='a.txt', mime_type='text/plain', purpose='assistants')",
    "batches": "page = client.batches.list(limit=1)",
    "models": "models = client.models.list()",
    "fine_tuning": "jobs = client.fine_tuning.jobs.list(limit=1)",
    "vector_stores": "stores = client.vector_stores.list(limit=1)",
    "conversations": "conversation = client.conversations.create(items=[...])",
    "realtime": "secret = client.realtime.client_secrets.create(session={'type': 'realtime', 'model': model})",
    "webhooks": "client.webhooks.verify_signature(payload, headers, secret=webhook_secret)",
    "evals": "evals = client.evals.list(limit=1)",
    "containers": "containers = client.containers.list(limit=1)",
    "skills": "skills = client.skills.list(limit=1)",
    "videos": "videos = client.videos.list(limit=1)",
    "images": "image = client.images.generate(model=image_model, prompt='a cat')",
    "audio": "transcript = client.audio.transcriptions.create(model=audio_model, file=audio_file)",
    "beta": "assistants = client.beta.assistants.list(limit=1)",
    "client_with_options": "resp = client.with_options(timeout=20.0).responses.create(model=model, input=user_text)",
    "client_with_raw_response": "raw = client.with_raw_response.responses.create(model=model, input=user_text)",
    "client_with_streaming_response": "with client.with_streaming_response.responses.create(model=model, input=user_text) as stream: ...",
}


_FEATURE_ASYNC_EXAMPLES: dict[str, str] = {
    feature_id: sync_example.replace("client.", "async_client.")
    for feature_id, sync_example in _FEATURE_SYNC_EXAMPLES.items()
}


_FEATURE_EXERCISES: dict[str, dict[str, str]] = {
    feature_id: {
        "concept": f"理解题：解释 `{feature_id}` 与 `responses` 在职责上的差异，并给出一个必须使用 `{feature_id}` 的场景。",
        "hands_on": f"动手题：基于 `{feature_id}` 写一个最小示例，记录 status、耗时和失败分类。",
    }
    for feature_id in FEATURE_IDS
}


def _get_registry_map() -> dict[str, FeatureRegistryItem]:
    return {item["feature_id"]: item for item in FEATURE_REGISTRY}


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def _get_probe_limit() -> int:
    return _parse_int_env("OPENAI_API_PROBE_LIMIT", DEFAULT_PROBE_LIMIT)


def _get_probe_timeout_seconds() -> float:
    return _parse_float_env("OPENAI_API_PROBE_TIMEOUT_SECONDS", DEFAULT_PROBE_TIMEOUT_SECONDS)


def _env_include_side_effect_calls() -> bool:
    return _read_bool_env("OPENAI_API_INCLUDE_SIDE_EFFECT_CALLS", default=False)


def _get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("missing DEEPSEEK_API_KEY; online mode requires a valid API key")
    return OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def _get_introspection_client() -> OpenAI:
    return OpenAI(api_key="introspection-only", base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def _get_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _get_embedding_model() -> str:
    return os.getenv("DEEPSEEK_EMBEDDING_MODEL", _get_model())


def _get_moderation_model() -> str:
    return os.getenv("DEEPSEEK_MODERATION_MODEL", _get_model())


def _get_image_model() -> str:
    return os.getenv("DEEPSEEK_IMAGE_MODEL", _get_model())


def _get_audio_model() -> str:
    return os.getenv("DEEPSEEK_AUDIO_MODEL", _get_model())


def _extract_param_names(typed_dict_cls: type[object]) -> list[str]:
    annotations = getattr(typed_dict_cls, "__annotations__", {})
    if not isinstance(annotations, dict):
        return []
    return sorted(str(name) for name in annotations)


def _extract_required_param_names(typed_dict_cls: type[object]) -> list[str]:
    required_keys = getattr(typed_dict_cls, "__required_keys__", set())
    if isinstance(required_keys, (set, frozenset)) and required_keys:
        return sorted(str(name) for name in required_keys)

    try:
        hints = get_type_hints(typed_dict_cls, include_extras=True)
    except (NameError, TypeError, AttributeError):
        return []

    fallback_required: list[str] = []
    for name, annotation in hints.items():
        if get_origin(annotation) is Required:
            fallback_required.append(str(name))
    return sorted(fallback_required)


def _extract_signature_details(target: object | None) -> tuple[list[str], list[str], str | None]:
    if target is None:
        return [], [], "SDK 中未找到可提取签名的目标调用。"
    if not callable(target):
        return [], [], "目标对象不是可调用对象。"

    callable_target = cast(Callable[..., object], target)

    try:
        signature = inspect.signature(callable_target)
    except (TypeError, ValueError) as exc:
        return [], [], f"签名提取失败：{exc.__class__.__name__}: {exc}"

    names: list[str] = []
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        names.append(name)
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        } and parameter.default is inspect.Parameter.empty:
            required.append(name)

    if not names:
        return [], [], "签名中没有可用参数。"
    return names, required, None


def _get_feature_callable(client: OpenAI, feature_id: str) -> object | None:
    if feature_id == "responses":
        return client.responses.create
    if feature_id == "chat":
        return client.chat.completions.create
    if feature_id == "completions":
        return client.completions.create
    if feature_id == "embeddings":
        return client.embeddings.create
    if feature_id == "moderations":
        return client.moderations.create
    if feature_id == "files":
        return client.files.list
    if feature_id == "uploads":
        return client.uploads.create
    if feature_id == "batches":
        return client.batches.list
    if feature_id == "models":
        return client.models.list
    if feature_id == "fine_tuning":
        return client.fine_tuning.jobs.list
    if feature_id == "vector_stores":
        return client.vector_stores.list
    if feature_id == "conversations":
        return client.conversations.create
    if feature_id == "realtime":
        return client.realtime.client_secrets.create
    if feature_id == "webhooks":
        return client.webhooks.verify_signature
    if feature_id == "evals":
        return client.evals.list
    if feature_id == "containers":
        return client.containers.list
    if feature_id == "skills":
        return client.skills.list
    if feature_id == "videos":
        return client.videos.list
    if feature_id == "images":
        return client.images.generate
    if feature_id == "audio":
        return client.audio.transcriptions.create
    if feature_id == "beta":
        return client.beta.assistants.list
    if feature_id == "client_with_options":
        return client.with_options(timeout=_get_probe_timeout_seconds()).responses.create
    if feature_id == "client_with_raw_response":
        return client.with_raw_response.responses.create
    if feature_id == "client_with_streaming_response":
        return client.with_streaming_response.responses.create
    return None


def _clip_text(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _extract_response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output_items = getattr(response, "output", None)
    if not isinstance(output_items, list):
        return ""

    text_fragments: list[str] = []
    for item in output_items:
        content_items = getattr(item, "content", None)
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            maybe_text = getattr(content, "text", None)
            if isinstance(maybe_text, str) and maybe_text.strip():
                text_fragments.append(maybe_text.strip())

    return " ".join(text_fragments).strip()


def _extract_chat_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    parts.append(maybe_text)
        return " ".join(part.strip() for part in parts if isinstance(part, str)).strip()
    return ""


def _extract_completion_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return ""
    text = getattr(choices[0], "text", None)
    return text.strip() if isinstance(text, str) else ""


def _extract_data_count(payload: object) -> int | None:
    data = getattr(payload, "data", None)
    if isinstance(data, list):
        return len(data)
    return None


def _extract_provider_error_type(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None

    error_obj = body.get("error")
    if not isinstance(error_obj, dict):
        return None

    maybe_type = error_obj.get("type")
    if isinstance(maybe_type, str) and maybe_type.strip():
        return maybe_type.strip()

    maybe_code = error_obj.get("code")
    if isinstance(maybe_code, str) and maybe_code.strip():
        return maybe_code.strip()
    return None


def _classify_api_error(exc: Exception) -> tuple[FeatureStatus, int | None, str | None]:
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return "network_error", None, exc.__class__.__name__

    status_obj = getattr(exc, "status_code", None)
    status_code = int(status_obj) if isinstance(status_obj, int) else None
    provider_error_type = _extract_provider_error_type(exc) or exc.__class__.__name__

    if status_code in {401, 403}:
        return "auth_error", status_code, provider_error_type
    if status_code in {404, 405, 501}:
        return "unsupported_by_provider", status_code, provider_error_type
    if status_code == 429:
        return "rate_limited", status_code, provider_error_type
    if isinstance(status_code, int) and status_code >= 500:
        return "provider_server_error", status_code, provider_error_type
    return "unknown_error", status_code, provider_error_type


def _build_error_result(endpoint: str, exc: Exception) -> EndpointCallResult:
    status, http_status, provider_error_type = _classify_api_error(exc)
    summary = f"{endpoint} 调用失败：{exc.__class__.__name__}: {str(exc)}"
    return {
        "endpoint": endpoint,
        "status": status,
        "http_status": http_status,
        "provider_error_type": provider_error_type,
        "summary": summary,
    }


def _build_probe_error_result(feature_id: str, call_mode: CallMode, exc: Exception) -> FeatureProbeResult:
    status, http_status, provider_error_type = _classify_api_error(exc)
    return {
        "feature_id": feature_id,
        "status": status,
        "http_status": http_status,
        "provider_error_type": provider_error_type,
        "summary": f"{feature_id} 调用失败：{exc.__class__.__name__}: {str(exc)}",
        "call_mode": call_mode,
        "executed": True,
    }


def _build_probe_ok_result(feature_id: str, summary: str, call_mode: CallMode) -> FeatureProbeResult:
    return {
        "feature_id": feature_id,
        "status": "ok",
        "http_status": None,
        "provider_error_type": None,
        "summary": summary,
        "call_mode": call_mode,
        "executed": True,
    }


def _build_probe_skip_result(feature_id: str, reason: str) -> FeatureProbeResult:
    return {
        "feature_id": feature_id,
        "status": "skipped_by_policy",
        "http_status": None,
        "provider_error_type": None,
        "summary": reason,
        "call_mode": "side_effect",
        "executed": False,
    }


def _probe_responses(client: OpenAI, text: str) -> str:
    response = client.responses.create(
        model=_get_model(),
        input=text,
        instructions="你是教学助手。请用不超过80字总结 LLM API 能力。",
        max_output_tokens=80,
        timeout=_get_probe_timeout_seconds(),
    )
    response_text = _extract_response_text(response) or "已返回响应对象（未提取到文本）"
    return f"responses 成功：{_clip_text(response_text)}"


def _probe_chat(client: OpenAI, text: str) -> str:
    response = client.chat.completions.create(
        model=_get_model(),
        messages=[{"role": "user", "content": text}],
        max_tokens=80,
        timeout=_get_probe_timeout_seconds(),
    )
    content = _extract_chat_text(response) or "已返回 chat completion（未提取到文本）"
    return f"chat.completions 成功：{_clip_text(content)}"


def _probe_completions(client: OpenAI, text: str) -> str:
    response = client.completions.create(
        model=_get_model(),
        prompt=text,
        max_tokens=64,
        timeout=_get_probe_timeout_seconds(),
    )
    content = _extract_completion_text(response) or "已返回 completion（未提取到文本）"
    return f"completions 成功：{_clip_text(content)}"


def _probe_embeddings(client: OpenAI, text: str) -> str:
    response = client.embeddings.create(
        model=_get_embedding_model(),
        input=[text],
        encoding_format="float",
        timeout=_get_probe_timeout_seconds(),
    )
    data = getattr(response, "data", None)
    vector_count = len(data) if isinstance(data, list) else 0

    first_dim: int | None = None
    if isinstance(data, list) and data:
        first_embedding = getattr(data[0], "embedding", None)
        if isinstance(first_embedding, list):
            first_dim = len(first_embedding)

    dim_text = str(first_dim) if isinstance(first_dim, int) else "未知"
    return f"embeddings 成功：向量条数={vector_count}，首条维度={dim_text}"


def _probe_moderations(client: OpenAI, text: str) -> str:
    response = client.moderations.create(
        model=_get_moderation_model(),
        input=text,
        timeout=_get_probe_timeout_seconds(),
    )
    results = getattr(response, "results", None)
    flagged_text = "未知"
    if isinstance(results, list) and results:
        flagged_obj = getattr(results[0], "flagged", None)
        if isinstance(flagged_obj, bool):
            flagged_text = str(flagged_obj)

    return f"moderations 成功：flagged={flagged_text}"


def _probe_files(client: OpenAI, _text: str) -> str:
    page = client.files.list(limit=_get_probe_limit(), timeout=_get_probe_timeout_seconds())
    count = _extract_data_count(page)
    return f"files.list 成功：返回条数={count if isinstance(count, int) else '未知'}"


def _probe_uploads(client: OpenAI, _text: str) -> str:
    upload = client.uploads.create(
        bytes=1,
        filename="openai-api-probe.txt",
        mime_type="text/plain",
        purpose="assistants",
        timeout=_get_probe_timeout_seconds(),
    )
    upload_id = getattr(upload, "id", None)
    upload_id_text = str(upload_id) if upload_id is not None else "未知"
    return f"uploads.create 成功：upload_id={upload_id_text}"


def _probe_batches(client: OpenAI, _text: str) -> str:
    page = client.batches.list(limit=_get_probe_limit(), timeout=_get_probe_timeout_seconds())
    count = _extract_data_count(page)
    return f"batches.list 成功：返回条数={count if isinstance(count, int) else '未知'}"


def _probe_models(client: OpenAI, _text: str) -> str:
    page = client.models.list(timeout=_get_probe_timeout_seconds())
    count = _extract_data_count(page)
    return f"models.list 成功：模型数量={count if isinstance(count, int) else '未知'}"


def _probe_fine_tuning(client: OpenAI, _text: str) -> str:
    page = client.fine_tuning.jobs.list(limit=_get_probe_limit(), timeout=_get_probe_timeout_seconds())
    count = _extract_data_count(page)
    return f"fine_tuning.jobs.list 成功：返回条数={count if isinstance(count, int) else '未知'}"


def _probe_vector_stores(client: OpenAI, _text: str) -> str:
    page = client.vector_stores.list(limit=_get_probe_limit(), timeout=_get_probe_timeout_seconds())
    count = _extract_data_count(page)
    return f"vector_stores.list 成功：返回条数={count if isinstance(count, int) else '未知'}"


def _probe_conversations(client: OpenAI, _text: str) -> str:
    response = client.conversations.create(timeout=_get_probe_timeout_seconds())
    conversation_id = getattr(response, "id", None)
    conversation_id_text = str(conversation_id) if conversation_id is not None else "未知"
    return f"conversations.create 成功：conversation_id={conversation_id_text}"


def _probe_realtime(client: OpenAI, _text: str) -> str:
    response = client.realtime.client_secrets.create(
        session={"type": "realtime", "model": _get_model()},
        timeout=_get_probe_timeout_seconds(),
    )
    secret_obj = getattr(response, "client_secret", None)
    expires_at = getattr(secret_obj, "expires_at", None)
    expires_text = str(expires_at) if expires_at is not None else "未知"
    return f"realtime.client_secrets.create 成功：expires_at={expires_text}"


def _probe_webhooks(client: OpenAI, _text: str) -> str:
    has_verify = hasattr(client.webhooks, "verify_signature")
    has_unwrap = hasattr(client.webhooks, "unwrap")
    return f"webhooks 为本地能力：verify_signature={has_verify}，unwrap={has_unwrap}（未发起网络请求）"


def _probe_evals(client: OpenAI, _text: str) -> str:
    page = client.evals.list(limit=_get_probe_limit(), timeout=_get_probe_timeout_seconds())
    count = _extract_data_count(page)
    return f"evals.list 成功：返回条数={count if isinstance(count, int) else '未知'}"


def _probe_containers(client: OpenAI, _text: str) -> str:
    page = client.containers.list(limit=_get_probe_limit(), timeout=_get_probe_timeout_seconds())
    count = _extract_data_count(page)
    return f"containers.list 成功：返回条数={count if isinstance(count, int) else '未知'}"


def _probe_skills(client: OpenAI, _text: str) -> str:
    page = client.skills.list(limit=_get_probe_limit(), timeout=_get_probe_timeout_seconds())
    count = _extract_data_count(page)
    return f"skills.list 成功：返回条数={count if isinstance(count, int) else '未知'}"


def _probe_videos(client: OpenAI, _text: str) -> str:
    page = client.videos.list(limit=_get_probe_limit(), timeout=_get_probe_timeout_seconds())
    count = _extract_data_count(page)
    return f"videos.list 成功：返回条数={count if isinstance(count, int) else '未知'}"


def _probe_images(client: OpenAI, text: str) -> str:
    response = client.images.generate(
        model=_get_image_model(),
        prompt=_clip_text(text, 80),
        n=1,
        timeout=_get_probe_timeout_seconds(),
    )
    count = _extract_data_count(response)
    return f"images.generate 成功：返回条数={count if isinstance(count, int) else '未知'}"


def _probe_audio(client: OpenAI, _text: str) -> str:
    fake_audio = ("probe.wav", b"RIFF\x24\x00\x00\x00WAVEfmt ", "audio/wav")
    response = client.audio.transcriptions.create(
        model=_get_audio_model(),
        file=fake_audio,
        timeout=_get_probe_timeout_seconds(),
    )
    if isinstance(response, str):
        return f"audio.transcriptions.create 成功：{_clip_text(response)}"

    text_value = getattr(response, "text", None)
    if isinstance(text_value, str):
        return f"audio.transcriptions.create 成功：{_clip_text(text_value)}"
    return "audio.transcriptions.create 成功：已返回转写对象"


def _probe_beta(client: OpenAI, _text: str) -> str:
    page = client.beta.assistants.list(limit=_get_probe_limit(), timeout=_get_probe_timeout_seconds())
    count = _extract_data_count(page)
    return f"beta.assistants.list 成功：返回条数={count if isinstance(count, int) else '未知'}"


def _probe_client_with_options(client: OpenAI, text: str) -> str:
    response = client.with_options(timeout=_get_probe_timeout_seconds()).responses.create(
        model=_get_model(),
        input=text,
        max_output_tokens=60,
    )
    response_text = _extract_response_text(response) or "已返回响应对象（未提取到文本）"
    return f"client.with_options 成功：{_clip_text(response_text)}"


def _probe_client_with_raw_response(client: OpenAI, text: str) -> str:
    raw_response = client.with_raw_response.responses.create(
        model=_get_model(),
        input=text,
        max_output_tokens=60,
        timeout=_get_probe_timeout_seconds(),
    )
    parsed = raw_response.parse()
    response_text = _extract_response_text(parsed) or "已返回响应对象（未提取到文本）"
    return f"client.with_raw_response 成功：status={raw_response.status_code}，text={_clip_text(response_text)}"


def _probe_client_with_streaming_response(client: OpenAI, text: str) -> str:
    with client.with_streaming_response.responses.create(
        model=_get_model(),
        input=text,
        max_output_tokens=60,
        timeout=_get_probe_timeout_seconds(),
    ) as stream_response:
        parsed = stream_response.parse()
        response_text = _extract_response_text(parsed) or "已返回响应对象（未提取到文本）"
        return (
            f"client.with_streaming_response 成功：status={stream_response.status_code}，"
            f"text={_clip_text(response_text)}"
        )


def _run_feature_probe(client: OpenAI, feature_id: str, user_text: str) -> str:
    if feature_id == "responses":
        return _probe_responses(client, user_text)
    if feature_id == "chat":
        return _probe_chat(client, user_text)
    if feature_id == "completions":
        return _probe_completions(client, user_text)
    if feature_id == "embeddings":
        return _probe_embeddings(client, user_text)
    if feature_id == "moderations":
        return _probe_moderations(client, user_text)
    if feature_id == "files":
        return _probe_files(client, user_text)
    if feature_id == "uploads":
        return _probe_uploads(client, user_text)
    if feature_id == "batches":
        return _probe_batches(client, user_text)
    if feature_id == "models":
        return _probe_models(client, user_text)
    if feature_id == "fine_tuning":
        return _probe_fine_tuning(client, user_text)
    if feature_id == "vector_stores":
        return _probe_vector_stores(client, user_text)
    if feature_id == "conversations":
        return _probe_conversations(client, user_text)
    if feature_id == "realtime":
        return _probe_realtime(client, user_text)
    if feature_id == "webhooks":
        return _probe_webhooks(client, user_text)
    if feature_id == "evals":
        return _probe_evals(client, user_text)
    if feature_id == "containers":
        return _probe_containers(client, user_text)
    if feature_id == "skills":
        return _probe_skills(client, user_text)
    if feature_id == "videos":
        return _probe_videos(client, user_text)
    if feature_id == "images":
        return _probe_images(client, user_text)
    if feature_id == "audio":
        return _probe_audio(client, user_text)
    if feature_id == "beta":
        return _probe_beta(client, user_text)
    if feature_id == "client_with_options":
        return _probe_client_with_options(client, user_text)
    if feature_id == "client_with_raw_response":
        return _probe_client_with_raw_response(client, user_text)
    if feature_id == "client_with_streaming_response":
        return _probe_client_with_streaming_response(client, user_text)
    raise ValueError(f"unsupported feature id: {feature_id}")


def build_capability_overview() -> list[dict[str, object]]:
    """构建 LLM API 能力全景矩阵。"""

    return [
        {
            "capability": "文本生成",
            "api_surface": ["responses", "chat.completions", "completions"],
            "what_it_does": "根据输入生成自然语言文本，可用于问答、总结、写作。",
            "when_to_use": "任何需要模型直接输出自然语言结果的场景。",
        },
        {
            "capability": "工具调用",
            "api_surface": ["responses.tools", "chat.completions.tools"],
            "what_it_does": "模型返回函数调用参数，应用侧执行工具后再回填结果。",
            "when_to_use": "需要让模型访问业务函数、数据库或外部系统时。",
        },
        {
            "capability": "结构化输出",
            "api_surface": ["responses.text", "chat.completions.response_format"],
            "what_it_does": "约束模型输出为 JSON 等稳定结构，便于程序解析。",
            "when_to_use": "需要自动化流水线消费模型输出时。",
        },
        {
            "capability": "多模态输入输出",
            "api_surface": ["images", "audio", "videos", "responses.input"],
            "what_it_does": "在同一次请求中处理文本、图片、音频与视频等内容。",
            "when_to_use": "图文问答、语音交互、视觉内容生成。",
        },
        {
            "capability": "检索与知识库",
            "api_surface": ["embeddings", "vector_stores", "files"],
            "what_it_does": "向量化、文件托管与知识检索基础设施。",
            "when_to_use": "RAG 检索、语义检索、企业知识库问答。",
        },
        {
            "capability": "安全与治理",
            "api_surface": ["moderations", "webhooks", "models"],
            "what_it_does": "审核、回调验签与能力探测。",
            "when_to_use": "UGC 合规、事件回调校验、生产可观测。",
        },
        {
            "capability": "批处理与编排",
            "api_surface": ["batches", "skills", "containers", "conversations"],
            "what_it_does": "异步批处理、技能编排与运行时容器化能力。",
            "when_to_use": "高吞吐离线任务与复杂流程编排。",
        },
        {
            "capability": "客户端调用模式",
            "api_surface": ["with_options", "with_raw_response", "with_streaming_response"],
            "what_it_does": "按请求覆盖配置、拿原始响应、流式消费结果。",
            "when_to_use": "调试复杂 API 行为或低延迟交互时。",
        },
    ]


def build_parameter_reference() -> dict[str, dict[str, object]]:
    """兼容历史接口：核心 4 端点参数参考。"""

    full_reference = build_full_parameter_reference()

    responses = full_reference.get("responses", {})
    chat = full_reference.get("chat", {})
    embeddings = full_reference.get("embeddings", {})
    moderations = full_reference.get("moderations", {})

    return {
        "responses": {
            "core_params": responses.get("core_params", _FEATURE_CORE_PARAMS["responses"]),
            "all_param_names": responses.get("all_param_names", []),
            "required_params": responses.get("required_params", []),
        },
        "chat_completions": {
            "core_params": chat.get("core_params", _FEATURE_CORE_PARAMS["chat"]),
            "all_param_names": chat.get("all_param_names", []),
            "required_params": chat.get("required_params", []),
        },
        "embeddings": {
            "core_params": embeddings.get("core_params", _FEATURE_CORE_PARAMS["embeddings"]),
            "all_param_names": embeddings.get("all_param_names", []),
            "required_params": embeddings.get("required_params", []),
        },
        "moderations": {
            "core_params": moderations.get("core_params", _FEATURE_CORE_PARAMS["moderations"]),
            "all_param_names": moderations.get("all_param_names", []),
            "required_params": moderations.get("required_params", []),
        },
    }


def build_full_parameter_reference() -> dict[str, dict[str, object]]:
    """全量参数索引：覆盖 FEATURE_REGISTRY 的每个 feature。"""

    introspection_client = _get_introspection_client()
    reference: dict[str, dict[str, object]] = {}

    for feature in FEATURE_REGISTRY:
        feature_id = feature["feature_id"]
        core_params = _FEATURE_CORE_PARAMS.get(feature_id, {})

        if feature_id == "responses":
            all_param_names = _extract_param_names(ResponseCreateParamsBase)
            required_params = _extract_required_param_names(ResponseCreateParamsBase)
            extraction_note: str | None = None
        elif feature_id == "chat":
            all_param_names = _extract_param_names(CompletionCreateParamsBase)
            required_params = _extract_required_param_names(CompletionCreateParamsBase)
            extraction_note = None
        elif feature_id == "embeddings":
            all_param_names = _extract_param_names(EmbeddingCreateParams)
            required_params = _extract_required_param_names(EmbeddingCreateParams)
            extraction_note = None
        elif feature_id == "moderations":
            all_param_names = _extract_param_names(ModerationCreateParams)
            required_params = _extract_required_param_names(ModerationCreateParams)
            extraction_note = None
        else:
            target = _get_feature_callable(introspection_client, feature_id)
            all_param_names, required_params, extraction_note = _extract_signature_details(target)

        payload: dict[str, object] = {
            "core_params": core_params,
            "all_param_names": all_param_names,
            "required_params": required_params,
        }

        if extraction_note:
            payload["extraction_note"] = extraction_note

        reference[feature_id] = payload

    return reference


def build_feature_lesson_catalog() -> list[FeatureLesson]:
    """构建全功能教学卡片目录。"""

    full_reference = build_full_parameter_reference()
    lessons: list[FeatureLesson] = []

    for feature in FEATURE_REGISTRY:
        feature_id = feature["feature_id"]
        ref_payload = full_reference.get(feature_id, {})
        core_params_obj = ref_payload.get("core_params")
        all_params_obj = ref_payload.get("all_param_names")
        required_params_obj = ref_payload.get("required_params")

        core_params = core_params_obj if isinstance(core_params_obj, dict) else {}
        all_param_names = all_params_obj if isinstance(all_params_obj, list) else []
        required_params = required_params_obj if isinstance(required_params_obj, list) else []

        exercise_bundle = _FEATURE_EXERCISES.get(
            feature_id,
            {
                "concept": "理解题：解释该 API 的职责边界。",
                "hands_on": "动手题：写一个最小可运行示例并记录状态。",
            },
        )

        lesson: FeatureLesson = {
            "feature_id": feature_id,
            "display_name": feature["display_name"],
            "category": feature["category"],
            "stability": feature["stability"],
            "side_effect_level": feature["side_effect_level"],
            "api_surface": [feature["resource_path"]],
            "what_it_does": feature["what_it_does"],
            "when_to_use": feature["when_to_use"],
            "core_params": cast(dict[str, str], core_params),
            "all_param_names": [str(item) for item in all_param_names],
            "required_params": [str(item) for item in required_params],
            "sync_example": _FEATURE_SYNC_EXAMPLES.get(feature_id, f"client.{feature['resource_path']}(...)"),
            "async_example": _FEATURE_ASYNC_EXAMPLES.get(feature_id, f"async_client.{feature['resource_path']}(...)"),
            "exercise_concept": str(exercise_bundle.get("concept", "")),
            "exercise_hands_on": str(exercise_bundle.get("hands_on", "")),
        }
        lessons.append(lesson)

    return lessons


def _normalize_user_text(user_text: str) -> str:
    return user_text.strip() or DEFAULT_PROMPT


def _probe_feature(
    client: OpenAI,
    feature: FeatureRegistryItem,
    user_text: str,
    include_side_effect_calls: bool,
) -> FeatureProbeResult:
    feature_id = feature["feature_id"]

    if feature["default_probe_mode"] == "requires_opt_in" and not include_side_effect_calls:
        return _build_probe_skip_result(
            feature_id,
            "已按策略跳过：该功能属于潜在高副作用调用，需 include_side_effect_calls=True 才执行。",
        )

    call_mode: CallMode = "side_effect" if feature["default_probe_mode"] == "requires_opt_in" else "safe"

    try:
        summary = _run_feature_probe(client, feature_id, user_text)
        return _build_probe_ok_result(feature_id, summary, call_mode)
    except Exception as exc:
        return _build_probe_error_result(feature_id, call_mode, exc)


def probe_all_features(user_text: str, include_side_effect_calls: bool = False) -> list[FeatureProbeResult]:
    """执行全 feature probe，并返回结构化结果。"""

    prompt = _normalize_user_text(user_text)
    include_side_effect = include_side_effect_calls or _env_include_side_effect_calls()

    client = _get_client()
    results: list[FeatureProbeResult] = []
    for feature in FEATURE_REGISTRY:
        results.append(_probe_feature(client, feature, prompt, include_side_effect))
    return results


def _summarize_probe_results(results: list[FeatureProbeResult]) -> str:
    counter = Counter(result["status"] for result in results)
    status_summary = "；".join(f"{status}={counter[status]}" for status in sorted(counter))
    detail_summary = " | ".join(f"{result['feature_id']}={result['status']}" for result in results)
    return f"状态统计：{status_summary}。明细：{detail_summary}"


def run_full_demo(user_text: str, include_side_effect_calls: bool = False) -> DemoResult:
    """执行全功能教学流程。"""

    prompt = _normalize_user_text(user_text)
    include_side_effect = include_side_effect_calls or _env_include_side_effect_calls()

    trace: list[TraceEvent] = []

    feature_catalog = build_feature_lesson_catalog()
    trace.append(
        {
            "event": "feature_lesson_catalog_built",
            "total_features": len(feature_catalog),
            "features": feature_catalog,
        }
    )

    full_parameter_reference = build_full_parameter_reference()
    trace.append(
        {
            "event": "full_parameter_reference_built",
            "total_features": len(full_parameter_reference),
            "full_parameter_reference": full_parameter_reference,
        }
    )

    trace.append(
        {
            "event": "feature_probe_started",
            "include_side_effect_calls": include_side_effect,
            "probe_feature_count": len(FEATURE_REGISTRY),
        }
    )

    probe_results = probe_all_features(prompt, include_side_effect_calls=include_side_effect)
    for item in probe_results:
        trace.append({"event": "feature_probe_result", "result": item})

    status_summary = _summarize_probe_results(probe_results)
    final_answer = (
        "这是 OpenAI API 概念 + DeepSeek 兼容实调结果。"
        f"已覆盖 OpenAI client 全功能（{len(FEATURE_REGISTRY)} 项）。"
        f"side-effect 策略：include_side_effect_calls={include_side_effect}。"
        f"{status_summary}"
    )
    trace.append({"event": "model_final_answer", "content": final_answer})

    return {"final_answer": final_answer, "trace": trace}


def request_responses_demo(user_text: str) -> EndpointCallResult:
    """在线调用 responses 端点并返回结构化结果。"""

    client = _get_client()
    prompt = _normalize_user_text(user_text)

    try:
        summary = _probe_responses(client, prompt)
        return {
            "endpoint": "responses",
            "status": "ok",
            "http_status": None,
            "provider_error_type": None,
            "summary": summary,
        }
    except Exception as exc:
        return _build_error_result("responses", exc)


def request_embeddings_demo(text: str) -> EndpointCallResult:
    """在线调用 embeddings 端点并返回结构化结果。"""

    client = _get_client()
    prompt = _normalize_user_text(text)

    try:
        summary = _probe_embeddings(client, prompt)
        return {
            "endpoint": "embeddings",
            "status": "ok",
            "http_status": None,
            "provider_error_type": None,
            "summary": summary,
        }
    except Exception as exc:
        return _build_error_result("embeddings", exc)


def request_moderations_demo(text: str) -> EndpointCallResult:
    """在线调用 moderations 端点并返回结构化结果。"""

    client = _get_client()
    prompt = _normalize_user_text(text)

    try:
        summary = _probe_moderations(client, prompt)
        return {
            "endpoint": "moderations",
            "status": "ok",
            "http_status": None,
            "provider_error_type": None,
            "summary": summary,
        }
    except Exception as exc:
        return _build_error_result("moderations", exc)


def run_demo(user_text: str) -> DemoResult:
    """执行核心三端点教学流程（兼容历史行为）。"""

    prompt = _normalize_user_text(user_text)
    trace: list[TraceEvent] = []

    capability_overview = build_capability_overview()
    trace.append(
        {
            "event": "capability_overview_built",
            "total_capabilities": len(capability_overview),
            "capabilities": capability_overview,
        }
    )

    parameter_reference = build_parameter_reference()
    trace.append(
        {
            "event": "parameter_reference_built",
            "endpoints": list(parameter_reference.keys()),
            "parameter_reference": parameter_reference,
        }
    )

    responses_result = request_responses_demo(prompt)
    trace.append({"event": "online_call_responses", "result": responses_result})

    embeddings_result = request_embeddings_demo(prompt)
    trace.append({"event": "online_call_embeddings", "result": embeddings_result})

    moderations_result = request_moderations_demo(prompt)
    trace.append({"event": "online_call_moderations", "result": moderations_result})

    endpoint_results = [responses_result, embeddings_result, moderations_result]
    status_summary = "；".join(f"{item['endpoint']}={item['status']}" for item in endpoint_results)
    detail_summary = " | ".join(f"{item['endpoint']}: {item['summary']}" for item in endpoint_results)
    final_answer = f"这是 OpenAI API 概念 + DeepSeek 兼容实调结果：{status_summary}。{detail_summary}"

    trace.append({"event": "model_final_answer", "content": final_answer})
    return {"final_answer": final_answer, "trace": trace}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在线演示 OpenAI API 能力与参数（DeepSeek 兼容）")
    parser.add_argument("prompt", nargs="*", help="可选：覆盖默认用户输入")
    parser.add_argument(
        "--scope",
        choices=("core", "all"),
        default="core",
        help="core=仅三端点; all=全功能探测",
    )
    parser.add_argument(
        "--include-side-effects",
        choices=("y", "n"),
        default="n",
        help="是否执行高副作用 probe（默认 n）",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    user_text = " ".join(args.prompt).strip() or DEFAULT_PROMPT
    include_side_effect = args.include_side_effects == "y"

    if args.scope == "all":
        result = run_full_demo(user_text, include_side_effect_calls=include_side_effect)
    else:
        result = run_demo(user_text)

    print("=== TRACE ===")
    for index, event in enumerate(result["trace"], start=1):
        print(f"[{index}]")
        print(json.dumps(event, ensure_ascii=False, indent=2))

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])


if __name__ == "__main__":
    _main()
