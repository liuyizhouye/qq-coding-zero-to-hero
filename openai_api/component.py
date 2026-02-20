"""openai_api 教学模块（OpenAI 概念 + DeepSeek OpenAI 兼容实调）。"""

from __future__ import annotations

import argparse
import json
import os
from typing import Required, TypedDict, get_origin, get_type_hints

from openai import APIConnectionError, APITimeoutError, OpenAI
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from openai.types.embedding_create_params import EmbeddingCreateParams
from openai.types.moderation_create_params import ModerationCreateParams
from openai.types.responses.response_create_params import ResponseCreateParamsBase

DEFAULT_PROMPT = "请用简短中文说明 LLM API 常见能力与参数作用。"
ONLINE_ENDPOINTS = ("responses", "embeddings", "moderations")
TraceEvent = dict[str, object]


class EndpointCallResult(TypedDict):
    """统一端点调用结果结构。"""

    endpoint: str
    status: str
    http_status: int | None
    provider_error_type: str | None
    summary: str


class DemoResult(TypedDict):
    """教学模块统一返回结构。"""

    final_answer: str
    trace: list[TraceEvent]


def _get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("missing DEEPSEEK_API_KEY; online mode requires a valid API key")
    return OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def _get_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _get_embedding_model() -> str:
    return os.getenv("DEEPSEEK_EMBEDDING_MODEL", _get_model())


def _get_moderation_model() -> str:
    return os.getenv("DEEPSEEK_MODERATION_MODEL", _get_model())


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


def build_capability_overview() -> list[dict[str, object]]:
    """构建 LLM API 能力全景矩阵。"""

    return [
        {
            "capability": "文本生成",
            "api_surface": ["responses", "chat.completions"],
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
            "api_surface": ["responses.input", "chat.completions.messages"],
            "what_it_does": "在同一次请求中处理文本、图片、文件等输入。",
            "when_to_use": "文档理解、图文问答、跨模态信息整合。",
        },
        {
            "capability": "检索向量",
            "api_surface": ["embeddings"],
            "what_it_does": "将文本映射到向量空间，用于相似度搜索与语义检索。",
            "when_to_use": "RAG 检索、语义去重、召回排序前置步骤。",
        },
        {
            "capability": "安全审核",
            "api_surface": ["moderations"],
            "what_it_does": "对输入内容做风险分类，识别潜在违规或高风险内容。",
            "when_to_use": "用户生成内容审核、合规前置检查。",
        },
        {
            "capability": "批处理与异步",
            "api_surface": ["batches", "responses.background"],
            "what_it_does": "将大量任务异步提交处理，提升吞吐与调度能力。",
            "when_to_use": "离线大批量生成、低实时性任务。",
        },
        {
            "capability": "文件与向量库",
            "api_surface": ["files", "vector_stores"],
            "what_it_does": "上传、管理与检索知识文件，构建可追溯知识基座。",
            "when_to_use": "企业知识库问答、文档检索增强。",
        },
        {
            "capability": "实时交互",
            "api_surface": ["realtime"],
            "what_it_does": "通过实时会话进行低延迟、持续式模型交互。",
            "when_to_use": "语音助手、实时协作、低延迟控制场景。",
        },
    ]


def build_parameter_reference() -> dict[str, dict[str, object]]:
    """构建参数参考：核心参数详解 + 全量参数索引。"""

    return {
        "responses": {
            "core_params": {
                "model": "模型名称，决定能力、价格、延迟与可用特性。",
                "input": "输入内容，可是字符串或结构化输入项。",
                "instructions": "系统级指令，用于约束回答风格与目标。",
                "tools": "可被模型调用的工具定义集合。",
                "tool_choice": "控制工具选择策略（自动/指定/禁用）。",
                "temperature": "采样随机度；越高越发散，越低越稳定。",
                "max_output_tokens": "限制输出 token 上限，控制成本与长度。",
                "reasoning": "推理模型相关配置（按模型支持情况生效）。",
                "text": "文本输出配置，可用于结构化输出。",
                "store": "是否保存响应供后续检索或追踪。",
            },
            "all_param_names": _extract_param_names(ResponseCreateParamsBase),
            "required_params": _extract_required_param_names(ResponseCreateParamsBase),
        },
        "chat_completions": {
            "core_params": {
                "model": "模型名称。",
                "messages": "对话消息数组，是 chat 接口核心输入。",
                "tools": "函数/工具定义，支持 function calling。",
                "tool_choice": "控制工具调用行为。",
                "temperature": "采样随机度。",
                "max_completion_tokens": "输出 token 上限（新字段）。",
                "response_format": "可配置 JSON schema 等结构化输出。",
                "parallel_tool_calls": "是否允许并行工具调用。",
                "seed": "尽量提高可复现性（非严格确定性）。",
            },
            "all_param_names": _extract_param_names(CompletionCreateParamsBase),
            "required_params": _extract_required_param_names(CompletionCreateParamsBase),
        },
        "embeddings": {
            "core_params": {
                "model": "向量模型名称。",
                "input": "待向量化的文本或 token 序列。",
                "dimensions": "返回向量维度（模型支持时可配置）。",
                "encoding_format": "向量编码格式（float/base64）。",
                "user": "终端用户标识，用于安全监控与审计。",
            },
            "all_param_names": _extract_param_names(EmbeddingCreateParams),
            "required_params": _extract_required_param_names(EmbeddingCreateParams),
        },
        "moderations": {
            "core_params": {
                "input": "待审核文本或多模态内容。",
                "model": "审核模型名称。",
            },
            "all_param_names": _extract_param_names(ModerationCreateParams),
            "required_params": _extract_required_param_names(ModerationCreateParams),
        },
    }


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


def _classify_api_error(exc: Exception) -> tuple[str, int | None, str | None]:
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


def _clip_text(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def request_responses_demo(user_text: str) -> EndpointCallResult:
    """在线调用 responses 端点并返回结构化结果。"""

    client = _get_client()
    try:
        response = client.responses.create(
            model=_get_model(),
            input=user_text,
            instructions="你是教学助手。请用不超过80字总结LLM API能力。",
        )
        response_text = _extract_response_text(response) or "已返回响应对象（未提取到文本内容）。"
        return {
            "endpoint": "responses",
            "status": "ok",
            "http_status": None,
            "provider_error_type": None,
            "summary": f"responses 调用成功：{_clip_text(response_text)}",
        }
    except Exception as exc:
        return _build_error_result("responses", exc)


def request_embeddings_demo(text: str) -> EndpointCallResult:
    """在线调用 embeddings 端点并返回结构化结果。"""

    client = _get_client()
    try:
        response = client.embeddings.create(
            model=_get_embedding_model(),
            input=[text],
            encoding_format="float",
        )
        data = getattr(response, "data", None)
        vector_count = len(data) if isinstance(data, list) else 0

        first_dim: int | None = None
        if isinstance(data, list) and data:
            first_embedding = getattr(data[0], "embedding", None)
            if isinstance(first_embedding, list):
                first_dim = len(first_embedding)

        dim_text = str(first_dim) if isinstance(first_dim, int) else "未知"
        summary = f"embeddings 调用成功：向量条数={vector_count}，首条维度={dim_text}。"
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
    try:
        response = client.moderations.create(
            model=_get_moderation_model(),
            input=text,
        )
        results = getattr(response, "results", None)
        flagged_text = "未知"
        if isinstance(results, list) and results:
            flagged_obj = getattr(results[0], "flagged", None)
            if isinstance(flagged_obj, bool):
                flagged_text = str(flagged_obj)

        return {
            "endpoint": "moderations",
            "status": "ok",
            "http_status": None,
            "provider_error_type": None,
            "summary": f"moderations 调用成功：flagged={flagged_text}。",
        }
    except Exception as exc:
        return _build_error_result("moderations", exc)


def run_demo(user_text: str) -> DemoResult:
    """执行完整教学流程并输出结构化 trace。"""

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

    responses_result = request_responses_demo(user_text)
    trace.append({"event": "online_call_responses", "result": responses_result})

    embeddings_result = request_embeddings_demo(user_text)
    trace.append({"event": "online_call_embeddings", "result": embeddings_result})

    moderations_result = request_moderations_demo(user_text)
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
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    user_text = " ".join(args.prompt).strip() or DEFAULT_PROMPT
    result = run_demo(user_text)

    print("=== TRACE ===")
    for index, event in enumerate(result["trace"], start=1):
        print(f"[{index}]")
        print(json.dumps(event, ensure_ascii=False, indent=2))

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])


if __name__ == "__main__":
    _main()
