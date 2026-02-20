import openai
import pytest
import httpx

import openai_api.component as openai_api_component


def test_build_capability_overview_contains_expected_categories() -> None:
    overview = openai_api_component.build_capability_overview()

    assert isinstance(overview, list)
    assert overview

    categories = {str(item.get("capability")) for item in overview}
    assert "文本生成" in categories
    assert "工具调用" in categories
    assert "检索向量" in categories
    assert "安全审核" in categories


def test_build_parameter_reference_contains_expected_sections() -> None:
    reference = openai_api_component.build_parameter_reference()

    for key in ("responses", "chat_completions", "embeddings", "moderations"):
        assert key in reference
        payload = reference[key]
        assert isinstance(payload.get("core_params"), dict)
        assert isinstance(payload.get("all_param_names"), list)
        assert payload["all_param_names"]
        assert isinstance(payload.get("required_params"), list)


class _FakeStatusError(Exception):
    def __init__(self, status_code: int, error_type: str = "invalid_request_error") -> None:
        super().__init__(f"status={status_code}")
        self.status_code = status_code
        self.body = {"error": {"type": error_type}}


def test_classify_api_error_for_typical_status_codes() -> None:
    assert openai_api_component._classify_api_error(_FakeStatusError(401))[0] == "auth_error"
    assert openai_api_component._classify_api_error(_FakeStatusError(404))[0] == "unsupported_by_provider"
    assert openai_api_component._classify_api_error(_FakeStatusError(429))[0] == "rate_limited"
    assert openai_api_component._classify_api_error(_FakeStatusError(500))[0] == "provider_server_error"
    assert openai_api_component._classify_api_error(_FakeStatusError(418))[0] == "unknown_error"


def test_classify_api_error_for_network_exception() -> None:
    request = httpx.Request("GET", "https://example.com")
    exc = openai.APIConnectionError(message="network down", request=request)

    status, http_status, provider_type = openai_api_component._classify_api_error(exc)
    assert status == "network_error"
    assert http_status is None
    assert isinstance(provider_type, str)


def test_run_demo_trace_order_is_stable_with_monkeypatched_online_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_responses(_user_text: str) -> openai_api_component.EndpointCallResult:
        return {
            "endpoint": "responses",
            "status": "ok",
            "http_status": None,
            "provider_error_type": None,
            "summary": "responses ok",
        }

    def fake_embeddings(_text: str) -> openai_api_component.EndpointCallResult:
        return {
            "endpoint": "embeddings",
            "status": "unsupported_by_provider",
            "http_status": 404,
            "provider_error_type": "not_found",
            "summary": "embeddings unsupported",
        }

    def fake_moderations(_text: str) -> openai_api_component.EndpointCallResult:
        return {
            "endpoint": "moderations",
            "status": "ok",
            "http_status": None,
            "provider_error_type": None,
            "summary": "moderations ok",
        }

    monkeypatch.setattr(openai_api_component, "request_responses_demo", fake_responses)
    monkeypatch.setattr(openai_api_component, "request_embeddings_demo", fake_embeddings)
    monkeypatch.setattr(openai_api_component, "request_moderations_demo", fake_moderations)

    result = openai_api_component.run_demo("任意输入")
    events = [item["event"] for item in result["trace"]]

    assert events == [
        "capability_overview_built",
        "parameter_reference_built",
        "online_call_responses",
        "online_call_embeddings",
        "online_call_moderations",
        "model_final_answer",
    ]
    assert "OpenAI API 概念 + DeepSeek 兼容实调结果" in result["final_answer"]


@pytest.mark.online
def test_request_responses_demo_online_schema() -> None:
    result = openai_api_component.request_responses_demo("请用一句话介绍 LLM API。")
    assert result["endpoint"] == "responses"
    assert result["status"] in {
        "ok",
        "auth_error",
        "unsupported_by_provider",
        "rate_limited",
        "provider_server_error",
        "network_error",
        "unknown_error",
    }
    assert isinstance(result["summary"], str)


@pytest.mark.online
def test_request_embeddings_demo_online_schema() -> None:
    result = openai_api_component.request_embeddings_demo("embedding 测试文本")
    assert result["endpoint"] == "embeddings"
    assert result["status"] in {
        "ok",
        "auth_error",
        "unsupported_by_provider",
        "rate_limited",
        "provider_server_error",
        "network_error",
        "unknown_error",
    }
    assert isinstance(result["summary"], str)


@pytest.mark.online
def test_request_moderations_demo_online_schema() -> None:
    result = openai_api_component.request_moderations_demo("moderation 测试文本")
    assert result["endpoint"] == "moderations"
    assert result["status"] in {
        "ok",
        "auth_error",
        "unsupported_by_provider",
        "rate_limited",
        "provider_server_error",
        "network_error",
        "unknown_error",
    }
    assert isinstance(result["summary"], str)


@pytest.mark.online
def test_run_demo_online_returns_trace_without_breaking() -> None:
    result = openai_api_component.run_demo("请总结 LLM API 能力与参数。")
    events = [item["event"] for item in result["trace"]]

    assert events == [
        "capability_overview_built",
        "parameter_reference_built",
        "online_call_responses",
        "online_call_embeddings",
        "online_call_moderations",
        "model_final_answer",
    ]
    assert isinstance(result["final_answer"], str)
    assert result["final_answer"].strip()
