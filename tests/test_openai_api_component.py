from __future__ import annotations

from typing import cast

import httpx
import openai
import pytest

import openai_api.component as openai_api_component

EXPECTED_FEATURE_IDS = {
    "responses",
    "chat",
    "completions",
    "embeddings",
    "moderations",
    "files",
    "uploads",
    "batches",
    "models",
    "fine_tuning",
    "vector_stores",
    "conversations",
    "realtime",
    "webhooks",
    "evals",
    "containers",
    "skills",
    "videos",
    "images",
    "audio",
    "beta",
    "client_with_options",
    "client_with_raw_response",
    "client_with_streaming_response",
}


def test_feature_registry_covers_all_expected_features() -> None:
    feature_ids = {item["feature_id"] for item in openai_api_component.FEATURE_REGISTRY}
    assert feature_ids == EXPECTED_FEATURE_IDS


def test_build_feature_lesson_catalog_schema_is_complete() -> None:
    lessons = openai_api_component.build_feature_lesson_catalog()

    assert isinstance(lessons, list)
    assert len(lessons) == len(openai_api_component.FEATURE_REGISTRY)

    required_keys = {
        "feature_id",
        "display_name",
        "category",
        "stability",
        "side_effect_level",
        "api_surface",
        "what_it_does",
        "when_to_use",
        "core_params",
        "all_param_names",
        "required_params",
        "sync_example",
        "async_example",
        "exercise_concept",
        "exercise_hands_on",
    }

    for lesson in lessons:
        assert required_keys.issubset(set(lesson.keys()))
        assert lesson["feature_id"] in EXPECTED_FEATURE_IDS
        assert isinstance(lesson["api_surface"], list)
        assert isinstance(lesson["core_params"], dict)
        assert isinstance(lesson["all_param_names"], list)
        assert isinstance(lesson["required_params"], list)
        assert isinstance(lesson["sync_example"], str) and lesson["sync_example"].strip()
        assert isinstance(lesson["async_example"], str) and lesson["async_example"].strip()
        assert isinstance(lesson["exercise_concept"], str) and lesson["exercise_concept"].strip()
        assert isinstance(lesson["exercise_hands_on"], str) and lesson["exercise_hands_on"].strip()


def test_build_full_parameter_reference_contains_all_features() -> None:
    reference = openai_api_component.build_full_parameter_reference()

    assert set(reference.keys()) == EXPECTED_FEATURE_IDS

    for feature_id, payload in reference.items():
        assert isinstance(payload.get("core_params"), dict)
        all_param_names = payload.get("all_param_names")
        assert isinstance(all_param_names, list)

        if all_param_names:
            assert all(isinstance(item, str) for item in all_param_names)
        else:
            note = payload.get("extraction_note")
            assert isinstance(note, str)
            assert note.strip()

        required_params = payload.get("required_params")
        assert isinstance(required_params, list)
        assert all(isinstance(item, str) for item in required_params)


def test_build_parameter_reference_keeps_backward_compat_sections() -> None:
    reference = openai_api_component.build_parameter_reference()

    for key in ("responses", "chat_completions", "embeddings", "moderations"):
        assert key in reference
        payload = reference[key]
        assert isinstance(payload.get("core_params"), dict)
        assert isinstance(payload.get("all_param_names"), list)
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


def test_probe_feature_skips_side_effect_calls_when_disabled() -> None:
    feature = next(item for item in openai_api_component.FEATURE_REGISTRY if item["feature_id"] == "uploads")

    result = openai_api_component._probe_feature(
        cast(openai_api_component.OpenAI, object()),
        feature,
        user_text="probe",
        include_side_effect_calls=False,
    )

    assert result["feature_id"] == "uploads"
    assert result["status"] == "skipped_by_policy"
    assert result["executed"] is False
    assert result["call_mode"] == "side_effect"


def test_probe_all_features_stays_structured_with_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyClient:
        pass

    def fake_probe(
        _client: openai_api_component.OpenAI,
        feature: openai_api_component.FeatureRegistryItem,
        _user_text: str,
        include_side_effect_calls: bool,
    ) -> openai_api_component.FeatureProbeResult:
        if feature["default_probe_mode"] == "requires_opt_in" and not include_side_effect_calls:
            return {
                "feature_id": feature["feature_id"],
                "status": "skipped_by_policy",
                "http_status": None,
                "provider_error_type": None,
                "summary": "skipped by test policy",
                "call_mode": "side_effect",
                "executed": False,
            }

        return {
            "feature_id": feature["feature_id"],
            "status": "ok",
            "http_status": None,
            "provider_error_type": None,
            "summary": "ok by monkeypatch",
            "call_mode": "safe" if feature["default_probe_mode"] == "safe_only" else "side_effect",
            "executed": True,
        }

    monkeypatch.setattr(openai_api_component, "_get_client", lambda: cast(openai_api_component.OpenAI, _DummyClient()))
    monkeypatch.setattr(openai_api_component, "_probe_feature", fake_probe)

    results = openai_api_component.probe_all_features("测试", include_side_effect_calls=False)

    assert len(results) == len(openai_api_component.FEATURE_REGISTRY)
    assert any(item["status"] == "skipped_by_policy" for item in results)
    assert all(item["feature_id"] in EXPECTED_FEATURE_IDS for item in results)


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


def test_run_full_demo_trace_order_is_stable_with_monkeypatched_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_lessons = [
        {
            "feature_id": "responses",
            "display_name": "Responses",
            "category": "generation",
            "stability": "stable",
            "side_effect_level": "none",
            "api_surface": ["responses.create"],
            "what_it_does": "mock",
            "when_to_use": "mock",
            "core_params": {"model": "mock"},
            "all_param_names": ["model", "input"],
            "required_params": ["model", "input"],
            "sync_example": "client.responses.create(...)",
            "async_example": "async_client.responses.create(...)",
            "exercise_concept": "c",
            "exercise_hands_on": "h",
        }
    ]

    fake_reference = {
        "responses": {
            "core_params": {"model": "mock"},
            "all_param_names": ["model", "input"],
            "required_params": ["model", "input"],
        }
    }

    fake_probe_results = [
        {
            "feature_id": "responses",
            "status": "ok",
            "http_status": None,
            "provider_error_type": None,
            "summary": "ok",
            "call_mode": "safe",
            "executed": True,
        },
        {
            "feature_id": "images",
            "status": "skipped_by_policy",
            "http_status": None,
            "provider_error_type": None,
            "summary": "skipped",
            "call_mode": "side_effect",
            "executed": False,
        },
    ]

    monkeypatch.setattr(openai_api_component, "build_feature_lesson_catalog", lambda: fake_lessons)
    monkeypatch.setattr(openai_api_component, "build_full_parameter_reference", lambda: fake_reference)
    monkeypatch.setattr(openai_api_component, "probe_all_features", lambda *_args, **_kwargs: fake_probe_results)

    result = openai_api_component.run_full_demo("任意输入", include_side_effect_calls=False)
    events = [item["event"] for item in result["trace"]]

    assert events[:3] == [
        "feature_lesson_catalog_built",
        "full_parameter_reference_built",
        "feature_probe_started",
    ]
    assert events.count("feature_probe_result") == len(fake_probe_results)
    assert events[-1] == "model_final_answer"
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
def test_probe_all_features_online_includes_models_and_containers() -> None:
    results = openai_api_component.probe_all_features("请总结 API 能力", include_side_effect_calls=False)
    result_map = {item["feature_id"]: item for item in results}

    for feature_id in ("models", "containers"):
        assert feature_id in result_map
        assert result_map[feature_id]["status"] in {
            "ok",
            "auth_error",
            "unsupported_by_provider",
            "rate_limited",
            "provider_server_error",
            "network_error",
            "skipped_by_policy",
            "unknown_error",
        }


@pytest.mark.online
def test_run_full_demo_online_returns_final_answer_and_trace() -> None:
    result = openai_api_component.run_full_demo("请总结 LLM API 功能", include_side_effect_calls=False)

    assert isinstance(result["final_answer"], str)
    assert result["final_answer"].strip()

    events = [item["event"] for item in result["trace"]]
    assert events[0] == "feature_lesson_catalog_built"
    assert events[1] == "full_parameter_reference_built"
    assert events[2] == "feature_probe_started"
    assert events[-1] == "model_final_answer"
