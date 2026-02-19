import json
import math

import pytest

import function_call.component as function_call_component


def test_calc_portfolio_value_total_is_correct() -> None:
    positions = [
        {"symbol": "AAPL", "qty": 10, "price": 180.5},
        {"symbol": "TSLA", "qty": -3, "price": 210},
        {"symbol": "SPY", "qty": 2, "price": 500},
    ]
    result = function_call_component.calc_portfolio_value(positions)
    assert math.isclose(result["total_value"], 2175.0, rel_tol=0.0, abs_tol=1e-9)


def test_run_demo_handles_error_event_without_breaking_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request_model_function_call(_user_text: str) -> dict[str, object]:
        return {
            "type": "function_call",
            "name": "calc_portfolio_value",
            "call_id": "call_test_error",
            "arguments": json.dumps({"positions": []}, ensure_ascii=False),
        }

    def fake_request_model_final_answer(*, user_text: str, tool_output: dict[str, object], call_id: str) -> str:
        assert "error" in tool_output
        return "已捕获工具执行错误。"

    monkeypatch.setattr(function_call_component, "request_model_function_call", fake_request_model_function_call)
    monkeypatch.setattr(function_call_component, "request_model_final_answer", fake_request_model_final_answer)

    result = function_call_component.run_demo("任意输入")
    trace = result["trace"]
    events = [item["event"] for item in trace]

    assert events == [
        "model_function_call",
        "local_function_result",
        "function_call_output",
        "model_final_answer",
    ]
    assert trace[1]["status"] == "error"
    output_payload = json.loads(str(trace[2]["output"]))
    assert "error" in output_payload
    assert isinstance(result["final_answer"], str)
    assert result["final_answer"]


@pytest.mark.online
def test_request_model_function_call_online_schema() -> None:
    call = function_call_component.request_model_function_call(
        "请计算组合总市值: AAPL 10@180.5, TSLA -3@210, SPY 2@500。"
    )

    assert call["type"] == "function_call"
    assert call["name"] == "calc_portfolio_value"
    assert str(call["call_id"]).startswith("call_")

    arguments_obj = json.loads(str(call["arguments"]))
    assert isinstance(arguments_obj, dict)
    positions = arguments_obj.get("positions")
    assert isinstance(positions, list)


@pytest.mark.online
def test_run_demo_online_trace_order_and_fields() -> None:
    result = function_call_component.run_demo("请计算组合总市值: AAPL 10@180.5, TSLA -3@210, SPY 2@500。")
    trace = result["trace"]
    events = [item["event"] for item in trace]

    assert events == [
        "model_function_call",
        "local_function_result",
        "function_call_output",
        "model_final_answer",
    ]

    first = trace[0]
    second = trace[1]
    third = trace[2]

    assert first["call_id"] == third["call_id"]
    assert "arguments" in first
    assert second["status"] in {"ok", "error"}
    assert "output" in third
    assert isinstance(result["final_answer"], str)
    assert result["final_answer"].strip()
