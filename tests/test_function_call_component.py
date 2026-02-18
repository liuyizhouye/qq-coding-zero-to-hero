import json
import math

from function_call.component import calc_portfolio_value, run_demo


def test_calc_portfolio_value_total_is_correct() -> None:
    positions = [
        {"symbol": "AAPL", "qty": 10, "price": 180.5},
        {"symbol": "TSLA", "qty": -3, "price": 210},
        {"symbol": "SPY", "qty": 2, "price": 500},
    ]
    result = calc_portfolio_value(positions)
    assert math.isclose(result["total_value"], 2175.0, rel_tol=0.0, abs_tol=1e-9)


def test_run_demo_trace_order_and_fields() -> None:
    result = run_demo("请计算组合总市值: AAPL 10@180.5, TSLA -3@210, SPY 2@500。")
    trace = result["trace"]
    assert isinstance(trace, list)
    events = [item["event"] for item in trace]
    assert events == [
        "model_function_call",
        "local_function_result",
        "function_call_output",
        "model_final_answer",
    ]

    first = trace[0]
    third = trace[2]
    assert first["call_id"] == third["call_id"]
    assert "arguments" in first
    assert "output" in third
    assert isinstance(result["final_answer"], str)


def test_run_demo_handles_error_without_breaking_flow() -> None:
    result = run_demo("请计算组合总市值: 无可用持仓")
    trace = result["trace"]

    assert trace[1]["event"] == "local_function_result"
    assert trace[1]["status"] == "error"

    output_payload = json.loads(str(trace[2]["output"]))
    assert "error" in output_payload
    assert "输入存在问题" in str(result["final_answer"])
    assert trace[3]["event"] == "model_final_answer"
