import argparse
import json
import re
from typing import TypedDict
from uuid import uuid4

DEFAULT_PROMPT = "请计算组合总市值: AAPL 10@180.5, TSLA -3@210, SPY 2@500。请先调用工具再回答。"
POSITION_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9._-]*)\s*(-?\d+(?:\.\d+)?)@(-?\d+(?:\.\d+)?)")
TraceEvent = dict[str, object]


class DemoResult(TypedDict):
    final_answer: str
    trace: list[TraceEvent]


def calc_portfolio_value(positions: list[dict[str, float | str]]) -> dict[str, float]:
    if not positions:
        raise ValueError("positions must contain at least one item")

    total = 0.0
    for index, row in enumerate(positions, start=1):
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            raise ValueError(f"row {index} missing symbol")
        try:
            qty = float(row["qty"])
            price = float(row["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"row {index} has invalid qty/price") from exc
        total += qty * price
    return {"total_value": total}


def _extract_positions(user_text: str) -> list[dict[str, float | str]]:
    matches = POSITION_PATTERN.findall(user_text)
    return [
        {"symbol": symbol.upper(), "qty": float(qty_text), "price": float(price_text)}
        for symbol, qty_text, price_text in matches
    ]


def mock_model_function_call(user_text: str) -> dict[str, object]:
    arguments = {"positions": _extract_positions(user_text)}
    return {
        "type": "function_call",
        "name": "calc_portfolio_value",
        "call_id": f"call_{uuid4().hex[:10]}",
        "arguments": json.dumps(arguments, ensure_ascii=False),
    }


def run_demo(user_text: str) -> DemoResult:
    trace: list[TraceEvent] = []

    call = mock_model_function_call(user_text)
    call_id = str(call["call_id"])
    arguments_text = str(call["arguments"])
    trace.append(
        {
            "event": "model_function_call",
            "type": "function_call",
            "name": call["name"],
            "call_id": call_id,
            "arguments": arguments_text,
        }
    )

    output_payload: dict[str, object]
    try:
        parsed = json.loads(arguments_text)
        if not isinstance(parsed, dict):
            raise ValueError("arguments must be a JSON object")
        positions = parsed.get("positions")
        if not isinstance(positions, list):
            raise ValueError("arguments.positions must be a list")
        result = calc_portfolio_value(positions)
        trace.append({"event": "local_function_result", "status": "ok", "result": result})
        output_payload = {"result": result}
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc)
        trace.append({"event": "local_function_result", "status": "error", "error": error_text})
        output_payload = {"error": error_text}

    output_text = json.dumps(output_payload, ensure_ascii=False)
    trace.append(
        {
            "event": "function_call_output",
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_text,
        }
    )

    if "error" in output_payload:
        final_answer = f"工具调用已完成，但输入存在问题：{output_payload['error']}"
    else:
        result_block = output_payload["result"]
        assert isinstance(result_block, dict)
        total_value = float(result_block["total_value"])
        final_answer = f"组合总市值为 {total_value:.2f}。"

    trace.append({"event": "model_final_answer", "content": final_answer})
    return {"final_answer": final_answer, "trace": trace}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线演示 function_call -> 本地函数执行 -> function_call_output")
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
