import argparse
import json
import os
from typing import TypedDict
from uuid import uuid4

from openai import OpenAI

DEFAULT_PROMPT = "请计算组合总市值: AAPL 10@180.5, TSLA -3@210, SPY 2@500。请先调用工具再回答。"
TraceEvent = dict[str, object]


class DemoResult(TypedDict):
    final_answer: str
    trace: list[TraceEvent]


def _get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("missing DEEPSEEK_API_KEY; online mode requires a valid API key")
    return OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def _get_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _request_json(system_prompt: str, user_prompt: str) -> dict[str, object]:
    client = _get_client()
    response = client.chat.completions.create(
        model=_get_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )

    message = response.choices[0].message.content
    if not isinstance(message, str):
        raise ValueError("model did not return text content")

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model output is not strict JSON: {message}") from exc

    if not isinstance(payload, dict):
        raise ValueError("model JSON output must be an object")
    return payload


def calc_portfolio_value(positions: list[dict[str, float | str]]) -> dict[str, float]:
    """根据持仓列表计算组合总市值。"""
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


def request_model_function_call(user_text: str) -> dict[str, object]:
    system_prompt = (
        "You convert a Chinese finance request into a function call argument object. "
        "Return strict JSON only, no markdown."
    )
    user_prompt = (
        "目标函数: calc_portfolio_value\n"
        "请从用户输入中抽取 positions。\n"
        "JSON schema:\n"
        "{\n"
        '  "name": "calc_portfolio_value",\n'
        '  "arguments": {\n'
        '    "positions": [\n'
        '      {"symbol": "AAPL", "qty": 10, "price": 180.5}\n'
        "    ]\n"
        "  }\n"
        "}\n"
        "规则: symbol 用字符串；qty/price 用数字；若无法解析则返回空数组。\n"
        f"用户输入: {user_text}"
    )

    payload = _request_json(system_prompt, user_prompt)
    name = str(payload.get("name", ""))
    if name != "calc_portfolio_value":
        raise ValueError("model returned invalid function name")

    arguments_obj = payload.get("arguments")
    if not isinstance(arguments_obj, dict):
        raise ValueError("model returned invalid arguments object")

    positions_obj = arguments_obj.get("positions")
    if not isinstance(positions_obj, list):
        raise ValueError("model returned invalid arguments.positions")

    normalized_positions: list[dict[str, float | str]] = []
    for index, item in enumerate(positions_obj, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"model returned non-object position at index {index}")
        try:
            symbol = str(item["symbol"]).upper().strip()
            qty = float(item["qty"])
            price = float(item["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"model returned invalid position at index {index}") from exc
        normalized_positions.append({"symbol": symbol, "qty": qty, "price": price})

    call_id = f"call_{uuid4().hex[:10]}"
    return {
        "type": "function_call",
        "name": "calc_portfolio_value",
        "call_id": call_id,
        "arguments": json.dumps({"positions": normalized_positions}, ensure_ascii=False),
    }


def request_model_final_answer(user_text: str, tool_output: dict[str, object], call_id: str) -> str:
    system_prompt = "You summarize tool execution results for end users. Return strict JSON only."
    user_prompt = (
        "请根据以下信息给出中文最终回答。\n"
        "JSON schema:\n"
        "{\n"
        '  "final_answer": "string"\n'
        "}\n"
        f"用户输入: {user_text}\n"
        f"call_id: {call_id}\n"
        f"工具输出(JSON): {json.dumps(tool_output, ensure_ascii=False)}"
    )

    payload = _request_json(system_prompt, user_prompt)
    final_answer = payload.get("final_answer")
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError("model returned invalid final_answer")
    return final_answer.strip()


def run_demo(user_text: str) -> DemoResult:
    trace: list[TraceEvent] = []

    call = request_model_function_call(user_text)
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

    final_answer = request_model_final_answer(user_text=user_text, tool_output=output_payload, call_id=call_id)
    trace.append({"event": "model_final_answer", "content": final_answer})
    return {"final_answer": final_answer, "trace": trace}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在线演示 function_call -> 本地函数执行 -> function_call_output")
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
