import argparse
import json
from typing import Any

from function_call.component import (
    DEFAULT_PROMPT,
    calc_portfolio_value,
    request_model_function_call,
    run_demo,
)


def build_debug_state(user_text: str = DEFAULT_PROMPT) -> dict[str, object]:
    step1_user_text = user_text.strip() or DEFAULT_PROMPT

    step2_call = request_model_function_call(step1_user_text)

    step3_call_id = str(step2_call["call_id"])
    step3_arguments_text = str(step2_call["arguments"])

    step3_arguments_obj: dict[str, object] | None
    try:
        parsed_args = json.loads(step3_arguments_text)
        step3_arguments_obj = parsed_args if isinstance(parsed_args, dict) else None
    except json.JSONDecodeError:
        step3_arguments_obj = None

    step4_positions_obj = step3_arguments_obj.get("positions") if isinstance(step3_arguments_obj, dict) else None
    step4_positions = step4_positions_obj if isinstance(step4_positions_obj, list) else []

    step5_local_result: dict[str, object]
    try:
        local_result = calc_portfolio_value(step4_positions)
        step5_local_result = {"status": "ok", "result": local_result}
    except Exception as exc:
        step5_local_result = {"status": "error", "error": str(exc)}

    step6_demo_result = run_demo(step1_user_text)

    debug_state: dict[str, object] = {
        "step1_user_text": step1_user_text,
        "step2_call": step2_call,
        "step3_call_id": step3_call_id,
        "step3_arguments_text": step3_arguments_text,
        "step3_arguments_obj": step3_arguments_obj,
        "step4_positions": step4_positions,
        "step5_local_result": step5_local_result,
        "demo_result": step6_demo_result,
    }
    return debug_state


def run_debug(user_text: str = DEFAULT_PROMPT) -> dict[str, object]:
    return build_debug_state(user_text=user_text)


def _print_demo_result(demo_result: dict[str, Any]) -> None:
    trace_obj = demo_result.get("trace")
    if isinstance(trace_obj, list):
        print("=== TRACE ===")
        for index, event in enumerate(trace_obj, start=1):
            print(f"[{index}]")
            print(json.dumps(event, ensure_ascii=False, indent=2))

    final_answer = demo_result.get("final_answer")
    if isinstance(final_answer, str):
        print("\n=== FINAL ANSWER ===")
        print(final_answer)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="function_call debug main: F5 逐行观察在线 function call 流程")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="调试输入文本")
    parser.add_argument("--print-trace", choices=("y", "n"), default="y", help="是否打印 trace")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    debug_state = build_debug_state(user_text=str(args.prompt))

    if args.print_trace == "y":
        demo_obj = debug_state.get("demo_result")
        if isinstance(demo_obj, dict):
            _print_demo_result(demo_obj)


if __name__ == "__main__":
    _main()
