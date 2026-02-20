import argparse
import json
from typing import Any

from openai_api.component import (
    DEFAULT_PROMPT,
    build_capability_overview,
    build_parameter_reference,
    request_embeddings_demo,
    request_moderations_demo,
    request_responses_demo,
    run_demo,
)


def build_debug_state(user_text: str = DEFAULT_PROMPT) -> dict[str, object]:
    step1_user_text = user_text.strip() or DEFAULT_PROMPT
    step2_capability_overview = build_capability_overview()
    step3_parameter_reference = build_parameter_reference()
    step4_responses_result = request_responses_demo(step1_user_text)
    step5_embeddings_result = request_embeddings_demo(step1_user_text)
    step6_moderations_result = request_moderations_demo(step1_user_text)
    step7_demo_result = run_demo(step1_user_text)

    debug_state: dict[str, object] = {
        "step1_user_text": step1_user_text,
        "step2_capability_overview": step2_capability_overview,
        "step3_parameter_reference": step3_parameter_reference,
        "step4_responses_result": step4_responses_result,
        "step5_embeddings_result": step5_embeddings_result,
        "step6_moderations_result": step6_moderations_result,
        "demo_result": step7_demo_result,
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
    parser = argparse.ArgumentParser(description="openai_api debug main: F5 逐行观察 API 能力与参数")
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
