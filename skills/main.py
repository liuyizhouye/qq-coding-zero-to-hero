import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Any

from skills.component import (
    DEFAULT_PROMPT,
    build_execution_plan,
    load_skill_catalog,
    match_skill_scores,
    run_demo,
    select_skill,
)


def build_debug_state(user_text: str = DEFAULT_PROMPT) -> dict[str, object]:
    step1_user_text = user_text.strip() or DEFAULT_PROMPT

    step2_catalog = load_skill_catalog()
    step3_scores = match_skill_scores(step1_user_text, step2_catalog)
    step4_selection = select_skill(step1_user_text, step2_catalog)

    step5_plan: list[str] | None
    if step4_selection is not None:
        step5_plan = build_execution_plan(step4_selection)
    else:
        step5_plan = None

    step6_demo_result = run_demo(step1_user_text)

    debug_state: dict[str, object] = {
        "step1_user_text": step1_user_text,
        "step2_catalog": step2_catalog,
        "step3_scores": step3_scores,
        "step4_selection": step4_selection,
        "step5_plan": step5_plan,
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
    parser = argparse.ArgumentParser(description="skills debug main：用于 F5 逐行观察匹配与规划")
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
