import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Any

from mcp.component import DEFAULT_PROMPT, build_mcp_tool_config, mock_mcp_approval_request, run_demo


def build_debug_state(user_text: str = DEFAULT_PROMPT, approve: bool = True) -> dict[str, object]:
    step1_user_text = user_text.strip() or DEFAULT_PROMPT
    step1_approve = approve

    step2_tool_config = build_mcp_tool_config()
    step3_approval_request = mock_mcp_approval_request()
    step4_approval_request_id = str(step3_approval_request["approval_request_id"])

    step5_demo_result = run_demo(user_text=step1_user_text, approve=step1_approve)

    debug_state: dict[str, object] = {
        "step1_user_text": step1_user_text,
        "step1_approve": step1_approve,
        "step2_tool_config": step2_tool_config,
        "step3_approval_request": step3_approval_request,
        "step4_approval_request_id": step4_approval_request_id,
        "demo_result": step5_demo_result,
    }
    return debug_state


def run_debug(user_text: str = DEFAULT_PROMPT, approve: bool = True) -> dict[str, object]:
    return build_debug_state(user_text=user_text, approve=approve)


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
    parser = argparse.ArgumentParser(description="mcp debug main：用于 F5 逐行观察审批流程")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="调试输入文本")
    parser.add_argument("--approve", choices=("y", "n"), default="y", help="审批决策")
    parser.add_argument("--print-trace", choices=("y", "n"), default="y", help="是否打印 trace")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    approve = args.approve == "y"
    debug_state = build_debug_state(user_text=str(args.prompt), approve=approve)

    if args.print_trace == "y":
        demo_obj = debug_state.get("demo_result")
        if isinstance(demo_obj, dict):
            _print_demo_result(demo_obj)


if __name__ == "__main__":
    _main()
