import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Any

from react_architecture.component import build_architecture_matrix, run_demo


def build_debug_state() -> dict[str, object]:
    step1_matrix = build_architecture_matrix()

    step2_mpa_row = next((item for item in step1_matrix if item.get("architecture") == "MPA"), None)
    step3_spa_row = next((item for item in step1_matrix if item.get("architecture") == "SPA_CSR"), None)
    step4_ssr_row = next((item for item in step1_matrix if item.get("architecture") == "SSR"), None)
    step5_ssg_row = next((item for item in step1_matrix if item.get("architecture") == "SSG"), None)

    step6_demo_result = run_demo()

    debug_state: dict[str, object] = {
        "step1_matrix": step1_matrix,
        "step2_mpa_row": step2_mpa_row,
        "step3_spa_row": step3_spa_row,
        "step4_ssr_row": step4_ssr_row,
        "step5_ssg_row": step5_ssg_row,
        "demo_result": step6_demo_result,
    }
    return debug_state


def run_debug() -> dict[str, object]:
    return build_debug_state()


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
    parser = argparse.ArgumentParser(description="react_architecture debug main：用于 F5 逐行观察架构矩阵")
    parser.add_argument("--print-trace", choices=("y", "n"), default="y", help="是否打印 trace")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    debug_state = build_debug_state()

    if args.print_trace == "y":
        demo_obj = debug_state.get("demo_result")
        if isinstance(demo_obj, dict):
            _print_demo_result(demo_obj)


if __name__ == "__main__":
    _main()
