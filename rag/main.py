import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Any

from rag.component import DEFAULT_QUERY, chunk_report, load_report_text, retrieve_top_k, run_demo, synthesize_answer


def build_debug_state(query: str | None = None, top_k: int = 3) -> dict[str, object]:
    step1_query = query.strip() if isinstance(query, str) and query.strip() else DEFAULT_QUERY
    step1_top_k = top_k

    step2_report_text = load_report_text()
    step3_chunks = chunk_report(step2_report_text)
    step4_hits = retrieve_top_k(step3_chunks, step1_query, top_k=step1_top_k)
    step5_answer = synthesize_answer(step1_query, step4_hits)
    step6_demo_result = run_demo(query=step1_query, top_k=step1_top_k)

    debug_state: dict[str, object] = {
        "step1_query": step1_query,
        "step1_top_k": step1_top_k,
        "step2_report_text": step2_report_text,
        "step3_chunks": step3_chunks,
        "step4_hits": step4_hits,
        "step5_answer": step5_answer,
        "demo_result": step6_demo_result,
    }
    return debug_state


def run_debug(query: str | None = None, top_k: int = 3) -> dict[str, object]:
    return build_debug_state(query=query, top_k=top_k)


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
    parser = argparse.ArgumentParser(description="rag debug main：用于 F5 逐行观察检索与引用式回答")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="调试 query")
    parser.add_argument("--top-k", type=int, default=3, help="检索 top-k")
    parser.add_argument("--print-trace", choices=("y", "n"), default="y", help="是否打印 trace")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    debug_state = build_debug_state(query=str(args.query), top_k=int(args.top_k))

    if args.print_trace == "y":
        demo_obj = debug_state.get("demo_result")
        if isinstance(demo_obj, dict):
            _print_demo_result(demo_obj)


if __name__ == "__main__":
    _main()
