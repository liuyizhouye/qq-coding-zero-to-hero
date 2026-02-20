import argparse
import json
from typing import Any

from decorator.component import CountCalls, run_demo, timed, with_tag


def build_debug_state() -> dict[str, object]:
    trace_local: list[dict[str, object]] = []

    @CountCalls
    @with_tag("billing")
    @timed
    def sample_total(subtotal: float, tax_rate: float, *, _trace: list[dict[str, object]] | None = None) -> float:
        total = subtotal * (1.0 + tax_rate)
        if _trace is not None:
            _trace.append(
                {
                    "event": "decorated_function_body",
                    "subtotal": subtotal,
                    "tax_rate": tax_rate,
                    "total": round(total, 2),
                }
            )
        return round(total, 2)

    step1_trace_local = trace_local
    step2_first_call_result = sample_total(100.0, 0.13, _trace=step1_trace_local)
    step3_second_call_result = sample_total(80.0, 0.13, _trace=step1_trace_local)
    step4_call_count = int(getattr(sample_total, "call_count", 0))

    step5_demo_result = run_demo()

    debug_state: dict[str, object] = {
        "step1_trace_local": step1_trace_local,
        "step2_first_call_result": step2_first_call_result,
        "step3_second_call_result": step3_second_call_result,
        "step4_call_count": step4_call_count,
        "demo_result": step5_demo_result,
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
    parser = argparse.ArgumentParser(description="decorator debug main：用于 F5 逐行观察装饰器链路")
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
