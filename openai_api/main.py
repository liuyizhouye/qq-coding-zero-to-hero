import argparse
import json
import os
from typing import Any

from openai_api.component import (
    DEFAULT_PROMPT,
    build_capability_overview,
    build_feature_lesson_catalog,
    build_full_parameter_reference,
    build_parameter_reference,
    probe_all_features,
    request_embeddings_demo,
    request_moderations_demo,
    request_responses_demo,
    run_demo,
    run_full_demo,
)


def _default_include_side_effect_calls() -> bool:
    raw = os.getenv("OPENAI_API_INCLUDE_SIDE_EFFECT_CALLS")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_debug_state(
    user_text: str = DEFAULT_PROMPT,
    scope: str = "all",
    include_side_effect_calls: bool = False,
) -> dict[str, object]:
    step1_user_text = user_text.strip() or DEFAULT_PROMPT
    step1_scope = scope if scope in {"core", "all"} else "all"
    step1_include_side_effect_calls = include_side_effect_calls

    if step1_scope == "core":
        step2_capability_overview = build_capability_overview()
        step3_parameter_reference = build_parameter_reference()
        step4_responses_result = request_responses_demo(step1_user_text)
        step5_embeddings_result = request_embeddings_demo(step1_user_text)
        step6_moderations_result = request_moderations_demo(step1_user_text)
        step7_demo_result = run_demo(step1_user_text)

        debug_state_core: dict[str, object] = {
            "step1_user_text": step1_user_text,
            "step1_scope": step1_scope,
            "step1_include_side_effect_calls": step1_include_side_effect_calls,
            "step2_capability_overview": step2_capability_overview,
            "step3_parameter_reference": step3_parameter_reference,
            "step4_responses_result": step4_responses_result,
            "step5_embeddings_result": step5_embeddings_result,
            "step6_moderations_result": step6_moderations_result,
            "demo_result": step7_demo_result,
        }
        return debug_state_core

    step2_feature_catalog = build_feature_lesson_catalog()
    step3_full_parameter_reference = build_full_parameter_reference()
    step4_feature_probe_results = probe_all_features(
        step1_user_text,
        include_side_effect_calls=step1_include_side_effect_calls,
    )
    step5_demo_result = run_full_demo(
        step1_user_text,
        include_side_effect_calls=step1_include_side_effect_calls,
    )

    debug_state_all: dict[str, object] = {
        "step1_user_text": step1_user_text,
        "step1_scope": step1_scope,
        "step1_include_side_effect_calls": step1_include_side_effect_calls,
        "step2_feature_catalog": step2_feature_catalog,
        "step3_full_parameter_reference": step3_full_parameter_reference,
        "step4_feature_probe_results": step4_feature_probe_results,
        "demo_result": step5_demo_result,
    }
    return debug_state_all


def run_debug(
    user_text: str = DEFAULT_PROMPT,
    scope: str = "all",
    include_side_effect_calls: bool = False,
) -> dict[str, object]:
    return build_debug_state(
        user_text=user_text,
        scope=scope,
        include_side_effect_calls=include_side_effect_calls,
    )


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
    parser.add_argument("--scope", choices=("core", "all"), default="all", help="core=三端点; all=全功能")
    parser.add_argument(
        "--include-side-effects",
        choices=("y", "n"),
        default="y" if _default_include_side_effect_calls() else "n",
        help="是否执行高副作用 probe（默认 n）",
    )
    parser.add_argument("--print-trace", choices=("y", "n"), default="y", help="是否打印 trace")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    include_side_effect_calls = args.include_side_effects == "y"
    debug_state = build_debug_state(
        user_text=str(args.prompt),
        scope=str(args.scope),
        include_side_effect_calls=include_side_effect_calls,
    )

    if args.print_trace == "y":
        demo_obj = debug_state.get("demo_result")
        if isinstance(demo_obj, dict):
            _print_demo_result(demo_obj)


if __name__ == "__main__":
    _main()
