import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Any

from fastapi.testclient import TestClient

from web_data_flow.component import calc_order_summary, create_app, run_demo

DEFAULT_PAYLOAD = {
    "items": [
        {"sku": "Keyboard", "qty": 2, "unit_price": 49.5},
        {"sku": "Mouse", "qty": 1, "unit_price": 25.0},
    ],
    "tax_rate": 0.1,
}


def build_debug_state() -> dict[str, object]:
    step1_payload = DEFAULT_PAYLOAD
    step2_calc_preview = calc_order_summary(step1_payload["items"], tax_rate=float(step1_payload["tax_rate"]))

    step3_app = create_app()
    with TestClient(step3_app) as client:
        response = client.post("/api/orders/summary", json=step1_payload)
    step4_response_status = response.status_code
    step4_response_body = response.json()

    step5_demo_result = run_demo()

    debug_state: dict[str, object] = {
        "step1_payload": step1_payload,
        "step2_calc_preview": step2_calc_preview,
        "step3_app": step3_app,
        "step4_response_status": step4_response_status,
        "step4_response_body": step4_response_body,
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
    parser = argparse.ArgumentParser(description="web_data_flow debug main：用于 F5 逐行观察请求/响应")
    parser.add_argument("--mode", choices=("demo", "serve"), default="demo", help="demo=逐行调试, serve=启动 API")
    parser.add_argument("--host", default="127.0.0.1", help="服务监听地址（mode=serve 时生效）")
    parser.add_argument("--port", type=int, default=8000, help="服务端口（mode=serve 时生效）")
    parser.add_argument("--print-trace", choices=("y", "n"), default="y", help="是否打印 trace")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    if args.mode == "serve":
        import uvicorn

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return

    debug_state = build_debug_state()
    if args.print_trace == "y":
        demo_obj = debug_state.get("demo_result")
        if isinstance(demo_obj, dict):
            _print_demo_result(demo_obj)


if __name__ == "__main__":
    _main()
