import argparse
import json
from typing import TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

TraceEvent = dict[str, object]


class DemoResult(TypedDict):
    final_answer: str
    trace: list[TraceEvent]


class OrderItem(BaseModel):
    sku: str = Field(min_length=1)
    qty: float
    unit_price: float


class OrderSummaryRequest(BaseModel):
    items: list[OrderItem]
    tax_rate: float = 0.1


def calc_order_summary(items: list[dict[str, object]], tax_rate: float = 0.1) -> dict[str, float]:
    """计算订单汇总金额（小计、税额、总额）。"""
    if not items:
        raise ValueError("items must contain at least one item")
    if tax_rate < 0:
        raise ValueError("tax_rate must be >= 0")

    subtotal = 0.0
    for index, row in enumerate(items, start=1):
        sku = str(row.get("sku", "")).strip()
        if not sku:
            raise ValueError(f"row {index} missing sku")
        try:
            qty = float(row["qty"])
            unit_price = float(row["unit_price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"row {index} has invalid qty/unit_price") from exc
        if qty <= 0:
            raise ValueError(f"row {index} qty must be > 0")
        if unit_price < 0:
            raise ValueError(f"row {index} unit_price must be >= 0")
        subtotal += qty * unit_price

    tax_amount = subtotal * tax_rate
    total = subtotal + tax_amount
    return {
        "subtotal": round(subtotal, 2),
        "tax_amount": round(tax_amount, 2),
        "total": round(total, 2),
    }


def create_app() -> FastAPI:
    """创建可独立运行的 FastAPI 应用。"""
    app = FastAPI(title="web_data_flow demo", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def get_health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/orders/summary")
    def post_order_summary(payload: OrderSummaryRequest) -> dict[str, object]:
        try:
            result = calc_order_summary(
                [item.model_dump() for item in payload.items],
                tax_rate=payload.tax_rate,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **result,
            "currency": "USD",
        }

    return app


def run_demo() -> DemoResult:
    """使用固定请求演示前后端 REST/JSON 通信全过程。"""
    trace: list[TraceEvent] = []
    payload = {
        "items": [
            {"sku": "Keyboard", "qty": 2, "unit_price": 49.5},
            {"sku": "Mouse", "qty": 1, "unit_price": 25.0},
        ],
        "tax_rate": 0.1,
    }

    trace.append({"event": "frontend_prepare_request", "payload": payload})

    path = "/api/orders/summary"
    trace.append({"event": "http_request_sent", "method": "POST", "path": path})
    trace.append({"event": "backend_request_received", "method": "POST", "path": path, "payload": payload})

    app = create_app()
    with TestClient(app) as client:
        response = client.post(path, json=payload)

    response_body = response.json()
    trace.append(
        {
            "event": "backend_response_generated",
            "status_code": response.status_code,
            "body": response_body,
        }
    )

    trace.append(
        {
            "event": "frontend_response_parsed",
            "status_code": response.status_code,
            "data": response_body,
        }
    )

    total = float(response_body["total"])
    final_answer = f"前后端数据流演示完成：订单总额 {total:.2f} USD，链路已通过 REST/JSON 打通。"
    trace.append({"event": "model_final_answer", "content": final_answer})

    return {
        "final_answer": final_answer,
        "trace": trace,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="web_data_flow: REST/JSON 前后端数据传输教学模块")
    parser.add_argument("--mode", choices=("demo", "serve"), default="demo", help="demo=打印教学 trace，serve=启动本地 API")
    parser.add_argument("--host", default="127.0.0.1", help="服务监听地址（mode=serve 时生效）")
    parser.add_argument("--port", type=int, default=8000, help="服务端口（mode=serve 时生效）")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    if args.mode == "serve":
        import uvicorn

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return

    result = run_demo()
    print("=== TRACE ===")
    for index, event in enumerate(result["trace"], start=1):
        print(f"[{index}]")
        print(json.dumps(event, ensure_ascii=False, indent=2))

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])


if __name__ == "__main__":
    _main()
