from fastapi.testclient import TestClient

from web_data_flow.component import calc_order_summary, create_app, run_demo


def test_calc_order_summary_returns_expected_values() -> None:
    result = calc_order_summary(
        [
            {"sku": "Keyboard", "qty": 2, "unit_price": 49.5},
            {"sku": "Mouse", "qty": 1, "unit_price": 25.0},
        ],
        tax_rate=0.1,
    )

    assert result == {"subtotal": 124.0, "tax_amount": 12.4, "total": 136.4}


def test_health_endpoint_returns_200() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_orders_summary_endpoint_returns_deterministic_fields() -> None:
    payload = {
        "items": [
            {"sku": "Keyboard", "qty": 2, "unit_price": 49.5},
            {"sku": "Mouse", "qty": 1, "unit_price": 25.0},
        ],
        "tax_rate": 0.1,
    }

    with TestClient(create_app()) as client:
        response = client.post("/api/orders/summary", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "subtotal": 124.0,
        "tax_amount": 12.4,
        "total": 136.4,
        "currency": "USD",
    }


def test_orders_summary_invalid_payload_returns_422_or_400() -> None:
    # 这个请求能通过 JSON 结构校验，但会触发业务参数校验（qty <= 0）。
    payload = {
        "items": [{"sku": "Keyboard", "qty": 0, "unit_price": 49.5}],
        "tax_rate": 0.1,
    }

    with TestClient(create_app()) as client:
        response = client.post("/api/orders/summary", json=payload)

    assert response.status_code in (400, 422)


def test_run_demo_trace_order_and_final_answer() -> None:
    result = run_demo()
    trace = result["trace"]
    events = [item["event"] for item in trace]

    assert events == [
        "frontend_prepare_request",
        "http_request_sent",
        "backend_request_received",
        "backend_response_generated",
        "frontend_response_parsed",
        "model_final_answer",
    ]
    assert isinstance(result["final_answer"], str)
