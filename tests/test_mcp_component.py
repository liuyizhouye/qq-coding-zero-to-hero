from mcp.component import run_demo


def test_run_demo_approve_true_has_approval_and_result() -> None:
    result = run_demo("什么是 MCP？", approve=True)
    trace = result["trace"]
    events = [item["event"] for item in trace]

    assert "mcp_approval_request" in events
    assert "mcp_approval_response" in events
    assert "mcp_tool_result" in events
    assert events[-1] == "model_final_answer"
    assert isinstance(result["final_answer"], str)
    assert "已批准并完成 MCP 调用" in result["final_answer"]


def test_run_demo_approve_false_stops_early() -> None:
    result = run_demo("什么是 MCP？", approve=False)
    trace = result["trace"]
    events = [item["event"] for item in trace]

    assert "mcp_approval_request" in events
    assert "mcp_approval_response" in events
    assert "mcp_tool_result" not in events
    assert events[-1] == "model_final_answer"
    assert "拒绝" in result["final_answer"]


def test_trace_contains_approval_request_id() -> None:
    result = run_demo("什么是 MCP？", approve=True)
    trace = result["trace"]

    approval_request = next(item for item in trace if item["event"] == "mcp_approval_request")
    approval_response = next(item for item in trace if item["event"] == "mcp_approval_response")

    assert "approval_request_id" in approval_request
    assert approval_request["approval_request_id"] == approval_response["approval_request_id"]
