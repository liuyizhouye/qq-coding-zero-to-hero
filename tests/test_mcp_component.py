import pytest

import mcp.component as mcp_component


def test_build_mcp_tool_config_contains_required_fields() -> None:
    config = mcp_component.build_mcp_tool_config()

    assert config["type"] == "mcp"
    assert "server_label" in config
    assert "server_url" in config
    assert config["require_approval"] == "always"
    assert isinstance(config["allowed_tools"], list)


@pytest.mark.online
def test_request_mcp_approval_request_online_schema() -> None:
    config = mcp_component.build_mcp_tool_config()
    request = mcp_component.request_mcp_approval_request("什么是 MCP？", config)

    assert request["type"] == "mcp_approval_request"
    assert str(request["approval_request_id"]).startswith("apr_")
    assert isinstance(request.get("name"), str)
    assert isinstance(request.get("arguments"), dict)


@pytest.mark.online
def test_run_demo_approve_true_contains_approval_and_tool_result() -> None:
    result = mcp_component.run_demo("什么是 MCP？", approve=True)
    trace = result["trace"]
    events = [item["event"] for item in trace]

    assert events[0] == "mcp_tool_config"
    assert events[1] == "mcp_approval_request"
    assert events[2] == "mcp_approval_response"
    assert "mcp_tool_result" in events
    assert events[-1] == "model_final_answer"
    assert isinstance(result["final_answer"], str)
    assert result["final_answer"].strip()


@pytest.mark.online
def test_run_demo_approve_false_stops_without_tool_result() -> None:
    result = mcp_component.run_demo("什么是 MCP？", approve=False)
    trace = result["trace"]
    events = [item["event"] for item in trace]

    assert events[0] == "mcp_tool_config"
    assert events[1] == "mcp_approval_request"
    assert events[2] == "mcp_approval_response"
    assert "mcp_tool_result" not in events
    assert events[-1] == "model_final_answer"
    assert isinstance(result["final_answer"], str)
    assert result["final_answer"].strip()


@pytest.mark.online
def test_trace_contains_consistent_approval_request_id() -> None:
    result = mcp_component.run_demo("解释 MCP 审批流", approve=True)
    trace = result["trace"]

    approval_request = next(item for item in trace if item["event"] == "mcp_approval_request")
    approval_response = next(item for item in trace if item["event"] == "mcp_approval_response")

    assert "approval_request_id" in approval_request
    assert approval_request["approval_request_id"] == approval_response["approval_request_id"]
