import argparse
import json
import os
from typing import TypedDict
from uuid import uuid4

# 本模块是一个“mcp 审批流教学最小闭环”：
# 1) 生成审批请求（mcp_approval_request）
# 2) 回填审批结果（mcp_approval_response）
# 3) 根据批准/拒绝走不同分支
# 4) 输出结构化 trace 供学习和审计
DEFAULT_PROMPT = "先通过 MCP 查一句：什么是 Model Context Protocol。"
TraceEvent = dict[str, object]


class DemoResult(TypedDict):
    final_answer: str
    trace: list[TraceEvent]


def build_mcp_tool_config(server_label: str = "demo-mcp") -> dict[str, object]:
    """构造 MCP 工具配置。

    这里允许通过环境变量覆写，方便你后续从 mock 切到真实服务。
    """
    server_url = os.getenv("MCP_SERVER_URL", "https://mock-mcp.local/server")
    allowed = os.getenv("MCP_ALLOWED_TOOLS", "ask_question,read_wiki_structure")
    allowed_tools = [item.strip() for item in allowed.split(",") if item.strip()]
    return {
        "type": "mcp",
        "server_label": server_label,
        "server_url": server_url,
        "require_approval": "always",
        "allowed_tools": allowed_tools,
    }


def mock_mcp_approval_request() -> dict[str, object]:
    """模拟模型侧发出的审批请求。

    approval_request_id 是审批链路的关联键，响应时必须原样带回。
    """
    return {
        "type": "mcp_approval_request",
        "approval_request_id": f"apr_{uuid4().hex[:10]}",
        "server_label": "demo-mcp",
        "name": "ask_question",
        "arguments": {"question": "什么是 Model Context Protocol?"},
    }


def run_demo(user_text: str, approve: bool = True) -> DemoResult:
    """执行 MCP 审批流程演示。

    approve=True 走“允许执行工具”分支；
    approve=False 走“立即终止”分支。
    """
    trace: list[TraceEvent] = []

    # Step 1: 记录本轮将要使用的 MCP 工具配置（便于审计上下文）
    config = build_mcp_tool_config()
    trace.append({"event": "mcp_tool_config", "config": config})

    # Step 2: 模拟模型发起审批请求
    approval_request = mock_mcp_approval_request()
    trace.append({"event": "mcp_approval_request", **approval_request})

    # Step 3: 操作员对同一 approval_request_id 给出批准/拒绝
    approval_response = {
        "type": "mcp_approval_response",
        "approval_request_id": approval_request["approval_request_id"],
        "approve": approve,
        "reason": "approved by operator (mock)" if approve else "rejected by operator (mock)",
    }
    trace.append({"event": "mcp_approval_response", **approval_response})

    # Step 4a: 拒绝分支，流程在此停止，不再进入工具执行
    if not approve:
        final_answer = "已拒绝 MCP 工具调用，流程终止。"
        trace.append({"event": "model_final_answer", "content": final_answer})
        return {"final_answer": final_answer, "trace": trace}

    # Step 4b: 批准分支，模拟远程工具执行结果
    tool_result = {
        "tool_name": str(approval_request["name"]),
        "result": "Model Context Protocol (MCP) 是让模型与外部工具进行标准化通信的协议（mock）。",
    }
    trace.append({"event": "mcp_tool_result", **tool_result})

    # Step 5: 汇总流程并给出最终回答
    final_answer = f"已批准并完成 MCP 调用。{tool_result['result']} 用户问题：{user_text}"
    trace.append({"event": "model_final_answer", "content": final_answer})
    return {"final_answer": final_answer, "trace": trace}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线演示 mcp_approval_request -> mcp_approval_response 的审批流程")
    parser.add_argument("--approve", choices=("y", "n"), default="y", help="审批决策: y=批准, n=拒绝")
    parser.add_argument("prompt", nargs="*", help="可选：覆盖默认用户输入")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    user_text = " ".join(args.prompt).strip() or DEFAULT_PROMPT
    approve = args.approve == "y"
    result = run_demo(user_text=user_text, approve=approve)

    # 教学场景下打印 trace，便于对照审批链路的每个事件。
    print("=== TRACE ===")
    for index, event in enumerate(result["trace"], start=1):
        print(f"[{index}]")
        print(json.dumps(event, ensure_ascii=False, indent=2))

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])


if __name__ == "__main__":
    _main()
