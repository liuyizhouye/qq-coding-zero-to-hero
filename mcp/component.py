"""mcp 教学模块（在线模式）。

这个模块聚焦审批流本身，而不是远端工具细节：
1) 模型提出 mcp_approval_request
2) 本地给出 mcp_approval_response
3) 根据 approve 分支继续或终止
4) 模型输出最终总结
"""

import argparse
import json
import os
from typing import TypedDict
from uuid import uuid4

from openai import OpenAI

DEFAULT_PROMPT = "先通过 MCP 查一句：什么是 Model Context Protocol。"
TraceEvent = dict[str, object]


class DemoResult(TypedDict):
    """教学模块统一返回结构。"""

    final_answer: str
    trace: list[TraceEvent]


def _get_client() -> OpenAI:
    """创建 DeepSeek(OpenAI 兼容) 客户端。"""

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("missing DEEPSEEK_API_KEY; online mode requires a valid API key")
    return OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def _get_model() -> str:
    """读取模型名，默认 deepseek-chat。"""

    return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _request_json(system_prompt: str, user_prompt: str) -> dict[str, object]:
    """统一的在线请求入口，要求模型严格输出 JSON。"""

    client = _get_client()
    response = client.chat.completions.create(
        model=_get_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )

    message = response.choices[0].message.content
    if not isinstance(message, str):
        raise ValueError("model did not return text content")

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model output is not strict JSON: {message}") from exc

    if not isinstance(payload, dict):
        raise ValueError("model JSON output must be an object")
    return payload


def build_mcp_tool_config(server_label: str = "demo-mcp") -> dict[str, object]:
    """构造 MCP 工具配置。

    这里把审批策略和工具白名单集中在一个对象里，便于 trace 展示与测试断言。
    """

    server_url = os.getenv("MCP_SERVER_URL", "https://demo-mcp.local/server")
    allowed = os.getenv("MCP_ALLOWED_TOOLS", "ask_question,read_wiki_structure")
    allowed_tools = [item.strip() for item in allowed.split(",") if item.strip()]
    return {
        "type": "mcp",
        "server_label": server_label,
        "server_url": server_url,
        "require_approval": "always",
        "allowed_tools": allowed_tools,
    }


def request_mcp_approval_request(user_text: str, config: dict[str, object]) -> dict[str, object]:
    """请求模型生成 mcp_approval_request 结构。"""

    system_prompt = (
        "You generate a MCP approval request object. "
        "Return strict JSON only, no markdown."
    )
    user_prompt = (
        "请根据用户意图生成待审批工具调用。\n"
        "JSON schema:\n"
        "{\n"
        '  "name": "ask_question",\n'
        '  "arguments": {"question": "string"},\n'
        '  "reason": "string"\n'
        "}\n"
        f"用户输入: {user_text}\n"
        f"允许工具: {json.dumps(config.get('allowed_tools', []), ensure_ascii=False)}"
    )

    payload = _request_json(system_prompt, user_prompt)
    name = str(payload.get("name", "")).strip()
    arguments_obj = payload.get("arguments")

    if not name:
        raise ValueError("model returned empty tool name")
    if not isinstance(arguments_obj, dict):
        raise ValueError("model returned invalid arguments")

    question = str(arguments_obj.get("question", "")).strip()
    if not question:
        raise ValueError("model returned empty arguments.question")

    return {
        "type": "mcp_approval_request",
        "approval_request_id": f"apr_{uuid4().hex[:10]}",
        "server_label": str(config.get("server_label", "demo-mcp")),
        "name": name,
        "arguments": {"question": question},
    }


def request_mcp_final_answer(
    user_text: str,
    approve: bool,
    approval_request: dict[str, object],
    tool_result: dict[str, object] | None,
) -> str:
    """让模型把审批流程翻译成最终可读结论。"""

    system_prompt = "You summarize MCP approval workflow results. Return strict JSON only."
    user_prompt = (
        "请根据审批流程给出中文总结。\n"
        "JSON schema:\n"
        "{\n"
        '  "final_answer": "string"\n'
        "}\n"
        f"用户输入: {user_text}\n"
        f"审批是否通过: {approve}\n"
        f"审批请求: {json.dumps(approval_request, ensure_ascii=False)}\n"
        f"工具结果: {json.dumps(tool_result, ensure_ascii=False) if tool_result is not None else 'null'}"
    )

    payload = _request_json(system_prompt, user_prompt)
    final_answer = payload.get("final_answer")
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError("model returned invalid final_answer")
    return final_answer.strip()


def run_demo(user_text: str, approve: bool = True) -> DemoResult:
    """执行 MCP 审批演示流程。"""

    trace: list[TraceEvent] = []

    # 1) 写入工具配置事件。
    config = build_mcp_tool_config()
    trace.append({"event": "mcp_tool_config", "config": config})

    # 2) 请求模型生成审批请求。
    approval_request = request_mcp_approval_request(user_text=user_text, config=config)
    trace.append({"event": "mcp_approval_request", **approval_request})

    # 3) 本地模拟审批决策。
    approval_response = {
        "type": "mcp_approval_response",
        "approval_request_id": approval_request["approval_request_id"],
        "approve": approve,
        "reason": "approved by operator" if approve else "rejected by operator",
    }
    trace.append({"event": "mcp_approval_response", **approval_response})

    # 4a) 拒绝分支：立即结束。
    if not approve:
        final_answer = request_mcp_final_answer(
            user_text=user_text,
            approve=False,
            approval_request=approval_request,
            tool_result=None,
        )
        trace.append({"event": "model_final_answer", "content": final_answer})
        return {"final_answer": final_answer, "trace": trace}

    # 4b) 通过分支：继续写入工具执行结果。
    tool_result = {
        "tool_name": str(approval_request["name"]),
        "result": "已执行 MCP 工具调用（在线模式示例）。",
    }
    trace.append({"event": "mcp_tool_result", **tool_result})

    final_answer = request_mcp_final_answer(
        user_text=user_text,
        approve=True,
        approval_request=approval_request,
        tool_result=tool_result,
    )
    trace.append({"event": "model_final_answer", "content": final_answer})
    return {"final_answer": final_answer, "trace": trace}


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="在线演示 mcp_approval_request -> mcp_approval_response 审批流程")
    parser.add_argument("--approve", choices=("y", "n"), default="y", help="审批决策: y=批准, n=拒绝")
    parser.add_argument("prompt", nargs="*", help="可选：覆盖默认用户输入")
    return parser.parse_args()


def _main() -> None:
    """CLI 入口：打印完整 trace。"""

    args = _parse_args()
    user_text = " ".join(args.prompt).strip() or DEFAULT_PROMPT
    approve = args.approve == "y"
    result = run_demo(user_text=user_text, approve=approve)

    print("=== TRACE ===")
    for index, event in enumerate(result["trace"], start=1):
        print(f"[{index}]")
        print(json.dumps(event, ensure_ascii=False, indent=2))

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])


if __name__ == "__main__":
    _main()
