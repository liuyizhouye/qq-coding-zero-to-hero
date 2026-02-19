# mcp 模块

这个模块演示在线 MCP 审批流：

1. LLM 生成 `mcp_approval_request`
2. 操作员给出 `mcp_approval_response`
3. `approve=True/False` 进入不同分支
4. LLM 产出最终总结

## 核心概念

- `mcp_approval_request`: 模型请求调用外部能力。
- `approval_request_id`: 请求与审批响应的关联键。
- `mcp_approval_response`: 明确的策略决策（批准/拒绝）。

## 在线前置条件

```env
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

无 key / 网络异常 / 非 JSON 输出都会直接失败。

## 代码实现映射

- `build_mcp_tool_config(...)`: 生成 MCP 工具配置。
- `request_mcp_approval_request(...)`: 在线生成审批请求对象。
- `request_mcp_final_answer(...)`: 在线生成最终总结。
- `run_demo(user_text, approve=True)`: 审批分支主流程。

## 分支行为

- `approve=False`: 在 `mcp_approval_response` 后立即终止，不执行工具结果分支。
- `approve=True`: 继续写入 `mcp_tool_result`，再生成最终回答。

## trace 关键事件

- `mcp_tool_config`
- `mcp_approval_request`
- `mcp_approval_response`
- `mcp_tool_result`（仅批准分支）
- `model_final_answer`

## 运行

```bash
python mcp/component.py --approve y
python mcp/component.py --approve n
python mcp/component.py --approve y "请解释 MCP 审批机制"
```

## 常见错误

- `missing DEEPSEEK_API_KEY`
- `model output is not strict JSON`
- 401/429/超时等 API 调用异常

## 扩展方向

- 把 `approve` 决策改为策略引擎 + 人工复核。
- 记录审批日志到数据库。
- 增加多级审批链路。
