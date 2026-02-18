# mcp 模块

本模块演示 MCP 审批流程（离线 mock 版）：

1. 生成 `mcp_approval_request`
2. 操作员返回 `mcp_approval_response`
3. 根据批准/拒绝进入不同分支

## 关键概念

- `mcp_approval_request`：模型请求你审批一次远程工具调用。
- `mcp_approval_response`：你对这次调用给出批准或拒绝。
- `approval_request_id`：审批请求唯一标识，回填审批结果时必须带上。

## 独立运行组件

批准分支：

```bash
python mcp/component.py --approve y
```

拒绝分支：

```bash
python mcp/component.py --approve n
```

可选：带自定义问题

```bash
python mcp/component.py --approve y "MCP 在工具调用里解决了什么问题？"
```

## 两条分支说明

- `approve=True`
  - 会继续执行 mock MCP 工具结果并输出最终答案。
- `approve=False`
  - 会立刻终止并返回拒绝说明。

## Notebook

`walkthrough.ipynb` 会对比批准与拒绝两条路径，并展示：

- `approval_request_id`
- `approve`
- `reason`
- 后续状态变化（是否进入工具执行）
