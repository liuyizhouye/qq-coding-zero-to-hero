# mcp 模块

本模块聚焦 MCP 审批流（approval flow）机制，而不是远程工具本身。

核心流程：

1. 模型发出 `mcp_approval_request`
2. 操作员给出 `mcp_approval_response`
3. 系统根据批准/拒绝走不同分支

## 这个模块解决什么问题

MCP 让模型可调用外部能力，但外部调用可能涉及权限、成本、数据安全。审批流的价值是把“是否允许调用”交给操作员显式决策。

## 概念背后的原理

### `mcp_approval_request`

它表示模型在请求执行某个远端工具动作。该事件应包含足够上下文（工具名、参数、server label）供人判断风险。

### `approval_request_id`

审批流的关联键。`mcp_approval_response` 必须携带这个 ID，才能保证“这条同意/拒绝意见”绑定到正确请求。

### `mcp_approval_response`

它是一个显式策略决策：

- `approve=True`：允许继续执行
- `approve=False`：拒绝并终止当前调用链

## 实现过程（对应代码）

### 1) 构造 MCP 工具配置

`build_mcp_tool_config(...)` 负责聚合配置：

- `server_url`
- `server_label`
- `require_approval=always`
- `allowed_tools` 白名单

### 2) 生成审批请求

`mock_mcp_approval_request()` 模拟模型请求审批，输出请求对象和 `approval_request_id`。

### 3) 写入审批响应

`run_demo(user_text, approve=...)` 根据参数生成 `mcp_approval_response`：

- `approve=True` -> reason: approved
- `approve=False` -> reason: rejected

### 4) 分支执行

- 拒绝分支：立即返回 `model_final_answer`（流程终止）
- 批准分支：继续生成 `mcp_tool_result` 并返回最终答案

## Trace 设计

默认会包含这些关键事件：

1. `mcp_tool_config`
2. `mcp_approval_request`
3. `mcp_approval_response`
4. `mcp_tool_result`（仅批准时存在）
5. `model_final_answer`

这个事件模型能直接映射真实审批系统的审计日志结构。

## 如何运行

批准分支：

```bash
python mcp/component.py --approve y
```

拒绝分支：

```bash
python mcp/component.py --approve n
```

自定义问题：

```bash
python mcp/component.py --approve y "MCP 在工具调用里解决了什么问题？"
```

## Notebook 学习重点

`walkthrough.ipynb` 重点看三件事：

1. `approval_request_id` 在请求/响应中的一致性。
2. 批准与拒绝分支的事件差异。
3. 为什么审批事件应先于工具执行事件。

## 常见误区

- 误区：批准只是 UI 行为，不影响流程。
  - 实际：批准是流程控制信号，决定是否允许进入后续执行。

- 误区：拒绝后还能继续执行同一请求。
  - 实际：拒绝分支应立即终止该调用链。

## 可扩展方向

1. 接入真实风险策略（按工具类别、参数范围自动建议 approve/reject）。
2. 审批结果持久化（数据库审计日志）。
3. 增加多级审批（operator -> admin）。
