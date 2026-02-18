# function_call 模块

本模块演示 OpenAI function calling 的最小闭环（离线 mock 版）：

1. 模型返回 `function_call`
2. 本地执行 Python 函数
3. 用同一个 `call_id` 回填 `function_call_output`
4. 模型输出最终答案

## 关键概念

- `function_call`：模型请求调用你定义的函数。
- `call_id`：一次函数调用的唯一标识，回填输出时必须一致。
- `function_call_output`：函数执行完成后反馈给模型的结果。

## 独立运行组件

```bash
python function_call/component.py
```

可选：传入自定义输入

```bash
python function_call/component.py "请计算: NVDA 2@100, MSFT 1@300"
```

## 默认案例

默认输入：

```text
请计算组合总市值: AAPL 10@180.5, TSLA -3@210, SPY 2@500。请先调用工具再回答。
```

默认输出中会看到 `trace`，事件顺序固定：

1. `model_function_call`
2. `local_function_result`
3. `function_call_output`
4. `model_final_answer`

## 常见错误与排查

- 错误：`positions must contain at least one item`
  - 原因：输入文本没有匹配到 `SYMBOL qty@price` 格式。
  - 处理：确认类似 `AAPL 10@180.5` 的写法存在。

- 错误：`row X has invalid qty/price`
  - 原因：数量或价格不是数字。
  - 处理：把 `qty` 和 `price` 改成可解析的数值。

## Notebook

`walkthrough.ipynb` 会逐步展示：

- `call_id`
- `arguments`
- `output`
- 完整 `run_demo` 轨迹
