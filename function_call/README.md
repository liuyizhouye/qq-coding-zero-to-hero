# function_call 模块

本模块用一个“组合市值计算”案例讲清 function calling 的核心机制：

1. 模型生成函数调用请求（`function_call`）
2. 本地代码执行业务函数
3. 把执行结果用同一个 `call_id` 回填（`function_call_output`）
4. 模型基于工具结果生成最终自然语言答案

## 这个模块解决什么问题

当你希望模型“先调用工具再回答”时，需要一个稳定协议把模型侧和本地代码连接起来。本模块演示的就是这个协议最小闭环。

## 概念背后的原理

### `function_call`

模型并不是直接执行 Python，它只会返回一个结构化调用意图（函数名 + 参数字符串）。真正执行发生在你的本地代码里。

### `call_id`

`call_id` 是调用会话的关联键。你回填结果时必须携带同一个 `call_id`，模型才能知道“这个输出对应哪次调用”。

### `function_call_output`

它是“工具执行结果的结构化回传”。没有这一步，模型无法把工具结果纳入后续推理。

## 实现过程（对应代码）

### 1) 输入解析

`_extract_positions(user_text)` 用正则提取 `SYMBOL qty@price`，把自然语言转换为结构化持仓列表。

### 2) 生成 mock 调用

`mock_model_function_call(user_text)` 组装调用对象：

- `type=function_call`
- `name=calc_portfolio_value`
- `call_id=随机唯一值`
- `arguments=JSON 字符串`

### 3) 执行业务函数

`calc_portfolio_value(positions)` 做参数校验和累加计算：

- 校验持仓是否为空
- 校验 symbol/qty/price 合法性
- 输出 `{"total_value": float}`

### 4) 处理异常与成功分支

`run_demo(user_text)` 中会：

- 解析 `arguments`
- 成功时记录 `local_function_result(status=ok)`
- 失败时记录 `local_function_result(status=error)`

### 5) 回填输出

`run_demo(...)` 会构造 `function_call_output` 事件，并带回同一个 `call_id`。

### 6) 生成最终回答

- 成功：输出格式化总市值
- 失败：输出错误解释，但流程不中断

## Trace 设计

事件顺序固定：

1. `model_function_call`
2. `local_function_result`
3. `function_call_output`
4. `model_final_answer`

这个顺序是测试与教学可复现的关键。

## 如何运行

```bash
python function_call/component.py
```

自定义输入：

```bash
python function_call/component.py "请计算: NVDA 2@100, MSFT 1@300"
```

## Notebook 学习重点

`walkthrough.ipynb` 建议按以下问题观察：

1. `arguments` 在“字符串”和“字典”之间如何转换？
2. `call_id` 为什么必须在前后保持一致？
3. 错误输入时，为什么仍然返回 `final_answer`？

## 常见错误

- `positions must contain at least one item`
  - 说明没有解析到任何持仓。
- `row X has invalid qty/price`
  - 说明某一行数量或价格不是合法数字。

## 可扩展方向

1. 支持币种与汇率转换。
2. 支持多个 function_call 串联。
3. 把 mock 调用替换为真实 OpenAI Responses 工具调用。
