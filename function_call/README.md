# function_call 模块

这个模块演示在线 function calling 的完整闭环：

1. LLM 生成 `function_call`（含 `call_id` 与参数 JSON）
2. 本地执行 `calc_portfolio_value`
3. 回填 `function_call_output`
4. LLM 基于工具输出生成最终回答

## 核心概念

- `function_call`: 模型返回“要调用哪个函数、传什么参数”。
- `call_id`: 关联同一次调用与回填结果的关键字段。
- `function_call_output`: 本地函数执行结果的结构化回传。

## 在线前置条件

```env
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

缺少 key、网络不可达、模型返回非 JSON 都会直接报错。

## 代码实现映射

- `request_model_function_call(user_text)`: 在线请求函数调用对象，要求严格 JSON。
- `calc_portfolio_value(positions)`: 本地纯函数，计算 `sum(qty * price)`。
- `request_model_final_answer(...)`: 在线请求最终回答 JSON。
- `run_demo(user_text)`: 组装完整 trace 并返回统一协议。

## trace 事件顺序（固定）

1. `model_function_call`
2. `local_function_result`
3. `function_call_output`
4. `model_final_answer`

## 运行

```bash
python function_call/component.py
python function_call/component.py "请计算组合总市值: AAPL 10@180.5, TSLA -3@210, SPY 2@500。"
```

## 常见错误

- `missing DEEPSEEK_API_KEY`: 没有配置在线密钥。
- `model output is not strict JSON`: 模型返回了非 JSON 文本。
- `row X has invalid qty/price`: 参数结构正确，但字段值不合法。

## 扩展方向

- 增加多工具串联调用。
- 在 trace 中增加 token 用量与延迟统计。
- 对 `arguments` 增加更严格 schema 校验。
