# web_data_flow 模块

这个模块用于学习“网页前端和后端如何通过 HTTP/JSON 交换数据”。

## 模块解决的问题

前端页面点击按钮后，数据怎么到后端？后端怎么算结果？结果又怎么回到前端？
本模块用一个最小订单汇总案例，把整条链路拆成可观察步骤。

## 核心概念

### 1) HTTP 请求/响应生命周期

1. 前端构造请求体（JSON）。
2. 前端通过 `POST /api/orders/summary` 发送请求。
3. 后端解析 JSON、执行业务函数。
4. 后端返回状态码 + JSON 结果。
5. 前端解析结果并更新界面。

### 2) REST + JSON

- REST：通过 URL + HTTP 方法表达资源操作。
- JSON：前后端共享的结构化数据格式。

### 3) 状态码意义

- `200`：请求成功。
- `400`：请求结构合法，但业务参数不合法。
- `422`：请求体字段缺失/类型不匹配（FastAPI 自动校验）。

## 代码实现映射（`component.py`）

- `calc_order_summary(...)`：纯业务计算，负责金额逻辑与参数校验。
- `create_app()`：创建 FastAPI 应用并暴露 API。
- `run_demo()`：固定样例演示完整链路并输出 `trace`。

## trace 事件顺序

固定顺序如下（便于测试和 notebook 观察）：

1. `frontend_prepare_request`
2. `http_request_sent`
3. `backend_request_received`
4. `backend_response_generated`
5. `frontend_response_parsed`
6. `model_final_answer`

## 运行方式

### 1) 运行离线教学演示

```bash
python web_data_flow/component.py --mode demo
```

### 2) 启动本地 API 服务

```bash
python web_data_flow/component.py --mode serve --host 127.0.0.1 --port 8000
```

### 3) 手动请求示例

```bash
curl -X POST http://127.0.0.1:8000/api/orders/summary \
  -H "Content-Type: application/json" \
  -d '{"items":[{"sku":"Keyboard","qty":2,"unit_price":49.5}],"tax_rate":0.1}'
```

## 常见错误与排查

- 报 `422 Unprocessable Entity`
原因：请求 JSON 字段缺失或类型错误。
排查：检查 `items/qty/unit_price/tax_rate` 的键名和类型。

- 报 `400` 且提示 `qty must be > 0`
原因：业务参数不合法。
排查：确保 `qty > 0` 且 `unit_price >= 0`。

- 前端请求跨域失败（CORS）
原因：浏览器默认限制跨域。
排查：确认后端已启用 CORS，且前端来源是 `localhost:3000` 或 `127.0.0.1:3000`。

## 可扩展方向

1. 加入订单折扣与优惠券逻辑。
2. 增加数据库持久化（SQLite/PostgreSQL）。
3. 增加鉴权（JWT）与用户态订单接口。
