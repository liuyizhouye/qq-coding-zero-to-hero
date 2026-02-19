# rag 模块

这个模块演示最小 RAG 闭环（检索本地、生成在线）：

1. 读取 `RAG-deep-research-report.md`
2. 切分 chunk（带 overlap）
3. NumPy 余弦检索 Top-k
4. 把命中片段交给在线 LLM 生成回答与引用
5. 输出 `final_answer + trace`

## 模块定位

- 检索层离线可复现（便于理解算法）
- 生成层在线（更接近真实生产链路）
- 不提供离线生成 fallback

## 在线前置条件

```env
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## 关键函数

- `load_report_text(path)`: 读取报告，缺失则 `FileNotFoundError`。
- `chunk_report(text, chunk_size, overlap)`: 切分文本。
- `retrieve_top_k(chunks, query, top_k)`: 余弦相似度检索。
- `request_answer_payload(query, hits)`: 在线生成严格 JSON（`final_answer` + `citations`）。
- `run_demo(query, top_k)`: 统一 trace 主流程。

## trace 事件顺序（固定）

1. `report_loaded`
2. `report_chunked`
3. `query_prepared`
4. `retrieval_top_k`
5. `answer_synthesized`
6. `model_final_answer`

## 运行

```bash
python rag/component.py
python rag/component.py "RAG-Sequence 和 RAG-Token 有什么区别？"
python rag/component.py --top-k 5 "RAG 的核心组件有哪些？"
```

## 常见错误

- `report file not found`: 报告文件缺失。
- `report file is empty`: 报告为空。
- `query has no valid tokens`: 查询无有效 token。
- `model output is not strict JSON`: 生成阶段返回非 JSON。

## 扩展方向

- 替换为真实 embedding 模型。
- 增加 rerank。
- 加入混合检索（BM25 + dense）。
