# rag 模块

这个模块演示一个纯离线的最小 RAG（Retrieval-Augmented Generation）闭环：

1. 读取研究报告文本。
2. 按段落切成可检索 chunk。
3. 用 NumPy 做向量化与余弦相似度检索。
4. 取 Top-k 证据并生成带引用的回答。
5. 输出完整 trace，方便观察每一步中间状态。

## 模块目标与最小 RAG 定义

最小 RAG 不追求模型能力上限，而是追求机制可观察：

- 有知识库输入（报告文本）。
- 有检索过程（Top-k 命中）。
- 有生成结果（回答 + 引用）。
- 有可审计轨迹（trace）。

## 为什么这个实现是“最小闭环”

很多 RAG 示例依赖外部 API、向量数据库和复杂编排，初学者不容易看清核心机制。
这个模块只保留必要组件：

- 知识库：`RAG-deep-research-report.md`
- 检索器：token 频次向量 + 余弦相似度
- 生成器：模板化回答 + chunk 引用

这样可以在本地稳定复现流程。

## 从报告到答案的流程图（文本）

1. `load_report_text` 读取报告内容。
2. `chunk_report` 将文本切分成重叠 chunk。
3. `retrieve_top_k` 对 query 进行检索并排序。
4. `synthesize_answer` 将命中证据拼装成回答。
5. `run_demo` 输出统一 `{"final_answer": ..., "trace": ...}`。

## 检索机制解释（余弦相似度、Top-k）

- 先把 chunk 和 query 都转成 token 频次向量。
- 做 L2 归一化后，向量内积等价于余弦相似度。
- 分数越高表示越相关。
- 取分数最高的 Top-k 作为证据。

## 回答生成机制（模板 + 引用）

`synthesize_answer` 不调用外部模型，而是：

1. 用最高分 chunk 生成结论提示。
2. 列出所有命中片段及其分数。
3. 每条证据标注 `[chunk_XXX]` 引用编号。

## 运行方式

默认 query：

```bash
python rag/component.py
```

自定义 query：

```bash
python rag/component.py "RAG-Sequence 和 RAG-Token 有什么区别？"
```

调整召回数量：

```bash
python rag/component.py --top-k 5 "RAG 的核心组件有哪些？"
```

## 常见错误与排查

- `FileNotFoundError: report file not found`
原因：根目录缺少 `RAG-deep-research-report.md`。
排查：确认文件在仓库根目录，且命名完全一致。

- `ValueError: report file is empty`
原因：报告文件为空。
排查：补充有效文本内容。

- `ValueError: query has no valid tokens`
原因：query 为空或只有无效字符。
排查：使用包含中文/英文/数字的自然语言 query。

## 可扩展方向

1. 替换为真实 embedding 模型。
2. 引入 BM25 + 向量混合检索。
3. 增加 rerank（如交叉编码器）。
4. 增加答案质量评估（recall@k、groundedness）。
