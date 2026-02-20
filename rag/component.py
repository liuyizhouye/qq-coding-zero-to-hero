"""rag 教学模块（检索本地、生成在线）。

流程：
1) 读取报告
2) 切分 chunk
3) 计算相似度并检索 top-k
4) 让模型基于命中片段生成答案与引用
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import TypedDict

import numpy as np
from openai import OpenAI

TraceEvent = dict[str, object]
DEFAULT_QUERY = "RAG-Sequence 和 RAG-Token 有什么区别？"
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = MODULE_DIR / "RAG-deep-research-report.md"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


class RetrievedChunk(TypedDict):
    """检索命中片段结构。"""

    chunk_id: str
    score: float
    text: str


class DemoResult(TypedDict):
    """教学模块统一返回结构。"""

    final_answer: str
    trace: list[TraceEvent]


def _get_client() -> OpenAI:
    """创建在线客户端。"""

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("missing DEEPSEEK_API_KEY; online mode requires a valid API key")
    return OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def _get_model() -> str:
    """读取模型名，默认 deepseek-chat。"""

    return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _request_json(system_prompt: str, user_prompt: str) -> dict[str, object]:
    """请求模型并要求返回严格 JSON。"""

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


def load_report_text(path: str | Path = DEFAULT_REPORT_PATH) -> str:
    """读取报告原文。"""

    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"report file not found: {path}")
    text = report_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("report file is empty")
    return text


def chunk_report(text: str, chunk_size: int = 420, overlap: int = 80) -> list[str]:
    """按近似字符长度切分报告，并保留重叠区。

    overlap 的目的：降低语义在边界处被硬切断的风险。
    """

    if not text.strip():
        raise ValueError("text is empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if not paragraphs:
        raise ValueError("no valid paragraphs found")

    grouped: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue

        candidate = f"{current}\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            grouped.append(current)
            current = paragraph

    if current:
        grouped.append(current)

    if not grouped:
        raise ValueError("no chunks produced")

    if len(grouped) == 1 or overlap == 0:
        return grouped

    chunks = [grouped[0]]
    for index in range(1, len(grouped)):
        prefix = grouped[index - 1][-overlap:].strip()
        if prefix:
            chunks.append(f"{prefix}\n{grouped[index]}")
        else:
            chunks.append(grouped[index])
    return chunks


def _tokenize(text: str) -> list[str]:
    """把中英文与数字 token 提取并统一小写。"""

    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _build_tf_matrix(tokenized_texts: list[list[str]], vocabulary: list[str]) -> np.ndarray:
    """构建词频矩阵（TF）。"""

    vocab_to_index = {term: idx for idx, term in enumerate(vocabulary)}
    matrix = np.zeros((len(tokenized_texts), len(vocabulary)), dtype=np.float32)

    for row_index, tokens in enumerate(tokenized_texts):
        for token in tokens:
            col_index = vocab_to_index.get(token)
            if col_index is not None:
                matrix[row_index, col_index] += 1.0
    return matrix


def retrieve_top_k(chunks: list[str], query: str, top_k: int = 3) -> list[RetrievedChunk]:
    """基于余弦相似度检索 top-k 片段。"""

    if not chunks:
        raise ValueError("chunks must not be empty")
    query_text = query.strip()
    if not query_text:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    chunk_tokens = [_tokenize(chunk) for chunk in chunks]
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        raise ValueError("query has no valid tokens")

    # 建立统一词表：文档与查询向量必须落在同一维度空间。
    vocabulary = sorted({token for tokens in chunk_tokens + [query_tokens] for token in tokens})
    if not vocabulary:
        raise ValueError("vocabulary is empty")

    doc_matrix = _build_tf_matrix(chunk_tokens, vocabulary)
    query_vector = _build_tf_matrix([query_tokens], vocabulary)[0]

    # 先归一化再点乘，等价于余弦相似度。
    doc_norm = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    doc_norm[doc_norm == 0.0] = 1.0
    normalized_docs = doc_matrix / doc_norm

    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0.0:
        raise ValueError("query vector norm is zero")
    normalized_query = query_vector / query_norm

    scores = normalized_docs @ normalized_query
    limit = min(top_k, len(chunks))
    ranked_indices = sorted(range(len(chunks)), key=lambda idx: (-float(scores[idx]), idx))[:limit]

    hits: list[RetrievedChunk] = []
    for idx in ranked_indices:
        hits.append(
            {
                "chunk_id": f"chunk_{idx + 1:03d}",
                "score": float(scores[idx]),
                "text": chunks[idx],
            }
        )
    return hits


def request_answer_payload(query: str, hits: list[RetrievedChunk]) -> dict[str, object]:
    """请求模型基于命中片段生成答案与 citations。"""

    if not hits:
        raise ValueError("hits must not be empty")

    system_prompt = "You are a RAG answer synthesizer. Return strict JSON only, no markdown."
    user_prompt = (
        "请基于给定检索片段回答问题，并给出引用 chunk_id。\n"
        "JSON schema:\n"
        "{\n"
        '  "final_answer": "string",\n'
        '  "citations": ["chunk_001", "chunk_002"]\n'
        "}\n"
        "要求: 只引用给定 hits 中存在的 chunk_id。\n"
        f"query: {query}\n"
        f"hits: {json.dumps(hits, ensure_ascii=False)}"
    )

    payload = _request_json(system_prompt, user_prompt)
    final_answer = payload.get("final_answer")
    citations_obj = payload.get("citations")

    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError("model returned invalid final_answer")
    if not isinstance(citations_obj, list) or not citations_obj:
        raise ValueError("model returned invalid citations")

    citations = [str(item) for item in citations_obj]
    valid_ids = {hit["chunk_id"] for hit in hits}
    if any(citation not in valid_ids for citation in citations):
        raise ValueError("model returned citation outside retrieved hits")

    return {
        "final_answer": final_answer.strip(),
        "citations": citations,
    }


def synthesize_answer(query: str, hits: list[RetrievedChunk]) -> str:
    """生成答案文本（仅返回 answer，不返回 citations）。"""

    payload = request_answer_payload(query, hits)
    return str(payload["final_answer"])


def run_demo(query: str | None = None, top_k: int = 3) -> DemoResult:
    """执行 RAG 演示主流程，并输出可观察 trace。"""

    trace: list[TraceEvent] = []

    # 1) 读取报告。
    report_text = load_report_text()
    trace.append(
        {
            "event": "report_loaded",
            "path": str(DEFAULT_REPORT_PATH),
            "char_count": len(report_text),
        }
    )

    # 2) 文本切分。
    chunks = chunk_report(report_text)
    trace.append(
        {
            "event": "report_chunked",
            "chunk_count": len(chunks),
            "chunk_size": 420,
            "overlap": 80,
        }
    )

    # 3) 准备查询。
    prepared_query = query.strip() if isinstance(query, str) else ""
    if not prepared_query:
        prepared_query = DEFAULT_QUERY
    trace.append(
        {
            "event": "query_prepared",
            "query": prepared_query,
            "top_k": top_k,
        }
    )

    # 4) top-k 检索。
    hits = retrieve_top_k(chunks, prepared_query, top_k=top_k)
    trace.append(
        {
            "event": "retrieval_top_k",
            "top_k": min(top_k, len(chunks)),
            "hits": hits,
        }
    )

    # 5) 模型生成带引用答案。
    answer_payload = request_answer_payload(prepared_query, hits)
    trace.append(
        {
            "event": "answer_synthesized",
            "citations": answer_payload["citations"],
        }
    )

    final_answer = str(answer_payload["final_answer"])
    trace.append({"event": "model_final_answer", "content": final_answer})
    return {"final_answer": final_answer, "trace": trace}


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="在线 RAG 演示：加载报告 -> 切分 -> 检索 -> API 引用式回答")
    parser.add_argument("--top-k", type=int, default=3, help="返回的检索片段数")
    parser.add_argument("prompt", nargs="*", help="可选：自定义 query")
    return parser.parse_args()


def _main() -> None:
    """CLI 入口：打印 trace 与最终答案。"""

    args = _parse_args()
    query = " ".join(args.prompt).strip() or None
    result = run_demo(query=query, top_k=args.top_k)

    print("=== TRACE ===")
    for index, event in enumerate(result["trace"], start=1):
        print(f"[{index}]")
        print(json.dumps(event, ensure_ascii=False, indent=2))

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])


if __name__ == "__main__":
    _main()
