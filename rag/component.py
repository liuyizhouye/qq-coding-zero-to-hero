import argparse
import json
import re
from pathlib import Path
from typing import TypedDict

import numpy as np

TraceEvent = dict[str, object]
DEFAULT_QUERY = "RAG-Sequence 和 RAG-Token 有什么区别？"
DEFAULT_REPORT_PATH = "RAG-deep-research-report.md"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


class RetrievedChunk(TypedDict):
    chunk_id: str
    score: float
    text: str


class DemoResult(TypedDict):
    final_answer: str
    trace: list[TraceEvent]


def load_report_text(path: str = DEFAULT_REPORT_PATH) -> str:
    """读取研究报告原文，作为 RAG 知识库输入。"""
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"report file not found: {path}")
    text = report_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("report file is empty")
    return text


def chunk_report(text: str, chunk_size: int = 420, overlap: int = 80) -> list[str]:
    """按段落聚合并加入固定重叠，生成可检索 chunk。"""
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
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _build_tf_matrix(tokenized_texts: list[list[str]], vocabulary: list[str]) -> np.ndarray:
    vocab_to_index = {term: idx for idx, term in enumerate(vocabulary)}
    matrix = np.zeros((len(tokenized_texts), len(vocabulary)), dtype=np.float32)

    for row_index, tokens in enumerate(tokenized_texts):
        for token in tokens:
            col_index = vocab_to_index.get(token)
            if col_index is not None:
                matrix[row_index, col_index] += 1.0
    return matrix


def retrieve_top_k(chunks: list[str], query: str, top_k: int = 3) -> list[RetrievedChunk]:
    """用 token 频次 + 余弦相似度检索最相关的 chunk。"""
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

    vocabulary = sorted({token for tokens in chunk_tokens + [query_tokens] for token in tokens})
    if not vocabulary:
        raise ValueError("vocabulary is empty")

    doc_matrix = _build_tf_matrix(chunk_tokens, vocabulary)
    query_vector = _build_tf_matrix([query_tokens], vocabulary)[0]

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


def _compact_text(text: str, max_chars: int = 88) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return f"{one_line[:max_chars].rstrip()}..."


def synthesize_answer(query: str, hits: list[RetrievedChunk]) -> str:
    """将检索命中片段拼装为带引用的模板化回答。"""
    query_text = query.strip()
    if not query_text:
        raise ValueError("query must not be empty")
    if not hits:
        raise ValueError("hits must not be empty")

    first_hit = hits[0]
    conclusion = _compact_text(first_hit["text"], max_chars=70)

    evidence_lines = [
        f"- [{hit['chunk_id']}] score={hit['score']:.4f} | {_compact_text(hit['text'])}" for hit in hits
    ]

    return (
        f"针对问题“{query_text}”，我基于检索到的报告片段给出结论：{conclusion}\n"
        "下面是可追溯证据（按相关性排序）：\n"
        f"{'\n'.join(evidence_lines)}"
    )


def run_demo(query: str | None = None, top_k: int = 3) -> DemoResult:
    """执行离线最小 RAG 流程并返回统一 trace。"""
    trace: list[TraceEvent] = []

    report_text = load_report_text()
    trace.append(
        {
            "event": "report_loaded",
            "path": DEFAULT_REPORT_PATH,
            "char_count": len(report_text),
        }
    )

    chunks = chunk_report(report_text)
    trace.append(
        {
            "event": "report_chunked",
            "chunk_count": len(chunks),
            "chunk_size": 420,
            "overlap": 80,
        }
    )

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

    hits = retrieve_top_k(chunks, prepared_query, top_k=top_k)
    trace.append(
        {
            "event": "retrieval_top_k",
            "top_k": min(top_k, len(chunks)),
            "hits": hits,
        }
    )

    answer = synthesize_answer(prepared_query, hits)
    trace.append(
        {
            "event": "answer_synthesized",
            "citations": [hit["chunk_id"] for hit in hits],
        }
    )

    trace.append({"event": "model_final_answer", "content": answer})
    return {"final_answer": answer, "trace": trace}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线最小 RAG 演示：加载报告 -> 切分 -> 检索 -> 引用式回答")
    parser.add_argument("--top-k", type=int, default=3, help="返回的检索片段数")
    parser.add_argument("prompt", nargs="*", help="可选：自定义 query")
    return parser.parse_args()


def _main() -> None:
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
