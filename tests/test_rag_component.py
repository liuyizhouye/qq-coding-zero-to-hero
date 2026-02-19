from pathlib import Path

import pytest

from rag.component import (
    chunk_report,
    load_report_text,
    retrieve_top_k,
    run_demo,
    synthesize_answer,
)


def test_load_report_text_reads_file() -> None:
    text = load_report_text()
    assert isinstance(text, str)
    assert len(text) > 0


def test_load_report_text_raises_when_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.md"
    with pytest.raises(FileNotFoundError):
        load_report_text(str(missing_path))


def test_chunk_report_returns_non_empty_and_overlap_behavior() -> None:
    text = "\n".join([f"paragraph-{idx} {idx * 'x'}" for idx in range(1, 80)])
    chunks = chunk_report(text, chunk_size=120, overlap=20)

    assert len(chunks) >= 2
    # 第二个 chunk 应该包含第一个 chunk 的尾部重叠片段。
    overlap_seed = chunks[0][-20:].strip()
    assert overlap_seed
    assert chunks[1].startswith(overlap_seed)


def test_retrieve_top_k_returns_sorted_scores() -> None:
    chunks = [
        "RAG combines retrieval and generation for grounded answers.",
        "BM25 is a sparse retrieval baseline in information retrieval.",
        "RAG-Token allows token-level document switching.",
    ]
    hits = retrieve_top_k(chunks, "What is RAG-Token?", top_k=2)

    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]
    assert hits[0]["chunk_id"].startswith("chunk_")


def test_synthesize_answer_contains_chunk_citations() -> None:
    hits = [
        {"chunk_id": "chunk_003", "score": 0.9, "text": "RAG-Sequence uses one set of docs for the sequence."},
        {"chunk_id": "chunk_004", "score": 0.7, "text": "RAG-Token can switch documents per token."},
    ]
    answer = synthesize_answer("RAG-Sequence vs RAG-Token", hits)

    assert "[chunk_003]" in answer
    assert "[chunk_004]" in answer


def test_run_demo_returns_protocol_and_event_order() -> None:
    result = run_demo()

    assert "final_answer" in result
    assert "trace" in result
    events = [event["event"] for event in result["trace"]]
    assert events == [
        "report_loaded",
        "report_chunked",
        "query_prepared",
        "retrieval_top_k",
        "answer_synthesized",
        "model_final_answer",
    ]


def test_run_demo_with_custom_query_keeps_query_in_trace() -> None:
    query = "RAG 的核心组件有哪些？"
    result = run_demo(query=query, top_k=2)

    query_event = next(item for item in result["trace"] if item["event"] == "query_prepared")
    assert query_event["query"] == query
    retrieval_event = next(item for item in result["trace"] if item["event"] == "retrieval_top_k")
    hits = retrieval_event["hits"]
    assert isinstance(hits, list)
    assert len(hits) == 2
