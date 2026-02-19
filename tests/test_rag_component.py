from pathlib import Path

import pytest

import rag.component as rag_component


def test_load_report_text_reads_file() -> None:
    text = rag_component.load_report_text()
    assert isinstance(text, str)
    assert len(text) > 0


def test_load_report_text_raises_when_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.md"
    with pytest.raises(FileNotFoundError):
        rag_component.load_report_text(str(missing_path))


def test_chunk_report_returns_non_empty_and_overlap_behavior() -> None:
    text = "\n".join([f"paragraph-{idx} {idx * 'x'}" for idx in range(1, 80)])
    chunks = rag_component.chunk_report(text, chunk_size=120, overlap=20)

    assert len(chunks) >= 2
    overlap_seed = chunks[0][-20:].strip()
    assert overlap_seed
    assert chunks[1].startswith(overlap_seed)


def test_retrieve_top_k_returns_sorted_scores() -> None:
    chunks = [
        "RAG combines retrieval and generation for grounded answers.",
        "BM25 is a sparse retrieval baseline in information retrieval.",
        "RAG-Token allows token-level document switching.",
    ]
    hits = rag_component.retrieve_top_k(chunks, "What is RAG-Token?", top_k=2)

    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]
    assert hits[0]["chunk_id"].startswith("chunk_")


@pytest.mark.online
def test_request_answer_payload_online_contains_valid_citations() -> None:
    hits = [
        {
            "chunk_id": "chunk_001",
            "score": 0.9,
            "text": "RAG-Sequence uses one fixed retrieved set for whole sequence.",
        },
        {
            "chunk_id": "chunk_002",
            "score": 0.8,
            "text": "RAG-Token can switch documents at token level.",
        },
    ]
    payload = rag_component.request_answer_payload("RAG-Sequence 和 RAG-Token 的区别", hits)

    assert isinstance(payload["final_answer"], str)
    assert payload["final_answer"].strip()
    assert isinstance(payload["citations"], list)
    assert payload["citations"]
    assert set(payload["citations"]).issubset({"chunk_001", "chunk_002"})


@pytest.mark.online
def test_run_demo_returns_protocol_and_event_order() -> None:
    result = rag_component.run_demo()

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
    assert isinstance(result["final_answer"], str)
    assert result["final_answer"].strip()


@pytest.mark.online
def test_run_demo_with_custom_query_keeps_query_in_trace() -> None:
    query = "RAG 的核心组件有哪些？"
    result = rag_component.run_demo(query=query, top_k=2)

    query_event = next(item for item in result["trace"] if item["event"] == "query_prepared")
    assert query_event["query"] == query

    retrieval_event = next(item for item in result["trace"] if item["event"] == "retrieval_top_k")
    hits = retrieval_event["hits"]
    assert isinstance(hits, list)
    assert len(hits) == 2

    answer_event = next(item for item in result["trace"] if item["event"] == "answer_synthesized")
    assert isinstance(answer_event["citations"], list)
