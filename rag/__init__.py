"""rag lesson package."""

from .component import DemoResult, RetrievedChunk, chunk_report, load_report_text, retrieve_top_k, run_demo, synthesize_answer

__all__ = [
    "DemoResult",
    "RetrievedChunk",
    "chunk_report",
    "load_report_text",
    "retrieve_top_k",
    "run_demo",
    "synthesize_answer",
]
