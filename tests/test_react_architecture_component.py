from react_architecture.component import build_architecture_matrix, run_demo


def test_build_architecture_matrix_contains_all_core_architectures() -> None:
    matrix = build_architecture_matrix()
    names = {item["architecture"] for item in matrix}

    assert names == {"MPA", "SPA_CSR", "SSR", "SSG"}


def test_run_demo_trace_events_are_complete_and_ordered() -> None:
    result = run_demo()
    events = [item["event"] for item in result["trace"]]

    assert events == [
        "architecture_matrix_built",
        "mpa_flow",
        "spa_csr_flow",
        "ssr_flow",
        "ssg_flow",
        "beginner_recommendation",
        "model_final_answer",
    ]


def test_run_demo_has_beginner_path_recommendation() -> None:
    result = run_demo()

    recommendation_event = next(item for item in result["trace"] if item["event"] == "beginner_recommendation")
    steps = recommendation_event["steps"]

    assert isinstance(steps, list)
    assert any("MPA" in str(step) for step in steps)
    assert any("SPA" in str(step) for step in steps)
    assert any("SSR" in str(step) or "SSG" in str(step) for step in steps)
    assert "先 MPA，再 SPA(CSR)，再 SSR/SSG" in result["final_answer"]
