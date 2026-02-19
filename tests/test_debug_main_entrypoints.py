import importlib

MODULES = [
    "function_call.main",
    "mcp.main",
    "skills.main",
    "decorator.main",
    "web_data_flow.main",
    "react_architecture.main",
    "rag.main",
]


def test_main_modules_are_importable_and_have_debug_interfaces() -> None:
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert hasattr(module, "build_debug_state")
        assert hasattr(module, "run_debug")


def test_build_debug_state_returns_dict_with_demo_result() -> None:
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        builder = getattr(module, "build_debug_state")
        debug_state = builder()

        assert isinstance(debug_state, dict)
        assert "demo_result" in debug_state
        assert isinstance(debug_state["demo_result"], dict)


def test_mcp_build_debug_state_supports_approve_branches() -> None:
    module = importlib.import_module("mcp.main")

    state_yes = module.build_debug_state(approve=True)
    state_no = module.build_debug_state(approve=False)

    assert state_yes["step1_approve"] is True
    assert state_no["step1_approve"] is False


def test_rag_build_debug_state_supports_custom_query_and_top_k() -> None:
    module = importlib.import_module("rag.main")
    custom_query = "RAG 的核心组件有哪些？"

    state = module.build_debug_state(query=custom_query, top_k=2)

    assert state["step1_query"] == custom_query
    hits = state["step4_hits"]
    assert isinstance(hits, list)
    assert len(hits) == 2
