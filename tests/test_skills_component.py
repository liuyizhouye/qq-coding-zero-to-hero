import pytest

import skills.component as skills_component


def test_load_skill_catalog_has_expected_shape() -> None:
    catalog = skills_component.load_skill_catalog()

    assert isinstance(catalog, list)
    assert len(catalog) >= 2
    for item in catalog:
        assert isinstance(item["name"], str)
        assert isinstance(item["description"], str)
        assert isinstance(item["triggers"], list)


@pytest.mark.online
def test_request_skill_routing_online_schema() -> None:
    catalog = skills_component.load_skill_catalog()
    routing = skills_component.request_skill_routing(
        "请使用 skill-creator，帮我创建一个新技能模板。",
        catalog,
    )

    assert isinstance(routing["matched"], bool)
    assert isinstance(routing["score"], int)
    assert isinstance(routing["trigger_hits"], list)
    assert isinstance(routing["execution_plan"], list)
    assert isinstance(routing["explanation"], str)

    if routing["matched"]:
        selected_name = routing["selected_name"]
        assert isinstance(selected_name, str)
        valid_names = {item["name"] for item in catalog}
        assert selected_name in valid_names


@pytest.mark.online
def test_run_demo_online_trace_protocol() -> None:
    result = skills_component.run_demo("请帮我创建一个新技能，用于规范化代码评审流程。")
    trace = result["trace"]
    events = [item["event"] for item in trace]

    assert events[0] == "skill_catalog_loaded"
    assert events[1] == "skill_match_scores"
    assert events[-1] == "model_final_answer"
    assert isinstance(result["final_answer"], str)
    assert result["final_answer"].strip()


@pytest.mark.online
def test_run_demo_online_when_selected_includes_plan_event() -> None:
    result = skills_component.run_demo("请明确使用 skill-creator，为我创建新技能。")
    trace = result["trace"]
    events = [item["event"] for item in trace]

    if "skill_selected" in events:
        selected_event = next(item for item in trace if item["event"] == "skill_selected")
        plan_event = next(item for item in trace if item["event"] == "skill_execution_plan")
        assert isinstance(selected_event.get("name"), str)
        assert isinstance(plan_event.get("plan"), list)
