from skills.component import load_skill_catalog, run_demo, select_skill


def test_select_skill_prefers_creator_for_create_requests() -> None:
    catalog = load_skill_catalog()
    selection = select_skill("请帮我创建一个新技能，用于代码评审。", catalog)
    assert selection is not None
    assert selection["name"] == "skill-creator"
    assert selection["score"] > 0


def test_select_skill_prefers_installer_for_install_requests() -> None:
    catalog = load_skill_catalog()
    selection = select_skill("请从 GitHub 仓库安装一个 skill。", catalog)
    assert selection is not None
    assert selection["name"] == "skill-installer"
    assert selection["score"] > 0


def test_run_demo_trace_has_selection_and_plan_when_matched() -> None:
    result = run_demo("请帮我创建一个新技能，用于规范化代码评审流程。")
    trace = result["trace"]
    events = [item["event"] for item in trace]

    assert events == [
        "skill_catalog_loaded",
        "skill_match_scores",
        "skill_selected",
        "skill_execution_plan",
        "model_final_answer",
    ]
    assert "skill-creator" in result["final_answer"]


def test_run_demo_handles_no_match() -> None:
    result = run_demo("今天天气如何？")
    trace = result["trace"]
    events = [item["event"] for item in trace]

    assert events == [
        "skill_catalog_loaded",
        "skill_match_scores",
        "model_final_answer",
    ]
    assert "未匹配到专用 skill" in result["final_answer"]
