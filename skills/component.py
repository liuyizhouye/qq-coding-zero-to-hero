import argparse
import json
from typing import TypedDict

DEFAULT_PROMPT = "请帮我创建一个新技能，用于规范化代码评审流程。"
TraceEvent = dict[str, object]


class SkillDefinition(TypedDict):
    name: str
    description: str
    triggers: list[str]


class SkillSelection(TypedDict):
    name: str
    description: str
    triggers: list[str]
    score: int
    trigger_hits: list[str]
    name_hit: bool


class DemoResult(TypedDict):
    final_answer: str
    trace: list[TraceEvent]


def load_skill_catalog() -> list[SkillDefinition]:
    return [
        {
            "name": "skill-creator",
            "description": "用于创建或更新一个技能，包含 SKILL.md、资源规划与验证流程。",
            "triggers": ["创建", "新建", "更新", "编写", "设计", "skill", "技能"],
        },
        {
            "name": "skill-installer",
            "description": "用于列出可安装技能，或从 curated 列表 / GitHub 路径安装技能。",
            "triggers": ["安装", "列出", "列表", "github", "仓库", "下载", "skill", "技能"],
        },
    ]


def match_skill_scores(user_text: str, catalog: list[SkillDefinition] | None = None) -> list[dict[str, object]]:
    if catalog is None:
        catalog = load_skill_catalog()
    text = user_text.lower()
    scored: list[dict[str, object]] = []
    for skill in catalog:
        trigger_hits = [token for token in skill["triggers"] if token.lower() in text]
        name_hit = skill["name"].lower() in text
        score = len(trigger_hits) + (2 if name_hit else 0)
        scored.append(
            {
                "name": skill["name"],
                "score": score,
                "trigger_hits": trigger_hits,
                "name_hit": name_hit,
            }
        )
    return sorted(scored, key=lambda item: (-int(item["score"]), str(item["name"])))


def select_skill(user_text: str, catalog: list[SkillDefinition] | None = None) -> SkillSelection | None:
    if catalog is None:
        catalog = load_skill_catalog()
    ranked = match_skill_scores(user_text, catalog)
    if not ranked:
        return None
    top = ranked[0]
    if int(top["score"]) <= 0:
        return None
    matched = next(item for item in catalog if item["name"] == top["name"])
    return {
        "name": matched["name"],
        "description": matched["description"],
        "triggers": matched["triggers"],
        "score": int(top["score"]),
        "trigger_hits": list(top["trigger_hits"]),
        "name_hit": bool(top["name_hit"]),
    }


def build_execution_plan(selection: SkillSelection) -> list[str]:
    if selection["name"] == "skill-creator":
        return [
            "确认 skill 的目标任务与触发语句。",
            "规划可复用资源（scripts/references/assets）。",
            "编写或更新 SKILL.md（包含 name/description）。",
            "运行 quick_validate.py 做结构校验。",
        ]
    if selection["name"] == "skill-installer":
        return [
            "确认来源（curated 列表或 GitHub 仓库路径）。",
            "安装到 $CODEX_HOME/skills 并检查目录结构。",
            "校验 SKILL.md 的触发描述是否准确。",
            "执行一次最小触发用例验证安装结果。",
        ]
    return [
        "明确用户目标。",
        "选择可复用流程。",
        "执行并验证结果。",
    ]


def run_demo(user_text: str) -> DemoResult:
    trace: list[TraceEvent] = []
    catalog = load_skill_catalog()

    trace.append(
        {
            "event": "skill_catalog_loaded",
            "skills": [{"name": item["name"], "description": item["description"]} for item in catalog],
        }
    )

    scores = match_skill_scores(user_text, catalog)
    trace.append({"event": "skill_match_scores", "user_text": user_text, "scores": scores})

    selection = select_skill(user_text, catalog)
    if selection is None:
        final_answer = "未匹配到专用 skill。请补充你是要“创建/更新 skill”还是“安装 skill”。"
        trace.append({"event": "model_final_answer", "content": final_answer})
        return {"final_answer": final_answer, "trace": trace}

    trace.append(
        {
            "event": "skill_selected",
            "name": selection["name"],
            "score": selection["score"],
            "trigger_hits": selection["trigger_hits"],
        }
    )

    plan = build_execution_plan(selection)
    trace.append({"event": "skill_execution_plan", "name": selection["name"], "plan": plan})

    numbered_plan = " ".join(f"{idx}. {step}" for idx, step in enumerate(plan, start=1))
    final_answer = f"已匹配到 {selection['name']}。建议执行：{numbered_plan}"
    trace.append({"event": "model_final_answer", "content": final_answer})
    return {"final_answer": final_answer, "trace": trace}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线演示 skills 模块：发现技能 -> 匹配 -> 规划")
    parser.add_argument("prompt", nargs="*", help="可选：覆盖默认用户输入")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    user_text = " ".join(args.prompt).strip() or DEFAULT_PROMPT
    result = run_demo(user_text)

    print("=== TRACE ===")
    for index, event in enumerate(result["trace"], start=1):
        print(f"[{index}]")
        print(json.dumps(event, ensure_ascii=False, indent=2))

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])


if __name__ == "__main__":
    _main()
