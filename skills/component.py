import argparse
import json
import os
from typing import TypedDict

from openai import OpenAI

DEFAULT_PROMPT = "请帮我创建一个新技能，用于规范化代码评审流程。"
TraceEvent = dict[str, object]


class SkillDefinition(TypedDict):
    name: str
    description: str
    triggers: list[str]


class DemoResult(TypedDict):
    final_answer: str
    trace: list[TraceEvent]


def _get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("missing DEEPSEEK_API_KEY; online mode requires a valid API key")
    return OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def _get_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _request_json(system_prompt: str, user_prompt: str) -> dict[str, object]:
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


def request_skill_routing(user_text: str, catalog: list[SkillDefinition]) -> dict[str, object]:
    system_prompt = "You are a skill router. Return strict JSON only, no markdown."
    user_prompt = (
        "请根据用户输入从 catalog 中选择最匹配技能。\n"
        "JSON schema:\n"
        "{\n"
        '  "matched": true,\n'
        '  "selected_name": "skill-creator",\n'
        '  "score": 3,\n'
        '  "trigger_hits": ["创建", "skill"],\n'
        '  "execution_plan": ["步骤1", "步骤2"],\n'
        '  "explanation": "string"\n'
        "}\n"
        "如果无匹配: matched=false, selected_name=null, execution_plan=[]。\n"
        f"catalog: {json.dumps(catalog, ensure_ascii=False)}\n"
        f"user_text: {user_text}"
    )

    payload = _request_json(system_prompt, user_prompt)

    matched = bool(payload.get("matched", False))
    selected_name_obj = payload.get("selected_name")
    selected_name = str(selected_name_obj).strip() if isinstance(selected_name_obj, str) else None

    if matched:
        if not selected_name:
            raise ValueError("model returned matched=true but selected_name is empty")
        valid_names = {item["name"] for item in catalog}
        if selected_name not in valid_names:
            raise ValueError("model selected unknown skill")

    score = int(payload.get("score", 0))
    trigger_hits_obj = payload.get("trigger_hits", [])
    execution_plan_obj = payload.get("execution_plan", [])
    explanation_obj = payload.get("explanation", "")

    trigger_hits = [str(item) for item in trigger_hits_obj] if isinstance(trigger_hits_obj, list) else []
    execution_plan = [str(item) for item in execution_plan_obj] if isinstance(execution_plan_obj, list) else []
    explanation = str(explanation_obj).strip()

    return {
        "matched": matched,
        "selected_name": selected_name,
        "score": score,
        "trigger_hits": trigger_hits,
        "execution_plan": execution_plan,
        "explanation": explanation,
    }


def run_demo(user_text: str) -> DemoResult:
    trace: list[TraceEvent] = []
    catalog = load_skill_catalog()

    trace.append(
        {
            "event": "skill_catalog_loaded",
            "skills": [{"name": item["name"], "description": item["description"]} for item in catalog],
        }
    )

    routing = request_skill_routing(user_text, catalog)

    scores_payload: list[dict[str, object]]
    if routing["matched"] and isinstance(routing["selected_name"], str):
        scores_payload = [
            {
                "name": routing["selected_name"],
                "score": int(routing["score"]),
                "trigger_hits": list(routing["trigger_hits"]),
                "name_hit": True,
            }
        ]
    else:
        scores_payload = []

    trace.append(
        {
            "event": "skill_match_scores",
            "user_text": user_text,
            "scores": scores_payload,
            "routing": routing,
        }
    )

    if not routing["matched"] or not isinstance(routing["selected_name"], str):
        final_answer = routing["explanation"] or "未匹配到专用 skill。请补充更具体的技能意图。"
        trace.append({"event": "model_final_answer", "content": final_answer})
        return {"final_answer": final_answer, "trace": trace}

    selected_name = routing["selected_name"]
    trace.append(
        {
            "event": "skill_selected",
            "name": selected_name,
            "score": int(routing["score"]),
            "trigger_hits": list(routing["trigger_hits"]),
        }
    )

    execution_plan = list(routing["execution_plan"])
    trace.append({"event": "skill_execution_plan", "name": selected_name, "plan": execution_plan})

    if execution_plan:
        numbered_plan = " ".join(f"{idx}. {step}" for idx, step in enumerate(execution_plan, start=1))
        final_answer = f"已匹配到 {selected_name}。{routing['explanation']} 建议执行：{numbered_plan}"
    else:
        final_answer = f"已匹配到 {selected_name}。{routing['explanation']}"

    trace.append({"event": "model_final_answer", "content": final_answer})
    return {"final_answer": final_answer, "trace": trace}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在线演示 skills 模块：发现技能 -> 匹配 -> 规划")
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
