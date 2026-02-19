import json
from typing import TypedDict

TraceEvent = dict[str, object]


class DemoResult(TypedDict):
    final_answer: str
    trace: list[TraceEvent]


def build_architecture_matrix() -> list[dict[str, str]]:
    """返回入门阶段最关键的网页架构对比矩阵。"""
    return [
        {
            "architecture": "MPA",
            "render_timing": "服务端拼好完整 HTML 后再返回",
            "data_fetch": "每次跳转都由后端重新拉数据",
            "seo": "天然友好",
            "first_screen": "通常快",
            "interaction": "整页跳转，局部交互弱",
        },
        {
            "architecture": "SPA_CSR",
            "render_timing": "浏览器下载 JS 后客户端渲染",
            "data_fetch": "页面加载后由前端发 API 请求",
            "seo": "需额外策略",
            "first_screen": "可能较慢",
            "interaction": "局部刷新与交互体验强",
        },
        {
            "architecture": "SSR",
            "render_timing": "每次请求都在服务器渲染页面",
            "data_fetch": "服务器渲染阶段先拉数据",
            "seo": "友好",
            "first_screen": "通常快于 CSR",
            "interaction": "首屏好 + 可继续水合交互",
        },
        {
            "architecture": "SSG",
            "render_timing": "构建期预渲染成静态页面",
            "data_fetch": "构建阶段提前拉数据",
            "seo": "友好",
            "first_screen": "极快（可走 CDN）",
            "interaction": "更新依赖重建或增量再生成",
        },
    ]


def run_demo() -> DemoResult:
    """输出架构差异对比 trace，并给出入门学习顺序建议。"""
    trace: list[TraceEvent] = []

    matrix = build_architecture_matrix()
    trace.append({"event": "architecture_matrix_built", "matrix": matrix})

    trace.append(
        {
            "event": "mpa_flow",
            "steps": [
                "浏览器请求页面",
                "后端渲染 HTML",
                "浏览器整页替换",
            ],
            "core_characteristic": "整页跳转，后端主导渲染",
        }
    )

    trace.append(
        {
            "event": "spa_csr_flow",
            "steps": [
                "浏览器先加载 JS",
                "React 在客户端渲染",
                "前端再请求 API 更新状态",
            ],
            "core_characteristic": "前端主导渲染，交互灵活",
        }
    )

    trace.append(
        {
            "event": "ssr_flow",
            "steps": [
                "请求到达服务器",
                "服务器拉数据并渲染 HTML",
                "浏览器首屏展示后完成水合",
            ],
            "core_characteristic": "兼顾首屏与 SEO",
        }
    )

    trace.append(
        {
            "event": "ssg_flow",
            "steps": [
                "构建时预生成页面",
                "静态文件部署到 CDN",
                "用户请求直接命中静态资源",
            ],
            "core_characteristic": "性能稳定，更新依赖构建策略",
        }
    )

    recommendation = [
        "先理解 MPA 的请求-响应思路",
        "再掌握 SPA(CSR) 的状态与 API 调用",
        "然后学习 SSR 与 SSG 的渲染时机差异",
        "最后再扩展 BFF、微服务与实时通信",
    ]
    trace.append({"event": "beginner_recommendation", "steps": recommendation})

    final_answer = "React 架构学习建议：先 MPA，再 SPA(CSR)，再 SSR/SSG，最后扩展到更复杂架构。"
    trace.append({"event": "model_final_answer", "content": final_answer})

    return {"final_answer": final_answer, "trace": trace}


def _main() -> None:
    result = run_demo()

    print("=== TRACE ===")
    for index, event in enumerate(result["trace"], start=1):
        print(f"[{index}]")
        print(json.dumps(event, ensure_ascii=False, indent=2))

    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])


if __name__ == "__main__":
    _main()
