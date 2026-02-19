# qq-coding-zero-to-hero

一个面向初学者的 Python 学习仓库，目标是把“概念”拆成可以独立运行、可观察中间过程、可测试验证的教学模块。

## 项目目标

这个项目强调三件事：

1. 先理解原理，再看代码。
2. 每个知识点都可以独立运行。
3. 每个知识点都能看到完整中间状态（trace / notebook）。

## 模块总览

- `function_call/`：理解 function calling 的闭环（调用请求 -> 本地执行 -> 输出回填）。
- `mcp/`：理解 MCP 审批流（approval request -> approval response -> 分支处理）。
- `skills/`：理解 skill 的匹配和路由策略（catalog -> scoring -> selection -> plan）。
- `nanoGPT/`：第三方引入子项目（上游仓库内容，主要用于扩展学习）。

## 统一设计原则（背后原理）

### 1) Offline First（离线优先）

教学模块默认使用 mock 数据，不依赖外部 API。这样可以把注意力集中在流程机制，而不是网络和鉴权问题。

### 2) Trace First（可观测优先）

每个 `run_demo(...)` 都返回统一结构：

```python
{
  "final_answer": str,
  "trace": list[dict[str, object]]
}
```

`trace` 是学习核心：你可以按事件顺序看到系统状态如何一步步演进。

### 3) Deterministic Core（确定性核心）

核心逻辑尽量确定性，便于测试和复现。即使带随机标识（例如 `call_id` / `approval_request_id`），关键行为和事件顺序依然稳定。

### 4) Testable Units（可测试单元）

每个模块都把关键逻辑写成可调用函数，而不是只写脚本入口。这样既能做 CLI 演示，也能做自动化测试。

### 5) Notebook Hygiene（Notebook 可维护）

仓库内设置了 pre-commit 规范化流程，自动清理 notebook 输出和执行计数，减少无意义 git diff。

## 目录结构

```text
qq-coding-zero-to-hero/
  function_call/
    README.md
    component.py
    walkthrough.ipynb
  mcp/
    README.md
    component.py
    walkthrough.ipynb
  skills/
    README.md
    component.py
    walkthrough.ipynb
  tests/
    test_function_call_component.py
    test_mcp_component.py
    test_skills_component.py
  scripts/
    notebook_guard.py
  .githooks/
    pre-commit
  README.md
  requirements.txt
  pyrightconfig.json
  .env.example
```

## 快速开始

```bash
cd "/Users/liuyizhou/Documents/qq-coding-zero-to-hero"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 一次性本地配置

```bash
python -m ipykernel install --user --name qq-coding-zero-to-hero --display-name "Python (qq-coding-zero-to-hero)"
git config core.hooksPath .githooks
```

这一步完成后：

- Notebook 默认 kernel 会落到项目专用环境。
- 提交前会自动规范化 notebook，降低提交噪音。

## 模块运行方式

```bash
python function_call/component.py
python mcp/component.py --approve y
python mcp/component.py --approve n
python skills/component.py
```

## Notebook 学习方式

```bash
python -m jupyter lab
```

按模块打开：

- `function_call/walkthrough.ipynb`
- `mcp/walkthrough.ipynb`
- `skills/walkthrough.ipynb`

推荐方法：先读模块 README 的“原理 + 流程”，再跑 notebook 观察中间状态。

## 测试

```bash
python -m pytest -q
```

## 如何继续扩展模块

新增一个知识点时，保持同样结构：

1. `module_name/README.md`：写清功能、原理、实现步骤。
2. `module_name/component.py`：提供可测试函数 + CLI 入口。
3. `module_name/walkthrough.ipynb`：展示关键中间变量。
4. `tests/test_module_name_component.py`：覆盖主路径和错误路径。

## 备注

- 自建模块默认离线 mock，不依赖 OpenAI key。
- `.env.example` 是未来扩展真实 API 流程时的可选配置。
- `nanoGPT/` 为外部项目内容，建议按其上游 `README.md` 学习。
