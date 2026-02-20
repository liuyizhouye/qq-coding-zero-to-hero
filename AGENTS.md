# AGENTS.md

## Purpose & Scope
- 本文档是 `qq-coding-zero-to-hero` 的仓库级长期记忆，必须作为后续协作与改动的统一执行基线。
- 作用范围必须覆盖仓库根目录下所有教学模块与测试目录，不允许只按单模块局部约定执行。
- 任何涉及运行入口、调试方式、类型检查、代码风格的变更，必须先对照本文档。
- 反例关键词（禁止）：`按感觉改配置`、`只在当前文件修到不报错`。

## Repo Snapshot
- `function_call/`：在线 function calling 闭环（模型函数调用 -> 本地函数 -> 回填 -> 最终回答）。
- `mcp/`：在线 MCP 审批流（`mcp_approval_request` / `mcp_approval_response` / 分支处理）。
- `skills/`：在线 skill 路由（catalog -> LLM 路由 JSON -> 执行计划）。
- `rag/`：本地检索 + 在线生成（切分/Top-k 检索 + LLM 回答）。
- `decorator/`：Python 装饰器教学模块（函数、参数化、类装饰器）。
- `web_data_flow/`：FastAPI REST/JSON 前后端数据流教学模块。
- `react_architecture/`：MPA / SPA(CSR) / SSR / SSG 架构对比与 Next.js 最小实战。
- `tests/`：统一测试入口（含 `online` 标记测试）。
- `nanoGPT/`：第三方引入目录，必须按上游项目管理，不在本仓库教学规则内随意重构。

## Environment Baseline
- Python 解释器必须使用：`/Users/liuyizhou/.venvs/vscode-py312/bin/python`。
- 类型检查主配置必须是：`pyrightconfig.json`。
- VSCode 基线配置必须是：`.vscode/settings.json` 与 `.vscode/launch.json`。
- 工作区必须打开仓库根目录：`qq-coding-zero-to-hero`。
- 正确命令示例：
  - `cat pyrightconfig.json`
  - `cat .vscode/settings.json`
  - `cat .vscode/launch.json`

## Canonical Run & Debug
- 调试入口必须统一使用模块启动：`python -m <package>.main`。
- 正确调试命令示例：
  - `python -m function_call.main --print-trace n`
  - `python -m mcp.main --approve y --print-trace n`
  - `python -m skills.main --print-trace n`
  - `python -m decorator.main --print-trace n`
  - `python -m web_data_flow.main --mode demo --print-trace n`
  - `python -m react_architecture.main --print-trace n`
  - `python -m rag.main --print-trace n`
- 组件演示必须继续使用脚本路径：`python <module>/component.py`。
- 正确组件命令示例：
  - `python function_call/component.py`
  - `python web_data_flow/component.py --mode serve --host 127.0.0.1 --port 8000`
- 测试命令必须统一：`python -m pytest -q`。
- 反例关键词（禁止）：`python function_call/main.py`、`python mcp/main.py`（会导致包导入上下文不稳定）。

## Python Import & Structure Rules
- `main.py` 中禁止使用 `sys.path` 注入修复导入（例如 `sys.path.insert(...)`）。
- 导入区禁止使用 `try/except` 兜底导入（例如 `try: from pkg.x ... except ModuleNotFoundError: from x ...`）。
- 模块级导入必须放在文件顶部，且必须按工具规则排序。
- 正确示例：`from function_call.component import run_demo`（位于顶部导入块）。
- 反例关键词（禁止）：`PROJECT_ROOT = Path(__file__).resolve().parents[1]`、`Module level import not at top of file`。

## Pylance/Pyright Rules
- 当仓库存在 `pyrightconfig.json` 时，VSCode 设置中禁止配置 `python.analysis.extraPaths`。
- Python 版本与虚拟环境必须以 `pyrightconfig.json` 为主，不允许在多个配置源重复覆盖。
- `.vscode/launch.json` 中教学模块调试项必须使用 `"module": "<package>.main"`。
- 正确检查命令示例：`npx -y pyright function_call/main.py`。
- 反例关键词（禁止）：`settingsNotOverridable`、`python.analysis.extraPaths cannot be set when a pyrightconfig.json or pyproject.toml is being used`。

## Lint/Type Hygiene Rules
- 不允许保留无效 `# noqa` 指令；若规则未启用，必须删除对应 `# noqa`。
- 若出现 `RUF100`，必须视为需要立即清理的代码卫生问题。
- 仅可在确有业务语义（例如教学 trace 不中断）时使用宽异常捕获；其余场景必须收窄异常类型。
- 正确检查命令示例：
  - `uvx ruff check function_call/main.py function_call/component.py`
  - `npx -y pyright function_call/main.py`
- 反例关键词（禁止）：`Unused noqa directive`、`# noqa: BLE001`（在未启用 BLE 规则时）。

## LLM Modules Runtime Contract
- 在线模块（`function_call`、`mcp`、`skills`、`rag`）必须依赖以下环境变量：
  - `DEEPSEEK_API_KEY`
  - `DEEPSEEK_BASE_URL`
  - `DEEPSEEK_MODEL`
- 缺少 `DEEPSEEK_API_KEY`、网络异常、模型返回非 JSON，必须显式失败；禁止静默降级或伪造成功。
- 密钥必须通过本地环境变量注入，禁止写入仓库文件与文档示例中的真实值。
- 正确命令示例：`python -m function_call.main --print-trace n`（在已配置环境变量前提下）。
- 反例关键词（禁止）：`离线兜底自动成功`、`把真实 key 写进 .env 并提交`。

## Change Checklist
- 每次改动运行入口、调试配置、类型检查配置后，必须逐项自检：
  - [ ] `pyrightconfig.json` 仍是类型检查主配置源。
  - [ ] `.vscode/settings.json` 不包含 `python.analysis.extraPaths`。
  - [ ] `.vscode/launch.json` 对教学模块仍使用 `module` 启动。
  - [ ] `main.py` 无 `sys.path` 注入、无导入区 `try/except` 兜底。
  - [ ] 无无效 `# noqa`（尤其 `RUF100`）。
  - [ ] 至少执行 1 条运行命令与 1 条检查命令。
- 最小验证命令集合（每次改配置后至少执行）：
  - `python -m decorator.main --print-trace n`
  - `npx -y pyright function_call/main.py`
  - `uvx ruff check function_call/main.py`

## Maintenance Mechanism
- 触发更新条件（任一命中即必须更新本文档）：
  - 配置变更：`pyrightconfig.json`、`.vscode/settings.json`、`.vscode/launch.json`。
  - 启动方式变更：`python -m <package>.main` 或 `component.py` 命令约定变化。
  - 规则变更：导入规范、lint/type 规范、异常处理策略变化。
  - 模块变更：新增/删除教学模块或模块职责变化。
- 责任归属必须遵循：谁改能力，谁同步更新 `AGENTS.md`。
- 最小维护节奏必须遵循：每月至少一次轻审（核对命令可执行、规则未过期）。
- 维护执行步骤必须遵循：
  - 1) 更新文档。
  - 2) 运行最小验证命令集合。
  - 3) 在决策日志追加记录。

## Decision Log Template
```md
### [YYYY-MM-DD] 变更标题
- 变更内容：
- 变更原因：
- 影响范围：
- 验证命令：
  - `python -m decorator.main --print-trace n`
  - `npx -y pyright function_call/main.py`
  - `uvx ruff check function_call/main.py`
- 结论：
- 记录人：
```

最后验证日期：2026-02-20

验证命令列表：
- `python -m decorator.main --print-trace n`
- `npx -y pyright function_call/main.py`
- `uvx ruff check function_call/main.py`
- `rg -n "sys\.path\.insert\(|PROJECT_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[1\]|python\.analysis\.extraPaths|# noqa: BLE001" function_call mcp skills decorator web_data_flow react_architecture rag .vscode pyrightconfig.json`
