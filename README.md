# qq-coding-zero-to-hero

一个面向初学者的模块化学习仓库：每个知识点拆成“文档 + 组件 + notebook + 测试”。

## 模块总览

- `function_call/`: 在线 function calling 闭环（模型函数调用 -> 本地函数 -> 回填 -> 最终回答）。
- `openai_api/`: OpenAI API 功能与参数教学（DeepSeek OpenAI 兼容实调）。
- `mcp/`: 在线 MCP 审批流（`mcp_approval_request` / `mcp_approval_response` / 分支处理）。
- `skills/`: 在线 skill 路由（catalog -> LLM 路由 JSON -> 计划）。
- `rag/`: 本地检索 + 在线生成（报告切分/Top-k 检索 + LLM 引用式回答）。
- `decorator/`: Python 装饰器机制教学。
- `web_data_flow/`: FastAPI REST/JSON 前后端数据传输。
- `react_architecture/`: MPA / SPA(CSR) / SSR / SSG 架构对比 + Next.js 最小实战。
- `nanoGPT/`: 第三方引入子项目（保持上游内容）。

## 在线模式要求（LLM 模块）

以下模块只支持在线 API，不再提供离线仿真：

- `function_call`
- `openai_api`
- `mcp`
- `skills`
- `rag`

必需环境变量：

```env
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
RUN_LLM_API_TESTS=0
```

说明：

- 缺少 `DEEPSEEK_API_KEY` 时会直接报错。
- 网络异常、401/429、模型返回非 JSON 都会直接失败（教学上可见）。
- 不要把真实 key 提交到仓库。

## 安装

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

## 运行模块

```bash
python function_call/component.py
python openai_api/component.py
python mcp/component.py --approve y
python mcp/component.py --approve n
python skills/component.py
python rag/component.py
python rag/component.py --top-k 5 "RAG-Sequence 和 RAG-Token 有什么区别？"

python decorator/component.py
python web_data_flow/component.py --mode demo
python web_data_flow/component.py --mode serve --host 127.0.0.1 --port 8000
python react_architecture/component.py
```

## Notebook

```bash
python -m jupyter lab
```

- `function_call/walkthrough.ipynb`
- `openai_api/walkthrough.ipynb`
- `mcp/walkthrough.ipynb`
- `skills/walkthrough.ipynb`
- `rag/walkthrough.ipynb`
- `decorator/walkthrough.ipynb`
- `web_data_flow/walkthrough.ipynb`
- `react_architecture/walkthrough.ipynb`

### 细讲版 walkthrough 标准

- 每个教学 notebook 必须按固定顺序组织：`目标与先修` -> `流程总览` -> `环境与依赖检查` -> `步骤拆解（逐步）` -> `端到端结果` -> `常见错误` -> `总结`。
- 每个代码单元前必须有 markdown 讲解，且必须包含：`本步做什么`、`为什么这样做`、`输入`、`输出`、`观察点`。
- 每个代码单元必须展示中间过程（变量、结构化对象或 trace）；禁止只调用函数不展示结果。
- 在线模块（`function_call/openai_api/mcp/skills/rag`）缺少 `DEEPSEEK_API_KEY` 时必须显式失败，不做静默跳过。
- 建议学习顺序：`decorator` -> `web_data_flow` -> `react_architecture` -> `function_call` -> `openai_api` -> `mcp` -> `skills` -> `rag`。

## 测试

默认执行本地测试（在线测试自动跳过）：

```bash
python -m pytest -q
```

执行在线测试：

```bash
export RUN_LLM_API_TESTS=1
python -m pytest -q
```

## VS Code F5 Debug

- 调试入口在各模块 `main.py`。
- 配置文件：`.vscode/launch.json`、`.vscode/settings.json`。
- 推荐从 `build_debug_state(...)` 的 `step*` 变量开始逐行观察。

## Notebook 规范化

```bash
python scripts/notebook_guard.py
```
