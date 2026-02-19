# qq-coding-zero-to-hero

一个面向初学者的模块化学习仓库：每个知识点都拆成“可读文档 + 可运行组件 + 可观察 notebook + 可测试用例”。

## 项目目标

1. 先理解原理，再看代码实现。
2. 每个知识点都能独立运行，不依赖复杂外部环境。
3. 每个知识点都能看到中间过程（trace / notebook）。

## 模块总览

- `function_call/`：函数调用闭环（`function_call` -> 本地执行 -> `function_call_output`）。
- `mcp/`：MCP 审批流（`mcp_approval_request` -> `mcp_approval_response` -> 分支处理）。
- `skills/`：技能路由（catalog -> scoring -> selection -> plan）。
- `decorator/`：函数装饰器、参数化装饰器、类装饰器的调用顺序与状态传递。
- `web_data_flow/`：真实 FastAPI REST/JSON 前后端数据传输。
- `react_architecture/`：架构差异对比（MPA / SPA(CSR) / SSR / SSG）+ Next.js 最小实战。
- `rag/`：离线最小 RAG（报告切分 -> 检索 -> 引用式回答）。
- `nanoGPT/`：第三方引入子项目（保持上游内容）。

## 统一教学协议

每个模块都尽量提供：

1. `README.md`：知识点与原理。
2. `component.py`：可独立运行的组件（CLI + 可测试函数）。
3. `walkthrough.ipynb`：中间变量与过程观察。
4. `tests/test_*.py`：关键路径自动化验证。

多数教学模块通过统一返回结构输出：

```python
{
  "final_answer": str,
  "trace": list[dict[str, object]]
}
```

## 安装（Python）

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

这会让 notebook 默认使用统一 kernel，并在提交前自动规范化 notebook 输出。

## 运行 Python 模块

```bash
python function_call/component.py
python mcp/component.py --approve y
python mcp/component.py --approve n
python skills/component.py
python decorator/component.py
python web_data_flow/component.py --mode demo
python web_data_flow/component.py --mode serve --host 127.0.0.1 --port 8000
python react_architecture/component.py
python rag/component.py
python rag/component.py "RAG-Sequence 和 RAG-Token 有什么区别？"
```

## 打开 Notebook

```bash
python -m jupyter lab
```

按模块打开：

- `function_call/walkthrough.ipynb`
- `mcp/walkthrough.ipynb`
- `skills/walkthrough.ipynb`
- `decorator/walkthrough.ipynb`
- `web_data_flow/walkthrough.ipynb`
- `react_architecture/walkthrough.ipynb`
- `rag/walkthrough.ipynb`

## React 架构实战（Next.js）

`react_architecture/frontend_next/` 是真实前端子工程，用于直观看到 CSR / SSR / SSG 行为差异。

### Node 基线

- Node 20 LTS
- npm

### 启动步骤（双终端）

终端 A（Python API）：

```bash
python web_data_flow/component.py --mode serve --host 127.0.0.1 --port 8000
```

终端 B（Next.js）：

```bash
cd react_architecture/frontend_next
cp .env.local.example .env.local
npm install
npm run dev
```

访问：

- `http://localhost:3000/`
- `http://localhost:3000/csr`
- `http://localhost:3000/ssr`
- `http://localhost:3000/ssg`

可选：刷新 SSG 构建快照

```bash
npm run refresh:ssg
```

## 测试

```bash
python -m pytest -q
```

## Notebook 规范化

```bash
python scripts/notebook_guard.py
```

## 目录结构（关键部分）

```text
qq-coding-zero-to-hero/
  function_call/
  mcp/
  skills/
  decorator/
  web_data_flow/
  react_architecture/
    frontend_next/
  rag/
  tests/
  scripts/
  .githooks/
  README.md
  requirements.txt
  pyrightconfig.json
```

## 备注

- 教学模块默认离线/本地可跑，不强依赖外网 API。
- `.env.example` 是未来扩展真实服务时的可选配置。
- `nanoGPT/` 按上游 README 学习。
