# qq-coding-zero-to-hero

Python 学习项目：function_call + MCP（离线模块化）

这是一个按“知识点模块”组织的学习项目。每个模块都包含：

1. 中文讲解与案例 `README.md`
2. 可独立运行的 Python 组件 `component.py`
3. 逐步展示中间变量与函数过程的 `walkthrough.ipynb`

当前模块：

- `function_call/`
- `mcp/`

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
  tests/
    test_function_call_component.py
    test_mcp_component.py
  README.md
  requirements.txt
  pyrightconfig.json
  .env.example
```

## 安装

```bash
cd "/Users/liuyizhou/Documents/qq-coding-zero-to-hero"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行模块

`function_call`：

```bash
python function_call/component.py
```

`mcp` 批准分支：

```bash
python mcp/component.py --approve y
```

`mcp` 拒绝分支：

```bash
python mcp/component.py --approve n
```

## 打开 Notebook

```bash
python -m jupyter lab
```

然后分别打开：

- `function_call/walkthrough.ipynb`
- `mcp/walkthrough.ipynb`

## 运行测试

```bash
python -m pytest -q
```

## 说明

- 默认教学流程是离线 mock，不依赖 API Key。
- `.env.example` 中的变量是可选配置，主要用于你后续扩展到真实 API 场景。
- 推荐为每个项目使用独立解释器（`.venv` 或 `uv run`），避免全局包版本差异导致的编辑器/运行结果不一致。
