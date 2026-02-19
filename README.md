# qq-coding-zero-to-hero

Python 学习项目：function_call + MCP + skills + nanoGPT（模块化）

这是一个按“知识点模块”组织的学习项目。

自建教学模块（`function_call`、`mcp`、`skills`）包含：

1. 中文讲解与案例 `README.md`
2. 可独立运行的 Python 组件 `component.py`
3. 逐步展示中间变量与函数过程的 `walkthrough.ipynb`

当前模块：

- `function_call/`
- `mcp/`
- `skills/`
- `nanoGPT/`（引入自 https://github.com/karpathy/nanoGPT）

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
  nanoGPT/
    README.md
    train.py
    model.py
    ...（保持上游仓库结构）
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

## 安装

```bash
cd "/Users/liuyizhou/Documents/qq-coding-zero-to-hero"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 一次性 Notebook 配置

```bash
python -m ipykernel install --user --name qq-coding-zero-to-hero --display-name "Python (qq-coding-zero-to-hero)"
git config core.hooksPath .githooks
```

说明：

- notebook 默认 kernel 已固定为 `Python (qq-coding-zero-to-hero)`。
- 提交时会自动执行 `scripts/notebook_guard.py`，清理 `outputs/execution_count`，减少无意义 git 变更。

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

`skills`：

```bash
python skills/component.py
```

`nanoGPT`：

```bash
cd nanoGPT
# 按上游说明准备数据、安装依赖并运行
# 详见 nanoGPT/README.md
```

## 打开 Notebook

```bash
python -m jupyter lab
```

然后分别打开：

- `function_call/walkthrough.ipynb`
- `mcp/walkthrough.ipynb`
- `skills/walkthrough.ipynb`

## 运行测试

```bash
python -m pytest -q
```

## 说明

- 默认教学流程是离线 mock，不依赖 API Key。
- `.env.example` 中的变量是可选配置，主要用于你后续扩展到真实 API 场景。
- 推荐为每个项目使用独立解释器（`.venv` 或 `uv run`），避免全局包版本差异导致的编辑器/运行结果不一致。
