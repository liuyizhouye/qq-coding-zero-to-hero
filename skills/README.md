# skills 模块

本模块演示 Skills 的最小闭环（离线 mock 版）：

1. 发现可用 skill 目录（示例：`skill-creator`、`skill-installer`）
2. 按用户输入做触发匹配与打分
3. 选出最匹配 skill 并生成执行计划

## 关键概念

- `skill catalog`：当前会话可用的技能清单（名称、描述、触发词）。
- `trigger match`：根据用户文本命中关键词，给每个 skill 打分。
- `selection`：选出分数最高且大于 0 的 skill。

## 独立运行组件

```bash
python skills/component.py
```

可选：传入自定义输入

```bash
python skills/component.py "请从 GitHub 仓库安装一个 skills 模板"
```

## 默认案例

默认输入：

```text
请帮我创建一个新技能，用于规范化代码评审流程。
```

默认输出中会看到 `trace`，事件顺序：

1. `skill_catalog_loaded`
2. `skill_match_scores`
3. `skill_selected`
4. `skill_execution_plan`
5. `model_final_answer`

## 常见错误与排查

- 现象：最终结果提示“未匹配到专用 skill”
  - 原因：输入里没有明显“创建/更新 skill”或“安装 skill”信号。
  - 处理：在输入中加上关键动作词，例如“创建”“安装”“GitHub 仓库”。

## Notebook

`walkthrough.ipynb` 会逐步展示：

- skill 清单
- 匹配分数
- 选中结果
- 完整 `run_demo` 轨迹
