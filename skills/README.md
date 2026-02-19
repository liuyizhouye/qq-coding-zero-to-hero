# skills 模块

本模块演示“技能路由”的核心机制：系统如何从 skill catalog 中选择最匹配的技能并生成执行计划。

流程主线：

1. 加载 skill catalog
2. 对用户输入做触发词匹配与打分
3. 选出最高分 skill
4. 生成该 skill 对应的执行计划

## 这个模块解决什么问题

当系统内存在多个 skill（例如 `skill-creator`、`skill-installer`）时，需要一个可解释、可测试的选择机制，而不是“拍脑袋选一个”。

## 概念背后的原理

### Skill Catalog

catalog 是系统可用能力的注册表。每个 skill 至少包含：

- `name`
- `description`
- `triggers`

这让“能力定义”和“能力选择”分离，便于扩展。

### Trigger Match + Scoring

匹配策略采用可解释打分：

- 命中触发词：每个 +1
- 直接命中 skill 名称：额外 +2

这样能兼顾语义提示（触发词）和显式指定（直接写 skill 名）。

### Selection

按分数降序排序，取第一名；若最高分 `<= 0` 则判定无匹配，返回引导提示。

### Execution Plan

选中 skill 后，不直接执行复杂动作，而先返回结构化计划。这样用户可以先审阅再执行，降低误操作风险。

## 实现过程（对应代码）

### 1) 定义数据结构

- `SkillDefinition`：catalog 中的技能结构
- `SkillSelection`：选择结果结构（含 score、命中详情）
- `DemoResult`：统一输出协议

### 2) 加载 catalog

`load_skill_catalog()` 返回当前可用技能清单（示例含 `skill-creator` 与 `skill-installer`）。

### 3) 计算匹配分数

`match_skill_scores(user_text, catalog)`：

- 统一小写比对
- 统计触发词命中
- 计算总分并排序

### 4) 选择目标技能

`select_skill(...)`：

- 读取最高分
- 若分值不足，返回 `None`
- 若命中成功，返回完整 `SkillSelection`

### 5) 生成执行计划

`build_execution_plan(selection)` 按 skill name 返回步骤列表。

### 6) 组装 trace 与最终回答

`run_demo(user_text)` 按事件写入 trace，最后输出可读回答。

## Trace 设计

事件顺序：

1. `skill_catalog_loaded`
2. `skill_match_scores`
3. `skill_selected`（若有匹配）
4. `skill_execution_plan`（若有匹配）
5. `model_final_answer`

无匹配场景会跳过第 3、4 步，直接进入最终回答。

## 如何运行

```bash
python skills/component.py
```

自定义输入：

```bash
python skills/component.py "请从 GitHub 仓库安装一个 skills 模板"
```

## Notebook 学习重点

`walkthrough.ipynb` 建议关注：

1. `trigger_hits` 与 `score` 的对应关系。
2. 为什么 `name_hit` 给予额外权重。
3. 无匹配场景如何优雅退化。

## 常见问题

- 现象：总是选不到 skill
  - 原因：输入缺少触发词或 skill 名称。
  - 处理：加入“创建/更新/安装/GitHub/skill”等关键词。

- 现象：两个 skill 分数相同
  - 处理：当前实现使用名称字典序作为稳定 tie-break，保证结果可复现。

## 可扩展方向

1. 引入权重配置文件（不同 trigger 权重不同）。
2. 接入 embedding 相似度，提升语义匹配能力。
3. 记录历史选择反馈，做在线调优。
