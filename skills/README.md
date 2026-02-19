# skills 模块

这个模块演示在线 skill 路由：

1. 加载本地 skill catalog
2. 把用户输入 + catalog 发送给 LLM
3. LLM 返回严格 JSON 路由结果（匹配、分数、执行计划）
4. 生成最终回答与 trace

## 为什么要这样做

当系统里有多个技能（如 `skill-creator`、`skill-installer`）时，需要统一的路由层。这个模块把“技能定义（catalog）”和“技能选择（LLM routing）”分离，便于扩展与审计。

## 在线前置条件

```env
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

无 key、网络失败、非 JSON 输出都会直接报错。

## 代码实现映射

- `load_skill_catalog()`: 返回技能清单。
- `request_skill_routing(user_text, catalog)`: 在线请求路由 JSON。
- `run_demo(user_text)`: 写入 trace 并返回统一协议。

## trace 事件

- `skill_catalog_loaded`
- `skill_match_scores`
- `skill_selected`（若匹配）
- `skill_execution_plan`（若匹配）
- `model_final_answer`

## 运行

```bash
python skills/component.py
python skills/component.py "请使用 skill-creator 帮我创建技能模板"
```

## 常见错误

- `missing DEEPSEEK_API_KEY`
- `model output is not strict JSON`
- `model selected unknown skill`（模型选了 catalog 外名称）

## 扩展方向

- 为 catalog 增加版本号和灰度发布。
- 对路由结果增加 schema 校验层。
- 追加反馈闭环（用户确认后反哺路由策略）。
