# openai_api 模块

这个模块用于讲清三件事：

1. OpenAI API 生态里“LLM 能做什么”
2. 常见端点“有哪些参数、每个参数有什么作用”
3. 在当前仓库环境下，如何用 DeepSeek（OpenAI 兼容）做在线实调并解读兼容差异

## 模块目标（OpenAI 概念 + DeepSeek 兼容实调）

- 教学目标是 OpenAI API 能力认知，不是绑定某一家供应商。
- 运行层采用 DeepSeek 的 OpenAI 兼容接口（`DEEPSEEK_*` 环境变量）。
- 输出会明确标注每个端点是 `ok` 还是 `unsupported_by_provider` 等状态。

## 功能全景总览

`build_capability_overview()` 覆盖以下能力类别：

- 文本生成（`responses` / `chat.completions`）
- 工具调用（function calling）
- 结构化输出（JSON schema / JSON mode）
- 多模态输入输出（文本/图片/文件）
- 检索向量（`embeddings`）
- 安全审核（`moderations`）
- 批处理与异步（`batches` / `background`）
- 文件与向量库（`files` / `vector_stores`）
- 实时交互（`realtime`）

## 三个重点端点及核心参数说明

### 1) `responses`

- 核心参数：`model`、`input`、`instructions`、`tools`、`tool_choice`
- 采样控制：`temperature`、`top_p`
- 成本控制：`max_output_tokens`
- 结构化输出：`text`
- 推理控制：`reasoning`

### 2) `embeddings`

- 核心参数：`model`、`input`
- 维度控制：`dimensions`（模型支持时）
- 编码格式：`encoding_format`
- 审计字段：`user`

### 3) `moderations`

- 核心参数：`input`、`model`
- 用途：输入风险识别与合规前置检查

## 全量参数索引来源（SDK 2.21.0 TypedDict）

`build_parameter_reference()` 会从 SDK 类型定义自动提取参数名与必填参数：

- `ResponseCreateParamsBase`
- `CompletionCreateParamsBase`（用于对比，不强制在线调用）
- `EmbeddingCreateParams`
- `ModerationCreateParams`

这样可以保证“参数清单”随 SDK 升级同步更新，不靠手写维护。

## 在线运行与常见错误解释

必需环境变量（沿用当前仓库约定）：

```env
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

运行：

```bash
python openai_api/component.py
python -m openai_api.main --print-trace y
```

常见状态解释：

- `ok`: 调用成功并返回可解析结果
- `unsupported_by_provider`: 供应商兼容层未实现该端点/方法
- `auth_error`: 密钥无效或无权限
- `rate_limited`: 触发频率限制
- `provider_server_error`: 供应商服务端异常
- `network_error`: 网络/超时问题
- `unknown_error`: 其他未归类错误

## 供应商兼容差异说明

- 本模块教学语义是 OpenAI API 概念，但在线实调使用 DeepSeek OpenAI 兼容层。
- 兼容层可能不覆盖所有端点或参数，出现 `unsupported_by_provider` 属于预期教学结果。
- 该结果不表示参数概念无效，而是表示“当前 base_url 的实现覆盖范围有限”。
