# openai_api 模块（全功能教学版）

## 1. 模块目标与范围

- 目标：覆盖 `OpenAI` client 的全部资源入口与客户端调用模式。
- 教学形态：每个功能都提供“作用说明 + 核心参数 + 同步示例 + 异步对照 + 2 题练习”。
- 运行方式：在线实调默认 safe mode；高副作用功能仅在显式开关开启后执行。
- 结果契约：每个功能都返回结构化状态（`ok` / `unsupported_by_provider` / `auth_error` 等），不中断总流程。

## 2. 功能总目录（按资源 + client 操作）

- `responses`: `responses.create` | category=`generation` | stability=`stable` | side_effect=`none`
- `chat`: `chat.completions.create` | category=`generation` | stability=`stable` | side_effect=`none`
- `completions`: `completions.create` | category=`generation` | stability=`compat_risk` | side_effect=`none`
- `embeddings`: `embeddings.create` | category=`retrieval` | stability=`stable` | side_effect=`none`
- `moderations`: `moderations.create` | category=`safety` | stability=`stable` | side_effect=`none`
- `files`: `files.list` | category=`data` | stability=`stable` | side_effect=`low`
- `uploads`: `uploads.create` | category=`data` | stability=`preview` | side_effect=`high`
- `batches`: `batches.list` | category=`orchestration` | stability=`stable` | side_effect=`low`
- `models`: `models.list` | category=`management` | stability=`stable` | side_effect=`none`
- `fine_tuning`: `fine_tuning.jobs.list` | category=`training` | stability=`preview` | side_effect=`low`
- `vector_stores`: `vector_stores.list` | category=`retrieval` | stability=`stable` | side_effect=`low`
- `conversations`: `conversations.create` | category=`conversation` | stability=`preview` | side_effect=`high`
- `realtime`: `realtime.client_secrets.create` | category=`realtime` | stability=`preview` | side_effect=`low`
- `webhooks`: `webhooks.verify_signature` | category=`integration` | stability=`stable` | side_effect=`none`
- `evals`: `evals.list` | category=`evaluation` | stability=`preview` | side_effect=`low`
- `containers`: `containers.list` | category=`runtime` | stability=`compat_risk` | side_effect=`low`
- `skills`: `skills.list` | category=`orchestration` | stability=`compat_risk` | side_effect=`low`
- `videos`: `videos.list` | category=`multimodal` | stability=`preview` | side_effect=`low`
- `images`: `images.generate` | category=`multimodal` | stability=`stable` | side_effect=`high`
- `audio`: `audio.transcriptions.create` | category=`multimodal` | stability=`stable` | side_effect=`high`
- `beta`: `beta.assistants.list` | category=`preview` | stability=`preview` | side_effect=`low`
- `client_with_options`: `client.with_options(...).responses.create` | category=`client_ops` | stability=`stable` | side_effect=`none`
- `client_with_raw_response`: `client.with_raw_response.responses.create` | category=`client_ops` | stability=`stable` | side_effect=`none`
- `client_with_streaming_response`: `client.with_streaming_response.responses.create` | category=`client_ops` | stability=`stable` | side_effect=`none`

## 3. 每个功能教学卡片

### Responses（`responses`）

- 功能作用：统一生成接口，支持文本、多模态与工具调用。
- 何时使用：需要统一处理生成请求、工具调用和结构化输出时。
- 兼容风险：stability=`stable`，side_effect_level=`none`
- API surface：`responses.create`
- 全量参数数：`28`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `model`: 模型名称，决定能力、延迟与成本。
  - `input`: 输入内容，可为文本或结构化输入。
  - `instructions`: 系统级指令，约束回答风格和任务边界。
  - `tools`: 函数/工具定义，支持 tool calling。
  - `temperature`: 采样随机度，越高越发散。
  - `max_output_tokens`: 限制输出长度与成本。
- 同步最小示例：
```python
response = client.responses.create(model=model, input=user_text)
```
- 异步对照片段：
```python
response = async_client.responses.create(model=model, input=user_text)
```
- 练习题（理解）：
  - 理解题：解释 `responses` 与 `responses` 在职责上的差异，并给出一个必须使用 `responses` 的场景。
- 练习题（动手）：
  - 动手题：基于 `responses` 写一个最小示例，记录 status、耗时和失败分类。

### Chat Completions（`chat`）

- 功能作用：经典对话生成接口，适合消息式输入。
- 何时使用：已有 chat 消息结构或迁移旧项目时。
- 兼容风险：stability=`stable`，side_effect_level=`none`
- API surface：`chat.completions.create`
- 全量参数数：`34`
- 必填参数：`messages, model`
- 核心参数：
  - `model`: 模型名称。
  - `messages`: 消息数组，chat 接口核心输入。
  - `tools`: 可调用工具定义。
  - `tool_choice`: 工具选择策略。
  - `temperature`: 采样随机度。
  - `max_completion_tokens`: 输出 token 上限。
- 同步最小示例：
```python
resp = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': user_text}])
```
- 异步对照片段：
```python
resp = async_client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': user_text}])
```
- 练习题（理解）：
  - 理解题：解释 `chat` 与 `responses` 在职责上的差异，并给出一个必须使用 `chat` 的场景。
- 练习题（动手）：
  - 动手题：基于 `chat` 写一个最小示例，记录 status、耗时和失败分类。

### Completions（`completions`）

- 功能作用：旧版文本 completion 接口。
- 何时使用：维护历史 prompt-only 接口兼容时。
- 兼容风险：stability=`compat_risk`，side_effect_level=`none`
- API surface：`completions.create`
- 全量参数数：`22`
- 必填参数：`model, prompt`
- 核心参数：
  - `model`: completion 模型名称（多为历史接口）。
  - `prompt`: 纯文本 prompt。
  - `max_tokens`: 输出长度上限。
  - `temperature`: 采样随机度。
  - `top_p`: 核采样概率阈值。
- 同步最小示例：
```python
resp = client.completions.create(model=model, prompt=user_text, max_tokens=64)
```
- 异步对照片段：
```python
resp = async_client.completions.create(model=model, prompt=user_text, max_tokens=64)
```
- 练习题（理解）：
  - 理解题：解释 `completions` 与 `responses` 在职责上的差异，并给出一个必须使用 `completions` 的场景。
- 练习题（动手）：
  - 动手题：基于 `completions` 写一个最小示例，记录 status、耗时和失败分类。

### Embeddings（`embeddings`）

- 功能作用：将文本编码为向量。
- 何时使用：RAG 召回、语义搜索和相似度计算场景。
- 兼容风险：stability=`stable`，side_effect_level=`none`
- API surface：`embeddings.create`
- 全量参数数：`5`
- 必填参数：`input, model`
- 核心参数：
  - `model`: 向量模型名称。
  - `input`: 待向量化文本。
  - `dimensions`: 向量维度（模型支持时可配置）。
  - `encoding_format`: 向量编码格式（float/base64）。
- 同步最小示例：
```python
vec = client.embeddings.create(model=embed_model, input=[user_text])
```
- 异步对照片段：
```python
vec = async_client.embeddings.create(model=embed_model, input=[user_text])
```
- 练习题（理解）：
  - 理解题：解释 `embeddings` 与 `responses` 在职责上的差异，并给出一个必须使用 `embeddings` 的场景。
- 练习题（动手）：
  - 动手题：基于 `embeddings` 写一个最小示例，记录 status、耗时和失败分类。

### Moderations（`moderations`）

- 功能作用：对输入内容进行风险审核。
- 何时使用：UGC 审核、合规前置检查。
- 兼容风险：stability=`stable`，side_effect_level=`none`
- API surface：`moderations.create`
- 全量参数数：`2`
- 必填参数：`input`
- 核心参数：
  - `model`: 审核模型名称。
  - `input`: 待审核文本或多模态内容。
- 同步最小示例：
```python
moderation = client.moderations.create(model=mod_model, input=user_text)
```
- 异步对照片段：
```python
moderation = async_client.moderations.create(model=mod_model, input=user_text)
```
- 练习题（理解）：
  - 理解题：解释 `moderations` 与 `responses` 在职责上的差异，并给出一个必须使用 `moderations` 的场景。
- 练习题（动手）：
  - 动手题：基于 `moderations` 写一个最小示例，记录 status、耗时和失败分类。

### Files（`files`）

- 功能作用：文件上传与管理。
- 何时使用：需要托管知识文件、训练数据或中间文件时。
- 兼容风险：stability=`stable`，side_effect_level=`low`
- API surface：`files.list`
- 全量参数数：`8`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `purpose`: 文件用途标识。
  - `file`: 上传文件内容。
  - `limit`: 列表分页大小。
- 同步最小示例：
```python
page = client.files.list(limit=1)
```
- 异步对照片段：
```python
page = async_client.files.list(limit=1)
```
- 练习题（理解）：
  - 理解题：解释 `files` 与 `responses` 在职责上的差异，并给出一个必须使用 `files` 的场景。
- 练习题（动手）：
  - 动手题：基于 `files` 写一个最小示例，记录 status、耗时和失败分类。

### Uploads（`uploads`）

- 功能作用：分片上传能力，用于大文件。
- 何时使用：需要对大文件做断点/分片上传时。
- 兼容风险：stability=`preview`，side_effect_level=`high`
- API surface：`uploads.create`
- 全量参数数：`9`
- 必填参数：`bytes, filename, mime_type, purpose`
- 核心参数：
  - `bytes`: 总字节数。
  - `filename`: 上传文件名。
  - `mime_type`: 文件类型。
  - `purpose`: 用途标识。
- 同步最小示例：
```python
upload = client.uploads.create(bytes=1024, filename='a.txt', mime_type='text/plain', purpose='assistants')
```
- 异步对照片段：
```python
upload = async_client.uploads.create(bytes=1024, filename='a.txt', mime_type='text/plain', purpose='assistants')
```
- 练习题（理解）：
  - 理解题：解释 `uploads` 与 `responses` 在职责上的差异，并给出一个必须使用 `uploads` 的场景。
- 练习题（动手）：
  - 动手题：基于 `uploads` 写一个最小示例，记录 status、耗时和失败分类。

### Batches（`batches`）

- 功能作用：批量异步任务管理。
- 何时使用：离线高吞吐任务调度时。
- 兼容风险：stability=`stable`，side_effect_level=`low`
- API surface：`batches.list`
- 全量参数数：`6`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `input_file_id`: 批处理输入文件 ID。
  - `endpoint`: 批处理目标端点。
  - `completion_window`: 批处理时间窗口。
- 同步最小示例：
```python
page = client.batches.list(limit=1)
```
- 异步对照片段：
```python
page = async_client.batches.list(limit=1)
```
- 练习题（理解）：
  - 理解题：解释 `batches` 与 `responses` 在职责上的差异，并给出一个必须使用 `batches` 的场景。
- 练习题（动手）：
  - 动手题：基于 `batches` 写一个最小示例，记录 status、耗时和失败分类。

### Models（`models`）

- 功能作用：模型元信息查询。
- 何时使用：动态能力探测、模型白名单和版本检查。
- 兼容风险：stability=`stable`，side_effect_level=`none`
- API surface：`models.list`
- 全量参数数：`4`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `model`: 模型 ID。
- 同步最小示例：
```python
models = client.models.list()
```
- 异步对照片段：
```python
models = async_client.models.list()
```
- 练习题（理解）：
  - 理解题：解释 `models` 与 `responses` 在职责上的差异，并给出一个必须使用 `models` 的场景。
- 练习题（动手）：
  - 动手题：基于 `models` 写一个最小示例，记录 status、耗时和失败分类。

### Fine Tuning（`fine_tuning`）

- 功能作用：微调任务生命周期管理。
- 何时使用：需要领域定制模型训练与追踪。
- 兼容风险：stability=`preview`，side_effect_level=`low`
- API surface：`fine_tuning.jobs.list`
- 全量参数数：`7`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `training_file`: 训练文件 ID。
  - `model`: 基座模型。
  - `hyperparameters`: 训练超参数。
- 同步最小示例：
```python
jobs = client.fine_tuning.jobs.list(limit=1)
```
- 异步对照片段：
```python
jobs = async_client.fine_tuning.jobs.list(limit=1)
```
- 练习题（理解）：
  - 理解题：解释 `fine_tuning` 与 `responses` 在职责上的差异，并给出一个必须使用 `fine_tuning` 的场景。
- 练习题（动手）：
  - 动手题：基于 `fine_tuning` 写一个最小示例，记录 status、耗时和失败分类。

### Vector Stores（`vector_stores`）

- 功能作用：向量库管理和检索。
- 何时使用：文档检索增强、知识库问答。
- 兼容风险：stability=`stable`，side_effect_level=`low`
- API surface：`vector_stores.list`
- 全量参数数：`8`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `name`: 向量库名称。
  - `file_ids`: 关联文件 ID 列表。
  - `limit`: 分页大小。
- 同步最小示例：
```python
stores = client.vector_stores.list(limit=1)
```
- 异步对照片段：
```python
stores = async_client.vector_stores.list(limit=1)
```
- 练习题（理解）：
  - 理解题：解释 `vector_stores` 与 `responses` 在职责上的差异，并给出一个必须使用 `vector_stores` 的场景。
- 练习题（动手）：
  - 动手题：基于 `vector_stores` 写一个最小示例，记录 status、耗时和失败分类。

### Conversations（`conversations`）

- 功能作用：会话容器管理。
- 何时使用：需要跨请求保留会话上下文时。
- 兼容风险：stability=`preview`，side_effect_level=`high`
- API surface：`conversations.create`
- 全量参数数：`6`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `items`: 初始会话输入项。
  - `metadata`: 会话元数据。
- 同步最小示例：
```python
conversation = client.conversations.create(items=[...])
```
- 异步对照片段：
```python
conversation = async_client.conversations.create(items=[...])
```
- 练习题（理解）：
  - 理解题：解释 `conversations` 与 `responses` 在职责上的差异，并给出一个必须使用 `conversations` 的场景。
- 练习题（动手）：
  - 动手题：基于 `conversations` 写一个最小示例，记录 status、耗时和失败分类。

### Realtime（`realtime`）

- 功能作用：实时会话和低延迟交互。
- 何时使用：语音助手、实时协作与 streaming 交互。
- 兼容风险：stability=`preview`，side_effect_level=`low`
- API surface：`realtime.client_secrets.create`
- 全量参数数：`6`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `session`: 实时会话配置。
  - `expires_after`: 临时令牌有效期。
- 同步最小示例：
```python
secret = client.realtime.client_secrets.create(session={'type': 'realtime', 'model': model})
```
- 异步对照片段：
```python
secret = async_client.realtime.client_secrets.create(session={'type': 'realtime', 'model': model})
```
- 练习题（理解）：
  - 理解题：解释 `realtime` 与 `responses` 在职责上的差异，并给出一个必须使用 `realtime` 的场景。
- 练习题（动手）：
  - 动手题：基于 `realtime` 写一个最小示例，记录 status、耗时和失败分类。

### Webhooks（`webhooks`）

- 功能作用：Webhook 验签与回调事件校验。
- 何时使用：服务端接收异步回调并做安全校验。
- 兼容风险：stability=`stable`，side_effect_level=`none`
- API surface：`webhooks.verify_signature`
- 全量参数数：`4`
- 必填参数：`payload, headers`
- 核心参数：
  - `payload`: 原始回调 payload。
  - `headers`: 回调请求头。
  - `secret`: Webhook 签名密钥。
- 同步最小示例：
```python
client.webhooks.verify_signature(payload, headers, secret=webhook_secret)
```
- 异步对照片段：
```python
async_client.webhooks.verify_signature(payload, headers, secret=webhook_secret)
```
- 练习题（理解）：
  - 理解题：解释 `webhooks` 与 `responses` 在职责上的差异，并给出一个必须使用 `webhooks` 的场景。
- 练习题（动手）：
  - 动手题：基于 `webhooks` 写一个最小示例，记录 status、耗时和失败分类。

### Evals（`evals`）

- 功能作用：模型评测任务管理。
- 何时使用：构建自动化评测基线与质量回归。
- 兼容风险：stability=`preview`，side_effect_level=`low`
- API surface：`evals.list`
- 全量参数数：`8`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `name`: 评测名称。
  - `data_source_config`: 评测数据源配置。
  - `testing_criteria`: 评测判定标准。
- 同步最小示例：
```python
evals = client.evals.list(limit=1)
```
- 异步对照片段：
```python
evals = async_client.evals.list(limit=1)
```
- 练习题（理解）：
  - 理解题：解释 `evals` 与 `responses` 在职责上的差异，并给出一个必须使用 `evals` 的场景。
- 练习题（动手）：
  - 动手题：基于 `evals` 写一个最小示例，记录 status、耗时和失败分类。

### Containers（`containers`）

- 功能作用：容器化运行时资源管理。
- 何时使用：需要隔离运行环境或沙箱资源时。
- 兼容风险：stability=`compat_risk`，side_effect_level=`low`
- API surface：`containers.list`
- 全量参数数：`8`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `name`: 容器名称。
  - `limit`: 分页大小。
- 同步最小示例：
```python
containers = client.containers.list(limit=1)
```
- 异步对照片段：
```python
containers = async_client.containers.list(limit=1)
```
- 练习题（理解）：
  - 理解题：解释 `containers` 与 `responses` 在职责上的差异，并给出一个必须使用 `containers` 的场景。
- 练习题（动手）：
  - 动手题：基于 `containers` 写一个最小示例，记录 status、耗时和失败分类。

### Skills（`skills`）

- 功能作用：技能编排与版本管理。
- 何时使用：需要平台化路由能力与技能版本治理。
- 兼容风险：stability=`compat_risk`，side_effect_level=`low`
- API surface：`skills.list`
- 全量参数数：`7`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `name`: 技能名称。
  - `description`: 技能说明。
  - `tools`: 技能工具定义。
- 同步最小示例：
```python
skills = client.skills.list(limit=1)
```
- 异步对照片段：
```python
skills = async_client.skills.list(limit=1)
```
- 练习题（理解）：
  - 理解题：解释 `skills` 与 `responses` 在职责上的差异，并给出一个必须使用 `skills` 的场景。
- 练习题（动手）：
  - 动手题：基于 `skills` 写一个最小示例，记录 status、耗时和失败分类。

### Videos（`videos`）

- 功能作用：视频生成任务与结果管理。
- 何时使用：视频生成、重混与轮询下载场景。
- 兼容风险：stability=`preview`，side_effect_level=`low`
- API surface：`videos.list`
- 全量参数数：`7`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `prompt`: 视频生成提示词。
  - `model`: 视频模型名称。
  - `seconds`: 视频时长。
- 同步最小示例：
```python
videos = client.videos.list(limit=1)
```
- 异步对照片段：
```python
videos = async_client.videos.list(limit=1)
```
- 练习题（理解）：
  - 理解题：解释 `videos` 与 `responses` 在职责上的差异，并给出一个必须使用 `videos` 的场景。
- 练习题（动手）：
  - 动手题：基于 `videos` 写一个最小示例，记录 status、耗时和失败分类。

### Images（`images`）

- 功能作用：图像生成与编辑。
- 何时使用：营销素材、创意草图、视觉内容生成。
- 兼容风险：stability=`stable`，side_effect_level=`high`
- API surface：`images.generate`
- 全量参数数：`18`
- 必填参数：`prompt`
- 核心参数：
  - `prompt`: 图像生成提示词。
  - `model`: 图像模型名称。
  - `size`: 图像尺寸。
  - `quality`: 图像质量档位。
- 同步最小示例：
```python
image = client.images.generate(model=image_model, prompt='a cat')
```
- 异步对照片段：
```python
image = async_client.images.generate(model=image_model, prompt='a cat')
```
- 练习题（理解）：
  - 理解题：解释 `images` 与 `responses` 在职责上的差异，并给出一个必须使用 `images` 的场景。
- 练习题（动手）：
  - 动手题：基于 `images` 写一个最小示例，记录 status、耗时和失败分类。

### Audio（`audio`）

- 功能作用：音频转写、翻译与语音合成。
- 何时使用：语音转文本、字幕生成与语音交互。
- 兼容风险：stability=`stable`，side_effect_level=`high`
- API surface：`audio.transcriptions.create`
- 全量参数数：`16`
- 必填参数：`file, model`
- 核心参数：
  - `file`: 音频文件输入。
  - `model`: 音频模型名称。
  - `language`: 语种提示。
  - `response_format`: 返回格式。
- 同步最小示例：
```python
transcript = client.audio.transcriptions.create(model=audio_model, file=audio_file)
```
- 异步对照片段：
```python
transcript = async_client.audio.transcriptions.create(model=audio_model, file=audio_file)
```
- 练习题（理解）：
  - 理解题：解释 `audio` 与 `responses` 在职责上的差异，并给出一个必须使用 `audio` 的场景。
- 练习题（动手）：
  - 动手题：基于 `audio` 写一个最小示例，记录 status、耗时和失败分类。

### Beta（`beta`）

- 功能作用：实验性 API 集合。
- 何时使用：提前验证新能力并评估升级影响。
- 兼容风险：stability=`preview`，side_effect_level=`low`
- API surface：`beta.assistants.list`
- 全量参数数：`8`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `limit`: 分页大小。
- 同步最小示例：
```python
assistants = client.beta.assistants.list(limit=1)
```
- 异步对照片段：
```python
assistants = async_client.beta.assistants.list(limit=1)
```
- 练习题（理解）：
  - 理解题：解释 `beta` 与 `responses` 在职责上的差异，并给出一个必须使用 `beta` 的场景。
- 练习题（动手）：
  - 动手题：基于 `beta` 写一个最小示例，记录 status、耗时和失败分类。

### Client with_options/copy（`client_with_options`）

- 功能作用：基于同一 client 快速覆盖 timeout/retry/header 等选项。
- 何时使用：按请求级别调整客户端配置时。
- 兼容风险：stability=`stable`，side_effect_level=`none`
- API surface：`client.with_options(...).responses.create`
- 全量参数数：`33`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `timeout`: 临时覆盖请求超时。
  - `max_retries`: 临时覆盖重试次数。
- 同步最小示例：
```python
resp = client.with_options(timeout=20.0).responses.create(model=model, input=user_text)
```
- 异步对照片段：
```python
resp = async_client.with_options(timeout=20.0).responses.create(model=model, input=user_text)
```
- 练习题（理解）：
  - 理解题：解释 `client_with_options` 与 `responses` 在职责上的差异，并给出一个必须使用 `client_with_options` 的场景。
- 练习题（动手）：
  - 动手题：基于 `client_with_options` 写一个最小示例，记录 status、耗时和失败分类。

### Client with_raw_response（`client_with_raw_response`）

- 功能作用：直接获取原始 HTTP 响应对象。
- 何时使用：需要调试状态码、headers、原始 body 时。
- 兼容风险：stability=`stable`，side_effect_level=`none`
- API surface：`client.with_raw_response.responses.create`
- 全量参数数：`33`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `status_code`: 原始 HTTP 状态码。
  - `headers`: 原始响应头。
  - `parse()`: 将原始响应解析为 SDK 对象。
- 同步最小示例：
```python
raw = client.with_raw_response.responses.create(model=model, input=user_text)
```
- 异步对照片段：
```python
raw = async_client.with_raw_response.responses.create(model=model, input=user_text)
```
- 练习题（理解）：
  - 理解题：解释 `client_with_raw_response` 与 `responses` 在职责上的差异，并给出一个必须使用 `client_with_raw_response` 的场景。
- 练习题（动手）：
  - 动手题：基于 `client_with_raw_response` 写一个最小示例，记录 status、耗时和失败分类。

### Client with_streaming_response（`client_with_streaming_response`）

- 功能作用：流式读取响应并按需解析。
- 何时使用：低延迟输出、逐段消费与长响应场景。
- 兼容风险：stability=`stable`，side_effect_level=`none`
- API surface：`client.with_streaming_response.responses.create`
- 全量参数数：`33`
- 必填参数：`无显式必填`（通常由默认值或服务器侧策略决定）
- 核心参数：
  - `stream`: 流式读取响应体。
  - `parse()`: 流结束后解析完整对象。
- 同步最小示例：
```python
with client.with_streaming_response.responses.create(model=model, input=user_text) as stream: ...
```
- 异步对照片段：
```python
with async_client.with_streaming_response.responses.create(model=model, input=user_text) as stream: ...
```
- 练习题（理解）：
  - 理解题：解释 `client_with_streaming_response` 与 `responses` 在职责上的差异，并给出一个必须使用 `client_with_streaming_response` 的场景。
- 练习题（动手）：
  - 动手题：基于 `client_with_streaming_response` 写一个最小示例，记录 status、耗时和失败分类。

## 4. 实调策略（safe mode vs side-effect mode）

- 默认策略：safe mode，只执行低副作用调用（如 `list/retrieve` 或轻量 `create`）。
- side-effect mode：通过 `--include-side-effects y` 或 `OPENAI_API_INCLUDE_SIDE_EFFECT_CALLS=1` 开启。
- 高副作用功能示例：`uploads`、`conversations`、`images`、`audio`。
- 若供应商不支持，结果会标记为 `unsupported_by_provider`，不视为流程失败。

## 5. 常见错误与状态解释

- `ok`: 调用成功并返回可解析结果。
- `unsupported_by_provider`: 兼容层未实现该功能或端点。
- `auth_error`: key 无效或权限不足（401/403）。
- `rate_limited`: 触发频控（429）。
- `provider_server_error`: 供应商服务端错误（5xx）。
- `network_error`: 网络连接或超时问题。
- `skipped_by_policy`: 因高副作用策略被跳过。
- `unknown_error`: 未归类错误（含本地参数错误）。

## 6. 命令与验证

必需环境变量：

```env
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
OPENAI_API_INCLUDE_SIDE_EFFECT_CALLS=0
OPENAI_API_PROBE_TIMEOUT_SECONDS=30
OPENAI_API_PROBE_LIMIT=1
```

运行（全功能）：

```bash
python -m openai_api.main --scope all --print-trace y
python -m openai_api.main --scope all --include-side-effects y --print-trace y
```

运行（兼容旧三端点）：

```bash
python -m openai_api.main --scope core --print-trace y
```

检查：

```bash
python -m pytest -q -k openai_api
npx -y pyright openai_api/main.py openai_api/component.py
uvx ruff check openai_api
```
