# RAG（检索增强生成）技术深度研究报告

## 执行摘要

检索增强生成（RAG, Retrieval‑Augmented Generation）是一类将“外部检索（Retrieval）”与“大模型生成（Generation）”结合的体系：模型在回答问题或生成内容前，先从显式知识库（文档、网页、数据库等）检索相关证据，再将证据作为上下文输入生成器，从而提升**可更新性、可追溯性与事实一致性**，并在很多知识密集型任务上优于仅靠参数记忆的生成模型。citeturn0search0turn2search2turn3search0turn3search4

从原始论文视角，RAG通过可微检索器访问“非参数化记忆”（通常是向量索引），并将检索到的文档视为潜变量进行边缘化推断；经典实现区分 RAG‑Sequence 与 RAG‑Token 两种边缘化方式。citeturn0search0turn0search4

工程上，RAG落地的关键不在“是否接入向量库”本身，而在于：数据切分与元数据设计、向量化模型选择、索引结构与检索策略（稠密/稀疏/混合）、多路召回与融合（如RRF）、生成阶段的上下文组织与提示工程、以及可度量的离线/在线评估闭环。citeturn3search0turn9search9turn3search3turn3search1turn3search2

成本与性能方面，RAG会引入检索与存储开销：向量维度越高，存储与带宽越大；ANN索引（HNSW/IVF/PQ/DiskANN等）会在召回率、延迟、构建时间与内存之间权衡；混合检索与重排可提升质量但可能提高端到端延迟。citeturn4search4turn11search21turn11search3turn1search4turn11search0

## 概念、原理与体系结构

### 定义与“参数化记忆 + 非参数化记忆”的视角

RAG可理解为：生成模型（参数化记忆）在推理时通过检索器访问外部语料（非参数化记忆），把“与问题相关的证据”注入上下文，从而减少纯生成的无依据扩写，并支持通过更新索引来更新知识，而不必每次都重新训练大模型。citeturn0search4turn2search2turn3search4

entity["people","Patrick Lewis","rag paper author"]等人在原始RAG论文中明确提出：检索到的文档可作为潜变量，模型对多个可能文档进行边缘化以产生输出分布；这一框架同时兼容“检索‑阅读‑生成”的端到端微调。citeturn0search0turn0search4

### 两种经典RAG边缘化：RAG‑Sequence 与 RAG‑Token

原始论文给出两种形式：  
- **RAG‑Sequence**：整段生成序列共享同一检索文档集合（更像“先定证据，再写答案”）。citeturn0search0  
- **RAG‑Token**：不同token在生成时可以基于不同检索文档（更细粒度、更灵活，但推理复杂度更高）。citeturn0search0  

这两者体现了RAG的一个核心设计维度：**证据与生成的耦合粒度**（sequence级 vs token级），在质量、可解释性与计算代价上会呈现不同取舍。citeturn0search0turn3search0

### 与FiD（Fusion‑in‑Decoder）的结构差异

entity["people","Gautier Izacard","fid paper author"]与合作者提出的 Fusion‑in‑Decoder（FiD）是一类“检索‑生成”结构：对每条检索段落分别编码（encoder对每段独立处理），再在解码器侧通过跨注意力把多段表示融合生成答案；论文指出这种结构在处理多段证据时，计算随段落数呈线性增长（相对一些联合编码方式的更差扩展性），同时有利于汇聚多段证据。citeturn0search14

在工程直觉上：**RAG更强调“检索作为潜变量的概率建模”**，而 FiD 更强调“多段证据的结构化融合（fusion）”。两者常被视为现代RAG系统中“生成器侧融合策略”的代表范式之一。citeturn0search0turn0search14turn3search4

## 核心组件与检索‑生成融合

### 检索器（Retriever）

RAG的检索器通常可分为稀疏检索、稠密检索与混合检索三类：

稀疏检索以 BM25 为典型，属于概率相关框架（PRF）的经典家族，利用词频、逆文档频率与长度归一化计算相关性，长期作为工业检索基线并在诸多系统中默认启用。citeturn1search6turn6search2turn9search29turn2search13

稠密检索以双塔（bi‑encoder/dual‑encoder）为典型：把查询与段落编码到同一向量空间，以向量相似度检索。Dense Passage Retrieval（DPR）展示了仅用稠密向量也能在开放域QA的top‑k段落召回上显著超越Lucene‑BM25基线，并推动“稠密向量检索 + 生成/阅读器”的端到端方案成为主流。citeturn0search1turn0search26turn0search30

entity["people","Vladimir Karpukhin","dpr paper author"]等人的DPR论文还明确强调：检索器可以用“top‑k检索准确率”等指标独立评估（例如top‑20是否包含答案证据），这也奠定了RAG工程评估“分组件度量”的方法论基础。citeturn0search26

混合检索（Hybrid Search）将稀疏（BM25）与稠密（向量）并行召回，再进行融合或重排，以同时覆盖“关键词精确匹配”和“语义近邻”。citeturn9search9turn12search1turn8search4

### 索引（Index）与向量化（Embedding）

向量化模型将文本/代码等转为稠密向量（embedding），用于相似度计算。以 OpenAI 的 embedding 文档为例：向量用于搜索、聚类、推荐等；`text-embedding-3-small` 默认维度为1536，`text-embedding-3-large` 默认3072，并支持通过 `dimensions` 参数降低输出维度以控制存储与计算开销。citeturn1search11turn1search3turn1search7turn7search0

索引则是为加速kNN/ANN搜索而构建的数据结构。FAISS作为经典向量相似度库，提供从精确暴力检索到IVF、HNSW、PQ等多种索引结构，支持CPU与GPU实现，并包含评估与参数调优工具。citeturn0search3turn5search2turn11search0turn0search7

Milvus这类向量数据库进一步把索引、持久化、分布式扩展与多租户治理做成服务化形态，并支持多种索引类型（如IVF、HNSW、PQ、DiskANN等）及稀疏/稠密向量。citeturn4search4turn1search0turn11search3turn11search1

### 检索策略：Top‑k、过滤、多段召回与多样性重排

常见策略包括：

- Top‑k：最简单的召回方式，直接取相似度最高的k段。citeturn2search4turn0search26  
- 过滤（metadata filtering）：在向量检索时结合结构化条件（时间、权限、业务标签等），在企业知识库/客服场景尤其关键。citeturn2search4turn10search0turn12search0  
- 多样性重排（MMR）：在“相关性”与“新颖性/去冗余”之间折中，以减少检索结果同质化。MMR由 Carbonell 等提出，用线性组合度量相关与冗余，常用于检索与摘要场景的多样性排序。citeturn8search3  
- 多路召回（multi‑retriever / multi‑vector）：用不同向量字段、不同embedding模型或“稀疏+稠密”并行召回，再融合。Milvus在多向量混合检索中强调可同时对多个向量字段进行ANN，并通过重排器整合结果。citeturn8search11turn8search1turn8search7  

### 融合方法：从得分融合到RRF、再到生成器侧融合

融合主要发生在两处：**检索融合**与**生成融合**。

检索融合的代表是 Reciprocal Rank Fusion（RRF）。RRF由 Cormack 等提出，核心思想是只基于排名位置进行融合，不要求不同检索器的分数可比，并在实践中表现稳健。citeturn3search3turn9search4

在工业实现中，entity["company","Elastic","search company"]的官方文档明确推荐用RRF做混合检索融合，并提供RRF的REST API与公式说明。citeturn9search9turn9search4

Milvus也提供RRF Ranker用于混合检索，强调其基于排序位置平衡多路召回、避免手工权重调参。citeturn8search0turn8search7

生成器侧融合的代表包括：  
- RAG‑Token/RAG‑Sequence 的“对文档潜变量边缘化”机制。citeturn0search0  
- FiD 的“编码器独立编码每段证据，解码器侧跨注意力融合”。citeturn0search14  

## 实现范式与工具生态

### 稀疏检索 vs 稠密检索 vs 混合检索

稀疏检索（BM25）在关键词精确匹配、罕见实体/产品型号、强结构化语料上往往更稳健；稠密检索（向量）在同义改写、语义相近表达上更强；混合检索试图把两者优势叠加，避免“纯向量错过关键字、纯BM25错过语义”这类互补失误。citeturn8search2turn9search29turn12search1turn9search9

### 主流向量检索/混合检索工具对比

下表以“实现形态 + 检索能力 + 典型取舍”为主线，比较主流开源组件（含用户点名的FAISS/Milvus/Elasticsearch/BM25相关栈），用于选型时快速定位。

| 组件/系统 | 形态 | 主要检索能力 | 典型优势 | 典型局限 |
|---|---|---|---|---|
| FAISS | 本地库（C++/Python），可CPU/GPU | 稠密向量相似度搜索；多种索引（Flat/IVF/HNSW/PQ等） | 高性能、算法族丰富、易嵌入本地应用与批处理；支持GPU加速 | 更像“索引库”而非“数据库”：分布式、权限、事务、运维能力需自行补齐 |
| Milvus | 向量数据库（可单机/分布式） | 稠密/稀疏向量；多索引（IVF/HNSW/PQ/DiskANN等）；支持混合检索与重排 | 服务化、可扩展；支持多向量/混合检索与RRF等重排；具备RBAC等治理能力 | 运维与容量规划更复杂；索引与参数需要面向场景调优 |
| Elasticsearch | 搜索引擎（分布式） | 倒排（BM25）+ 向量kNN；支持混合检索与RRF | 全文检索生态成熟，过滤/聚合强；向量检索与混合检索逐步完善；支持文档/字段级安全 | 向量能力与参数体系受版本、索引结构与存储开销影响；需关注dense_vector存储策略 |
| OpenSearch | 搜索引擎（分布式） | k‑NN向量检索（含多种方法/引擎）；与全文检索结合 | 开源社区活跃；k‑NN能力系统化并支持不同获取kNN的方法 | 与不同发行版/插件版本耦合，能力随版本演进需验证 |
| Weaviate | 向量数据库 | 向量检索 + 关键词（BM25/BM25F）混合检索（并行+融合） | 内置混合检索与融合；面向AI应用的API抽象较完整 | 具体融合/归一化策略需要理解与调参 |
| Qdrant | 向量数据库 | 向量检索 + 过滤（payload索引） | 对过滤与payload索引有较系统的工程化建议与机制 | 选型仍需结合部署形态、混合检索策略与生态 |
| Chroma | 面向AI应用的开源检索/向量存储 | 向量检索；强调开发者易用与本地运行 | 上手快、适合原型与中小规模应用 | 大规模分布式、治理与高并发场景需评估其云形态或替换后端 |

与表中结论对应的权威依据包括：FAISS官方文档与索引wiki对其“稠密向量相似度与多索引家族”的定位说明；Milvus官方索引文档对其支持稠密/稀疏向量与多索引的描述；Elasticsearch官方kNN/混合检索与RRF文档；OpenSearch对kNN向量检索的技术说明；Weaviate对BM25+向量混合检索的说明；Qdrant对过滤与payload索引机制的说明；Chroma对其“面向AI的开源检索引擎/本地运行”的定位描述。citeturn0search3turn11search0turn4search4turn1search0turn2search4turn9search9turn9search4turn12search3turn12search7turn12search1turn12search4turn12search2

### 端到端框架生态：组件化编排与评估工具

工程落地通常会用“框架把组件连起来”，典型包括：  
- LlamaIndex强调向量存储与索引在RAG中的核心地位，并提供VectorStoreIndex等抽象，覆盖数据摄取到查询的常见链路。citeturn6search0turn6search4  
- Haystack明确将Retriever用于RAG检索阶段，并提供“从零搭建RAG QA pipeline”的教程，体现其端到端管线化理念。citeturn9search2turn9search22turn9search14  

评估方面，RAGAS提供面向RAG管线的组件级指标（如faithfulness、context recall/precision等）；TruLens提出RAG triad（context relevance、groundedness、answer relevance）以度量检索‑生成链路的“相关性与扎根性”。citeturn3search1turn3search2turn3search5

## 端到端流程、工程实践与示意图

### 端到端工作流程：离线“建库”与在线“问答”两条链

RAG系统通常分为两条主链：

离线链（Indexing / Ingestion）：收集文档 → 清洗去噪 → 切分成chunk → 生成embedding → 写入向量库/搜索引擎 → 建索引。citeturn6search4turn6search0turn9search14

在线链（Querying）：用户问题 →（可选）查询改写/扩展 → 计算查询向量/关键词查询 → 多路召回 → 融合/重排 → 组织上下文与提示 → 大模型生成 →（可选）引用证据/返回溯源。citeturn3search4turn9search9turn8search0turn2search3

### Mermaid：架构与数据流示意

```mermaid
flowchart TB
  subgraph Offline[离线：数据准备与索引构建]
    A[原始数据\nPDF/HTML/Markdown/代码/工单] --> B[清洗与解析\n去重/去噪/结构化元数据]
    B --> C[切分Chunk\n段落/滑窗/层级结构]
    C --> D[向量化Embedding\n稠密/稀疏]
    D --> E[(索引/库)\n向量库 or 搜索引擎\nFAISS/Milvus/Elasticsearch]
    B --> F[倒排索引(可选)\nBM25]
    F --> E
  end

  subgraph Online[在线：检索增强生成]
    Q[用户查询] --> Q1[查询处理(可选)\n改写/扩展/结构化过滤]
    Q1 --> R1[稠密检索\nkNN/ANN]
    Q1 --> R2[稀疏检索\nBM25]
    R1 --> M[融合/重排\nRRF/加权融合/MMR/交叉编码器]
    R2 --> M
    M --> P[上下文构造\n证据拼接/去重/压缩]
    P --> G[生成器/大模型\n回答+引用证据]
    G --> O[输出\n答案/引用/置信度/日志]
  end

  E --> R1
  E --> R2
```

### 工程实践步骤清单（表格）

下表给出“从0到可上线”的实践步骤，强调可验证产物与常见坑位（默认无特定约束，按通用企业RAG场景编排）。

| 阶段 | 关键产物 | 关键决策点 | 常见坑与调优抓手 |
|---|---|---|---|
| 数据准备 | 统一语料与元数据规范（来源、时间、权限、版本、文档层级） | 是否需要权限隔离/多租户；是否保留行级/段级定位 | 元数据缺失导致无法过滤与溯源；版本漂移导致“旧答案” |
| 清洗与切分 | chunk集合（含定位信息） | chunk大小、重叠策略、按标题/段落/代码块分割 | chunk太大导致上下文浪费；太小导致语义断裂；可用“父子chunk”或自动合并策略缓解（例如返回父文档/段落）citeturn9search30 |
| 向量化模型选择 | embedding模型与维度方案 | 语种与领域适配；向量维度与成本；是否可降维 | 维度过高导致存储/带宽拉升；可用`dimensions`降维（若模型支持）citeturn1search11 |
| 索引构建 | 向量索引（HNSW/IVF/PQ/DiskANN等） | 数据量级、延迟目标、内存预算、是否上SSD | IVF通常更省内存、构建更快但需调参；HNSW召回好但更吃内存；超内存规模可用DiskANN等磁盘索引citeturn11search5turn11search3turn11search1 |
| 检索器配置 | top‑k、过滤、并发与超时策略 | 稀疏/稠密/混合；是否需要多路召回 | 仅向量易漏关键词；仅BM25易漏语义；建议混合并用RRF融合citeturn9search9turn9search4 |
| 融合与重排 | 融合器（RRF/加权/线性归一化等），可选重排器 | 是否需要交叉编码器重排；如何限时 | RRF不依赖分数尺度且少调参；加权/线性融合需归一与权重调试citeturn3search3turn9search4turn9search9 |
| 提示工程与生成 | Prompt模板、引用格式、拒答策略 | 上下文拼接方式；是否强制引用证据 | 提示注入与“检索片段伪指令”风险；需隔离指令与数据、做内容过滤citeturn2search24turn2search3 |
| 评估与回归测试 | 指标与测试用例集、离线评测脚本 | 评估粒度：检索、上下文、生成 | 可用RAGAS/TruLens度量faithfulness、context precision/recall、groundedness等，并建立回归集citeturn3search1turn3search2 |

### 可直接运行的简短Python示例：FAISS + OpenAI Embeddings + 生成回答

说明：示例展示最小可用RAG骨架（本地FAISS索引 + OpenAI embeddings + Responses/Chat生成）。OpenAI官方文档给出embeddings维度、`dimensions`参数与API调用方式；Responses/Chat接口可用于生成答案。citeturn1search11turn7search3turn7search5turn7search15turn10search2

```python
# pip install -U openai faiss-cpu numpy

import os
import numpy as np
import faiss
from openai import OpenAI

# 1) 准备一个小型语料库（示例：真实项目中应来自文档切分后的chunks）
docs = [
    "RAG 将外部检索到的文档作为上下文注入生成过程，以提升事实性与可更新性。",
    "BM25 是经典稀疏检索方法，基于词频、逆文档频率与长度归一化进行排序。",
    "FAISS 是高效向量相似度搜索库，可用IndexFlatIP做内积检索（配合归一化可近似余弦相似度）。",
]

# 2) 使用 OpenAI 生成 embeddings
client = OpenAI()  # 需要环境变量 OPENAI_API_KEY
EMBED_MODEL = "text-embedding-3-small"

def embed_texts(texts):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        # 可选：降维以减少存储/加速（需与索引维度一致）
        # dimensions=768,
    )
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")

doc_vecs = embed_texts(docs)

# 3) 构建 FAISS 索引（用内积检索；先做L2归一化=>内积≈余弦相似度）
faiss.normalize_L2(doc_vecs)
dim = doc_vecs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(doc_vecs)

# 4) 查询：检索Top-k文档
query = "RAG 的核心思想是什么？"
q_vec = embed_texts([query])
faiss.normalize_L2(q_vec)

k = 2
scores, ids = index.search(q_vec, k)
retrieved = [(int(i), float(s), docs[int(i)]) for i, s in zip(ids[0], scores[0])]

context = "\n\n".join([f"[Doc {i} | score={s:.3f}]\n{t}" for i, s, t in retrieved])

# 5) 生成回答（Responses API；也可换成 chat.completions.create）
prompt = f"""你是一个严谨的中文技术助手。
请仅基于给定上下文回答问题；如果上下文不足以回答，请明确说“上下文不足”。
问题：{query}

上下文：
{context}
"""

resp = client.responses.create(
    model="gpt-5",  # 替换为你账号可用的模型
    instructions="回答要简洁，但要给出依据并引用Doc编号。",
    input=prompt
)

print(resp.output_text)
```

## 性能、成本与质量权衡

### 端到端延迟：检索 + 重排 + 生成

RAG的时延通常可拆为：  
1) embedding计算（查询向量化）；2) 检索（kNN/BM25）；3) 融合/重排；4) 生成。citeturn6search4turn2search4turn9search9

若引入FiD式多段证据融合或更强重排器，质量往往更好，但解码器侧跨注意力与更长上下文会增加推理开销；相关研究也指出FiD类结构推理时间瓶颈往往在decoder侧。citeturn0search14turn0search6

### 存储：向量维度、精度与“存哪里”

向量存储的下界可用“维度×元素字节数”估算。Milvus中文索引文档给出示例：128维float向量占用 128×4=512字节；同理可推得1536维float向量约6KB，3072维约12KB（不含索引额外结构与元数据）。citeturn4search4turn1search11

Elasticsearch在向量检索上也强调存储策略：官方性能优化建议可通过mapping把 `dense_vector` 从 `_source` 中排除，以避免返回与存储大向量带来的索引膨胀；并支持 `byte`（int8）向量以降低内存占用、提升缓存效率。citeturn11search33turn11search21

此外，Elasticsearch近期也在推进更低存储精度（如bfloat16存储）等方向，以在磁盘占用与性能之间做权衡。citeturn11search2turn11search6

### 吞吐：ANN索引结构与硬件（CPU/GPU/SSD）

FAISS官方文档指出其包含可在GPU上实现的算法，用于提升大规模向量相似度搜索性能；相应研究也讨论了GPU在十亿级相似度搜索中的加速价值。citeturn0search3turn0search7

Milvus的索引解释文档强调：当数据规模超过RAM时，可以考虑mmap与DiskANN等机制，把部分索引结构放到磁盘上以缓解内存压力；其DiskANN文档把定位明确为“超内存规模、仍保持准确率与速度”的磁盘向量检索方案。citeturn11search1turn11search3

### 召回与生成准确性：质量不是单点最优，而是系统级最优

DPR等研究显示检索召回提升会显著影响端到端QA效果，因此“先把检索做对”通常是RAG质量提升的最高杠杆之一。citeturn0search1turn0search26

同时，综述研究普遍将RAG优化拆为检索前（pre‑retrieval）、检索、检索后（post‑retrieval）与生成四块，并强调系统复杂度提升后需要系统化评估与工程闭环。citeturn3search4turn3search0

## 常见问题与调优

### 幻觉仍可能发生：RAG是“减少”而非“消除”

RAG的动机之一是通过显式证据提升事实性并降低无依据生成；但如果检索到的上下文不相关、被污染或被提示注入，生成仍可能偏离证据。citeturn2search2turn3search4turn2search3

调优思路通常是“先证据、后生成”：提高检索质量（召回与精确度）、增强重排、在提示中要求“仅基于上下文”、并在输出中强制引用证据段落或给出“上下文不足”的拒答路径。citeturn3search1turn3search2turn2search24

### 检索噪声与“相似但无用”的上下文

语义检索可能返回“主题相关但不含答案”的材料；关键词检索可能返回“命中词但语义无关”的材料。混合检索与融合（尤其RRF）常用于缓解两者各自的系统性偏差。citeturn9search9turn9search4turn8search4

进一步的工程增强包括：用MMR降低冗余、用交叉编码器重排提升精确度、用AutoMerging/父子chunk返回更完整语境等。citeturn8search3turn9search30turn9search2

### 上下文长度限制：证据选择与压缩成为核心能力

FiD与RAG类结构都依赖把“检索证据”塞入上下文，而大模型上下文窗口有限，因此必须在“更多证据”与“更低时延/更少token成本”之间取舍。相关研究与工程实践通常采用：Top‑k截断、去重、段落合并、基于问题的摘要压缩、以及多轮检索（不足则再检索）等策略。citeturn0search14turn4search6turn3search4

### 过拟合：检索器/重排器对特定分布“学得太像”

在可微端到端训练或领域微调中，检索器可能对训练集分布过拟合，导致跨域查询召回下降；这类风险在“训练数据小、语料更新频繁、标签噪声高”的场景更突出。综述工作通常将其归入RAG系统复杂化后的训练与泛化挑战，需要通过更稳健的评估集、对抗样例、以及持续回归测试来控制。citeturn3search4turn3search0

### 评估指标与测试用例：必须覆盖“检索正确 + 生成扎根”

推荐把评估拆为三层：

检索层：Recall@k、MRR、nDCG、top‑k包含答案证据比例（DPR式指标表述）。citeturn0search26turn12search24

上下文层与答案层：RAGAS提供context recall/precision、faithfulness等；TruLens提供RAG triad（context relevance、groundedness、answer relevance）。citeturn3search1turn3search2turn3search5

回归用例：应包含实体/数字/日期精确问答、否定与对比问题、权限隔离问题、以及“提示注入/恶意文档”用例，以覆盖安全与可靠性风险。citeturn2search24turn2search3

## 部署与安全隐私，以及参考资料

### 部署建议：本地/云、CPU/GPU、混合检索

本地原型常用“FAISS + 轻量embedding + 小模型生成”，优点是部署简单、成本可控；但当数据规模、并发、权限治理与可观测性要求上升时，向量数据库或搜索引擎（Milvus/Elasticsearch/OpenSearch等）更适合承担服务化检索层。citeturn0search3turn11search22turn2search4turn12search3

硬件建议可按瓶颈选配：embedding与生成更吃GPU（或高性能CPU）；检索侧取决于索引结构与数据规模——HNSW等内存索引更依赖RAM，DiskANN等更依赖SSD与IO链路；若使用Elasticsearch，需关注dense_vector是否存入_source、是否采用byte向量等以控制存储与缓存效率。citeturn11search3turn11search21turn11search33

混合检索通常建议默认启用（BM25+向量）并选择低调参的融合方式（如RRF），在企业知识库、客服与文档检索中更稳健。citeturn9search9turn9search4turn8search4

### 安全与隐私：提示注入、敏感信息、访问控制与合规

entity["organization","OWASP","open web app security project"]将Prompt Injection列为LLM应用的首要风险之一，并指出攻击者可通过构造输入操纵模型行为；其防护要点包括隔离指令与数据、对不可信内容进行约束与过滤。citeturn2search7turn2search3turn2search24

敏感信息泄露也是常见风险：模型可能在回答中泄漏PII或机密信息，尤其当检索层把不该给用户看的文档召回并注入上下文时。citeturn10search3turn10search11

访问控制方面，Elasticsearch支持文档级与字段级安全控制，用于确保不同用户只能检索到被授权的文档与字段；Milvus也提供RBAC以在集合/数据库/实例层面做细粒度权限控制。citeturn10search0turn10search1

关于“把数据发给模型服务商”的隐私问题，OpenAI平台文档说明：自2023‑03‑01起，发送到OpenAI API的数据默认不会用于训练或改进模型（除非显式选择加入数据共享）。citeturn10search2

entity["organization","NIST","us standards institute"]发布的AI风险管理框架（AI RMF 1.0）强调可信AI的多维特性（安全、透明、隐私增强等），可作为RAG系统在治理、评估与上线流程中进行风险分解与控制的参考框架。citeturn6search3turn6search7turn6search17

### 参考资料（中英文链接）

为满足“权威来源优先”的要求，以下优先列出原始论文、官方文档与主流开源库文档；其中尽量同时给出中文页面（若有）与英文页面。

**原始论文与综述（英文为主）**  
1) RAG原始论文（NeurIPS 2020）：https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf citeturn0search0  
2) DPR原始论文（EMNLP 2020 / arXiv）：https://arxiv.org/abs/2004.04906 citeturn0search1  
3) FiD原始论文（arXiv）：https://arxiv.org/pdf/2007.01282 citeturn0search14  
4) BM25综述（PRF: BM25 and Beyond）：https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf citeturn1search6  
5) Okapi at TREC‑3（BM25早期经典）：https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/okapi_trec3.pdf citeturn6search2  
6) RAG综述（Gao et al., 2023）：https://arxiv.org/abs/2312.10997 citeturn3search0  
7) 检索增强文本生成综述（Huang & Huang, 2024）：https://arxiv.org/abs/2404.10981 citeturn3search4  
8) RRF论文（Cormack et al., SIGIR 2009）：https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf citeturn3search3  
9) MMR原始论文（Carbonell & Goldstein, 1998）：https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf citeturn8search3  

**核心组件官方文档（含中文入口）**  
1) FAISS官方文档（英文）：https://faiss.ai/index.html citeturn0search3  
2) FAISS索引家族（英文wiki）：https://github.com/facebookresearch/faiss/wiki/Faiss-indexes citeturn11search0  
3) Milvus索引（中文）：https://milvus.io/docs/zh/index.md citeturn4search4  
4) Milvus索引解释（英文）：https://milvus.io/docs/index-explained.md citeturn11search1  
5) Milvus全文检索（BM25）与混合检索（英文）：https://milvus.io/docs/full_text_search_with_milvus.md citeturn8search4  
6) Elasticsearch kNN（英文）：https://www.elastic.co/docs/solutions/search/vector/knn citeturn2search4  
7) Elasticsearch混合检索（英文）：https://www.elastic.co/docs/solutions/search/hybrid-search citeturn9search9  
8) Elasticsearch RRF（英文）：https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion citeturn9search4  
9) Elasticsearch kNN教程（中文：Elastic Labs）：https://www.elastic.co/search-labs/cn/tutorials/search-tutorial/vector-search/nearest-neighbor-search citeturn4search9  
10) Hugging Face Transformers RAG文档（英文）：https://huggingface.co/docs/transformers/en/model_doc/rag citeturn2search2  
11) Hugging Face Transformers RAG文档（中文/繁体入口）：https://huggingface.tw/docs/transformers/model_doc/rag citeturn4search2  
12) OpenAI embeddings指南（英文）：https://developers.openai.com/api/docs/guides/embeddings/ citeturn7search0  
13) OpenAI Responses API（英文）：https://platform.openai.com/docs/api-reference/responses citeturn7search5  
14) OpenAI 数据控制（英文）：https://developers.openai.com/api/docs/guides/your-data/ citeturn10search2  

**评估与安全（英文为主）**  
1) RAGAS指标文档：https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ citeturn3search18  
2) TruLens RAG Triad：https://www.trulens.org/getting_started/core_concepts/rag_triad/ citeturn3search2  
3) OWASP LLM01 Prompt Injection：https://genai.owasp.org/llmrisk/llm01-prompt-injection/ citeturn2search3  
4) NIST AI RMF 1.0（PDF）：https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf citeturn6search3