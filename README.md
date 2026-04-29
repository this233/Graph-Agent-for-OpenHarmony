# DualGraph-Agent-for-OpenHarmony

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

一个面向 **OpenHarmony** 的 **图谱增强智能体（Graph-Augmented Agent）**。系统由两层组成：

- **Agent 层（`OpenHarmonyAssistant/`）**：负责对话、规划与 Generative UI 输出，把用户查询路由到下层工具。
- **DualGraphRAG 工具层（`src/hipporag/` + `OpenHarmonyAssistant/chatbox/`）**：在 [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) 基础上深度改造，把 Markdown 文档解析成 **结构-语义双层异构图谱（Dual Graph）**，并使用 **代价感知的 Best-First 图扩散** 完成 4 阶段多模态检索，作为 Agent 的核心检索工具。

整体面向"文档级问答 + 跨文件 API 推理 + 富交互 UI 应答"三类场景。

> 命名说明：仓库根包仍叫 `hipporag`（代码导入路径），但本项目在节点结构、检索流程、服务化方式上做了较大改造，建议按本 README 而非上游文档为准。

---

## 目录

- [项目亮点](#项目亮点)
- [系统架构](#系统架构)
- [知识图谱：节点与边](#知识图谱节点与边)
- [构建流程（5 步）](#构建流程5-步)
- [检索流程（4 阶段）](#检索流程4-阶段)
- [快速开始](#快速开始)
- [服务化与 API](#服务化与-api)
- [配置参数](#配置参数)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 项目亮点

1. **细粒度多模态分域嵌入**  
   文档段落被解析为 6 类图节点 + 1 类纯向量节点，**每类独立向量库**，避免长文本淹没代码 / 表格 / 图片 / 实体的信号：
   - 图节点：`file` / `chunk` / `code` / `table` / `image` / `entity`
   - 纯向量节点（不入图，仅作为种子来源）：`fact`（OpenIE 三元组）

2. **结构 + 语义二级异构图谱**  
   - **结构层**：来自 Markdown 标题层级与超链接 —— `File → Chunk → SubChunk`、`Chunk → Code/Table/Image`、`Chunk → File`（跳转）。
   - **语义层**：来自 OpenIE 抽取的三元组 + 实体 KNN 同义词边 + 段落↔实体关联边。
   - 两层之间通过 `Chunk ↔ Entity` 桥接，形成"外圈结构 + 内圈语义"的双层网络。

3. **代价感知 Best-First 图扩散（非 PPR）**  
   线上检索（`retrieve_v2`）使用堆驱动的 Best-First 扩散，从 Fact 命中实体 + 命中文件出发，沿"低代价边"多跳传播：  
   `score = sim(node, query) · init_weight / (1 + cumulative_cost)`  
   不同类型边给出不同代价（`synonymy / semantic / passage / structural`），并配有时间预算、扩展数、frontier 大小等多重早停，**比 PPR 全图迭代延迟更可控**，更契合在线 API 推理。

4. **图片感知哈希去重**  
   建图时对图片用 dHash/pHash + LSH 桶 + 可选 SSIM 做近重复合并，保留最高分辨率代表，被合并图片的边会自动重定向。

5. **完整的服务化能力**  
   双进程架构：`hipporag_service`（预加载图谱、专注检索）+ `server_text`（API 网关、调用 LLM 生成最终回答），配套自带前端页面。

---

## 系统架构

```
┌────────────────────────────────────────────────────────────────────┐
│                  User · Browser (frontend.html)                    │
└─────────────────────────────┬──────────────────────────────────────┘
                              │ HTTP
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│   Agent 层 · OpenHarmonyAssistant   (server_text.py · port 8000)   │
│   • 对话规划 / 工具路由 / Generative UI 输出                        │
│   • 把用户查询喂给下层 DualGraphRAG，再把检索结果喂给 Chat LLM      │
└─────────────────────────────┬──────────────────────────────────────┘
                              │ HTTP（作为 Agent 的检索工具调用）
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│         DualGraphRAG 工具层 · hipporag_service.py (port 8001)       │
│   • 预加载 Embedding / Reranker / 双层异构知识图谱                  │
│   • 执行 4 阶段检索：召回 → Rerank → 代价感知图扩散 → 终排          │
│   • 返回多模态检索结果（chunks / codes / tables / images）          │
└────────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 位置 | 说明 |
| --- | --- | --- |
| `HippoRAG` | `src/hipporag/HippoRAG.py` | 核心引擎：图谱构建 + 检索 |
| `DocumentProcessor` | `src/hipporag/document_processor.py` | Markdown → 层次化 JSON 解析 |
| `OpenIE` | `src/hipporag/information_extraction/` | 实体与三元组抽取（在线 / 离线） |
| `EmbeddingStoreV2` | `src/hipporag/embedding_store_v2.py` | 多类型向量存储 |
| `TransformersCrossEncoderReranker` | `src/hipporag/rerankers/` | 本地 bge-reranker 精排 |
| `hipporag_service.py` | `OpenHarmonyAssistant/chatbox/` | FastAPI 检索服务 |
| `server_text.py` | `OpenHarmonyAssistant/chatbox/` | FastAPI 网关 + 命令行 |
| `frontend.html` | `OpenHarmonyAssistant/chatbox/` | Web 前端 |

---

## 知识图谱：节点与边

### 节点（6 类入图 + 1 类纯向量）

| 节点 | ID 前缀 | 嵌入文本 | 是否入图 |
| --- | --- | --- | --- |
| 文件 | `file-`   | 文件摘要 | 是 |
| 段落 | `chunk-`  | 段落摘要（embedding 内容用 `filter_chunk.content`：去掉代码/表/图引用后的纯文本） | 是 |
| 代码块 | `code-`  | 代码块摘要（LLM 生成） | 是 |
| 表格 | `table-`  | 表格摘要（LLM 生成） | 是 |
| 图片 | `image-`  | 图片 caption（MLLM 生成，配感知哈希去重） | 是 |
| 实体 | `entity-` | `"name: description"` | 是 |
| 事实 | `fact-`   | 三元组字符串 `(h, r, t)` | **否（仅向量库，作为图扩散种子的来源）** |

> 注：原版 HippoRAG 的"Fact 节点入图 + PPR"链路在本项目中已**不再使用于线上检索**。线上 `retrieve_v2` 用的是基于代价的 Best-First 扩散，Fact 仅在向量层作为 seed 生成器。

### 边（5 类，由 `_classify_edge_type` 统一打标）

```
① structural   包含关系（contains 边自动加反向）
   File ──contains──▶ Chunk ──contains──▶ SubChunk / Code / Table / Image

② jump         段落级超链接（Markdown [text](xxx.md)）
   Chunk ──jump──▶ File

③ passage      段落↔实体（chunk 内出现过的实体）
   Chunk ──contains──▶ Entity

④ semantic     实体三元组（双向，权重=共现频次）
   Entity ◀──(h, r, t)──▶ Entity

⑤ synonymy    同义扩展（实体向量 KNN + 阈值 ≥ synonymy_edge_sim_threshold）
   Entity ◀──sim──▶ Entity
```

### 边代价（用于扩散打分）

| 边类型 | 代价公式 |
| --- | --- |
| `synonymy`   | `max(0.01, 1 - weight)` |
| `semantic`   | `0.5 / max(weight, 0.1)` |
| `passage`    | `0.2`（固定） |
| `structural` | `0.3`（固定） |
| 其他/未知    | `0.5` |

直观含义：**结构与段落-实体边便宜，弱语义/弱同义边贵**，从而控制噪声、控制扩展规模。

---

## 构建流程（5 步）

> 所有命令默认在仓库根目录执行，输出根目录为 `outputs/Harmony_docs_zh_cn/`。

```bash
conda activate hipporag
```

### 步骤 1 · 文档结构解析

`DocumentProcessor` 把 Markdown 解析为层次化 JSON：

- 按 `#` ~ `######` 递归分块
- 抽取 ≥ 5 行的代码块 / Markdown / HTML 表格（带前后 `context_lines` 上下文）
- 抽取图片引用并解析相对路径
- 抽取段落内的 `[text](*.md)` 超链接生成 `jump`
- 同时输出去掉代码/表/图引用的 `filter_chunk.content`，供后续 OpenIE 使用

```bash
python src/hipporag/document_processor.py >> index.log 2>&1
# 默认对应：/root/code/docs/zh-cn → outputs/Harmony_docs_zh_cn/markdown_parse/structure.json
```

也可以代码调用：

```python
from src.hipporag.document_processor import DocumentProcessor
proc = DocumentProcessor(context_lines=15)
proc.process_directory(
    "/path/to/docs/zh-cn",
    "outputs/Harmony_docs_zh_cn/markdown_parse/structure.json",
)
```

### 步骤 2 · 摘要生成（File / Chunk / Code / Table）

```bash
python generate_abstracts.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/structure.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/abstract.json \
    --backup outputs/Harmony_docs_zh_cn/markdown_parse_bak/abstract.json \
    --max-workers 20 \
    >> index.log 2>&1
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--max-workers` | 20 | 并发线程数 |
| `--backup` | - | 失败时复用已有摘要 |
| `--disable-fallback` | False | 关闭备用 LLM 配置 |
| `--dry-run` | False | 只统计任务 |

### 步骤 3 · 图片描述（MLLM Caption）

```bash
# 第一轮
python generate_image_captions.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/abstract.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions.json \
    --backup outputs/Harmony_docs_zh_cn/markdown_parse_bak/abstract.json \
    --max-workers 10

# 重试失败项
python retry_failed_captions.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions_final.json
```

### 步骤 4 · 实体与三元组抽取（OpenIE）

`extract_entities_triples.py` 会把 OpenIE 结果写回每个 `chunk.filter_chunk.extracted_entities / extracted_triples`：

```bash
python extract_entities_triples.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions_final.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/triples.json \
    --batch-size 200 \
    --openie-mode online \
    >> index.log 2>&1
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--batch-size` | 200 | 批大小 |
| `--openie-mode` | online | `online` 走 OpenAI 协议；`offline` 走 vLLM |
| `--llm-name` | 见脚本 | LLM 名称 |
| `--llm-base-url` | - | 自定义端点 |
| `--disable-fallback` | False | 关闭备用配置 |

每个 chunk 输出形如：

```json
"filter_chunk": {
  "content": "ArkTS 是 OpenHarmony 的主要开发语言…",
  "extracted_entities": [
    ["ArkTS", "OpenHarmony 主推的应用开发语言"],
    ["OpenHarmony", "操作系统"]
  ],
  "extracted_triples": [
    ["ArkTS", "是…的主要开发语言", "OpenHarmony"]
  ]
}
```

### 步骤 5 · 知识图谱构建与索引

启动 `hipporag_service.py` 时会自动调用 `HippoRAG.index_from_json` 完成：

1. 各类节点向量化并写入对应嵌入存储（`file/chunk/code/table/image/entity/fact`）
2. 图片感知哈希去重（dHash + 可选 SSIM + LSH 桶）
3. 添加 5 类边（structural / jump / passage / semantic / synonymy）
4. 同义词边补全 → `augment_graph` → 写入磁盘
5. 保存节点元信息（面包屑导航 `breadcrumb` + Gitee URL）

也可手动触发：

```python
import json
from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

cfg = BaseConfig()
cfg.save_dir = "outputs/Harmony_docs_zh_cn"
cfg.llm_name = "deepseek-v3.2-exp"
cfg.llm_base_url = "https://api.modelarts-maas.com/openai/v1"
cfg.embedding_model_name = "Qwen3-Embedding-4B"

hipporag = HippoRAG(global_config=cfg)
with open("outputs/Harmony_docs_zh_cn/markdown_parse/triples.json") as f:
    json_structure = json.load(f)
hipporag.index_from_json(json_structure)
```

### 数据流

```
structure.json → abstract.json → with_captions.json
              → with_captions_final.json → triples.json → 图谱(igraph)
```

---

## 检索流程（4 阶段）

`retrieve_v2` 是真正在线上跑的方法，分 4 个阶段：

```
┌──────────────────────────────────────────────────────────────────────┐
│ 阶段 1 · 多路向量召回                                                 │
│   Fact-top100 / File-top100 / Chunk-top100                            │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 阶段 2 · Rerank（Cross-encoder bge-reranker-v2-m3，本地）              │
│   Fact → top50 / File → top50 / Chunk → top50                         │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 阶段 3 · 代价感知 Best-First 图扩散                                    │
│   • 种子：top-Fact 解出的 Entity（init_weight=0.8）                    │
│           + top-File（init_weight=0.9）                                │
│   • 必选 Chunk：top-Fact 来源 chunk + 阶段 2 top-Chunk                 │
│   • 评分：score = sim(node, q) · init_weight / (1 + cumulative_cost)   │
│   • 收集：Chunk / Code / Table / Image                                 │
│   • 早停：time_budget_s / max_expansions /                             │
│           per_node_neighbor_limit / max_frontier_size /                │
│           min_enqueue_score / 候选已收满                               │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 阶段 4 · 最终 Rerank（默认 LLM 后端） + 合并输出                        │
│   Chunk → 10 / Code → 2 / Table → 2 / Image → 2                       │
│   合并为可直接展示的 Markdown（带面包屑、Gitee URL、ID）               │
└──────────────────────────────────────────────────────────────────────┘
```

### 关键参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `fact_candidate_k` / `file_candidate_k` / `chunk_candidate_k` | 100 | 阶段 1 三路召回数 |
| `fact_top_k` / `file_top_k` / `chunk_top_k` | 50 | 阶段 2 精排保留数 |
| `spread_chunk_k` | 100 | 阶段 3 chunk 候选上限 |
| `spread_code_k` / `spread_table_k` / `spread_image_k` | 5 | 阶段 3 模态候选上限 |
| `spread_time_budget_s` | 2.0 | 扩散时间预算（秒） |
| `spread_max_expansions` | 12000 | 最大扩展节点数 |
| `spread_per_node_neighbor_limit` | 96 | 单节点最多扩展邻居数（按低代价优先） |
| `spread_max_frontier_size` | 20000 | 优先队列最大长度 |
| `spread_min_enqueue_score` | 0.01 | 入队最小分数阈值 |
| `final_chunk_k` | 10 | 最终返回 chunk 数 |
| `final_code_k` / `final_table_k` / `final_image_k` | 2 | 最终返回各模态数 |
| `generate_report` | false | 是否生成 LLM 整合报告 |
| `verbose` | true | 是否打印检索过程 |

> `server_text.py` 的默认网关配置略有不同（`fact_candidate_k=300`、`chunk_candidate_k=300`、`fact_top_k=10` 等），见 `DEFAULT_RETRIEVAL_CONFIG`。

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+（推荐 GPU；建议 ≥ 24GB 显存）
- 网络可访问 LLM/Embedding API（或本地 vLLM）

### 安装

```bash
pip install -r requirements.txt
pip install -e .
```

### 启动服务

**1. HippoRAG 服务（端口 8001，预加载较慢）**

```bash
cd OpenHarmonyAssistant/chatbox
python hipporag_service.py --port 8001
```

看到 `✅ HippoRAG 初始化完成` 后即可。

**2. API 网关（端口 8000，秒级启动）**

```bash
# 新开终端
cd OpenHarmonyAssistant/chatbox
python server_text.py --server --port 8000
```

**3. 浏览器打开** `http://localhost:8000/`

### 命令行交互

```bash
cd OpenHarmonyAssistant/chatbox
python server_text.py -i        # 交互模式（默认 RAG + LLM）
python server_text.py "如何创建一个 ArkTS 页面？"  # 单次提问
python server_text.py "如何使用 @State？" --no-llm  # 仅检索
```

---

## 服务化与 API

### 端点速查

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/` (8000)         | GET  | 返回前端页面 |
| `/chat` (8000)     | POST | RAG + LLM 完整问答 |
| `/retrieve` (8000) | POST | 仅检索（不调用 LLM） |
| `/health` (8000)   | GET  | 健康检查（含下游 8001 状态） |
| `/retrieve` (8001) | POST | 直接调用底层 HippoRAG 检索 |
| `/health` (8001)   | GET  | HippoRAG 服务健康检查 |

### `/chat` 示例

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何创建一个 ArkTS 页面？",
    "use_rag": true,
    "use_llm": true
  }'
```

只检索：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "如何使用 @State 装饰器？", "use_rag": true, "use_llm": false}'
```

自定义检索参数：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何创建一个 ArkTS 页面？",
    "retrieval_config": {
      "final_chunk_k": 5,
      "final_code_k": 3,
      "spread_chunk_k": 80,
      "verbose": true
    }
  }'
```

### 响应（节选）

```json
{
  "query": "如何创建一个 ArkTS 页面？",
  "rag_context": "# 相关文档段落 1\n- ID: chunk-...\n- 文件: https://gitee.com/...\n- 定位: H1 / H2 / H3\n\n...",
  "llm_response": "要创建一个 ArkTS 页面，您需要…",
  "rag_chunks_count": 10,
  "rag_codes_count": 2,
  "rag_tables_count": 2,
  "rag_images_count": 2,
  "timing": {
    "stage1_embedding": 1.23,
    "stage2_rerank": 2.34,
    "stage3_spread": 0.56,
    "stage4_final_rerank": 3.45,
    "stage4_merge": 0.01,
    "rag_total": 7.59,
    "llm": 5.67,
    "total": 13.26
  }
}
```

更多 curl / jq 用法见 [`OpenHarmonyAssistant/chatbox/API_README.md`](OpenHarmonyAssistant/chatbox/API_README.md)。

---

## 配置参数

### `BaseConfig`（`src/hipporag/utils/config_utils.py`）

**LLM**

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `llm_name` | gpt-4o-mini | OpenIE / 最终精排 / QA 用的 LLM |
| `llm_base_url` | None | 自定义 OpenAI-兼容端点 |
| `max_new_tokens` | 2048 | 单次推理最大 token |
| `temperature` | 0 | 采样温度 |

**Embedding**

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `embedding_model_name` | nvidia/NV-Embed-v2 | 嵌入模型 |
| `embedding_batch_size` | 10000 | 嵌入批大小 |
| `embedding_max_seq_len` | 2048 | 最大序列长度 |
| `embedding_return_as_normalized` | True | 是否归一化 |

**图谱构建**

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `synonymy_edge_topk` | 2047 | 实体 KNN 候选数 |
| `synonymy_edge_sim_threshold` | 0.95 | 同义词边相似度阈值 |
| `is_directed_graph` | False | 是否有向 |
| `enable_image_content_dedup` | True | 是否启用图片感知哈希去重 |
| `image_dedup_hash_method` | dhash | `dhash` / `phash` |
| `image_dedup_hamming_threshold` | 6 | 汉明距离阈值 |
| `enable_image_dedup_ssim` | False | 是否额外 SSIM 复核 |

### 服务侧默认配置（`hipporag_service.py`）

| 项 | 默认 |
| --- | --- |
| `save_dir` | `outputs/Harmony_docs_zh_cn` |
| `llm_name` | `deepseek-v3.2-exp` |
| `llm_base_url` | `https://api.modelarts-maas.com/openai/v1` |
| `embedding_model_name` | `Qwen3-Embedding-4B` |
| `rerank_backend` | `transformers` |
| `rerank_model_name` | 本地 `bge-reranker-v2-m3` |
| `final_rerank_backend` | `llm` |

### 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HIPPORAG_SERVICE_URL` | `http://localhost:8001` | 网关访问的下游 HippoRAG 地址 |
| `CHAT_MODEL_NAME` | `qwen3-coder-480b-a35b-instruct` | 网关侧 Chat LLM |
| `CHAT_BASE_URL` | `https://api.modelarts-maas.com/v2` | 网关侧 Chat LLM 端点 |
| `OPENAI_API_KEY` / `MAAS_API_KEY` | - | LLM 鉴权 |
| `RERANK_BACKEND` | `transformers` | 阶段 2 rerank 后端 |
| `RERANK_MODEL_NAME` | `bge-reranker-v2-m3` 路径 | rerank 模型 |
| `RERANK_DEVICE` / `RERANK_BATCH_SIZE` / `RERANK_MAX_LENGTH` | 见代码 | rerank 性能调优 |
| `FINAL_RERANK_BACKEND` | `llm` | 阶段 4 最终 rerank 后端 |

---

## 项目结构

```
Graph-Agent-for-OpenHarmony/
├── src/hipporag/                        # 核心代码
│   ├── HippoRAG.py                      # 主类：图谱构建 + 4 阶段检索
│   ├── document_processor.py            # Markdown → 层次化 JSON
│   ├── abstract_generator.py            # 摘要生成器
│   ├── embedding_store_v2.py            # 多类型向量存储
│   ├── rerank.py                        # DSPy 风格 fact rerank
│   ├── information_extraction/          # OpenIE（在线 / 离线）
│   ├── embedding_model/                 # Embedding 适配
│   ├── llm/                             # LLM 客户端
│   ├── rerankers/                       # Cross-encoder rerank
│   ├── prompts/                         # 提示模板
│   └── utils/                           # 配置 / 工具
│
├── OpenHarmonyAssistant/chatbox/        # Web 服务 + 前端
│   ├── hipporag_service.py              # FastAPI 检索服务（端口 8001）
│   ├── server_text.py                   # FastAPI 网关 + CLI（端口 8000）
│   ├── frontend.html                    # Web 前端
│   └── API_README.md                    # API 详细文档
│
├── extract_entities_triples.py          # 步骤 4：实体三元组抽取
├── generate_abstracts.py                # 步骤 2：摘要生成
├── generate_image_captions.py           # 步骤 3：图片 caption
├── retry_failed_captions.py             # 步骤 3 重试
├── markdown_parser.py                   # Markdown 解析辅助
├── demo_retrieval.py / demo_*.py        # 各类示例
│
├── outputs/Harmony_docs_zh_cn/          # 默认输出目录（索引 + 嵌入）
├── bge-reranker-v2-m3/                  # 本地 rerank 模型（自备）
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 常见问题

**Q1. 启动 `hipporag_service.py` 卡住或 OOM？**  
- 检查 `nvidia-smi`，确认 CUDA 可用；显存不足可缩小 `embedding_batch_size`。  
- 第一次启动会读取 `outputs/Harmony_docs_zh_cn/` 下全部嵌入并构建 igraph，**预热时间在分钟级**属正常。

**Q2. 检索结果为空？**  
- 确认 `outputs/Harmony_docs_zh_cn/` 下嵌入文件存在；
- `curl http://localhost:8001/health` 看 `hipporag_initialized` 是否为 `true`；
- 用 `verbose: true` 调用 `/retrieve` 查看每阶段日志。

**Q3. `/chat` 超时？**  
- 减小 `final_chunk_k` / `spread_chunk_k`；
- 缩短 `spread_time_budget_s`；
- 确认下游 LLM API 可达。

**Q4. 如何增量加文档？**  
1. 重新跑 `DocumentProcessor.process_directory` 生成新 `structure.json`；
2. 顺次执行步骤 2~4；
3. 重新运行 `index_from_json`（或重启 `hipporag_service.py`）。  
代码内 `add_synonymy_edges` / `add_new_nodes` / `add_new_edges` 已支持增量合并。

**Q5. 想换 LLM / Embedding？**  
- 换 LLM：修改 `hipporag_service.py` 中 `llm_model_name` / `llm_base_url`，或通过 `OPENAI_API_KEY` + 自定义 `BaseConfig`；
- 换 Embedding：修改 `embedding_model_name`；如果向量维度变化，**必须重新构建索引**。

---

## 维护者备注

- **`HippoRAG.py` 接近 8000 行**，修改 `index_from_json` / `retrieve_v2` / `graph_spread_with_similarity` 前请先理解 `node_to_node_stats / fact_to_chunk_id / fact_to_entities` 三个核心字典。
- 新增节点类型：在 `index_from_json` 增 `*_embedding_store`、在 `_extract_nodes_from_json` 增解析、在 `_classify_edge_type` 增分类；同时在 `retrieve_v2` 阶段 3/4 增加候选收集与最终精排逻辑。
- 新增边类型：在 `_classify_edge_type` 中分类，并在 `graph_spread_with_similarity::get_edge_cost` 给出代价。
- 调试单查询：

```python
from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

hipporag = HippoRAG(global_config=BaseConfig(save_dir="outputs/Harmony_docs_zh_cn"))
hipporag.prepare_retrieval_objects()
result = hipporag.retrieve_v2(["如何使用 @State？"], verbose=True)
```

---

## 许可证

MIT。详见 [LICENSE](LICENSE)。

## 更新日志

- **v2.2** 重写 README，对齐线上 `retrieve_v2` 实际行为（代价感知 Best-First 扩散，非 PPR）；新增图片感知哈希去重说明。
- **v2.1** 引入层次化 JSON 索引（`index_from_json`）。
- **v2.0** 4 阶段检索流程上线。
- **v1.0** 基础 HippoRAG 实现。
