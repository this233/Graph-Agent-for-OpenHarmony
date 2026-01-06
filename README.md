# HippoRAG: 基于知识图谱的检索增强生成系统

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

HippoRAG 是一个受人类海马体记忆机制启发的先进 RAG（检索增强生成）系统，结合了开放信息抽取(OpenIE)、知识图谱构建、多层次向量嵌入和个性化 PageRank 算法，用于高质量的文档检索和问答。

---

## 📁 目录

- [系统架构](#系统架构)
- [知识图谱构建流程](#知识图谱构建流程)
- [检索流程（4阶段）](#检索流程4阶段)
- [快速开始](#快速开始)
- [API 使用文档](#api-使用文档)
- [配置参数详解](#配置参数详解)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 系统架构

### 整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              前端 (frontend.html)                         │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ HTTP
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      API 网关 (server_text.py:8000)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  • 接收用户请求                                                      │ │
│  │  • 调用 HippoRAG 服务进行检索                                        │ │
│  │  • 调用 LLM 生成回答                                                 │ │
│  │  • 格式化输出结果                                                    │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ HTTP
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  HippoRAG 服务 (hipporag_service.py:8001)                 │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  • 预加载 Embedding 模型和知识图谱                                   │ │
│  │  • 执行 4 阶段检索流程                                               │ │
│  │  • 返回多类型检索结果                                                │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 描述 | 位置 |
|------|------|------|
| **HippoRAG** | 核心检索引擎，实现知识图谱构建和检索 | `src/hipporag/HippoRAG.py` |
| **DocumentProcessor** | 文档解析器，处理 Markdown 文件结构 | `src/hipporag/document_processor.py` |
| **OpenIE** | 开放信息抽取模块，提取实体和三元组 | `src/hipporag/information_extraction/` |
| **EmbeddingStoreV2** | 向量嵌入存储，支持多类型内容 | `src/hipporag/embedding_store_v2.py` |
| **TransformersCrossEncoderReranker** | 本地 Rerank 模型 | `src/hipporag/rerankers/` |

---

## 知识图谱构建流程

### TL;DR 快速命令

```bash
conda activate hipporag

# 步骤1: 提取文档结构（包含图片）
python src/hipporag/document_processor.py >> index.log 2>&1

# 步骤2: 生成全文摘要（失败时复用备份）
python generate_abstracts.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/structure.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/abstract.json \
    --backup outputs/Harmony_docs_zh_cn/markdown_parse_bak/abstract.json \
    >> index.log 2>&1

# 步骤3: 生成图片caption
python generate_image_captions.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/abstract.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions.json \
    --backup outputs/Harmony_docs_zh_cn/markdown_parse_bak/abstract.json \
    >> index.log 2>&1

python retry_failed_captions.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions_final.json

# 步骤4: 提取实体和三元组
python extract_entities_triples.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions_final.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/triples.json \
    >> index.log 2>&1

# 步骤5: 构建知识图谱索引（通过启动服务自动完成）
cd OpenHarmonyAssistant/chatbox
python hipporag_service.py --port 8001
```

### 流程概览

```
步骤1                   步骤2                   步骤3
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Markdown 文档  │ ──▶ │   文档结构解析   │ ──▶ │   摘要生成      │
│   (.md 文件)     │     │   (层次化JSON)   │     │   (LLM)        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
步骤5                   步骤4                   步骤3续
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   知识图谱构建   │ ◀── │   实体三元组提取  │ ◀── │   图片描述生成   │
│   (igraph)      │     │   (OpenIE)      │     │   (MLLM)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   向量嵌入存储   │ ──▶ │   索引完成       │
│   (多类型)      │     │   (可检索)       │
└─────────────────┘     └─────────────────┘
```

**数据流向**:
```
structure.json → abstract.json → with_captions.json → with_captions_final.json → triples.json
    步骤1           步骤2              步骤3                 步骤3重试              步骤4
```

### 第一步：文档结构解析

使用 `DocumentProcessor` 将 Markdown 文档解析为层次化 JSON 结构。

**核心代码位置**: `src/hipporag/document_processor.py`

**处理内容**:
- **Markdown 标题分层**: 根据 `#`、`##`、`###` 等标题层级递归分块
- **代码块提取**: 提取 ≥5 行的代码块，保留上下文
- **表格提取**: 提取 ≥5 行的 Markdown/HTML 表格
- **图片提取**: 提取图片引用，解析相对路径
- **超链接提取**: 识别文档间的跳转关系

**执行命令**:
```bash
# 直接运行脚本（使用脚本内预设的路径配置）
python src/hipporag/document_processor.py >> index.log 2>&1

# 或者通过代码调用
python -c "
from src.hipporag.document_processor import DocumentProcessor
processor = DocumentProcessor(context_lines=15)
result = processor.process_directory(
    '/root/code/docs/zh-cn',
    'outputs/Harmony_docs_zh_cn/markdown_parse/structure.json'
)
"
```

**输出路径**: `outputs/Harmony_docs_zh_cn/markdown_parse/structure.json`

**输出 JSON 结构示例**:
```json
{
  "file-abc123...": {
    "abstract": "",
    "file_path": "/path/to/document.md",
    "content": "完整文件内容...",
    "chunks": {
      "chunk-def456...": {
        "abstract": "",
        "content": "段落内容...",
        "metadata": {"h1": "标题1", "h2": "子标题"},
        "jump": {},
        "images": {},
        "codes": {
          "code-ghi789...": {
            "abstract": "",
            "content": "代码内容..."
          }
        },
        "tables": {},
        "filter_chunk": {
          "content": "过滤后的纯文本内容",
          "extracted_entities": [],
          "extracted_triples": []
        },
        "chunks": { /* 嵌套子段落 */ }
      }
    }
  }
}
```

### 第二步：摘要生成

使用 LLM 为各层次内容生成摘要。

**核心代码位置**: `generate_abstracts.py`, `src/hipporag/abstract_generator.py`

**执行命令**:
```bash
python generate_abstracts.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/structure.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/abstract.json \
    --backup outputs/Harmony_docs_zh_cn/markdown_parse_bak/abstract.json \
    --max-workers 20
```

**参数说明**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_file` | 必填 | 输入的 JSON 结构文件 |
| `output_file` | 必填 | 输出文件路径 |
| `--max-workers` | 20 | 并发线程数 |
| `--backup` | 无 | 备份文件，调用失败时复用已有摘要 |
| `--disable-fallback` | False | 禁用备用 LLM 配置 |
| `--dry-run` | False | 只分析任务，不实际生成 |

### 第三步：图片描述生成

使用多模态 LLM 为图片生成描述（caption）。

**核心代码位置**: `generate_image_captions.py`, `retry_failed_captions.py`

**执行命令**:
```bash
# 第一轮：生成图片caption
python generate_image_captions.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/abstract.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions.json \
    --backup outputs/Harmony_docs_zh_cn/markdown_parse_bak/abstract.json \
    --max-workers 10

# 第二轮：重试失败的图片
python retry_failed_captions.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions_final.json
```

**参数说明**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_file` | 必填 | 输入的 JSON 文件（包含摘要） |
| `output_file` | 必填 | 输出文件路径 |
| `--backup` | 无 | 备份文件，调用失败时复用已有caption |
| `--max-workers` | 10 | 并发线程数 |

### 第四步：实体和三元组提取

使用 OpenIE 从 `filter_chunk` 中提取实体和关系三元组。

**核心代码位置**: `extract_entities_triples.py`, `src/hipporag/information_extraction/`

**执行命令**:
```bash
python extract_entities_triples.py \
    outputs/Harmony_docs_zh_cn/markdown_parse/with_captions_final.json \
    outputs/Harmony_docs_zh_cn/markdown_parse/triples.json \
    --batch-size 200 \
    --openie-mode online
```

**参数说明**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_file` | 必填 | 输入的 JSON 文件（包含图片描述） |
| `output_file` | 必填 | 输出文件路径 |
| `--batch-size` | 200 | 批处理大小 |
| `--openie-mode` | online | OpenIE 模式：`online` 或 `offline` |
| `--llm-name` | gpt-3.5-turbo | LLM 模型名称 |
| `--llm-base-url` | 无 | 自定义 API 端点 |
| `--disable-fallback` | False | 禁用备用配置 |

**提取结果示例**:
```json
{
  "filter_chunk": {
    "content": "ArkTS 是 OpenHarmony 的主要开发语言...",
    "extracted_entities": ["ArkTS", "OpenHarmony", "开发语言"],
    "extracted_triples": [
      ["ArkTS", "是...的主要开发语言", "OpenHarmony"]
    ]
  }
}
```

### 第五步：知识图谱构建与索引

使用 `HippoRAG.index_from_json()` 构建完整的知识图谱。

**核心代码位置**: `src/hipporag/HippoRAG.py` 中的 `index_from_json` 方法

**构建内容**:

1. **多类型节点向量化**:
   - File 节点：文件摘要嵌入
   - Chunk 节点：段落摘要嵌入
   - Code 节点：代码摘要嵌入
   - Table 节点：表格摘要嵌入
   - Image 节点：图片描述嵌入
   - Entity 节点：实体名称嵌入
   - Fact 节点：三元组嵌入

2. **图结构边构建**:
   - **结构边**: File → Chunk, Chunk → 子Chunk, Chunk → Code/Table/Image
   - **语义边**: Entity ↔ Entity（通过三元组关系）
   - **同义词边**: 相似实体间的连接（基于向量相似度）
   - **超链接边**: 文档间的跳转关系

**完整建图脚本示例**:
```python
import json
import sys
sys.path.append('src')

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# 配置
cfg = BaseConfig()
cfg.save_dir = 'outputs/your_project'
cfg.llm_name = 'deepseek-v3.2-exp'
cfg.llm_base_url = 'https://api.modelarts-maas.com/openai/v1'
cfg.embedding_model_name = 'Qwen3-Embedding-4B'

# 初始化
hipporag = HippoRAG(global_config=cfg)

# 加载 JSON 结构
with open('output_with_triples.json', 'r') as f:
    json_structure = json.load(f)

# 构建索引
hipporag.index_from_json(json_structure)
```

---

## 检索流程（4阶段）

HippoRAG v2 使用 4 阶段检索流程：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         阶段1: Embedding 候选获取                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │   Fact 检索  │  │   File 检索  │  │  Chunk 检索  │                     │
│  │  (100个候选) │  │  (100个候选) │  │  (100个候选) │                     │
│  └─────────────┘  └─────────────┘  └─────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         阶段2: Rerank 精排                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │ Fact → 50个  │  │ File → 50个  │  │ Chunk → 50个 │                     │
│  └─────────────┘  └─────────────┘  └─────────────┘                      │
│  使用 TransformersCrossEncoder (bge-reranker-v2-m3) 本地精排             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         阶段3: 图谱扩散                                  │
│  基于知识图谱进行广度优先扩散，从精排后的节点出发：                         │
│  • 沿结构边扩散：Chunk → Code/Table/Image                                │
│  • 沿语义边扩散：Entity → 相关 Entity → 关联 Chunk                        │
│  • 扩散参数控制：时间预算、最大扩展数、邻居限制                             │
│                                                                          │
│  输出：Chunk(100), Code(5), Table(5), Image(5)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         阶段4: 最终精排 + 合并输出                        │
│  使用 LLM 对扩散结果进行最终精排：                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Chunk → 10个 │  │ Code → 2个  │  │ Table → 2个  │  │ Image → 2个  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                                          │
│  合并输出：格式化为可展示的 Markdown 文本                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 检索参数说明

**阶段1 参数（候选数量）**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fact_candidate_k` | 100 | Fact 向量检索的候选数量 |
| `file_candidate_k` | 100 | File 向量检索的候选数量 |
| `chunk_candidate_k` | 100 | Chunk 向量检索的候选数量 |

**阶段2 参数（精排保留数量）**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fact_top_k` | 50 | Fact 精排后保留数量 |
| `file_top_k` | 50 | File 精排后保留数量 |
| `chunk_top_k` | 50 | Chunk 精排后保留数量 |

**阶段3 参数（图扩散）**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `spread_chunk_k` | 100 | 扩散后 Chunk 数量上限 |
| `spread_code_k` | 5 | 扩散后 Code 数量上限 |
| `spread_table_k` | 5 | 扩散后 Table 数量上限 |
| `spread_image_k` | 5 | 扩散后 Image 数量上限 |
| `spread_time_budget_s` | 2.0 | 扩散时间预算（秒） |
| `spread_max_expansions` | 12000 | 最大扩展节点数 |
| `spread_per_node_neighbor_limit` | 96 | 每节点最大扩展邻居数 |
| `spread_max_frontier_size` | 20000 | 优先队列最大长度 |
| `spread_min_enqueue_score` | 0.01 | 入队最小分数阈值 |

**阶段4 参数（最终输出）**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `final_chunk_k` | 10 | 最终返回的 Chunk 数量 |
| `final_code_k` | 2 | 最终返回的 Code 数量 |
| `final_table_k` | 2 | 最终返回的 Table 数量 |
| `final_image_k` | 2 | 最终返回的 Image 数量 |
| `generate_report` | False | 是否生成 LLM 整合报告 |

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+ (GPU 加速)
- 足够的显存（建议 ≥24GB）

### 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

### 启动服务

**1. 启动 HippoRAG 服务（需要预加载模型，较慢）**

```bash
cd /root/code/HippoRAG-main/OpenHarmonyAssistant/chatbox
python hipporag_service.py --port 8001
```

等待看到 `✅ HippoRAG 初始化完成` 后服务就绑定了。

**2. 启动 API 网关（快速）**

```bash
# 新开一个终端
cd /root/code/HippoRAG-main/OpenHarmonyAssistant/chatbox
python server_text.py --server --port 8000
```

**3. 访问前端**

浏览器打开 `http://localhost:8000/`

---

## API 使用文档

### API 端点概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 返回前端页面 |
| `/chat` | POST | 问答接口（RAG + LLM） |
| `/retrieve` | POST | 单独检索接口 |
| `/health` | GET | 健康检查 |

### /chat 接口

**基本用法**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何创建一个ArkTS页面？",
    "use_rag": true,
    "use_llm": true
  }'
```

**只检索，不调用 LLM**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何创建一个ArkTS页面？",
    "use_rag": true,
    "use_llm": false
  }'
```

**自定义检索参数**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何创建一个ArkTS页面？",
    "use_rag": true,
    "use_llm": true,
    "retrieval_config": {
      "final_chunk_k": 5,
      "final_code_k": 3,
      "final_table_k": 2,
      "final_image_k": 2
    }
  }'
```

**完整参数示例**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何创建一个ArkTS页面？",
    "use_rag": true,
    "use_llm": false,
    "retrieval_config": {
      "fact_candidate_k": 100,
      "file_candidate_k": 100,
      "chunk_candidate_k": 100,
      "fact_top_k": 50,
      "file_top_k": 50,
      "chunk_top_k": 50,
      "spread_chunk_k": 100,
      "spread_code_k": 5,
      "spread_table_k": 5,
      "spread_image_k": 5,
      "final_chunk_k": 10,
      "final_code_k": 2,
      "final_table_k": 2,
      "final_image_k": 2,
      "verbose": true
    }
  }'
```

### /retrieve 接口

**基本用法**:
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何使用@State装饰器？"
  }'
```

**自定义参数**:
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何使用@State装饰器？",
    "retrieval_config": {
      "final_chunk_k": 5,
      "final_code_k": 3
    }
  }'
```

### 直接调用 HippoRAG 服务

```bash
# 使用默认参数
curl -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何创建一个ArkTS页面？"
  }'

# 自定义参数
curl -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何创建一个ArkTS页面？",
    "final_chunk_k": 5,
    "final_code_k": 3,
    "final_table_k": 2,
    "final_image_k": 2,
    "verbose": true
  }'
```

### 请求参数说明

#### ChatRequest 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | string | 必填 | 查询问题 |
| `use_rag` | bool | true | 是否启用 RAG 检索 |
| `use_llm` | bool | true | 是否调用 LLM 生成回答 |
| `retrieval_config` | object | null | 检索参数配置（可选） |

#### retrieval_config 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fact_candidate_k` | 100 | 阶段1：事实候选数量 |
| `file_candidate_k` | 100 | 阶段1：文件候选数量 |
| `chunk_candidate_k` | 100 | 阶段1：段落候选数量 |
| `fact_top_k` | 50 | 阶段2：精排后保留的事实数 |
| `file_top_k` | 50 | 阶段2：精排后保留的文件数 |
| `chunk_top_k` | 50 | 阶段2：精排后保留的段落数 |
| `spread_chunk_k` | 100 | 阶段3：扩散后的段落数 |
| `spread_code_k` | 5 | 阶段3：扩散后的代码数 |
| `spread_table_k` | 5 | 阶段3：扩散后的表格数 |
| `spread_image_k` | 5 | 阶段3：扩散后的图片数 |
| `final_chunk_k` | 10 | 阶段4：最终返回的段落数 |
| `final_code_k` | 2 | 阶段4：最终返回的代码数 |
| `final_table_k` | 2 | 阶段4：最终返回的表格数 |
| `final_image_k` | 2 | 阶段4：最终返回的图片数 |
| `generate_report` | false | 是否生成 LLM 整合报告 |
| `verbose` | true | 是否输出详细日志 |

### 响应格式

#### ChatResponse

```json
{
  "query": "如何创建一个ArkTS页面？",
  "rag_context": "# 相关文档段落 1\n...",
  "llm_response": "要创建一个ArkTS页面，您需要...",
  "rag_used": true,
  "llm_used": true,
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
  },
  "answer": "要创建一个ArkTS页面，您需要...",
  "timing_rag": 7.59,
  "timing_llm": 5.67
}
```

#### RetrieveResponse

```json
{
  "query": "如何使用@State装饰器？",
  "context": "# 相关文档段落 1\n...",
  "chunks_count": 10,
  "codes_count": 2,
  "tables_count": 2,
  "images_count": 2,
  "timing": 7.59
}
```

#### HealthResponse

```json
{
  "status": "ok",
  "hipporag_service_url": "http://localhost:8001",
  "hipporag_available": true,
  "timestamp": "2025-12-25T16:30:00.000000"
}
```

### 格式化输出（使用 jq）

```bash
# 安装 jq
apt install jq

# 格式化 JSON 输出
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "如何创建一个ArkTS页面？", "use_rag": true, "use_llm": false}' \
  | jq .

# 只看耗时信息
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "如何创建一个ArkTS页面？", "use_rag": true, "use_llm": false}' \
  | jq '.timing'

# 只看 LLM 回答
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "如何创建一个ArkTS页面？", "use_rag": true, "use_llm": true}' \
  | jq -r '.llm_response'
```

---

## 配置参数详解

### BaseConfig 核心配置

**位置**: `src/hipporag/utils/config_utils.py`

#### LLM 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `llm_name` | gpt-4o-mini | LLM 模型名称 |
| `llm_base_url` | None | LLM API 基础 URL |
| `max_new_tokens` | 2048 | 每次推理最大新 Token 数 |
| `temperature` | 0 | 采样温度 |
| `response_format` | {"type": "json_object"} | 响应格式 |

#### Embedding 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_model_name` | nvidia/NV-Embed-v2 | 嵌入模型名称 |
| `embedding_batch_size` | 10000 | 嵌入批处理大小 |
| `embedding_max_seq_len` | 2048 | 最大序列长度 |
| `embedding_return_as_normalized` | True | 是否归一化嵌入 |

#### 图构建配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `synonymy_edge_topk` | 2047 | 同义词边 KNN 检索 k 值 |
| `synonymy_edge_sim_threshold` | 0.95 | 同义词边相似度阈值 |
| `is_directed_graph` | False | 是否为有向图 |

#### 检索配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `linking_top_k` | 5 | 每步检索的链接节点数 |
| `rerank_candidate_k` | 50 | 重排序候选数量 |
| `retrieval_top_k` | 200 | 每步检索的文档数 |
| `damping` | 0.5 | PPR 阻尼因子 |

#### 存储配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `save_dir` | None | 保存目录 |
| `save_openie` | True | 是否保存 OpenIE 结果 |
| `force_index_from_scratch` | False | 强制重新构建索引 |
| `force_openie_from_scratch` | False | 强制重新进行 OpenIE |

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HIPPORAG_SERVICE_URL` | http://localhost:8001 | HippoRAG 服务地址 |
| `CHAT_MODEL_NAME` | qwen3-coder-480b-a35b-instruct | Chat LLM 模型名称 |
| `CHAT_BASE_URL` | https://api.modelarts-maas.com/v2 | Chat LLM API 地址 |
| `OPENAI_API_KEY` | - | OpenAI API Key |
| `RERANK_BACKEND` | transformers | Rerank 后端 |
| `RERANK_MODEL_NAME` | bge-reranker-v2-m3 路径 | Rerank 模型名称 |
| `FINAL_RERANK_BACKEND` | llm | 最终 Rerank 后端 |

---

## 项目结构

```
HippoRAG-main/
├── src/hipporag/                    # 核心代码
│   ├── HippoRAG.py                  # 主类：知识图谱构建和检索
│   ├── document_processor.py        # 文档解析器
│   ├── abstract_generator.py        # 摘要生成器
│   ├── embedding_store_v2.py        # 向量嵌入存储
│   ├── rerank.py                    # DSPy Rerank
│   ├── information_extraction/      # 信息抽取模块
│   │   ├── openie_openai.py         # 在线 OpenIE
│   │   └── openie_vllm_offline.py   # 离线 OpenIE
│   ├── embedding_model/             # 嵌入模型
│   ├── llm/                         # LLM 接口
│   ├── rerankers/                   # Rerank 模型
│   ├── prompts/                     # 提示模板
│   └── utils/                       # 工具函数
│       ├── config_utils.py          # 配置管理
│       └── misc_utils.py            # 杂项工具
│
├── OpenHarmonyAssistant/chatbox/    # Web 服务
│   ├── hipporag_service.py          # HippoRAG 服务（端口 8001）
│   ├── server_text.py               # API 网关（端口 8000）
│   ├── frontend.html                # 前端页面
│   └── API_README.md                # API 文档
│
├── extract_entities_triples.py      # 实体三元组提取脚本
├── generate_abstracts.py            # 摘要生成脚本
├── generate_image_captions.py       # 图片描述生成脚本
├── markdown_parser.py               # Markdown 解析器
│
├── bge-reranker-v2-m3/              # 本地 Rerank 模型
├── outputs/                         # 输出目录
│   └── Harmony_docs_zh_cn/          # OpenHarmony 文档索引
│
├── requirements.txt                 # Python 依赖
├── setup.py                         # 安装配置
└── README.md                        # 本文档
```

---

## 常见问题

### 1. HippoRAG 服务启动失败

**问题**: 显存不足或模型加载失败

**解决方案**:
```bash
# 检查 GPU 状态
nvidia-smi

# 确保 CUDA 可用
python -c "import torch; print(torch.cuda.is_available())"

# 如果显存不足，可以尝试减少 tensor_parallel_size
```

### 2. 检索结果为空

**问题**: 索引未正确构建

**解决方案**:
```bash
# 检查索引目录是否存在
ls -la outputs/Harmony_docs_zh_cn/

# 确认 embedding 文件存在
ls -la outputs/Harmony_docs_zh_cn/deepseek-v3.2-exp_Qwen3-Embedding-4B/
```

### 3. API 调用超时

**问题**: 检索或 LLM 调用时间过长

**解决方案**:
- 减少检索参数中的 k 值
- 确保 HippoRAG 服务已完成初始化
- 检查网络连接和 API 可用性

### 4. 如何添加新文档

**流程**:
1. 将新 Markdown 文件放入文档目录
2. 运行 `DocumentProcessor.process_directory()` 生成 JSON
3. 运行 `generate_abstracts.py` 生成摘要
4. 运行 `extract_entities_triples.py` 提取实体和三元组
5. 调用 `HippoRAG.index_from_json()` 重新构建索引

---

## 维护者须知

### 核心文件说明

1. **`HippoRAG.py`**: 7900+ 行，包含所有核心逻辑，修改需谨慎
2. **`document_processor.py`**: 文档解析逻辑，修改会影响 JSON 结构
3. **`server_text.py`**: API 网关，修改会影响所有 API 行为
4. **`hipporag_service.py`**: 检索服务，修改会影响检索结果

### 添加新功能建议

1. 新的检索参数：在 `RetrieveRequest` 和 `retrieve_v2` 中添加
2. 新的内容类型：需要修改 `index_from_json` 和 `retrieve_v2`
3. 新的 API 端点：在 `server_text.py` 中添加

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 单独测试检索
from src.hipporag import HippoRAG
hipporag = HippoRAG(global_config=cfg)
hipporag.prepare_retrieval_objects()
result = hipporag.retrieve_v2(["测试查询"], verbose=True)
```

---

## 许可证

请参阅 [LICENSE](LICENSE) 文件。

---

## 更新日志

- **v2.1**: 添加层次化 JSON 索引支持
- **v2.0**: 实现 4 阶段检索流程
- **v1.0**: 基础 HippoRAG 实现

