## 常见 Rerank 模型（2024~2025 RAG 实战里最常见的一类）

本项目的“普通 rerank”指 **Cross-Encoder / Reranker**：输入为 \((query, doc)\) 对，输出相关性分数，用于对候选集合做重排。它通常比 LLM prompt rerank 快、便宜、稳定（但效果取决于模型与领域）。

### 开源（Transformers，本地推理）

- **`BAAI/bge-reranker-v2-m3`（推荐默认，多语种/中文强）**  
  - 特点：多语种覆盖好，中文效果稳定；RAG里常见“Embedding + Reranker”组合之一。  
  - 代价：模型较大，CPU会慢，建议GPU。

- **`BAAI/bge-reranker-large` / `BAAI/bge-reranker-base`**  
  - 特点：经典 bge reranker 系列；base更轻、速度更快但效果略弱。  
  - 适合：英文占比高或对吞吐更敏感时。

- **`jinaai/jina-reranker-v2-base-multilingual`**  
  - 特点：多语种，工程落地较多。  
  - 适合：中英混合且希望模型体量/速度折中。

- **`mixedbread-ai/mxbai-rerank-large-v1`**  
  - 特点：英文 RAG 场景口碑不错。  
  - 适合：英文文档为主、追求更强排序质量。

### 商用 API（不在本改造范围内，但可对比）

- **Cohere Rerank（`rerank-*`）**  
  - 特点：服务化稳定、易用；质量较强但有调用成本与网络开销。

### 选型建议（非常简化版）

- **中文/多语种优先**：`BAAI/bge-reranker-v2-m3`  
- **更轻更快**：`BAAI/bge-reranker-base` 或 `jinaai/jina-reranker-v2-base-multilingual`  
- **英文优先且追求效果**：`mixedbread-ai/mxbai-rerank-large-v1`

### 本项目如何切换

在 `OpenHarmonyAssistant/chatbox/server_text.py` 中通过环境变量切换：

- `RERANK_BACKEND=transformers|none`（本次改造已禁用 LLM rerank）
- `RERANK_MODEL_NAME=...`
- `RERANK_DEVICE=auto|cpu|cuda`
- `RERANK_BATCH_SIZE=32`
- `RERANK_MAX_LENGTH=512`



