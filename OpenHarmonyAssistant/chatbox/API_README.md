# OpenHarmony RAG API 使用说明

## 📁 架构说明

```
前端 (frontend.html)
    ↓ HTTP
API网关 (server_text.py:8000) ──→ LLM API
    ↓ HTTP
HippoRAG服务 (hipporag_service.py:8001)
```

## 🚀 启动步骤

### 1. 启动 HippoRAG 服务（慢，只需启动一次）

```bash
cd /root/code/HippoRAG-main/OpenHarmonyAssistant/chatbox
python hipporag_service.py --port 8001
```

等待看到 `✅ HippoRAG 初始化完成` 后，服务就绑定了。

### 2. 启动 API 网关（快，可随时重启）

```bash
# 新开一个终端
cd /root/code/HippoRAG-main/OpenHarmonyAssistant/chatbox
python server_text.py --server --port 8000
```

### 3. 访问前端

- 浏览器打开 `http://localhost:8000/`
- 或直接打开 `frontend.html` 文件

---

## 📡 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 返回前端页面 |
| `/chat` | POST | 问答接口（RAG + LLM） |
| `/retrieve` | POST | 单独检索接口 |
| `/health` | GET | 健康检查 |

---

## 🔧 curl 使用示例

### 健康检查

```bash
# 检查 API 网关
curl http://localhost:8000/health

# 检查 HippoRAG 服务
curl http://localhost:8001/health
```

---

### /chat 接口

#### 基本用法（使用默认配置）

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何创建一个ArkTS页面？",
    "use_rag": true,
    "use_llm": true
  }'
```

#### 只检索，不调用 LLM

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何创建一个ArkTS页面？",
    "use_rag": true,
    "use_llm": false
  }'
```

#### 只调用 LLM，不检索

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是OpenHarmony？",
    "use_rag": false,
    "use_llm": true
  }'
```

#### 自定义检索参数

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

#### 完整参数示例

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

---

### /retrieve 接口

#### 基本用法

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何使用@State装饰器？"
  }'
```

#### 自定义参数

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

---

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

---

## 📊 格式化输出（使用 jq）

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

# 只看检索结果
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "如何创建一个ArkTS页面？", "use_rag": true, "use_llm": false}' \
  | jq -r '.rag_context'
```

---

## 📝 参数说明

### ChatRequest 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | string | 必填 | 查询问题 |
| `use_rag` | bool | true | 是否启用 RAG 检索 |
| `use_llm` | bool | true | 是否调用 LLM 生成回答 |
| `retrieval_config` | object | null | 检索参数配置（可选） |

### retrieval_config 参数

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
| `generate_report` | false | 是否生成报告 |
| `verbose` | true | 是否输出详细日志 |

---

## 📤 响应格式

### ChatResponse

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

### RetrieveResponse

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

### HealthResponse

```json
{
  "status": "ok",
  "hipporag_service_url": "http://localhost:8001",
  "hipporag_available": true,
  "timestamp": "2025-12-25T16:30:00.000000"
}
```

---

## ⚙️ 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HIPPORAG_SERVICE_URL` | http://localhost:8001 | HippoRAG 服务地址 |
| `CHAT_MODEL_NAME` | qwen3-coder-480b-a35b-instruct | LLM 模型名称 |
| `CHAT_BASE_URL` | https://api.modelarts-maas.com/v2 | LLM API 地址 |
| `OPENAI_API_KEY` | - | API Key |

---

## 🔗 命令行参数

### server_text.py

```bash
python server_text.py --help

# 参数：
#   query              查询问题（可选）
#   --server           启动 API 服务器
#   --port PORT        服务器端口（默认 8000）
#   --hipporag-url URL HippoRAG 服务地址
#   --no-rag           禁用 RAG 检索
#   --no-llm           禁用 LLM 回答
#   -i, --interactive  交互模式
```

### hipporag_service.py

```bash
python hipporag_service.py --help

# 参数：
#   --port PORT  服务端口（默认 8001）
#   --host HOST  监听地址（默认 0.0.0.0）
```



