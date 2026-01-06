"""
RAG + LLM API 网关服务
功能：
1. 连接独立的 HippoRAG 服务进行检索
2. 调用 LLM 生成回答
3. 分页显示检索结果和LLM响应
4. 显示4阶段耗时详情

架构：
    前端 (frontend.html) 
        ↓
    API网关 (server_text.py, 端口 8000) 
        ↓
    HippoRAG服务 (hipporag_service.py, 端口 8001)
"""

import os
import re
import sys
import asyncio
import datetime
import requests
from typing import Optional, List, Dict, Any

import loguru
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = loguru.logger

# LLM 配置
TEMPERATURE = 0.7

# HippoRAG 服务配置
HIPPORAG_SERVICE_URL = os.environ.get("HIPPORAG_SERVICE_URL", "http://localhost:8001")

# ==================== 清除代理设置 ====================
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

# ==================== 配置 ====================

# Chat LLM 配置
LLM_CONFIG = {
    'chat_model_name': os.environ.get("CHAT_MODEL_NAME", "qwen3-coder-480b-a35b-instruct"),
    'chat_base_url': os.environ.get("CHAT_BASE_URL", "https://api.modelarts-maas.com/v2"),
}

# 系统提示词
SYSTEM_PROMPT = """你是 OpenHarmony 智能助手，请根据提供的上下文信息准确回答用户问题。

回答要求：
1. 基于提供的上下文进行回答，不要编造信息
2. 如果上下文中没有相关信息，请明确告知
3. 回答应该清晰、有条理
4. 如果涉及代码，请提供完整的代码示例
5. 使用中文回答"""

def get_api_key() -> str:
    """读取 API Key"""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MAAS_API_KEY")
    if not api_key:
        api_key = "7cL3yLtnQ09_nvQtdlOJVQeJLgjy9O7wfTgb31NNbuAsB_xvDfitBqbYSyKsqCPemEo-n4oH_S2WA6IApbaB8g"
        logger.warning("未检测到 OPENAI_API_KEY/MAAS_API_KEY，使用内置默认Key")
    return api_key


def check_hipporag_service() -> bool:
    """检查 HippoRAG 服务是否可用"""
    try:
        resp = requests.get(f"{HIPPORAG_SERVICE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ==================== LLM 调用 ====================

async def call_llm_async(messages: List[Dict]) -> Optional[str]:
    """异步调用 LLM"""
    api_key = get_api_key()
    model_name = LLM_CONFIG['chat_model_name']
    base_url = LLM_CONFIG['chat_base_url']
    api_url = base_url.rstrip("/") + "/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    
    def _do_request():
        logger.info(f"[LLM] 调用模型: {model_name}")
        resp = requests.post(api_url, headers=headers, json=payload, timeout=600, verify=False)
        resp.raise_for_status()
        return resp.json()
    
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _do_request)
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"[LLM] 调用失败: {e}")
        return None


def call_llm_sync(messages: List[Dict]) -> Optional[str]:
    """同步调用 LLM"""
    api_key = get_api_key()
    model_name = LLM_CONFIG['chat_model_name']
    base_url = LLM_CONFIG['chat_base_url']
    api_url = base_url.rstrip("/") + "/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    
    try:
        logger.info(f"[LLM] 调用模型: {model_name}")
        resp = requests.post(api_url, headers=headers, json=payload, timeout=600, verify=False)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"[LLM] 调用失败: {e}")
        return None


def build_llm_prompt(query: str, rag_context: str) -> List[Dict]:
    """构建LLM调用的消息"""
    if rag_context:
        user_content = f"""【参考上下文】:
{rag_context}

【用户问题】:
{query}"""
    else:
        user_content = query
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ==================== RAG 检索（调用 HippoRAG 服务）====================

# 默认检索配置
DEFAULT_RETRIEVAL_CONFIG = {
    "fact_candidate_k": 300,
    "file_candidate_k": 100,
    "chunk_candidate_k": 300,
    "fact_top_k": 10,
    "file_top_k": 10,
    "chunk_top_k": 30,
    "spread_chunk_k": 80,
    "spread_code_k": 5,
    "spread_table_k": 5,
    "spread_image_k": 5,
    "final_chunk_k": 10,
    "final_code_k": 2,
    "final_table_k": 2,
    "final_image_k": 2,
    "generate_report": False,
    "verbose": True
}


def retrieve_rag_context(query: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    调用 HippoRAG 服务进行检索
    
    Args:
        query: 查询文本
        config: 可选的检索配置，未提供的参数使用默认值
    """
    try:
        # 合并配置：默认值 + 用户配置
        payload = {"query": query, **DEFAULT_RETRIEVAL_CONFIG}
        if config:
            payload.update(config)
        
        resp = requests.post(
            f"{HIPPORAG_SERVICE_URL}/retrieve",
            json=payload,
            timeout=300  # 5分钟超时
        )
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("success"):
            return data.get("result")
        else:
            logger.error(f"[RAG] HippoRAG 服务返回错误: {data.get('error')}")
            return None
    except requests.exceptions.ConnectionError:
        logger.error(f"[RAG] 无法连接到 HippoRAG 服务: {HIPPORAG_SERVICE_URL}")
        logger.error("[RAG] 请确保 HippoRAG 服务已启动: python hipporag_service.py --port 8001")
        return None
    except Exception as e:
        logger.error(f"[RAG] 检索失败: {e}")
        return None


def increase_heading_level(content: str) -> str:
    """将内容开头的 markdown 标题层级 +1（每个 # 前面再加一个 #）
    
    只处理开头连续的标题行，一旦遇到非标题的正文内容，后面的行都保持原样。
    这样可以避免误处理正文中的代码注释。
    """
    lines = content.split('\n')
    result = []
    found_content = False  # 是否已遇到正文内容
    heading_pattern = re.compile(r'^(#{1,6})(\s+)')
    
    for line in lines:
        if found_content:
            # 已经遇到正文，后面的都不处理
            result.append(line)
        elif heading_pattern.match(line):
            # 是标题行，增加层级
            result.append(heading_pattern.sub(r'#\1\2', line))
        elif line.strip() == '':
            # 空行，继续检查后面的行
            result.append(line)
        else:
            # 遇到第一个非空非标题行（正文开始），停止处理
            found_content = True
            result.append(line)
    
    return '\n'.join(result)


def format_rag_context(rag_result: Dict[str, Any]) -> str:
    """将 RAG 检索结果格式化为可直接展示的合并文本（不调用LLM）"""
    if not rag_result:
        return ""

    logger.info("[RAG] 合并输出 chunks/codes/tables/images")
    context_parts = []
    
    # Chunks
    chunks = rag_result.get('chunks', {})
    chunk_ids = chunks.get('ids', [])
    chunk_contents = chunks.get('contents', [])
    
    chunk_metas = chunks.get('metadata', [])
    if chunk_contents:
        # context_parts.append("## 相关文档段落\n")
        for i, (chunk_id, content) in enumerate(zip(chunk_ids, chunk_contents)):
            meta = chunk_metas[i] if i < len(chunk_metas) else {}
            file_path = meta.get("file_path", "")
            breadcrumb = meta.get("breadcrumb", "")
            content_preview = content[:2000] if len(content) > 2000 else content
            content_preview = increase_heading_level(content_preview)
            context_parts.append(f"# 相关文档段落 {i+1}\n- ID: {chunk_id}\n- 文件: {file_path}\n- 定位: {breadcrumb}\n\n{content_preview}\n")
    
    # Codes
    codes = rag_result.get('codes', {})
    code_ids = codes.get('ids', [])
    code_contents = codes.get('contents', [])
    code_metas = codes.get('metadata', [])
    
    if code_contents:
        # context_parts.append("\n## 相关代码\n")
        for i, (code_id, content) in enumerate(zip(code_ids, code_contents)):
            meta = code_metas[i] if i < len(code_metas) else {}
            file_path = meta.get("file_path", "")
            breadcrumb = meta.get("breadcrumb", "")
            content_preview = content[:1500] if len(content) > 1500 else content
            content_preview = increase_heading_level(content_preview)
            context_parts.append(f"# 相关代码 {i+1}\n- ID: {code_id}\n- 文件: {file_path}\n- 定位: {breadcrumb}\n\n```\n{content_preview}\n```\n")
    
    # Tables
    tables = rag_result.get('tables', {})
    table_ids = tables.get('ids', [])
    table_contents = tables.get('contents', [])
    table_metas = tables.get('metadata', [])
    
    if table_contents:
        # context_parts.append("\n## 相关表格\n")
        for i, (table_id, content) in enumerate(zip(table_ids, table_contents)):
            meta = table_metas[i] if i < len(table_metas) else {}
            file_path = meta.get("file_path", "")
            breadcrumb = meta.get("breadcrumb", "")
            content_preview = content[:1000] if len(content) > 1000 else content
            content_preview = increase_heading_level(content_preview)
            context_parts.append(f"# 相关表格 {i+1}\n- ID: {table_id}\n- 文件: {file_path}\n- 定位: {breadcrumb}\n\n{content_preview}\n")

    # Images
    images = rag_result.get('images', {})
    image_ids = images.get('ids', [])
    image_contents = images.get('contents', [])
    image_metas = images.get('metadata', [])

    if image_ids:
        # context_parts.append("\n## 相关图片\n")
        for i, image_id in enumerate(image_ids):
            meta = image_metas[i] if i < len(image_metas) else {}
            gitee_url = meta.get("gitee_url", "") or meta.get("file_path", "")
            local_path = meta.get("local_path", "") or meta.get("absolute_path", "")
            md_file_path = meta.get("md_file_path", "")
            parent_chunk_id = meta.get("parent_chunk_id", "")
            breadcrumb = meta.get("breadcrumb", "")
            caption = image_contents[i] if i < len(image_contents) else ""
            context_parts.append(
                f"# 相关图片 {i+1}\n- ID: {image_id}\n- Gitee URL: {gitee_url}\n- 本地路径: {local_path}\n- 来源MD: {md_file_path}\n- 来源Chunk: {parent_chunk_id}\n- 定位: {breadcrumb}\n- Caption: {caption}\n"
            )
    
    return "\n".join(context_parts)


def log_rag_result(rag_result: Dict[str, Any], query: str):
    """打印 RAG 检索结果摘要"""
    if not rag_result:
        logger.warning(f"[RAG] 查询 '{query}' 无检索结果")
        return
    
    chunks = rag_result.get('chunks', {})
    codes = rag_result.get('codes', {})
    tables = rag_result.get('tables', {})
    images = rag_result.get('images', {})
    timing = rag_result.get('timing', {})
    
    print(f"\n{'='*60}")
    print(f"RAG 检索结果: {query}")
    print(f"{'='*60}")
    print(f"📄 Chunks: {len(chunks.get('ids', []))} 个")
    print(f"💻 Codes: {len(codes.get('ids', []))} 个")
    print(f"📊 Tables: {len(tables.get('ids', []))} 个")
    print(f"🖼️ Images: {len(images.get('ids', []))} 个")
    print(f"⏱️ 总耗时: {timing.get('total', 0):.2f}s")
    print(f"{'='*60}\n")


# ==================== FastAPI 应用 ====================

app = FastAPI(title="OpenHarmony RAG Chat API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 前端页面 ====================

from fastapi.responses import HTMLResponse, FileResponse

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        # 替换默认的 API 地址为当前服务地址
        html_content = html_content.replace(
            'value="http://localhost:8000"',
            'value=""'  # 空值表示使用当前页面的地址
        )
        # 修改 getApiUrl 函数，使其在地址为空时使用当前页面的 origin
        html_content = html_content.replace(
            "return apiUrlInput.value.trim().replace(/\\/$/, '');",
            "const url = apiUrlInput.value.trim(); return url ? url.replace(/\\/$/, '') : window.location.origin;"
        )
        return HTMLResponse(content=html_content)
    else:
        return HTMLResponse(content="<h1>frontend.html not found</h1><p>请确保 frontend.html 文件存在于同一目录</p>")


# ==================== 请求/响应模型 ====================

class RetrievalConfig(BaseModel):
    """检索参数配置（所有参数都有默认值）"""
    # 候选数量（阶段1）
    fact_candidate_k: int = 100
    file_candidate_k: int = 100
    chunk_candidate_k: int = 100
    # 精排后保留数量（阶段2）
    fact_top_k: int = 50
    file_top_k: int = 50
    chunk_top_k: int = 50
    # 扩散后保留数量（阶段3）
    spread_chunk_k: int = 100
    spread_code_k: int = 5
    spread_table_k: int = 5
    spread_image_k: int = 5
    # 最终返回数量（阶段4）
    final_chunk_k: int = 10
    final_code_k: int = 2
    final_table_k: int = 2
    final_image_k: int = 2
    # 其他
    generate_report: bool = False
    verbose: bool = True


class ChatRequest(BaseModel):
    query: str
    use_rag: bool = True
    use_llm: bool = True  # 是否调用LLM生成回答
    # 检索配置（可选，使用默认值）
    retrieval_config: Optional[RetrievalConfig] = None


class TimingInfo(BaseModel):
    """4阶段耗时详情"""
    stage1_embedding: float = 0.0    # 阶段1: Embedding检索
    stage2_rerank: float = 0.0       # 阶段2: Rerank精排
    stage3_spread: float = 0.0       # 阶段3: 图谱扩散
    stage4_final_rerank: float = 0.0 # 阶段4: 最终精排
    stage4_merge: float = 0.0        # 阶段4: 输出合并
    rag_total: float = 0.0           # RAG总耗时
    llm: float = 0.0                 # LLM生成耗时
    total: float = 0.0               # 总耗时


class ChatResponse(BaseModel):
    query: str
    # 检索结果（第一页）
    rag_context: str = ""            # 格式化的检索结果
    # LLM响应（第二页）
    llm_response: str = ""           # LLM生成的回答
    # 统计信息
    rag_used: bool = False
    llm_used: bool = False
    rag_chunks_count: int = 0
    rag_codes_count: int = 0
    rag_tables_count: int = 0
    rag_images_count: int = 0
    # 详细耗时
    timing: TimingInfo = TimingInfo()
    # 兼容旧字段
    answer: str = ""                 # 兼容：默认返回LLM响应
    timing_rag: float = 0.0
    timing_llm: float = 0.0


class RetrieveRequest(BaseModel):
    query: str
    # 检索配置（可选，使用默认值）
    retrieval_config: Optional[RetrievalConfig] = None


class RetrieveResponse(BaseModel):
    query: str
    context: str
    chunks_count: int = 0
    codes_count: int = 0
    tables_count: int = 0
    images_count: int = 0
    timing: float = 0.0


# ==================== 异步 RAG 检索 ====================

async def retrieve_rag_context_async(query: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """异步 RAG 检索"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, retrieve_rag_context, query, config)


# ==================== API 端点 ====================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
    """
    主接口：RAG检索 + LLM回答
    返回：
    - rag_context: 格式化的检索结果（第一页）
    - llm_response: LLM生成的回答（第二页）
    - timing: 4阶段详细耗时
    """
    import time
    
    rag_result = None
    rag_context = ""
    llm_response = ""
    
    # 初始化timing
    timing_info = TimingInfo()
    total_start = time.time()
    
    # Step 1: RAG 检索
    if body.use_rag:
        logger.info(f"[RAG] 开始检索: {body.query}")
        try:
            # 构建检索配置
            retrieval_config = body.retrieval_config.dict() if body.retrieval_config else None
            rag_result = await retrieve_rag_context_async(body.query, retrieval_config)
            if rag_result:
                log_rag_result(rag_result, body.query)
                rag_context = format_rag_context(rag_result)
                logger.info(f"[RAG] 检索完成，上下文长度: {len(rag_context)} 字符")
                
                # 提取RAG各阶段耗时
                rag_timing = rag_result.get('timing', {})
                timing_info.stage1_embedding = rag_timing.get('stage1_embedding', 0.0)
                timing_info.stage2_rerank = rag_timing.get('stage2_rerank', 0.0)
                timing_info.stage3_spread = rag_timing.get('stage3_spread', 0.0)
                timing_info.stage4_final_rerank = rag_timing.get('stage4_final_rerank', 0.0)
                timing_info.stage4_merge = rag_timing.get('stage4_merge_output', 0.0)
                timing_info.rag_total = rag_timing.get('total', 0.0)
        except Exception as e:
            logger.error(f"[RAG] 检索失败: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 2: LLM 生成回答
    if body.use_llm:
        logger.info(f"[LLM] 开始生成回答")
        llm_start = time.time()
        try:
            messages = build_llm_prompt(body.query, rag_context)
            llm_response = await call_llm_async(messages)
            if not llm_response:
                llm_response = ""
            logger.info(f"[LLM] 生成完成，回答长度: {len(llm_response)} 字符")
        except Exception as e:
            logger.error(f"[LLM] 生成失败: {e}")
            import traceback
            traceback.print_exc()
        timing_info.llm = time.time() - llm_start
    
    # 计算总耗时
    timing_info.total = time.time() - total_start
    
    # 统计 RAG 结果
    chunks_count = len(rag_result.get('chunks', {}).get('ids', [])) if rag_result else 0
    codes_count = len(rag_result.get('codes', {}).get('ids', [])) if rag_result else 0
    tables_count = len(rag_result.get('tables', {}).get('ids', [])) if rag_result else 0
    images_count = len(rag_result.get('images', {}).get('ids', [])) if rag_result else 0
    
    return ChatResponse(
        query=body.query,
        # 分页内容
        rag_context=rag_context,
        llm_response=llm_response,
        # 统计信息
        rag_used=body.use_rag and rag_result is not None,
        llm_used=body.use_llm and bool(llm_response),
        rag_chunks_count=chunks_count,
        rag_codes_count=codes_count,
        rag_tables_count=tables_count,
        rag_images_count=images_count,
        # 详细耗时
        timing=timing_info,
        # 兼容旧字段
        answer=llm_response or rag_context,
        timing_rag=timing_info.rag_total,
        timing_llm=timing_info.llm
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(body: RetrieveRequest):
    """
    单独的检索接口：只进行 RAG 检索，返回格式化后的上下文
    """
    import time
    
    rag_start = time.time()
    # 构建检索配置
    retrieval_config = body.retrieval_config.dict() if body.retrieval_config else None
    rag_result = await retrieve_rag_context_async(body.query, retrieval_config)
    timing = time.time() - rag_start
    
    if not rag_result:
        return RetrieveResponse(
            query=body.query,
            context="",
            timing=timing
        )
    
    log_rag_result(rag_result, body.query)
    context = format_rag_context(rag_result)
    
    return RetrieveResponse(
        query=body.query,
        context=context,
        chunks_count=len(rag_result.get('chunks', {}).get('ids', [])),
        codes_count=len(rag_result.get('codes', {}).get('ids', [])),
        tables_count=len(rag_result.get('tables', {}).get('ids', [])),
        images_count=len(rag_result.get('images', {}).get('ids', [])),
        timing=timing
    )


@app.get("/health")
async def health_check():
    """健康检查"""
    hipporag_available = check_hipporag_service()
    return {
        "status": "ok",
        "hipporag_service_url": HIPPORAG_SERVICE_URL,
        "hipporag_available": hipporag_available,
        "timestamp": datetime.datetime.now().isoformat()
    }


# ==================== 同步问答函数（命令行用）====================

def ask(query: str, use_rag: bool = True, use_llm: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """
    同步问答函数（用于命令行）
    返回：包含 rag_context, llm_response, timing 的字典
    """
    import time
    
    result = {
        'query': query,
        'rag_context': '',
        'llm_response': '',
        'timing': {
            'stage1_embedding': 0.0,
            'stage2_rerank': 0.0,
            'stage3_spread': 0.0,
            'stage4_final_rerank': 0.0,
            'rag_total': 0.0,
            'llm': 0.0,
            'total': 0.0
        }
    }
    
    total_start = time.time()
    rag_result = None
    
    # Step 1: RAG 检索
    if use_rag:
        logger.info(f"[RAG] 开始检索: {query}")
        try:
            rag_result = retrieve_rag_context(query, {"verbose": verbose})
            if rag_result:
                if verbose:
                    log_rag_result(rag_result, query)
                result['rag_context'] = format_rag_context(rag_result)
                logger.info(f"[RAG] 检索完成，上下文长度: {len(result['rag_context'])} 字符")
                
                # 提取各阶段耗时
                rag_timing = rag_result.get('timing', {})
                result['timing']['stage1_embedding'] = rag_timing.get('stage1_embedding', 0.0)
                result['timing']['stage2_rerank'] = rag_timing.get('stage2_rerank', 0.0)
                result['timing']['stage3_spread'] = rag_timing.get('stage3_spread', 0.0)
                result['timing']['stage4_final_rerank'] = rag_timing.get('stage4_final_rerank', 0.0)
                result['timing']['rag_total'] = rag_timing.get('total', 0.0)
        except Exception as e:
            logger.error(f"[RAG] 检索失败: {e}")
    
    # Step 2: LLM 生成回答
    if use_llm:
        logger.info(f"[LLM] 开始生成回答")
        llm_start = time.time()
        try:
            messages = build_llm_prompt(query, result['rag_context'])
            response = call_llm_sync(messages)
            result['llm_response'] = response if response else ""
            logger.info(f"[LLM] 生成完成，回答长度: {len(result['llm_response'])} 字符")
        except Exception as e:
            logger.error(f"[LLM] 生成失败: {e}")
        result['timing']['llm'] = time.time() - llm_start
    
    result['timing']['total'] = time.time() - total_start
    
    return result


def interactive_mode():
    """交互式问答模式"""
    print("\n" + "="*60)
    print("OpenHarmony RAG 问答系统")
    print("输入问题进行查询，输入 'quit' 或 'exit' 退出")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("\n🔍 请输入问题: ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            print("\n⏳ 正在检索并生成回答...\n")
            result = ask(query, use_rag=True, use_llm=True, verbose=True)
            
            # 显示耗时
            timing = result['timing']
            print("\n" + "="*60)
            print("⏱️ 各阶段耗时:")
            print(f"  阶段1 Embedding:  {timing['stage1_embedding']:.2f}s")
            print(f"  阶段2 Rerank:     {timing['stage2_rerank']:.2f}s")
            print(f"  阶段3 图谱扩散:   {timing['stage3_spread']:.2f}s")
            print(f"  阶段4 最终精排:   {timing['stage4_final_rerank']:.2f}s")
            print(f"  RAG 总耗时:       {timing['rag_total']:.2f}s")
            print(f"  LLM 生成:         {timing['llm']:.2f}s")
            print(f"  总耗时:           {timing['total']:.2f}s")
            print("="*60)
            
            # 显示LLM回答
            print("\n📝 LLM 回答:")
            print("-"*60)
            print(result['llm_response'] or "无回答")
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\n已中断，再见！")
            break
        except Exception as e:
            logger.error(f"发生错误: {e}")


# ==================== 启动入口 ====================

def main():
    """命令行入口"""
    global HIPPORAG_SERVICE_URL
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenHarmony RAG API 网关服务")
    parser.add_argument("query", nargs="?", help="查询问题")
    parser.add_argument("--server", action="store_true", help="启动 API 服务器")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口（默认 8000）")
    parser.add_argument("--hipporag-url", type=str, default=None, help="HippoRAG 服务地址（默认 http://localhost:8001）")
    parser.add_argument("--no-rag", action="store_true", help="禁用 RAG 检索")
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM 回答")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    
    args = parser.parse_args()
    
    # 更新 HippoRAG 服务地址
    if args.hipporag_url:
        HIPPORAG_SERVICE_URL = args.hipporag_url
    
    if args.server:
        # 启动 API 服务器
        import uvicorn
        print("\n" + "="*60)
        print("🚀 RAG API 网关服务")
        print("="*60)
        print(f"📡 API 地址: http://0.0.0.0:{args.port}")
        print(f"🔗 HippoRAG 服务: {HIPPORAG_SERVICE_URL}")
        print("-"*60)
        print("API 端点:")
        print(f"  POST /chat     - 问答接口")
        print(f"  POST /retrieve - 检索接口")
        print(f"  GET  /health   - 健康检查")
        print("-"*60)
        print("📄 前端页面请用浏览器打开: frontend.html")
        print("="*60)
        
        # 检查 HippoRAG 服务
        if check_hipporag_service():
            print("✅ HippoRAG 服务已连接")
        else:
            print("⚠️  警告: HippoRAG 服务未连接")
            print(f"   请先启动: python hipporag_service.py --port 8001")
        print("")
        
        uvicorn.run("server_text:app", host="0.0.0.0", port=args.port, reload=False)
    elif args.interactive:
        interactive_mode()
    elif args.query:
        result = ask(args.query, use_rag=not args.no_rag, use_llm=not args.no_llm, verbose=True)
        
        # 显示耗时
        timing = result['timing']
        print("\n" + "="*60)
        print("⏱️ 各阶段耗时:")
        print(f"  阶段1 Embedding:  {timing['stage1_embedding']:.2f}s")
        print(f"  阶段2 Rerank:     {timing['stage2_rerank']:.2f}s")
        print(f"  阶段3 图谱扩散:   {timing['stage3_spread']:.2f}s")
        print(f"  阶段4 最终精排:   {timing['stage4_final_rerank']:.2f}s")
        print(f"  RAG 总耗时:       {timing['rag_total']:.2f}s")
        print(f"  LLM 生成:         {timing['llm']:.2f}s")
        print(f"  总耗时:           {timing['total']:.2f}s")
        print("="*60)
        
        # 显示LLM回答
        print("\n📝 回答:")
        print("="*60)
        print(result['llm_response'] or result['rag_context'] or "无结果")
    else:
        # 默认启动服务器
        import uvicorn
        print("\n" + "="*60)
        print("🚀 RAG API 网关服务")
        print("="*60)
        print(f"📡 API 地址: http://0.0.0.0:8000")
        print(f"🔗 HippoRAG 服务: {HIPPORAG_SERVICE_URL}")
        print("-"*60)
        print("API 端点:")
        print(f"  POST /chat     - 问答接口")
        print(f"  POST /retrieve - 检索接口")
        print(f"  GET  /health   - 健康检查")
        print("-"*60)
        print("📄 前端页面请用浏览器打开: frontend.html")
        print("="*60)
        
        # 检查 HippoRAG 服务
        if check_hipporag_service():
            print("✅ HippoRAG 服务已连接")
        else:
            print("⚠️  警告: HippoRAG 服务未连接")
            print(f"   请先启动: python hipporag_service.py --port 8001")
        print("")
        
        uvicorn.run("server_text:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

