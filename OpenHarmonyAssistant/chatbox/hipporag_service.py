"""
HippoRAG 独立服务
用于预加载 HippoRAG 模型，提供 HTTP API 供其他服务调用

启动方式：
    python hipporag_service.py --port 8001

API 端点：
    POST /retrieve - 执行检索
    GET  /health   - 健康检查
"""

import os
import sys
import time
import argparse
from typing import Optional, List, Dict, Any

import loguru
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = loguru.logger

# ==================== 清除代理设置 ====================
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

# ==================== 配置 ====================

HIPPORAG_CONFIG = {
    'save_dir': os.path.join(PROJECT_ROOT, 'outputs/Harmony_docs_zh_cn'),
    'llm_model_name': 'deepseek-v3.2-exp',
    'llm_base_url': 'https://api.modelarts-maas.com/openai/v1',
    'embedding_model_name': 'Qwen3-Embedding-4B',
    # rerank（阶段2使用本地transformers提速）
    'rerank_backend': os.environ.get("RERANK_BACKEND", "transformers"),
    'rerank_model_name': os.environ.get("RERANK_MODEL_NAME", "/root/code/HippoRAG-main/bge-reranker-v2-m3"),
    'rerank_device': os.environ.get("RERANK_DEVICE", "auto"),
    'rerank_batch_size': int(os.environ.get("RERANK_BATCH_SIZE", "32")),
    'rerank_max_length': int(os.environ.get("RERANK_MAX_LENGTH", "512")),
    # 阶段4最终rerank：使用LLM精排
    'final_rerank_backend': os.environ.get("FINAL_RERANK_BACKEND", "llm"),
}

# 全局 HippoRAG 实例
hipporag_instance = None
init_time = 0.0


def init_hipporag():
    """初始化 HippoRAG 实例"""
    global hipporag_instance, init_time
    
    if hipporag_instance is not None:
        return hipporag_instance
    
    logger.info("🔧 开始初始化 HippoRAG...")
    start_time = time.time()
    
    from src.hipporag import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig
    
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "BQm_Gkd1EoTcHkJfVf31dTWfMIOsW3_mKIDfM5j-MvvwNM5jNl9XnLOjvNjEOuDiIWoKb-DIphdRWt2gOoNwBw"
        logger.warning("未检测到 OPENAI_API_KEY，使用内置默认Key")

    cfg = BaseConfig()
    cfg.save_dir = HIPPORAG_CONFIG['save_dir']
    cfg.llm_name = HIPPORAG_CONFIG['llm_model_name']
    cfg.llm_base_url = HIPPORAG_CONFIG['llm_base_url']
    cfg.embedding_model_name = HIPPORAG_CONFIG['embedding_model_name']
    cfg.rerank_backend = HIPPORAG_CONFIG['rerank_backend']
    cfg.rerank_model_name = HIPPORAG_CONFIG['rerank_model_name']
    cfg.rerank_device = HIPPORAG_CONFIG['rerank_device']
    cfg.rerank_batch_size = HIPPORAG_CONFIG['rerank_batch_size']
    cfg.rerank_max_length = HIPPORAG_CONFIG['rerank_max_length']
    cfg.final_rerank_backend = HIPPORAG_CONFIG['final_rerank_backend']

    hipporag_instance = HippoRAG(global_config=cfg)
    init_time = time.time() - start_time
    
    logger.info(f"✅ HippoRAG 初始化完成，耗时: {init_time:.2f}s")
    return hipporag_instance


# ==================== FastAPI 应用 ====================

app = FastAPI(title="HippoRAG Service", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 请求/响应模型 ====================

class RetrieveRequest(BaseModel):
    query: str
    # 检索参数（可选，使用默认值）
    fact_candidate_k: int = 100
    file_candidate_k: int = 100
    chunk_candidate_k: int = 100
    fact_top_k: int = 50
    file_top_k: int = 50
    chunk_top_k: int = 50
    spread_chunk_k: int = 100
    spread_code_k: int = 5
    spread_table_k: int = 5
    spread_image_k: int = 5
    final_chunk_k: int = 10
    final_code_k: int = 2
    final_table_k: int = 2
    final_image_k: int = 2
    generate_report: bool = False
    verbose: bool = True


class RetrieveResponse(BaseModel):
    success: bool
    query: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ==================== API 端点 ====================

@app.get("/health")
async def health_check():
    """健康检查"""
    import datetime
    return {
        "status": "ok",
        "service": "hipporag",
        "hipporag_initialized": hipporag_instance is not None,
        "init_time": f"{init_time:.2f}s" if init_time > 0 else "N/A",
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(body: RetrieveRequest):
    """
    执行 HippoRAG 检索
    """
    import asyncio
    
    if hipporag_instance is None:
        return RetrieveResponse(
            success=False,
            query=body.query,
            error="HippoRAG 未初始化"
        )
    
    def _retrieve():
        try:
            results = hipporag_instance.retrieve_v2(
                queries=[body.query],
                fact_candidate_k=body.fact_candidate_k,
                file_candidate_k=body.file_candidate_k,
                chunk_candidate_k=body.chunk_candidate_k,
                fact_top_k=body.fact_top_k,
                file_top_k=body.file_top_k,
                chunk_top_k=body.chunk_top_k,
                spread_chunk_k=body.spread_chunk_k,
                spread_code_k=body.spread_code_k,
                spread_table_k=body.spread_table_k,
                spread_image_k=body.spread_image_k,
                final_chunk_k=body.final_chunk_k,
                final_code_k=body.final_code_k,
                final_table_k=body.final_table_k,
                final_image_k=body.final_image_k,
                generate_report=body.generate_report,
                verbose=body.verbose
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(f"检索失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _retrieve)
        return RetrieveResponse(
            success=True,
            query=body.query,
            result=result
        )
    except Exception as e:
        return RetrieveResponse(
            success=False,
            query=body.query,
            error=str(e)
        )


# ==================== 启动入口 ====================

def main():
    parser = argparse.ArgumentParser(description="HippoRAG 独立服务")
    parser.add_argument("--port", type=int, default=8001, help="服务端口（默认 8001）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    args = parser.parse_args()
    
    # 启动前先初始化 HippoRAG
    print("\n" + "="*60)
    print("🚀 HippoRAG 独立服务")
    print("="*60)
    print("\n⏳ 正在预加载 HippoRAG 模型（这可能需要几分钟）...\n")
    
    init_hipporag()
    
    print("\n" + "="*60)
    print(f"✅ 服务启动: http://{args.host}:{args.port}")
    print("="*60)
    print("API 端点:")
    print(f"  POST /retrieve - 执行检索")
    print(f"  GET  /health   - 健康检查")
    print("="*60 + "\n")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()

