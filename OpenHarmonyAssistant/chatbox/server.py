"""
OpenHarmony Assistant Server
整合 HippoRAG 检索 + LLM 问答的统一服务端
"""

import os
import sys
import datetime
import asyncio
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import requests
import loguru
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 延迟导入 HippoRAG，避免 multiprocessing spawn 冲突
if TYPE_CHECKING:
    from src.hipporag import HippoRAG

logger = loguru.logger

# ==================== 清除代理设置 ====================
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

# ==================== 读取前端页面 ====================

BASE_DIR = os.path.dirname(__file__)
FRONTEND_PATH = "/root/code/OpenHarmonyAssistant/chatbox/index.html"

try:
    with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
        HTML_PAGE = f.read()
except FileNotFoundError:
    HTML_PAGE = "<html><body><h1>前端页面未找到</h1></body></html>"
    logger.warning(f"前端页面未找到: {FRONTEND_PATH}")

# ==================== ModelArts MaaS / OpenAI兼容 配置 ====================
#
# 说明：
# - HippoRAG 内部（包括阶段2/4 rerank、阶段5素材选取）使用的是 OpenAI 兼容的 base_url + model_name
# - 为了让 server 的“最终回答模型调用”与 HippoRAG 对齐，这里也默认走 OpenAI 兼容接口

TEMPERATURE = 0.6

def get_api_key() -> str:
    """
    读取 API Key（优先 OpenAI 兼容的 OPENAI_API_KEY，其次 MAAS_API_KEY）。
    注意：仓库里历史上存在硬编码 key；这里保留向后兼容，但会输出告警。
    """
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MAAS_API_KEY")
    if not api_key:
        # 向后兼容：避免直接跑不起来（建议通过环境变量注入）
        api_key = "7cL3yLtnQ09_nvQtdlOJVQeJLgjy9O7wfTgb31NNbuAsB_xvDfitBqbYSyKsqCPemEo-n4oH_S2WA6IApbaB8g"
        logger.warning("未检测到 OPENAI_API_KEY/MAAS_API_KEY，正在使用内置默认Key（不推荐，建议改为环境变量注入）")
    return api_key

# ==================== HippoRAG 配置与初始化 ====================

HIPPORAG_CONFIG = {
    'save_dir': os.path.join(PROJECT_ROOT, 'outputs/Harmony_docs_zh_cn'),
    'llm_model_name': 'deepseek-v3.2-exp',
    'llm_base_url': 'https://api.modelarts-maas.com/openai/v1',
    'embedding_model_name': 'Qwen3-Embedding-4B',
}

# 全局 HippoRAG 实例
hipporag_instance = None

def get_hipporag():
    """获取或创建 HippoRAG 实例（懒加载单例）"""
    global hipporag_instance
    if hipporag_instance is None:
        logger.info("🔧 初始化 HippoRAG 实例...")
        # 延迟导入，避免 multiprocessing spawn 冲突
        from src.hipporag import HippoRAG
        
        # 尽量与 demo_openai.py 行为一致：HippoRAG 走 OpenAI兼容接口
        # 建议通过环境变量注入 OPENAI_API_KEY；若没有则保留旧逻辑兜底（并告警）
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "BQm_Gkd1EoTcHkJfVf31dTWfMIOsW3_mKIDfM5j-MvvwNM5jNl9XnLOjvNjEOuDiIWoKb-DIphdRWt2gOoNwBw"
            logger.warning("未检测到 OPENAI_API_KEY，正在使用内置默认Key（不推荐，建议改为环境变量注入）")
        
        hipporag_instance = HippoRAG(
            save_dir=HIPPORAG_CONFIG['save_dir'],
            llm_model_name=HIPPORAG_CONFIG['llm_model_name'],
            llm_base_url=HIPPORAG_CONFIG['llm_base_url'],
            embedding_model_name=HIPPORAG_CONFIG['embedding_model_name']
        )
        logger.info("✅ HippoRAG 初始化完成")
    return hipporag_instance

# ==================== RAG 检索函数 ====================

async def retrieve_rag_context(query: str, verbose: bool = True) -> Dict[str, Any]:
    """
    使用 HippoRAG 检索相关上下文（参考 demo_openai.py）
    
    Args:
        query: 用户查询
        verbose: 是否输出详细信息
    
    Returns:
        包含检索结果的字典，如果 generate_report=True 则包含 report 字段
    """
    def _do_retrieve():
        hipporag = get_hipporag()
        results = hipporag.retrieve_v2(
            queries=[query],
            # 阶段1参数
            fact_candidate_k=50,
            file_candidate_k=30,
            chunk_candidate_k=30,
            # 阶段2参数
            fact_top_k=10,
            file_top_k=10,
            chunk_top_k=10,
            # 阶段3参数
            spread_chunk_k=30,
            spread_code_k=10,
            spread_table_k=10,
            spread_image_k=10,
            # 阶段4参数
            final_chunk_k=10,
            final_code_k=3,
            final_table_k=3,
            final_image_k=3,
            # 阶段5: 开启LLM报告生成（参考 demo_openai.py）
            # 这会调用 _generate_report 生成 frontend_prompt
            generate_report=True,
            verbose=verbose
        )
        return results[0] if results else None
    
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _do_retrieve)

def log_rag_result(rag_result: Dict[str, Any], query: str):
    """打印 RAG 检索结果的详细日志"""
    if not rag_result:
        logger.warning(f"[RAG] 查询 '{query}' 无检索结果")
        return
    
    print(f"\n{'#'*70}")
    print(f"# RAG 检索结果汇总: {query}")
    print(f"{'#'*70}")
    
    # Chunks
    chunks = rag_result.get('chunks', {})
    chunk_ids = chunks.get('ids', [])
    chunk_contents = chunks.get('contents', [])
    print(f"\n📄 Chunk: {len(chunk_ids)} 个")
    for i, (cid, content) in enumerate(zip(chunk_ids[:5], chunk_contents[:5])):
        preview = content[:150].replace('\n', ' ') if content else ''
        print(f"  [{i+1}] {cid}")
        print(f"      {preview}...")
    
    # Codes
    codes = rag_result.get('codes', {})
    code_ids = codes.get('ids', [])
    code_contents = codes.get('contents', [])
    print(f"\n💻 Code: {len(code_ids)} 个")
    for i, (cid, content) in enumerate(zip(code_ids[:3], code_contents[:3])):
        preview = content[:100].replace('\n', ' ') if content else ''
        print(f"  [{i+1}] {cid}")
        print(f"      {preview}...")
    
    # Tables
    tables = rag_result.get('tables', {})
    table_ids = tables.get('ids', [])
    table_contents = tables.get('contents', [])
    print(f"\n📊 Table: {len(table_ids)} 个")
    for i, (tid, content) in enumerate(zip(table_ids[:3], table_contents[:3])):
        preview = content[:100].replace('\n', ' ') if content else ''
        print(f"  [{i+1}] {tid}")
        print(f"      {preview}...")
    
    # Images（包含URL信息）
    images = rag_result.get('images', {})
    image_ids = images.get('ids', [])
    image_contents = images.get('contents', [])
    image_metadata = images.get('metadata', [])
    print(f"\n🖼️ Image: {len(image_ids)} 个")
    for i, iid in enumerate(image_ids[:5]):
        meta = image_metadata[i] if i < len(image_metadata) else {}
        content = image_contents[i] if i < len(image_contents) else ''
        gitee_url = meta.get('gitee_url', '')
        caption = meta.get('caption', '')
        width = meta.get('width')
        height = meta.get('height')
        
        print(f"  [{i+1}] {iid}")
        if gitee_url:
            print(f"      URL: {gitee_url}")
        if width and height:
            print(f"      尺寸: {width}x{height}")
        preview = (caption or content)[:80].replace('\n', ' ') if (caption or content) else ''
        print(f"      描述: {preview}...")
    
    # Report（与 HippoRAG 阶段5结构对齐）
    if 'report' in rag_result:
        report = rag_result['report']
        print(f"\n📝 LLM生成报告:")
        if report.get('success'):
            report_data = report.get('report', {})
            print(f"  ✅ 生成成功")
            answer = report_data.get('answer', {}) if isinstance(report_data, dict) else {}
            summary = answer.get('summary', '') if isinstance(answer, dict) else ''
            key_points = answer.get('key_points', []) if isinstance(answer, dict) else []
            sections = answer.get('sections', []) if isinstance(answer, dict) else []
            selection = report_data.get('selection', {}) if isinstance(report_data, dict) else {}
            print(f"  📋 核心摘要: {(summary or '无')[:150]}...")
            print(f"  📌 关键要点: {len(key_points)}个")
            for kp in (key_points or [])[:3]:
                print(f"      - {kp}")
            print(f"  📄 内容章节: {len(sections)}个")
            if isinstance(selection, dict):
                print(f"  📎 选取结果: chunk={len(selection.get('selected_chunks', []))}, code={len(selection.get('selected_codes', []))}, table={len(selection.get('selected_tables', []))}, image={len(selection.get('selected_images', []))}")
            # frontend_prompt 长度
            frontend_prompt = report.get('frontend_prompt', '')
            print(f"  📄 Frontend Prompt 长度: {len(frontend_prompt)} 字符")
        else:
            print(f"  ❌ 生成失败: {report.get('error', '未知错误')}")
    
    # Timing
    timing = rag_result.get('timing', {})
    print(f"\n⏱️ 耗时统计:")
    print(f"  阶段1 (Embedding): {timing.get('stage1_embedding', 0):.2f}s")
    print(f"  阶段2 (Rerank): {timing.get('stage2_rerank', 0):.2f}s")
    print(f"  阶段3 (扩散): {timing.get('stage3_spread', 0):.2f}s")
    print(f"  阶段4 (最终Rerank): {timing.get('stage4_final_rerank', 0):.2f}s")
    if 'stage5_report' in timing:
        print(f"  阶段5 (报告生成): {timing.get('stage5_report', 0):.2f}s")
    print(f"  总计: {timing.get('total', 0):.2f}s")
    print(f"{'#'*70}\n")

def format_rag_context(rag_result: Dict[str, Any]) -> str:
    """
    将 RAG 检索结果格式化为 LLM 可用的上下文文本
    
    优先使用 HippoRAG 生成的 frontend_prompt（参考 demo_openai.py）
    如果没有则 fallback 到手动拼接
    
    Args:
        rag_result: retrieve_v2 返回的单个查询结果
    
    Returns:
        格式化后的上下文字符串
    """
    if not rag_result:
        return ""
    
    # 优先使用 HippoRAG 生成的 frontend_prompt（经过LLM整合的高质量Prompt）
    if 'report' in rag_result:
        report = rag_result['report']
        if report.get('success') and report.get('frontend_prompt'):
            logger.info("[RAG] 使用 HippoRAG 生成的 frontend_prompt")
            return report['frontend_prompt']
    
    # Fallback: 手动拼接（如果 generate_report=False 或报告生成失败）
    logger.info("[RAG] Fallback: 手动拼接 RAG 上下文")
    context_parts = []
    
    # 1. 添加 Chunk 内容
    chunks = rag_result.get('chunks', {})
    chunk_ids = chunks.get('ids', [])
    chunk_contents = chunks.get('contents', [])
    
    if chunk_contents:
        context_parts.append("## 📄 相关文档段落\n")
        for i, (chunk_id, content) in enumerate(zip(chunk_ids[:5], chunk_contents[:5])):
            # 截取合理长度
            content_preview = content[:2000] if len(content) > 2000 else content
            context_parts.append(f"### 段落 {i+1}\n```\n{content_preview}\n```\n")
    
    # 2. 添加 Code 内容
    codes = rag_result.get('codes', {})
    code_ids = codes.get('ids', [])
    code_contents = codes.get('contents', [])
    
    if code_contents:
        context_parts.append("\n## 💻 相关代码\n")
        for i, (code_id, content) in enumerate(zip(code_ids[:3], code_contents[:3])):
            content_preview = content[:1500] if len(content) > 1500 else content
            context_parts.append(f"### 代码 {i+1}\n```\n{content_preview}\n```\n")
    
    # 3. 添加 Table 内容
    tables = rag_result.get('tables', {})
    table_ids = tables.get('ids', [])
    table_contents = tables.get('contents', [])
    
    if table_contents:
        context_parts.append("\n## 📊 相关表格\n")
        for i, (table_id, content) in enumerate(zip(table_ids[:3], table_contents[:3])):
            content_preview = content[:1000] if len(content) > 1000 else content
            context_parts.append(f"### 表格 {i+1}\n```\n{content_preview}\n```\n")
    
    # 4. 添加 Image（包含URL和描述）
    images = rag_result.get('images', {})
    image_ids = images.get('ids', [])
    image_contents = images.get('contents', [])  # 这是AI生成的描述
    image_metadata = images.get('metadata', [])
    
    if image_ids:
        context_parts.append("\n## 🖼️ 相关图片\n")
        for i, iid in enumerate(image_ids[:5]):
            # 获取对应的 metadata
            meta = image_metadata[i] if i < len(image_metadata) else {}
            content = image_contents[i] if i < len(image_contents) else ''
            
            gitee_url = meta.get('gitee_url', '')
            caption = meta.get('caption', content)  # 优先用 caption，否则用 content
            breadcrumb = meta.get('breadcrumb', '')
            width = meta.get('width')
            height = meta.get('height')
            
            context_parts.append(f"### 图片 {i+1}")
            if gitee_url:
                context_parts.append(f"- **URL**: {gitee_url}")
                context_parts.append(f"- **Markdown引用**: ![图片{i+1}]({gitee_url})")
            if width and height:
                context_parts.append(f"- **尺寸**: {width}x{height}")
            if caption:
                context_parts.append(f"- **描述**: {caption}")
            if breadcrumb:
                context_parts.append(f"- **来源**: {breadcrumb}")
            context_parts.append("")
    
    return "\n".join(context_parts)

# ==================== LLM 调用 ====================

async def get_system_prompt() -> str:
    """读取自定义的 OpenHarmony System Prompt"""
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ours/step2_system_prompt_zh.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"System prompt 文件未找到: {prompt_path}")
        return "你是 OpenHarmony 智能助手，请根据提供的上下文回答用户问题。"

async def call_llm(messages: List[Dict]) -> Optional[str]:
    """
    调用 LLM（OpenAI兼容 /chat/completions）。
    注意：最终“前端HTML生成”更适合使用 coder 类模型；默认使用 qwen3-coder，
    仍可通过环境变量 CHAT_MODEL_NAME 覆盖。
    """
    api_key = get_api_key()
    # 按 ModelArts MaaS 示例：最终前端/HTML生成更适合 coder 模型；默认用全名，避免平台侧歧义
    model_name = os.environ.get("CHAT_MODEL_NAME") or "qwen3-coder-480b-a35b-instruct"

    # 按 ModelArts MaaS 示例：默认走 v2/chat/completions；仍可通过 CHAT_BASE_URL 覆盖
    base_url = os.environ.get("CHAT_BASE_URL") or "https://api.modelarts-maas.com/v2"
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
        logger.info(f"Sending request to {model_name} via OpenAI-compatible endpoint...")
        # 与仓库内其他 ModelArts 调用保持一致：verify=False 避免部分环境证书链问题
        resp = requests.post(api_url, headers=headers, json=payload, timeout=600, verify=False)
        resp.raise_for_status()
        return resp.json()

    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _do_request)
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

def extract_html(content: str) -> str:
    """从 Markdown 代码块中提取 HTML"""
    if "```html" in content:
        try:
            return content.split("```html", 1)[1].split("```", 1)[0].strip()
        except Exception:
            return content
    return content

def harden_html_for_external_assets(html: str) -> str:
    """
    针对“防盗链(Referer校验)”的站点做兼容：
    - 注入 <meta name="referrer" content="no-referrer">，让页面内资源请求不带 Referer
    - 给 <img> 补 referrerpolicy="no-referrer"
    说明：这不会绕过真正的鉴权，仅用于解决外链图片的 Referer 防盗链问题。
    """
    import re

    if not html or "<html" not in html.lower():
        return html

    meta_tag = '<meta name="referrer" content="no-referrer">'

    # 1) 尽量插入到 <head> 内部（优先放在 charset/meta 之后也无妨）
    if re.search(r"<meta\s+name=['\"]referrer['\"]", html, flags=re.IGNORECASE):
        out = html
    else:
        m = re.search(r"<head[^>]*>", html, flags=re.IGNORECASE)
        if m:
            insert_pos = m.end()
            out = html[:insert_pos] + "\n  " + meta_tag + html[insert_pos:]
        else:
            out = meta_tag + "\n" + html

    # 2) 给 img 补 referrerpolicy
    out = re.sub(
        r"<img\b(?![^>]*\breferrerpolicy=)([^>]*?)>",
        r'<img\1 referrerpolicy="no-referrer">',
        out,
        flags=re.IGNORECASE,
    )
    return out

def ensure_images_in_html(html: str, rag_result: Optional[Dict[str, Any]]) -> str:
    """
    兜底：如果 LLM 没有把图片放进 HTML，但 RAG 已经检索/选中了图片，则自动注入图片 Hero/图集。
    目标：保证“有图必用”，避免仅靠 LLM 遵循 system prompt 的不稳定性。
    """
    import re
    from urllib.parse import quote

    if not html or not rag_result:
        return html
    if re.search(r"<img\b", html, flags=re.IGNORECASE):
        return html

    images = rag_result.get("images", {}) if isinstance(rag_result, dict) else {}
    metas = images.get("metadata", []) if isinstance(images, dict) else []
    if not metas:
        return html

    # 优先使用阶段5 selection 选中的图片；否则用 metadata 列表中的前几张
    selected_ids = []
    try:
        report = rag_result.get("report", {}) if isinstance(rag_result, dict) else {}
        report_data = report.get("report", {}) if isinstance(report, dict) else {}
        selection = report_data.get("selection", {}) if isinstance(report_data, dict) else {}
        selected_ids = selection.get("selected_images", []) if isinstance(selection, dict) else []
    except Exception:
        selected_ids = []

    id_set = set([str(x) for x in selected_ids]) if selected_ids else set()
    chosen = []
    for m in metas:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id", ""))
        if id_set and mid and mid not in id_set:
            continue
        url = (m.get("gitee_url") or "").strip()
        if not url:
            continue
        chosen.append(m)
        if len(chosen) >= 2:  # 默认最多注入2张，避免撑爆页面
            break
    if not chosen:
        # fallback：取前两张有 url 的
        for m in metas:
            if isinstance(m, dict) and (m.get("gitee_url") or "").strip():
                chosen.append(m)
                if len(chosen) >= 2:
                    break
    if not chosen:
        return html

    # 注入一个简洁的 Hero Gallery（尽量不影响原页面结构）
    figures = []
    for m in chosen:
        url = (m.get("gitee_url") or "").strip()
        caption = (m.get("caption") or "图片").strip()
        breadcrumb = (m.get("breadcrumb") or "").strip()
        # 若后续要升级为“服务端图片代理”，这里保留一个可替换点（当前直接用外链）
        safe_url = url
        figures.append(
            f"""
            <figure class="rounded-xl overflow-hidden border border-slate-700/60 bg-slate-900/40">
              <img src="{safe_url}" alt="{caption}" referrerpolicy="no-referrer" class="w-full h-auto block"/>
              <figcaption class="px-4 py-3 text-sm text-slate-300">
                <div class="font-medium">{caption}</div>
                {f'<div class="text-xs text-slate-400 mt-1">{breadcrumb}</div>' if breadcrumb else ''}
              </figcaption>
            </figure>
            """.strip()
        )
    injected = f"""
<!-- [AUTO] Injected image(s) because LLM output had no <img> but RAG provided images -->
<section class="mx-auto max-w-6xl px-4 py-6">
  <div class="mb-3 text-slate-300 text-sm">相关图片</div>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    {"".join(figures)}
  </div>
</section>
""".strip()

    # 尝试插入到 <body> 后；否则插到最前面
    m = re.search(r"<body[^>]*>", html, flags=re.IGNORECASE)
    if m:
        pos = m.end()
        return html[:pos] + "\n" + injected + "\n" + html[pos:]
    return injected + "\n" + html

def repair_img_srcs_from_rag(html: str, rag_result: Optional[Dict[str, Any]]) -> str:
    """
    修复模型生成的“坏 img src”（常见于把尺寸/其他文本误混进 src 属性导致 URL 断裂）。
    策略：
    - 如果存在 <img> 但其 src 不是 http(s) URL，则用 RAG 选中的第一张图片 gitee_url 替换
    - 只做最小侵入式替换，避免破坏模型输出结构
    """
    import re

    if not html or not rag_result:
        return html

    # 收集候选图片 URL（优先 selection）
    images = rag_result.get("images", {}) if isinstance(rag_result, dict) else {}
    metas = images.get("metadata", []) if isinstance(images, dict) else []
    if not metas:
        return html

    selected_ids = []
    try:
        report = rag_result.get("report", {}) if isinstance(rag_result, dict) else {}
        report_data = report.get("report", {}) if isinstance(report, dict) else {}
        selection = report_data.get("selection", {}) if isinstance(report_data, dict) else {}
        selected_ids = selection.get("selected_images", []) if isinstance(selection, dict) else []
    except Exception:
        selected_ids = []

    id_set = set([str(x) for x in selected_ids]) if selected_ids else set()
    candidate_urls: List[str] = []
    for m in metas:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id", ""))
        if id_set and mid and mid not in id_set:
            continue
        url = (m.get("gitee_url") or "").strip()
        if url.startswith("http"):
            candidate_urls.append(url)
    if not candidate_urls:
        for m in metas:
            if isinstance(m, dict):
                url = (m.get("gitee_url") or "").strip()
                if url.startswith("http"):
                    candidate_urls.append(url)
                    break
    if not candidate_urls:
        return html

    fallback_url = candidate_urls[0]

    # 替换坏 src（不以 http/https 开头，或明显被截断如 https://gite）
    def _fix_img(match: re.Match) -> str:
        tag = match.group(0)
        src = match.group("src") or ""
        src_strip = src.strip()
        if src_strip.startswith("http://") or src_strip.startswith("https://"):
            # 额外兜底：明显截断的域名
            if src_strip.startswith("https://gite") and "gitee.com" not in src_strip:
                return re.sub(r'src\s*=\s*"[^"]*"', f'src="{fallback_url}"', tag, count=1, flags=re.IGNORECASE)
            return tag
        # 非 URL 直接替换
        return re.sub(r'src\s*=\s*"[^"]*"', f'src="{fallback_url}"', tag, count=1, flags=re.IGNORECASE)

    # 仅处理双引号形式（我们生成/大部分模型也输出双引号）
    pattern = re.compile(r"<img\b[^>]*\bsrc\s*=\s*\"(?P<src>[^\"]*)\"[^>]*>", flags=re.IGNORECASE)
    return pattern.sub(_fix_img, html)

# ==================== FastAPI Web 接口 ====================

app = FastAPI(title="OpenHarmony Assistant API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

# ==================== 请求/响应模型 ====================

class ChatRequest(BaseModel):
    query: str
    use_rag: bool = True  # 是否使用 RAG 检索
    rag_context: Optional[str] = ""  # 手动提供的上下文（可选）

class ChatResponse(BaseModel):
    html: str
    raw: str
    rag_used: bool = False
    rag_chunks_count: int = 0
    rag_codes_count: int = 0
    rag_tables_count: int = 0
    rag_images_count: int = 0
    timing_rag: float = 0.0
    timing_llm: float = 0.0

class RetrieveRequest(BaseModel):
    query: str
    fact_candidate_k: int = 50
    file_candidate_k: int = 30
    chunk_candidate_k: int = 30
    fact_top_k: int = 10
    file_top_k: int = 10
    chunk_top_k: int = 10
    final_chunk_k: int = 10
    final_code_k: int = 3
    final_table_k: int = 3
    final_image_k: int = 3

class RetrieveResponse(BaseModel):
    query: str
    chunks: Dict[str, Any]
    codes: Dict[str, Any]
    tables: Dict[str, Any]
    images: Dict[str, Any]
    timing: Dict[str, float]

# ==================== 核心 API 端点 ====================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
    """
    主聊天接口：
    1. 如果 use_rag=True，先调用 HippoRAG 检索相关上下文
    2. 将检索结果 + 用户问题发送给 LLM
    3. 返回生成的 HTML 回答
    """
    import time
    
    rag_result = None
    rag_context = body.rag_context or ""
    timing_rag = 0.0
    timing_llm = 0.0
    
    # Step 1: RAG 检索
    if body.use_rag:
        logger.info(f"[RAG] 开始检索: {body.query}")
        rag_start = time.time()
        try:
            rag_result = await retrieve_rag_context(body.query, verbose=True)
            if rag_result:
                # 打印详细的 RAG 结果日志
                log_rag_result(rag_result, body.query)
                rag_context = format_rag_context(rag_result)
                logger.info(f"[RAG] 检索完成，上下文长度: {len(rag_context)}")
        except Exception as e:
            logger.error(f"[RAG] 检索失败: {e}")
            import traceback
            traceback.print_exc()
        timing_rag = time.time() - rag_start
    
    # Step 2: 构建 Prompt 并调用 LLM
    system_prompt = await get_system_prompt()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    system_prompt = system_prompt.replace("%%%DATE%%%", current_date)

    user_content = f"""
【参考上下文 (RAG Context)】:
{rag_context}

【用户问题】:
{body.query}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    logger.info(f"[CHAT] Query: {body.query}")
    llm_start = time.time()
    response_content = await call_llm(messages)
    timing_llm = time.time() - llm_start

    if not response_content:
        return ChatResponse(
            html="<p>抱歉，我这边出了一点问题，请稍后再试。</p>",
            raw="LLM call failed or empty response.",
            rag_used=body.use_rag,
            timing_rag=timing_rag,
            timing_llm=timing_llm
        )

    html_content = extract_html(response_content)
    html_content = harden_html_for_external_assets(html_content)
    html_content = repair_img_srcs_from_rag(html_content, rag_result)
    html_content = ensure_images_in_html(html_content, rag_result)
    
    # 保存结果
    try:
        import json
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "chat_html")
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # 保存 HTML
        html_path = os.path.join(save_dir, f"{ts}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # 保存原始 LLM 输出
        raw_path = os.path.join(save_dir, f"{ts}.raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(response_content)
        
        # 保存 RAG 中间结果（参考 demo_openai.py，包含完整 metadata 和 report）
        if rag_result:
            rag_path = os.path.join(save_dir, f"{ts}.rag.json")
            rag_save_data = {
                'query': body.query,
                'chunks': {
                    'ids': rag_result.get('chunks', {}).get('ids', []),
                    'contents': rag_result.get('chunks', {}).get('contents', []),
                    'metadata': rag_result.get('chunks', {}).get('metadata', [])
                },
                'codes': {
                    'ids': rag_result.get('codes', {}).get('ids', []),
                    'contents': rag_result.get('codes', {}).get('contents', []),
                    'metadata': rag_result.get('codes', {}).get('metadata', [])
                },
                'tables': {
                    'ids': rag_result.get('tables', {}).get('ids', []),
                    'contents': rag_result.get('tables', {}).get('contents', []),
                    'metadata': rag_result.get('tables', {}).get('metadata', [])
                },
                'images': {
                    'ids': rag_result.get('images', {}).get('ids', []),
                    'contents': rag_result.get('images', {}).get('contents', []),
                    'metadata': rag_result.get('images', {}).get('metadata', [])
                },
                'timing': rag_result.get('timing', {}),
                'rag_context': rag_context
            }
            
            # 保存 report 信息（如果有）
            if 'report' in rag_result:
                report = rag_result['report']
                rag_save_data['report'] = {
                    'success': report.get('success', False),
                    # 与 HippoRAG 结构对齐：report 内含 answer(占位) + selection(关键) 等
                    'report_data': report.get('report', {}),
                    'error': report.get('error', '')
                }
                # 方便调试：落盘 frontend_prompt（真实材料）
                rag_save_data['frontend_prompt'] = report.get('frontend_prompt', '')
            
            with open(rag_path, "w", encoding="utf-8") as f:
                json.dump(rag_save_data, f, ensure_ascii=False, indent=2)
            logger.info(f"[RAG] 中间结果已保存: {rag_path}")
            
            # 单独保存 frontend_prompt（参考 demo_openai.py）
            if 'report' in rag_result and rag_result['report'].get('success'):
                prompt_path = os.path.join(save_dir, f"{ts}.frontend_prompt.md")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(rag_result['report'].get('frontend_prompt', ''))
                logger.info(f"[RAG] Frontend Prompt 已保存: {prompt_path}")
        
        logger.info(f"[CHAT] Saved HTML to {html_path}")
    except Exception as e:
        logger.error(f"[CHAT] Failed to save: {e}")
    
    # 统计 RAG 结果数量
    chunks_count = len(rag_result.get('chunks', {}).get('ids', [])) if rag_result else 0
    codes_count = len(rag_result.get('codes', {}).get('ids', [])) if rag_result else 0
    tables_count = len(rag_result.get('tables', {}).get('ids', [])) if rag_result else 0
    images_count = len(rag_result.get('images', {}).get('ids', [])) if rag_result else 0
    
    return ChatResponse(
        html=html_content,
        raw=response_content,
        rag_used=body.use_rag and rag_result is not None,
        rag_chunks_count=chunks_count,
        rag_codes_count=codes_count,
        rag_tables_count=tables_count,
        rag_images_count=images_count,
        timing_rag=timing_rag,
        timing_llm=timing_llm
    )

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(body: RetrieveRequest):
    """
    单独的检索接口：只进行 RAG 检索，不调用 LLM
    可用于前端预览检索结果或调试
    """
    def _do_retrieve():
        hipporag = get_hipporag()
        results = hipporag.retrieve_v2(
            queries=[body.query],
            fact_candidate_k=body.fact_candidate_k,
            file_candidate_k=body.file_candidate_k,
            chunk_candidate_k=body.chunk_candidate_k,
            fact_top_k=body.fact_top_k,
            file_top_k=body.file_top_k,
            chunk_top_k=body.chunk_top_k,
            spread_chunk_k=30,
            spread_code_k=10,
            spread_table_k=10,
            spread_image_k=10,
            final_chunk_k=body.final_chunk_k,
            final_code_k=body.final_code_k,
            final_table_k=body.final_table_k,
            final_image_k=body.final_image_k,
            generate_report=True,
            verbose=False
        )
        return results[0] if results else None
    
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _do_retrieve)
    
    if not result:
        return RetrieveResponse(
            query=body.query,
            chunks={'ids': [], 'contents': []},
            codes={'ids': [], 'contents': []},
            tables={'ids': [], 'contents': []},
            images={'ids': [], 'contents': []},
            timing={'total': 0}
        )
    
    return RetrieveResponse(
        query=result['query'],
        chunks=result['chunks'],
        codes=result['codes'],
        tables=result['tables'],
        images=result['images'],
        timing=result['timing']
    )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "hipporag_initialized": hipporag_instance is not None,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/init_rag")
async def init_rag():
    """手动初始化 HippoRAG（预热）"""
    def _init():
        return get_hipporag() is not None
    
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(None, _init)
    
    return {
        "success": success,
        "message": "HippoRAG 初始化成功" if success else "初始化失败"
    }

# ==================== 启动 ====================

if __name__ == "__main__":
    import multiprocessing
    # 设置 multiprocessing start method 为 spawn（在 main guard 内）
    multiprocessing.set_start_method('spawn', force=True)
    
    import uvicorn
    # 启动 Web API，前端请求 http://localhost:8000/chat
    # 注意：关闭 reload 模式，因为 vLLM 多进程与 uvicorn reload 不兼容
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
