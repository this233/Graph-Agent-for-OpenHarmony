


from fastapi.responses import HTMLResponse


import os
import datetime
import asyncio
from typing import Optional

import requests
import loguru
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = loguru.logger

# ==================== 读取前端页面 ====================

BASE_DIR = os.path.dirname(__file__)
FRONTEND_PATH = "/root/code/OpenHarmonyAssistant/chatbox/index.html"

with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
    HTML_PAGE = f.read()

# ==================== ModelArts MaaS 配置====================

# API_URL = "https://api.modelarts-maas.com/v1/chat/completions"
API_URL = "https://api.modelarts-maas.com/v2/chat/completions"
# MODEL_NAME = "DeepSeek-V3"
MODEL_NAME = "qwen3-coder-480b-a35b-instruct"
TEMPERATURE = 0.6

def get_api_key() -> str:
    """
    从环境变量 MAAS_API_KEY 读取，如果没有则使用一个默认值（不推荐写死在代码里，仅开发测试用）。
    """
    api_key = os.environ.get("MAAS_API_KEY")
    if not api_key:
        # api_key = "mATSRxNA6sLbRzfYOU9St9y0OcaX37-47V0FK49XMySBn-T-olZ4SJM8C3lJXtwBJ3F4vEjMv2tUbZGKf7EEuQ"
        
        api_key = "7cL3yLtnQ09_nvQtdlOJVQeJLgjy9O7wfTgb31NNbuAsB_xvDfitBqbYSyKsqCPemEo-n4oH_S2WA6IApbaB8g"
        # api_key= "BQm_Gkd1EoTcHkJfVf31dTWfMIOsW3_mKIDfM5j-MvvwNM5jNl9XnLOjvNjEOuDiIWoKb-DIphdRWt2gOoNwBw"
        
    return api_key

# ==================== Prompt、LLM 调用等 ====================

async def get_system_prompt() -> str:
    """读取自定义的 OpenHarmony System Prompt"""
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ours/step2_system_prompt_zh.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


async def call_llm(messages):
    """
    调用华为 ModelArts MaaS 的 DeepSeek-V3 接口。
    因为 requests 是同步的，所以用 run_in_executor 包一层，避免阻塞事件循环。
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_api_key()}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "stream": False,  # 先用非流式，简单一点
    }

    def _do_request():
        logger.info(f"Sending request to {MODEL_NAME} via ModelArts MaaS...")
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()

    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _do_request)
        # 假设返回格式兼容 OpenAI：
        # { "choices": [ { "message": { "content": "..." } } ] }
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

# ==================== FastAPI Web 接口 ====================

app = FastAPI()

# HTML_PAGE = """
# <!DOCTYPE html>
# <html lang="zh-CN">
# <head>
#   <meta charset="UTF-8" />
#   <title>OpenHarmony 助手 · Chat</title>
#   <style>
#     body {
#       margin: 0;
#       font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
#       background: radial-gradient(circle at top, #1e293b 0, #020617 50%, #000 100%);
#       color: #e5e7eb;
#       height: 100vh;
#       display: flex;
#       justify-content: center;
#       align-items: stretch;
#     }
#     .chat-wrapper {
#       width: 100%;
#       max-width: 960px;
#       height: 100vh;
#       padding: 16px;
#       display: flex;
#     }
#     .chat-card {
#       background: rgba(15, 23, 42, 0.9);
#       border-radius: 24px;
#       border: 1px solid rgba(148, 163, 184, 0.3);
#       box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
#       flex: 1;
#       display: flex;
#       flex-direction: column;
#       overflow: hidden;
#       backdrop-filter: blur(18px);
#     }
#     .chat-header {
#       padding: 14px 20px;
#       border-bottom: 1px solid rgba(30, 64, 175, 0.6);
#       display: flex;
#       align-items: center;
#       justify-content: space-between;
#       background: linear-gradient(to right, rgba(15, 118, 110, 0.32), rgba(37, 99, 235, 0.32));
#     }
#     .chat-header-title {
#       display: flex;
#       align-items: center;
#       gap: 10px;
#     }
#     .chat-header-orb {
#       width: 24px;
#       height: 24px;
#       border-radius: 999px;
#       background: conic-gradient(from 0deg, #22d3ee, #818cf8, #f97316, #22d3ee);
#       position: relative;
#       animation: spin 6s linear infinite;
#     }
#     .chat-header-orb::after {
#       content: "";
#       position: absolute;
#       inset: 3px;
#       border-radius: inherit;
#       background: radial-gradient(circle at 30% 30%, #f9fafb, #312e81);
#     }
#     .chat-header h1 {
#       font-size: 15px;
#       margin: 0;
#     }
#     .chat-header span {
#       font-size: 11px;
#       color: #cbd5f5;
#       opacity: 0.8;
#     }
#     .chat-body {
#       flex: 1;
#       padding: 16px 18px;
#       overflow-y: auto;
#       display: flex;
#       flex-direction: column;
#       gap: 10px;
#     }
#     .bubble-row {
#       display: flex;
#       margin-bottom: 4px;
#     }
#     .bubble-row.user {
#       justify-content: flex-end;
#     }
#     .bubble-row.assistant {
#       justify-content: flex-start;
#     }
#     .bubble {
#       max-width: 72%;
#       padding: 10px 12px;
#       border-radius: 18px;
#       font-size: 14px;
#       line-height: 1.5;
#       word-wrap: break-word;
#       white-space: pre-wrap;
#     }
#     .bubble.user-bubble {
#       background: linear-gradient(to right, #0ea5e9, #22c55e);
#       color: #0b1120;
#       border-bottom-right-radius: 4px;
#     }
#     .bubble.assistant-bubble {
#       background: rgba(15, 23, 42, 0.9);
#       border: 1px solid rgba(148, 163, 184, 0.4);
#       border-bottom-left-radius: 4px;
#     }
#     .chat-footer {
#       border-top: 1px solid rgba(51, 65, 85, 0.9);
#       padding: 10px 14px 14px;
#     }
#     .chat-footer form {
#       display: flex;
#       flex-direction: column;
#       gap: 8px;
#     }
#     .chat-footer-top {
#       display: flex;
#       gap: 8px;
#     }
#     .chat-input {
#       flex: 4;
#       border-radius: 16px;
#       border: 1px solid rgba(75, 85, 99, 0.9);
#       background: rgba(15, 23, 42, 0.9);
#       padding: 8px 10px;
#       color: #e5e7eb;
#       font-size: 14px;
#       resize: none;
#       min-height: 40px;
#       max-height: 80px;
#     }
#     .chat-input:focus {
#       outline: none;
#       border-color: #38bdf8;
#       box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.4);
#     }
#     .chat-context {
#       flex: 3;
#       border-radius: 16px;
#       border: 1px dashed rgba(55, 65, 81, 0.9);
#       background: rgba(15, 23, 42, 0.7);
#       padding: 8px 10px;
#       color: #9ca3af;
#       font-size: 12px;
#       resize: none;
#       min-height: 40px;
#       max-height: 80px;
#     }
#     .chat-footer-bottom {
#       display: flex;
#       justify-content: space-between;
#       align-items: center;
#       gap: 8px;
#     }
#     .hint {
#       font-size: 11px;
#       color: #9ca3af;
#       opacity: 0.8;
#     }
#     .send-button {
#       padding: 6px 18px;
#       border-radius: 999px;
#       border: none;
#       font-size: 13px;
#       font-weight: 500;
#       color: #0b1120;
#       background: linear-gradient(to right, #22d3ee, #818cf8);
#       cursor: pointer;
#       display: inline-flex;
#       align-items: center;
#       gap: 6px;
#       box-shadow: 0 8px 24px rgba(56, 189, 248, 0.3);
#       transition: transform 0.08s ease, box-shadow 0.08s ease, opacity 0.1s ease;
#     }
#     .send-button:hover {
#       transform: translateY(-1px);
#       box-shadow: 0 10px 32px rgba(59, 130, 246, 0.4);
#     }
#     .send-button:active {
#       transform: translateY(0);
#       box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
#     }
#     .send-button:disabled {
#       opacity: 0.6;
#       cursor: not-allowed;
#       box-shadow: none;
#     }
#     /* 思考中动画 */
#     .thinking-container {
#       display: flex;
#       align-items: center;
#       gap: 10px;
#     }
#     .thinking-avatar {
#       width: 22px;
#       height: 22px;
#       border-radius: 999px;
#       background: conic-gradient(#22d3ee, #818cf8, #f97316, #22d3ee);
#       position: relative;
#       animation: breathe 1.8s ease-in-out infinite;
#     }
#     .thinking-avatar::after {
#       content: "";
#       position: absolute;
#       inset: 3px;
#       border-radius: inherit;
#       background: radial-gradient(circle at 30% 30%, #e5e7eb, #111827);
#     }
#     .thinking-dots {
#       display: flex;
#       align-items: flex-end;
#       gap: 4px;
#     }
#     .thinking-dot {
#       width: 6px;
#       height: 6px;
#       border-radius: 999px;
#       background: linear-gradient(to bottom right, #38bdf8, #a855f7);
#       opacity: 0.4;
#       animation: thinking-bounce 1.1s infinite;
#     }
#     .thinking-dot:nth-child(2) {
#       animation-delay: 0.16s;
#     }
#     .thinking-dot:nth-child(3) {
#       animation-delay: 0.32s;
#     }
#     @keyframes thinking-bounce {
#       0%, 80%, 100% {
#         transform: translateY(0);
#         opacity: 0.4;
#       }
#       40% {
#         transform: translateY(-4px);
#         opacity: 1;
#       }
#     }
#     @keyframes breathe {
#       0%, 100% {
#         transform: scale(1);
#         box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.2);
#       }
#       50% {
#         transform: scale(1.05);
#         box-shadow: 0 0 0 10px rgba(56, 189, 248, 0);
#       }
#     }
#     @keyframes spin {
#       to {
#         transform: rotate(360deg);
#       }
#     }
#     .chat-body::-webkit-scrollbar {
#       width: 6px;
#     }
#     .chat-body::-webkit-scrollbar-thumb {
#       background: rgba(148, 163, 184, 0.4);
#       border-radius: 999px;
#     }
#   </style>
# </head>
# <body>
#   <div class="chat-wrapper">
#     <div class="chat-card">
#       <div class="chat-header">
#         <div class="chat-header-title">
#           <div class="chat-header-orb"></div>
#           <div>
#             <h1>OpenHarmony 助手 · 实时对话</h1>
#             <span>当 LLM 在“思考”时，看看它的脑子在发光 ✨</span>
#           </div>
#         </div>
#       </div>
#       <div class="chat-body" id="chatBody">
#         <div class="bubble-row assistant">
#           <div class="bubble assistant-bubble">
#             你好，我是 OpenHarmony 相关文档的小助手。可以问我架构、部件设计、SysCap 等问题～
#           </div>
#         </div>
#       </div>
#       <div class="chat-footer">
#         <form id="chatForm">
#           <div class="chat-footer-top">
#             <textarea
#               id="queryInput"
#               class="chat-input"
#               placeholder="输入你的问题，例如：请介绍 OpenHarmony 的技术架构，特别是内核层和框架层。"
#             ></textarea>
#             <textarea
#               id="contextInput"
#               class="chat-context"
#               placeholder="可选：填入你检索到的一些 RAG 文本片段（暂时也可以先留空）"
#             ></textarea>
#           </div>
#           <div class="chat-footer-bottom">
#             <div class="hint">回车发送，Shift + 回车换行。</div>
#             <button type="submit" class="send-button" id="sendBtn">
#               <span>发送</span><span>➤</span>
#             </button>
#           </div>
#         </form>
#       </div>
#     </div>
#   </div>
#   <script>
#     const API_URL = "/chat"; // 和后端同域

#     const chatBody = document.getElementById("chatBody");
#     const chatForm = document.getElementById("chatForm");
#     const queryInput = document.getElementById("queryInput");
#     const contextInput = document.getElementById("contextInput");
#     const sendBtn = document.getElementById("sendBtn");

#     let isWaiting = false;
#     let thinkingBubbleEl = null;

#     function appendUserMessage(text) {
#       const row = document.createElement("div");
#       row.className = "bubble-row user";
#       const bubble = document.createElement("div");
#       bubble.className = "bubble user-bubble";
#       bubble.textContent = text;
#       row.appendChild(bubble);
#       chatBody.appendChild(row);
#       chatBody.scrollTop = chatBody.scrollHeight;
#     }

#     function appendThinkingBubble() {
#       const row = document.createElement("div");
#       row.className = "bubble-row assistant";
#       const bubble = document.createElement("div");
#       bubble.className = "bubble assistant-bubble";

#       const container = document.createElement("div");
#       container.className = "thinking-container";

#       const avatar = document.createElement("div");
#       avatar.className = "thinking-avatar";

#       const dots = document.createElement("div");
#       dots.className = "thinking-dots";
#       for (let i = 0; i < 3; i++) {
#         const dot = document.createElement("div");
#         dot.className = "thinking-dot";
#         dots.appendChild(dot);
#       }

#       container.appendChild(avatar);
#       container.appendChild(dots);
#       bubble.appendChild(container);
#       row.appendChild(bubble);

#       chatBody.appendChild(row);
#       chatBody.scrollTop = chatBody.scrollHeight;

#       thinkingBubbleEl = bubble;
#     }

#     function replaceThinkingWithAnswer(html) {
#       if (!thinkingBubbleEl) return;
#       thinkingBubbleEl.innerHTML = html;
#       chatBody.scrollTop = chatBody.scrollHeight;
#       thinkingBubbleEl = null;
#     }

#     async function sendMessage(query, ragContext) {
#       if (!query.trim() || isWaiting) return;
#       isWaiting = true;
#       sendBtn.disabled = true;

#       appendUserMessage(query);
#       appendThinkingBubble();

#       try {
#         const res = await fetch(API_URL, {
#           method: "POST",
#           headers: { "Content-Type": "application/json" },
#           body: JSON.stringify({ query: query, rag_context: ragContext || "" }),
#         });
#         if (!res.ok) throw new Error("HTTP " + res.status);
#         const data = await res.json();
#         replaceThinkingWithAnswer(data.html || "（空响应）");
#       } catch (e) {
#         console.error(e);
#         replaceThinkingWithAnswer("抱歉，我这边出了一点错误，请稍后再试。");
#       } finally {
#         isWaiting = false;
#         sendBtn.disabled = false;
#       }
#     }

#     chatForm.addEventListener("submit", (e) => {
#       e.preventDefault();
#       const query = queryInput.value.trim();
#       const ctx = contextInput.value.trim();
#       if (!query) return;
#       sendMessage(query, ctx);
#       queryInput.value = "";
#     });

#     queryInput.addEventListener("keydown", (e) => {
#       if (e.key === "Enter" && !e.shiftKey) {
#         e.preventDefault();
#         chatForm.requestSubmit();
#       }
#     });
#   </script>
# </body>
# </html>
# """
# HTML_PAGE = """<!DOCTYPE html>
# <html lang="zh-CN">
# <head>
#   <meta charset="UTF-8" />
#   <title>OpenHarmony 助手 · HTML 对话预览</title>
#   <style>
#     * {
#       box-sizing: border-box;
#     }
#     html, body {
#       margin: 0;
#       padding: 0;
#       height: 100%;
#       width: 100%;
#     }
#     body {
#       font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
#       background: radial-gradient(circle at top, #1e293b 0, #020617 50%, #000 100%);
#       color: #e5e7eb;
#     }

#     /* 整个页面铺满屏幕的容器 */
#     .chat-wrapper {
#       width: 100%;
#       height: 100vh;
#       padding: 0;
#       display: flex;
#     }

#     .chat-card {
#       background: rgba(15, 23, 42, 0.92);
#       border-radius: 0; /* 改为顶到底的全屏布局 */
#       border-top: 1px solid rgba(148, 163, 184, 0.3);
#       border-left: none;
#       border-right: none;
#       border-bottom: none;
#       flex: 1;
#       display: flex;
#       flex-direction: column;
#       overflow: hidden;
#       backdrop-filter: blur(18px);
#     }

#     .chat-header {
#       padding: 12px 18px;
#       border-bottom: 1px solid rgba(30, 64, 175, 0.6);
#       display: flex;
#       align-items: center;
#       justify-content: space-between;
#       background: linear-gradient(to right, rgba(15, 118, 110, 0.35), rgba(37, 99, 235, 0.35));
#     }
#     .chat-header-title {
#       display: flex;
#       align-items: center;
#       gap: 10px;
#     }
#     .chat-header-orb {
#       width: 24px;
#       height: 24px;
#       border-radius: 999px;
#       background: conic-gradient(from 0deg, #22d3ee, #818cf8, #f97316, #22d3ee);
#       position: relative;
#       animation: spin 6s linear infinite;
#     }
#     .chat-header-orb::after {
#       content: "";
#       position: absolute;
#       inset: 3px;
#       border-radius: inherit;
#       background: radial-gradient(circle at 30% 30%, #f9fafb, #312e81);
#     }
#     .chat-header h1 {
#       font-size: 15px;
#       margin: 0;
#     }
#     .chat-header span {
#       font-size: 11px;
#       color: #cbd5f5;
#       opacity: 0.8;
#     }

#     /* 上部：对话区（滚动） */
#     .chat-body {
#       flex: 1;
#       padding: 12px 16px;
#       overflow-y: auto;
#       display: flex;
#       flex-direction: column;
#       gap: 10px;
#     }

#     .bubble-row {
#       display: flex;
#       margin-bottom: 4px;
#       width: 100%;
#     }
#     .bubble-row.user {
#       justify-content: flex-end;
#     }
#     .bubble-row.assistant {
#       justify-content: flex-start;
#     }

#     .bubble {
#       width: 100%; /* 关键：助手气泡占满行，方便展示大 HTML */
#       padding: 8px 10px;
#       border-radius: 18px;
#       font-size: 14px;
#       line-height: 1.5;
#       word-wrap: break-word;
#       white-space: pre-wrap;
#     }
#     .bubble.user-bubble {
#       width: auto;
#       max-width: 70%;
#       background: linear-gradient(to right, #0ea5e9, #22c55e);
#       color: #0b1120;
#       border-bottom-right-radius: 4px;
#       align-self: flex-end;
#     }
#     .bubble.assistant-bubble {
#       background: rgba(15, 23, 42, 0.98);
#       border: 1px solid rgba(148, 163, 184, 0.4);
#       border-bottom-left-radius: 4px;
#     }

#     /* 助手 HTML 页面预览 iframe */
#     .assistant-html-iframe {
#       width: 100%;
#       min-height: 500px;  /* 可以按需调整默认高度 */
#       border: none;
#       border-radius: 16px;
#       background: #ffffff;
#       overflow: auto;
#     }
#     @media (min-height: 900px) {
#       .assistant-html-iframe {
#         min-height: 700px;
#       }
#     }

#     /* 下部：输入区 */
#     .chat-footer {
#       border-top: 1px solid rgba(51, 65, 85, 0.95);
#       padding: 8px 14px 10px;
#       background: rgba(15, 23, 42, 0.98);
#     }
#     .chat-footer form {
#       display: flex;
#       flex-direction: column;
#       gap: 6px;
#     }
#     .chat-footer-top {
#       display: flex;
#       gap: 8px;
#     }
#     .chat-input {
#       flex: 4;
#       border-radius: 12px;
#       border: 1px solid rgba(75, 85, 99, 0.9);
#       background: rgba(15, 23, 42, 0.9);
#       padding: 6px 8px;
#       color: #e5e7eb;
#       font-size: 14px;
#       resize: none;
#       min-height: 40px;
#       max-height: 80px;
#     }
#     .chat-input:focus {
#       outline: none;
#       border-color: #38bdf8;
#       box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.4);
#     }
#     .chat-context {
#       flex: 3;
#       border-radius: 12px;
#       border: 1px dashed rgba(55, 65, 81, 0.9);
#       background: rgba(15, 23, 42, 0.7);
#       padding: 6px 8px;
#       color: #9ca3af;
#       font-size: 12px;
#       resize: none;
#       min-height: 40px;
#       max-height: 80px;
#     }
#     .chat-footer-bottom {
#       display: flex;
#       justify-content: space-between;
#       align-items: center;
#       gap: 8px;
#     }
#     .hint {
#       font-size: 11px;
#       color: #9ca3af;
#       opacity: 0.8;
#     }
#     .send-button {
#       padding: 6px 18px;
#       border-radius: 999px;
#       border: none;
#       font-size: 13px;
#       font-weight: 500;
#       color: #0b1120;
#       background: linear-gradient(to right, #22d3ee, #818cf8);
#       cursor: pointer;
#       display: inline-flex;
#       align-items: center;
#       gap: 6px;
#       box-shadow: 0 8px 24px rgba(56, 189, 248, 0.3);
#       transition: transform 0.08s ease, box-shadow 0.08s ease, opacity 0.1s ease;
#     }
#     .send-button:hover {
#       transform: translateY(-1px);
#       box-shadow: 0 10px 32px rgba(59, 130, 246, 0.4);
#     }
#     .send-button:active {
#       transform: translateY(0);
#       box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
#     }
#     .send-button:disabled {
#       opacity: 0.6;
#       cursor: not-allowed;
#       box-shadow: none;
#     }

#     /* “思考中”动画 */
#     .thinking-container {
#       display: flex;
#       align-items: center;
#       gap: 10px;
#     }
#     .thinking-avatar {
#       width: 22px;
#       height: 22px;
#       border-radius: 999px;
#       background: conic-gradient(#22d3ee, #818cf8, #f97316, #22d3ee);
#       position: relative;
#       animation: breathe 1.8s ease-in-out infinite;
#     }
#     .thinking-avatar::after {
#       content: "";
#       position: absolute;
#       inset: 3px;
#       border-radius: inherit;
#       background: radial-gradient(circle at 30% 30%, #e5e7eb, #111827);
#     }
#     .thinking-dots {
#       display: flex;
#       align-items: flex-end;
#       gap: 4px;
#     }
#     .thinking-dot {
#       width: 6px;
#       height: 6px;
#       border-radius: 999px;
#       background: linear-gradient(to bottom right, #38bdf8, #a855f7);
#       opacity: 0.4;
#       animation: thinking-bounce 1.1s infinite;
#     }
#     .thinking-dot:nth-child(2) {
#       animation-delay: 0.16s;
#     }
#     .thinking-dot:nth-child(3) {
#       animation-delay: 0.32s;
#     }
#     @keyframes thinking-bounce {
#       0%, 80%, 100% {
#         transform: translateY(0);
#         opacity: 0.4;
#       }
#       40% {
#         transform: translateY(-4px);
#         opacity: 1;
#       }
#     }
#     @keyframes breathe {
#       0%, 100% {
#         transform: scale(1);
#         box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.2);
#       }
#       50% {
#         transform: scale(1.05);
#         box-shadow: 0 0 0 10px rgba(56, 189, 248, 0);
#       }
#     }
#     @keyframes spin {
#       to {
#         transform: rotate(360deg);
#       }
#     }

#     .chat-body::-webkit-scrollbar {
#       width: 6px;
#     }
#     .chat-body::-webkit-scrollbar-thumb {
#       background: rgba(148, 163, 184, 0.4);
#       border-radius: 999px;
#     }
#   </style>
# </head>
# <body>
#   <div class="chat-wrapper">
#     <div class="chat-card">
#       <div class="chat-header">
#         <div class="chat-header-title">
#           <div class="chat-header-orb"></div>
#           <div>
#             <h1>OpenHarmony 助手 · 实时 HTML 对话</h1>
#             <span>LLM 返回的是完整 HTML 页面，这里直接嵌入预览 👀</span>
#           </div>
#         </div>
#       </div>

#       <div class="chat-body" id="chatBody">
#         <div class="bubble-row assistant">
#           <div class="bubble assistant-bubble">
#             你好，我是 OpenHarmony 文档助手。<br/>
#             每次回答我都会生成一份完整 HTML 页面，并在下方直接预览。
#           </div>
#         </div>
#       </div>

#       <div class="chat-footer">
#         <form id="chatForm">
#           <div class="chat-footer-top">
#             <textarea
#               id="queryInput"
#               class="chat-input"
#               placeholder="输入你的问题，例如：请介绍 OpenHarmony 的技术架构。"
#             ></textarea>
#             <textarea
#               id="contextInput"
#               class="chat-context"
#               placeholder="可选：填 RAG 检索到的上下文"
#             ></textarea>
#           </div>
#           <div class="chat-footer-bottom">
#             <div class="hint">回车发送，Shift + 回车换行。</div>
#             <button type="submit" class="send-button" id="sendBtn">
#               <span>发送</span><span>➤</span>
#             </button>
#           </div>
#         </form>
#       </div>
#     </div>
#   </div>

#   <script>
#     // 与后端同域的话可以用相对路径；如果是单独 html 文件，改成 http://localhost:8000/chat
#     const API_URL = "/chat";

#     const chatBody = document.getElementById("chatBody");
#     const chatForm = document.getElementById("chatForm");
#     const queryInput = document.getElementById("queryInput");
#     const contextInput = document.getElementById("contextInput");
#     const sendBtn = document.getElementById("sendBtn");

#     let isWaiting = false;
#     let thinkingBubbleEl = null;

#     function appendUserMessage(text) {
#       const row = document.createElement("div");
#       row.className = "bubble-row user";

#       const bubble = document.createElement("div");
#       bubble.className = "bubble user-bubble";
#       bubble.textContent = text;

#       row.appendChild(bubble);
#       chatBody.appendChild(row);
#       chatBody.scrollTop = chatBody.scrollHeight;
#     }

#     function appendThinkingBubble() {
#       const row = document.createElement("div");
#       row.className = "bubble-row assistant";

#       const bubble = document.createElement("div");
#       bubble.className = "bubble assistant-bubble";

#       const container = document.createElement("div");
#       container.className = "thinking-container";

#       const avatar = document.createElement("div");
#       avatar.className = "thinking-avatar";

#       const dots = document.createElement("div");
#       dots.className = "thinking-dots";
#       for (let i = 0; i < 3; i++) {
#         const dot = document.createElement("div");
#         dot.className = "thinking-dot";
#         dots.appendChild(dot);
#       }

#       container.appendChild(avatar);
#       container.appendChild(dots);
#       bubble.appendChild(container);
#       row.appendChild(bubble);

#       chatBody.appendChild(row);
#       chatBody.scrollTop = chatBody.scrollHeight;

#       thinkingBubbleEl = bubble;
#     }

#     // ✅ 关键：把 LLM 返回的 HTML 用 iframe 渲染出来
#     function replaceThinkingWithAnswer(html) {
#       if (!thinkingBubbleEl) return;

#       const iframe = document.createElement("iframe");
#       iframe.className = "assistant-html-iframe";
#       iframe.srcdoc = html;
#       // 如果你想稍微限制一下 iframe 权限，可以打开 sandbox：
#       // iframe.setAttribute("sandbox", "allow-same-origin allow-scripts allow-forms allow-popups");

#       thinkingBubbleEl.innerHTML = "";
#       thinkingBubbleEl.appendChild(iframe);

#       chatBody.scrollTop = chatBody.scrollHeight;
#       thinkingBubbleEl = null;
#     }

#     async function sendMessage(query, ragContext) {
#       if (!query.trim() || isWaiting) return;

#       isWaiting = true;
#       sendBtn.disabled = true;

#       appendUserMessage(query);
#       appendThinkingBubble();

#       try {
#         const res = await fetch(API_URL, {
#           method: "POST",
#           headers: { "Content-Type": "application/json" },
#           body: JSON.stringify({
#             query: query,
#             rag_context: ragContext || "",
#           }),
#         });

#         if (!res.ok) {
#           throw new Error("HTTP " + res.status);
#         }

#         const data = await res.json();
#         replaceThinkingWithAnswer(data.html || "<p>（空响应）</p>");
#       } catch (err) {
#         console.error(err);
#         replaceThinkingWithAnswer("<p>抱歉，我这边出了一点错误，请稍后再试。</p>");
#       } finally {
#         isWaiting = false;
#         sendBtn.disabled = false;
#       }
#     }

#     chatForm.addEventListener("submit", (e) => {
#       e.preventDefault();
#       const query = queryInput.value.trim();
#       const ctx = contextInput.value.trim();
#       if (!query) return;

#       sendMessage(query, ctx);
#       queryInput.value = "";
#     });

#     queryInput.addEventListener("keydown", (e) => {
#       if (e.key === "Enter" && !e.shiftKey) {
#         e.preventDefault();
#         chatForm.requestSubmit();
#       }
#     });
#   </script>
# </body>
# </html>
# """


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    rag_context: Optional[str] = ""


class ChatResponse(BaseModel):
    html: str   # 提取出来的 html
    raw: str    # 原始模型输出


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
    """
    前端对话调用的接口：
    - 输入：用户 query + 可选的 rag_context（你以后接 RAG 可以用）
    - 输出：{ html, raw }
    """
    system_prompt = await get_system_prompt()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    system_prompt = system_prompt.replace("%%%DATE%%%", current_date)

    user_content = f"""
【参考上下文 (RAG Context)】:
{body.rag_context or ""}

【用户问题】:
{body.query}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    logger.info(f"[CHAT] Query: {body.query}")
    response_content = await call_llm(messages)

    if not response_content:
        return ChatResponse(
            html="<p>抱歉，我这边出了一点问题，请稍后再试。</p>",
            raw="LLM call failed or empty response.",
        )

    html_content = extract_html(response_content)
    # ==================== >>> 保存本次返回的 HTML 和原始内容 ====================
    try:
        # 保存目录：输出在项目下的 output/chat_html 下
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "chat_html")
        os.makedirs(save_dir, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # 简单用时间戳作为文件名，避免字符问题
        base_name = f"{ts}"

        # 仅提取后的 HTML
        html_path = os.path.join(save_dir, base_name + ".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 完整原始模型输出（含 markdown、说明等），便于排查
        raw_path = os.path.join(save_dir, base_name + ".raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(response_content)

        logger.info(f"[CHAT] Saved HTML to {html_path}, raw to {raw_path}")
    except Exception as e:
        logger.error(f"[CHAT] Failed to save HTML/RAW response: {e}")
    # ==================== ====================
    return ChatResponse(
        html=html_content,
        raw=response_content,
    )

# ==================== 保留你原来的 run_test_case 逻辑（可选） ====================

async def run_test_case(query, mock_rag_context, output_filename):
    """保留你的原测试逻辑，不影响 Web 接口"""
    system_prompt = await get_system_prompt()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    system_prompt = system_prompt.replace("%%%DATE%%%", current_date)

    user_content = f"""
【参考上下文 (RAG Context)】:
{mock_rag_context}

【用户问题】:
{query}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    logger.info(f"Testing Query: {query}")
    response_content = await call_llm(messages)

    if response_content:
        html_content = extract_html(response_content)
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Result saved to {output_path}")
    else:
        logger.error("No response received.")


if __name__ == "__main__":
    import uvicorn
    # 启动 Web API，前端请求 http://localhost:8000/chat
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)