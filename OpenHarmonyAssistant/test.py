import asyncio
import os
from openai import AsyncOpenAI
import loguru

# 配置 Logger
logger = loguru.logger

# 配置 Client (参考 translate.py)
# 注意：实际运行时请确保环境变量或此处 Key 有效
client = AsyncOpenAI(
    api_key="",
    base_url="https://openai.app.msh.team/v1",
    timeout=600.0,
)

MODEL_NAME = 'qwen3-coder' # 使用 translate.py 中的模型

async def get_system_prompt():
    """读取自定义的 OpenHarmony System Prompt"""
    # prompt_path = os.path.join(os.path.dirname(__file__), 'ours/openharmony_prompt.md')
    prompt_path = os.path.join(os.path.dirname(__file__), 'ours/step2_system_prompt_zh.md')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

async def call_llm(messages):
    """调用 LLM"""
    try:
        logger.info(f"Sending request to {MODEL_NAME}...")
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

def extract_html(content):
    """从 Markdown 代码块中提取 HTML"""
    if "```html" in content:
        return content.split("```html")[1].split("```")[0].strip()
    return content

async def run_test_case(query, mock_rag_context, output_filename):
    """运行单个测试用例"""
    system_prompt = await get_system_prompt()
    
    # 注入日期占位符
    import datetime
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    system_prompt = system_prompt.replace("%%%DATE%%%", current_date)

    # 构造 Prompt：包含 Context 和 用户 Query
    user_content = f"""
【参考上下文 (RAG Context)】:
{mock_rag_context}

【用户问题】:
{query}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    logger.info(f"Testing Query: {query}")
    response_content = await call_llm(messages)
    
    if response_content:
        html_content = extract_html(response_content)
        
        # 确保输出目录存在
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Result saved to {output_path}")
    else:
        logger.error("No response received.")

async def main():
    # --- 测试用例 1: 技术架构 ---
    query_1 = "请介绍 OpenHarmony 的技术架构，特别是内核层和框架层。"
    
    # 模拟的 RAG 检索结果 (摘自 OpenHarmony-Overview_zh.md)
    context_1 = """
    OpenHarmony整体遵从分层设计，从下向上依次为：内核层、系统服务层、框架层和应用层。
    
    1. **内核层**:
       - 内核子系统：采用多内核（Linux内核或者LiteOS）设计。
       - 驱动子系统：驱动框架（HDF）是系统硬件生态开放的基础。
    
    2. **系统服务层**: 是核心能力集合，包含分布式软总线、分布式数据管理等。
    
    3. **框架层**: 
       - 为应用开发提供了C/C++/JS等多语言的用户程序框架和Ability框架。
       - 适用于JS语言的ArkUI框架。
       - 各种软硬件服务对外开放的多语言框架API。
       
    图片资源:
    - 架构图 URL: https://gitee.com/openharmony/docs/raw/master/zh-cn/figures/1.png
    """
    
    await run_test_case(query_1, context_1, "openharmony_arch.html")

    # --- 测试用例 2: 解释部件化设计 ---
    query_2 = "OpenHarmony 的部件设计原则是什么？如何划分部件？"
    
    # 模拟的 RAG 检索结果 (摘自 OpenHarmony部件设计和开发指南.md)
    context_2 = """
    组件化、部件化、模块化是指软件基于组件、部件、模块解耦。
    
    **部件定义**: 系统能力的基本单元，以源码为划分依据，具有独立的文件和目录。
    
    **划分原则**:
    - 具备独立的代码目录。
    - 可独立编译出库或可执行文件。
    - 可独立测试和验证。
    
    **规则**:
    - 规则1.1: 部件应当实现独立自制原则，解耦。
    - 规则1.2: 禁止系统通用部件依赖特定芯片。
    - 规则1.5: 禁止部件间反向依赖、循环依赖。
    
    **SysCap (系统能力)**: 由部件提供，每个SysCap绑定一个或多个应用API。
    """
    
    await run_test_case(query_2, context_2, "openharmony_component.html")

if __name__ == "__main__":
    asyncio.run(main())

