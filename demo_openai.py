import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG

# 清除代理设置
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

def load_triples_json(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

os.environ['OPENAI_API_KEY'] = 'BQm_Gkd1EoTcHkJfVf31dTWfMIOsW3_mKIDfM5j-MvvwNM5jNl9XnLOjvNjEOuDiIWoKb-DIphdRWt2gOoNwBw'

def main():

    # Prepare datasets and evaluation
    # docs = [
    #     "Oliver Badman is a politician.",
    #     "George Rankin is a politician.",
    #     "Thomas Marwick is a politician.",
    #     "Cinderella attended the royal ball.",
    #     "The prince used the lost glass slipper to search the kingdom.",
    #     "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    #     "Erik Hort's birthplace is Montebello.",
    #     "Marina is bom in Minsk.",
    #     "Montebello is a part of Rockland County."
    # ]
    my_json_path = "outputs/Harmony_docs_zh_cn/markdown_parse/triples.json"
    my_json_data = load_triples_json(my_json_path)

    # save_dir = 'outputs/openai'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    # llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    # embedding_model_name = 'text-embedding-3-small'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    save_dir = 'outputs/Harmony_docs_zh_cn'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    # llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    llm_model_name = 'deepseek-v3.2-exp'  # 华为 ModelArts MaaS 模型名称
    # embedding_model_name = 'text-embedding-3-small'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    embedding_model_name = 'Qwen3-Embedding-4B'

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url='https://api.modelarts-maas.com/openai/v1',  # OpenAI 兼容接口
                        # embedding_base_url='https://api.vveai.com/v1',
                        embedding_model_name=embedding_model_name
                        # embedding_base_url='127.0.0.1:11434',
                        # embedding_model_name='dengcao/Qwen3-Embedding-8B:Q5_K_M'
                        )

    # ====== 重排序参数配置 ======
    # rerank_candidate_k: 送入LLM重排序的候选事实数量（默认50）
    #   - 增大此值可以弥补embedding不准确的问题
    #   - 但会增加LLM调用的token消耗
    # linking_top_k: 重排序后最终保留的事实数量（默认5）
    hipporag.global_config.rerank_candidate_k = 100  # 候选事实数量
    hipporag.global_config.linking_top_k = 10       # 最终保留的事实数量
    
    print(f"📋 重排序配置:")
    print(f"   - rerank_candidate_k (候选事实数): {hipporag.global_config.rerank_candidate_k}")
    print(f"   - linking_top_k (最终保留数): {hipporag.global_config.linking_top_k}")

    # Run indexing
    # hipporag.index_from_json(my_json_data)

    # return
    # Separate Retrieval & QA
    queries = [
        "What is OpenHarmony's purpose?",
        "What is OpenHarmony's architecture?",
        "What is OpenHarmony's core features?",
        "What is OpenHarmony's development model?",
        "What is OpenHarmony's development process?",
        "What is OpenHarmony's development tools?",
        "What is OpenHarmony's development environment?",
        "What is OpenHarmony's development language?",
        "What is OpenHarmony's development framework?",
    ]

    # For Evaluation
    # answers = [
    #     ["Politician"],
    #     ["By going to the ball."],
    #     ["Rockland County"]
    # ]

    # gold_docs = [
    #     ["George Rankin is a politician."],
    #     ["Cinderella attended the royal ball.",
    #      "The prince used the lost glass slipper to search the kingdom.",
    #      "When the slipper fit perfectly, Cinderella was reunited with the prince."],
    #     ["Erik Hort's birthplace is Montebello.",
    #      "Montebello is a part of Rockland County."]
    # ]

    # 使用带debug输出的检索方法，查看每一步的中间结果
    # num_to_retrieve 可以设置为较小的值以减少输出
    query_solutions = hipporag.retrieve_with_debug(
        queries=queries[:2],  # 先只测试前2个查询
        num_to_retrieve=10,   # 只检索10个文档（而不是默认的200个）
        verbose=True          # 开启详细输出
    )
    
    # 最终汇总
    print(f"\n{'#'*70}")
    print(f"# 最终检索结果汇总")
    print(f"{'#'*70}")
    total_tokens_estimate = 0
    for i, solution in enumerate(query_solutions):
        doc_tokens = sum(len(doc)//4 for doc in solution.docs) if solution.docs else 0
        total_tokens_estimate += doc_tokens
        print(f"\n查询 {i+1}: {solution.question}")
        print(f"  - 检索文档数: {len(solution.docs) if solution.docs else 0}")
        print(f"  - 估计tokens: ~{doc_tokens}")
        if solution.doc_scores is not None and len(solution.doc_scores) > 0:
            print(f"  - 分数范围: {min(solution.doc_scores):.4f} ~ {max(solution.doc_scores):.4f}")
    
    print(f"\n总token估计: ~{total_tokens_estimate}")
    
    # 暂时不执行QA，先看检索结果
    # query_solutions, response_messages, metadata = hipporag.rag_qa(queries=queries)
    # 
    # # 打印结果
    # for i, (query, solution) in enumerate(zip(queries, query_solutions)):
    #     print(f"\n===== 查询 {i+1} =====")
    #     print(f"问题: {query}")
    #     print(f"答案: {solution.answer}")
    #     print(f"检索到的文档数: {len(solution.retrieved_docs) if solution.retrieved_docs else 0}")

if __name__ == "__main__":
    main()
