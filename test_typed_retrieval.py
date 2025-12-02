#!/usr/bin/env python3
"""
测试分类型检索方法 - 测试新版加权图搜索
"""
import os
import time

# 清除代理设置
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

os.environ['OPENAI_API_KEY'] = 'BQm_Gkd1EoTcHkJfVf31dTWfMIOsW3_mKIDfM5j-MvvwNM5jNl9XnLOjvNjEOuDiIWoKb-DIphdRWt2gOoNwBw'

from src.hipporag import HippoRAG

def main():
    # 使用与 demo_openai.py 相同的配置
    save_dir = 'outputs/Harmony_docs_zh_cn'
    llm_model_name = 'deepseek-v3.2-exp'
    embedding_model_name = 'Qwen3-Embedding-4B'
    
    print("="*70)
    print("🔧 初始化 HippoRAG")
    print("="*70)
    
    # 初始化 HippoRAG
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        llm_base_url='https://api.modelarts-maas.com/openai/v1',
        embedding_model_name=embedding_model_name
    )
    
    # 配置重排序参数
    hipporag.global_config.rerank_candidate_k = 100
    hipporag.global_config.linking_top_k = 10
    
    print("📋 重排序配置:")
    print(f"   - rerank_candidate_k (候选事实数): {hipporag.global_config.rerank_candidate_k}")
    print(f"   - linking_top_k (最终保留数): {hipporag.global_config.linking_top_k}")
    
    # 准备检索对象
    print("\n📊 准备检索对象...")
    hipporag.prepare_retrieval_objects()
    
    print("\n📈 索引统计:")
    print(f"   - 事实数量: {len(hipporag.fact_node_keys)}")
    print(f"   - chunk数量: {len(hipporag.passage_node_keys)}")
    print(f"   - code数量: {len(hipporag.code_node_keys) if hasattr(hipporag, 'code_node_keys') else 0}")
    print(f"   - table数量: {len(hipporag.table_node_keys) if hasattr(hipporag, 'table_node_keys') else 0}")
    print(f"   - 实体数量: {len(hipporag.entity_node_keys)}")
    print(f"   - 图节点数: {hipporag.graph.vcount()}")
    print(f"   - 图边数: {hipporag.graph.ecount()}")
    
    # 测试查询
    queries = ["OpenHarmony的架构是什么？", "OpenHarmony的核心功能有哪些？", "如何做贡献"]
    
    # 获取查询嵌入
    hipporag.get_query_embeddings(queries)
    
    for q_idx, query in enumerate(queries):
        print(f"\n{'#'*70}")
        print(f"# 查询 {q_idx+1}/{len(queries)}: {query}")
        print(f"{'#'*70}")
        
        # 第一步：获取事实和文件
        print("\n📋 第一步：事实检索 + 重排序")
        query_fact_scores = hipporag.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, rerank_log = hipporag.rerank_facts(query, query_fact_scores)
        
        print(f"   重排序后的事实数量: {len(top_k_facts)}")
        for i, fact in enumerate(top_k_facts[:5]):
            print(f"   [{i+1}] {fact}")
        
        # 获取相关文件
        top_files = []
        if hasattr(hipporag, 'file_node_keys') and len(hipporag.file_node_keys) > 0:
            query_file_scores, file_keys = hipporag.get_file_scores(query)
            if len(query_file_scores) > 0:
                _, top_files, _ = hipporag.rerank_files(query, query_file_scores, file_keys)
                print(f"\n   相关文件数量: {len(top_files)}")
                for i, f in enumerate(top_files[:3]):
                    print(f"   [{i+1}] {f.get('file_path', '')} (分数={f.get('score', 0):.4f})")
        
        # 第二步：加权图搜索
        print("\n" + "="*70)
        print("🆕 第二步：加权图搜索 (weighted_graph_search)")
        print("="*70)
        
        start_time = time.time()
        weighted_results = hipporag.weighted_graph_search(
            query=query,
            top_k_facts=top_k_facts,
            top_files=top_files,
            candidate_k=100,
            max_hop=3,
            alpha=0.3,   # 边权重系数
            beta=0.4,    # 节点相似度系数
            gamma=0.2,   # 路径衰减系数
            delta=0.1,   # 边类型加成系数
            verbose=True
        )
        search_time = time.time() - start_time
        
        print(f"\n⏱️ 加权搜索耗时: {search_time:.3f}s")
        
        # 第三步：V2图搜索（加权搜索 + 可选LLM重排序）
        print("\n" + "="*70)
        print("🚀 第三步：V2图搜索 (graph_neighbor_rerank_v2)")
        print("="*70)
        
        start_time = time.time()
        v2_results = hipporag.graph_neighbor_rerank_v2(
            query=query,
            top_k_facts=top_k_facts,
            top_files=top_files,
            candidate_k=100,
            chunk_top_k=10,
            table_top_k=5,
            code_top_k=5,
            max_hop=3,
            use_llm_rerank=False,  # 先不使用LLM重排序，直接看加权搜索效果
            alpha=0.3,
            beta=0.4,
            gamma=0.2,
            delta=0.1,
            verbose=True
        )
        v2_time = time.time() - start_time
        
        print(f"\n⏱️ V2搜索总耗时: {v2_time:.3f}s")
        
        # 打印结果摘要
        print("\n" + "#"*70)
        print("# 结果摘要")
        print("#"*70)
        
        print("\n🆕 加权搜索结果 (weighted_graph_search):")
        for content_type in ['chunk', 'table', 'code']:
            data = weighted_results.get(content_type, [])
            print(f"   {content_type}: {len(data)}个")
            if data:
                print(f"      Top-3分数: {[f'{item[1]:.4f}' for item in data[:3]]}")
        
        print("\n🚀 V2搜索结果 (graph_neighbor_rerank_v2):")
        for content_type in ['chunk', 'table', 'code']:
            data = v2_results.get(content_type, ([], [], []))
            contents, scores, keys = data
            print(f"   {content_type}: {len(contents)}个")
            if len(scores) > 0:
                print(f"      Top-3分数: {[f'{s:.4f}' for s in scores[:3]]}")

if __name__ == "__main__":
    main()
