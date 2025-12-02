#!/usr/bin/env python3
"""
检索功能演示脚本

该脚本展示如何使用HippoRAG进行检索和问答，支持两种模式：
1. 传统段落检索模式
2. 层次化索引检索模式

使用方法:
    python demo_retrieval.py --mode hierarchical --triples_json outputs/Harmony_docs_zh_cn/markdown_parse/triples.json
    python demo_retrieval.py --mode traditional
"""

import os
import json
import argparse
import logging
from typing import List, Optional

# 设置路径以便导入HippoRAG模块
import sys
sys.path.append('src')

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置API密钥（根据实际情况修改）
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', 'your-api-key-here')


def traditional_retrieval_example():
    """传统段落检索示例"""
    logger.info("=== 传统段落检索示例 ===")
    
    # 准备文档
    docs = [
        "HippoRAG是一个基于知识图谱的检索增强生成系统。",
        "它模拟人类海马体的记忆机制，支持多跳推理。",
        "系统支持从Markdown文档构建层次化知识图谱。",
        "检索功能包括事实检索、认知记忆和图搜索。",
        "问答功能基于检索到的文档生成答案。"
    ]
    
    # 配置
    config = BaseConfig(
        save_dir='outputs/demo_traditional',
        llm_name='gpt-4o-mini',
        embedding_model_name='nvidia/NV-Embed-v2',
        retrieval_top_k=3,
        qa_top_k=3
    )
    
    # 初始化HippoRAG
    hipporag = HippoRAG(global_config=config)
    
    # 索引文档
    logger.info("开始索引文档...")
    hipporag.index(docs)
    logger.info("索引完成")
    
    # 准备查询
    queries = [
        "HippoRAG是什么？",
        "检索功能包括哪些步骤？"
    ]
    
    # 检索
    logger.info("开始检索...")
    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=3)
    
    # 显示检索结果
    for i, result in enumerate(retrieval_results):
        logger.info(f"\n查询 {i+1}: {result.question}")
        logger.info(f"检索到的文档数量: {len(result.docs)}")
        for j, doc in enumerate(result.docs):
            logger.info(f"  文档 {j+1}: {doc[:100]}...")
    
    # 检索增强生成
    logger.info("\n开始问答...")
    qa_results, responses, metadata = hipporag.rag_qa(queries=queries)
    
    # 显示问答结果
    for i, result in enumerate(qa_results):
        logger.info(f"\n查询 {i+1}: {result.question}")
        logger.info(f"答案: {result.answer}")
        logger.info(f"使用的文档数: {len(result.docs)}")
    
    return qa_results


def hierarchical_retrieval_example(triples_json_path: str):
    """层次化索引检索示例"""
    logger.info("=== 层次化索引检索示例 ===")
    
    # 加载层次化JSON结构
    if not os.path.exists(triples_json_path):
        logger.error(f"文件不存在: {triples_json_path}")
        logger.info("请先运行知识库构建流程生成triples.json文件")
        return None
    
    logger.info(f"加载JSON结构: {triples_json_path}")
    with open(triples_json_path, 'r', encoding='utf-8') as f:
        json_structure = json.load(f)
    
    # 统计节点数量
    num_files = len([k for k in json_structure.keys() if k.startswith('file-')])
    logger.info(f"检测到 {num_files} 个文件节点")
    
    # 配置
    config = BaseConfig(
        save_dir='outputs/Harmony_docs_zh_cn',
        llm_name='gpt-4o-mini',
        embedding_model_name='Qwen/Qwen2.5-7B-Instruct',  # 根据实际情况修改
        retrieval_top_k=5,
        qa_top_k=5,
        linking_top_k=5
    )
    
    # 初始化HippoRAG
    hipporag = HippoRAG(global_config=config)
    
    # 执行层次化索引
    logger.info("开始层次化索引...")
    hipporag.index_from_json(json_structure)
    logger.info("索引完成")
    
    # 查看图统计信息（如果支持）
    try:
        graph_info = hipporag.get_graph_info()
        logger.info(f"图统计信息: {graph_info}")
    except:
        pass
    
    # 准备查询
    queries = [
        "如何配置API?",
        "代码示例在哪里?",
        "有哪些主要功能?"
    ]
    
    # 检索
    logger.info("开始检索...")
    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=5)
    
    # 显示检索结果
    for i, result in enumerate(retrieval_results):
        logger.info(f"\n查询 {i+1}: {result.question}")
        logger.info(f"检索到的文档数量: {len(result.docs)}")
        for j, doc in enumerate(result.docs[:3]):  # 只显示前3个
            if isinstance(doc, dict):
                content = doc.get('content', str(doc))[:100]
            else:
                content = str(doc)[:100]
            logger.info(f"  文档 {j+1}: {content}...")
    
    # 检索增强生成
    logger.info("\n开始问答...")
    qa_results, responses, metadata = hipporag.rag_qa(queries=queries)
    
    # 显示问答结果
    for i, result in enumerate(qa_results):
        logger.info(f"\n查询 {i+1}: {result.question}")
        logger.info(f"答案: {result.answer}")
        if hasattr(result, 'docs') and result.docs:
            logger.info(f"使用的文档数: {len(result.docs)}")
    
    return qa_results


def main():
    parser = argparse.ArgumentParser(description="HippoRAG检索功能演示")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['traditional', 'hierarchical'],
        default='traditional',
        help='检索模式: traditional (传统段落检索) 或 hierarchical (层次化索引检索)'
    )
    parser.add_argument(
        '--triples_json',
        type=str,
        default='outputs/Harmony_docs_zh_cn/markdown_parse/triples.json',
        help='层次化JSON文件路径（仅用于hierarchical模式）'
    )
    parser.add_argument(
        '--queries',
        type=str,
        nargs='+',
        help='自定义查询列表（可选）'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'traditional':
        # 传统检索示例
        if args.queries:
            # 如果提供了自定义查询，可以修改函数接受参数
            logger.info("注意: 传统模式当前使用示例查询")
        traditional_retrieval_example()
    else:
        # 层次化检索示例
        hierarchical_retrieval_example(args.triples_json)


if __name__ == "__main__":
    main()

