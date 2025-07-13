"""
HippoRAG: 基于知识图谱的检索增强生成系统

本文件实现了HippoRAG框架的核心功能，这是一个受人类海马体记忆机制启发的
先进RAG（检索增强生成）系统。

================================================================================
系统架构概述
================================================================================

HippoRAG系统由以下核心组件构成：

1. 开放信息抽取(OpenIE)模块：
   - 从文档中自动提取结构化的实体-关系-实体三元组
   - 支持在线和离线两种处理模式
   - 使用大语言模型进行高质量的信息抽取

2. 多层次嵌入系统：
   - 段落嵌入：对原始文档段落进行向量化
   - 实体嵌入：对提取的实体进行向量化
   - 事实嵌入：对关系三元组进行向量化

3. 知识图谱构建：
   - 实体关系图：基于三元组建立实体间的连接
   - 段落-实体连接：建立文档段落与实体的关联
   - 同义词边：基于语义相似度的实体连接

4. 认知记忆机制：
   - 模拟人类海马体的记忆筛选功能
   - 使用DSPy过滤器进行智能事实重排序
   - 提高检索结果的相关性和准确性

5. 个性化PageRank搜索：
   - 在知识图谱上进行权重传播
   - 结合事实检索和密集检索的优势
   - 提供全局一致的文档排序

================================================================================
工作流程说明
================================================================================

索引阶段 (index):
1. 文档分割和向量化
2. OpenIE提取实体和关系
3. 构建多层次的嵌入存储
4. 建立知识图谱结构
5. 添加同义词边扩展连接

检索阶段 (retrieve):
1. 事实检索：查找与查询相关的事实三元组
2. 认知记忆：使用重排序器筛选高质量事实
3. 图搜索：基于PersonalizedPageRank在图上传播权重
4. 结果合成：结合图搜索和密集检索的结果

问答阶段 (rag_qa):
1. 基于检索结果构建上下文提示
2. 使用大语言模型生成答案
3. 提取和验证最终答案

================================================================================
核心创新点
================================================================================

1. 双重记忆机制：
   - 图结构记忆：存储实体关系的结构化知识
   - 向量记忆：存储语义相似度信息

2. 认知记忆筛选：
   - 模拟人类大脑的记忆筛选过程
   - 不仅考虑相似度，还考虑逻辑一致性

3. 多模态检索融合：
   - 结合符号化的图搜索和神经化的向量检索
   - 在准确性和召回率之间取得平衡

4. 增量更新支持：
   - 支持动态添加和删除文档
   - 维护数据一致性和图结构完整性

================================================================================
使用场景
================================================================================

- 复杂多跳问答：需要推理多个相关事实的问题
- 知识密集型任务：需要大量背景知识的推理任务  
- 长文档理解：需要在长文档中定位相关信息
- 实时知识更新：需要动态更新知识库的应用

作者：HippoRAG团队
版本：2.0
许可：请参考LICENSE文件
"""

import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store_v2 import EmbeddingStoreV2
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig
from .utils.misc_utils import compute_mdhash_id

import vllm
from vllm import LLM
import torch

logger = logging.getLogger(__name__)

# 设置全局logger配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 确保当前模块的logger也设置为INFO级别
logger.setLevel(logging.INFO)

class HippoRAG:
    """
    HippoRAG: 基于知识图谱的检索增强生成系统
    
    该类实现了一个完整的RAG系统，结合了开放信息抽取(OpenIE)、知识图谱构建、
    向量嵌入存储和个性化PageRank算法，用于高质量的文档检索和问答。
    
    核心特性:
    - 自动构建知识图谱：从文档中提取实体和关系三元组
    - 多层次嵌入：分别对段落、实体和事实进行向量化
    - 图搜索：基于个性化PageRank算法的检索
    - 认知记忆：模拟人类记忆机制的事实重排序
    """

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        初始化HippoRAG实例及其相关组件
        
        Args:
            global_config (BaseConfig, optional): 全局配置对象，包含所有系统设置
            save_dir (str, optional): 存储目录，用于保存模型输出和中间结果
            llm_model_name (str, optional): 大语言模型名称，用于OpenIE和问答
            llm_base_url (str, optional): LLM服务的基础URL（用于API调用）
            embedding_model_name (str, optional): 嵌入模型名称，用于向量化文本
            embedding_base_url (str, optional): 嵌入模型服务的基础URL
            azure_endpoint (str, optional): Azure OpenAI端点
            azure_embedding_endpoint (str, optional): Azure嵌入服务端点
            
        主要组件:
            - llm_model: 用于信息抽取和问答的语言模型
            - openie: 开放信息抽取模块（在线或离线模式）
            - graph: 知识图谱（使用igraph库）
            - embedding_model: 文本嵌入模型
            - *_embedding_store: 三个嵌入存储器（段落、实体、事实）
            - prompt_template_manager: 提示模板管理器
            - rerank_filter: 事实重排序过滤器
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # 如果指定了参数，则覆盖配置文件中的设置
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"HippoRAG init with config:\n  {_print_config}\n")

        # 创建模型特定的工作目录，避免不同模型间的冲突
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        # 初始化大语言模型
        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        # 根据配置选择在线或离线OpenIE模式
        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)

        # 初始化知识图谱
        self.graph = self.initialize_graph()

        # 初始化嵌入模型（离线模式下为None）
        # if self.global_config.openie_mode == 'offline':
        #     self.embedding_model = None
        # else:
        #     self.embedding_model = _get_embedding_model_class(
        #         embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
        #                                                                       embedding_model_name=self.global_config.embedding_model_name)
        
        self.embedding_model = LLM(model="../Qwen3-Embedding-4B", task="embed",dtype=torch.float16)


        # 初始化五个嵌入存储器：文件、段落、代码、表格、实体、事实
        self.file_embedding_store = EmbeddingStoreV2(self.embedding_model,
                                                   os.path.join(self.working_dir, "file_embeddings"),
                                                   self.global_config.embedding_batch_size, 'file') # file_id,摘要,embedding
        self.chunk_embedding_store = EmbeddingStoreV2(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk') # chunk_id,摘要,embedding
        self.code_embedding_store = EmbeddingStoreV2(self.embedding_model,
                                                     os.path.join(self.working_dir, "code_embeddings"),
                                                     self.global_config.embedding_batch_size, 'code') # code_id,代码摘要,embedding
        self.table_embedding_store = EmbeddingStoreV2(self.embedding_model,
                                                     os.path.join(self.working_dir, "table_embeddings"),
                                                     self.global_config.embedding_batch_size, 'table') # table_id,表格摘要,embedding
        self.entity_embedding_store = EmbeddingStoreV2(self.embedding_model, 
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity') # entity_id,实体,embedding
        self.fact_embedding_store = EmbeddingStoreV2(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact') # fact_id,事实,embedding
        

        # 初始化提示模板管理器
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        # OpenIE结果保存路径
        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        # 初始化重排序过滤器
        self.rerank_filter = DSPyFilter(self)

        # 系统状态标志
        self.ready_to_retrieve = False

        # 性能计时器
        self.ppr_time = 0  # PageRank算法耗时
        self.rerank_time = 0  # 重排序耗时
        self.all_retrieval_time = 0  # 总检索耗时

        # 实体到段落映射（用于增量更新）
        self.ent_node_to_chunk_ids: Optional[Dict[str, set]] = None

    def initialize_graph(self):
        """
        初始化知识图谱
        
        尝试从pickle文件加载预存的图谱，如果文件不存在或配置要求从头构建，
        则创建新的有向/无向图。
        
        Returns:
            ig.Graph: 初始化后的图谱对象
        
        图谱特性:
        - 支持有向和无向图
        - 自动持久化到pickle文件
        - 支持增量更新
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        # 如果不强制从头构建且存在预存图谱，则加载
        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def _batch_encode_texts(self, texts, instruction=None, norm=True):
        """
        使用新的嵌入模型API对文本进行批量编码
        
        Args:
            texts: 要编码的文本列表或单个文本
            instruction: 指令文本（暂时未使用）
            norm: 是否归一化（暂时未使用）
            
        Returns:
            embeddings: 嵌入向量
        """
        if isinstance(texts, str):
            texts = [texts]
        
        outputs = self.embedding_model.embed(texts)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        
        if len(texts) == 1:
            return embeddings[0]
        else:
            return embeddings

    def pre_openie(self,  docs: List[str]):
        """
        预处理OpenIE（离线模式）
        
        在离线模式下预先进行开放信息抽取，提取所有文档的实体和关系三元组。
        这个步骤是为了在后续的在线索引中使用预处理的结果。
        
        Args:
            docs (List[str]): 待处理的文档列表
            
        Note:
            执行完成后会抛出断言错误，提示用户运行在线索引进行后续检索
        """
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        # 获取尚未处理的文档段落
        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        # 加载已有的OpenIE结果
        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        # 对新文档进行OpenIE处理
        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        # 保存OpenIE结果
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        文档索引：HippoRAG的核心索引流程
        
        基于HippoRAG框架对给定文档进行索引，包括：
        1. 开放信息抽取(OpenIE) - 提取实体和关系三元组
        2. 向量嵌入 - 分别对段落、实体、事实进行向量化
        3. 知识图谱构建 - 构建实体关系图和段落连接图
        4. 同义词边扩展 - 基于相似度添加同义词连接
        
        Args:
            docs (List[str]): 待索引的文档列表，每个文档为一个字符串
            
        流程说明:
        1. OpenIE阶段：提取实体、关系和事实
        2. 嵌入阶段：对三种类型的内容分别进行向量化
        3. 图构建阶段：
           - 添加事实边（实体间的关系连接）
           - 添加段落边（段落与实体的连接）
           - 添加同义词边（相似实体间的连接）
        4. 图增强和保存
        """
        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        # 离线模式下先进行预处理
        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        # 将文档插入段落嵌入存储器
        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        # 加载已有OpenIE结果，确定需要处理的新段落
        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        # 对新段落进行OpenIE处理
        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        # 保存OpenIE结果
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        # 重新格式化OpenIE结果
        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)

        # 准备数据存储
        chunk_ids = list(chunk_to_rows.keys())

        # 处理三元组并提取实体节点
        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        # 对实体进行向量化编码
        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        # 对事实进行向量化编码
        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        # 构建知识图谱
        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}  # 节点间连接统计
        self.ent_node_to_chunk_ids = {}  # 实体到段落的映射

        # 添加事实边（实体间的关系连接）
        self.add_fact_edges(chunk_ids, chunk_triples)
        # 添加段落边（段落与实体的连接）
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            # 添加同义词边（基于相似度的实体连接）
            self.add_synonymy_edges()

            # 增强图谱并保存
            self.augment_graph()
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        文档删除：从所有数据结构中删除指定文档
        
        安全地从HippoRAG系统中删除文档，包括段落、相关实体和事实。
        注意：只删除仅出现在被删除文档中的三元组和实体，保持数据一致性。
        
        Args:
            docs_to_delete (List[str]): 待删除的文档列表
            
        删除策略:
        1. 删除文档段落
        2. 识别仅在被删除文档中出现的事实和实体
        3. 从嵌入存储器中删除相应数据
        4. 从知识图谱中删除对应节点
        5. 更新OpenIE结果文件
        
        保留策略:
        - 在其他文档中也出现的实体和事实会被保留
        - 确保知识图谱的完整性
        """

        #Making sure that all the necessary structures have been built.
        # 确保所有必要的结构都已构建完成
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        # 获取待删除段落的ID
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        #Find triples in chunks to delete
        # 查找待删除段落中的三元组
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        # 过滤掉在其他未删除段落中也出现的三元组
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        #Filter out entities that appear in unaltered chunks
        # 过滤掉在其他未删除段落中也出现的实体
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        # 保存更新后的OpenIE结果
        self.save_openie_results(all_openie_info_with_deletes)

        # 从嵌入存储器中删除数据
        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        # 从知识图谱中删除节点
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: Optional[int] = None,
                 gold_docs: Optional[List[List[str]]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        HippoRAG检索：模拟人类记忆的多步骤检索过程
        
        实现基于HippoRAG框架的文档检索，模拟人类大脑海马体的记忆机制：
        1. 事实检索 - 基于查询找到相关事实
        2. 认知记忆 - 使用重排序器改进事实选择（模拟人类记忆筛选）
        3. 密集段落评分 - 传统向量相似度检索
        4. 个性化PageRank重排序 - 基于图结构的全局排序
        
        Args:
            queries (List[str]): 查询字符串列表
            num_to_retrieve (int, optional): 每个查询返回的文档数量，
                默认使用配置中的retrieval_top_k值
            gold_docs (List[List[str]], optional): 金标准文档列表，用于评估
                
        Returns:
            List[QuerySolution] 或 Tuple[List[QuerySolution], Dict]:
                如果未启用检索评估，返回QuerySolution对象列表
                如果启用评估，额外返回包含评估指标的字典
                
        检索流程详解:
        1. 事实检索阶段：
           - 使用查询向量与事实向量进行相似度计算
           - 获取最相关的候选事实
           
        2. 认知记忆阶段（Recognition Memory）：
           - 使用DSPy过滤器对事实进行重排序
           - 模拟人类记忆中的事实筛选过程
           - 提高事实的相关性和准确性
           
        3. 图搜索阶段：
           - 基于选中的事实确定相关实体
           - 使用个性化PageRank算法在知识图谱上传播权重
           - 结合密集检索分数进行最终排序
           
        4. 降级策略：
           - 如果重排序后没有相关事实，回退到纯密集检索
           
        Note:
            长查询在重排序后可能没有相关事实，此时会默认使用密集段落检索结果
        """
        retrieve_start_time = time.time()  # Record start time
        # 记录检索开始时间

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k # 200

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        # 预处理：获取所有查询的向量嵌入
        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            # 第一步：事实检索 - 计算查询与事实的相似度分数
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            
            # 第二步：认知记忆 - 重排序事实以提高相关性
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            # 第三步：检索策略选择
            if len(top_k_facts) == 0:
                # 降级策略：如果没有相关事实，使用纯密集检索
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                # 主要策略：基于事实和图结构的混合检索
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)

            # 构建检索结果
            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time
        # 记录检索结束时间

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        # 性能统计日志
        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # Evaluate retrieval
        # 检索评估（如果提供了金标准文档）
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        检索增强生成问答：完整的HippoRAG问答流水线
        
        实现基于HippoRAG框架的端到端问答系统，结合先进的检索和生成能力：
        1. 检索阶段 - 使用HippoRAG检索相关文档（如果输入是字符串查询）
        2. 生成阶段 - 基于检索文档使用LLM生成答案
        3. 评估阶段 - 可选的检索和答案质量评估
        
        Args:
            queries (List[Union[str, QuerySolution]]): 查询列表，可以是：
                - 字符串：需要先进行检索，然后问答
                - QuerySolution对象：已包含检索结果，直接进行问答
            gold_docs (Optional[List[List[str]]]): 金标准文档列表，用于检索评估
            gold_answers (Optional[List[List[str]]]): 金标准答案列表，用于问答评估
                
        Returns:
            根据评估配置返回不同的元组:
            基础返回 (总是包含):
                - List[QuerySolution]: 包含答案和元数据的查询解决方案列表
                - List[str]: LLM的原始响应消息列表
                - List[Dict]: 每个查询的元数据字典列表
            扩展返回 (如果启用评估):
                - Dict: 检索阶段的整体评估结果（如适用）
                - Dict: 问答评估指标（精确匹配和F1分数）
                
        工作流程:
        1. 检索阶段（如需要）：
           - 如果输入是字符串，使用HippoRAG检索相关文档
           - 如果输入是QuerySolution，跳过检索直接使用已有文档
           
        2. 问答生成阶段：
           - 基于检索到的文档构建提示
           - 使用LLM进行推理生成答案
           - 从LLM响应中提取最终答案
           
        3. 评估阶段（可选）：
           - 检索评估：使用Recall@K指标
           - 问答评估：使用精确匹配(EM)和F1分数
           
        特性:
        - 支持多种输入格式的灵活处理
        - 自动提示模板选择（基于数据集）
        - 综合的性能评估指标
        - 详细的元数据记录
        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)
        print(f"queries_solutions: {queries_solutions}", flush=True)
        print(f"all_response_message: {all_response_message}", flush=True)
        print(f"all_metadata: {all_metadata}", flush=True)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        问答推理：基于检索文档的生成式问答
        
        使用大语言模型对给定的查询解决方案进行问答推理，
        基于检索到的相关文档生成准确的答案。
        
        Args:
            queries (List[QuerySolution]): 包含查询和检索文档的解决方案列表
                
        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]:
                - 更新后的QuerySolution列表（包含生成的答案）
                - LLM的原始响应消息列表
                - 推理过程的元数据字典列表
                
        问答流程:
        1. 提示构建阶段：
           - 为每个查询构建包含检索文档的提示
           - 使用数据集特定的提示模板
           - 格式化检索段落为上下文
           
        2. LLM推理阶段：
           - 批量发送提示到语言模型
           - 并行处理多个查询以提高效率
           - 收集原始响应和元数据
           
        3. 答案提取阶段：
           - 从LLM响应中解析最终答案
           - 处理格式异常和解析错误
           - 更新QuerySolution对象
           
        提示模板策略:
        - 优先使用数据集特定的模板
        - 回退到通用MUSIQUE模板
        - 支持自定义角色映射
        
        异常处理:
        - 答案解析失败时使用完整响应
        - 记录详细的错误信息
        - 确保流程的鲁棒性
        """
        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        添加事实边：构建基于三元组的实体关系图
        
        基于从文档中提取的关系三元组，在知识图谱中建立实体间的连接。
        这是知识图谱构建的核心步骤，将结构化的事实转化为图的边。
        
        Args:
            chunk_ids (List[str]): 文档段落的唯一标识符列表
            chunk_triples (List[Tuple]): 每个段落对应的三元组列表
                每个三元组格式为: (主体实体, 关系, 客体实体)
                
        处理流程:
        1. 遍历每个段落的所有三元组
        2. 为三元组中的主体和客体实体生成哈希ID
        3. 建立实体间的双向连接统计
        4. 维护实体到段落的映射关系
        
        数据结构更新:
        - node_to_node_stats: 记录节点间的连接权重
        - ent_node_to_chunk_ids: 维护实体到包含它的段落ID集合的映射
        
        图结构特性:
        - 实体间关系建立双向连接（无向图特性）
        - 连接权重反映实体间关系的频次
        - 支持同一实体在多个段落中出现的情况
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        添加段落边：连接文档段落与其中的实体
        
        建立段落节点与实体节点之间的连接，使得在图搜索时能够
        从相关实体导航到包含这些实体的文档段落。
        
        Args:
            chunk_ids (List[str]): 段落标识符列表
            chunk_triple_entities (List[List[str]]): 每个段落中包含的实体列表
                
        Returns:
            int: 新添加到图中的段落节点数量
            
        处理逻辑:
        1. 检查当前图中已存在的节点
        2. 为每个新段落建立与其实体的连接
        3. 设置固定权重(1.0)表示段落-实体关联
        
        重要性:
        - 实现了从事实检索到段落检索的桥梁
        - 支持基于实体相关性的段落排序
        - 为个性化PageRank提供段落权重传播路径
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        添加同义词边：基于语义相似度扩展图连接
        
        通过计算实体间的向量相似度，识别并连接语义相似的实体，
        增强知识图谱的连通性和检索的召回率。
        
        处理流程:
        1. 获取所有实体的向量嵌入
        2. 执行K近邻(KNN)搜索找到相似实体
        3. 基于相似度阈值过滤候选同义词
        4. 建立高质量的同义词连接
        
        过滤条件:
        - 实体长度 > 2个字符（过滤过短的实体）
        - 相似度分数 >= 阈值
        - 限制每个实体的同义词数量（≤ 100）
        
        配置参数:
        - synonymy_edge_topk: 每个实体的候选同义词数量
        - synonymy_edge_sim_threshold: 相似度阈值
        - synonymy_edge_query_batch_size: 查询批处理大小
        - synonymy_edge_key_batch_size: 键值批处理大小
        
        重要性:
        - 处理实体表述的多样性（同一概念的不同表达）
        - 提高检索的鲁棒性
        - 增强图的连通性，改善PageRank传播效果
        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold: # or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        加载已有OpenIE结果：支持增量更新的智能加载机制
        
        从指定文件加载已存在的OpenIE结果，并与新内容合并，同时标准化索引。
        如果文件不存在或配置为从头开始重建，则准备新条目进行处理。
        
        Args:

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        图增强：构建完整的知识图谱结构
        
        通过添加新节点和新边来扩展图结构，完成知识图谱的构建过程。
        这是索引流程的最后阶段，将所有收集的实体、段落和关系信息
        整合成一个完整的图结构。
        
        处理流程:
        1. 添加新节点 - 将所有类型的节点加入图中（文件、段落、代码、表格、实体、事实）
        2. 添加新边 - 建立节点间的连接关系（结构性边和语义边）
        3. 记录完成状态并输出图信息
        
        重要性:
        - 完成从数据到图结构的转换
        - 为后续的图搜索和PageRank计算提供基础
        - 确保图的完整性和一致性
        """

        # 添加所有新节点到图中
        self.add_new_nodes()
        # 添加所有新边到图中
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        添加新节点：将所有类型的节点批量加入图中
        
        从所有嵌入存储器中获取节点信息，与图中现有节点进行比较，
        识别并批量添加新节点。支持层次化结构的多种节点类型。
        
        处理逻辑:
        1. 获取图中现有节点列表
        2. 从所有嵌入存储器中获取节点信息
        3. 识别尚未添加到图中的新节点
        4. 批量添加新节点及其属性
        
        节点类型:
        - 文件节点：从file_embedding_store获取
        - 段落节点：从chunk_embedding_store获取
        - 代码节点：从code_embedding_store获取
        - 表格节点：从table_embedding_store获取
        - 实体节点：从entity_embedding_store获取
        - 事实节点：从fact_embedding_store获取
        
        优化特性:
        - 批量操作提高效率
        - 避免重复添加已存在的节点
        - 保持节点属性的完整性
        """

        # 获取图中现有节点，建立名称到节点的映射
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        # 从所有嵌入存储器中获取ID到行的映射
        all_node_stores = []
        
        # 获取各类型节点（如果存在数据）
        try:
            file_to_row = self.file_embedding_store.get_all_id_to_rows()
            if file_to_row:
                all_node_stores.append(file_to_row)
        except Exception as e:
            logger.debug(f"No file nodes to add: {e}")
        
        try:
            chunk_to_row = self.chunk_embedding_store.get_all_id_to_rows()
            if chunk_to_row:
                all_node_stores.append(chunk_to_row)
        except Exception as e:
            logger.debug(f"No chunk nodes to add: {e}")
            
        try:
            code_to_row = self.code_embedding_store.get_all_id_to_rows()
            if code_to_row:
                all_node_stores.append(code_to_row)
        except Exception as e:
            logger.debug(f"No code nodes to add: {e}")
            
        try:
            table_to_row = self.table_embedding_store.get_all_id_to_rows()
            if table_to_row:
                all_node_stores.append(table_to_row)
        except Exception as e:
            logger.debug(f"No table nodes to add: {e}")
            
        try:
            entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
            if entity_to_row:
                all_node_stores.append(entity_to_row)
        except Exception as e:
            logger.debug(f"No entity nodes to add: {e}")
            
        # try:
        #     fact_to_row = self.fact_embedding_store.get_all_id_to_rows()
        #     if fact_to_row:
        #         all_node_stores.append(fact_to_row)
        # except Exception as e:
        #     logger.debug(f"No fact nodes to add: {e}")

        # 合并所有节点信息
        node_to_rows = {}
        for store in all_node_stores:
            node_to_rows.update(store)

        # 准备新节点的属性字典
        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id  # 设置节点名称
            # 只处理不在现有节点中的新节点
            if node_id not in existing_nodes:
                # 为每个属性准备列表
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        # 如果有新节点，批量添加到图中
        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)
            logger.info(f"Added {len(next(iter(new_nodes.values())))} new nodes to graph")

    def add_new_edges(self):
        """
        添加新边：将节点间的连接关系加入图中
        
        处理node_to_node_stats中记录的所有边信息，验证边的有效性，
        并将有效的边批量添加到图结构中。支持多种类型的边。
        
        处理流程:
        1. 构建邻接表和逆邻接表
        2. 准备边的源节点、目标节点和权重信息
        3. 验证边的有效性（确保两端节点都存在）
        4. 批量添加有效边到图中
        
        边类型包括:
        - 结构性边：层次关系边（contains）、跳转边（jump）
        - 语义边：实体间的关系连接（基于三元组）
        - 同义词边：相似实体间的连接（基于相似度）
        - 段落边：段落与实体的连接
        
        验证机制:
        - 检查源节点和目标节点是否都存在于图中
        - 过滤自环边（源节点等于目标节点）
        - 记录无效边的警告信息
        
        数据结构:
        - graph_adj_list: 正向邻接表
        - graph_inverse_adj_list: 反向邻接表
        - 边权重信息保存在边属性中
        """

        # 构建邻接表和反向邻接表
        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        
        # 统计边的类型
        edge_type_counts = {
            'structural': 0,  # 结构性边
            'semantic': 0,    # 语义边
            'synonymy': 0,    # 同义词边
            'passage': 0      # 段落边
        }
        
        # 遍历所有节点间的统计信息
        for edge, weight in self.node_to_node_stats.items():
            # 跳过自环边
            if edge[0] == edge[1]: 
                continue
            
            # 构建邻接表
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            # 准备边信息
            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            
            # 判断边的类型
            edge_type = self._classify_edge_type(edge[0], edge[1])
            edge_type_counts[edge_type] += 1
            
            edge_metadata.append({
                "weight": weight,
                "type": edge_type
            })

        # 验证边的有效性并准备添加
        valid_edges, valid_weights = [], {"weight": [], "type": []}
        current_node_ids = set(self.graph.vs["name"])
        
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            # 检查源节点和目标节点是否都存在于图中
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                edge_type = edge_d.get("type", "unknown")
                valid_weights["weight"].append(weight)
                valid_weights["type"].append(edge_type)
            else:
                # 记录无效边的警告
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        
        # 批量添加有效边到图中
        if valid_edges:
            self.graph.add_edges(
                valid_edges,
                attributes=valid_weights
            )
            logger.info(f"Added {len(valid_edges)} edges to graph")
            logger.info(f"Edge type distribution: {edge_type_counts}")
        else:
            logger.warning("No valid edges to add to graph")
            
    def _classify_edge_type(self, source_id: str, target_id: str) -> str:
        """
        分类边的类型
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            
        Returns:
            str: 边的类型 ('structural', 'semantic', 'synonymy', 'passage')
        """
        # 结构性边：文件到段落、段落到代码/表格、段落到子段落
        if ((source_id.startswith('file-') and target_id.startswith('chunk-')) or
            (source_id.startswith('chunk-') and target_id.startswith('code-')) or
            (source_id.startswith('chunk-') and target_id.startswith('table-')) or
            (source_id.startswith('chunk-') and target_id.startswith('chunk-'))):
            return 'structural'
        
        # 段落边：段落到实体
        if source_id.startswith('chunk-') and target_id.startswith('entity-'):
            return 'passage'
        
        # 实体间的边需要进一步区分
        if source_id.startswith('entity-') and target_id.startswith('entity-'):
            edge_weight = self.node_to_node_stats.get((source_id, target_id), 0)
            
            # 同义词边：权重是相似度分数（通常是0-1之间的浮点数）
            # 这种边是通过add_synonymy_edges方法添加的
            if (isinstance(edge_weight, float) and 0 < edge_weight < 1):
                return 'synonymy'
            
            # 语义边：权重是整数（通常是三元组的共现次数）
            # 这种边是通过add_fact_edges方法添加的
            else:
                return 'semantic'
        
        # 默认为结构性边
        return 'structural'

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        获取图的详细信息：支持层次化结构的统计信息
        
        统计各类节点和连接的数量，包括：
        - 文件节点数量
        - 段落节点数量  
        - 代码块节点数量
        - 表格节点数量
        - 实体节点数量
        - 事实数量
        - 各类边的数量统计
        
        Returns:
            Dict: 包含图统计信息的字典
        """
        graph_info = {}

        # 获取各类节点数量
        try:
            file_nodes_keys = self.file_embedding_store.get_all_ids()
            graph_info["num_file_nodes"] = len(set(file_nodes_keys))
        except:
            graph_info["num_file_nodes"] = 0

        try:
            chunk_nodes_keys = self.chunk_embedding_store.get_all_ids()
            graph_info["num_chunk_nodes"] = len(set(chunk_nodes_keys))
        except:
            graph_info["num_chunk_nodes"] = 0

        try:
            code_nodes_keys = self.code_embedding_store.get_all_ids()
            graph_info["num_code_nodes"] = len(set(code_nodes_keys))
        except:
            graph_info["num_code_nodes"] = 0

        try:
            table_nodes_keys = self.table_embedding_store.get_all_ids()
            graph_info["num_table_nodes"] = len(set(table_nodes_keys))
        except:
            graph_info["num_table_nodes"] = 0

        try:
            entity_nodes_keys = self.entity_embedding_store.get_all_ids()
            graph_info["num_entity_nodes"] = len(set(entity_nodes_keys))
        except:
            graph_info["num_entity_nodes"] = 0

        # 计算总节点数
        graph_info["num_total_nodes"] = (graph_info["num_file_nodes"] + 
                                        graph_info["num_chunk_nodes"] +
                                        graph_info["num_code_nodes"] + 
                                        graph_info["num_table_nodes"] +
                                        graph_info["num_entity_nodes"])

        # 获取事实数量
        try:
            graph_info["num_extracted_facts"] = len(self.fact_embedding_store.get_all_ids())
        except:
            graph_info["num_extracted_facts"] = 0

        # 统计边的数量和类型
        if hasattr(self, 'node_to_node_stats') and self.node_to_node_stats:
            # 统计不同类型的边
            structure_edges = 0  # 结构性边
            semantic_edges = 0   # 语义边（实体关系）
            synonymy_edges = 0   # 同义词边
            
            all_node_sets = set()
            try:
                all_node_sets.update(chunk_nodes_keys)
            except:
                pass
            try:
                all_node_sets.update(entity_nodes_keys)
            except:
                pass
            
            for (node1, node2) in self.node_to_node_stats:
                # 判断边的类型
                if (node1.startswith('file-') or node1.startswith('chunk-') or 
                    node1.startswith('code-') or node1.startswith('table-')):
                    structure_edges += 1
                elif (node1.startswith('entity-') and node2.startswith('entity-')):
                    # 这里可以进一步区分是语义边还是同义词边
                    # 简化处理：假设实体间的边都是语义边
                    semantic_edges += 1
                else:
                    # 其他边归类为结构边
                    structure_edges += 1
            
            graph_info["num_structure_edges"] = structure_edges
            graph_info["num_semantic_edges"] = semantic_edges
            graph_info["num_total_edges"] = len(self.node_to_node_stats)
        else:
            graph_info["num_structure_edges"] = 0
            graph_info["num_semantic_edges"] = 0
            graph_info["num_total_edges"] = 0

        return graph_info

    def prepare_retrieval_objects(self):
        """
        准备检索对象：为快速检索初始化内存数据结构
        
        将磁盘存储的数据加载到内存中，构建高效的检索所需的各种映射和索引，
        确保检索过程的高性能执行。
        
        初始化内容:
        1. 查询嵌入缓存字典
        2. 节点键列表（实体、段落、事实）
        3. 图节点映射（名称到索引）
        4. 向量嵌入矩阵（实体、段落、事实）
        5. OpenIE结果和映射关系
        
        数据一致性检查:
        - 验证图节点数量与嵌入存储器的一致性
        - 检查缺失节点并自动修复
        - 确保所有映射关系的完整性
        
        性能优化:
        - 预加载所有向量到内存（numpy数组）
        - 构建快速查找的哈希映射
        - 避免检索时的磁盘I/O开销
        
        Note:
            此方法必须在执行任何检索操作前调用
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        # 初始化查询嵌入缓存字典，用于存储查询的向量表示
        # 分别缓存用于事实检索和段落检索的查询嵌入
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        # 从各个嵌入存储器中获取所有节点的键列表
        # 这些键用于后续的向量检索和图搜索
        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # 实体节点键列表
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # 段落节点键列表
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids()) # 事实节点键列表

        # 数据一致性检查：验证图中的节点数量与嵌入存储器中的节点数量是否匹配
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # 如果图为空但存在节点数据，需要重新构建图
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # 创建节点名称到图顶点索引的映射关系
        # 这个映射用于在图搜索时快速定位节点
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # 节点键到图索引的映射
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # 检查所有实体和段落节点是否都在图中存在
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # 如果发现缺失节点，重新构建图结构
                self.add_new_nodes()
                self.save_igraph()
                # 更新映射关系
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # 创建节点键到图索引的快速查找列表
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # 实体节点的图索引列表
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # 段落节点的图索引列表
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # 如果映射创建失败，初始化为空列表
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        # 将所有向量嵌入加载到内存中的numpy数组，提高检索性能
        # 避免检索时频繁的磁盘I/O操作
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))
        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        # 加载现有的OpenIE结果，用于构建事实到文档的映射关系
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        # 初始化处理后的三元组到文档的映射字典
        # 用于跟踪每个事实三元组出现在哪些文档中
        self.proc_triples_to_docs = {}

        # 构建三元组到文档的映射关系
        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    # 对三元组进行文本处理并创建映射
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        # 如果实体节点到段落ID的映射不存在，需要重新构建
        if self.ent_node_to_chunk_ids is None:
            # 重新格式化OpenIE结果为标准化格式
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # 检查数据长度是否匹配，确保数据完整性
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                # 为缺失的段落创建空的OpenIE结果条目
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response="",  # 修复：使用空字符串而不是None
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response="",  # 修复：使用空字符串而不是None
                            metadata={},
                            triples=[]
                        )

            # 准备段落三元组数据，用于构建图边
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            # 初始化节点统计和实体到段落的映射
            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            # 添加基于事实的图边连接
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        # 标记检索对象已准备完毕，可以开始执行检索操作
        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            # debug;flush
            print(f"all_query_strings: {all_query_strings}", flush=True)
            query_embeddings_for_triple = self._batch_encode_texts(all_query_strings,
                                                                   instruction=get_query_instruction('query_to_fact'),
                                                                   norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self._batch_encode_texts(all_query_strings,
                                                                    instruction=get_query_instruction('query_to_passage'),
                                                                    norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        计算事实相关性分数：查询与事实库的语义匹配
        
        通过向量相似度计算查询与预存事实嵌入之间的标准化相似度分数，
        这是HippoRAG事实检索阶段的核心步骤。
        
        Args:
            query (str): 输入查询文本
            
        Returns:
            np.ndarray: 标准化的相似度分数数组，形状为(#facts,)
                分数范围为[0,1]，分数越高表示事实与查询越相关
                
        计算流程:
        1. 获取查询的向量嵌入（针对事实检索优化的指令）
        2. 计算查询向量与所有事实向量的点积相似度
        3. 使用min-max标准化将分数归一化到[0,1]区间
        
        异常处理:
        - 如果查询嵌入不存在，重新编码
        - 如果事实库为空，返回空数组并记录警告
        
        Note:
            使用专门的查询指令('query_to_fact')以优化查询-事实匹配效果
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self._batch_encode_texts(query,
                                                       instruction=get_query_instruction('query_to_fact'),
                                                       norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        密集段落检索：传统向量相似度检索
        
        基于预训练嵌入模型进行查询-段落的密集向量检索，
        作为HippoRAG的基础检索方法和降级策略。
        
        Args:
            query (str): 输入查询字符串
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - sorted_doc_ids: 按相关性排序的文档ID数组
                - sorted_doc_scores: 标准化的相关性分数数组
                
        检索流程:
        1. 查询编码：
           - 获取查询的向量嵌入（段落检索指令）
           - 如不存在则重新编码
           
        2. 相似度计算：
           - 计算查询向量与所有段落向量的点积
           - 处理维度匹配和数值稳定性
           
        3. 结果排序：
           - 使用min-max标准化归一化分数
           - 按分数降序排列文档
           
        应用场景:
        - HippoRAG的降级策略（无相关事实时）
        - 传统RAG系统的主要检索方法
        - 与图搜索结果的对比基准
        
        特点:
        - 高效的向量运算
        - 语义理解能力
        - 不依赖图结构
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self._batch_encode_texts(query,
                                                       instruction=get_query_instruction('query_to_passage'),
                                                       norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于事实实体的图搜索算法
        
        使用个性化PageRank (PPR)和密集检索模型，基于事实相似性和相关性计算文档分数。
        该函数将相关事实的信号与段落相似性和基于图的搜索相结合，以增强结果排名。

        Parameters:
            query (str): 需要进行相似性和相关性计算的输入查询字符串
            link_top_k (int): 从链接分数映射中包含的顶级短语数量，用于下游处理
            query_fact_scores (np.ndarray): 表示每个提供事实的事实-查询相似性的分数数组
            top_k_facts (List[Tuple]): 顶级事实列表，每个事实表示为主语、谓语和宾语的元组
            top_k_fact_indices (List[str]): query_fact_scores数组中顶级事实对应的索引或标识符
            passage_node_weight (float): 缩放图中段落分数的默认权重

        Returns:
            Tuple[np.ndarray, np.ndarray]: 包含两个数组的元组：
                - 第一个数组对应根据分数排序的文档ID
                - 第二个数组包含与排序文档ID关联的PPR分数
        """
        # 基于前面步骤选择的事实分配短语权重
        linking_score_map = {}  # 从短语到包含该短语的事实的平均分数的映射
        phrase_scores = {}  # 存储每个短语的所有事实分数，无论它们是否存在于知识图谱中
        phrase_weights = np.zeros(len(self.graph.vs['name']))  # 初始化短语权重数组
        passage_weights = np.zeros(len(self.graph.vs['name']))  # 初始化段落权重数组

        # 遍历顶级事实，为相关短语分配权重
        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()  # 事实的主语短语（转为小写）
            predicate_phrase = f[1].lower()  # 事实的谓语短语（转为小写）
            object_phrase = f[2].lower()  # 事实的宾语短语（转为小写）
            
            # 获取当前事实的相关性分数
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
                
            # 处理主语和宾语短语（谓语通常不作为实体处理）
            for phrase in [subject_phrase, object_phrase]:
                # 计算短语的哈希ID
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                # 获取短语在图中的节点ID
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                # 如果短语在图中存在
                if phrase_id is not None:
                    # 设置短语权重为事实分数
                    phrase_weights[phrase_id] = fact_score

                    # 检查实体到文档块的映射是否存在且非空
                    if self.ent_node_to_chunk_ids is not None and len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        # 根据包含该实体的文档块数量进行权重归一化
                        phrase_weights[phrase_id] /= len(self.ent_node_to_chunk_ids[phrase_key])

                # 记录短语分数用于后续平均值计算
                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        # 计算每个短语的平均事实分数
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        # 如果指定了链接top-k，则过滤权重以保留顶级短语
        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)

        # 根据选择的密集检索模型获取段落分数
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        # 对DPR分数进行最小-最大归一化
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        # 为每个检索到的段落分配权重
        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]  # 获取段落节点键
            passage_dpr_score = normalized_dpr_sorted_scores[i]  # 获取归一化的DPR分数
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]  # 获取段落在图中的节点ID
            # 设置段落权重（乘以权重因子）
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            # 获取段落文本内容
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            # 将段落文本和分数添加到链接分数映射中
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        # 将短语权重和段落权重合并为一个数组用于PPR计算
        node_weights = phrase_weights + passage_weights

        # 限制linking_score_map的大小，只保留前30个最高分数的项目
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        # 确保至少有一些权重大于0，否则PPR算法无法运行
        assert sum(node_weights) > 0, f'在给定事实的图中未找到短语: {top_k_facts}'

        # 基于之前分配的段落和短语权重运行PPR算法
        ppr_start = time.time()  # 记录PPR开始时间
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()  # 记录PPR结束时间

        # 累加PPR计算时间
        self.ppr_time += (ppr_end - ppr_start)

        # 验证返回的文档数量与语料库大小一致
        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"文档概率长度 {len(ppr_sorted_doc_ids)} != 语料库长度 {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """
        事实重排序：认知记忆机制的核心实现
        
        模拟人类认知记忆中的事实筛选过程，使用DSPy过滤器对初步检索的
        事实进行智能重排序，提高事实的相关性和质量。
        
        Args:
            query (str): 输入查询文本
            query_fact_scores (np.ndarray): 事实相关性分数数组
            
        Returns:
            Tuple[List[int], List[Tuple], dict]:
                - top_k_fact_indices: 重排序后的事实索引列表
                - top_k_facts: 重排序后的事实三元组列表
                - rerank_log: 重排序过程的日志信息
                
        重排序流程:
        1. 基于分数选择候选事实（top-k选择）
        2. 从嵌入存储器获取事实内容
        3. 使用DSPy过滤器进行智能重排序
        4. 返回优化后的事实列表和处理日志
        
        认知记忆机制:
        - 模拟人类大脑海马体的记忆筛选功能
        - 不仅考虑相似度，还考虑事实的逻辑一致性
        - 过滤噪声事实，提升检索质量
        
        异常处理:
        - 处理空事实库的情况
        - 捕获重排序过程中的异常
        - 提供详细的错误日志
        
        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # 加载配置参数
        link_top_k: int = self.global_config.linking_top_k
        
        # 检查是否有可用的事实进行重排序
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # 根据分数获取前k个事实
            if len(query_fact_scores) <= link_top_k:
                # 如果事实数量少于请求数量，使用所有事实
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # 否则获取前k个事实（按分数降序排列）
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # 获取实际的事实ID列表
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            
            # 从嵌入存储器中获取事实内容
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            
            # 解析事实内容，将字符串转换为三元组
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            # 使用DSPy过滤器对事实进行重排序
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            # 构建重排序日志，记录重排序前后的事实
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            # 异常处理：记录错误并返回空结果
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        个性化PageRank算法：图搜索的核心排序机制
        
        在知识图谱上运行个性化PageRank算法，计算节点的相关性分数。
        该算法模拟随机游走过程，从相关实体出发传播权重到文档段落。
        
        Args:
            reset_prob (np.ndarray): 重置概率分布，指定每个节点的初始权重
                数组大小必须等于图中节点数量，NaN和负值会被替换为0
            damping (float, optional): 阻尼因子，控制随机游走的探索程度
                取值范围[0,1]，默认0.5。值越高，越依赖图结构；值越低，越依赖初始权重
                
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - sorted_doc_ids: 按相关性分数降序排列的文档段落ID数组
                - sorted_doc_scores: 对应的相关性分数数组
                
        算法特性:
        1. 个性化重置：基于查询相关的实体设置重置概率
        2. 权重传播：在知识图谱上传播相关性权重
        3. 全局排序：考虑图的全局结构进行排序
        
        实现细节:
        - 使用igraph的高效prpack实现
        - 支持有向和无向图
        - 考虑边权重进行加权传播
        - 仅返回文档段落节点的分数
        
        应用场景:
        - HippoRAG的最终排序阶段
        - 结合事实检索和密集检索的结果
        - 提供全局一致的文档排序
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

    def index_from_json(self, json_structure: Dict[str, Any]):
        """
        从层次化JSON结构索引：支持复杂文档结构的HippoRAG索引
        
        基于层次化JSON结构构建知识图谱，支持文件、段落、代码块、表格和实体等多种节点类型：
        1. 文件摘要生成和向量化
        2. 段落摘要生成和向量化  
        3. 代码块和表格的摘要及向量化
        4. 实体提取和关系构建
        5. 构建多层次的图结构连接
        
        Args:
            json_structure (Dict[str, Any]): 层次化的JSON文档结构
                格式如用户描述的triples.json结构
                
        JSON结构说明:
        - file-xxxx: 文件节点
          - abstract: 文件摘要
          - file_path: 文件路径
          - content: 文件内容
          - chunks: 一级切分的段落
            - chunk-xxxx: 段落节点
              - abstract: 段落摘要
              - content: 段落内容
              - jump: 超链接信息
              - codes: 代码块实体
              - tables: 表格实体
              - filter_chunk: 过滤后的段落内容及提取的实体关系
              - chunks: 子级段落（递归结构）
        """
        logger.info(f"Indexing from hierarchical JSON structure")

        # 收集所有节点信息
        all_files = []
        all_chunks = []
        all_codes = []
        all_tables = []
        all_entities = []
        all_facts = []
        
        # 图结构连接信息
        structure_edges = []  # 结构性边 (parent -> child, jump等)
        semantic_edges = []   # 语义边 (实体关系)
        
        # 递归处理JSON结构
        self._extract_nodes_from_json(json_structure, all_files, all_chunks, all_codes, 
                                     all_tables, all_entities, all_facts,
                                     structure_edges, semantic_edges)
        
        logger.info(f"Extracted {len(all_files)} files, {len(all_chunks)} chunks, "
                   f"{len(all_codes)} codes, {len(all_tables)} tables, "
                   f"{len(all_entities)} entities, {len(all_facts)} facts")

        # 对各类节点进行向量化编码 
        logger.info(f"Encoding files") # 7348
        if all_files:
            self.file_embedding_store.insert_strings([file[0] for file in all_files], [file[1] for file in all_files], [file[2] for file in all_files])
            
        logger.info(f"Encoding chunks") # 68225
        if all_chunks:
            self.chunk_embedding_store.insert_strings([chunk[0] for chunk in all_chunks], [chunk[1] for chunk in all_chunks], [chunk[2] for chunk in all_chunks])
            
        logger.info(f"Encoding codes") # 20776
        if all_codes:
            self.code_embedding_store.insert_strings([code[0] for code in all_codes], [code[1] for code in all_codes], [code[2] for code in all_codes])
            
        logger.info(f"Encoding tables") # 17117
        if all_tables:
            self.table_embedding_store.insert_strings([table[0] for table in all_tables], [table[1] for table in all_tables], [table[2] for table in all_tables])
            
        logger.info(f"Encoding entities") # 328956
        if all_entities:
            self.entity_embedding_store.insert_strings([entity[0] for entity in all_entities], [entity[1] for entity in all_entities], [entity[2] for entity in all_entities])
            
        logger.info(f"Encoding facts") # 542375
        if all_facts:
            self.fact_embedding_store.insert_strings([fact[0] for fact in all_facts], [fact[1] for fact in all_facts], [fact[2] for fact in all_facts])

        # 构建知识图谱
        logger.info(f"Constructing hierarchical graph")
        
        self.node_to_node_stats = {}  # 节点间连接统计
        self.ent_node_to_chunk_ids = {}  # 实体到段落的映射
        
        # 添加结构性边（层次关系、跳转关系等）
        self._add_structure_edges(structure_edges)
        
        # 添加语义边（实体关系）
        self._add_semantic_edges(semantic_edges)
        
        # 添加同义词边（基于相似度的实体连接）
        if len(all_entities) > 0:
            self.add_synonymy_edges()

        # 增强图谱并保存
        self.augment_graph()
        self.save_igraph()
        
        logger.info(f"Hierarchical indexing completed!")
        print(self.get_graph_info())

    def _extract_nodes_from_json(self, json_structure: Dict[str, Any], 
                                all_files: List[Tuple[str, str, str]], all_chunks: List[Tuple[str, str, str]],
                                all_codes: List[Tuple[str, str, str]], all_tables: List[Tuple[str, str, str]],
                                all_entities: List[Tuple[str, str, str]], all_facts: List[Triple],
                                structure_edges: List[Tuple[str, str, str, float]],
                                semantic_edges: List[Triple],
                                parent_id: str = None):
        """
        递归提取JSON结构中的所有节点和边信息
        
        Args:
            json_structure: JSON结构
            all_*: 各类节点的收集列表
            structure_edges: 结构性边列表 (source_id, target_id, edge_type, weight)
            semantic_edges: 语义边列表 (三元组)
            parent_id: 父节点ID
        """
        for node_id, node_data in json_structure.items():
            if node_id.startswith('file-'):
                # 处理文件节点
                file_abstract = node_data.get('abstract', '')
                file_content = node_data.get('content','')
                if file_abstract and file_content:
                    all_files.append((node_id, file_content, file_abstract))
                    
                # 处理文件的chunk子节点
                chunks_data = node_data.get('chunks', {})
                if chunks_data:
                    self._extract_nodes_from_json(chunks_data, all_files, all_chunks,
                                                 all_codes, all_tables, all_entities, all_facts,
                                                 structure_edges, semantic_edges, parent_id=node_id)
                    
            elif node_id.startswith('chunk-'):
                # 处理段落节点
                chunk_abstract = node_data.get('abstract', '')
                chunk_content = node_data.get('content', '')
                
                # 使用摘要或内容作为嵌入文本
                # embed_text = chunk_abstract if chunk_abstract else chunk_content
                if chunk_abstract and chunk_content:
                    all_chunks.append((node_id, chunk_content, chunk_abstract))
                
                # 添加段落到父节点的边
                if parent_id:
                    structure_edges.append((parent_id, node_id, 'contains', 1.0))
                
                # 处理跳转关系
                jump_data = node_data.get('jump', {})
                for jump_file_id, jump_info in jump_data.items():
                    structure_edges.append((node_id, jump_file_id, 'jump', 1.0))
                
                # 处理代码块
                codes_data = node_data.get('codes', {})
                for code_id, code_info in codes_data.items():
                    code_abstract = code_info.get('abstract', '')
                    code_content = code_info.get('content', '')
                    # embed_text = code_abstract if code_abstract else code_content
                    if code_abstract and code_content:
                        all_codes.append((code_id,code_content,code_abstract))
                        structure_edges.append((node_id, code_id, 'contains', 1.0))
                
                # 处理表格
                tables_data = node_data.get('tables', {})
                for table_id, table_info in tables_data.items():
                    table_abstract = table_info.get('abstract', '')
                    table_content = table_info.get('content', '')
                    # embed_text = table_abstract if table_abstract else table_content
                    if table_abstract and table_content:
                        all_tables.append((table_id,table_content,table_abstract))
                        structure_edges.append((node_id, table_id, 'contains', 1.0))
                
                # 处理过滤后的chunk中的实体和关系
                filter_chunk = node_data.get('filter_chunk', {})
                if filter_chunk:
                    # 初始化实体到段落的映射（如果为None）
                    if self.ent_node_to_chunk_ids is None:
                        self.ent_node_to_chunk_ids = {}
                    
                    # 从三元组中提取实体
                    triples = filter_chunk.get('extracted_triples', [])
                    entities_from_triples = set()
                    
                    # 收集所有三元组中的实体
                    for triple in triples:
                        if isinstance(triple, list) and len(triple) == 3:
                            subject, predicate, obj = triple
                            entities_from_triples.add(str(subject))
                            entities_from_triples.add(str(obj))
                    
                    # 构建实体定义映射
                    entity_definitions = {}
                    extracted_entities = filter_chunk.get('extracted_entities', [])
                    for entity_info in extracted_entities:
                        if isinstance(entity_info, list) and len(entity_info) >= 2:
                            entity_name, entity_desc = str(entity_info[0]), str(entity_info[1])
                            entity_definitions[entity_name] = entity_desc
                    
                    # 处理从三元组中提取的实体
                    for entity_name in entities_from_triples:
                        # 查找实体定义
                        entity_desc = entity_definitions.get(entity_name, '')
                        
                        # 构建嵌入文本和哈希
                        if entity_desc:
                            embed_text = f"{entity_name}: {entity_desc}"
                        else:
                            embed_text = f"{entity_name}"
                        
                        entity_id = compute_mdhash_id(embed_text, "entity-")
                        all_entities.append((entity_id, embed_text, embed_text))
                        
                        # 添加实体到段落的边
                        structure_edges.append((node_id, entity_id, 'contains', 1.0))
                        
                        # 维护实体到段落的映射
                        if entity_id not in self.ent_node_to_chunk_ids:
                            self.ent_node_to_chunk_ids[entity_id] = set()
                        self.ent_node_to_chunk_ids[entity_id].add(node_id)
                    
                    # 提取关系三元组
                    for triple in triples:
                        if isinstance(triple, list) and len(triple) == 3:
                            processed_triple = (str(triple[0]), str(triple[1]), str(triple[2]))
                            embed_text = str(processed_triple)
                            all_facts.append((compute_mdhash_id(embed_text, "fact-"), embed_text, embed_text))
                            entity_desc_1 = entity_definitions.get(processed_triple[0], '')
                            entity_desc_2 = entity_definitions.get(processed_triple[2], '')
                            embed_text_1 = f"{processed_triple[0]}: {entity_desc_1}" if entity_desc_1 else processed_triple[0]
                            embed_text_2 = f"{processed_triple[2]}: {entity_desc_2}" if entity_desc_2 else processed_triple[2]
                            semantic_edges.append((compute_mdhash_id(embed_text_1, "entity-"), processed_triple[1], compute_mdhash_id(embed_text_2, "entity-")))
                
                # 递归处理子chunks
                sub_chunks = node_data.get('chunks', {})
                if sub_chunks:
                    self._extract_nodes_from_json(sub_chunks, all_files, all_chunks,
                                                 all_codes, all_tables, all_entities, all_facts,
                                                 structure_edges, semantic_edges, parent_id=node_id)

    def _add_structure_edges(self, structure_edges: List[Tuple[str, str, str, float]]):
        """
        添加结构性边到图中
        
        Args:
            structure_edges: 结构性边列表 (source_id, target_id, edge_type, weight)
        """
        logger.info(f"Adding {len(structure_edges)} structural edges")
        
        for source_id, target_id, edge_type, weight in structure_edges:
            self.node_to_node_stats[(source_id, target_id)] = weight
            
            # 对于包含关系，也添加反向边以便双向导航
            if edge_type == 'contains':
                self.node_to_node_stats[(target_id, source_id)] = weight

    def _add_semantic_edges(self, semantic_edges: List[Triple]):
        """
        添加语义边（实体关系）到图中
        
        Args:
            semantic_edges: 语义边列表（三元组）
        """
        logger.info(f"Adding {len(semantic_edges)} semantic edges")
        
        for triple in semantic_edges:
            if len(triple) == 3:
                subject_id, predicate, obj_id = triple
                
                # 添加双向连接（无向图）
                self.node_to_node_stats[(subject_id, obj_id)] = self.node_to_node_stats.get((subject_id, obj_id), 0.0) + 1
                self.node_to_node_stats[(obj_id, subject_id)] = self.node_to_node_stats.get((obj_id, subject_id), 0.0) + 1
