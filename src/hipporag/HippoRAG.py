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

================================================================================
层次化结构支持（新增功能）
================================================================================

基于index_from_json的层次化RAG-QA增强：

1. 多类型内容索引：
   - 文件级别索引：处理完整文档的摘要和内容
   - 段落级别索引：传统的文档分块处理
   - 代码块索引：专门处理代码片段的摘要和内容
   - 表格索引：处理结构化表格数据

2. 智能检索策略：
   - 自动检测内容结构类型（传统vs层次化）
   - 多类型内容的统一相似度计算
   - 基于内容类型的优化检索算法

3. 增强的问答体验：
   - 内容类型感知的提示构建
   - 支持不同类型内容的上下文整合
   - 改进的答案生成质量

4. 向后兼容性：
   - 完全兼容原有的段落检索模式
   - 自动降级到传统检索机制
   - 无缝的API过渡

使用方法：
```python
# 传统方式（仍然支持）
hippo_rag.index(docs)  # 段落索引
results = hippo_rag.rag_qa(queries)

# 层次化方式（新增）
hippo_rag.index_from_json(json_structure)  # 层次化索引
results = hippo_rag.rag_qa(queries)  # 自动使用层次化检索

# 完整示例
results = hippo_rag.hierarchical_rag_qa_example(json_structure, queries)
```

作者：HippoRAG团队
版本：2.1（层次化增强版）
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
        
        # 初始化嵌入模型，启用多GPU tensor并行
        # tensor_parallel_size: 使用所有可用GPU进行张量并行计算
        # 这样embedding计算会自动分布到所有GPU上
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        
        num_gpus = torch.cuda.device_count()
        logger.info(f"初始化Embedding模型，检测到 {num_gpus} 个GPU")
        
        self.embedding_model = LLM(
            model="../Qwen3-Embedding-4B", 
            task="embed",
            dtype=torch.float16,
            tensor_parallel_size=num_gpus,  # 使用所有GPU进行张量并行
            gpu_memory_utilization=0.8,  # GPU显存利用率
            trust_remote_code=True,
            max_model_len=8192,  # 根据模型支持的最大长度设置
        )


        # 初始化七个嵌入存储器：文件、段落、代码、表格、图片、实体、事实
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
        self.image_embedding_store = EmbeddingStoreV2(self.embedding_model,
                                                     os.path.join(self.working_dir, "image_embeddings"),
                                                     self.global_config.embedding_batch_size, 'image') # image_id,caption,embedding
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
        
        # 事实到段落映射（fact_id → chunk_id，fact来源的chunk）
        self.fact_to_chunk_id: Dict[str, str] = {}
        
        # 事实到实体映射（fact_id → (subject_entity_id, object_entity_id)）
        self.fact_to_entities: Dict[str, Tuple[str, str]] = {}
        
        # 节点元信息映射（node_id → metadata dict），用于存储面包屑导航等信息
        # 格式: {node_id: {"breadcrumb": "h1标题 > h2标题 > h3标题", "metadata": {...}}}
        self.node_id_to_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_node_metadata()  # 尝试加载已保存的元信息

    def _get_node_metadata_path(self) -> str:
        """获取节点元信息保存路径"""
        return os.path.join(self.working_dir, "node_metadata.json")
    
    def _load_node_metadata(self):
        """从文件加载节点元信息"""
        metadata_path = self._get_node_metadata_path()
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.node_id_to_metadata = json.load(f)
                logger.info(f"Loaded {len(self.node_id_to_metadata)} node metadata entries from {metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to load node metadata: {e}")
                self.node_id_to_metadata = {}
        else:
            self.node_id_to_metadata = {}
    
    def _save_node_metadata(self):
        """保存节点元信息到文件"""
        metadata_path = self._get_node_metadata_path()
        try:
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.node_id_to_metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.node_id_to_metadata)} node metadata entries to {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save node metadata: {e}")
    
    def _build_breadcrumb(self, metadata: Dict[str, str]) -> str:
        """
        从metadata构建面包屑导航字符串
        
        Args:
            metadata: 包含h1, h2, h3...的字典
            
        Returns:
            str: 面包屑字符串，如 "一级标题 > 二级标题 > 三级标题"
        """
        if not metadata:
            return ""
        
        breadcrumb_parts = []
        for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            if level in metadata:
                breadcrumb_parts.append(metadata[level])
        
        return " > ".join(breadcrumb_parts) if breadcrumb_parts else ""
    
    def get_node_breadcrumb(self, node_id: str) -> str:
        """
        获取节点的面包屑导航
        
        Args:
            node_id: 节点ID
            
        Returns:
            str: 面包屑字符串
        """
        if node_id in self.node_id_to_metadata:
            return self.node_id_to_metadata[node_id].get('breadcrumb', '')
        return ''

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

    def _convert_to_gitee_url(self, local_path: str, base_local: str = "/root/code/docs") -> str:
        """
        将本地路径转换为 Gitee URL
        
        Args:
            local_path: 本地文件路径
            base_local: 本地基础路径
            
        Returns:
            str: Gitee URL
            
        示例:
            /root/code/docs/zh-cn/figures/1.png 
            -> https://gitee.com/openharmony/docs/raw/master/zh-cn/figures/1.png
        """
        if local_path.startswith(('http://', 'https://', 'ftp://')):
            return local_path  # 已经是远程路径，直接返回
        
        # 提取相对路径部分
        if local_path.startswith(base_local):
            relative_path = local_path[len(base_local):].lstrip('/')
        else:
            # 如果不是以 base_local 开头，尝试提取 docs 之后的部分
            if '/docs/' in local_path:
                relative_path = local_path.split('/docs/', 1)[1]
            else:
                relative_path = local_path
        
        return f"https://gitee.com/openharmony/docs/raw/master/{relative_path}"

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
            
            # 第二步B：文件摘要检索和重排序（如果有文件索引）
            top_files = []
            if hasattr(self, 'file_node_keys') and len(self.file_node_keys) > 0:
                query_file_scores, file_keys = self.get_file_scores(query)
                if len(query_file_scores) > 0:
                    _, top_files, _ = self.rerank_files(query, query_file_scores, file_keys)
            
            rerank_end = time.time()
            self.rerank_time += rerank_end - rerank_start

            # 第三步：检索策略选择
            if len(top_k_facts) == 0 and len(top_files) == 0:
                # 降级策略：如果没有相关事实和文件，使用纯密集检索
                logger.info('No facts or files found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                # 主要策略：基于事实、文件和图结构的混合检索
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight,
                                                                                         top_files=top_files)

            # 构建检索结果
            if len(self.all_content_node_keys) > len(self.passage_node_keys):
                # 使用层次化检索结果
                top_k_docs = []
                for idx in sorted_doc_ids[:num_to_retrieve]:
                    content_key = self.all_content_node_keys[idx]
                    content_type = self.all_content_node_types[idx]
                    content_data = self._get_content_by_type_and_key(content_type, content_key)
                    top_k_docs.append(content_data)
            else:
                # 使用原始段落检索结果
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

    def retrieve_with_debug(self,
                            queries: List[str],
                            num_to_retrieve: Optional[int] = None,
                            verbose: bool = True) -> List[QuerySolution]:
        """
        带详细Debug输出的检索方法
        
        与retrieve方法相同，但在每个步骤输出详细的中间结果，便于调试和理解检索流程。
        
        Args:
            queries (List[str]): 查询字符串列表
            num_to_retrieve (int, optional): 每个查询返回的文档数量
            verbose (bool): 是否输出详细信息，默认True
                
        Returns:
            List[QuerySolution]: 检索结果列表
        """
        def debug_print(msg, data=None, max_items=5, max_chars=200):
            """辅助函数：格式化打印debug信息"""
            if not verbose:
                return
            print(f"\n{'='*60}")
            print(f"🔍 {msg}")
            print(f"{'='*60}")
            if data is not None:
                if isinstance(data, (list, tuple)):
                    print(f"  数量: {len(data)}")
                    for i, item in enumerate(data[:max_items]):
                        item_str = str(item)
                        if len(item_str) > max_chars:
                            item_str = item_str[:max_chars] + "..."
                        print(f"  [{i+1}] {item_str}")
                    if len(data) > max_items:
                        print(f"  ... 还有 {len(data) - max_items} 项")
                elif isinstance(data, dict):
                    print(f"  数量: {len(data)}")
                    for i, (k, v) in enumerate(list(data.items())[:max_items]):
                        v_str = str(v)
                        if len(v_str) > max_chars:
                            v_str = v_str[:max_chars] + "..."
                        print(f"  [{k}]: {v_str}")
                elif isinstance(data, np.ndarray):
                    print(f"  形状: {data.shape}")
                    print(f"  统计: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
                    if len(data) > 0:
                        top_indices = np.argsort(data)[-max_items:][::-1]
                        print(f"  Top-{max_items}分数: {[f'{data[i]:.4f}' for i in top_indices]}")
                else:
                    print(f"  {data}")
        
        retrieve_start_time = time.time()
        
        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k
        
        rerank_candidate_k = getattr(self.global_config, 'rerank_candidate_k', 50)
        file_rerank_candidate_k = getattr(self.global_config, 'file_rerank_candidate_k', 50)
        file_linking_top_k = getattr(self.global_config, 'file_linking_top_k', 5)
        debug_print("检索配置", {
            'num_to_retrieve': num_to_retrieve,
            'linking_top_k (最终保留事实数)': self.global_config.linking_top_k,
            'rerank_candidate_k (候选事实数)': rerank_candidate_k,
            'file_rerank_candidate_k (候选文件数)': file_rerank_candidate_k,
            'file_linking_top_k (最终保留文件数)': file_linking_top_k,
            'damping': self.global_config.damping,
            'passage_node_weight': self.global_config.passage_node_weight
        })
        
        if not self.ready_to_retrieve:
            debug_print("准备检索对象...")
            self.prepare_retrieval_objects()
        
        # 统计信息
        debug_print("索引统计", {
            '事实数量': len(self.fact_node_keys) if hasattr(self, 'fact_node_keys') else 0,
            '段落数量': len(self.passage_node_keys) if hasattr(self, 'passage_node_keys') else 0,
            '实体数量': len(self.entity_node_keys) if hasattr(self, 'entity_node_keys') else 0,
            '图节点数': self.graph.vcount() if hasattr(self, 'graph') else 0,
            '图边数': self.graph.ecount() if hasattr(self, 'graph') else 0,
            '层次化内容数': len(self.all_content_node_keys) if hasattr(self, 'all_content_node_keys') else 0
        })
        
        # 获取查询嵌入
        debug_print("获取查询嵌入...")
        self.get_query_embeddings(queries)
        
        retrieval_results = []
        
        for q_idx, query in enumerate(queries):
            print(f"\n{'#'*70}")
            print(f"# 查询 {q_idx + 1}/{len(queries)}: {query}")
            print(f"{'#'*70}")
            
            # ====== 第一步：事实检索 ======
            step1_start = time.time()
            debug_print("步骤1: 事实检索 (get_fact_scores)")
            query_fact_scores = self.get_fact_scores(query)
            step1_time = time.time() - step1_start
            
            if len(query_fact_scores) > 0:
                debug_print(f"事实相似度分数 (耗时: {step1_time:.3f}s)", query_fact_scores)
                
                # 显示将送入重排序的候选事实（使用rerank_candidate_k）
                display_k = min(rerank_candidate_k, len(query_fact_scores), 20)  # 最多显示20个
                top_indices = np.argsort(query_fact_scores)[-display_k:][::-1]
                print(f"\n  📋 Top-{display_k} 候选事实三元组 (将送入LLM重排序):")
                for i, idx in enumerate(top_indices):
                    fact_id = self.fact_node_keys[idx]
                    try:
                        fact_content = self.fact_embedding_store.get_row(fact_id)['content']
                        print(f"    [{i+1}] 分数={query_fact_scores[idx]:.4f} | {fact_content}")
                    except:
                        print(f"    [{i+1}] 分数={query_fact_scores[idx]:.4f} | (无法获取内容)")
                if rerank_candidate_k > display_k:
                    print(f"    ... 还有 {rerank_candidate_k - display_k} 个候选事实")
            else:
                debug_print("没有可用的事实进行评分")
            
            # ====== 第二步：认知记忆 - 事实重排序 ======
            step2_start = time.time()
            debug_print("步骤2: 认知记忆 - 事实重排序 (rerank_facts)")
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            step2_time = time.time() - step2_start
            
            print(f"\n  ⏱️ 重排序耗时: {step2_time:.3f}s")
            facts_before = rerank_log.get('facts_before_rerank', [])
            print(f"\n  📋 重排序前的候选事实 ({len(facts_before)}个):")
            display_before = min(len(facts_before), 10)  # 显示前10个
            for i, fact in enumerate(facts_before[:display_before]):
                print(f"    [{i+1}] {fact}")
            if len(facts_before) > display_before:
                print(f"    ... 还有 {len(facts_before) - display_before} 个")
            
            print(f"\n  📋 重排序后的事实 ({len(top_k_facts)}个) [最终用于图搜索]:")
            for i, fact in enumerate(top_k_facts):
                print(f"    [{i+1}] {fact}")
            
            # ====== 第二步B：文件摘要检索和重排序 ======
            step2b_start = time.time()
            top_files = []
            if hasattr(self, 'file_node_keys') and len(self.file_node_keys) > 0:
                debug_print("步骤2B: 文件摘要检索和重排序 (get_file_scores + rerank_files)")
                query_file_scores, file_keys = self.get_file_scores(query)
                
                if len(query_file_scores) > 0:
                    debug_print(f"文件相似度分数", query_file_scores)
                    
                    # 显示将送入LLM重排序的候选文件（使用file_rerank_candidate_k）
                    display_k = min(file_rerank_candidate_k, len(query_file_scores))
                    top_file_indices = np.argsort(query_file_scores)[-display_k:][::-1]
                    # 只显示前15个，避免输出过多
                    show_k = min(15, display_k)
                    print(f"\n  📂 Top-{display_k} 候选文件 (将送入LLM重排序，显示前{show_k}个):")
                    print(f"  {'-'*60}")
                    for i, idx in enumerate(top_file_indices[:show_k]):
                        try:
                            file_key = file_keys[idx]
                            row = self.file_embedding_store.get_row(file_key)
                            summary = row.get('summary', '')
                            file_path = row.get('file_path', '')
                            print(f"    [{i+1}] 分数={query_file_scores[idx]:.4f}")
                            print(f"        路径: {file_path}")
                            print(f"        摘要: {summary[:300]}{'...' if len(summary) > 300 else ''}")
                            print()
                        except Exception as e:
                            print(f"    [{i+1}] 分数={query_file_scores[idx]:.4f} | (无法获取信息: {e})")
                    if display_k > show_k:
                        print(f"    ... 还有 {display_k - show_k} 个候选文件")
                    
                    # 文件重排序
                    _, top_files, file_rerank_log = self.rerank_files(query, query_file_scores, file_keys)
                    
                    print(f"\n  📂 重排序后的文件 ({len(top_files)}个，目标保留{file_linking_top_k}个) [LLM筛选后，用于图搜索]:")
                    print(f"  {'-'*60}")
                    for i, f in enumerate(top_files):
                        print(f"    [{i+1}] 分数={f.get('score', 0):.4f}")
                        print(f"        路径: {f.get('file_path', '')}")
                        print(f"        摘要: {f.get('summary', '')[:300]}{'...' if len(f.get('summary', '')) > 300 else ''}")
                        print()
            else:
                debug_print("没有文件索引，跳过文件检索")
            
            step2b_time = time.time() - step2b_start
            self.rerank_time += step2_time + step2b_time
            
            # ====== 第三步：图邻居搜索 + 分类型LLM重排序 ======
            step3_start = time.time()
            entities_for_search = set()
            
            if len(top_k_facts) == 0 and len(top_files) == 0:
                debug_print("步骤3: 降级策略 - 分类型DPR检索 (_dpr_by_type)")
                type_results = self._dpr_by_type(query)
                retrieval_method = "DPR by Type"
                # 构建各类型结果
                chunk_result = self._build_typed_result('chunk', type_results.get('chunk'), num_to_retrieve)
                table_result = self._build_typed_result('table', type_results.get('table'), num_to_retrieve // 2)
                code_result = self._build_typed_result('code', type_results.get('code'), num_to_retrieve // 2)
            else:
                debug_print("步骤3: 图邻居搜索 + LLM重排序 (graph_neighbor_rerank)")
                
                # 显示将用于图搜索的实体
                for fact in top_k_facts:
                    entities_for_search.add(fact[0].lower())  # subject
                    entities_for_search.add(fact[2].lower())  # object
                print(f"\n  🔗 用于图搜索的实体 ({len(entities_for_search)}个):")
                for i, ent in enumerate(list(entities_for_search)[:10]):
                    print(f"    [{i+1}] {ent}")
                
                if top_files:
                    print(f"\n  📂 用于图搜索的文件 ({len(top_files)}个):")
                    for i, f in enumerate(top_files[:5]):
                        summary_preview = f.get('summary', '')[:150]
                        print(f"    [{i+1}] {f.get('file_path', '')} (分数={f.get('score', 0):.4f})")
                        print(f"        摘要: {summary_preview}{'...' if len(f.get('summary', '')) > 150 else ''}")
                
                # 使用新的图邻居搜索 + LLM重排序方法
                type_results = self.graph_neighbor_rerank(
                    query=query,
                    top_k_facts=top_k_facts,
                    top_files=top_files,
                    chunk_candidate_k=50,
                    table_candidate_k=20,
                    code_candidate_k=20,
                    chunk_top_k=num_to_retrieve,
                    table_top_k=num_to_retrieve // 2,
                    code_top_k=num_to_retrieve // 2,
                    max_hop=2,
                    verbose=True
                )
                retrieval_method = "Graph Neighbor + LLM Rerank"
                
                # 直接构建结果（graph_neighbor_rerank 返回的格式不同）
                chunk_data = type_results.get('chunk', ([], np.array([]), []))
                table_data = type_results.get('table', ([], np.array([]), []))
                code_data = type_results.get('code', ([], np.array([]), []))
                
                chunk_result = TypedContentResult(
                    content_type='chunk',
                    contents=chunk_data[0] if len(chunk_data[0]) > 0 else [],
                    scores=chunk_data[1] if len(chunk_data[1]) > 0 else np.array([]),
                    keys=chunk_data[2] if len(chunk_data[2]) > 0 else []
                )
                table_result = TypedContentResult(
                    content_type='table',
                    contents=table_data[0] if len(table_data[0]) > 0 else [],
                    scores=table_data[1] if len(table_data[1]) > 0 else np.array([]),
                    keys=table_data[2] if len(table_data[2]) > 0 else []
                )
                code_result = TypedContentResult(
                    content_type='code',
                    contents=code_data[0] if len(code_data[0]) > 0 else [],
                    scores=code_data[1] if len(code_data[1]) > 0 else np.array([]),
                    keys=code_data[2] if len(code_data[2]) > 0 else []
                )
            
            step3_time = time.time() - step3_start
            
            debug_print(f"检索方法: {retrieval_method} (耗时: {step3_time:.3f}s)")
            
            # 打印分类型结果
            print(f"\n  {'='*50}")
            print(f"  📊 分类型检索结果")
            print(f"  {'='*50}")
            
            # Chunk 结果
            print(f"\n  📁 Chunk 结果 (共{len(chunk_result.contents)}个):")
            print(f"  {'-'*50}")
            for i in range(min(5, len(chunk_result.contents))):
                score = chunk_result.scores[i] if i < len(chunk_result.scores) else 0
                content = chunk_result.contents[i]
                preview = content[:150].replace('\n', ' ') + "..." if len(content) > 150 else content.replace('\n', ' ')
                print(f"    [{i+1}] 分数={score:.6f}, 长度={len(content)}字符")
                print(f"        {preview}")
            
            # Table 结果
            print(f"\n  📁 Table 结果 (共{len(table_result.contents)}个):")
            print(f"  {'-'*50}")
            for i in range(min(3, len(table_result.contents))):
                score = table_result.scores[i] if i < len(table_result.scores) else 0
                content = table_result.contents[i]
                preview = content[:150].replace('\n', ' ') + "..." if len(content) > 150 else content.replace('\n', ' ')
                print(f"    [{i+1}] 分数={score:.6f}, 长度={len(content)}字符")
                print(f"        {preview}")
            
            # Code 结果
            print(f"\n  📁 Code 结果 (共{len(code_result.contents)}个):")
            print(f"  {'-'*50}")
            for i in range(min(3, len(code_result.contents))):
                score = code_result.scores[i] if i < len(code_result.scores) else 0
                content = code_result.contents[i]
                preview = content[:150].replace('\n', ' ') + "..." if len(content) > 150 else content.replace('\n', ' ')
                print(f"    [{i+1}] 分数={score:.6f}, 长度={len(content)}字符")
                print(f"        {preview}")
            
            # 合并为兼容的 top_k_docs（用于返回 QuerySolution）
            top_k_docs = chunk_result.contents[:num_to_retrieve]
            
            # ====== 统计汇总 ======
            total_tokens = sum(len(doc)//4 for doc in top_k_docs)
            print(f"\n  📊 本次查询统计:")
            print(f"     - Chunk数: {len(chunk_result.contents)}, Table数: {len(table_result.contents)}, Code数: {len(code_result.contents)}")
            print(f"     - 总token估计: ~{total_tokens}")
            print(f"     - 步骤1 (事实检索): {step1_time:.3f}s")
            print(f"     - 步骤2 (事实重排序): {step2_time:.3f}s")
            print(f"     - 步骤2B(文件检索+重排): {step2b_time:.3f}s")
            print(f"     - 步骤3 (图邻居+重排序): {step3_time:.3f}s")
            print(f"     - 使用实体数: {len(entities_for_search)}")
            print(f"     - 使用文件数: {len(top_files)}")
            
            # 同时返回分类型结果
            retrieval_results.append(QuerySolution(
                question=query, 
                docs=top_k_docs, 
                doc_scores=chunk_result.scores[:num_to_retrieve] if len(chunk_result.scores) > 0 else np.array([])
            ))
        
        # 总体统计
        retrieve_end_time = time.time()
        total_time = retrieve_end_time - retrieve_start_time
        
        print(f"\n{'='*70}")
        print(f"📈 检索总体统计")
        print(f"{'='*70}")
        print(f"  查询数量: {len(queries)}")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  平均每查询: {total_time/len(queries):.2f}s")
        print(f"  重排序总耗时: {self.rerank_time:.2f}s")
        print(f"  PPR总耗时: {self.ppr_time:.2f}s")
        
        return retrieval_results

    def retrieve_v2(
        self,
        queries: List[str],
        fact_candidate_k: int = 100,
        file_candidate_k: int = 50,
        chunk_candidate_k: int = 50,
        fact_top_k: int = 10,
        file_top_k: int = 5,
        chunk_top_k: int = 5,
        spread_chunk_k: int = 15,
        spread_code_k: int = 10,
        spread_table_k: int = 10,
        spread_image_k: int = 10,
        final_chunk_k: int = 10,
        final_code_k: int = 3,
        final_table_k: int = 3,
        final_image_k: int = 3,
        generate_report: bool = False,
        verbose: bool = True
    ) -> List[Dict]:
        """
        新版检索流程 v2：基于图扩散的检索
        
        流程:
        1. Embedding候选获取: Fact(100), File(50), Chunk(50)
        2. LLM Rerank: Fact(10), File(5), Chunk(5)
        3. 图扩散: 必选Chunk + 扩散候选
        4. 最终LLM Rerank: Chunk(10), Code(3), Table(3), Image(3)
        5. (可选) LLM报告生成: 整合所有信息生成结构化报告
        
        Args:
            queries: 查询列表
            fact_candidate_k: Fact候选数量
            file_candidate_k: File候选数量
            chunk_candidate_k: Chunk候选数量
            fact_top_k: Fact重排后保留数量
            file_top_k: File重排后保留数量
            chunk_top_k: Chunk重排后保留数量
            spread_chunk_k: 扩散Chunk上限
            spread_code_k: 扩散Code上限
            spread_table_k: 扩散Table上限
            spread_image_k: 扩散Image上限
            final_chunk_k: 最终Chunk数量
            final_code_k: 最终Code数量
            final_table_k: 最终Table数量
            final_image_k: 最终Image数量
            generate_report: 是否生成LLM整合报告（供前端页面生成器使用）
            verbose: 是否输出详细信息
            
        Returns:
            List[Dict]: 每个查询的检索结果，如果generate_report=True则包含report字段
        """
        retrieve_start = time.time()
        
        if not self.ready_to_retrieve:
            if verbose:
                print("🔧 准备检索对象...")
            self.prepare_retrieval_objects()
        
        # 获取查询嵌入
        self.get_query_embeddings(queries)
        
        results = []
        
        for q_idx, query in enumerate(queries):
            if verbose:
                print(f"\n{'#'*70}")
                print(f"# 查询 {q_idx + 1}/{len(queries)}: {query}")
                print(f"{'#'*70}")
            
            query_result = self._retrieve_single_v2(
                query=query,
                fact_candidate_k=fact_candidate_k,
                file_candidate_k=file_candidate_k,
                chunk_candidate_k=chunk_candidate_k,
                fact_top_k=fact_top_k,
                file_top_k=file_top_k,
                chunk_top_k=chunk_top_k,
                spread_chunk_k=spread_chunk_k,
                spread_code_k=spread_code_k,
                spread_table_k=spread_table_k,
                spread_image_k=spread_image_k,
                final_chunk_k=final_chunk_k,
                final_code_k=final_code_k,
                final_table_k=final_table_k,
                final_image_k=final_image_k,
                verbose=verbose
            )
            
            # ========== 阶段5: LLM报告生成 (可选) ==========
            if generate_report:
                report_result = self._generate_report(
                    query=query,
                    chunks_data=query_result['chunks'],
                    codes_data=query_result['codes'],
                    tables_data=query_result['tables'],
                    images_data=query_result['images'],
                    verbose=verbose
                )
                query_result['report'] = report_result
                query_result['timing']['stage5_report'] = report_result.get('timing', 0)
                query_result['timing']['total'] += report_result.get('timing', 0)
            
            results.append(query_result)
        
        total_time = time.time() - retrieve_start
        if verbose:
            print(f"\n{'='*70}")
            print(f"📈 检索总体统计")
            print(f"{'='*70}")
            print(f"  查询数量: {len(queries)}")
            print(f"  总耗时: {total_time:.2f}s")
            print(f"  平均每查询: {total_time/len(queries):.2f}s")
            if generate_report:
                print(f"  包含LLM报告生成: 是")
        
        return results
    
    def _retrieve_single_v2(
        self,
        query: str,
        fact_candidate_k: int,
        file_candidate_k: int,
        chunk_candidate_k: int,
        fact_top_k: int,
        file_top_k: int,
        chunk_top_k: int,
        spread_chunk_k: int,
        spread_code_k: int,
        spread_table_k: int,
        spread_image_k: int,
        final_chunk_k: int,
        final_code_k: int,
        final_table_k: int,
        final_image_k: int,
        verbose: bool
    ) -> Dict:
        """单个查询的v2检索实现"""
        
        # ========== 阶段1: Embedding候选获取 ==========
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 阶段1: Embedding候选获取")
            print(f"{'='*60}")
        
        stage1_start = time.time()
        
        # 1.1 Fact检索
        query_fact_scores = self.get_fact_scores(query)
        if len(query_fact_scores) > 0:
            fact_top_indices = np.argsort(query_fact_scores)[-fact_candidate_k:][::-1]
            fact_candidate_ids = [self.fact_node_keys[i] for i in fact_top_indices]
            fact_candidate_scores = query_fact_scores[fact_top_indices]
        else:
            fact_candidate_ids = []
            fact_candidate_scores = np.array([])
        
        # 1.2 File检索
        if hasattr(self, 'file_node_keys') and len(self.file_node_keys) > 0:
            query_file_scores, file_keys = self.get_file_scores(query)
            if len(query_file_scores) > 0:
                file_top_indices = np.argsort(query_file_scores)[-file_candidate_k:][::-1]
                file_candidate_ids = [file_keys[i] for i in file_top_indices]
                file_candidate_scores = query_file_scores[file_top_indices]
            else:
                file_candidate_ids = []
                file_candidate_scores = np.array([])
        else:
            file_candidate_ids = []
            file_candidate_scores = np.array([])
        
        # 1.3 Chunk检索
        query_chunk_scores = self.get_passage_scores(query)
        if len(query_chunk_scores) > 0:
            chunk_top_indices = np.argsort(query_chunk_scores)[-chunk_candidate_k:][::-1]
            chunk_candidate_ids = [self.passage_node_keys[i] for i in chunk_top_indices]
            chunk_candidate_scores = query_chunk_scores[chunk_top_indices]
        else:
            chunk_candidate_ids = []
            chunk_candidate_scores = np.array([])
        
        stage1_time = time.time() - stage1_start
        
        if verbose:
            print(f"\n  📊 阶段1统计: 耗时 {stage1_time:.3f}s")
            
            # Fact候选详情
            print(f"\n  📋 Fact候选: {len(fact_candidate_ids)}个 (top-{fact_candidate_k})")
            print(f"  {'-'*60}")
            for i, (fid, score) in enumerate(zip(fact_candidate_ids[:10], fact_candidate_scores[:10] if len(fact_candidate_scores) > 0 else [])):
                try:
                    fact_row = self.fact_embedding_store.get_row(fid)
                    fact_content = fact_row.get('content', fact_row.get('summary', ''))
                    print(f"    [{i+1}] score={score:.4f}")
                    print(f"        ID: {fid}")
                    print(f"        内容: {fact_content}")
                except Exception as e:
                    print(f"    [{i+1}] score={score:.4f} | ID: {fid} | (获取失败: {e})")
            if len(fact_candidate_ids) > 10:
                print(f"    ... 还有 {len(fact_candidate_ids) - 10} 个")
            
            # File候选详情
            print(f"\n  📂 File候选: {len(file_candidate_ids)}个 (top-{file_candidate_k})")
            print(f"  {'-'*60}")
            for i, (fid, score) in enumerate(zip(file_candidate_ids[:5], file_candidate_scores[:5] if len(file_candidate_scores) > 0 else [])):
                try:
                    file_row = self.file_embedding_store.get_row(fid)
                    file_path = file_row.get('file_path', '')
                    summary = file_row.get('summary', file_row.get('content', ''))
                    print(f"    [{i+1}] score={score:.4f}")
                    print(f"        ID: {fid}")
                    print(f"        路径: {file_path}")
                    print(f"        摘要: {summary[:500]}{'...' if len(summary) > 500 else ''}")
                except Exception as e:
                    print(f"    [{i+1}] score={score:.4f} | ID: {fid} | (获取失败: {e})")
            if len(file_candidate_ids) > 5:
                print(f"    ... 还有 {len(file_candidate_ids) - 5} 个")
            
            # Chunk候选详情
            print(f"\n  📄 Chunk候选: {len(chunk_candidate_ids)}个 (top-{chunk_candidate_k})")
            print(f"  {'-'*60}")
            for i, (cid, score) in enumerate(zip(chunk_candidate_ids[:5], chunk_candidate_scores[:5] if len(chunk_candidate_scores) > 0 else [])):
                try:
                    chunk_row = self.chunk_embedding_store.get_row(cid)
                    summary = chunk_row.get('summary', '')
                    content = chunk_row.get('content', '')
                    print(f"    [{i+1}] score={score:.4f}")
                    print(f"        ID: {cid}")
                    print(f"        摘要: {summary[:300]}{'...' if len(summary) > 300 else ''}")
                    print(f"        内容预览: {content[:200].replace(chr(10), ' ')}{'...' if len(content) > 200 else ''}")
                except Exception as e:
                    print(f"    [{i+1}] score={score:.4f} | ID: {cid} | (获取失败: {e})")
            if len(chunk_candidate_ids) > 5:
                print(f"    ... 还有 {len(chunk_candidate_ids) - 5} 个")
        
        # ========== 阶段2: LLM Rerank ==========
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 阶段2: LLM Rerank")
            print(f"{'='*60}")
        
        stage2_start = time.time()
        
        # 2.1 Fact重排序
        top_k_fact_indices, top_k_facts, _ = self.rerank_facts(query, query_fact_scores)
        top_k_facts = top_k_facts[:fact_top_k]
        top_k_fact_ids = [self.fact_node_keys[i] for i in top_k_fact_indices[:fact_top_k]]
        
        if verbose:
            print(f"\n  📋 LLM重排序后的Fact ({len(top_k_facts)}个，目标保留{fact_top_k}个):")
            print(f"  {'-'*60}")
            for i, (fact, fact_id) in enumerate(zip(top_k_facts, top_k_fact_ids)):
                print(f"    [{i+1}] ID: {fact_id}")
                print(f"        三元组: {fact}")
                # 获取源chunk
                source_chunk = self.fact_to_chunk_id.get(fact_id, '未知')
                print(f"        来源Chunk: {source_chunk}")
            if len(top_k_facts) == 0:
                print(f"    (无)")
        
        # 2.2 File重排序
        if len(file_candidate_ids) > 0:
            _, top_files, _ = self.rerank_files(query, np.array(file_candidate_scores), file_candidate_ids)
            top_files = top_files[:file_top_k]
        else:
            top_files = []
        
        if verbose:
            print(f"\n  📂 LLM重排序后的File ({len(top_files)}个，目标保留{file_top_k}个):")
            print(f"  {'-'*60}")
            for i, f in enumerate(top_files):
                print(f"    [{i+1}] Key: {f.get('key', '')}")
                print(f"        路径: {f.get('file_path', '')}")
                print(f"        分数: {f.get('score', 0):.4f}")
                summary = f.get('summary', '')
                print(f"        摘要: {summary[:400]}{'...' if len(summary) > 400 else ''}")
            if len(top_files) == 0:
                print(f"    (无)")
        
        # 2.3 Chunk重排序
        if len(chunk_candidate_ids) > 0:
            # 获取候选chunk的内容
            chunk_contents = []
            for chunk_id in chunk_candidate_ids[:chunk_candidate_k]:
                try:
                    row = self.chunk_embedding_store.get_row(chunk_id)
                    content = row.get('content', row.get('summary', ''))
                    chunk_contents.append(content)
                except:
                    chunk_contents.append('')
            
            # LLM重排序
            top_chunk_indices, top_chunk_contents, _ = self._rerank_contents(
                query, chunk_contents, chunk_candidate_ids[:chunk_candidate_k],
                content_type='chunk', len_after_rerank=chunk_top_k
            )
            top_chunk_ids = [chunk_candidate_ids[i] for i in top_chunk_indices]
        else:
            top_chunk_ids = []
            top_chunk_contents = []
        
        if verbose:
            print(f"\n  📄 LLM重排序后的Chunk ({len(top_chunk_ids)}个，目标保留{chunk_top_k}个):")
            print(f"  {'-'*60}")
            for i, (chunk_id, content) in enumerate(zip(top_chunk_ids, top_chunk_contents)):
                print(f"    [{i+1}] ID: {chunk_id}")
                print(f"        长度: {len(content)}字符")
                # 显示摘要和内容预览
                content_preview = content[:300].replace('\n', ' ')
                print(f"        内容预览: {content_preview}{'...' if len(content) > 300 else ''}")
            if len(top_chunk_ids) == 0:
                print(f"    (无)")
        
        stage2_time = time.time() - stage2_start
        if verbose:
            print(f"\n  📊 阶段2统计: 耗时 {stage2_time:.3f}s")
            print(f"      Fact重排: {len(top_k_facts)}个")
            print(f"      File重排: {len(top_files)}个")
            print(f"      Chunk重排: {len(top_chunk_ids)}个")
        
        # ========== 阶段3: 图扩散 ==========
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 阶段3: 图扩散")
            print(f"{'='*60}")
        
        stage3_start = time.time()
        
        # 3.1 必选Chunk（Fact来源 + 阶段2高质量Chunk）
        must_have_chunks = set()
        
        # 从Fact获取源Chunk
        for fact_id in top_k_fact_ids:
            source_chunk = self.fact_to_chunk_id.get(fact_id)
            if source_chunk:
                must_have_chunks.add(source_chunk)
        
        # 添加阶段2的高质量Chunk
        must_have_chunks.update(top_chunk_ids)
        
        if verbose:
            print(f"\n  📌 必选Chunk: {len(must_have_chunks)}个")
            print(f"  {'-'*60}")
            fact_source_count = len([fid for fid in top_k_fact_ids if self.fact_to_chunk_id.get(fid)])
            print(f"      - Fact来源: {fact_source_count}个")
            print(f"      - 阶段2高质量: {len(top_chunk_ids)}个")
            print(f"\n      必选Chunk列表:")
            for i, chunk_id in enumerate(list(must_have_chunks)[:10]):
                try:
                    row = self.chunk_embedding_store.get_row(chunk_id)
                    summary = row.get('summary', row.get('content', ''))[:200]
                    print(f"        [{i+1}] {chunk_id}")
                    print(f"            摘要: {summary}...")
                except:
                    print(f"        [{i+1}] {chunk_id}")
            if len(must_have_chunks) > 10:
                print(f"        ... 还有 {len(must_have_chunks) - 10} 个")
        
        # 3.2 构建种子节点
        seed_entities = []
        for fact_id in top_k_fact_ids:
            entity_tuple = self.fact_to_entities.get(fact_id)
            if entity_tuple:
                subject_id, object_id = entity_tuple
                if subject_id:
                    seed_entities.append((subject_id, 0.8))
                if object_id:
                    seed_entities.append((object_id, 0.8))
        
        # 去重
        seed_entities = list(set(seed_entities))
        
        seed_files = [(f.get('key'), 0.9) for f in top_files if f.get('key')]
        
        if verbose:
            print(f"\n  🌱 种子节点:")
            print(f"  {'-'*60}")
            print(f"      Entity种子: {len(seed_entities)}个 (权重0.8)")
            for i, (ent_id, weight) in enumerate(seed_entities[:10]):
                try:
                    ent_row = self.entity_embedding_store.get_row(ent_id)
                    ent_name = ent_row.get('content', ent_row.get('summary', ent_id))
                    print(f"        [{i+1}] {ent_name} (ID: {ent_id})")
                except:
                    print(f"        [{i+1}] {ent_id}")
            if len(seed_entities) > 10:
                print(f"        ... 还有 {len(seed_entities) - 10} 个")
            
            print(f"\n      File种子: {len(seed_files)}个 (权重0.9)")
            for i, (file_key, weight) in enumerate(seed_files):
                try:
                    file_row = self.file_embedding_store.get_row(file_key)
                    file_path = file_row.get('file_path', file_key)
                    print(f"        [{i+1}] {file_path}")
                except:
                    print(f"        [{i+1}] {file_key}")
        
        # 3.3 图扩散
        query_embedding = self.query_to_embedding['passage'].get(query)
        if query_embedding is None:
            query_embedding = self._batch_encode_texts(query)
        query_embedding = np.array(query_embedding)
        
        spread_candidates = self.graph_spread_with_similarity(
            query=query,
            query_embedding=query_embedding,
            must_have_chunks=must_have_chunks,
            seed_entities=seed_entities,
            seed_files=seed_files,
            max_cost=1.0,
            max_chunk_candidates=spread_chunk_k,
            max_code_candidates=spread_code_k,
            max_table_candidates=spread_table_k,
            max_image_candidates=spread_image_k,
            verbose=verbose
        )
        
        stage3_time = time.time() - stage3_start
        if verbose:
            print(f"\n  📊 阶段3统计: 耗时 {stage3_time:.3f}s")
            print(f"  {'='*80}")
            print(f"      图扩散结果汇总:")
            print(f"        - Chunk扩散: {len(spread_candidates.get('chunk', []))}个 (上限{spread_chunk_k})")
            print(f"        - Code扩散: {len(spread_candidates.get('code', []))}个 (上限{spread_code_k})")
            print(f"        - Table扩散: {len(spread_candidates.get('table', []))}个 (上限{spread_table_k})")
            print(f"        - Image扩散: {len(spread_candidates.get('image', []))}个 (上限{spread_image_k})")
            
            # 详细展示所有扩散Chunk结果（含文件路径和面包屑）
            print(f"\n      扩散Chunk详情 (共{len(spread_candidates.get('chunk', []))}个):")
            print(f"      {'-'*70}")
            for i, (cid, score) in enumerate(spread_candidates.get('chunk', [])):
                try:
                    row = self.chunk_embedding_store.get_row(cid)
                    file_path = row.get('file_path', '')
                    summary = row.get('summary', '')[:200]
                    content = row.get('content', '')[:300]
                    breadcrumb = self.get_node_breadcrumb(cid)
                    print(f"        [{i+1}] 分数={score:.4f}")
                    print(f"            ID: {cid}")
                    print(f"            文件路径: {file_path}")
                    print(f"            📍 定位: {breadcrumb if breadcrumb else '(无层级信息)'}")
                    print(f"            摘要: {summary}...")
                    print(f"            内容预览: {content.replace(chr(10), ' ')[:200]}...")
                except:
                    print(f"        [{i+1}] 分数={score:.4f} | ID: {cid} | (获取详情失败)")
            
            # 详细展示所有扩散Code结果（含文件路径和面包屑）
            print(f"\n      扩散Code详情 (共{len(spread_candidates.get('code', []))}个):")
            print(f"      {'-'*70}")
            for i, (cid, score) in enumerate(spread_candidates.get('code', [])):
                try:
                    row = self.code_embedding_store.get_row(cid)
                    file_path = row.get('file_path', '')
                    summary = row.get('summary', '')[:200]
                    content = row.get('content', '')[:300]
                    breadcrumb = self.get_node_breadcrumb(cid)
                    print(f"        [{i+1}] 分数={score:.4f}")
                    print(f"            ID: {cid}")
                    print(f"            文件路径: {file_path}")
                    print(f"            📍 定位: {breadcrumb if breadcrumb else '(无层级信息)'}")
                    print(f"            摘要: {summary}...")
                    print(f"            代码预览: {content.replace(chr(10), ' ')[:200]}...")
                except:
                    print(f"        [{i+1}] 分数={score:.4f} | ID: {cid} | (获取详情失败)")
            
            # 详细展示所有扩散Table结果（含文件路径和面包屑）
            print(f"\n      扩散Table详情 (共{len(spread_candidates.get('table', []))}个):")
            print(f"      {'-'*70}")
            for i, (tid, score) in enumerate(spread_candidates.get('table', [])):
                try:
                    row = self.table_embedding_store.get_row(tid)
                    file_path = row.get('file_path', '')
                    summary = row.get('summary', '')[:200]
                    content = row.get('content', '')[:300]
                    breadcrumb = self.get_node_breadcrumb(tid)
                    print(f"        [{i+1}] 分数={score:.4f}")
                    print(f"            ID: {tid}")
                    print(f"            文件路径: {file_path}")
                    print(f"            📍 定位: {breadcrumb if breadcrumb else '(无层级信息)'}")
                    print(f"            摘要: {summary}...")
                    print(f"            表格预览: {content.replace(chr(10), ' ')[:200]}...")
                except:
                    print(f"        [{i+1}] 分数={score:.4f} | ID: {tid} | (获取详情失败)")
            
            # 详细展示所有扩散Image结果（含文件路径和面包屑）
            print(f"\n      扩散Image详情 (共{len(spread_candidates.get('image', []))}个):")
            print(f"      {'-'*70}")
            for i, (iid, score) in enumerate(spread_candidates.get('image', [])):
                try:
                    row = self.image_embedding_store.get_row(iid)
                    file_path = row.get('file_path', '')  # gitee_url
                    absolute_path = row.get('content', '')  # 本地绝对路径
                    caption = row.get('summary', '')
                    breadcrumb = self.get_node_breadcrumb(iid)
                    # 获取来源信息
                    meta_info = self.node_id_to_metadata.get(iid, {})
                    md_file_path = meta_info.get('md_file_path', '')
                    parent_chunk_id = meta_info.get('parent_chunk_id', '')
                    print(f"        [{i+1}] 分数={score:.4f}")
                    print(f"            ID: {iid}")
                    print(f"            图片Gitee URL: {file_path}")
                    print(f"            图片本地路径: {absolute_path}")
                    print(f"            📄 来源MD文件: {md_file_path}")
                    print(f"            📦 来源Chunk ID: {parent_chunk_id}")
                    print(f"            📍 定位: {breadcrumb if breadcrumb else '(无层级信息)'}")
                    print(f"            Caption: {caption}")
                except:
                    print(f"        [{i+1}] 分数={score:.4f} | ID: {iid} | (获取详情失败)")
        
        # ========== 阶段4: 最终LLM Rerank ==========
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 阶段4: 最终LLM Rerank")
            print(f"{'='*60}")
        
        stage4_start = time.time()
        
        # 4.1 合并Chunk候选
        all_chunk_ids = list(must_have_chunks) + [c[0] for c in spread_candidates['chunk']]
        all_chunk_ids = list(dict.fromkeys(all_chunk_ids))  # 去重保持顺序
        
        if verbose:
            print(f"\n  📄 Chunk候选汇总: {len(all_chunk_ids)}个 (去重后)")
            print(f"      - 必选: {len(must_have_chunks)}个")
            print(f"      - 扩散: {len(spread_candidates['chunk'])}个")
            print(f"\n      待重排Chunk ID列表:")
            for i, cid in enumerate(all_chunk_ids[:8]):
                print(f"        [{i+1}] {cid}")
            if len(all_chunk_ids) > 8:
                print(f"        ... 还有 {len(all_chunk_ids) - 8} 个")
        
        # Chunk重排序
        if len(all_chunk_ids) > 0:
            chunk_contents = []
            for chunk_id in all_chunk_ids:
                try:
                    row = self.chunk_embedding_store.get_row(chunk_id)
                    content = row.get('content', row.get('summary', ''))
                    chunk_contents.append(content)
                except:
                    chunk_contents.append('')
            
            final_chunk_indices, final_chunk_contents, _ = self._rerank_contents(
                query, chunk_contents, all_chunk_ids,
                content_type='chunk', len_after_rerank=final_chunk_k
            )
            final_chunk_ids = [all_chunk_ids[i] for i in final_chunk_indices]
        else:
            final_chunk_ids = []
            final_chunk_contents = []
        
        # 4.2 Code重排序
        all_code_ids = [c[0] for c in spread_candidates['code']]
        if len(all_code_ids) > 0:
            code_contents = []
            for code_id in all_code_ids:
                try:
                    row = self.code_embedding_store.get_row(code_id)
                    content = row.get('content', row.get('summary', ''))
                    code_contents.append(content)
                except:
                    code_contents.append('')
            
            final_code_indices, final_code_contents, _ = self._rerank_contents(
                query, code_contents, all_code_ids,
                content_type='code', len_after_rerank=final_code_k
            )
            final_code_ids = [all_code_ids[i] for i in final_code_indices]
        else:
            final_code_ids = []
            final_code_contents = []
        
        # 4.3 Table重排序
        all_table_ids = [t[0] for t in spread_candidates['table']]
        if len(all_table_ids) > 0:
            table_contents = []
            for table_id in all_table_ids:
                try:
                    row = self.table_embedding_store.get_row(table_id)
                    content = row.get('content', row.get('summary', ''))
                    table_contents.append(content)
                except:
                    table_contents.append('')
            
            final_table_indices, final_table_contents, _ = self._rerank_contents(
                query, table_contents, all_table_ids,
                content_type='table', len_after_rerank=final_table_k
            )
            final_table_ids = [all_table_ids[i] for i in final_table_indices]
        else:
            final_table_ids = []
            final_table_contents = []
        
        # 4.4 Image重排序
        all_image_ids = [img[0] for img in spread_candidates['image']]
        if len(all_image_ids) > 0:
            image_contents = []
            for image_id in all_image_ids:
                try:
                    row = self.image_embedding_store.get_row(image_id)
                    content = row.get('summary', row.get('content', ''))  # image用summary
                    image_contents.append(content)
                except:
                    image_contents.append('')
            
            final_image_indices, final_image_contents, _ = self._rerank_contents(
                query, image_contents, all_image_ids,
                content_type='image', len_after_rerank=final_image_k
            )
            final_image_ids = [all_image_ids[i] for i in final_image_indices]
        else:
            final_image_ids = []
            final_image_contents = []
        
        stage4_time = time.time() - stage4_start
        
        if verbose:
            print(f"\n  📊 阶段4统计: 耗时 {stage4_time:.3f}s")
            
            # Chunk最终结果详情（含文件路径和面包屑）
            print(f"\n  📄 最终Chunk结果 ({len(final_chunk_ids)}个):")
            print(f"  {'='*80}")
            for i, (cid, content) in enumerate(zip(final_chunk_ids, final_chunk_contents)):
                # 获取文件路径
                try:
                    row = self.chunk_embedding_store.get_row(cid)
                    file_path = row.get('file_path', '')
                    summary = row.get('summary', '')
                except:
                    file_path = '(未找到)'
                    summary = ''
                
                # 获取面包屑导航
                breadcrumb = self.get_node_breadcrumb(cid)
                
                print(f"    [{i+1}] ====== Chunk ======")
                print(f"        ID: {cid}")
                print(f"        文件路径: {file_path}")
                print(f"        📍 定位: {breadcrumb if breadcrumb else '(无层级信息)'}")
                print(f"        摘要: {summary}")
                print(f"        内容长度: {len(content)}字符")
                print(f"        完整内容:")
                print(f"        {'-'*70}")
                # 显示完整内容（用于日志调试）
                content_lines = content.split('\n')
                for line in content_lines:
                    print(f"        | {line}")
                print(f"        {'-'*70}")
                print()
            
            # Code最终结果详情（含文件路径和面包屑）
            print(f"\n  💻 最终Code结果 ({len(final_code_ids)}个):")
            print(f"  {'='*80}")
            for i, (cid, content) in enumerate(zip(final_code_ids, final_code_contents)):
                # 获取文件路径
                try:
                    row = self.code_embedding_store.get_row(cid)
                    file_path = row.get('file_path', '')
                    summary = row.get('summary', '')
                except:
                    file_path = '(未找到)'
                    summary = ''
                
                # 获取面包屑导航
                breadcrumb = self.get_node_breadcrumb(cid)
                
                print(f"    [{i+1}] ====== Code ======")
                print(f"        ID: {cid}")
                print(f"        文件路径: {file_path}")
                print(f"        📍 定位: {breadcrumb if breadcrumb else '(无层级信息)'}")
                print(f"        摘要: {summary}")
                print(f"        内容长度: {len(content)}字符")
                print(f"        完整代码:")
                print(f"        {'-'*70}")
                content_lines = content.split('\n')
                for line in content_lines:
                    print(f"        | {line}")
                print(f"        {'-'*70}")
                print()
            
            # Table最终结果详情（含文件路径和面包屑）
            print(f"\n  📊 最终Table结果 ({len(final_table_ids)}个):")
            print(f"  {'='*80}")
            for i, (tid, content) in enumerate(zip(final_table_ids, final_table_contents)):
                # 获取文件路径
                try:
                    row = self.table_embedding_store.get_row(tid)
                    file_path = row.get('file_path', '')
                    summary = row.get('summary', '')
                except:
                    file_path = '(未找到)'
                    summary = ''
                
                # 获取面包屑导航
                breadcrumb = self.get_node_breadcrumb(tid)
                
                print(f"    [{i+1}] ====== Table ======")
                print(f"        ID: {tid}")
                print(f"        文件路径: {file_path}")
                print(f"        📍 定位: {breadcrumb if breadcrumb else '(无层级信息)'}")
                print(f"        摘要: {summary}")
                print(f"        内容长度: {len(content)}字符")
                print(f"        完整表格:")
                print(f"        {'-'*70}")
                content_lines = content.split('\n')
                for line in content_lines:
                    print(f"        | {line}")
                print(f"        {'-'*70}")
                print()
            
            # Image最终结果详情（含文件路径、面包屑和来源信息）
            print(f"\n  🖼️ 最终Image结果 ({len(final_image_ids)}个):")
            print(f"  {'='*80}")
            for i, (iid, content) in enumerate(zip(final_image_ids, final_image_contents)):
                # 获取文件路径
                try:
                    row = self.image_embedding_store.get_row(iid)
                    file_path = row.get('file_path', '')  # gitee_url
                    absolute_path = row.get('content', '')  # 本地绝对路径
                except:
                    file_path = '(未找到)'
                    absolute_path = ''
                
                # 获取面包屑导航和来源信息
                breadcrumb = self.get_node_breadcrumb(iid)
                meta_info = self.node_id_to_metadata.get(iid, {})
                md_file_path = meta_info.get('md_file_path', '')
                parent_chunk_id = meta_info.get('parent_chunk_id', '')
                gitee_md_url = meta_info.get('file_path', '')  # md文件的gitee url
                
                print(f"    [{i+1}] ====== Image ======")
                print(f"        ID: {iid}")
                print(f"        图片Gitee URL: {file_path}")
                print(f"        图片本地路径: {absolute_path}")
                print(f"        📄 来源MD文件: {md_file_path}")
                print(f"        📄 MD文件Gitee URL: {gitee_md_url}")
                print(f"        📦 来源Chunk ID: {parent_chunk_id}")
                print(f"        📍 定位: {breadcrumb if breadcrumb else '(无层级信息)'}")
                print(f"        Caption/摘要: {content}")
                print()
            
            # 总耗时汇总
            total_time = stage1_time + stage2_time + stage3_time + stage4_time
            print(f"\n  {'='*60}")
            print(f"  ⏱️ 本查询耗时汇总:")
            print(f"      阶段1 (Embedding候选): {stage1_time:.3f}s")
            print(f"      阶段2 (LLM Rerank): {stage2_time:.3f}s")
            print(f"      阶段3 (图扩散): {stage3_time:.3f}s")
            print(f"      阶段4 (最终Rerank): {stage4_time:.3f}s")
            print(f"      总计: {total_time:.3f}s")
        
        # 构建返回结果（不含报告，报告在retrieve_v2中统一生成）
        return {
            'query': query,
            'chunks': {
                'ids': final_chunk_ids,
                'contents': final_chunk_contents,
                'metadata': self._get_chunks_metadata(final_chunk_ids)
            },
            'codes': {
                'ids': final_code_ids,
                'contents': final_code_contents,
                'metadata': self._get_codes_metadata(final_code_ids)
            },
            'tables': {
                'ids': final_table_ids,
                'contents': final_table_contents,
                'metadata': self._get_tables_metadata(final_table_ids)
            },
            'images': {
                'ids': final_image_ids,
                'contents': final_image_contents,
                'metadata': self._get_images_metadata(final_image_ids)
            },
            'timing': {
                'stage1_embedding': stage1_time,
                'stage2_rerank': stage2_time,
                'stage3_spread': stage3_time,
                'stage4_final_rerank': stage4_time,
                'total': stage1_time + stage2_time + stage3_time + stage4_time
            }
        }
    
    def _get_chunks_metadata(self, chunk_ids: List[str]) -> List[Dict]:
        """获取chunks的元数据"""
        metadata_list = []
        for cid in chunk_ids:
            try:
                row = self.chunk_embedding_store.get_row(cid)
                breadcrumb = self.get_node_breadcrumb(cid)
                metadata_list.append({
                    'id': cid,
                    'file_path': row.get('file_path', ''),
                    'summary': row.get('summary', ''),
                    'breadcrumb': breadcrumb or ''
                })
            except:
                metadata_list.append({'id': cid, 'file_path': '', 'summary': '', 'breadcrumb': ''})
        return metadata_list
    
    def _get_codes_metadata(self, code_ids: List[str]) -> List[Dict]:
        """获取codes的元数据"""
        metadata_list = []
        for cid in code_ids:
            try:
                row = self.code_embedding_store.get_row(cid)
                breadcrumb = self.get_node_breadcrumb(cid)
                metadata_list.append({
                    'id': cid,
                    'file_path': row.get('file_path', ''),
                    'summary': row.get('summary', ''),
                    'breadcrumb': breadcrumb or ''
                })
            except:
                metadata_list.append({'id': cid, 'file_path': '', 'summary': '', 'breadcrumb': ''})
        return metadata_list
    
    def _get_tables_metadata(self, table_ids: List[str]) -> List[Dict]:
        """获取tables的元数据"""
        metadata_list = []
        for tid in table_ids:
            try:
                row = self.table_embedding_store.get_row(tid)
                breadcrumb = self.get_node_breadcrumb(tid)
                metadata_list.append({
                    'id': tid,
                    'file_path': row.get('file_path', ''),
                    'summary': row.get('summary', ''),
                    'breadcrumb': breadcrumb or ''
                })
            except:
                metadata_list.append({'id': tid, 'file_path': '', 'summary': '', 'breadcrumb': ''})
        return metadata_list
    
    def _get_images_metadata(self, image_ids: List[str]) -> List[Dict]:
        """获取images的元数据，包括图片尺寸信息"""
        metadata_list = []
        for iid in image_ids:
            try:
                row = self.image_embedding_store.get_row(iid)
                meta_info = self.node_id_to_metadata.get(iid, {})
                breadcrumb = self.get_node_breadcrumb(iid)
                local_path = row.get('content', '')  # 本地绝对路径
                
                # 获取图片尺寸信息
                image_size = self._get_image_size(local_path)
                
                metadata_list.append({
                    'id': iid,
                    'gitee_url': row.get('file_path', ''),  # 图片的gitee URL
                    'local_path': local_path,
                    'caption': row.get('summary', ''),       # 图片caption
                    'md_file_path': meta_info.get('md_file_path', ''),
                    'parent_chunk_id': meta_info.get('parent_chunk_id', ''),
                    'breadcrumb': breadcrumb or '',
                    'width': image_size.get('width'),
                    'height': image_size.get('height'),
                    'format': image_size.get('format', ''),
                    'file_size_kb': image_size.get('file_size_kb')
                })
            except:
                metadata_list.append({
                    'id': iid, 'gitee_url': '', 'local_path': '', 
                    'caption': '', 'md_file_path': '', 'parent_chunk_id': '', 'breadcrumb': '',
                    'width': None, 'height': None, 'format': '', 'file_size_kb': None
                })
        return metadata_list
    
    def _get_image_size(self, image_path: str) -> Dict:
        """
        获取图片的尺寸信息
        
        Args:
            image_path: 图片的本地路径
            
        Returns:
            Dict: 包含width, height, format, file_size_kb的字典
        """
        result = {'width': None, 'height': None, 'format': '', 'file_size_kb': None}
        
        if not image_path or not os.path.exists(image_path):
            return result
        
        try:
            # 获取文件大小
            file_size = os.path.getsize(image_path)
            result['file_size_kb'] = round(file_size / 1024, 2)
            
            # 尝试使用PIL获取图片尺寸
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    result['width'] = img.width
                    result['height'] = img.height
                    result['format'] = img.format or ''
            except ImportError:
                # PIL未安装，尝试使用其他方法
                # 对于常见格式，读取文件头获取尺寸
                result.update(self._get_image_size_from_header(image_path))
            except Exception as e:
                logger.debug(f"无法使用PIL读取图片 {image_path}: {e}")
                result.update(self._get_image_size_from_header(image_path))
                
        except Exception as e:
            logger.debug(f"获取图片信息失败 {image_path}: {e}")
        
        return result
    
    def _get_image_size_from_header(self, image_path: str) -> Dict:
        """
        从文件头读取图片尺寸（不依赖PIL）
        支持 PNG, JPEG, GIF 格式
        """
        result = {'width': None, 'height': None, 'format': ''}
        
        try:
            with open(image_path, 'rb') as f:
                header = f.read(32)
                
                # PNG
                if header[:8] == b'\x89PNG\r\n\x1a\n':
                    result['format'] = 'PNG'
                    if len(header) >= 24:
                        result['width'] = int.from_bytes(header[16:20], 'big')
                        result['height'] = int.from_bytes(header[20:24], 'big')
                        
                # JPEG
                elif header[:2] == b'\xff\xd8':
                    result['format'] = 'JPEG'
                    f.seek(0)
                    f.read(2)  # Skip SOI
                    while True:
                        marker = f.read(2)
                        if len(marker) < 2:
                            break
                        if marker[0] != 0xff:
                            break
                        if marker[1] == 0xc0 or marker[1] == 0xc2:  # SOF0 or SOF2
                            f.read(3)  # Skip length and precision
                            height_bytes = f.read(2)
                            width_bytes = f.read(2)
                            if len(height_bytes) == 2 and len(width_bytes) == 2:
                                result['height'] = int.from_bytes(height_bytes, 'big')
                                result['width'] = int.from_bytes(width_bytes, 'big')
                            break
                        else:
                            length_bytes = f.read(2)
                            if len(length_bytes) < 2:
                                break
                            length = int.from_bytes(length_bytes, 'big')
                            f.seek(length - 2, 1)
                            
                # GIF
                elif header[:6] in (b'GIF87a', b'GIF89a'):
                    result['format'] = 'GIF'
                    result['width'] = int.from_bytes(header[6:8], 'little')
                    result['height'] = int.from_bytes(header[8:10], 'little')
                    
                # WebP
                elif header[:4] == b'RIFF' and header[8:12] == b'WEBP':
                    result['format'] = 'WEBP'
                    # WebP 格式解析较复杂，这里只标记格式
                    
        except Exception as e:
            logger.debug(f"从文件头读取图片尺寸失败 {image_path}: {e}")
        
        return result

    def _call_report_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        verbose: bool = True
    ) -> Tuple[str, Dict]:
        """
        调用专门的报告生成模型 qwen3-235b-a22b
        
        使用华为 ModelArts MaaS API v2 直接调用，而不是使用默认的 LLM 模型
        
        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            verbose: 是否输出详细信息
            
        Returns:
            Tuple[str, Dict]: (响应内容, 元数据)
        """
        import requests
        import json
        import urllib3
        
        # 禁用 SSL 警告（因为使用 verify=False）
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        url = "https://api.modelarts-maas.com/v2/chat/completions"  # API v2
        api_key = os.environ.get('OPENAI_API_KEY', '')
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        data = {
            "model": "Kimi-K2",  # 使用 DeepSeek-R1 模型生成报告
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        if verbose:
            print(f"  🤖 使用报告专用模型: Kimi-K2")
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), verify=False, timeout=600)
            response.raise_for_status()
            
            result = response.json()
            
            response_content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            metadata = {
                'prompt_tokens': result.get('usage', {}).get('prompt_tokens', 0),
                'completion_tokens': result.get('usage', {}).get('completion_tokens', 0),
                'model': 'Kimi-K2'
            }
            
            return response_content, metadata
            
        except requests.exceptions.RequestException as e:
            logger.error(f"报告生成模型调用失败: {e}")
            raise RuntimeError(f"报告生成模型调用失败: {e}")

    def _generate_report(
        self,
        query: str,
        chunks_data: Dict,
        codes_data: Dict,
        tables_data: Dict,
        images_data: Dict,
        verbose: bool = True
    ) -> Dict:
        """
        阶段5: 调用LLM整合所有检索信息，生成用于前端页面生成的prompt
        
        输出格式：直接可用于前端生成器的Markdown格式prompt，包含：
        1. 用户问题和核心回答
        2. 详细内容章节（带来源引用）
        3. 相关图片（URL + 描述 + 来源）
        4. 相关表格（完整内容 + 来源）
        5. 相关代码（完整内容 + 来源）
        
        所有内容围绕用户问题筛选，无关内容会被LLM丢弃。
        使用专门的 qwen3-235b-a22b 模型来生成高质量报告。
        """
        from .utils.llm_utils import TextChatMessage
        import json
        import re
        
        stage5_start = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📝 阶段5: LLM报告生成（前端Prompt格式）")
            print(f"{'='*60}")
        
        # ========== 第一步：调用LLM筛选和整合内容 ==========
        # 构建检索上下文
        context_parts = []
        
        # Chunk内容
        if chunks_data['ids']:
            context_parts.append("=== 文档内容 ===")
            for i, (cid, content, meta) in enumerate(zip(
                chunks_data['ids'], chunks_data['contents'], chunks_data['metadata']
            )):
                context_parts.append(f"\n[DOC-{i+1}]")
                context_parts.append(f"ID: {cid}")
                context_parts.append(f"文件: {meta.get('file_path', '')}")
                context_parts.append(f"定位: {meta.get('breadcrumb', '')}")
                context_parts.append(f"内容:\n{content[:3000]}")
        
        # Code内容
        if codes_data['ids']:
            context_parts.append("\n\n=== 代码 ===")
            for i, (cid, content, meta) in enumerate(zip(
                codes_data['ids'], codes_data['contents'], codes_data['metadata']
            )):
                context_parts.append(f"\n[CODE-{i+1}]")
                context_parts.append(f"ID: {cid}")
                context_parts.append(f"文件: {meta.get('file_path', '')}")
                context_parts.append(f"定位: {meta.get('breadcrumb', '')}")
                context_parts.append(f"摘要: {meta.get('summary', '')}")
                context_parts.append(f"代码:\n```\n{content}\n```")
        
        # Table内容  
        if tables_data['ids']:
            context_parts.append("\n\n=== 表格 ===")
            for i, (tid, content, meta) in enumerate(zip(
                tables_data['ids'], tables_data['contents'], tables_data['metadata']
            )):
                context_parts.append(f"\n[TABLE-{i+1}]")
                context_parts.append(f"ID: {tid}")
                context_parts.append(f"文件: {meta.get('file_path', '')}")
                context_parts.append(f"定位: {meta.get('breadcrumb', '')}")
                context_parts.append(f"摘要: {meta.get('summary', '')}")
                context_parts.append(f"表格:\n{content}")
        
        # Image信息（含图片元信息）
        if images_data['ids']:
            context_parts.append("\n\n=== 图片 ===")
            for i, (iid, content, meta) in enumerate(zip(
                images_data['ids'], images_data['contents'], images_data['metadata']
            )):
                context_parts.append(f"\n[IMG-{i+1}]")
                context_parts.append(f"ID: {iid}")
                context_parts.append(f"URL: {meta.get('gitee_url', '')}")
                context_parts.append(f"来源文件: {meta.get('md_file_path', '')}")
                context_parts.append(f"定位: {meta.get('breadcrumb', '')}")
                context_parts.append(f"描述: {meta.get('caption', content)}")
                # 添加图片元信息
                width = meta.get('width')
                height = meta.get('height')
                if width and height:
                    context_parts.append(f"尺寸: {width}x{height}像素")
                img_format = meta.get('format', '')
                if img_format:
                    context_parts.append(f"格式: {img_format}")
                file_size_kb = meta.get('file_size_kb')
                if file_size_kb:
                    context_parts.append(f"文件大小: {file_size_kb}KB")
        
        context_text = "\n".join(context_parts)
        
        # 构建LLM Prompt - 要求输出用于前端生成的结构化内容
        # 前端生成器的唯一信息来源，所以要尽量丰富
        system_prompt = """你是技术文档分析助手。任务：筛选相关内容，生成详细丰富的JSON报告，这将作为前端页面生成器的**唯一信息来源**。

输出JSON格式：
{
    "answer": {
        "summary": "核心回答摘要（200-300字，全面概括问题的答案）",
        "key_points": ["关键要点1", "关键要点2", "..."],
        "sections": [
            {
                "title": "章节标题",
                "content": "详细内容（500-800字，尽量详尽）",
                "highlights": ["本章节要点1", "本章节要点2"],
                "sources": [{"file": "文件路径", "location": "定位", "relevance": "说明该来源如何相关"}]
            }
        ]
    },
    "images": [
        {
            "url": "图片URL",
            "caption": "图片说明（100字内，说明图片展示了什么）",
            "context": "图片在文档中的上下文说明",
            "relevance": "图片与问题的相关性说明",
            "source_file": "来源文件",
            "location": "定位路径",
            "dimensions": "尺寸信息（如有）",
            "suggested_display": "建议显示方式：full-width/inline/thumbnail"
        }
    ],
    "tables": [
        {
            "title": "表格标题",
            "description": "表格说明（描述表格包含什么数据）",
            "content": "完整的Markdown表格",
            "key_data": ["表格中的关键数据点1", "关键数据点2"],
            "source_file": "来源文件",
            "location": "定位路径"
        }
    ],
    "codes": [
        {
            "title": "代码标题",
            "language": "编程语言",
            "content": "完整代码",
            "explanation": "代码功能详细说明（100-200字）",
            "usage_notes": "使用注意事项",
            "related_apis": ["涉及的API1", "API2"],
            "source_file": "来源文件",
            "location": "定位路径"
        }
    ],
    "related_concepts": ["相关概念1", "相关概念2"],
    "further_reading": [{"title": "推荐阅读标题", "reason": "推荐理由"}]
}

规则：
- 只输出JSON，无其他内容
- 只保留与问题**相关**的内容，无关的设为空数组[]
- 内容要**详尽丰富**，这是前端生成器的唯一信息来源
- sections最多5个，每个content尽量详细（500-800字）
- 代码content保持完整，不要截断
- 表格保持原格式，并提取关键数据点
- 为图片提供充分的上下文说明和显示建议
- 提取相关概念和进一步阅读建议，帮助用户深入理解"""

        user_prompt = f"""用户问题：{query}

以下是检索到的内容，请筛选并整合与问题相关的信息：

{context_text}

请输出JSON格式的报告。注意：只保留与问题相关的内容。"""

        if verbose:
            print(f"  📤 上下文长度: {len(context_text)} 字符")
            print(f"  📤 包含: {len(chunks_data['ids'])}个Chunk, {len(codes_data['ids'])}个Code, {len(tables_data['ids'])}个Table, {len(images_data['ids'])}个Image")
        
        try:
            # 调用专门的报告生成模型 qwen3-235b-a22b
            response, metadata = self._call_report_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                verbose=verbose
            )
            
            if verbose:
                print(f"  📥 LLM响应长度: {len(response)} 字符")
                print(f"  📊 Token: prompt={metadata.get('prompt_tokens', 0)}, completion={metadata.get('completion_tokens', 0)}")
            
            # 解析JSON
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.strip()
            
            # 尝试解析JSON，如果失败则尝试修复
            try:
                report_data = json.loads(json_str)
            except json.JSONDecodeError as parse_error:
                if verbose:
                    print(f"  ⚠️ JSON解析失败，尝试修复: {parse_error}")
                # 尝试修复截断的JSON
                report_data = self._try_fix_truncated_json(json_str, verbose)
                if report_data is None:
                    raise parse_error
            
            # ========== 第二步：生成前端Prompt ==========
            frontend_prompt = self._build_frontend_prompt(query, report_data)
            
            stage5_time = time.time() - stage5_start
            
            if verbose:
                print(f"\n  ✅ 报告生成成功!")
                print(f"  📋 摘要: {report_data.get('answer', {}).get('summary', '')[:150]}...")
                print(f"  📄 内容章节: {len(report_data.get('answer', {}).get('sections', []))}个")
                print(f"  🖼️ 相关图片: {len(report_data.get('images', []))}个")
                print(f"  📊 相关表格: {len(report_data.get('tables', []))}个")
                print(f"  💻 相关代码: {len(report_data.get('codes', []))}个")
                print(f"  📝 前端Prompt长度: {len(frontend_prompt)} 字符")
                print(f"  ⏱️ 阶段5耗时: {stage5_time:.3f}s")
            
            return {
                'success': True,
                'report': report_data,
                'frontend_prompt': frontend_prompt,
                'timing': stage5_time
            }
            
        except json.JSONDecodeError as e:
            if verbose:
                print(f"  ❌ JSON解析失败: {e}")
                print(f"  📄 原始响应: {response[:500]}...")
            
            # 即使JSON解析失败，也尝试生成一个基础的前端prompt
            stage5_time = time.time() - stage5_start
            fallback_prompt = self._build_fallback_frontend_prompt(query, chunks_data, codes_data, tables_data, images_data)
            
            return {
                'success': False,
                'error': f'JSON解析失败: {str(e)}',
                'raw_response': response,
                'frontend_prompt': fallback_prompt,  # 提供降级的prompt
                'timing': stage5_time
            }
            
        except Exception as e:
            if verbose:
                print(f"  ❌ 报告生成失败: {e}")
            
            stage5_time = time.time() - stage5_start
            fallback_prompt = self._build_fallback_frontend_prompt(query, chunks_data, codes_data, tables_data, images_data)
            
            return {
                'success': False,
                'error': str(e),
                'frontend_prompt': fallback_prompt,
                'timing': stage5_time
            }
    
    def _try_fix_truncated_json(self, json_str: str, verbose: bool = False) -> Optional[Dict]:
        """
        尝试修复被截断的JSON字符串
        """
        import json
        
        # 策略1: 尝试找到最后一个完整的对象/数组
        # 从末尾向前查找可以闭合的位置
        brackets = []
        in_string = False
        escape_next = False
        last_valid_pos = 0
        
        for i, char in enumerate(json_str):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            
            if char in '{[':
                brackets.append((char, i))
            elif char == '}' and brackets and brackets[-1][0] == '{':
                brackets.pop()
                last_valid_pos = i + 1
            elif char == ']' and brackets and brackets[-1][0] == '[':
                brackets.pop()
                last_valid_pos = i + 1
        
        # 策略2: 尝试补全缺失的括号
        if brackets:
            # 先尝试闭合当前未闭合的字符串
            fixed_str = json_str
            if in_string:
                fixed_str += '"'
            
            # 然后逆序闭合所有未闭合的括号
            for bracket, _ in reversed(brackets):
                if bracket == '{':
                    fixed_str += '}'
                elif bracket == '[':
                    fixed_str += ']'
            
            try:
                result = json.loads(fixed_str)
                if verbose:
                    print(f"  ✅ JSON修复成功（补全括号）")
                return result
            except:
                pass
        
        # 策略3: 截取到最后一个完整位置
        if last_valid_pos > 0:
            try:
                result = json.loads(json_str[:last_valid_pos])
                if verbose:
                    print(f"  ✅ JSON修复成功（截取完整部分）")
                return result
            except:
                pass
        
        return None
    
    def _build_fallback_frontend_prompt(
        self,
        query: str,
        chunks_data: Dict,
        codes_data: Dict,
        tables_data: Dict,
        images_data: Dict
    ) -> str:
        """
        当LLM整合失败时，生成降级版本的前端Prompt
        直接使用原始检索内容，不经过LLM整合
        """
        lines = []
        
        lines.append("# 用户问题")
        lines.append(query)
        lines.append("")
        
        lines.append("# 检索到的相关内容")
        lines.append("")
        lines.append("> 注：以下内容为原始检索结果，未经LLM整合筛选")
        lines.append("")
        
        # 添加Chunk内容
        if chunks_data.get('ids'):
            lines.append("## 相关文档")
            lines.append("")
            for i, (cid, content, meta) in enumerate(zip(
                chunks_data['ids'][:5],  # 限制数量
                chunks_data['contents'][:5],
                chunks_data['metadata'][:5]
            ), 1):
                lines.append(f"### 文档 {i}")
                lines.append("")
                lines.append(content[:1500] + ('...' if len(content) > 1500 else ''))
                lines.append("")
                lines.append(f"**来源：** `{meta.get('file_path', '')}`")
                if meta.get('breadcrumb'):
                    lines.append(f"**定位：** {meta.get('breadcrumb')}")
                lines.append("")
        
        # 添加图片（含尺寸信息）
        if images_data.get('ids'):
            lines.append("## 相关图片")
            lines.append("")
            for i, (iid, content, meta) in enumerate(zip(
                images_data['ids'],
                images_data['contents'],
                images_data['metadata']
            ), 1):
                url = meta.get('gitee_url', '')
                caption = meta.get('caption', content)
                lines.append(f"### 图片 {i}")
                lines.append(f"![{caption}]({url})")
                lines.append("")
                lines.append(f"**描述：** {caption}")
                # 添加图片元信息
                width = meta.get('width')
                height = meta.get('height')
                if width and height:
                    lines.append(f"**尺寸：** {width}x{height}像素")
                img_format = meta.get('format', '')
                file_size_kb = meta.get('file_size_kb')
                if img_format or file_size_kb:
                    meta_parts = []
                    if img_format:
                        meta_parts.append(f"格式: {img_format}")
                    if file_size_kb:
                        meta_parts.append(f"大小: {file_size_kb}KB")
                    lines.append(f"**文件信息：** {', '.join(meta_parts)}")
                if meta.get('breadcrumb'):
                    lines.append(f"**定位：** {meta.get('breadcrumb')}")
                lines.append(f"**来源：** `{meta.get('md_file_path', '')}`")
                lines.append("")
        
        # 添加表格
        if tables_data.get('ids'):
            lines.append("## 相关表格")
            lines.append("")
            for i, (tid, content, meta) in enumerate(zip(
                tables_data['ids'],
                tables_data['contents'],
                tables_data['metadata']
            ), 1):
                lines.append(f"### 表格 {i}")
                lines.append("")
                lines.append(content)
                lines.append("")
                lines.append(f"**来源：** `{meta.get('file_path', '')}`")
                lines.append("")
        
        # 添加代码
        if codes_data.get('ids'):
            lines.append("## 相关代码")
            lines.append("")
            for i, (cid, content, meta) in enumerate(zip(
                codes_data['ids'],
                codes_data['contents'],
                codes_data['metadata']
            ), 1):
                lines.append(f"### 代码 {i}")
                lines.append("")
                if meta.get('summary'):
                    lines.append(f"**说明：** {meta.get('summary')}")
                    lines.append("")
                lines.append("```")
                lines.append(content)
                lines.append("```")
                lines.append("")
                lines.append(f"**来源：** `{meta.get('file_path', '')}`")
                lines.append("")
        
        return "\n".join(lines)
    
    def _build_frontend_prompt(self, query: str, report_data: Dict) -> str:
        """
        将报告数据转换为前端生成器可用的Prompt格式
        
        输出一个清晰的Markdown格式文档，包含所有必要信息供前端LLM生成HTML页面
        这是前端生成器的唯一信息来源，所以尽量详尽
        """
        lines = []
        
        # ===== 用户问题 =====
        lines.append("# 用户问题")
        lines.append(query)
        lines.append("")
        
        # ===== 核心回答 =====
        answer = report_data.get('answer', {})
        if answer.get('summary'):
            lines.append("# 核心回答")
            lines.append(answer['summary'])
            lines.append("")
            
            # 关键要点
            key_points = answer.get('key_points', [])
            if key_points:
                lines.append("## 关键要点")
                for kp in key_points:
                    lines.append(f"- {kp}")
                lines.append("")
        
        # ===== 详细内容 =====
        sections = answer.get('sections', [])
        if sections:
            lines.append("# 详细内容")
            lines.append("")
            for i, section in enumerate(sections, 1):
                lines.append(f"## {i}. {section.get('title', '内容')}")
                lines.append("")
                lines.append(section.get('content', ''))
                lines.append("")
                
                # 章节要点
                highlights = section.get('highlights', [])
                if highlights:
                    lines.append("**本节要点：**")
                    for h in highlights:
                        lines.append(f"- {h}")
                    lines.append("")
                
                # 来源信息
                sources = section.get('sources', [])
                if sources:
                    lines.append("**来源：**")
                    for src in sources:
                        lines.append(f"- 文件: `{src.get('file', '')}`")
                        if src.get('location'):
                            lines.append(f"  定位: {src.get('location')}")
                        if src.get('relevance'):
                            lines.append(f"  相关性: {src.get('relevance')}")
                lines.append("")
        
        # ===== 相关图片 =====
        images = report_data.get('images', [])
        if images:
            lines.append("# 相关图片")
            lines.append("")
            for i, img in enumerate(images, 1):
                lines.append(f"## 图片 {i}")
                lines.append(f"![{img.get('caption', '图片')}]({img.get('url', '')})")
                lines.append("")
                lines.append(f"**说明：** {img.get('caption', '')}")
                lines.append("")
                # 上下文信息
                if img.get('context'):
                    lines.append(f"**上下文：** {img.get('context')}")
                    lines.append("")
                # 相关性说明
                if img.get('relevance'):
                    lines.append(f"**与问题的相关性：** {img.get('relevance')}")
                    lines.append("")
                # 尺寸信息
                if img.get('dimensions'):
                    lines.append(f"**尺寸：** {img.get('dimensions')}")
                # 显示建议
                if img.get('suggested_display'):
                    lines.append(f"**建议显示方式：** {img.get('suggested_display')}")
                lines.append("")
                lines.append(f"**来源文件：** `{img.get('source_file', '')}`")
                if img.get('location'):
                    lines.append(f"**定位：** {img.get('location')}")
                lines.append("")
        
        # ===== 相关表格 =====
        tables = report_data.get('tables', [])
        if tables:
            lines.append("# 相关表格")
            lines.append("")
            for i, table in enumerate(tables, 1):
                lines.append(f"## 表格 {i}: {table.get('title', '数据表')}")
                lines.append("")
                # 表格描述
                if table.get('description'):
                    lines.append(f"**描述：** {table.get('description')}")
                    lines.append("")
                lines.append(table.get('content', ''))
                lines.append("")
                # 关键数据点
                key_data = table.get('key_data', [])
                if key_data:
                    lines.append("**关键数据：**")
                    for kd in key_data:
                        lines.append(f"- {kd}")
                    lines.append("")
                lines.append(f"**来源文件：** `{table.get('source_file', '')}`")
                if table.get('location'):
                    lines.append(f"**定位：** {table.get('location')}")
                lines.append("")
        
        # ===== 相关代码 =====
        codes = report_data.get('codes', [])
        if codes:
            lines.append("# 相关代码")
            lines.append("")
            for i, code in enumerate(codes, 1):
                lines.append(f"## 代码 {i}: {code.get('title', '示例代码')}")
                lines.append("")
                if code.get('explanation'):
                    lines.append(f"**功能说明：** {code.get('explanation')}")
                    lines.append("")
                # 使用注意事项
                if code.get('usage_notes'):
                    lines.append(f"**使用注意：** {code.get('usage_notes')}")
                    lines.append("")
                # 涉及的API
                related_apis = code.get('related_apis', [])
                if related_apis:
                    lines.append(f"**涉及API：** {', '.join(related_apis)}")
                    lines.append("")
                lang = code.get('language', '')
                lines.append(f"```{lang}")
                lines.append(code.get('content', ''))
                lines.append("```")
                lines.append("")
                lines.append(f"**来源文件：** `{code.get('source_file', '')}`")
                if code.get('location'):
                    lines.append(f"**定位：** {code.get('location')}")
                lines.append("")
        
        # ===== 相关概念 =====
        related_concepts = report_data.get('related_concepts', [])
        if related_concepts:
            lines.append("# 相关概念")
            lines.append("")
            for concept in related_concepts:
                lines.append(f"- {concept}")
            lines.append("")
        
        # ===== 推荐阅读 =====
        further_reading = report_data.get('further_reading', [])
        if further_reading:
            lines.append("# 推荐阅读")
            lines.append("")
            for fr in further_reading:
                if isinstance(fr, dict):
                    lines.append(f"- **{fr.get('title', '')}**: {fr.get('reason', '')}")
                else:
                    lines.append(f"- {fr}")
            lines.append("")
        
        return "\n".join(lines)

    def retrieve_by_type(self,
                         queries: List[str],
                         chunk_top_k: int = 10,
                         table_top_k: int = 5,
                         code_top_k: int = 5,
                         verbose: bool = True) -> List[TypedQuerySolution]:
        """
        分类型检索：分别返回 chunk、table、code 的排序结果
        
        与 retrieve 方法类似，但返回按类型分开的结果。
        
        Args:
            queries (List[str]): 查询字符串列表
            chunk_top_k (int): 返回的 chunk 数量
            table_top_k (int): 返回的 table 数量
            code_top_k (int): 返回的 code 数量
            verbose (bool): 是否输出详细信息
                
        Returns:
            List[TypedQuerySolution]: 分类型的检索结果列表
        """
        def debug_print(msg, data=None, max_items=5):
            """辅助函数：格式化打印debug信息"""
            if not verbose:
                return
            print(f"\n{'='*60}")
            print(f"🔍 {msg}")
            print(f"{'='*60}")
            if data is not None:
                if isinstance(data, dict):
                    for k, v in list(data.items())[:max_items]:
                        print(f"  [{k}]: {v}")
        
        retrieve_start_time = time.time()
        
        rerank_candidate_k = getattr(self.global_config, 'rerank_candidate_k', 50)
        file_rerank_candidate_k = getattr(self.global_config, 'file_rerank_candidate_k', 50)
        file_linking_top_k = getattr(self.global_config, 'file_linking_top_k', 5)
        
        if not self.ready_to_retrieve:
            debug_print("准备检索对象...")
            self.prepare_retrieval_objects()
        
        debug_print("索引统计", {
            '事实数量': len(self.fact_node_keys),
            'chunk数量': len(self.passage_node_keys),
            'code数量': len(self.code_node_keys),
            'table数量': len(self.table_node_keys),
            '图节点数': self.graph.vcount(),
        })
        
        # 获取查询嵌入
        self.get_query_embeddings(queries)
        
        retrieval_results = []
        
        for q_idx, query in enumerate(queries):
            if verbose:
                print(f"\n{'#'*70}")
                print(f"# 查询 {q_idx + 1}/{len(queries)}: {query}")
                print(f"{'#'*70}")
            
            # 第一步：事实检索
            step1_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            step1_time = time.time() - step1_start
            
            # 第二步：事实重排序
            step2_start = time.time()
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            step2_time = time.time() - step2_start
            
            if verbose:
                print(f"\n  ⏱️ 事实检索耗时: {step1_time:.3f}s, 重排序耗时: {step2_time:.3f}s")
                print(f"  📋 重排序后的事实 ({len(top_k_facts)}个):")
                for i, fact in enumerate(top_k_facts[:5]):
                    print(f"    [{i+1}] {fact}")
            
            # 第二步B：文件摘要检索和重排序
            step2b_start = time.time()
            top_files = []
            if hasattr(self, 'file_node_keys') and len(self.file_node_keys) > 0:
                query_file_scores, file_keys = self.get_file_scores(query)
                if len(query_file_scores) > 0:
                    _, top_files, _ = self.rerank_files(query, query_file_scores, file_keys)
            step2b_time = time.time() - step2b_start
            
            if verbose and top_files:
                print(f"\n  📂 相关文件 ({len(top_files)}个):")
                for i, f in enumerate(top_files[:3]):
                    print(f"    [{i+1}] {f.get('file_path', '')} (分数={f.get('score', 0):.4f})")
            
            # 第三步：图邻居搜索 + 分类型重排序
            step3_start = time.time()
            if len(top_k_facts) == 0 and len(top_files) == 0:
                # 降级到DPR，但仍然分类型返回
                if verbose:
                    print(f"\n  ⚠️ 无事实或文件，使用DPR降级检索")
                type_results = self._dpr_by_type(query)
                # 构建分类型结果
                chunk_result = self._build_typed_result('chunk', type_results.get('chunk'), chunk_top_k)
                table_result = self._build_typed_result('table', type_results.get('table'), table_top_k)
                code_result = self._build_typed_result('code', type_results.get('code'), code_top_k)
            else:
                # 使用新的图邻居搜索 + LLM重排序方法
                if verbose:
                    print(f"\n  🔍 使用图邻居搜索 + LLM重排序")
                type_results = self.graph_neighbor_rerank(
                    query=query,
                    top_k_facts=top_k_facts,
                    top_files=top_files,
                    chunk_candidate_k=50,
                    table_candidate_k=20,
                    code_candidate_k=20,
                    chunk_top_k=chunk_top_k,
                    table_top_k=table_top_k,
                    code_top_k=code_top_k,
                    max_hop=2,
                    verbose=verbose
                )
                # 直接构建结果（graph_neighbor_rerank 返回的格式不同）
                chunk_data = type_results.get('chunk', ([], np.array([]), []))
                table_data = type_results.get('table', ([], np.array([]), []))
                code_data = type_results.get('code', ([], np.array([]), []))
                
                chunk_result = TypedContentResult(
                    content_type='chunk',
                    contents=chunk_data[0] if len(chunk_data[0]) > 0 else [],
                    scores=chunk_data[1] if len(chunk_data[1]) > 0 else np.array([]),
                    keys=chunk_data[2] if len(chunk_data[2]) > 0 else []
                )
                table_result = TypedContentResult(
                    content_type='table',
                    contents=table_data[0] if len(table_data[0]) > 0 else [],
                    scores=table_data[1] if len(table_data[1]) > 0 else np.array([]),
                    keys=table_data[2] if len(table_data[2]) > 0 else []
                )
                code_result = TypedContentResult(
                    content_type='code',
                    contents=code_data[0] if len(code_data[0]) > 0 else [],
                    scores=code_data[1] if len(code_data[1]) > 0 else np.array([]),
                    keys=code_data[2] if len(code_data[2]) > 0 else []
                )
            step3_time = time.time() - step3_start
            
            if verbose:
                print(f"\n  ⏱️ 图邻居搜索+重排序总耗时: {step3_time:.3f}s")
                print(f"\n  📊 分类型检索结果:")
                self._print_typed_result("Chunk", chunk_result, 5)
                self._print_typed_result("Table", table_result, 3)
                self._print_typed_result("Code", code_result, 3)
            
            retrieval_results.append(TypedQuerySolution(
                question=query,
                chunks=chunk_result,
                tables=table_result,
                codes=code_result
            ))
        
        # 总体统计
        retrieve_end_time = time.time()
        total_time = retrieve_end_time - retrieve_start_time
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"📈 检索总体统计")
            print(f"{'='*70}")
            print(f"  查询数量: {len(queries)}")
            print(f"  总耗时: {total_time:.2f}s")
            print(f"  平均每查询: {total_time/len(queries):.2f}s")
        
        return retrieval_results

    def _dpr_by_type(self, query: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        分类型的DPR检索（降级方案）
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self._batch_encode_texts(query,
                                                       instruction=get_query_instruction('query_to_passage'),
                                                       norm=True)
        
        results = {}
        
        for content_type in ['chunk', 'code', 'table']:
            start_idx, end_idx = self.content_type_ranges.get(content_type, (0, 0))
            
            if start_idx >= end_idx:
                results[content_type] = (np.array([]), np.array([]), [])
                continue
            
            # 获取该类型的嵌入
            type_embeddings = self.all_content_embeddings[start_idx:end_idx]
            type_node_keys = self.all_content_node_keys[start_idx:end_idx]
            
            # 计算相似度
            type_scores = np.dot(type_embeddings, query_embedding.T)
            type_scores = np.squeeze(type_scores) if type_scores.ndim == 2 else type_scores
            type_scores = min_max_normalize(type_scores)
            
            # 排序
            sorted_local_ids = np.argsort(type_scores)[::-1]
            sorted_scores = type_scores[sorted_local_ids]
            sorted_node_keys = [type_node_keys[i] for i in sorted_local_ids]
            
            results[content_type] = (sorted_local_ids, sorted_scores, sorted_node_keys)
        
        return results

    def _build_typed_result(self, content_type: str, 
                           type_data: Tuple[np.ndarray, np.ndarray, List[str]],
                           top_k: int) -> TypedContentResult:
        """
        构建单一类型的检索结果
        """
        if type_data is None or len(type_data[0]) == 0:
            return TypedContentResult(
                content_type=content_type,
                contents=[],
                scores=np.array([]),
                keys=[]
            )
        
        sorted_ids, sorted_scores, sorted_keys = type_data
        
        # 获取top_k个内容
        top_keys = sorted_keys[:top_k]
        top_scores = sorted_scores[:top_k]
        
        # 获取内容文本
        contents = []
        for key in top_keys:
            content = self._get_content_by_type_and_key(content_type, key)
            contents.append(content)
        
        return TypedContentResult(
            content_type=content_type,
            contents=contents,
            scores=top_scores,
            keys=top_keys
        )

    def _print_typed_result(self, type_name: str, result: TypedContentResult, top_k: int = 3):
        """
        打印单一类型的检索结果
        """
        print(f"\n  📁 {type_name} (共{len(result.contents)}个，显示Top-{min(top_k, len(result.contents))}):")
        if len(result.contents) == 0:
            print(f"      无结果")
            return
        
        for i in range(min(top_k, len(result.contents))):
            score = result.scores[i] if i < len(result.scores) else 0
            content = result.contents[i]
            preview = content[:120] + "..." if len(content) > 120 else content
            preview = preview.replace('\n', ' ')
            print(f"      [{i+1}] 分数={score:.6f}")
            print(f"          {preview}")

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
            for idx, passage in enumerate(retrieved_passages):
                # 检查是否是层次化内容（包含类型标记）
                if passage.startswith('[文件]') or passage.startswith('[段落]') or passage.startswith('[代码]') or passage.startswith('[表格]'):
                    # 层次化内容，保持原有格式
                    prompt_user += f'参考内容 {idx+1}: {passage}\n\n'
                else:
                    # 传统段落内容
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
        - 图片节点：从image_embedding_store获取
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
            image_to_row = self.image_embedding_store.get_all_id_to_rows()
            if image_to_row:
                all_node_stores.append(image_to_row)
        except Exception as e:
            logger.debug(f"No image nodes to add: {e}")
            
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
        # 结构性边：文件到段落、段落到代码/表格/图片、段落到子段落
        if ((source_id.startswith('file-') and target_id.startswith('chunk-')) or
            (source_id.startswith('chunk-') and target_id.startswith('code-')) or
            (source_id.startswith('chunk-') and target_id.startswith('table-')) or
            (source_id.startswith('chunk-') and target_id.startswith('image-')) or
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
        
        # 保存映射数据
        self._save_mappings()
        
        logger.info(f"Saving graph completed!")
    
    def _save_mappings(self):
        """保存各种映射数据到JSON文件"""
        mappings_path = os.path.join(self.working_dir, "mappings.json")
        
        mappings_data = {
            'fact_to_chunk_id': self.fact_to_chunk_id,
            'fact_to_entities': self.fact_to_entities,
            # ent_node_to_chunk_ids 需要将set转换为list
            'ent_node_to_chunk_ids': {k: list(v) for k, v in (self.ent_node_to_chunk_ids or {}).items()}
        }
        
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(mappings_data, f, ensure_ascii=False)
        
        logger.info(f"Saved mappings to {mappings_path}: "
                   f"{len(self.fact_to_chunk_id)} fact->chunk, "
                   f"{len(self.fact_to_entities)} fact->entities, "
                   f"{len(self.ent_node_to_chunk_ids or {})} entity->chunks")
    
    def _load_mappings(self):
        """从JSON文件加载映射数据"""
        mappings_path = os.path.join(self.working_dir, "mappings.json")
        
        if os.path.exists(mappings_path):
            with open(mappings_path, 'r', encoding='utf-8') as f:
                mappings_data = json.load(f)
            
            self.fact_to_chunk_id = mappings_data.get('fact_to_chunk_id', {})
            self.fact_to_entities = {k: tuple(v) for k, v in mappings_data.get('fact_to_entities', {}).items()}
            # 将list转换回set
            ent_to_chunks = mappings_data.get('ent_node_to_chunk_ids', {})
            self.ent_node_to_chunk_ids = {k: set(v) for k, v in ent_to_chunks.items()}
            
            logger.info(f"Loaded mappings from {mappings_path}: "
                       f"{len(self.fact_to_chunk_id)} fact->chunk, "
                       f"{len(self.fact_to_entities)} fact->entities, "
                       f"{len(self.ent_node_to_chunk_ids)} entity->chunks")
        else:
            logger.info(f"No mappings file found at {mappings_path}, using empty mappings")

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
            image_nodes_keys = self.image_embedding_store.get_all_ids()
            graph_info["num_image_nodes"] = len(set(image_nodes_keys))
        except:
            graph_info["num_image_nodes"] = 0

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
                                        graph_info["num_image_nodes"] +
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
                    node1.startswith('code-') or node1.startswith('table-') or
                    node1.startswith('image-')):
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
        
        # 获取层次化结构的其他类型节点
        try:
            self.file_node_keys: List = list(self.file_embedding_store.get_all_ids()) # 文件节点键列表
        except:
            self.file_node_keys: List = []
            
        try:
            self.code_node_keys: List = list(self.code_embedding_store.get_all_ids()) # 代码节点键列表
        except:
            self.code_node_keys: List = []
            
        try:
            self.table_node_keys: List = list(self.table_embedding_store.get_all_ids()) # 表格节点键列表
        except:
            self.table_node_keys: List = []
        
        try:
            self.image_node_keys: List = list(self.image_embedding_store.get_all_ids()) # 图片节点键列表
        except:
            self.image_node_keys: List = []
            
        # 合并所有内容节点用于检索（文件、段落、代码、表格、图片）
        self.all_content_node_keys: List = (self.file_node_keys + self.passage_node_keys + 
                                           self.code_node_keys + self.table_node_keys + self.image_node_keys)
        self.all_content_node_types: List = (['file'] * len(self.file_node_keys) + 
                                           ['chunk'] * len(self.passage_node_keys) + 
                                           ['code'] * len(self.code_node_keys) + 
                                           ['table'] * len(self.table_node_keys) +
                                           ['image'] * len(self.image_node_keys))
        
        # 记录各类型在 all_content_node_keys 中的索引范围（用于分类型检索）
        file_end = len(self.file_node_keys)
        chunk_end = file_end + len(self.passage_node_keys)
        code_end = chunk_end + len(self.code_node_keys)
        table_end = code_end + len(self.table_node_keys)
        image_end = table_end + len(self.image_node_keys)
        
        self.content_type_ranges = {
            'file': (0, file_end),
            'chunk': (file_end, chunk_end),
            'code': (chunk_end, code_end),
            'table': (code_end, table_end),
            'image': (table_end, image_end)
        }
        logger.info(f"Content type ranges: file={self.content_type_ranges['file']}, chunk={self.content_type_ranges['chunk']}, "
                   f"code={self.content_type_ranges['code']}, table={self.content_type_ranges['table']}, image={self.content_type_ranges['image']}")

        # 数据一致性检查：验证图中的节点数量与嵌入存储器中的节点数量是否匹配
        expected_node_count = len(self.entity_node_keys) + len(self.all_content_node_keys)
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
            
            # 检查所有实体和内容节点是否都在图中存在
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_content_nodes = [node_key for node_key in self.all_content_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_content_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_content_nodes)} content nodes")
                # 如果发现缺失节点，重新构建图结构
                self.add_new_nodes()
                self.save_igraph()
                # 更新映射关系
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # 创建节点键到图索引的快速查找列表
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # 实体节点的图索引列表
            # 为了兼容性，保留passage_node_idxs，但也创建所有内容节点的索引
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # 段落节点的图索引列表
            self.all_content_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.all_content_node_keys] # 所有内容节点的图索引列表
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # 如果映射创建失败，初始化为空列表
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []
            self.all_content_node_idxs = []

        logger.info("Loading embeddings.")
        # 将所有向量嵌入加载到内存中的numpy数组，提高检索性能
        # 避免检索时频繁的磁盘I/O操作
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        logger.info(f"Loaded {len(self.entity_embeddings)} entity embeddings.")
        
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))
        logger.info(f"Loaded {len(self.passage_embeddings)} passage embeddings.")
        
        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))
        logger.info(f"Loaded {len(self.fact_embeddings)} fact embeddings.")
        
        # 批量加载层次化结构的所有内容类型嵌入（优化性能）
        logger.info("Loading hierarchical content embeddings...")
        all_content_embeddings_list = []
        
        # 批量获取各类型嵌入，避免逐个循环
        if self.file_node_keys:
            file_embeddings = self.file_embedding_store.get_embeddings(self.file_node_keys)
            all_content_embeddings_list.extend(file_embeddings)
            logger.info(f"Loaded {len(file_embeddings)} file embeddings.")
        
        if self.passage_node_keys:
            # passage_embeddings 已经加载过了，直接复用
            all_content_embeddings_list.extend(self.passage_embeddings.tolist())
            logger.info(f"Reused {len(self.passage_embeddings)} passage embeddings.")
        
        if self.code_node_keys:
            code_embeddings = self.code_embedding_store.get_embeddings(self.code_node_keys)
            all_content_embeddings_list.extend(code_embeddings)
            logger.info(f"Loaded {len(code_embeddings)} code embeddings.")
        
        if self.table_node_keys:
            table_embeddings = self.table_embedding_store.get_embeddings(self.table_node_keys)
            all_content_embeddings_list.extend(table_embeddings)
            logger.info(f"Loaded {len(table_embeddings)} table embeddings.")
        
        if all_content_embeddings_list:
            self.all_content_embeddings = np.array(all_content_embeddings_list)
        else:
            self.all_content_embeddings = np.array([])
        
        logger.info(f"Total content embeddings: {len(self.all_content_embeddings)}")

        # 【新增】加载映射数据（fact->chunk, fact->entities, entity->chunks）
        self._load_mappings()
        
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
        
        # 检查是否使用层次化结构
        self.is_hierarchical = len(self.all_content_node_keys) > len(self.passage_node_keys)
        if self.is_hierarchical:
            logger.info(f"Using hierarchical structure with {len(self.all_content_node_keys)} content nodes")
        else:
            logger.info(f"Using traditional structure with {len(self.passage_node_keys)} passage nodes")

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

    def get_file_scores(self, query: str) -> Tuple[np.ndarray, List[str]]:
        """
        计算文件摘要相关性分数：查询与文件摘要的语义匹配
        
        通过向量相似度计算查询与文件摘要嵌入之间的标准化相似度分数，
        用于在事实检索之外提供文件级别的相关性信息。
        
        Args:
            query (str): 输入查询文本
            
        Returns:
            Tuple[np.ndarray, List[str]]:
                - file_scores: 标准化的相似度分数数组，形状为(#files,)
                - file_keys: 文件节点键列表
        """
        if not hasattr(self, 'file_node_keys') or len(self.file_node_keys) == 0:
            logger.warning("No file nodes available for scoring.")
            return np.array([]), []
        
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self._batch_encode_texts(query,
                                                       instruction=get_query_instruction('query_to_passage'),
                                                       norm=True)
        
        try:
            # 获取文件嵌入（如果还没有加载到内存）
            if not hasattr(self, 'file_embeddings') or self.file_embeddings is None or len(self.file_embeddings) == 0:
                self.file_embeddings = np.array(self.file_embedding_store.get_embeddings(self.file_node_keys))
            
            # 计算相似度
            query_file_scores = np.dot(self.file_embeddings, query_embedding.T)
            query_file_scores = np.squeeze(query_file_scores) if query_file_scores.ndim == 2 else query_file_scores
            query_file_scores = min_max_normalize(query_file_scores)
            
            return query_file_scores, self.file_node_keys
            
        except Exception as e:
            logger.error(f"Error computing file scores: {str(e)}")
            return np.array([]), []

    def get_passage_scores(self, query: str) -> np.ndarray:
        """
        计算段落相关性分数：查询与段落的语义匹配
        
        通过向量相似度计算查询与段落嵌入之间的标准化相似度分数。
        
        Args:
            query (str): 输入查询文本
            
        Returns:
            np.ndarray: 标准化的相似度分数数组，形状为(#passages,)
        """
        if not hasattr(self, 'passage_node_keys') or len(self.passage_node_keys) == 0:
            logger.warning("No passage nodes available for scoring.")
            return np.array([])
        
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self._batch_encode_texts(query,
                                                       instruction=get_query_instruction('query_to_passage'),
                                                       norm=True)
        
        try:
            # 使用已加载的passage_embeddings
            if not hasattr(self, 'passage_embeddings') or self.passage_embeddings is None or len(self.passage_embeddings) == 0:
                self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))
            
            # 计算相似度
            query_passage_scores = np.dot(self.passage_embeddings, np.array(query_embedding).T)
            query_passage_scores = np.squeeze(query_passage_scores) if query_passage_scores.ndim == 2 else query_passage_scores
            query_passage_scores = min_max_normalize(query_passage_scores)
            
            return query_passage_scores
            
        except Exception as e:
            logger.error(f"Error computing passage scores: {str(e)}")
            return np.array([])

    def rerank_files(self, query: str, query_file_scores: np.ndarray, file_keys: List[str]) -> Tuple[List[int], List[Dict], dict]:
        """
        文件摘要重排序：使用LLM对候选文件进行智能筛选
        
        从embedding相似度top-k候选文件中，使用LLM筛选最相关的文件。
        类似于事实重排序的流程：50个候选 -> LLM筛选 -> 保留5个
        
        Args:
            query (str): 输入查询文本
            query_file_scores (np.ndarray): 文件相关性分数数组
            file_keys (List[str]): 文件键列表
            
        Returns:
            Tuple[List[int], List[Dict], dict]:
                - top_file_indices: 重排序后的文件索引列表
                - top_files: 重排序后的文件信息列表
                - rerank_log: 重排序过程的日志信息
        """
        # 使用文件专用配置参数
        file_candidate_k: int = getattr(self.global_config, 'file_rerank_candidate_k', 50)
        file_top_k: int = getattr(self.global_config, 'file_linking_top_k', 5)
        
        logger.info(f"File reranking: selecting top {file_candidate_k} candidates, will keep top {file_top_k} after reranking")
        
        if len(query_file_scores) == 0 or len(file_keys) == 0:
            logger.warning("No files available for reranking.")
            return [], [], {'files_before_rerank': [], 'files_after_rerank': []}
        
        try:
            # 获取top-k候选文件（默认50个）
            candidate_k = min(file_candidate_k, len(query_file_scores))
            if len(query_file_scores) <= candidate_k:
                candidate_file_indices = np.argsort(query_file_scores)[::-1].tolist()
            else:
                candidate_file_indices = np.argsort(query_file_scores)[-candidate_k:][::-1].tolist()
            
            # 获取候选文件的摘要信息
            candidate_files = []
            for idx in candidate_file_indices:
                file_key = file_keys[idx]
                try:
                    row = self.file_embedding_store.get_row(file_key)
                    file_info = {
                        'key': file_key,
                        'summary': row.get('summary', '')[:500],  # 截取摘要
                        'file_path': row.get('file_path', ''),
                        'score': float(query_file_scores[idx])
                    }
                    candidate_files.append(file_info)
                except Exception as e:
                    logger.warning(f"Failed to get file info for {file_key}: {e}")
            
            # 构建重排序prompt，发送所有候选文件给LLM
            files_for_rerank = []
            for i, f in enumerate(candidate_files):  # 发送所有候选文件
                files_for_rerank.append({
                    'id': i,
                    'summary': f['summary'],
                    'file_path': f['file_path']
                })
            
            # 使用LLM进行文件重排序
            try:
                rerank_prompt = f"""你是一个专业的技术文档检索助手。请根据用户的查询问题，从候选文件中选择最相关的文件。

## 选择标准（按优先级排序）：
1. **直接相关**：文件摘要直接回答或涉及查询问题的核心内容
2. **定义/概述**：包含查询主题的定义、概述或基本介绍
3. **技术细节**：包含查询主题的具体实现、配置或使用方法
4. **相关概念**：涉及与查询相关的上下游概念或依赖关系

## 注意事项：
- 优先选择内容具体的文件，而非泛泛的概述
- 如果查询涉及特定版本/API，优先选择对应版本的文档
- 避免选择仅在摘要中顺带提及查询关键词的文件

## 查询问题
{query}

## 候选文件（共{len(files_for_rerank)}个）
{json.dumps(files_for_rerank, ensure_ascii=False, indent=2)}

## 输出要求
请返回最相关文件的id列表，格式为JSON数组，如 [0, 3, 5]。最多选择{file_top_k}个文件。
只输出JSON数组，不要其他内容："""

                messages = [{"role": "user", "content": rerank_prompt}]
                response, _, _ = self.llm_model.infer(messages)
                
                # 解析LLM返回的文件id列表
                import re
                match = re.search(r'\[[\d,\s]*\]', response)
                if match:
                    selected_ids = json.loads(match.group())
                    # 限制最多保留file_top_k个
                    selected_ids = selected_ids[:file_top_k]
                    top_file_indices = [candidate_file_indices[i] for i in selected_ids if i < len(candidate_file_indices)]
                    top_files = [candidate_files[i] for i in selected_ids if i < len(candidate_files)]
                else:
                    # 如果解析失败，使用embedding分数最高的
                    logger.warning("Failed to parse LLM response, using embedding scores")
                    top_file_indices = candidate_file_indices[:file_top_k]
                    top_files = candidate_files[:file_top_k]
                    
            except Exception as e:
                logger.warning(f"LLM file reranking failed: {e}, using embedding scores")
                top_file_indices = candidate_file_indices[:file_top_k]
                top_files = candidate_files[:file_top_k]
            
            rerank_log = {
                'files_before_rerank': candidate_files,
                'files_after_rerank': top_files
            }
            
            return top_file_indices, top_files, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_files: {str(e)}")
            return [], [], {'files_before_rerank': [], 'files_after_rerank': [], 'error': str(e)}

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        密集段落检索：传统向量相似度检索（为了兼容性保留）
        
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
        # 如果有层次化内容，优先使用多类型检索
        if len(self.all_content_node_keys) > len(self.passage_node_keys):
            return self.dense_content_retrieval(query)
        
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

    def dense_content_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        多类型内容检索：支持文件、段落、代码、表格的层次化检索
        
        基于预训练嵌入模型进行查询与多种类型内容的密集向量检索，
        支持层次化JSON结构中的所有内容类型。
        
        Args:
            query (str): 输入查询字符串
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - sorted_content_ids: 按相关性排序的内容ID数组
                - sorted_content_scores: 标准化的相关性分数数组
                
        特点:
        - 支持文件、段落、代码、表格等多种内容类型
        - 统一的相似度计算和排序
        - 为层次化结构优化的检索机制
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self._batch_encode_texts(query,
                                                       instruction=get_query_instruction('query_to_passage'),
                                                       norm=True)
        
        if len(self.all_content_embeddings) == 0:
            # 回退到原始段落检索
            return self.dense_passage_retrieval_original(query)
        
        query_content_scores = np.dot(self.all_content_embeddings, query_embedding.T)
        query_content_scores = np.squeeze(query_content_scores) if query_content_scores.ndim == 2 else query_content_scores
        query_content_scores = min_max_normalize(query_content_scores)

        sorted_content_ids = np.argsort(query_content_scores)[::-1]
        sorted_content_scores = query_content_scores[sorted_content_ids.tolist()]
        return sorted_content_ids, sorted_content_scores

    def dense_passage_retrieval_original(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """原始的段落检索方法（内部使用）"""
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

    def _get_content_by_type_and_key(self, content_type: str, content_key: str) -> str:
        """
        根据内容类型和键获取内容数据
        
        Args:
            content_type: 内容类型 ('file', 'chunk', 'code', 'table')
            content_key: 内容键
            
        Returns:
            str: 格式化的内容字符串
        """
        try:
            if content_type == 'file':
                row = self.file_embedding_store.get_row(content_key)
                return f"[文件] {row.get('content', '')}"
            elif content_type == 'chunk':
                row = self.chunk_embedding_store.get_row(content_key)
                return f"[段落] {row.get('content', '')}"
            elif content_type == 'code':
                row = self.code_embedding_store.get_row(content_key)
                return f"[代码] {row.get('content', '')}"
            elif content_type == 'table':
                row = self.table_embedding_store.get_row(content_key)
                return f"[表格] {row.get('content', '')}"
            else:
                # 回退到chunk
                row = self.chunk_embedding_store.get_row(content_key)
                return row.get('content', '')
        except Exception as e:
            logger.warning(f"Failed to get content for {content_type}:{content_key}, error: {e}")
            return f"[{content_type}] Content not available"


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
                                        passage_node_weight: float = 0.05,
                                        top_files: List[Dict] = None,
                                        file_node_weight: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于事实实体和文件摘要的图搜索算法
        
        使用个性化PageRank (PPR)和密集检索模型，基于事实相似性、文件摘要相关性计算文档分数。
        该函数将相关事实的信号、文件摘要信号与段落相似性和基于图的搜索相结合，以增强结果排名。

        Parameters:
            query (str): 需要进行相似性和相关性计算的输入查询字符串
            link_top_k (int): 从链接分数映射中包含的顶级短语数量，用于下游处理
            query_fact_scores (np.ndarray): 表示每个提供事实的事实-查询相似性的分数数组
            top_k_facts (List[Tuple]): 顶级事实列表，每个事实表示为主语、谓语和宾语的元组
            top_k_fact_indices (List[str]): query_fact_scores数组中顶级事实对应的索引或标识符
            passage_node_weight (float): 缩放图中段落分数的默认权重
            top_files (List[Dict]): 重排序后的相关文件列表，每个包含key, summary, score等
            file_node_weight (float): 缩放图中文件节点分数的权重（默认0.1，比段落权重稍高）

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
        file_weights = np.zeros(len(self.graph.vs['name']))  # 初始化文件权重数组

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

                    # 记录短语分数用于后续平均值计算（只记录存在于图中的短语）
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

        # 为每个检索到的内容分配权重
        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            if len(self.all_content_node_keys) > len(self.passage_node_keys):
                # 使用层次化内容
                content_node_key = self.all_content_node_keys[dpr_sorted_doc_id]  # 获取内容节点键
                content_type = self.all_content_node_types[dpr_sorted_doc_id]  # 获取内容类型
                content_dpr_score = normalized_dpr_sorted_scores[i]  # 获取归一化的DPR分数
                content_node_id = self.node_name_to_vertex_idx[content_node_key]  # 获取内容在图中的节点ID
                # 设置内容权重（乘以权重因子）
                passage_weights[content_node_id] = content_dpr_score * passage_node_weight
                # 获取内容文本
                content_text = self._get_content_by_type_and_key(content_type, content_node_key)
                # 将内容文本和分数添加到链接分数映射中
                linking_score_map[content_text] = content_dpr_score * passage_node_weight
            else:
                # 回退到原始段落处理
                passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]  # 获取段落节点键
                passage_dpr_score = normalized_dpr_sorted_scores[i]  # 获取归一化的DPR分数
                passage_node_id = self.node_name_to_vertex_idx[passage_node_key]  # 获取段落在图中的节点ID
                # 设置段落权重（乘以权重因子）
                passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
                # 获取段落文本内容
                passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
                # 将段落文本和分数添加到链接分数映射中
                linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        # 为相关文件节点分配权重（如果有的话）
        if top_files and len(top_files) > 0:
            for file_info in top_files:
                file_key = file_info.get('key', '')
                file_score = file_info.get('score', 0.5)
                
                # 获取文件在图中的节点ID
                file_node_id = self.node_name_to_vertex_idx.get(file_key, None)
                if file_node_id is not None:
                    # 设置文件权重
                    file_weights[file_node_id] = file_score * file_node_weight
                    # 记录到linking_score_map
                    file_summary = file_info.get('summary', '')[:100]
                    linking_score_map[f"[文件] {file_summary}"] = file_score * file_node_weight
                    logger.debug(f"File node {file_key} assigned weight {file_weights[file_node_id]:.4f}")

        # 将短语权重、段落权重和文件权重合并为一个数组用于PPR计算
        node_weights = phrase_weights + passage_weights + file_weights

        # 限制linking_score_map的大小，只保留前30个最高分数的项目
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        # 确保至少有一些权重大于0，否则PPR算法无法运行
        if sum(node_weights) <= 0:
            logger.warning(f'在给定事实的图中未找到短语，尝试仅使用文件权重: {top_k_facts}')
            # 如果没有实体匹配，但有文件匹配，仍然可以继续
            if sum(file_weights) <= 0:
                raise AssertionError(f'在给定事实的图中未找到短语，也没有文件匹配: {top_k_facts}')

        # 基于之前分配的段落和短语权重运行PPR算法
        ppr_start = time.time()  # 记录PPR开始时间
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()  # 记录PPR结束时间

        # 累加PPR计算时间
        self.ppr_time += (ppr_end - ppr_start)

        # 验证返回的文档数量与语料库大小一致
        # 根据是否使用层次化结构选择正确的比较长度
        if len(self.all_content_node_keys) > len(self.passage_node_keys):
            expected_length = len(self.all_content_node_idxs)
        else:
            expected_length = len(self.passage_node_idxs)
        assert len(ppr_sorted_doc_ids) == expected_length, f"文档概率长度 {len(ppr_sorted_doc_ids)} != 语料库长度 {expected_length}"

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
        link_top_k: int = self.global_config.linking_top_k  # 最终选择的事实数量
        # 候选事实数量：用于送入LLM重排序的事实数量，应该比link_top_k大以弥补embedding不准确
        rerank_candidate_k: int = getattr(self.global_config, 'rerank_candidate_k', 50)
        
        # 检查是否有可用的事实进行重排序
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # 根据分数获取前k个候选事实（使用更大的rerank_candidate_k）
            if len(query_fact_scores) <= rerank_candidate_k:
                # 如果事实数量少于请求数量，使用所有事实
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # 否则获取前rerank_candidate_k个事实（按分数降序排列）
                candidate_fact_indices = np.argsort(query_fact_scores)[-rerank_candidate_k:][::-1].tolist()
            
            logger.info(f"Reranking: selecting top {len(candidate_fact_indices)} candidates from {len(query_fact_scores)} facts, will keep top {link_top_k} after reranking")
                
            # 获取实际的事实ID列表
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            
            # 从嵌入存储器中获取事实内容
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            
            # 解析事实内容，将字符串转换为三元组
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            # 使用DSPy过滤器对事实进行重排序，最终保留link_top_k个
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

        # 优先使用所有内容节点，如果没有则回退到段落节点
        if len(self.all_content_node_keys) > len(self.passage_node_keys):
            # 使用所有内容节点索引
            content_scores = np.array([pagerank_scores[idx] for idx in self.all_content_node_idxs])
            sorted_content_ids = np.argsort(content_scores)[::-1]
            sorted_content_scores = content_scores[sorted_content_ids.tolist()]
            return sorted_content_ids, sorted_content_scores
        else:
            # 回退到原始段落节点
            doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
            sorted_doc_ids = np.argsort(doc_scores)[::-1]
            sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]
            return sorted_doc_ids, sorted_doc_scores

    def run_ppr_by_type(self,
                        reset_prob: np.ndarray,
                        damping: float = 0.5) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        分类型的个性化PageRank算法：分别返回 chunk、table、code 的排序结果
        
        在知识图谱上运行个性化PageRank算法，然后按内容类型分别排序。
        
        Args:
            reset_prob (np.ndarray): 重置概率分布，指定每个节点的初始权重
            damping (float, optional): 阻尼因子，默认0.5
                
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]: 
                各类型的排序结果，键为类型名 ('chunk', 'table', 'code')
                值为元组 (sorted_ids, sorted_scores, node_keys)
                - sorted_ids: 该类型内按分数降序排列的局部索引
                - sorted_scores: 对应的PPR分数
                - node_keys: 对应的节点键列表
        """
        if damping is None: 
            damping = 0.5
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        
        # 运行PPR算法
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        results = {}
        
        # 为每种内容类型分别提取和排序分数
        for content_type in ['chunk', 'code', 'table']:
            start_idx, end_idx = self.content_type_ranges.get(content_type, (0, 0))
            
            if start_idx >= end_idx:
                # 该类型没有数据
                results[content_type] = (np.array([]), np.array([]), [])
                continue
            
            # 获取该类型的节点键和图索引
            type_node_keys = self.all_content_node_keys[start_idx:end_idx]
            type_graph_idxs = self.all_content_node_idxs[start_idx:end_idx]
            
            # 提取该类型节点的PPR分数
            type_scores = np.array([pagerank_scores[idx] for idx in type_graph_idxs])
            
            # 在该类型内按分数排序
            sorted_local_ids = np.argsort(type_scores)[::-1]
            sorted_scores = type_scores[sorted_local_ids]
            sorted_node_keys = [type_node_keys[i] for i in sorted_local_ids]
            
            results[content_type] = (sorted_local_ids, sorted_scores, sorted_node_keys)
        
        return results

    def graph_neighbor_rerank(self, query: str,
                               top_k_facts: List[Tuple],
                               top_files: List[Dict] = None,
                               chunk_candidate_k: int = 50,
                               table_candidate_k: int = 20,
                               code_candidate_k: int = 20,
                               chunk_top_k: int = 10,
                               table_top_k: int = 5,
                               code_top_k: int = 5,
                               max_hop: int = 2,
                               verbose: bool = False) -> Dict[str, Tuple[List[str], np.ndarray, List[str]]]:
        """
        图邻居搜索 + LLM重排序：基于事实和文件在图中找到相邻节点，然后分类型重排序
        
        流程:
        1. 从 rerank 得到的事实中提取实体节点
        2. 从 rerank 得到的文件中获取文件节点
        3. 在图中搜索与这些节点相邻的 chunk/table/code 节点（支持多跳）
        4. 收集各类型的候选节点
        5. 对每种类型的候选分别进行 LLM rerank
        
        Args:
            query (str): 输入查询字符串
            top_k_facts (List[Tuple]): 重排序后的事实列表
            top_files (List[Dict]): 重排序后的文件列表
            chunk_candidate_k (int): chunk 候选数量
            table_candidate_k (int): table 候选数量
            code_candidate_k (int): code 候选数量
            chunk_top_k (int): chunk 最终保留数量
            table_top_k (int): table 最终保留数量
            code_top_k (int): code 最终保留数量
            max_hop (int): 图搜索的最大跳数
            verbose (bool): 是否输出详细信息
            
        Returns:
            Dict[str, Tuple[List[str], np.ndarray, List[str]]]:
                各类型的排序结果，键为类型名 ('chunk', 'table', 'code')
                值为元组 (contents, scores, node_keys)
        """
        search_start = time.time()
        
        # ====== 步骤1: 提取起始节点 ======
        seed_node_ids = set()  # 图节点ID集合
        
        # 1.1 从事实中提取实体节点
        for fact in top_k_facts:
            subject_phrase = fact[0].lower()
            object_phrase = fact[2].lower()
            
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    seed_node_ids.add(phrase_id)
        
        # 1.2 从文件中获取文件节点
        if top_files:
            for file_info in top_files:
                file_key = file_info.get('key', '')
                file_id = self.node_name_to_vertex_idx.get(file_key, None)
                if file_id is not None:
                    seed_node_ids.add(file_id)
        
        if verbose:
            print(f"\n  🌱 起始节点数: {len(seed_node_ids)}")
        
        if len(seed_node_ids) == 0:
            logger.warning("No seed nodes found for graph neighbor search")
            return {
                'chunk': ([], np.array([]), []),
                'table': ([], np.array([]), []),
                'code': ([], np.array([]), [])
            }
        
        # ====== 步骤2: 图邻居搜索 ======
        # 收集各类型的候选节点
        chunk_candidates = {}  # key -> distance
        table_candidates = {}
        code_candidates = {}
        
        # 创建节点键到类型的映射
        chunk_keys_set = set(self.passage_node_keys)
        code_keys_set = set(self.code_node_keys) if hasattr(self, 'code_node_keys') else set()
        table_keys_set = set(self.table_node_keys) if hasattr(self, 'table_node_keys') else set()
        
        # BFS搜索相邻节点
        visited = set(seed_node_ids)
        current_level = seed_node_ids
        
        for hop in range(max_hop):
            next_level = set()
            
            for node_id in current_level:
                # 获取邻居节点
                neighbors = self.graph.neighbors(node_id, mode='all')
                
                for neighbor_id in neighbors:
                    if neighbor_id in visited:
                        continue
                    
                    visited.add(neighbor_id)
                    next_level.add(neighbor_id)
                    
                    # 获取节点键
                    node_key = self.graph.vs[neighbor_id]['name']
                    distance = hop + 1  # 距离 = 跳数
                    
                    # 根据节点键类型分类
                    if node_key in chunk_keys_set:
                        if node_key not in chunk_candidates:
                            chunk_candidates[node_key] = distance
                    elif node_key in code_keys_set:
                        if node_key not in code_candidates:
                            code_candidates[node_key] = distance
                    elif node_key in table_keys_set:
                        if node_key not in table_candidates:
                            table_candidates[node_key] = distance
            
            current_level = next_level
            
            if len(next_level) == 0:
                break
        
        if verbose:
            print(f"  🔍 图邻居搜索 (max_hop={max_hop}):")
            print(f"      - Chunk候选: {len(chunk_candidates)}")
            print(f"      - Table候选: {len(table_candidates)}")
            print(f"      - Code候选: {len(code_candidates)}")
        
        search_time = time.time() - search_start
        
        # ====== 步骤3: 分类型LLM重排序 ======
        results = {}
        
        for content_type, candidates, candidate_k, top_k in [
            ('chunk', chunk_candidates, chunk_candidate_k, chunk_top_k),
            ('table', table_candidates, table_candidate_k, table_top_k),
            ('code', code_candidates, code_candidate_k, code_top_k)
        ]:
            if len(candidates) == 0:
                results[content_type] = ([], np.array([]), [])
                continue
            
            # 按距离排序，选取前 candidate_k 个
            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1])[:candidate_k]
            candidate_keys = [k for k, _ in sorted_candidates]
            
            # 获取内容
            candidate_contents = []
            for key in candidate_keys:
                content = self._get_content_by_type_and_key(content_type, key)
                candidate_contents.append(content)
            
            if verbose:
                print(f"\n  📋 {content_type.upper()} 重排序: {len(candidate_keys)}个候选 -> {top_k}个")
            
            # LLM重排序
            if len(candidate_contents) > 0:
                rerank_start = time.time()
                top_indices, top_contents, _ = self._rerank_contents(
                    query, candidate_contents, candidate_keys, 
                    content_type=content_type, 
                    len_after_rerank=top_k
                )
                rerank_time = time.time() - rerank_start
                
                # 构建结果
                top_keys = [candidate_keys[i] for i in top_indices]
                # 使用重排序后的顺序作为分数（越靠前分数越高）
                scores = np.array([1.0 - i * 0.1 for i in range(len(top_keys))])
                
                if verbose:
                    print(f"      重排序耗时: {rerank_time:.3f}s")
                
                results[content_type] = (top_contents, scores, top_keys)
            else:
                results[content_type] = ([], np.array([]), [])
        
        total_time = time.time() - search_start
        if verbose:
            print(f"\n  ⏱️ 图邻居搜索+重排序总耗时: {total_time:.3f}s")
        
        return results

    def graph_spread_with_similarity(
        self,
        query: str,
        query_embedding: np.ndarray,
        must_have_chunks: Set[str],
        seed_entities: List[Tuple[str, float]],  # [(entity_id, weight), ...]
        seed_files: List[Tuple[str, float]],     # [(file_id, weight), ...]
        max_cost: float = 1.0,
        max_chunk_candidates: int = 15,
        max_code_candidates: int = 10,
        max_table_candidates: int = 10,
        max_image_candidates: int = 10,
        verbose: bool = False
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        基于代价和Query相似度的图扩散
        
        从种子节点（Entity、File）出发，沿低代价边扩散，
        综合考虑边代价和节点与Query的相似度进行评分。
        
        评分公式: score = similarity(node, query) / (1 + cost) × init_weight
        
        Args:
            query: 查询字符串
            query_embedding: 查询的embedding向量
            must_have_chunks: 必选的chunk集合（不参与扩散排序）
            seed_entities: 种子实体列表 [(entity_id, weight), ...]
            seed_files: 种子文件列表 [(file_id, weight), ...]
            max_cost: 最大累积代价
            max_chunk_candidates: 扩散chunk上限
            max_code_candidates: 扩散code上限
            max_table_candidates: 扩散table上限
            max_image_candidates: 扩散image上限
            verbose: 是否输出详细信息
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: 各类型候选结果
                {
                    'chunk': [(chunk_id, score), ...],
                    'code': [(code_id, score), ...],
                    'table': [(table_id, score), ...],
                    'image': [(image_id, score), ...]
                }
        """
        import heapq
        
        start_time = time.time()
        
        def get_node_similarity(node_id: str) -> float:
            """获取节点与Query的相似度"""
            try:
                if node_id.startswith('chunk-'):
                    emb = self.chunk_embedding_store.get_embedding(node_id)
                elif node_id.startswith('entity-'):
                    emb = self.entity_embedding_store.get_embedding(node_id)
                elif node_id.startswith('file-'):
                    emb = self.file_embedding_store.get_embedding(node_id)
                elif node_id.startswith('code-'):
                    emb = self.code_embedding_store.get_embedding(node_id)
                elif node_id.startswith('table-'):
                    emb = self.table_embedding_store.get_embedding(node_id)
                elif node_id.startswith('image-'):
                    emb = self.image_embedding_store.get_embedding(node_id)
                else:
                    return 0.0
                
                if emb is None:
                    return 0.0
                emb = np.array(emb)
                return float(np.dot(query_embedding.flatten(), emb.flatten()))
            except Exception as e:
                logger.debug(f"Error getting similarity for {node_id}: {e}")
                return 0.0
        
        def get_edge_cost(edge_type: str, edge_weight: float) -> float:
            """计算边的代价"""
            if edge_type == 'synonymy':
                # 同义词边：相似度越高，代价越低
                return max(0.01, 1.0 - edge_weight)  # 相似度0.95 → 代价0.05
            elif edge_type == 'semantic':
                # 语义边：共现次数越多，代价越低
                return 0.5 / max(edge_weight, 0.1)
            elif edge_type == 'passage':
                # chunk-entity边
                return 0.2
            elif edge_type == 'structural':
                # file-chunk, chunk-code等结构边
                return 0.3
            else:
                return 0.5
        
        # 候选结果
        candidates = {
            'chunk': [],
            'code': [],
            'table': [],
            'image': []
        }
        
        # 优先队列: (-score, node_id, cumulative_cost, init_weight)
        pq = []
        visited = set()
        
        # 初始化种子节点
        seed_count = 0
        for entity_id, weight in seed_entities:
            vertex_idx = self.node_name_to_vertex_idx.get(entity_id)
            if vertex_idx is not None:
                sim = get_node_similarity(entity_id)
                score = sim * weight
                heapq.heappush(pq, (-score, entity_id, 0.0, weight))
                seed_count += 1
        
        for file_id, weight in seed_files:
            vertex_idx = self.node_name_to_vertex_idx.get(file_id)
            if vertex_idx is not None:
                sim = get_node_similarity(file_id)
                score = sim * weight
                heapq.heappush(pq, (-score, file_id, 0.0, weight))
                seed_count += 1
        
        if verbose:
            print(f"\n  🌱 扩散种子节点数: {seed_count} (Entity: {len(seed_entities)}, File: {len(seed_files)})")
        
        # 扩散
        expansion_count = 0
        while pq:
            neg_score, node_id, cost, init_weight = heapq.heappop(pq)
            
            if node_id in visited:
                continue
            visited.add(node_id)
            expansion_count += 1
            
            current_score = -neg_score
            
            # 收集内容节点（排除必选的chunk）
            if node_id.startswith('chunk-') and node_id not in must_have_chunks:
                if len(candidates['chunk']) < max_chunk_candidates:
                    candidates['chunk'].append((node_id, current_score))
            elif node_id.startswith('code-'):
                if len(candidates['code']) < max_code_candidates:
                    candidates['code'].append((node_id, current_score))
            elif node_id.startswith('table-'):
                if len(candidates['table']) < max_table_candidates:
                    candidates['table'].append((node_id, current_score))
            elif node_id.startswith('image-'):
                if len(candidates['image']) < max_image_candidates:
                    candidates['image'].append((node_id, current_score))
            
            # 检查是否已收集足够
            total_candidates = sum(len(v) for v in candidates.values())
            if total_candidates >= (max_chunk_candidates + max_code_candidates + 
                                   max_table_candidates + max_image_candidates):
                break
            
            # 扩散到邻居
            vertex_idx = self.node_name_to_vertex_idx.get(node_id)
            if vertex_idx is None:
                continue
            
            # 获取所有邻居
            neighbor_indices = self.graph.neighbors(vertex_idx, mode='all')
            
            for neighbor_idx in neighbor_indices:
                neighbor_id = self.graph.vs[neighbor_idx]['name']
                
                if neighbor_id in visited:
                    continue
                
                # 获取边信息
                edge_id = self.graph.get_eid(vertex_idx, neighbor_idx, error=False)
                if edge_id < 0:
                    # 尝试反向查找
                    edge_id = self.graph.get_eid(neighbor_idx, vertex_idx, error=False)
                
                if edge_id >= 0:
                    edge = self.graph.es[edge_id]
                    edge_type = edge['type'] if 'type' in edge.attributes() else 'structural'
                    edge_weight = edge['weight'] if 'weight' in edge.attributes() else 1.0
                else:
                    edge_type = 'structural'
                    edge_weight = 1.0
                
                edge_cost = get_edge_cost(edge_type, edge_weight)
                new_cost = cost + edge_cost
                
                if new_cost > max_cost:
                    continue
                
                # 计算新得分
                neighbor_sim = get_node_similarity(neighbor_id)
                decay = 1.0 / (1.0 + new_cost)
                new_score = neighbor_sim * decay * init_weight
                
                if new_score > 0.01:  # 过滤低分
                    heapq.heappush(pq, (-new_score, neighbor_id, new_cost, init_weight))
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"  🔍 扩散完成: 访问{expansion_count}个节点, 耗时{elapsed_time:.3f}s")
            print(f"      Chunk候选: {len(candidates['chunk'])}")
            print(f"      Code候选: {len(candidates['code'])}")
            print(f"      Table候选: {len(candidates['table'])}")
            print(f"      Image候选: {len(candidates['image'])}")
        
        return candidates

    def weighted_graph_search(self, query: str,
                               top_k_facts: List[Tuple],
                               top_files: List[Dict] = None,
                               candidate_k: int = 100,
                               max_hop: int = 3,
                               alpha: float = 0.3,  # 边权重系数
                               beta: float = 0.4,   # 节点相似度系数
                               gamma: float = 0.2,  # 路径衰减系数
                               delta: float = 0.1,  # 边类型加成系数
                               verbose: bool = False) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        加权图搜索：综合点权、边权和多种因素选出最佳候选
        
        基于Dijkstra变体的加权搜索算法，综合考虑：
        1. 点权（Node Weight）：节点与query的语义相似度
        2. 边权（Edge Weight）：图中存储的边权重
        3. 路径衰减：距离种子节点的跳数惩罚
        4. 边类型加成：不同类型边的权重加成
        
        算法流程:
        1. 从 rerank 得到的事实和文件中提取种子节点
        2. 预计算所有节点的点权（与query的相似度）
        3. 使用优先队列按综合得分进行扩展搜索
        4. 返回综合得分最高的 candidate_k 个内容节点
        
        综合得分公式:
        score = α * path_edge_weight + β * node_similarity + γ * hop_decay + δ * edge_type_bonus
        
        其中:
        - path_edge_weight: 路径上边权的累积（归一化）
        - node_similarity: 节点内容与query的相似度
        - hop_decay: 1/(hop+1)，距离衰减
        - edge_type_bonus: 边类型加成（semantic > synonymy > passage > structural）
        
        Args:
            query (str): 输入查询字符串
            top_k_facts (List[Tuple]): 重排序后的事实列表
            top_files (List[Dict]): 重排序后的文件列表
            candidate_k (int): 最终返回的候选数量（默认100）
            max_hop (int): 图搜索的最大跳数（默认3）
            alpha (float): 边权重系数（默认0.3）
            beta (float): 节点相似度系数（默认0.4）
            gamma (float): 路径衰减系数（默认0.2）
            delta (float): 边类型加成系数（默认0.1）
            verbose (bool): 是否输出详细信息
            
        Returns:
            Dict[str, List[Tuple[str, float, str]]]:
                各类型的候选结果，键为类型名 ('chunk', 'table', 'code')
                值为列表 [(node_key, score, content), ...]，按score降序排列
        """
        import heapq
        search_start = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 加权图搜索 (Weighted Graph Search)")
            print(f"   参数: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
            print(f"   max_hop={max_hop}, candidate_k={candidate_k}")
            print(f"{'='*60}")
        
        # ====== 步骤1: 预计算节点点权（与query的相似度）======
        precompute_start = time.time()
        
        # 获取query的嵌入向量
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self._batch_encode_texts(query,
                                                       instruction=get_query_instruction('query_to_passage'),
                                                       norm=True)
        
        # 预计算实体节点的点权（实体与query的相似度）
        entity_node_weights = {}
        if len(self.entity_embeddings) > 0:
            entity_scores = np.dot(self.entity_embeddings, query_embedding.T)
            entity_scores = np.squeeze(entity_scores) if entity_scores.ndim == 2 else entity_scores
            entity_scores = min_max_normalize(entity_scores)
            for i, key in enumerate(self.entity_node_keys):
                entity_node_weights[key] = entity_scores[i]
        
        # 预计算内容节点的点权（内容与query的相似度）
        content_node_weights = {}
        
        # 段落节点
        if len(self.passage_embeddings) > 0:
            passage_scores = np.dot(self.passage_embeddings, query_embedding.T)
            passage_scores = np.squeeze(passage_scores) if passage_scores.ndim == 2 else passage_scores
            passage_scores = min_max_normalize(passage_scores)
            for i, key in enumerate(self.passage_node_keys):
                content_node_weights[key] = passage_scores[i]
        
        # 代码节点
        if hasattr(self, 'code_node_keys') and self.code_node_keys:
            try:
                code_embeddings = np.array(self.code_embedding_store.get_embeddings(self.code_node_keys))
                if len(code_embeddings) > 0:
                    code_scores = np.dot(code_embeddings, query_embedding.T)
                    code_scores = np.squeeze(code_scores) if code_scores.ndim == 2 else code_scores
                    code_scores = min_max_normalize(code_scores)
                    for i, key in enumerate(self.code_node_keys):
                        content_node_weights[key] = code_scores[i]
            except Exception as e:
                logger.debug(f"Failed to compute code node weights: {e}")
        
        # 表格节点
        if hasattr(self, 'table_node_keys') and self.table_node_keys:
            try:
                table_embeddings = np.array(self.table_embedding_store.get_embeddings(self.table_node_keys))
                if len(table_embeddings) > 0:
                    table_scores = np.dot(table_embeddings, query_embedding.T)
                    table_scores = np.squeeze(table_scores) if table_scores.ndim == 2 else table_scores
                    table_scores = min_max_normalize(table_scores)
                    for i, key in enumerate(self.table_node_keys):
                        content_node_weights[key] = table_scores[i]
            except Exception as e:
                logger.debug(f"Failed to compute table node weights: {e}")
        
        precompute_time = time.time() - precompute_start
        
        if verbose:
            print(f"\n  📊 预计算点权完成:")
            print(f"      - 实体节点点权: {len(entity_node_weights)}")
            print(f"      - 内容节点点权: {len(content_node_weights)}")
            print(f"      - 耗时: {precompute_time:.3f}s")
        
        # ====== 步骤2: 提取种子节点并初始化 ======
        seed_node_ids = set()
        seed_node_scores = {}  # 种子节点的初始分数
        
        # 2.1 从事实中提取实体节点，并用事实相关性作为初始分数
        fact_scores = self.get_fact_scores(query) if len(self.fact_embeddings) > 0 else np.array([])
        
        for fact_idx, fact in enumerate(top_k_facts):
            subject_phrase = fact[0].lower()
            object_phrase = fact[2].lower()
            
            # 获取该事实的相关性分数（如果可用）
            fact_score = 1.0  # 默认分数
            
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    seed_node_ids.add(phrase_id)
                    # 使用实体相似度和事实排名作为初始分数
                    entity_sim = entity_node_weights.get(phrase_key, 0.5)
                    rank_bonus = 1.0 / (fact_idx + 1)  # 排名靠前的事实加成更高
                    seed_node_scores[phrase_id] = max(
                        seed_node_scores.get(phrase_id, 0),
                        entity_sim * 0.5 + rank_bonus * 0.5
                    )
        
        # 2.2 从文件中获取文件节点
        if top_files:
            for file_idx, file_info in enumerate(top_files):
                file_key = file_info.get('key', '')
                file_id = self.node_name_to_vertex_idx.get(file_key, None)
                if file_id is not None:
                    seed_node_ids.add(file_id)
                    rank_bonus = 1.0 / (file_idx + 1)
                    seed_node_scores[file_id] = max(
                        seed_node_scores.get(file_id, 0),
                        rank_bonus
                    )
        
        if verbose:
            print(f"\n  🌱 种子节点数: {len(seed_node_ids)}")
        
        if len(seed_node_ids) == 0:
            logger.warning("No seed nodes found for weighted graph search")
            return {
                'chunk': [],
                'table': [],
                'code': []
            }
        
        # ====== 步骤3: 定义边类型权重 ======
        edge_type_weights = {
            'semantic': 1.0,      # 语义边（实体关系）权重最高
            'synonymy': 0.8,      # 同义词边（相似实体）
            'passage': 0.6,       # 段落边（段落-实体）
            'structural': 0.4,    # 结构性边（层次关系）
            'unknown': 0.3        # 未知类型
        }
        
        # ====== 步骤4: 加权图搜索（优先队列） ======
        # 使用最大堆，存储 (-score, node_id, hop, path_edge_weight_sum)
        # 注意：heapq是最小堆，所以用负分数
        
        chunk_keys_set = set(self.passage_node_keys)
        code_keys_set = set(self.code_node_keys) if hasattr(self, 'code_node_keys') else set()
        table_keys_set = set(self.table_node_keys) if hasattr(self, 'table_node_keys') else set()
        
        # 结果字典：node_key -> (score, content_type)
        candidate_scores = {}
        
        # 初始化优先队列：从种子节点开始
        pq = []
        visited_with_score = {}  # node_id -> best_score
        
        for seed_id in seed_node_ids:
            initial_score = seed_node_scores.get(seed_id, 0.5)
            heapq.heappush(pq, (-initial_score, seed_id, 0, 0.0, []))  # (neg_score, node_id, hop, path_edge_sum, path_types)
            visited_with_score[seed_id] = initial_score
        
        search_iterations = 0
        max_iterations = len(self.graph.vs) * 2  # 防止无限循环
        
        while pq and search_iterations < max_iterations:
            search_iterations += 1
            neg_score, node_id, hop, path_edge_sum, path_types = heapq.heappop(pq)
            current_score = -neg_score
            
            # 如果已经访问过且当前分数不比之前好，跳过
            if node_id in visited_with_score and visited_with_score[node_id] > current_score:
                continue
            
            # 获取节点键
            node_key = self.graph.vs[node_id]['name']
            
            # 如果是内容节点（chunk/code/table），记录候选
            if node_key in chunk_keys_set:
                if node_key not in candidate_scores or candidate_scores[node_key][0] < current_score:
                    candidate_scores[node_key] = (current_score, 'chunk')
            elif node_key in code_keys_set:
                if node_key not in candidate_scores or candidate_scores[node_key][0] < current_score:
                    candidate_scores[node_key] = (current_score, 'code')
            elif node_key in table_keys_set:
                if node_key not in candidate_scores or candidate_scores[node_key][0] < current_score:
                    candidate_scores[node_key] = (current_score, 'table')
            
            # 超过最大跳数则不再扩展
            if hop >= max_hop:
                continue
            
            # 获取邻居节点并扩展
            neighbors = self.graph.neighbors(node_id, mode='all')
            
            for neighbor_id in neighbors:
                # 获取边信息
                try:
                    edge_id = self.graph.get_eid(node_id, neighbor_id, error=False)
                    if edge_id < 0:
                        edge_id = self.graph.get_eid(neighbor_id, node_id, error=False)
                    
                    if edge_id >= 0:
                        edge_weight = self.graph.es[edge_id]['weight'] if 'weight' in self.graph.es.attributes() else 1.0
                        edge_type = self.graph.es[edge_id]['type'] if 'type' in self.graph.es.attributes() else 'unknown'
                    else:
                        edge_weight = 1.0
                        edge_type = 'unknown'
                except:
                    edge_weight = 1.0
                    edge_type = 'unknown'
                
                # 归一化边权重（假设边权重范围在0-10之间）
                normalized_edge_weight = min(edge_weight / 10.0, 1.0) if edge_weight > 1 else edge_weight
                
                # 获取邻居节点的点权（与query的相似度）
                neighbor_key = self.graph.vs[neighbor_id]['name']
                
                # 确定节点类型并获取相似度
                if neighbor_key in entity_node_weights:
                    node_similarity = entity_node_weights[neighbor_key]
                elif neighbor_key in content_node_weights:
                    node_similarity = content_node_weights[neighbor_key]
                else:
                    node_similarity = 0.3  # 默认相似度
                
                # 计算边类型加成
                edge_type_bonus = edge_type_weights.get(edge_type, 0.3)
                
                # 计算新的累积路径边权
                new_path_edge_sum = path_edge_sum + normalized_edge_weight
                new_path_types = path_types + [edge_type]
                
                # 计算综合得分
                new_hop = hop + 1
                hop_decay = 1.0 / (new_hop + 1)  # 距离衰减
                avg_path_edge_weight = new_path_edge_sum / new_hop if new_hop > 0 else 0
                
                # 综合得分公式
                new_score = (
                    alpha * avg_path_edge_weight +      # 边权贡献
                    beta * node_similarity +             # 节点相似度贡献
                    gamma * hop_decay +                  # 距离衰减贡献
                    delta * edge_type_bonus              # 边类型加成
                )
                
                # 如果新分数更好，加入队列
                if neighbor_id not in visited_with_score or visited_with_score[neighbor_id] < new_score:
                    visited_with_score[neighbor_id] = new_score
                    heapq.heappush(pq, (-new_score, neighbor_id, new_hop, new_path_edge_sum, new_path_types))
        
        search_time = time.time() - search_start - precompute_time
        
        if verbose:
            print(f"\n  🔍 加权搜索完成:")
            print(f"      - 搜索迭代次数: {search_iterations}")
            print(f"      - 候选节点总数: {len(candidate_scores)}")
            print(f"      - 搜索耗时: {search_time:.3f}s")
        
        # ====== 步骤5: 按类型整理结果 ======
        results = {
            'chunk': [],
            'table': [],
            'code': []
        }
        
        for node_key, (score, content_type) in candidate_scores.items():
            results[content_type].append((node_key, score))
        
        # 每种类型按分数排序，取前 candidate_k 个
        for content_type in results:
            results[content_type] = sorted(results[content_type], key=lambda x: x[1], reverse=True)[:candidate_k]
            
            # 获取内容
            results_with_content = []
            for node_key, score in results[content_type]:
                content = self._get_content_by_type_and_key(content_type, node_key)
                results_with_content.append((node_key, score, content))
            results[content_type] = results_with_content
        
        total_time = time.time() - search_start
        
        if verbose:
            print(f"\n  📋 最终结果:")
            print(f"      - Chunk候选: {len(results['chunk'])}")
            print(f"      - Table候选: {len(results['table'])}")
            print(f"      - Code候选: {len(results['code'])}")
            print(f"\n  ⏱️ 总耗时: {total_time:.3f}s")
            
            # 打印每种类型的top-5
            for content_type in ['chunk', 'table', 'code']:
                if results[content_type]:
                    print(f"\n  📄 {content_type.upper()} Top-5:")
                    for i, (key, score, content) in enumerate(results[content_type][:5]):
                        preview = content[:80] + "..." if len(content) > 80 else content
                        preview = preview.replace('\n', ' ')
                        print(f"      [{i+1}] score={score:.4f} | {preview}")
        
        return results

    def graph_neighbor_rerank_v2(self, query: str,
                                  top_k_facts: List[Tuple],
                                  top_files: List[Dict] = None,
                                  candidate_k: int = 100,
                                  chunk_top_k: int = 10,
                                  table_top_k: int = 5,
                                  code_top_k: int = 5,
                                  max_hop: int = 3,
                                  use_llm_rerank: bool = True,
                                  alpha: float = 0.3,
                                  beta: float = 0.4,
                                  gamma: float = 0.2,
                                  delta: float = 0.1,
                                  verbose: bool = False) -> Dict[str, Tuple[List[str], np.ndarray, List[str]]]:
        """
        图邻居搜索 V2：结合加权图搜索 + 可选LLM重排序
        
        改进版的图邻居搜索，使用加权搜索算法替代简单BFS：
        1. 使用weighted_graph_search获取综合得分最高的候选
        2. 可选择性地使用LLM对候选进行最终重排序
        
        Args:
            query (str): 输入查询字符串
            top_k_facts (List[Tuple]): 重排序后的事实列表
            top_files (List[Dict]): 重排序后的文件列表
            candidate_k (int): 从图搜索中获取的候选数量（默认100）
            chunk_top_k (int): chunk最终保留数量
            table_top_k (int): table最终保留数量
            code_top_k (int): code最终保留数量
            max_hop (int): 图搜索的最大跳数
            use_llm_rerank (bool): 是否使用LLM重排序（默认True）
            alpha (float): 边权重系数
            beta (float): 节点相似度系数
            gamma (float): 路径衰减系数
            delta (float): 边类型加成系数
            verbose (bool): 是否输出详细信息
            
        Returns:
            Dict[str, Tuple[List[str], np.ndarray, List[str]]]:
                各类型的排序结果，键为类型名 ('chunk', 'table', 'code')
                值为元组 (contents, scores, node_keys)
        """
        search_start = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 图邻居搜索 V2 (Weighted Search + Optional LLM Rerank)")
            print(f"{'='*60}")
        
        # ====== 步骤1: 加权图搜索 ======
        weighted_results = self.weighted_graph_search(
            query=query,
            top_k_facts=top_k_facts,
            top_files=top_files,
            candidate_k=candidate_k,
            max_hop=max_hop,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            verbose=verbose
        )
        
        # ====== 步骤2: 可选LLM重排序 ======
        results = {}
        
        for content_type, top_k in [('chunk', chunk_top_k), ('table', table_top_k), ('code', code_top_k)]:
            candidates = weighted_results.get(content_type, [])
            
            if len(candidates) == 0:
                results[content_type] = ([], np.array([]), [])
                continue
            
            candidate_keys = [item[0] for item in candidates]
            candidate_scores = [item[1] for item in candidates]
            candidate_contents = [item[2] for item in candidates]
            
            if use_llm_rerank and len(candidates) > top_k:
                # 使用LLM重排序
                if verbose:
                    print(f"\n  🔄 {content_type.upper()} LLM重排序: {len(candidates)}个候选 -> {top_k}个")
                
                rerank_start = time.time()
                try:
                    top_indices, top_contents_list, _ = self._rerank_contents(
                        query, candidate_contents, candidate_keys,
                        content_type=content_type,
                        len_after_rerank=top_k
                    )
                    rerank_time = time.time() - rerank_start
                    
                    # 构建结果
                    top_keys = [candidate_keys[i] for i in top_indices]
                    top_contents = [candidate_contents[i] for i in top_indices]
                    # 结合原始加权分数和重排序位置
                    scores = np.array([candidate_scores[i] * (1.0 - j * 0.05) for j, i in enumerate(top_indices)])
                    
                    if verbose:
                        print(f"      LLM重排序耗时: {rerank_time:.3f}s")
                    
                    results[content_type] = (top_contents, scores, top_keys)
                except Exception as e:
                    logger.warning(f"LLM rerank failed for {content_type}: {e}, falling back to weighted scores")
                    # 回退到加权分数排序
                    top_contents = candidate_contents[:top_k]
                    top_keys = candidate_keys[:top_k]
                    scores = np.array(candidate_scores[:top_k])
                    results[content_type] = (top_contents, scores, top_keys)
            else:
                # 直接使用加权分数排序的结果
                top_contents = candidate_contents[:top_k]
                top_keys = candidate_keys[:top_k]
                scores = np.array(candidate_scores[:top_k])
                results[content_type] = (top_contents, scores, top_keys)
        
        total_time = time.time() - search_start
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  ✅ V2搜索完成:")
            print(f"      - Chunk: {len(results['chunk'][0])}个")
            print(f"      - Table: {len(results['table'][0])}个")
            print(f"      - Code: {len(results['code'][0])}个")
            print(f"  ⏱️ 总耗时: {total_time:.3f}s")
            print(f"{'='*60}\n")
        
        return results

    def _rerank_contents(self, query: str, contents: List[str], keys: List[str], 
                         content_type: str, len_after_rerank: int) -> Tuple[List[int], List[str], dict]:
        """
        对内容进行LLM重排序
        
        Args:
            query: 查询字符串
            contents: 内容列表
            keys: 键列表
            content_type: 内容类型
            len_after_rerank: 重排序后保留的数量
            
        Returns:
            (top_indices, top_contents, rerank_log)
        """
        try:
            # 准备候选内容（截断太长的内容）
            truncated_contents = []
            for content in contents:
                if len(content) > 2000:
                    content = content[:2000] + "..."
                truncated_contents.append(content)
            
            # 调用重排序过滤器
            top_indices, top_contents_list, reranker_dict = self.rerank_filter.rerank_contents(
                query=query,
                contents=truncated_contents,
                content_type=content_type,
                len_after_rerank=len_after_rerank
            )
            
            # 返回原始完整内容
            top_full_contents = [contents[i] for i in top_indices]
            
            return top_indices, top_full_contents, reranker_dict
            
        except Exception as e:
            logger.error(f"Error in _rerank_contents: {str(e)}")
            # 降级：直接返回前 top_k 个
            top_k = min(len_after_rerank, len(contents))
            return list(range(top_k)), contents[:top_k], {'error': str(e)}

    def graph_search_by_type(self, query: str,
                              link_top_k: int,
                              query_fact_scores: np.ndarray,
                              top_k_facts: List[Tuple],
                              top_k_fact_indices: List[str],
                              passage_node_weight: float = 0.05,
                              top_files: List[Dict] = None,
                              file_node_weight: float = 0.1) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        分类型图搜索：分别返回 chunk、table、code 的排序结果（PPR版本，已弃用）
        
        与 graph_search_with_fact_entities 类似，但返回分类型的结果。
        建议使用 graph_neighbor_rerank 方法替代。
        
        Args:
            query (str): 输入查询字符串
            link_top_k (int): 从链接分数映射中包含的顶级短语数量
            query_fact_scores (np.ndarray): 事实-查询相似性的分数数组
            top_k_facts (List[Tuple]): 顶级事实列表
            top_k_fact_indices (List[str]): 顶级事实索引
            passage_node_weight (float): 段落分数权重
            top_files (List[Dict]): 相关文件列表
            file_node_weight (float): 文件节点权重
            
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
                各类型的排序结果，键为类型名 ('chunk', 'table', 'code')
        """
        # 复用原有的权重计算逻辑
        linking_score_map = {}
        phrase_scores = {}
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        file_weights = np.zeros(len(self.graph.vs['name']))

        # 遍历顶级事实，为相关短语分配权重
        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
                
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score
                    if self.ent_node_to_chunk_ids is not None and len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        phrase_weights[phrase_id] /= len(self.ent_node_to_chunk_ids[phrase_key])
                    if phrase not in phrase_scores:
                        phrase_scores[phrase] = []
                    phrase_scores[phrase].append(fact_score)

        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k, phrase_weights, linking_score_map)

        # DPR分数
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            if len(self.all_content_node_keys) > len(self.passage_node_keys):
                content_node_key = self.all_content_node_keys[dpr_sorted_doc_id]
                content_dpr_score = normalized_dpr_sorted_scores[i]
                content_node_id = self.node_name_to_vertex_idx[content_node_key]
                passage_weights[content_node_id] = content_dpr_score * passage_node_weight
            else:
                passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
                passage_dpr_score = normalized_dpr_sorted_scores[i]
                passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
                passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight

        # 文件权重
        if top_files and len(top_files) > 0:
            for file_info in top_files:
                file_key = file_info.get('key', '')
                file_score = file_info.get('score', 0.5)
                file_node_id = self.node_name_to_vertex_idx.get(file_key, None)
                if file_node_id is not None:
                    file_weights[file_node_id] = file_score * file_node_weight

        node_weights = phrase_weights + passage_weights + file_weights

        if sum(node_weights) <= 0:
            logger.warning(f'No weights found for graph search')
            if sum(file_weights) <= 0:
                # 返回空结果
                return {
                    'chunk': (np.array([]), np.array([]), []),
                    'table': (np.array([]), np.array([]), []),
                    'code': (np.array([]), np.array([]), [])
                }

        # 运行分类型PPR
        ppr_start = time.time()
        results = self.run_ppr_by_type(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()
        self.ppr_time += (ppr_end - ppr_start)

        return results

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
        all_images = []
        all_entities = []
        all_facts = []
        
        # 图结构连接信息
        structure_edges = []  # 结构性边 (parent -> child, jump等)
        semantic_edges = []   # 语义边 (实体关系)
        
        # 递归处理JSON结构
        self._extract_nodes_from_json(json_structure, all_files, all_chunks, all_codes, 
                                     all_tables, all_images, all_entities, all_facts,
                                     structure_edges, semantic_edges)
        
        logger.info(f"Extracted {len(all_files)} files, {len(all_chunks)} chunks, "
                   f"{len(all_codes)} codes, {len(all_tables)} tables, {len(all_images)} images, "
                   f"{len(all_entities)} entities, {len(all_facts)} facts")

        # 对各类节点进行向量化编码 
        logger.info(f"Encoding files") # 8275
        if all_files:
            # all_files 格式: [(node_id, content, abstract, file_path), ...]
            # 将本地文件路径转换为 Gitee URL
            gitee_file_paths = [self._convert_to_gitee_url(file[3]) if len(file) > 3 else '' for file in all_files]
            self.file_embedding_store.insert_strings(
                hash_ids=[file[0] for file in all_files], 
                contents=[file[1] for file in all_files], 
                summaries=[file[2] for file in all_files],
                file_paths=gitee_file_paths
            )
            
        logger.info(f"Encoding chunks") # 71496
        if all_chunks:
            # all_chunks 格式: [(node_id, content, abstract, file_path), ...]
            self.chunk_embedding_store.insert_strings(
                hash_ids=[chunk[0] for chunk in all_chunks], 
                contents=[chunk[1] for chunk in all_chunks], 
                summaries=[chunk[2] for chunk in all_chunks],
                file_paths=[chunk[3] if len(chunk) > 3 else '' for chunk in all_chunks]
            )
            
        logger.info(f"Encoding codes") # 27776
        if all_codes:
            # all_codes 格式: [(node_id, content, abstract, file_path), ...]
            self.code_embedding_store.insert_strings(
                hash_ids=[code[0] for code in all_codes], 
                contents=[code[1] for code in all_codes], 
                summaries=[code[2] for code in all_codes],
                file_paths=[code[3] if len(code) > 3 else '' for code in all_codes]
            )
            
        logger.info(f"Encoding tables") # 18732
        if all_tables:
            # all_tables 格式: [(node_id, content, abstract, file_path), ...]
            self.table_embedding_store.insert_strings(
                hash_ids=[table[0] for table in all_tables], 
                contents=[table[1] for table in all_tables], 
                summaries=[table[2] for table in all_tables],
                file_paths=[table[3] if len(table) > 3 else '' for table in all_tables]
            )
        
        logger.info(f"Encoding images") # 6954
        if all_images:
            # all_images 格式: [(node_id, absolute_path, caption/embed_text, gitee_url), ...]
            # 使用 caption 作为摘要（用于计算embedding），gitee_url 作为 file_path
            self.image_embedding_store.insert_strings(
                hash_ids=[img[0] for img in all_images],
                contents=[img[1] for img in all_images],    # absolute_path (原始内容)
                summaries=[img[2] for img in all_images],   # caption (用于embedding)
                file_paths=[img[3] for img in all_images]   # gitee_url
            )
            
        logger.info(f"Encoding entities") # 301490
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
        
        # 保存节点元信息（面包屑导航等）
        self._save_node_metadata()
        logger.info(f"Saved {len(self.node_id_to_metadata)} node metadata entries (breadcrumbs)")
        
        logger.info(f"Hierarchical indexing completed!")
        print(self.get_graph_info())

    def _extract_nodes_from_json(self, json_structure: Dict[str, Any], 
                                all_files: List[Tuple[str, str, str, str]], all_chunks: List[Tuple[str, str, str, str]],
                                all_codes: List[Tuple[str, str, str, str]], all_tables: List[Tuple[str, str, str, str]],
                                all_images: List[Tuple[str, str, str, str]],
                                all_entities: List[Tuple[str, str, str]], all_facts: List[Triple],
                                structure_edges: List[Tuple[str, str, str, float]],
                                semantic_edges: List[Triple],
                                parent_id: str = None,
                                current_file_path: str = ''):
        """
        递归提取JSON结构中的所有节点和边信息
        
        Args:
            json_structure: JSON结构
            all_files: 文件节点列表 [(node_id, content, abstract, file_path), ...]
            all_chunks: 段落节点列表 [(node_id, content, abstract, file_path), ...]
            all_codes: 代码节点列表 [(node_id, content, abstract, file_path), ...]
            all_tables: 表格节点列表 [(node_id, content, abstract, file_path), ...]
            all_images: 图片节点列表 [(node_id, absolute_path, caption, gitee_url), ...]
            all_entities: 实体节点列表
            all_facts: 事实三元组列表
            structure_edges: 结构性边列表 (source_id, target_id, edge_type, weight)
            semantic_edges: 语义边列表 (三元组)
            parent_id: 父节点ID
            current_file_path: 当前文件路径（用于chunk/code/table的文件定位）
        """
        for node_id, node_data in json_structure.items():
            if node_id.startswith('file-'):
                # 处理文件节点
                file_abstract = node_data.get('abstract', '')
                file_content = node_data.get('content','')
                file_path = node_data.get('file_path', '')  # 提取文件路径
                if file_abstract and file_content:
                    # 存储为4元组: (node_id, content, abstract, file_path)
                    all_files.append((node_id, file_content, file_abstract, file_path))
                    
                # 处理文件的chunk子节点，传递文件路径
                chunks_data = node_data.get('chunks', {})
                if chunks_data:
                    self._extract_nodes_from_json(chunks_data, all_files, all_chunks,
                                                 all_codes, all_tables, all_images, all_entities, all_facts,
                                                 structure_edges, semantic_edges, parent_id=node_id,
                                                 current_file_path=file_path)
                    
            elif node_id.startswith('chunk-'):
                # 处理段落节点
                chunk_abstract = node_data.get('abstract', '')
                chunk_content = node_data.get('content', '')
                
                # 转换为 Gitee URL
                gitee_file_path = self._convert_to_gitee_url(current_file_path) if current_file_path else ''
                
                # 提取并存储 metadata（面包屑导航）
                chunk_metadata = node_data.get('metadata', {})
                chunk_breadcrumb = self._build_breadcrumb(chunk_metadata)
                if chunk_breadcrumb or chunk_metadata:
                    self.node_id_to_metadata[node_id] = {
                        'breadcrumb': chunk_breadcrumb,
                        'metadata': chunk_metadata,
                        'file_path': gitee_file_path
                    }
                
                # 使用摘要或内容作为嵌入文本
                # embed_text = chunk_abstract if chunk_abstract else chunk_content
                if chunk_abstract and chunk_content:
                    # 存储为4元组: (node_id, content, abstract, file_path)
                    all_chunks.append((node_id, chunk_content, chunk_abstract, gitee_file_path))
                
                # 添加段落到父节点的边
                if parent_id:
                    structure_edges.append((parent_id, node_id, 'contains', 1.0))
                
                # 处理跳转关系
                jump_data = node_data.get('jump', {})
                for jump_file_id, jump_info in jump_data.items():
                    structure_edges.append((node_id, jump_file_id, 'jump', 1.0))
                
                # 处理代码块（继承父chunk的面包屑）
                codes_data = node_data.get('codes', {})
                for code_id, code_info in codes_data.items():
                    code_abstract = code_info.get('abstract', '')
                    code_content = code_info.get('content', '')
                    # embed_text = code_abstract if code_abstract else code_content
                    if code_abstract and code_content:
                        # 存储为4元组: (node_id, content, abstract, file_path)
                        all_codes.append((code_id, code_content, code_abstract, gitee_file_path))
                        structure_edges.append((node_id, code_id, 'contains', 1.0))
                        # code继承父chunk的面包屑
                        self.node_id_to_metadata[code_id] = {
                            'breadcrumb': chunk_breadcrumb,
                            'metadata': chunk_metadata,
                            'file_path': gitee_file_path,
                            'parent_chunk_id': node_id
                        }
                
                # 处理表格（继承父chunk的面包屑）
                tables_data = node_data.get('tables', {})
                for table_id, table_info in tables_data.items():
                    table_abstract = table_info.get('abstract', '')
                    table_content = table_info.get('content', '')
                    # embed_text = table_abstract if table_abstract else table_content
                    if table_abstract and table_content:
                        # 存储为4元组: (node_id, content, abstract, file_path)
                        all_tables.append((table_id, table_content, table_abstract, gitee_file_path))
                        structure_edges.append((node_id, table_id, 'contains', 1.0))
                        # table继承父chunk的面包屑
                        self.node_id_to_metadata[table_id] = {
                            'breadcrumb': chunk_breadcrumb,
                            'metadata': chunk_metadata,
                            'file_path': gitee_file_path,
                            'parent_chunk_id': node_id
                        }
                
                # 处理图片（继承父chunk的面包屑）
                images_data = node_data.get('images', {})
                for image_id, image_info in images_data.items():
                    caption = image_info.get('caption', '')
                    context = image_info.get('context', '')
                    absolute_path = image_info.get('absolute_path', '')
                    
                    # 使用 caption 作为嵌入文本（如果没有caption，使用context）
                    embed_text = caption if caption else context
                    if embed_text and absolute_path:
                        # 转换为 Gitee URL
                        gitee_url = self._convert_to_gitee_url(absolute_path)
                        # 存储为4元组: (node_id, absolute_path, caption/embed_text, gitee_url)
                        all_images.append((image_id, absolute_path, embed_text, gitee_url))
                        structure_edges.append((node_id, image_id, 'contains', 1.0))
                        # image继承父chunk的面包屑，并存储来源信息
                        self.node_id_to_metadata[image_id] = {
                            'breadcrumb': chunk_breadcrumb,
                            'metadata': chunk_metadata,
                            'file_path': gitee_file_path,  # gitee url
                            'md_file_path': current_file_path,  # 原始md文件路径
                            'parent_chunk_id': node_id,  # 来源chunk ID
                            'image_path': absolute_path  # 图片本地路径
                        }
                
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
                            fact_id = compute_mdhash_id(embed_text, "fact-")
                            all_facts.append((fact_id, embed_text, embed_text))
                            
                            # 计算实体ID
                            entity_desc_1 = entity_definitions.get(processed_triple[0], '')
                            entity_desc_2 = entity_definitions.get(processed_triple[2], '')
                            embed_text_1 = f"{processed_triple[0]}: {entity_desc_1}" if entity_desc_1 else processed_triple[0]
                            embed_text_2 = f"{processed_triple[2]}: {entity_desc_2}" if entity_desc_2 else processed_triple[2]
                            subject_entity_id = compute_mdhash_id(embed_text_1, "entity-")
                            object_entity_id = compute_mdhash_id(embed_text_2, "entity-")
                            
                            # 【新增】记录fact到chunk的映射
                            self.fact_to_chunk_id[fact_id] = node_id  # node_id是当前的chunk-xxxx
                            
                            # 【新增】记录fact到实体的映射
                            self.fact_to_entities[fact_id] = (subject_entity_id, object_entity_id)
                            
                            # 添加语义边
                            semantic_edges.append((subject_entity_id, processed_triple[1], object_entity_id))
                
                # 递归处理子chunks（继承当前文件路径）
                sub_chunks = node_data.get('chunks', {})
                if sub_chunks:
                    self._extract_nodes_from_json(sub_chunks, all_files, all_chunks,
                                                 all_codes, all_tables, all_images, all_entities, all_facts,
                                                 structure_edges, semantic_edges, parent_id=node_id,
                                                 current_file_path=current_file_path)

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

    def hierarchical_rag_qa_example(self, json_structure: Dict[str, Any], queries: List[str]):
        """
        层次化RAG问答的完整使用示例
        
        此方法展示了如何使用HippoRAG处理层次化JSON结构并进行问答。
        适用于包含文件、段落、代码、表格等多种内容类型的复杂文档结构。
        
        Args:
            json_structure: 层次化的JSON文档结构
            queries: 查询列表
            
        Returns:
            问答结果
            
        使用流程:
        1. 使用index_from_json索引层次化结构
        2. 使用rag_qa进行多类型内容的问答
        """
        logger.info("Starting hierarchical RAG-QA example")
        
        # 步骤1: 索引层次化JSON结构
        logger.info("Indexing hierarchical JSON structure...")
        self.index_from_json(json_structure)
        
        # 步骤2: 执行RAG问答
        logger.info("Performing RAG-QA on hierarchical content...")
        query_solutions, response_messages, metadata = self.rag_qa(queries)
        
        # 打印结果摘要
        logger.info(f"Completed {len(queries)} queries")
        for i, (query, solution) in enumerate(zip(queries, query_solutions)):
            logger.info(f"Query {i+1}: {query}")
            logger.info(f"Answer: {solution.answer}")
            
        return query_solutions, response_messages, metadata
