import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd
import torch

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput

logger = logging.getLogger(__name__)

class EmbeddingStoreV2:
    """
    嵌入向量存储类
    
    这个类用于管理文本、摘要及其对应的嵌入向量，提供持久化存储功能。
    使用外部传入的哈希ID作为唯一标识符，支持批量插入、查询和删除操作。
    数据以parquet格式存储在本地文件系统中。
    
    主要特性：
    - 支持外部传入的哈希ID，不再内部生成
    - 分别存储完整内容和摘要，用摘要计算embedding
    - 向后兼容旧版本数据格式
    """
    
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        初始化嵌入向量存储实例
        
        Args:
            embedding_model: 用于生成文本嵌入向量的模型
            db_filename: 数据存储目录的路径
            batch_size: 批处理大小，用于控制批量操作的数量
            namespace: 命名空间，用于数据隔离和哈希ID前缀
        
        功能说明:
        - 将传入的参数赋值给实例变量
        - 检查并创建存储目录（如果不存在）
        - 构建数据文件的完整路径（parquet格式）
        - 调用 _load_data() 方法加载已有数据
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        # 检查并创建工作目录
        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        # 构建数据文件路径，使用命名空间区分不同的数据集
        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        # 加载已有数据
        self._load_data()

    def _batch_encode_texts(self, texts):
        """
        使用新的嵌入模型API对文本进行批量编码
        
        Args:
            texts: 要编码的文本列表或单个文本
            
        Returns:
            embeddings: 嵌入向量列表
        """
        if isinstance(texts, str):
            texts = [texts]
        
        outputs = self.embedding_model.embed(texts)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        
        return embeddings.tolist()

    def get_missing_string_hash_ids(self, hash_ids: List[str], contents: List[str], summaries: List[str]):
        """
        获取尚未存储在数据库中的文本对应的哈希ID和内容
        
        Args:
            hash_ids: 外部传入的哈希ID列表
            contents: 完整内容列表
            summaries: 摘要列表（用于计算embedding）
            
        Returns:
            dict: 包含缺失文本的字典，键为哈希ID，值为包含hash_id、content和summary的字典
        
        功能说明:
        - 使用外部传入的哈希ID
        - 检查哪些哈希ID在数据库中不存在
        - 返回缺失的哈希ID及其对应的内容和摘要
        """
        # 验证输入参数长度一致
        if not (len(hash_ids) == len(contents) == len(summaries)):
            raise ValueError("hash_ids, contents, and summaries must have the same length")
        
        # 为每个哈希ID创建内容和摘要映射
        nodes_dict = {}
        for hash_id, content, summary in zip(hash_ids, contents, summaries):
            nodes_dict[hash_id] = {'content': content, 'summary': summary}

        # 获取所有哈希ID
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return {}

        # 获取已存在的哈希ID集合
        existing = self.hash_id_to_row.keys()

        # 筛选出不存在的哈希ID
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        # 返回缺失的数据字典
        return {h: {"hash_id": h, "content": nodes_dict[h]["content"], "summary": nodes_dict[h]["summary"]} for h in missing_ids}

    def insert_strings(self, hash_ids: List[str], contents: List[str], summaries: List[str], file_paths: List[str] = None):
        """
        批量插入文本字符串及其嵌入向量
        
        Args:
            hash_ids: 外部传入的哈希ID列表
            contents: 完整内容列表
            summaries: 摘要列表（用于计算embedding）
            file_paths: 文件路径列表（可选，用于文件类型的数据）
            
        功能说明:
        - 使用外部传入的哈希ID
        - 检查哪些哈希ID尚未存储
        - 对新摘要生成嵌入向量
        - 将新数据插入到存储中
        - 自动跳过已存在的记录
        """
        # 验证输入参数长度一致
        if not (len(hash_ids) == len(contents) == len(summaries)):
            raise ValueError("hash_ids, contents, and summaries must have the same length")
        
        # 如果提供了 file_paths，也需要验证长度
        if file_paths is not None and len(file_paths) != len(hash_ids):
            raise ValueError("file_paths must have the same length as hash_ids")
        
        # 为每个哈希ID创建内容、摘要和文件路径映射
        nodes_dict = {}
        for i, (hash_id, content, summary) in enumerate(zip(hash_ids, contents, summaries)):
            fp = file_paths[i] if file_paths else ''
            nodes_dict[hash_id] = {'content': content, 'summary': summary, 'file_path': fp}

        # 获取所有哈希ID
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # 没有要插入的内容

        # 获取已存在的哈希ID集合
        existing = self.hash_id_to_row.keys()

        # 筛选出需要插入的新哈希ID
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return {}  # 所有记录都已存在

        # 准备要编码的摘要（用于计算embedding）
        summaries_to_encode = [nodes_dict[hash_id]["summary"] for hash_id in missing_ids]
        # 准备要保存的完整内容
        contents_to_save = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        # 准备要保存的文件路径
        file_paths_to_save = [nodes_dict[hash_id]["file_path"] for hash_id in missing_ids]

        # 使用嵌入模型批量生成向量（基于摘要）
        missing_embeddings = self._batch_encode_texts(summaries_to_encode)

        # 插入新数据
        self._upsert(missing_ids, contents_to_save, summaries_to_encode, missing_embeddings, file_paths_to_save)

    def _load_data(self):
        """
        从parquet文件加载已有数据
        
        功能说明:
        - 如果数据文件存在，则读取并解析数据
        - 构建各种索引映射（哈希ID到索引、哈希ID到行数据等）
        - 如果文件不存在，则初始化空的数据结构
        - 验证数据完整性（哈希ID、文本、摘要、嵌入向量数量一致）
        - 支持可选的 file_path 字段（向后兼容）
        """
        if os.path.exists(self.filename):
            # 读取parquet文件
            df = pd.read_parquet(self.filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["content"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()
            
            # 处理摘要字段（兼容旧版本数据）
            if "summary" in df.columns:
                self.summaries = df["summary"].values.tolist()
            else:
                # 如果没有摘要字段，使用内容作为摘要以保持兼容性
                self.summaries = self.texts.copy()
            
            # 处理 file_path 字段（新增，向后兼容）
            if "file_path" in df.columns:
                self.file_paths = df["file_path"].values.tolist()
            else:
                # 如果没有 file_path 字段，初始化为空字符串列表
                self.file_paths = [''] * len(self.hash_ids)
            
            # 构建各种索引映射，提高查询效率
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t, "summary": s, "file_path": fp}
                for h, t, s, fp in zip(self.hash_ids, self.texts, self.summaries, self.file_paths)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            
            # 验证数据完整性
            assert len(self.hash_ids) == len(self.texts) == len(self.summaries) == len(self.embeddings) == len(self.file_paths)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            # 初始化空的数据结构
            self.hash_ids, self.texts, self.summaries, self.embeddings, self.file_paths = [], [], [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        """
        将数据保存到parquet文件
        
        功能说明:
        - 将内存中的数据构造为DataFrame
        - 保存为parquet格式文件
        - 重新构建索引映射，确保一致性
        - 记录保存的数据条数
        - 支持可选的 file_path 字段
        """
        # 构造DataFrame
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "summary": self.summaries,
            "embedding": self.embeddings,
            "file_path": self.file_paths
        })
        
        # 保存为parquet文件
        data_to_save.to_parquet(self.filename, index=False)
        
        # 重新构建索引映射
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t, "summary": s, "file_path": fp} 
            for h, t, s, fp in zip(self.hash_ids, self.texts, self.summaries, self.file_paths)
        }
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, summaries, embeddings, file_paths=None):
        """
        内部方法：插入或更新数据
        
        Args:
            hash_ids: 哈希ID列表
            texts: 文本列表
            summaries: 摘要列表
            embeddings: 嵌入向量列表
            file_paths: 文件路径列表（可选，默认为空字符串）
            
        功能说明:
        - 将新数据添加到内存中的列表
        - 调用保存方法持久化数据
        """
        # 如果没有提供 file_paths，使用空字符串填充
        if file_paths is None:
            file_paths = [''] * len(hash_ids)
        
        # 扩展数据列表
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.summaries.extend(summaries)
        self.file_paths.extend(file_paths)

        logger.info(f"Saving new records.")
        # 持久化到文件
        self._save_data()

    def delete(self, hash_ids):
        """
        删除指定哈希ID的数据记录
        
        Args:
            hash_ids: 要删除的哈希ID列表
            
        功能说明:
        - 找到要删除的记录在列表中的索引位置
        - 按索引倒序删除，避免索引位置变化的问题
        - 保存更新后的数据
        """
        # 获取要删除记录的索引
        indices = []
        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        # 按倒序排列索引，从后往前删除
        sorted_indices = np.sort(indices)[::-1]

        # 删除对应索引的数据
        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.summaries.pop(idx)
            self.embeddings.pop(idx)
            self.file_paths.pop(idx)

        logger.info(f"Saving record after deletion.")
        # 保存更新后的数据
        self._save_data()

    def get_row(self, hash_id):
        """
        根据哈希ID获取对应的行数据
        
        Args:
            hash_id: 哈希ID
            
        Returns:
            dict: 包含hash_id和content的字典
        """
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        """
        根据文本内容获取对应的哈希ID
        
        Args:
            text: 文本内容
            
        Returns:
            str: 对应的哈希ID
        """
        return self.text_to_hash_id[text]
    
    def get_summary(self, hash_id):
        """
        根据哈希ID获取对应的摘要
        
        Args:
            hash_id: 哈希ID
            
        Returns:
            str: 对应的摘要
        """
        return self.summaries[self.hash_id_to_idx[hash_id]]

    def get_rows(self, hash_ids, dtype=np.float32):
        """
        批量获取多个哈希ID对应的行数据
        
        Args:
            hash_ids: 哈希ID列表
            dtype: 数据类型（保留参数，与embeddings相关）
            
        Returns:
            dict: 哈希ID到行数据的映射字典
        """
        if not hash_ids:
            return {}

        # 批量获取行数据
        results = {id : self.hash_id_to_row[id] for id in hash_ids}
        return results

    def get_all_ids(self):
        """
        获取所有哈希ID的深拷贝列表
        
        Returns:
            List[str]: 所有哈希ID的列表副本
        """
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        """
        获取所有哈希ID到行数据映射的深拷贝
        
        Returns:
            dict: 哈希ID到行数据的映射字典副本
        """
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        """
        获取所有存储文本内容的集合
        
        Returns:
            set: 包含所有文本内容的集合
        """
        return set(row['content'] for row in self.hash_id_to_row.values())
    
    def get_all_summaries(self):
        """
        获取所有存储摘要内容的集合
        
        Returns:
            set: 包含所有摘要内容的集合
        """
        return set(row['summary'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        """
        获取单个哈希ID对应的嵌入向量
        
        Args:
            hash_id: 哈希ID
            dtype: 返回数组的数据类型，默认为float32
            
        Returns:
            np.ndarray: 对应的嵌入向量
        """
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> np.ndarray:
        """
        批量获取多个哈希ID对应的嵌入向量
        
        Args:
            hash_ids: 哈希ID列表
            dtype: 返回数组的数据类型，默认为float32
            
        Returns:
            np.ndarray: 包含所有对应嵌入向量的二维数组，形状为(len(hash_ids), embedding_dim)
        """
        if not hash_ids:
            return np.array([])

        # 获取对应的索引位置
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        # 根据索引获取嵌入向量
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings