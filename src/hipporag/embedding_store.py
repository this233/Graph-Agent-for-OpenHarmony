import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput

logger = logging.getLogger(__name__)

class EmbeddingStore:
    """
    嵌入向量存储类
    
    这个类用于管理文本及其对应的嵌入向量，提供持久化存储功能。
    使用哈希ID作为唯一标识符，支持批量插入、查询和删除操作。
    数据以parquet格式存储在本地文件系统中。
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

    def get_missing_string_hash_ids(self, texts: List[str]):
        """
        获取尚未存储在数据库中的文本对应的哈希ID和内容
        
        Args:
            texts: 要检查的文本列表
            
        Returns:
            dict: 包含缺失文本的字典，键为哈希ID，值为包含hash_id和content的字典
        
        功能说明:
        - 为每个输入文本生成哈希ID
        - 检查哪些哈希ID在数据库中不存在
        - 返回缺失的哈希ID及其对应的文本内容
        """
        # 为每个文本生成哈希ID和内容映射
        nodes_dict = {}
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # 获取所有哈希ID
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return {}

        # 获取已存在的哈希ID集合
        existing = self.hash_id_to_row.keys()

        # 筛选出不存在的哈希ID
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        # 返回缺失的数据字典
        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]):
        """
        批量插入文本字符串及其嵌入向量
        
        Args:
            texts: 要插入的文本列表
            
        功能说明:
        - 为每个文本生成唯一的哈希ID
        - 检查哪些文本尚未存储
        - 对新文本生成嵌入向量
        - 将新数据插入到存储中
        - 自动跳过已存在的文本
        """
        # 为每个文本生成哈希ID和内容映射
        nodes_dict = {}
        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

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

        # 准备要编码的文本
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        # 使用嵌入模型批量生成向量
        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        # 插入新数据
        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        """
        从parquet文件加载已有数据
        
        功能说明:
        - 如果数据文件存在，则读取并解析数据
        - 构建各种索引映射（哈希ID到索引、哈希ID到行数据等）
        - 如果文件不存在，则初始化空的数据结构
        - 验证数据完整性（哈希ID、文本、嵌入向量数量一致）
        """
        if os.path.exists(self.filename):
            # 读取parquet文件
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), df["embedding"].values.tolist()
            
            # 构建各种索引映射，提高查询效率
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            
            # 验证数据完整性
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            # 初始化空的数据结构
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        """
        将数据保存到parquet文件
        
        功能说明:
        - 将内存中的数据构造为DataFrame
        - 保存为parquet格式文件
        - 重新构建索引映射，确保一致性
        - 记录保存的数据条数
        """
        # 构造DataFrame
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        
        # 保存为parquet文件
        data_to_save.to_parquet(self.filename, index=False)
        
        # 重新构建索引映射
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        """
        内部方法：插入或更新数据
        
        Args:
            hash_ids: 哈希ID列表
            texts: 文本列表
            embeddings: 嵌入向量列表
            
        功能说明:
        - 将新数据添加到内存中的列表
        - 调用保存方法持久化数据
        """
        # 扩展数据列表
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

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
            self.embeddings.pop(idx)

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