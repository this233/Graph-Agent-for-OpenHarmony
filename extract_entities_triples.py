#!/usr/bin/env python3
"""
实体和三元组提取器

该脚本展示如何使用OpenIE为document_processor生成的JSON结构补充filter_chunk中的
extracted_entities和extracted_triples字段。
"""

import json
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, TypedDict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 定义ChunkInfo类型，与HippoRAG保持一致
class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]

# 设置API密钥
os.environ['OPENAI_API_KEY'] = 'siq3nBr8C75Pv89E0CQaKq4c3KTCpOREj8Umj8OMCM5ByKkBrHxm-IOPiLuFlEOjnU3HFE5Hv-sfLzShM8CCoA'

# 设置路径以便导入HippoRAG模块
import sys
sys.path.append('src')

from hipporag.information_extraction import OpenIE
from hipporag.information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from hipporag.llm import _get_llm_class
from hipporag.utils.config_utils import BaseConfig

class FallbackOpenIE:
    """
    带有备用配置的OpenIE
    
    当主配置的LLM调用失败时，自动切换到备用配置重试
    """
    
    def __init__(self, primary_config: BaseConfig, fallback_config: Optional[BaseConfig] = None):
        """
        初始化带备用配置的OpenIE
        
        Args:
            primary_config: 主配置
            fallback_config: 备用配置，当主配置失败时使用
        """
        self.primary_config = primary_config
        self.fallback_config = fallback_config
        self.logger = logging.getLogger(__name__)
        
        # 存储API密钥
        self.primary_api_key = os.environ.get('OPENAI_API_KEY')
        self.fallback_api_key = 'sk-0xdfGKYi0W6KOzcGC4B3958f6b6b482f8616A7E05eCa7aEb'
        
        # 统计fallback使用次数
        self.fallback_used_count = 0
        
        # 初始化主OpenIE（使用主API密钥）
        if self.primary_api_key:
            os.environ['OPENAI_API_KEY'] = self.primary_api_key
        self.primary_llm_model = _get_llm_class(primary_config)
        if primary_config.openie_mode == 'online':
            self.primary_openie = OpenIE(llm_model=self.primary_llm_model)
        elif primary_config.openie_mode == 'offline':
            self.primary_openie = VLLMOfflineOpenIE(primary_config)
        
        # 初始化备用OpenIE
        self.fallback_openie = None
        if fallback_config:
            try:
                # 临时切换到备用API密钥来初始化备用配置
                original_key = os.environ.get('OPENAI_API_KEY')
                os.environ['OPENAI_API_KEY'] = self.fallback_api_key
                
                try:
                    fallback_llm_model = _get_llm_class(fallback_config)
                    if fallback_config.openie_mode == 'online':
                        self.fallback_openie = OpenIE(llm_model=fallback_llm_model)
                    elif fallback_config.openie_mode == 'offline':
                        self.fallback_openie = VLLMOfflineOpenIE(fallback_config)
                    self.logger.info("备用OpenIE配置初始化成功")
                finally:
                    # 恢复原始API密钥
                    if original_key:
                        os.environ['OPENAI_API_KEY'] = original_key
                        
            except Exception as e:
                self.logger.warning(f"备用OpenIE配置初始化失败: {str(e)}")
    
    def batch_openie_with_fallback(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        使用备用配置机制进行批量OpenIE处理
        
        Args:
            chunks: 待处理的chunks字典
            
        Returns:
            Tuple[Dict, Dict]: NER结果字典和三元组结果字典，以及是否使用了备用配置的标记
        """
        primary_failed = False
        
        # 首先尝试主配置，确保使用主API密钥
        try:
            self.logger.info("使用主配置进行OpenIE处理...")
            # 确保使用主配置的API密钥
            if self.primary_api_key:
                os.environ['OPENAI_API_KEY'] = self.primary_api_key
            ner_results_dict, triple_results_dict = self.primary_openie.batch_openie(chunks)
            
            # 检查是否有成功的结果
            successful_ner = sum(1 for result in ner_results_dict.values() 
                               if result.unique_entities and not result.metadata.get('error'))
            successful_triples = sum(1 for result in triple_results_dict.values() 
                                   if result.triples and not result.metadata.get('error'))
            
            total_chunks = len(chunks)
            success_rate = (successful_ner + successful_triples) / (2 * total_chunks) if total_chunks > 0 else 0
            
            # 如果成功率大于50%，认为主配置工作正常
            if success_rate > 0.5:
                self.logger.info(f"主配置OpenIE处理成功，成功率: {success_rate:.1%}")
                return ner_results_dict, triple_results_dict
            else:
                primary_failed = True
                self.logger.warning(f"⚠️  主配置OpenIE处理成功率较低: {success_rate:.1%}")
                self.logger.warning(f"  成功的NER: {successful_ner}/{total_chunks}")
                self.logger.warning(f"  成功的三元组提取: {successful_triples}/{total_chunks}")
                self.logger.warning(f"  将尝试备用配置...")
                
        except Exception as e:
            primary_failed = True
            self.logger.warning(f"❌ 主配置OpenIE处理异常:")
            self.logger.warning(f"  错误: {str(e)}")
            self.logger.warning(f"  批次大小: {len(chunks)} 个chunks")
        
        # 如果主配置失败或成功率太低，且有备用配置，则尝试备用配置
        if self.fallback_openie and primary_failed:
            try:
                self.logger.info("=" * 60)
                self.logger.info("=== 使用备用配置重试OpenIE处理 ===")
                self.logger.info(f"批次大小: {len(chunks)} 个chunks")
                
                # 输出每个chunk的详细信息
                for i, (chunk_id, chunk_info) in enumerate(chunks.items()):
                    content = chunk_info['content']
                    content_preview = content[:150] + "..." if len(content) > 150 else content
                    self.logger.info(f"  Chunk {i+1}/{len(chunks)}")
                    self.logger.info(f"    ID: {chunk_id}")
                    self.logger.info(f"    内容长度: {len(content)} 字符")
                    self.logger.info(f"    内容预览: {content_preview}")
                    self.logger.info(f"    Token数: {chunk_info.get('num_tokens', 'N/A')}")
                    if i >= 2:  # 只显示前3个chunk的详细信息，避免日志过长
                        remaining = len(chunks) - 3
                        if remaining > 0:
                            self.logger.info(f"    ... 还有 {remaining} 个chunks")
                        break
                        
                self.logger.info("=" * 60)
                
                # 临时切换到备用API密钥
                original_key = os.environ.get('OPENAI_API_KEY')
                os.environ['OPENAI_API_KEY'] = self.fallback_api_key
                
                try:
                    ner_results_dict, triple_results_dict = self.fallback_openie.batch_openie(chunks)
                    self.fallback_used_count += 1
                    
                    # 检查备用配置的成功率
                    successful_ner = sum(1 for result in ner_results_dict.values() 
                                       if result.unique_entities and not result.metadata.get('error'))
                    successful_triples = sum(1 for result in triple_results_dict.values() 
                                           if result.triples and not result.metadata.get('error'))
                    
                    total_chunks = len(chunks)
                    success_rate = (successful_ner + successful_triples) / (2 * total_chunks) if total_chunks > 0 else 0
                    
                    self.logger.info(f"✅ 备用配置OpenIE处理完成，成功率: {success_rate:.1%}")
                    
                    # 输出备用配置处理结果摘要
                    total_entities = sum(len(result.unique_entities) for result in ner_results_dict.values())
                    total_triples = sum(len(result.triples) for result in triple_results_dict.values())
                    self.logger.info(f"备用配置提取结果: {total_entities} 个实体, {total_triples} 个三元组")
                    self.logger.info("=" * 60)
                    
                    # 标记结果使用了备用配置，用于后续统计调整
                    for result in ner_results_dict.values():
                        result.metadata['used_fallback'] = True
                    for result in triple_results_dict.values():
                        result.metadata['used_fallback'] = True
                    
                    return ner_results_dict, triple_results_dict
                    
                finally:
                    # 恢复原始API密钥
                    if original_key:
                        os.environ['OPENAI_API_KEY'] = original_key
                    
            except Exception as e:
                self.logger.error(f"备用配置OpenIE处理也失败: {str(e)}")
        
        # 所有配置都失败，返回空结果
        self.logger.error("主配置和备用配置都失败，返回空结果")
        empty_ner_results = {}
        empty_triple_results = {}
        
        for chunk_id in chunks.keys():
            from hipporag.utils.misc_utils import NerRawOutput, TripleRawOutput
            empty_ner_results[chunk_id] = NerRawOutput(
                chunk_id=chunk_id,
                response="",
                unique_entities=[],
                metadata={'error': '主配置和备用配置都失败'}
            )
            empty_triple_results[chunk_id] = TripleRawOutput(
                chunk_id=chunk_id,
                response="",
                triples=[],
                metadata={'error': '主配置和备用配置都失败'}
            )
        
        return empty_ner_results, empty_triple_results

class EntityTripleExtractor:
    """
    实体和三元组提取器
    
    用于从JSON结构中的filter_chunk内容提取实体和三元组
    """
    
    def __init__(self, global_config: BaseConfig, fallback_config: Optional[BaseConfig] = None):
        """
        初始化提取器
        
        Args:
            global_config: 全局配置对象
            fallback_config: 备用配置对象
        """
        self.global_config = global_config
        self.fallback_config = fallback_config
        self.stats = {
            'filter_chunks_processed': 0,
            'entities_extracted': 0,
            'triples_extracted': 0,
            'failed_calls': 0
        }
        
        # 使用带备用配置的OpenIE
        if fallback_config:
            self.openie = FallbackOpenIE(global_config, fallback_config)
            self.logger = logging.getLogger(__name__)
            self.logger.info("使用带备用配置的OpenIE")
        else:
            # 初始化LLM
            self.llm_model = _get_llm_class(self.global_config)
            
            # 根据配置选择OpenIE模式
            if self.global_config.openie_mode == 'online':
                self.openie = OpenIE(llm_model=self.llm_model)
            elif self.global_config.openie_mode == 'offline':
                self.openie = VLLMOfflineOpenIE(self.global_config)
            self.logger = logging.getLogger(__name__)
        
    @property
    def fallback_used_count(self):
        """获取备用配置使用次数"""
        if isinstance(self.openie, FallbackOpenIE):
            return self.openie.fallback_used_count
        return 0

    def _collect_filter_chunks(self, doc_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        递归收集所有需要处理的filter_chunk
        
        Args:
            doc_structure: 文档结构字典
            
        Returns:
            List[Dict]: 包含filter_chunk信息的任务列表
        """
        tasks = []
        
        def _recursive_collect(obj: Any, path: str = ""):
            """递归收集filter_chunk"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    if key == "filter_chunk" and isinstance(value, dict):
                        # 检查是否需要提取实体和三元组
                        content = value.get("content", "").strip()
                        entities = value.get("extracted_entities", [])
                        triples = value.get("extracted_triples", [])
                        
                        # 如果内容不为空且实体或三元组为空，则需要处理
                        if content and (not entities or not triples):
                            tasks.append({
                                "path": current_path,
                                "content": content,
                                "chunk_id": self._generate_chunk_id(content),
                                "current_entities": entities,
                                "current_triples": triples
                            })
                    else:
                        _recursive_collect(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _recursive_collect(item, f"{path}[{i}]")
        
        _recursive_collect(doc_structure)
        return tasks

    def _generate_chunk_id(self, content: str) -> str:
        """
        生成chunk ID（与document_processor保持一致）
        
        Args:
            content: chunk内容
            
        Returns:
            str: chunk ID
        """
        import hashlib
        return "chunk-" + hashlib.md5(content.encode()).hexdigest()

    def _extract_batch_chunks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        批量提取多个chunk的实体和三元组
        
        Args:
            tasks: 包含chunk信息的任务列表
            
        Returns:
            Dict: chunk_id -> 提取结果的字典
        """
        if not tasks:
            return {}
        
        try:
            # 准备批量chunk信息字典
            chunks_dict: Dict[str, ChunkInfo] = {}
            for task in tasks:
                chunk_id = task["chunk_id"]
                content = task["content"]
                
                chunk_info: ChunkInfo = {
                    "num_tokens": 0,  # 占位符，OpenIE不需要这个字段
                    "content": content,
                    "chunk_order": [],  # 占位符，OpenIE不需要这个字段
                    "full_doc_ids": []  # 占位符，OpenIE不需要这个字段
                }
                chunks_dict[chunk_id] = chunk_info
            
            self.logger.info(f"开始批量处理 {len(chunks_dict)} 个chunks...")
            
            # 批量调用OpenIE进行提取（使用带备用配置的方法）
            if isinstance(self.openie, FallbackOpenIE):
                ner_results_dict, triple_results_dict = self.openie.batch_openie_with_fallback(chunks_dict)
            else:
                ner_results_dict, triple_results_dict = self.openie.batch_openie(chunks_dict)
            
            # 整理结果
            results = {}
            fallback_success_count = 0
            
            for task in tasks:
                chunk_id = task["chunk_id"]
                
                ner_result = ner_results_dict.get(chunk_id)
                triple_result = triple_results_dict.get(chunk_id)
                
                entities = ner_result.unique_entities if ner_result else []
                triples = triple_result.triples if triple_result else []
                
                # 检查是否使用了备用配置且成功
                used_fallback = (ner_result and ner_result.metadata.get('used_fallback', False)) or \
                               (triple_result and triple_result.metadata.get('used_fallback', False))
                
                if used_fallback and (entities or triples):
                    fallback_success_count += 1
                
                results[chunk_id] = {
                    "extracted_entities": entities,
                    "extracted_triples": triples
                }
                
                # 更新统计
                self.stats['filter_chunks_processed'] += 1
                self.stats['entities_extracted'] += len(entities)
                self.stats['triples_extracted'] += len(triples)
                
                self.logger.debug(f"处理完成 chunk {chunk_id}: {len(entities)} 个实体, {len(triples)} 个三元组")
            
            # 如果备用配置成功了，需要从失败统计中减去（因为它们实际上是成功的）
            if fallback_success_count > 0:
                self.logger.info(f"备用配置成功处理了 {fallback_success_count} 个chunks")
                # 注意：这里不需要调整failed_calls，因为在批量处理中，
                # failed_calls是在整个批次失败时才增加的，而不是单个chunk失败时
                
            self.logger.info(f"批量处理完成，成功处理 {len(results)} 个chunks")
            return results
            
        except Exception as e:
            self.stats['failed_calls'] += len(tasks)
            self.logger.error(f"批量处理chunks时出错: {str(e)}")
            return {}

    def _update_structure_with_results(self, doc_structure: Dict[str, Any], 
                                     tasks: List[Dict[str, Any]], 
                                     results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        将提取结果更新到文档结构中
        
        Args:
            doc_structure: 原始文档结构
            tasks: 任务列表
            results: 提取结果
            
        Returns:
            Dict: 更新后的文档结构
        """
        # 创建深拷贝以避免修改原始结构
        import copy
        updated_structure = copy.deepcopy(doc_structure)
        
        # 构建路径到结果的映射
        path_to_result = {}
        for task in tasks:
            chunk_id = task["chunk_id"]
            if chunk_id in results:
                path_to_result[task["path"]] = results[chunk_id]
        
        def _update_recursive(obj: Any, path: str = ""):
            """递归更新结构"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    if key == "filter_chunk" and current_path in path_to_result:
                        # 更新filter_chunk的字段
                        result = path_to_result[current_path]
                        if "extracted_entities" in result:
                            obj[key]["extracted_entities"] = result["extracted_entities"]
                        if "extracted_triples" in result:
                            obj[key]["extracted_triples"] = result["extracted_triples"]
                    else:
                        _update_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _update_recursive(item, f"{path}[{i}]")
        
        _update_recursive(updated_structure)
        return updated_structure

    def process_document_structure(self, doc_structure: Dict[str, Any], 
                                 batch_size: int = 50) -> Dict[str, Any]:
        """
        处理整个文档结构，批量提取所有filter_chunk的实体和三元组
        
        Args:
            doc_structure: 文档结构字典
            batch_size: 批处理大小，每批处理的chunk数量
            
        Returns:
            Dict: 更新后的文档结构
        """
        self.logger.info("开始收集需要处理的filter_chunk...")
        
        # 收集所有需要处理的filter_chunk
        tasks = self._collect_filter_chunks(doc_structure)
        
        if not tasks:
            self.logger.info("没有找到需要处理的filter_chunk")
            return doc_structure
        
        self.logger.info(f"发现 {len(tasks)} 个filter_chunk需要处理")
        
        # 分批处理任务
        all_results = {}
        
        for i in tqdm(range(0, len(tasks), batch_size), desc="批量处理chunks"):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = self._extract_batch_chunks(batch_tasks)
            all_results.update(batch_results)
            
            self.logger.info(f"已完成 {min(i + batch_size, len(tasks))} / {len(tasks)} 个chunks")
        
        self.logger.info(f"总计成功处理 {len(all_results)} / {len(tasks)} 个filter_chunk")
        
        # 更新文档结构
        self.logger.info("更新文档结构...")
        updated_structure = self._update_structure_with_results(doc_structure, tasks, all_results)
        
        return updated_structure

    def save_updated_structure(self, updated_structure: Dict[str, Any], output_file: str):
        """
        保存更新后的文档结构
        
        Args:
            updated_structure: 更新后的文档结构
            output_file: 输出文件路径
        """
        try:
            # 确保输出目录存在
            output_dir = Path(output_file).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(updated_structure, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"更新后的结构已保存到: {output_file}")
            
        except Exception as e:
            self.logger.error(f"保存文件时出错: {str(e)}")
            raise


def create_fallback_config():
    """创建备用配置"""
    fallback_config = BaseConfig()
    fallback_config.llm_name = 'gpt-4.1'
    # fallback_config.llm_base_url = 'http://api.v36.cm/v1/'
    fallback_config.llm_base_url = 'https://api.vveai.com/v1'
    fallback_config.temperature = 0.1
    fallback_config.openie_mode = 'online'  # 备用配置使用在线模式
    
    # 不在这里修改API密钥，而是在实际使用时临时切换
    return fallback_config

def setup_logging(level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('entity_triple_extraction.log')
        ]
    )


def validate_input_file(file_path: str) -> bool:
    """验证输入文件格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 简单验证：检查是否包含文件级别的结构
        if not isinstance(data, dict):
            return False
            
        # 检查是否有文件级别的条目
        for key, value in data.items():
            if key.startswith('file-') and isinstance(value, dict):
                if 'content' in value and 'file_path' in value:
                    return True
        
        return False
        
    except Exception as e:
        print(f"验证输入文件失败: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="为文档结构JSON提取实体和三元组",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python extract_entities_triples.py input.json output.json
  python extract_entities_triples.py input.json output.json --batch-size 100 --llm-name gpt-4
  
支持的LLM模型:
  - gpt-3.5-turbo (默认)
  - gpt-4
  - gpt-4-turbo
  - DeepSeek-V3
  - 其他OpenAI兼容的模型
        """
    )
    
    parser.add_argument(
        "input_file", 
        help="输入的文档结构JSON文件路径"
    )
    
    parser.add_argument(
        "output_file", 
        help="输出的文档结构JSON文件路径"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=200, 
        help="批处理大小，每批处理的chunk数量 (默认: 50)"
    )
    
    parser.add_argument(
        "--llm-name", 
        default="gpt-3.5-turbo", 
        help="LLM模型名称 (默认: gpt-3.5-turbo)"
    )
    
    parser.add_argument(
        "--llm-base-url", 
        help="LLM服务的基础URL（用于自定义API端点）"
    )
    
    parser.add_argument(
        "--openie-mode", 
        choices=['online', 'offline'],
        default='online', 
        help="OpenIE模式：online或offline (默认: online)"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.1, 
        help="LLM生成温度 (默认: 0.1)"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="只分析任务，不实际进行提取"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="启用详细日志输出"
    )
    
    parser.add_argument(
        "--disable-fallback", 
        action="store_true", 
        help="禁用备用配置，只使用主配置"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    
    # 验证输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    if not validate_input_file(args.input_file):
        logger.error(f"输入文件格式无效: {args.input_file}")
        logger.error("请确保输入文件是由document_processor生成的有效JSON结构")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载文档结构
    logger.info(f"加载文档结构: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        doc_structure = json.load(f)
    
    logger.info(f"文档结构包含 {len(doc_structure)} 个文件")
    
    # 创建配置
    config = BaseConfig()
    config.llm_name = args.llm_name
    config.temperature = args.temperature
    config.openie_mode = args.openie_mode
    
    if args.llm_base_url:
        config.llm_base_url = args.llm_base_url
    
    # 使用DeepSeek配置
    # if args.llm_name == 'DeepSeek-V3':
    #     config.llm_base_url = 'https://api.modelarts-maas.com/v1'
    config.llm_base_url = 'https://api.modelarts-maas.com/v1'
    config.llm_name = 'DeepSeek-V3'

    # 创建备用配置
    fallback_config = None
    if not args.disable_fallback:
        try:
            fallback_config = create_fallback_config()
            logger.info("备用配置已创建")
        except Exception as e:
            logger.warning(f"创建备用配置失败: {str(e)}")

    # 创建提取器
    logger.info("初始化实体和三元组提取器...")
    if fallback_config and not args.disable_fallback:
        extractor = EntityTripleExtractor(global_config=config, fallback_config=fallback_config)
        logger.info("使用带备用配置的实体和三元组提取器")
    else:
        extractor = EntityTripleExtractor(global_config=config)
        logger.info("使用标准实体和三元组提取器")
    
    if args.dry_run:
        # 只分析任务，不实际提取
        logger.info("=== 执行预分析 ===")
        tasks = extractor._collect_filter_chunks(doc_structure)
        
        logger.info(f"发现 {len(tasks)} 个filter_chunk需要处理")
        
        # 统计内容长度
        total_chars = sum(len(task['content']) for task in tasks)
        avg_chars = total_chars / len(tasks) if tasks else 0
        
        logger.info(f"平均内容长度: {avg_chars:.1f} 字符")
        logger.info(f"总内容长度: {total_chars} 字符")
        
        # 估算耗时（假设每批需要30秒）
        num_batches = (len(tasks) + args.batch_size - 1) // args.batch_size
        estimated_time = num_batches * 30
        logger.info(f"预估批次数: {num_batches}")
        logger.info(f"预估处理时间: {estimated_time:.1f} 秒")
        
        return
    
    # 处理文档结构
    logger.info("开始提取实体和三元组...")
    try:
        updated_structure = extractor.process_document_structure(
            doc_structure, 
            batch_size=args.batch_size
        )
        
        # 保存结果
        logger.info(f"保存结果到: {args.output_file}")
        extractor.save_updated_structure(updated_structure, args.output_file)
        
        logger.info("实体和三元组提取完成！")
        
        # 输出统计信息
        stats = extractor.stats
        total_processed = stats['filter_chunks_processed']
        total_tasks = total_processed + stats['failed_calls']
        
        logger.info(f"处理统计:")
        logger.info(f"  总任务数: {total_tasks}")
        logger.info(f"  成功处理的filter_chunk: {total_processed}")
        logger.info(f"  失败的调用: {stats['failed_calls']}")
        if total_tasks > 0:
            success_rate = total_processed / total_tasks * 100
            logger.info(f"  总成功率: {success_rate:.1f}%")
        
        logger.info(f"  提取的实体: {stats['entities_extracted']}")
        logger.info(f"  提取的三元组: {stats['triples_extracted']}")
        
        # 输出备用配置使用统计
        fallback_count = extractor.fallback_used_count
        if fallback_count > 0:
            logger.info(f"  其中备用配置成功: {fallback_count} 批次")
            main_success = total_processed - fallback_count if total_processed >= fallback_count else total_processed
            logger.info(f"  主配置成功: {main_success} 批次")
        
    except KeyboardInterrupt:
        logger.warning("用户中断了处理过程")
        sys.exit(1)
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()